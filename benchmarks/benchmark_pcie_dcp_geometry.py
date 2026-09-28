"""Exact DCP geometry comparison at the TP9 k=3/k=5 verification shapes.

Reuses the established nine-GPU DCP benchmark's input, capture, and measurement
helpers. Each operation compares two candidate geometries with the served
512-thread/eight-block launch, preserving both raw bytes and LSE arithmetic.
"""

from __future__ import annotations

import json
import os
import pathlib
import statistics
import subprocess

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import benchmark_pcie_dcp_a2a as common


REFERENCE = (512, 8)
CANDIDATES = {"heads": ((512, 16), (256, 16)), "lse": ((128, 32), (256, 16))}


def _telemetry():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,pstate,clocks.sm,clocks.mem,"
         "temperature.gpu,power.draw,clocks_event_reasons.active", "--format=csv"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    return result.stdout


def _exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if not torch.equal(actual.contiguous().view(torch.uint8),
                       expected.contiguous().view(torch.uint8)):
        raise AssertionError("candidate geometry differs in output bytes")


def _worker(rank, world_size, port, operation, out_path):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=world_size)
    pool = None
    try:
        batches = common._batches()
        geometries = (REFERENCE, *CANDIDATES[operation])
        pool = common.PCIeDCPA2APool.from_process_group(
            process_group=dist.group.WORLD, device=device,
            max_batch_size=common.MAX_BATCH, total_heads=common.TOTAL_HEADS,
            head_dim=common.HEAD_DIM, query_head_dim=common.QUERY_HEAD_DIM,
        )
        channels = tuple(f"geometry:{operation}:{batch}:{threads}:{blocks}"
                         for batch in batches for threads, blocks in geometries)
        pool.prepare_channels(channels)
        for threads in {geometry[0] for geometry in geometries}:
            if operation == "heads":
                pool.prepare_graph_all_gather_heads(threads=threads,
                                                   channel_id=channels[0])
            else:
                pool.prepare_graph_lse_reduce_scatter(dtype=torch.bfloat16,
                                                      threads=threads,
                                                      channel_id=channels[0])
        result = {"metadata": common._metadata(world_size, "fp8"),
                  "operation": operation, "reference": REFERENCE,
                  "rows": [], "telemetry": []}
        for batch in batches:
            partial, lse = common._rank_inputs(rank, batch, device)
            query = common._rank_query(rank, world_size, batch,
                                       torch.float8_e4m3fn, device)
            shape = ((batch, common.TOTAL_HEADS, common.QUERY_HEAD_DIM)
                     if operation == "heads" else
                     (batch, common.TOTAL_HEADS // world_size, common.HEAD_DIM))
            dtype = torch.float8_e4m3fn if operation == "heads" else torch.bfloat16
            outputs = {geometry: torch.empty(shape, dtype=dtype, device=device)
                       for geometry in geometries}
            graphs = {}
            for geometry in geometries:
                threads, blocks = geometry
                if operation == "heads":
                    fn = lambda: pool.all_gather_heads(
                        query, outputs[geometry], threads=threads, block_limit=blocks)
                else:
                    fn = lambda: pool.lse_reduce_scatter(
                        partial, lse, outputs[geometry], threads=threads,
                        block_limit=blocks)
                graphs[geometry] = common._capture(
                    fn, pool, f"geometry:{operation}:{batch}:{threads}:{blocks}")
            # Keep the existing mathematical/transport oracle as a nonzero gate.
            graphs[REFERENCE].replay()
            torch.cuda.synchronize(device)
            if operation == "heads":
                expected = torch.cat([
                    common._rank_query(source, world_size, batch,
                                       torch.float8_e4m3fn, device)
                    for source in range(world_size)], dim=1)
                _exact(outputs[REFERENCE], expected)
            else:
                sources = [common._rank_inputs(source, batch, device)
                           for source in range(world_size)]
                expected = common.lse_reduce_scatter_reference(
                    torch.stack([item[0] for item in sources]),
                    torch.stack([item[1] for item in sources]), rank)
                torch.testing.assert_close(outputs[REFERENCE], expected,
                                           rtol=2e-2, atol=2e-2)
            assert bool(outputs[REFERENCE].float().abs().sum() > 0)
            # Two distinct inputs on the same captured addresses, every rank.
            for mutation in range(2):
                if mutation:
                    query.copy_(query.float().neg().to(query.dtype))
                    partial.neg_()
                    lse.neg_()
                for graph in graphs.values():
                    graph.replay()
                torch.cuda.synchronize(device)
                for candidate in CANDIDATES[operation]:
                    _exact(outputs[candidate], outputs[REFERENCE])
            if rank == 0:
                result["telemetry"].append({"batch": batch, "phase": "before",
                                            "csv": _telemetry()})
            for candidate in CANDIDATES[operation]:
                arms = {"reference": [], "candidate": []}
                paired_ratios = []
                for _ in range(3):
                    block = {"reference": [], "candidate": []}
                    for arm in ("reference", "candidate", "candidate", "reference"):
                        geometry = REFERENCE if arm == "reference" else candidate
                        value = common._measure(graphs[geometry], device,
                                                warmup=50, iterations=600)
                        block[arm].append(value)
                        arms[arm].append(value)
                    paired_ratios.append(statistics.mean(block["candidate"])
                                         / statistics.mean(block["reference"]))
                before = statistics.median(arms["reference"])
                after = statistics.median(arms["candidate"])
                row = {"batch": batch, "candidate": candidate, "bit_identical": True,
                       "reference_us": before, "candidate_us": after,
                       "candidate_over_reference": after / before,
                       "paired_ratios": paired_ratios, "samples_us": arms}
                result["rows"].append(row)
                if rank == 0:
                    print(json.dumps(row), flush=True)
                    pathlib.Path(out_path).write_text(json.dumps(result, indent=2) + "\n")
            if rank == 0:
                result["telemetry"].append({"batch": batch, "phase": "after",
                                            "csv": _telemetry()})
            for graph in graphs.values():
                graph.reset()
            torch.cuda.synchronize(device)
        if rank == 0:
            pathlib.Path(out_path).write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if pool is not None:
            pool.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    common._reject_production_geometry_overrides()
    overrides = [name for name in os.environ if name.startswith("B12X_PCIE_DCP_")
                 and name.endswith(("_THREADS", "_BLOCK_LIMIT")) and os.environ[name]]
    if overrides:
        raise ValueError(f"geometry must be controlled by benchmark arguments: {overrides}")
    operation = os.environ["K3_DCP_GEOMETRY_OPERATION"]
    if operation not in CANDIDATES:
        raise ValueError("K3_DCP_GEOMETRY_OPERATION must be heads or lse")
    if (common.TOTAL_HEADS, common.HEAD_DIM, common.QUERY_HEAD_DIM) != (99, 512, 576):
        raise ValueError("this qualification targets 99 heads, 512 output, 576 query")
    if torch.cuda.device_count() != 9:
        raise ValueError("this qualification requires exactly nine visible GPUs")
    mp.spawn(_worker, args=(9, common._free_port(), operation,
                           os.environ["K3_DCP_GEOMETRY_OUTPUT"]), nprocs=9, join=True)


if __name__ == "__main__":
    main()
