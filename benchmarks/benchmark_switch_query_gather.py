"""Exact TP9 query-gather comparison for a cascaded five/four switch fabric.

The logical rank order is supplied through CUDA_VISIBLE_DEVICES. Both schedules
use the same live runtime, allocations and pointers. Compare arbitrary BF16
bit patterns, including NaNs, before timing interleaved CUDA graph replays.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

GROUPS = ((0, 1, 2, 3, 8), (4, 5, 6, 7))


def telemetry():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pstate,clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active,memory.free",
            "--format=csv,noheader",
        ],
        text=True,
    )


def worker(rank, port, args):
    from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2A

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=9,
        timeout=timedelta(seconds=180),
    )
    free, _ = torch.cuda.mem_get_info()
    safe = torch.tensor(int(free > 640 * 1024**2))
    dist.all_reduce(safe, op=dist.ReduceOp.MIN)
    if not safe.item():
        raise RuntimeError(
            "The bounded query-gather probe needs 640 MiB free after CUDA initialization on every rank"
        )
    os.environ["B12X_PCIE_DCP_A2A_TRANSPORT"] = "push"
    os.environ.pop("B12X_PCIE_DCP_GATHER_GROUPS", None)
    runtime = PCIeDCPA2A.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_batch_size=16,
        total_heads=99,
        head_dim=512,
        query_head_dim=576,
    )
    records = []
    for batch in (1, 2, 4, 8, 16):
        cpu_inputs = [
            torch.randint(
                0,
                65536,
                (batch, 11, 576),
                generator=torch.Generator().manual_seed(3209 + source + batch * 100),
                dtype=torch.int32,
            ).to(torch.uint16)
            for source in range(9)
        ]
        source = cpu_inputs[rank].to(device).view(torch.bfloat16)
        expected = torch.cat(cpu_inputs, dim=1).view(torch.uint8)
        arms = {}
        for name, groups in (("flat", ()), ("relay", GROUPS)):
            runtime.gather_switch_groups = groups
            output = torch.empty((batch, 99, 576), device=device, dtype=torch.bfloat16)
            runtime.all_gather_heads(source, out=output)
            assert torch.equal(output.view(torch.uint8).cpu(), expected)
            runtime.prepare_graph_all_gather_heads()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                runtime.all_gather_heads(source, out=output)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(output.view(torch.uint8).cpu(), expected)
            arms[name] = (graph, output)
        for graph, _ in arms.values():
            for _ in range(12):
                graph.replay()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        hardware_before = telemetry() if rank == 0 else None
        timings = {name: [] for name in arms}
        for repeat in range(args.rounds):
            for name in list(arms) if repeat % 2 == 0 else list(reversed(arms)):
                dist.barrier()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(args.iterations):
                    arms[name][0].replay()
                end.record()
                end.synchronize()
                elapsed = torch.tensor(
                    start.elapsed_time(end) * 1000 / args.iterations,
                    dtype=torch.float64,
                )
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                timings[name].append(elapsed.item())
                assert torch.equal(arms[name][1].view(torch.uint8).cpu(), expected)
        assert allocated == torch.cuda.memory_allocated()
        hardware_after = telemetry() if rank == 0 else None
        # Both captured schedules must read changed contents at the same address.
        source.view(torch.uint16).zero_()
        for graph, output in arms.values():
            graph.replay()
            torch.cuda.synchronize()
            assert torch.count_nonzero(output.view(torch.uint8)) == 0
        row = dict(
            batch=batch,
            total_heads=99,
            query_head_dim=576,
            exact_bit_patterns=True,
            graph_replay_allocation=False,
            output_sha256=hashlib.sha256(expected.numpy().tobytes()).hexdigest(),
            timings_us=timings,
            medians_us={k: statistics.median(v) for k, v in timings.items()},
            hardware_before=hardware_before,
            hardware_after=hardware_after,
        )
        row["relay_over_flat"] = row["medians_us"]["relay"] / row["medians_us"]["flat"]
        records.append(row)
        if rank == 0:
            print(
                json.dumps(
                    {k: v for k, v in row.items() if not k.startswith("hardware")}
                ),
                flush=True,
            )
            args.out.write_text(
                json.dumps(
                    dict(status="native checks in progress", rows=records), indent=2
                )
                + "\n"
            )
        del arms, graph, output
    # Model execution interleaves Q gather with the LSE reduce-scatter on
    # the same double-buffered channel. The extra relay barrier must not
    # change that channel's epochs or expose stale payloads to the reduction.
    batch = 4
    source = torch.randn(batch, 11, 576, device=device, dtype=torch.bfloat16)
    partial = torch.randn(batch, 99, 512, device=device, dtype=torch.bfloat16)
    lse = torch.randn(batch, 99, device=device)
    mixed_arms = {}
    runtime.prepare_graph_lse_reduce_scatter(dtype=torch.bfloat16)
    for name, groups in (("flat", ()), ("relay", GROUPS)):
        runtime.gather_switch_groups = groups
        runtime.prepare_graph_all_gather_heads()
        queries = torch.empty(batch, 99, 576, device=device, dtype=torch.bfloat16)
        reduced = torch.empty(batch, 11, 512, device=device, dtype=torch.bfloat16)
        runtime.all_gather_heads(source, out=queries)
        runtime.lse_reduce_scatter(partial, lse, out=reduced)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runtime.all_gather_heads(source, out=queries)
            runtime.lse_reduce_scatter(partial, lse, out=reduced)
        graph.replay()
        torch.cuda.synchronize()
        if name == "flat":
            expected_queries, expected_reduced = queries.cpu(), reduced.cpu()
        else:
            assert torch.equal(queries.cpu(), expected_queries)
            assert torch.equal(reduced.cpu(), expected_reduced)
        mixed_arms[name] = (graph, queries, reduced)
    for _ in range(100):
        for graph, queries, reduced in mixed_arms.values():
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(queries.cpu(), expected_queries)
            assert torch.equal(reduced.cpu(), expected_reduced)
    del mixed_arms, graph, queries, reduced
    dist.barrier()
    runtime.close()
    if rank == 0:
        args.out.write_text(
            json.dumps(
                dict(
                status="qualified native query-gather screen",
                rows=records,
                mixed_query_lse_graphs=200,
                    max_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                ),
                indent=2,
            )
            + "\n"
        )
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(port, args), nprocs=9, join=True)


if __name__ == "__main__":
    main()
