"""Compare lossless PCIe gather schedules with exact output and graph checks.

Run only on idle GPUs. The rank order in CUDA_VISIBLE_DEVICES defines the ring.
--rows 576 measures a small-message shape; --rows 4608 measures the Kimi-K3
prefill chunk. Every rank needs at least 2 GiB free after context initialization
for the IPC mappings and graph resources. Use dedicated GPUs. GPU timing
is the maximum across ranks, interleaved in both orders. Both arms use the same
ring, counters, arithmetic and input tensors.
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


def digest(tensor):
    return hashlib.sha256(
        tensor.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


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
    from b12x.comm.pcie.pcie_dma import PCIeDmaAllReduce

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=args.world,
        timeout=timedelta(seconds=180),
    )
    free, _ = torch.cuda.mem_get_info()
    max_bytes = args.rows * 7168 * 2
    safe = torch.tensor(int(free > max(2 * 1024**3, 10 * max_bytes + 64 * 1024**2)))
    dist.all_reduce(safe, op=dist.ReduceOp.MIN)
    if not safe.item():
        raise RuntimeError("Insufficient free device memory for the bounded ring probe")
    os.environ["B12X_PCIE_DMA_GRAPH_REPLAY"] = "0"
    os.environ["B12X_PCIE_DMA_FP8"] = "0"
    ring = PCIeDmaAllReduce(
        exchange_group=dist.group.WORLD, device=device, max_bytes=max_bytes
    )
    ring.min_bytes = 0
    report = {
        "rank": rank,
        "gpu": str(torch.cuda.get_device_properties(rank)),
        "operations": {},
    }

    def compare(operation, run, mutate):
        arms = {}
        reference = None
        for name, enabled in (("reference", False), ("pipelined", True)):
            ring._pipeline_all_gather = enabled
            eager = tuple(t.cpu() for t in run())
            assert all(
                torch.isfinite(t).all() and torch.count_nonzero(t) for t in eager
            )
            if reference is None:
                reference = eager
            else:
                assert all(
                    torch.equal(a, b) for a, b in zip(reference, eager, strict=True)
                )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = run()
            graph.replay()
            torch.cuda.synchronize()
            assert all(
                torch.equal(a, b.cpu()) for a, b in zip(reference, output, strict=True)
            )
            arms[name] = (graph, output)
        allocated = torch.cuda.memory_allocated()
        for graph, _ in arms.values():
            for _ in range(10):
                graph.replay()
        torch.cuda.synchronize()
        samples = {name: [] for name in arms}
        for repeat in range(args.rounds):
            order = list(arms) if repeat % 2 == 0 else list(reversed(arms))
            for name in order:
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
                samples[name].append(elapsed.item())
                assert all(
                    torch.equal(a, b.cpu())
                    for a, b in zip(reference, arms[name][1], strict=True)
                )
        assert torch.cuda.memory_allocated() == allocated
        initial_hashes = [digest(t) for t in reference]
        # Reuse the captured pointers with different contents. A stale output
        # or a missing producer dependency cannot pass by repeating one input.
        mutate()
        ring._pipeline_all_gather = False
        changed = tuple(t.cpu() for t in run())
        assert [digest(t) for t in changed] != initial_hashes
        for graph, output in arms.values():
            graph.replay()
            torch.cuda.synchronize()
            assert all(
                torch.equal(a, b.cpu()) for a, b in zip(changed, output, strict=True)
            )
        result = {
            name: {"median_us": statistics.median(values), "samples_us": values}
            for name, values in samples.items()
        }
        result["time_over_reference"] = (
            result["pipelined"]["median_us"] / result["reference"]["median_us"]
        )
        result["output_sha256"] = initial_hashes
        result["exact_eager_graph_and_mutated_input"] = True
        report["operations"][operation] = result
        if rank == 0:
            print(operation, json.dumps(result), flush=True)
        del arms

    generator = torch.Generator().manual_seed(5603 + rank)
    source = (
        (torch.randn(args.rows, 7168, generator=generator) * 2 ** (rank - 4))
        .to(torch.bfloat16)
        .to(device)
    )
    output = torch.empty_like(source)

    def all_reduce():
        return (ring.all_reduce(source, out=output),)

    compare("all_reduce", all_reduce, lambda: source.add_(0.25))

    def in_place():
        output.copy_(source)
        return (ring.all_reduce(output, out=output),)

    compare("all_reduce_in_place", in_place, lambda: source.add_(0.5))

    first = torch.randn(args.rows, 104, generator=generator).to(device)
    second = (
        torch.randn(args.rows, 400, generator=generator).to(torch.bfloat16).to(device)
    )

    def gather_pair():
        return ring.all_gather_pair(first, second)

    def mutate_pair():
        first.add_(0.125)
        second.add_(0.25)

    compare("gather_pair", gather_pair, mutate_pair)
    assert torch.equal(ring._send_counters, ring._wait_counters)
    report["max_torch_allocated_bytes"] = torch.cuda.max_memory_allocated()
    reports = [None] * args.world
    dist.all_gather_object(reports, report)
    if rank == 0:
        for name in report["operations"]:
            assert (
                len({tuple(r["operations"][name]["output_sha256"]) for r in reports})
                == 1
            )
        args.out.write_text(
            json.dumps(
                {
                    "status": "qualified kernel screen",
                    "args": vars(args) | {"out": str(args.out)},
                    "ranks": reports,
                },
                indent=2,
            )
            + "\n"
        )
    ring.close()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=int, default=9)
    parser.add_argument("--rows", type=int, default=576)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    before = telemetry()
    mp.spawn(worker, args=(port, args), nprocs=args.world, join=True)
    report = json.loads(args.out.read_text())
    report["telemetry"] = {"before": before, "after": telemetry()}
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
