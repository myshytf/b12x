"""Compare TP9 static-peer push with the deployed, unmodified kernel source.

Run on nine idle physical GPUs. Output hashes and owner-ordered FP32 oracles
must pass before ABBA timing. Graphs retain input, output and IPC storage;
repeated replay includes input mutations and different CTA counts.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import statistics

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 9
SHAPES = "1x8,1x72,1x80,1x3584,1x7168,4x3584,4x7168,8x3584,8x7168,16x3584,16x7168"


def digest(tensor):
    return hashlib.sha256(
        tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def inputs_and_oracle(shape, sample):
    generator = torch.Generator().manual_seed(71903 + sample)
    values = torch.randn((WORLD, *shape), generator=generator)
    for rank in range(WORLD):
        values[rank].mul_(2.0 ** (rank - 4))
    # Include cancellation and exponent separation that distinguish FP32
    # rank orders, in addition to ordinary nonzero activation magnitudes.
    flat = values.view(WORLD, -1)
    flat[:, ::31] = torch.tensor(
        [1e20, 1.0, -1e20, 3.0, -2.0, 1e-10, -1e-10, 0.5, -0.25]
    )[:, None]
    flat[:, 1::37] = torch.tensor(
        [1e-30, -1e-20, 1e-10, -1.0, 2.0, 1e10, -1e10, 1e20, -1e20]
    )[:, None]
    values = values.to(torch.bfloat16)
    packs = values[0].numel() // 8
    base, remainder = divmod(packs, WORLD)
    expected = torch.empty(shape, dtype=torch.bfloat16)
    for owner in range(WORLD):
        begin = (owner * base + min(owner, remainder)) * 8
        end = begin + (base + int(owner < remainder)) * 8
        acc = torch.zeros(end - begin, dtype=torch.float32)
        for offset in range(WORLD):
            acc.add_(values[(owner + offset) % WORLD].flatten()[begin:end].float())
        expected.flatten()[begin:end] = acc.to(torch.bfloat16)
    return values, expected


def worker(rank, port, args):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=300),
    )
    from b12x.comm.pcie import pcie_twoshot_bf16 as runtime_module

    # Keep the deployed reference implementation independent of the candidate
    # branch, including its original rank operand and compilation constants.
    spec = importlib.util.spec_from_file_location(
        "b12x.comm.pcie._reference_twoshot_bf16_cute", args.reference_kernel
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    candidate_get = runtime_module.get_twoshot_bf16_allreduce_launcher
    candidate_prepared = runtime_module.is_twoshot_bf16_allreduce_launcher_prepared

    def dispatch(*positional, **keywords):
        mode = positional[7] if len(positional) > 7 else keywords.get("mode", "pull")
        fn = (
            reference.get_twoshot_bf16_allreduce_launcher
            if mode == "push"
            else candidate_get
        )
        return fn(*positional, **keywords)

    def prepared(*positional, **keywords):
        mode = positional[7] if len(positional) > 7 else keywords.get("mode", "pull")
        fn = (
            reference.is_twoshot_bf16_allreduce_launcher_prepared
            if mode == "push"
            else candidate_prepared
        )
        return fn(*positional, **keywords)

    runtime_module.get_twoshot_bf16_allreduce_launcher = dispatch
    runtime_module.is_twoshot_bf16_allreduce_launcher_prepared = prepared
    runtimes = {}
    for mode in ("push", "push_static"):
        runtime = runtime_module.PCIeTwoShotBF16.from_exchange_group(
            exchange_group=dist.group.WORLD, device=device, max_rows=49149, row_elems=8
        )
        runtime.all_reduce_mode = mode
        runtime.prepare_graph(operations=())
        runtimes[mode] = runtime
    records, retained = [], []

    def check(output, expected, context):
        actual = output.cpu()
        ok = (
            torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            and bool(torch.isfinite(actual).all())
            and bool(torch.count_nonzero(actual))
        )
        passed = torch.tensor(int(ok))
        dist.all_reduce(passed, op=dist.ReduceOp.MIN)
        if not bool(passed):
            raise AssertionError(f"rank {rank}: exact oracle failed for {context}")
        return digest(actual)

    for raw_shape in args.shapes.split(","):
        shape = tuple(int(x) for x in raw_shape.split("x"))
        inputs, expected = inputs_and_oracle(shape, 0)
        inp = inputs[rank].to(device)
        input_pointer = inp.data_ptr()
        outputs = {mode: torch.empty_like(inp) for mode in runtimes}
        addresses = {mode: value.data_ptr() for mode, value in outputs.items()}
        graphs = {}
        hashes = {mode: [] for mode in runtimes}
        for mode, runtime in runtimes.items():
            runtime.all_reduce(inp, out=outputs[mode])
            hashes[mode].append(check(outputs[mode], expected, (shape, mode, "eager")))
            graph = torch.cuda.CUDAGraph()
            with runtime.capture(operations=()), torch.cuda.graph(graph):
                for _ in range(args.calls_per_graph):
                    runtime.all_reduce(inp, out=outputs[mode])
            graphs[mode] = graph
        for sample in range(1, 4):
            inputs, expected = inputs_and_oracle(shape, sample)
            inp.copy_(inputs[rank])
            before = digest(inp)
            for mode in runtimes:
                outputs[mode].fill_(float("nan"))
                graphs[mode].replay()
                hashes[mode].append(
                    check(outputs[mode], expected, (shape, mode, sample))
                )
                assert digest(inp) == before and inp.data_ptr() == input_pointer
                assert outputs[mode].data_ptr() == addresses[mode]
        assert hashes["push"] == hashes["push_static"]
        if rank == 0:
            print(
                json.dumps({"shape": shape, "exact_eager_mutated_graph": True}),
                flush=True,
            )

        batches = []
        for mode in ("push", "push_static", "push_static", "push"):
            graph = graphs[mode]
            # Queue enough work to hide Python scheduling between GPU ranks.
            events = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(args.samples)
            ]
            for start, end in events:
                start.record()
                end.record()
            torch.cuda.synchronize()
            dist.barrier()
            for _ in range(8):
                graph.replay()
            for start, end in events:
                start.record()
                graph.replay()
                end.record()
            torch.cuda.synchronize()
            times = [
                start.elapsed_time(end) * 1000 / args.calls_per_graph
                for start, end in events
            ]
            all_times = [None] * WORLD
            dist.all_gather_object(all_times, times)
            batches.append(
                {
                    "mode": mode,
                    "rank_samples_us": all_times,
                    "max_rank_samples_us": [
                        max(items) for items in zip(*all_times, strict=True)
                    ],
                }
            )
            check(outputs[mode], expected, (shape, mode, "after-timing"))
        medians = {
            mode: statistics.median(
                value
                for b in batches
                if b["mode"] == mode
                for value in b["max_rank_samples_us"]
            )
            for mode in runtimes
        }
        record = {
            "shape": shape,
            "hashes": hashes,
            "exact": True,
            "batches": batches,
            "median_max_rank_us": medians,
            "candidate_over_reference_latency": medians["push_static"]
            / medians["push"],
        }
        records.append(record)
        retained.append((inp, outputs, graphs))
        if rank == 0:
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in record.items()
                        if key not in ("batches", "hashes")
                    }
                ),
                flush=True,
            )
            args.output.write_text(
                json.dumps({"status": "in-progress", "records": records}, indent=2)
            )
        if shape == (4, 7168):
            for mode in runtimes:
                dist.barrier()
                if rank == 0:
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as prof:
                        graphs[mode].replay()
                        torch.cuda.synchronize()
                    prof.export_chrome_trace(
                        str(args.output.with_name(mode + ".trace.json"))
                    )
                else:
                    graphs[mode].replay()
                    torch.cuda.synchronize()
                dist.barrier()

    properties = torch.cuda.get_device_properties(device)
    device_record = {
        "rank": rank,
        "name": properties.name,
        "uuid": str(properties.uuid),
        "sm_count": properties.multi_processor_count,
    }
    devices = [None] * WORLD
    dist.all_gather_object(devices, device_record)
    torch.cuda.synchronize()
    retained.clear()
    del graph, graphs
    for runtime in runtimes.values():
        runtime.close()
    if rank == 0:
        args.output.write_text(
            json.dumps(
                {
                    "status": "qualified-component",
                    "reference_sha256": hashlib.sha256(
                        args.reference_kernel.read_bytes()
                    ).hexdigest(),
                    "candidate_sha256": hashlib.sha256(
                        Path(
                            candidate_get.__wrapped__.__code__.co_filename
                        ).read_bytes()
                    ).hexdigest(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "devices": devices,
                    "calls_per_graph": args.calls_per_graph,
                    "records": records,
                    "limitation": "Synthetic communication inputs; no model throughput or quality qualification.",
                },
                indent=2,
            )
        )
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-kernel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shapes", default=SHAPES)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--calls-per-graph", type=int, default=32)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(port, args), nprocs=WORLD, join=True)


if __name__ == "__main__":
    main()
