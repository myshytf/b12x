"""Measure coupled QSRT input-rotation reuse through native W4A16 graph replay.

Synthetic K2 payloads use the Kimi hidden/intermediate geometry and a reduced
expert population so the probe fits beside a resident model. This is a kernel
screen, not a complete-model throughput measurement. Every arm uses identical
weights, activations and routing; output identity precedes timing.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch

from b12x.moe._shared.kernels.w4a16.host import make_w4a16_packed_buffers
from b12x.moe._shared.kernels.w4a16 import kernel
from b12x.moe._shared.kernels.w4a16.prepare import prepare_trellis256_moe_weights


def digest(tensor):
    return hashlib.sha256(
        tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=164)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--block-m", type=int, default=48)
    parser.add_argument(
        "--comparison", choices=("scratch", "prefetch"), default="scratch"
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--memory-limit-mib", type=int, default=384)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    print("device", torch.cuda.get_device_properties(0), "free", free, flush=True)
    if free < 512 * 1024**2:
        raise RuntimeError(
            "Kernel probe requires 512 MiB free after context initialization"
        )
    torch.cuda.set_per_process_memory_fraction(args.memory_limit_mib * 1024**2 / total)
    torch.manual_seed(71903)
    e, h, width, m, topk, bits = args.experts, 3584, args.width, args.tokens, 16, 2
    w13 = torch.randint(
        -32768,
        32767,
        (2, e, h // 16, width // 16, 16 * bits),
        dtype=torch.int16,
        device="cuda",
    )
    w2 = torch.randint(
        -32768,
        32767,
        (e, width // 16, h // 16, 16 * bits),
        dtype=torch.int16,
        device="cuda",
    )
    weights_before = [digest(w13), digest(w2)]
    scale = torch.ones((1, h), dtype=torch.float16, device="cuda")
    rotations = torch.ones((e, 3 * width), dtype=torch.float16, device="cuda")
    signs = (
        torch.randint(0, 2, (e, 3 * width), dtype=torch.int32, device="cuda") * 2 - 1
    ).to(torch.float16)
    prepared = prepare_trellis256_moe_weights(
        w13,
        w2,
        hidden_size=h,
        intermediate_size=width,
        num_experts=e,
        activation="situ",
        params_dtype=torch.float16,
        fc1_tile_n=128,
        fc2_tile_n=128,
        w13_layout="trellis3_t256_proj",
        trellis_bits=bits,
        codebook="sqg_xor_cheb_t12",
        gate_suh=scale,
        up_suh=scale,
        intermediate_rotations=rotations,
        down_svh=scale,
        tile_config=(128, 128, 128, 128),
    )
    prepared = replace(
        prepared,
        coupled_hadamard=True,
        intermediate_rotations=torch.cat((rotations, signs), dim=1).contiguous(),
    )
    prepared_before = [digest(prepared.w13), digest(prepared.w2)]
    source = (torch.randn((m, h), device="cuda") * 0.001).to(torch.bfloat16)
    ids = torch.rand((m, e), device="cuda").topk(topk, dim=1).indices.to(torch.int32)
    routes = torch.rand((m, topk), device="cuda")
    routes /= routes.sum(dim=1, keepdim=True)
    arms = {}
    active_outputs = {}
    constructors = []
    original_init = kernel.W4A16FusedMoeKernel.__init__

    def record_init(self, *pargs, **kwargs):
        original_init(self, *pargs, **kwargs)
        constructors.append(
            dict(
                token_major=self.token_major_rotation,
                cross_tile=self.fc1.cross_tile_prefetch,
                shared_bytes=self.shared_words * 4 + 16,
                ctas_per_sm=self.blocks_per_sm,
                threads=self.cta_threads,
                codebook=self.trellis_codebook,
                full_rotation=self.full_rotation,
                coupled=self.coupled_hadamard,
            )
        )

    kernel.W4A16FusedMoeKernel.__init__ = record_init
    variants = [("reference", "1", "1")]
    variants += (
        [("shared_scratch", "1", "1")]
        if args.comparison == "scratch"
        else [("without_prefetch", "1", "0")]
    )
    for name, token_major, cross_tile in variants:
        os.environ["B12X_W4A16_TOKEN_MAJOR_ROTATION"] = token_major
        os.environ["B12X_W4A16_CROSS_TILE_PREFETCH"] = cross_tile
        buffers = make_w4a16_packed_buffers(
            prepared,
            m=m,
            topk=topk,
            dtype=torch.float16,
            device=torch.device("cuda"),
            full_rotation=True,
            block_size_m=args.block_m,
        )
        if name == "shared_scratch":
            offset = m * topk * 2 * width
            rotation = (
                buffers.intermediate_cache13.flatten()
                .narrow(0, offset, m * h)
                .view(m, h)
            )
            buffers = replace(buffers, rotation_a_gate=rotation, rotation_a_up=rotation)

        # Poison the borrowed tail as well as FC1/FC2 storage before each
        # eager/capture validation; no result may depend on allocator contents.
        buffers.intermediate_cache13.fill_(float("nan"))
        buffers.intermediate_cache2.fill_(float("nan"))

        def run(active_m=m):
            return kernel.run_w4a16_moe(
                source[:active_m],
                prepared,
                routes[:active_m],
                ids[:active_m],
                activation="situ",
                intermediate_cache13=buffers.intermediate_cache13,
                intermediate_cache2=buffers.intermediate_cache2,
                output=buffers.output[:active_m],
                fc1_c_tmp=buffers.fc1_c_tmp,
                fc2_c_tmp=buffers.fc2_c_tmp,
                packed_route_indices=buffers.packed_route_indices,
                block_expert_ids=buffers.block_expert_ids,
                packed_route_count=buffers.packed_route_count,
                expert_offsets=buffers.expert_offsets,
                expert_counts=buffers.expert_counts,
                route_block_size_m=args.block_m,
                intermediate_rotation_scales=prepared.intermediate_rotations,
                full_rotation=True,
                suh_gate_table=prepared.gate_suh,
                suh_up_table=prepared.up_suh,
                svh_table=prepared.down_svh,
                rotation_a_gate=buffers.rotation_a_gate,
                rotation_a_up=buffers.rotation_a_up,
            )

        start_constructor = len(constructors)
        for active_m in sorted({min(m, x) for x in (1, 7, 8, 9, 47, 48, 49, m)}):
            buffers.intermediate_cache13.fill_(float("nan"))
            actual = run(active_m).clone()
            assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
            if name == "reference":
                active_outputs[active_m] = actual.cpu()
            else:
                torch.testing.assert_close(
                    actual.cpu(), active_outputs[active_m], rtol=0, atol=0
                )
        eager = run().clone()
        torch.cuda.synchronize()
        assert torch.isfinite(eager).all() and torch.count_nonzero(eager) > 0
        if arms:
            torch.testing.assert_close(
                eager, arms["reference"]["eager"], rtol=0, atol=0
            )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, eager, rtol=0, atol=0)
        buffers.intermediate_cache13.fill_(float("nan"))
        buffers.intermediate_cache2.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, eager, rtol=0, atol=0)
        arms[name] = dict(
            graph=graph,
            buffers=buffers,
            output=output,
            eager=eager,
            constructors=constructors[start_constructor:],
            samples=[],
        )
        print(name, arms[name]["constructors"], "correct", flush=True)
    kernel.W4A16FusedMoeKernel.__init__ = original_init
    for arm in arms.values():
        for _ in range(10):
            arm["graph"].replay()
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    for repeat in range(args.rounds):
        names = list(arms)
        if repeat % 2:
            names.reverse()
        for name in names:
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            begin.record()
            for _ in range(args.iterations):
                arms[name]["graph"].replay()
            end.record()
            end.synchronize()
            arms[name]["samples"].append(
                begin.elapsed_time(end) * 1000 / args.iterations
            )
    assert torch.cuda.memory_allocated() == allocated
    assert weights_before == [digest(w13), digest(w2)]
    assert prepared_before == [digest(prepared.w13), digest(prepared.w2)]
    baseline = statistics.median(arms["reference"]["samples"])
    report = dict(
        status="kernel screen",
        configuration=vars(args) | {"out": str(args.out)},
        gpu=torch.cuda.get_device_name(),
        allocated_bytes=allocated,
        max_allocated_bytes=torch.cuda.max_memory_allocated(),
        checked_active_tokens=sorted(active_outputs),
        weight_sha256=weights_before,
        rows={},
    )
    for name, arm in arms.items():
        torch.testing.assert_close(arm["output"], arm["eager"], rtol=0, atol=0)
        median = statistics.median(arm["samples"])
        report["rows"][name] = dict(
            median_us=median,
            samples_us=arm["samples"],
            time_over_reference=median / baseline,
            output_sha256=digest(arm["output"]),
            constructors=arm["constructors"],
        )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
