"""Native checkpoint A/B probe for the W4A16 FC2 metadata construction inside the MoE LUT-copy phase.

This is a one-GPU component experiment. It does not measure model throughput.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch


def digest(tensor):
    return hashlib.sha256(
        tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--widths", default="384,256")
    parser.add_argument("--m-values", default="1,2,4,8,16")
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--stages", action="store_true")
    parser.add_argument(
        "--external-sanitizer", action="store_true",
        help="Correctness only: leave CUPTI to Compute Sanitizer and omit timings.",
    )
    parser.add_argument(
        "--route-pattern",
        choices=("independent", "shared", "partial", "all"),
        default="all",
    )
    args = parser.parse_args()
    os.environ["B12X_W4A16_M8_CTA_THREADS"] = "256"
    os.environ["B12X_W4A16_GROUPED_DECODE"] = "0"
    os.environ["B12X_W4A16_REFERENCE_GROUPED"] = str(int(args.mode != 0))
    os.environ["B12X_W4A16_REFERENCE_GROUPED_PHASES"] = "fc2"
    os.environ["B12X_W4A16_GROUP_BUILDER_INLINE"] = str(int(args.mode == 2))
    torch.cuda.set_device(0)
    torch.manual_seed(71903)
    from b12x.moe import fused_moe
    from b12x.moe._shared.qsrt_sharding import plan_qsrt_tp9_rank
    from b12x.moe._shared.kernels.w4a16 import kernel as kernel_module
    from vllm.model_executor.layers.quantization.kquant_qsrt_atoms_v2 import (
        open_qsrt_atom_v2_extent,
        read_qsrt_atom_v2_layer_metadata,
    )

    observed = {}
    original_init = kernel_module.W4A16FusedMoeKernel.__init__

    def record_init(self, *positional, **keyword):
        original_init(self, *positional, **keyword)
        if args.mode == 2 and self.reference_grouped:
            assert self.reference_grouped_inline, {
                "message": "Inline preparation was not selected",
                "sm_budget": self.sms,
                "blocks_per_sm": self.blocks_per_sm,
                "direct_smem": self.sqg_xor_cheb_t12_direct_smem,
                "threads": self.cta_threads,
            }
        key = repr(self.__cache_key__)
        observed[key] = {
            "width": self.intermediate_size,
            "direct_smem": self.sqg_xor_cheb_t12_direct_smem,
            "direct_topk": self.direct_topk_routes,
            "grouped_decode": self.reference_grouped,
            "inline_builder": self.reference_grouped_inline,
            "fc1_grouped": self.fc1.reference_grouped_phase >= 0,
            "fc2_grouped": self.fc2.reference_grouped_phase >= 0,
            "fc1_n": self.fc1.tile_n,
            "fc1_threads": self.fc1.cta_threads,
            "fc2_threads": self.fc2.cta_threads,
            "fc1_k_slices": (self.fc1.cta_threads // self.fc1.red_threads),
            "fc2_k_slices": (self.fc2.cta_threads // self.fc2.red_threads),
            "moe_block_size": self.moe_block_size,
            "full_rotation": self.full_rotation,
            "token_major_rotation": self.token_major_rotation,
            "threads": self.cta_threads,
            "sm_budget": self.sms,
            "blocks_per_sm": self.blocks_per_sm,
            "shared_bytes": self.shared_words * 4 + 16,
        }

    kernel_module.W4A16FusedMoeKernel.__init__ = record_init
    selected_launches = []
    original_flat = kernel_module._w4a16_fused_moe_launch_flat
    original_compile = kernel_module.compile_w4a16_fused_moe
    launch_context = None

    def record_compile(*positional, **keyword):
        compiled = original_compile(*positional, **keyword)
        if launch_context is not None:
            expected = args.mode == 2 and 2 <= launch_context["m"] <= 8
            assert compiled.reference_grouped_inline == expected
            launch_context.update(
                inline_builder=compiled.reference_grouped_inline,
                kernel_symbol=compiled.kernel_symbol,
            )
        return compiled

    kernel_module.compile_w4a16_fused_moe = record_compile

    def record_flat(*positional, **keyword):
        nonlocal launch_context
        launch_context = {
            k: keyword[k] for k in ("m", "direct_topk_routes", "full_rotation")
        }
        try:
            result = original_flat(*positional, **keyword)
            assert "inline_builder" in launch_context
            selected_launches.append(launch_context)
            return result
        finally:
            launch_context = None

    kernel_module._w4a16_fused_moe_launch_flat = record_flat
    model = Path("/model")
    metadata = read_qsrt_atom_v2_layer_metadata(
        model / f"qsrt-layer-{args.layer:05d}.safetensors", layer=args.layer
    )
    sizes = [int(m) for m in args.m_values.split(",")]
    records = []
    patterns = (
        ("independent", "shared", "partial")
        if args.route_pattern == "all"
        else (args.route_pattern,)
    )
    for width in [int(w) for w in args.widths.split(",")]:
        rank = next(
            r
            for r in range(9)
            if plan_qsrt_tp9_rank(args.layer, r).intermediate_channels == width
        )
        weight_plan = fused_moe.plan_weights(
            quant_modes="w4a16",
            source_format="qsrt_sqg_e4m3",
            activation="situ",
            params_dtype=torch.bfloat16,
            num_experts=896,
            hidden_size=3584,
            intermediate_size=width,
            w13_layout="w13",
            trellis_bits=2,
            trellis_tile_config=(128, 128, 128, 128),
            qsrt_storage_format="qsrt_atoms_v2",
            qsrt_profile=metadata.profile,
        )
        with open_qsrt_atom_v2_extent(
            metadata, shard_count=9, shard_index=rank, device=None
        ) as (first, atoms):
            weights = fused_moe.prepare_weights(
                plan=weight_plan,
                params_dtype=torch.bfloat16,
                qsrt_atom_payload=atoms,
                qsrt_first_atom_slot=first,
                qsrt_layer_index=args.layer,
                gate_suh=metadata.gate_suh.unsqueeze(0).cuda(),
                up_suh=metadata.up_suh.unsqueeze(0).cuda(),
                down_svh=metadata.down_svh.unsqueeze(0).cuda(),
                qsrt_rotation_draws=metadata.rotation_draws,
            )
        plan = fused_moe.plan(
            fused_moe.Caps(
                max_tokens=max(sizes),
                num_topk=16,
                device=0,
                weight_plan=weights.plan,
                quant_mode="w4a16",
                route_num_experts=896,
                w4a16_block_size_m=8,
                w4a16_shared_input_rotation=True,
            )
        )
        spec = plan.scratch_specs()[0]
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        output = torch.empty((max(sizes), 3584), dtype=torch.bfloat16, device="cuda")
        mapping = torch.arange(896, dtype=torch.int32, device="cuda")
        for pattern, m in ((p, m) for p in patterns for m in sizes):
            torch.manual_seed(71903 + m + width)
            x = (torch.randn((m, 3584), device="cuda") * 0.1).to(torch.bfloat16)

            def new_ids():
                permutation = torch.randperm(896, device="cuda")
                if pattern == "shared":
                    return permutation[:16].expand(m, 16).contiguous().to(torch.int32)
                if pattern == "partial":
                    return torch.stack(
                        [
                            torch.cat(
                                (
                                    permutation[:8],
                                    permutation[8 + row * 8 : 16 + row * 8],
                                )
                            )
                            for row in range(m)
                        ]
                    ).to(torch.int32)
                return torch.stack(
                    [torch.randperm(896, device="cuda")[:16] for _ in range(m)]
                ).to(torch.int32)

            ids = new_ids()
            routing = torch.softmax(torch.randn((m, 16), device="cuda"), dim=-1)
            retained = None
            last_binding = None

            def run(
                plan=plan,
                scratch=scratch,
                weights=weights,
                mapping=mapping,
                output=output,
            ):
                nonlocal last_binding
                binding = fused_moe.bind(
                    plan,
                    scratch=scratch,
                    a=x,
                    experts=weights,
                    topk_weights=routing,
                    topk_ids=ids,
                    route_expert_map=mapping,
                    output=output[:m],
                )
                if retained is not None:
                    binding = dataclasses.replace(binding, retained_fc2_output=retained)
                last_binding = binding
                return fused_moe.run(binding=binding)

            eager = run().clone()
            chosen = selected_launches[-1]
            expected_direct = m <= 8
            assert chosen["direct_topk_routes"] == expected_direct, chosen
            if args.stages:
                retained = torch.empty(
                    (m * 16, 3584), dtype=torch.float16, device="cuda"
                )
                instrumented = run().clone()
                torch.accelerator.synchronize()
                # Retaining FC2 must expose the same output, not a new execution
                # reference. It leaves FC1's otherwise overwritten values live.
                torch.testing.assert_close(instrumented, eager, rtol=0, atol=0)
                assert last_binding is not None
                stage_path = args.output.with_name(
                    f"{args.output.stem}-width{width}-{pattern}-m{m}.stages.pt"
                )
                torch.save(
                    {
                        "x": x.cpu(),
                        "ids": ids.cpu(),
                        "weights": routing.cpu(),
                        "rotation": last_binding.rotation_a_gate.view(-1)[
                            : m * 3584
                        ].cpu(),
                        "fc1": last_binding.intermediate_cache13.view(-1)[
                            : m * 16 * width * 2
                        ].cpu(),
                        "activation": last_binding.intermediate_cache2.view(-1)[
                            : m * 16 * width
                        ].cpu(),
                        "fc2": retained.cpu(),
                        "output": eager.cpu(),
                    },
                    stage_path,
                )
            torch.accelerator.synchronize()
            if not torch.isfinite(eager).all() or not torch.count_nonzero(eager):
                raise AssertionError("Invalid or all-zero native output")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replay = run()
            graph.replay()
            torch.accelerator.synchronize()
            torch.testing.assert_close(replay, eager, rtol=0, atol=0)
            launch_blocks = []
            if m == 4 and not args.external_sanitizer:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profiler:
                    graph.replay()
                    torch.accelerator.synchronize()
                trace = args.output.with_name(
                    f"{args.output.stem}-width{width}-{pattern}-m4.trace.json"
                )
                profiler.export_chrome_trace(str(trace))
                events = json.loads(trace.read_text())["traceEvents"]
                launch_blocks = [
                    e["args"]["block"]
                    for e in events
                    if e.get("cat") == "kernel"
                    and "W4A16FusedMoeKernel" in e.get("name", "")
                ]
                assert any(
                    e.get("cat") == "kernel"
                    and e.get("name") == chosen["kernel_symbol"]
                    for e in events
                ), "The selected compiled MoE object was not observed on the GPU"
                builders = [
                    e
                    for e in events
                    if e.get("cat") == "kernel"
                    and "ReferenceGroupBuilder" in e.get("name", "")
                ]
                assert bool(builders) == (args.mode == 1), (
                    "Unexpected standalone builder path"
                )
                if args.mode == 1:
                    assert builders[0]["args"]["grid"][0] == 28
                    assert not any(
                        "FillFunctor<int>" in e.get("name", "")
                        for e in events
                        if e.get("cat") == "kernel"
                    )
                fills = [
                    e
                    for e in events
                    if e.get("cat") == "kernel"
                    and "FillFunctor<int>" in e.get("name", "")
                ]
                assert len(fills) == (0 if args.mode == 1 else 1)
                if not launch_blocks or any(b[0] != 256 for b in launch_blocks):
                    raise AssertionError(
                        f"Requested stable expert grouping did not execute: {launch_blocks}"
                    )
            timing_inputs = (x.clone(), ids.clone(), routing.clone())
            hashes = [digest(eager)]
            input_hashes = [[digest(x), digest(ids), digest(routing)]]
            for mutation in range(3):
                x.copy_((torch.randn_like(x.float()) * 0.1).to(x.dtype))
                ids.copy_(new_ids())
                if mutation == 2 and m > 1:
                    ids[-1].fill_(-1)
                routing.copy_(torch.softmax(torch.randn_like(routing), dim=-1))
                expected = run().clone()
                graph.replay()
                torch.accelerator.synchronize()
                torch.testing.assert_close(replay, expected, rtol=0, atol=0)
                if not torch.isfinite(replay).all() or not torch.count_nonzero(replay):
                    raise AssertionError("Invalid replay after input/routing update")
                hashes.append(digest(replay))
                input_hashes.append([digest(x), digest(ids), digest(routing)])
            for target, source in zip((x, ids, routing), timing_inputs, strict=True):
                target.copy_(source)
            graph.replay()
            torch.accelerator.synchronize()
            torch.testing.assert_close(replay, eager, rtol=0, atol=0)
            for _ in range(2 if args.external_sanitizer else 40):
                graph.replay()
            torch.accelerator.synchronize()
            samples = []
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(0 if args.external_sanitizer else args.samples):
                begin.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end) * 1000)
            record = {
                "mode": args.mode,
                "layer": args.layer,
                "rank": rank,
                "route_pattern": pattern,
                "width": width,
                "m": m,
                "output_dtype": str(output.dtype),
                "eager_graph_exact": True,
                "chosen_launch": chosen,
                "output_hashes": hashes,
                "launch_blocks": launch_blocks,
                "input_hashes": input_hashes,
                "median_us": statistics.median(samples) if samples else None,
                "samples_us": samples,
            }
            records.append(record)
            print(
                json.dumps({k: v for k, v in record.items() if k != "samples_us"}),
                flush=True,
            )
            del graph, eager, expected, replay, run, timing_inputs
        del weights, plan, scratch, output, mapping
        torch.accelerator.empty_cache()
    specializations_ok = all(
        r["chosen_launch"].get("inline_builder") == (args.mode == 2 and 2 <= r["m"] <= 8)
        for r in records
    )
    trace_specializations_ok = all(
        r["launch_blocks"] and all(b[0] == 256 for b in r["launch_blocks"])
        for r in records
        if r["m"] == 4
    )
    if not args.external_sanitizer:
        specializations_ok = specializations_ok and trace_specializations_ok
    result = {
        "status": "sanitizer-correctness-only" if args.external_sanitizer else "component-only",
        "trace_validation": "external-sanitizer-owns-cupti" if args.external_sanitizer else "qualified",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "kernel_source": kernel_module.__file__,
        "kernel_sha256": hashlib.sha256(
            Path(kernel_module.__file__).read_bytes()
        ).hexdigest(),
        "specializations": list(observed.values()),
        "records": records,
        "compiled_resources": [
            {
                k: getattr(v, k, None)
                for k in (
                    "kernel_symbol",
                    "registers_per_thread",
                    "local_memory_bytes",
                    "shared_memory_bytes",
                    "cta_threads",
                    "direct_topk_routes",
                    "schedule_whole_tiles",
                    "reference_grouped",
                    "reference_grouped_inline",
                )
            }
            for v in kernel_module._FUSED_CACHE.values()
        ],
        "specializations_ok": specializations_ok,
        "limitation": "One GPU, actual TP9 extents; no full-model throughput claim",
    }
    args.output.write_text(json.dumps(result, indent=2))
    print("SPECIALIZATIONS " + json.dumps(result["specializations"]), flush=True)
    if not specializations_ok:
        raise RuntimeError("The requested kernel specialization did not run")


if __name__ == "__main__":
    main()
