"""Compare common preparation with the preserved atom-preparation arithmetic."""

from dataclasses import fields

import pytest
import torch

from b12x.moe import fused_moe
from b12x.moe.checkpoints.qsrt import trellis_from_qsrt_atoms_v2
from tests.moe.test_trellis_qsrt_adapter import _atoms

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("first,slots", [(0, 4), (12, 12), (48, 8), (92, 4)])
def test_common_preparation_is_byte_exact_and_graph_safe(first, slots):
    atoms, _, args = _atoms(first, slots)
    bundles = atoms.unflatten(1, (8, atoms.shape[1] // 8))
    scales = torch.full((slots, 8, 96), 0.05, dtype=torch.float16)
    bundles[:, :, -192:].copy_(scales.view(torch.uint8))
    source, tensors = trellis_from_qsrt_atoms_v2(atoms, **args)
    legacy_plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="qsrt_sqg_e4m3",
        activation="situ",
        params_dtype=torch.bfloat16,
        num_experts=8,
        hidden_size=512,
        intermediate_size=slots * 32,
        trellis_bits=2,
        trellis_tile_config=(128, 128, 128, 128),
        qsrt_storage_format="qsrt_atoms_v2",
        qsrt_profile="k2_coupled_h512_h128",
    )
    legacy = fused_moe.prepare_weights(
        plan=legacy_plan,
        params_dtype=torch.bfloat16,
        qsrt_atom_payload=atoms,
        qsrt_first_atom_slot=first,
        qsrt_layer_index=1,
        gate_suh=args["gate_suh"].unsqueeze(0).cuda(),
        up_suh=args["up_suh"].unsqueeze(0).cuda(),
        down_svh=args["down_svh"].unsqueeze(0).cuda(),
        qsrt_rotation_draws=args["rotation_draws"],
    )
    plan = fused_moe.plan_weights(
        source=source,
        activation=fused_moe.ActivationSpec(
            mode="a16",
            nonlinearity="situ",
            io_dtype=torch.bfloat16,
            rotation_dtype=torch.float16,
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=8, hidden_size=512, intermediate_size=slots * 32
        ),
    )
    common = fused_moe.prepare_weights(
        plan=plan,
        weights=tensors,
        device="cuda",
        staging=fused_moe.TrellisStaging(max_experts=3),
    )
    before, after = legacy.representation.value, common.representation.value
    for field in fields(before):
        old, new = getattr(before, field.name), getattr(after, field.name)
        if isinstance(old, torch.Tensor):
            assert old.dtype == new.dtype and old.shape == new.shape, field.name
            torch.testing.assert_close(
                old.view(torch.uint8), new.view(torch.uint8), rtol=0, atol=0
            )
    assert after.gate_suh.data_ptr() == after.up_suh.data_ptr()
    assert common.plan.source_format == "b12x_trellis"
    assert after.params_dtype == before.params_dtype == torch.float16
    x = (torch.randn(4, 512, device="cuda") * 0.2).bfloat16()
    ids = torch.arange(4, dtype=torch.int32, device="cuda").repeat(4, 1)
    routing = torch.full((4, 4), 0.25, device="cuda")
    outputs = []
    for experts in (legacy, common):
        runtime = fused_moe.plan(
            fused_moe.Caps(
                max_tokens=4,
                num_topk=4,
                route_num_experts=8,
                device=0,
                weight_plan=experts.plan,
                quant_mode="w4a16",
                w4a16_block_size_m=8,
                w4a16_shared_input_rotation=True,
            )
        )
        spec = runtime.scratch_specs()[0]
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        output = torch.empty_like(x)
        binding = fused_moe.bind(
            runtime,
            scratch=scratch,
            a=x,
            experts=experts,
            topk_ids=ids,
            topk_weights=routing,
            output=output,
        )
        fused_moe.run(binding=binding)
        torch.cuda.synchronize()
        assert torch.isfinite(output).all() and torch.count_nonzero(output)
        expected = output.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fused_moe.run(binding=binding)
        output.fill_(float("nan"))
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        outputs.append(output.clone())
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
