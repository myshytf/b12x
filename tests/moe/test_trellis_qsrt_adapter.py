"""The atom container must preserve bytes and planning through the common API."""

from dataclasses import replace

import pytest
import torch

from b12x.moe import fused_moe
from b12x.moe.checkpoints.qsrt import trellis_from_qsrt_atoms_v2
from b12x.moe.fused_moe.trellis_layout import assemble_uniform_slots


def _atoms(first=0, slots=4, hidden=512, experts=8):
    section = (hidden // 16) * 128
    records = torch.randint(
        0, 256, (slots, experts, 3 * section + 192), dtype=torch.uint8
    )
    scales = torch.arange(slots * experts * 96).reshape(slots, experts, 3, 32).half()
    records[:, :, 3 * section :].copy_(
        scales.reshape(slots, experts, 96).view(torch.uint8)
    )
    args = dict(
        first_atom_slot=first,
        num_experts=experts,
        hidden_size=hidden,
        global_intermediate_size=3072,
        gate_suh=torch.ones(hidden).half(),
        up_suh=torch.full((hidden,), 2.0).half(),
        down_svh=torch.full((hidden,), 3.0).half(),
        rotation_draws=(torch.arange(experts) % 8).byte(),
    )
    return records.flatten(1), scales, args


@pytest.mark.parametrize("first,slots", [(0, 4), (12, 12), (48, 8), (92, 4)])
def test_qsrt_expert_strides_preserve_words_scales_and_global_extent(first, slots):
    atoms, scales, args = _atoms(first, slots)
    source, weights = trellis_from_qsrt_atoms_v2(atoms, **args)
    assert weights.codes.data_ptr() == atoms.data_ptr()
    assert source.extent.first_slot == first
    assert source.extent.intermediate_size == slots * 32
    expected_side = args["gate_suh"] if first < 48 else args["up_suh"]
    torch.testing.assert_close(
        weights.input_scales.vectors, expected_side, rtol=0, atol=0
    )
    expected_scales = scales.permute(1, 2, 0, 3).reshape(8, 3, slots * 32)
    torch.testing.assert_close(
        weights.intermediate_scales.vectors, expected_scales, rtol=0, atol=0
    )
    # A dense native container and an expert-strided container must assemble
    # to identical carrier words even across different staging boundaries.
    native = weights.codes.contiguous().flatten(1)
    dense = assemble_uniform_slots(
        native, num_experts=8, hidden_size=512, bits=2, device="cpu"
    )
    strided = assemble_uniform_slots(
        weights.codes,
        num_experts=8,
        hidden_size=512,
        bits=2,
        device="cpu",
        staging=fused_moe.TrellisStaging(max_experts=1),
    )
    for old, new in zip(dense, strided, strict=True):
        torch.testing.assert_close(old, new, rtol=0, atol=0)


@pytest.mark.parametrize("first,slots", [(44, 8), (96, 4), (1, 4), (0, 3)])
def test_qsrt_adapter_rejects_invalid_extent_before_preparation(first, slots):
    atoms, _, args = _atoms(first, slots)
    with pytest.raises(ValueError):
        trellis_from_qsrt_atoms_v2(atoms, **args)


def test_qsrt_common_plan_retains_io_precision_and_exact_scratch_capacity(monkeypatch):
    from b12x.moe.fused_moe import _impl

    monkeypatch.setattr(_impl, "get_num_sm", lambda _: 188)
    atoms, _, args = _atoms()
    source, _ = trellis_from_qsrt_atoms_v2(atoms, **args)
    common = fused_moe.plan_weights(
        source=source,
        geometry=fused_moe.MoEGeometry(
            num_experts=8, hidden_size=512, intermediate_size=128
        ),
        activation=fused_moe.ActivationSpec(
            mode="a16",
            nonlinearity="situ",
            io_dtype=torch.bfloat16,
            rotation_dtype=torch.float16,
        ),
    )
    legacy = fused_moe.plan_weights(
        source_format="qsrt_sqg_e4m3",
        quant_modes="w4a16",
        activation="situ",
        params_dtype=torch.bfloat16,
        num_experts=8,
        hidden_size=512,
        intermediate_size=128,
        trellis_bits=2,
        trellis_tile_config=(128, 128, 128, 128),
        qsrt_storage_format="qsrt_atoms_v2",
        qsrt_profile="k2_coupled_h512_h128",
    )
    assert common.source_format == "b12x_trellis"
    assert common.trellis_source == source
    assert common.io_dtype == legacy.io_dtype == "bfloat16"
    assert common.coupled_hadamard == legacy.coupled_hadamard
    for rows in (1, 4, 16, 48):
        caps = fused_moe.Caps(
            max_tokens=rows,
            num_topk=4,
            route_num_experts=8,
            device="cpu",
            weight_plan=legacy,
            quant_mode="w4a16",
            w4a16_block_size_m=8,
        )
        assert fused_moe.required_nbytes(caps) == fused_moe.required_nbytes(
            replace(caps, weight_plan=common)
        )


def test_bfloat16_rotation_is_not_silently_lowered_to_fp16():
    with pytest.raises(NotImplementedError, match="explicit FP16"):
        fused_moe.ActivationSpec(
            mode="a16", nonlinearity="situ", io_dtype=torch.bfloat16
        )


def test_arena_contract_ignores_load_coordinates_but_preserves_codebook_and_dtype():
    from b12x.moe.fused_moe.config import TrellisCodebook

    atoms, _, args = _atoms()
    source, _ = trellis_from_qsrt_atoms_v2(atoms, **args)
    source = replace(source, uniform_bits=3)

    def plan(value, dtype=torch.bfloat16):
        return fused_moe.plan_weights(
            source=value,
            activation=fused_moe.ActivationSpec(
                mode="a16",
                nonlinearity="situ",
                io_dtype=dtype,
                rotation_dtype=torch.float16,
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=8, hidden_size=512, intermediate_size=128
            ),
        )

    first = plan(source)
    second = plan(replace(source, extent=replace(source.extent, first_slot=48)))
    assert first != second
    assert first.execution_key == second.execution_key
    mcg = plan(
        replace(source, config=replace(source.config, codebook=TrellisCodebook.MCG))
    )
    assert first.execution_key != mcg.execution_key
    assert first.execution_key != plan(source, torch.float16).execution_key
