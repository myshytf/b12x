"""Equivalence of lifted frozen QSRT containers with the BTX load path.

Each test packs one rank extent in a frozen container layout, lifts it to
an in-memory BTX extent, and requires byte-identical prepared tensors
against a synthetically written BTX checkpoint carrying the same plane
words. Byte equality against the frozen containers' dedicated readers was
established when the lift landed beside them; those readers are removed.
"""

from __future__ import annotations

import pytest
import torch

from b12x.moe._shared.btx_schema import matrix_atom_bytes, rate_code
from b12x.moe._shared.kernels.w4a16.btx import prepare_btx_moe_weights
from b12x.moe._shared.kernels.w4a16.btx_compat import (
    lift_qsrt_atoms_v1_extent,
    lift_qsrt_atoms_v2_extent,
)
from b12x.moe._shared.kernels.w4a16.btx_synth import (
    BtxSynthConfig,
    synth_layer_payloads,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _rand_plane(generator, hidden_tiles, bits):
    return torch.randint(
        -(1 << 15),
        1 << 15,
        (hidden_tiles, 16 * bits),
        dtype=torch.int16,
        generator=generator,
    )


def _span(generator):
    raw = torch.rand((32,), generator=generator, dtype=torch.float32)
    return (0.5 + raw).to(torch.float16)


@requires_cuda
def test_v1_lift_matches_synth_btx(tmp_path) -> None:
    from b12x.moe._shared.kernels.w4a16.btx import read_btx_layer
    from b12x.moe._shared.kernels.w4a16.btx_synth import write_btx_checkpoint

    hidden, experts, layer_index, first_atom_slot = 256, 3, 1, 0
    expert_ids = torch.tensor([3, 7, 11], dtype=torch.int32)
    format_codes = torch.tensor([0x21, 0x02, 0x11], dtype=torch.uint8)
    physical_pair = first_atom_slot // 8
    rotation = (5 * expert_ids.to(torch.int64) + layer_index) % 12
    logical_pair = (physical_pair - rotation) % 12
    fc1_codes = torch.where(
        logical_pair < (format_codes.to(torch.int64) >> 4),
        rate_code(2, 4),
        rate_code(3, 3),
    ).to(torch.uint8)
    fc2_codes = torch.where(
        logical_pair < (format_codes.to(torch.int64) & 0xF),
        rate_code(2, 4),
        rate_code(3, 3),
    ).to(torch.uint8)

    config = BtxSynthConfig(
        codebook="sqg_e4m3",
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=256,
        moe_layer_indices=(layer_index,),
        bits=None,
        rate_tables={
            layer_index: (
                fc1_codes.reshape(1, -1).clone(),
                fc2_codes.reshape(1, -1).clone(),
            )
        },
        extent_alignment_slots=8,
        seed=41,
    )
    manifest = write_btx_checkpoint(tmp_path, config)
    btx_layer = read_btx_layer(
        tmp_path, manifest, layer_index, first_slot=0, slot_count=8
    )
    device = _device()
    from_synth = prepare_btx_moe_weights(
        btx_layer, activation="situ", device=device
    )

    payloads = synth_layer_payloads(config, layer_index)
    matrix_bytes = matrix_atom_bytes(hidden, 3, 3)
    bundle = 3 * matrix_bytes + 3 * 64
    payload = torch.zeros((8, experts, bundle), dtype=torch.uint8)
    for slot in range(8):
        for expert in range(experts):
            cursor = 0
            for matrix in range(3):
                low, high = payloads.planes[(expert, slot, matrix)]
                for plane in (low, high):
                    raw = plane.contiguous().view(torch.uint8).reshape(-1)
                    payload[slot, expert, cursor : cursor + raw.numel()] = raw
                    cursor += raw.numel()
            for matrix in range(3):
                payload[slot, expert, cursor : cursor + 64] = (
                    payloads.rotations[slot, expert, matrix]
                    .contiguous()
                    .view(torch.uint8)
                )
                cursor += 64

    lifted_layer = lift_qsrt_atoms_v1_extent(
        payload,
        first_atom_slot=first_atom_slot,
        layer_index=layer_index,
        expert_ids=expert_ids,
        format_codes=format_codes,
        hidden_size=hidden,
        global_intermediate_size=256,
        gate_suh=payloads.gate_suh,
        up_suh=payloads.up_suh,
        down_svh=payloads.down_svh,
    )
    lifted = prepare_btx_moe_weights(
        lifted_layer, activation="situ", device=device
    )

    assert torch.equal(lifted.w13, from_synth.w13)
    assert torch.equal(lifted.w2, from_synth.w2)
    assert torch.equal(
        lifted.intermediate_rotations, from_synth.intermediate_rotations
    )
    assert lifted.fc1_trellis_pair_kind == from_synth.fc1_trellis_pair_kind
    assert torch.equal(
        lifted.fc1_trellis_pair_modes, from_synth.fc1_trellis_pair_modes
    )
    assert torch.equal(
        lifted.fc2_trellis_pair_modes, from_synth.fc2_trellis_pair_modes
    )


@requires_cuda
def test_v2_pure_k2_lift_matches_synth_btx(tmp_path) -> None:
    from b12x.moe._shared.kernels.w4a16.btx import read_btx_layer
    from b12x.moe._shared.kernels.w4a16.btx_synth import write_btx_checkpoint

    hidden, global_i, experts, slots = 512, 512, 2, 8
    config = BtxSynthConfig(
        codebook="sqg_e4m3",
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=global_i,
        moe_layer_indices=(1,),
        bits=2,
        coupled=True,
        pre_block=512,
        post_block=128,
        extent_alignment_slots=4,
        extent_barriers=(8,),
        seed=21,
    )
    manifest = write_btx_checkpoint(tmp_path, config)
    btx_layer = read_btx_layer(
        tmp_path, manifest, 1, first_slot=0, slot_count=slots
    )
    device = _device()
    from_synth = prepare_btx_moe_weights(
        btx_layer,
        activation="situ",
        device=device,
        tile_config=(128, 128, 128, 128),
    )

    payloads = synth_layer_payloads(config, 1)
    section = matrix_atom_bytes(hidden, 2, 2)
    bundle = 3 * section + 3 * 64
    payload = torch.zeros((slots, experts * bundle), dtype=torch.uint8)
    for slot in range(slots):
        cursor = 0
        for expert in range(experts):
            for matrix in range(3):
                low, high = payloads.planes[(expert, slot, matrix)]
                for plane in (low, high):
                    raw = plane.contiguous().view(torch.uint8).reshape(-1)
                    payload[slot, cursor : cursor + raw.numel()] = raw
                    cursor += raw.numel()
            for matrix in range(3):
                payload[slot, cursor : cursor + 64] = (
                    payloads.rotations[slot, expert, matrix]
                    .contiguous()
                    .view(torch.uint8)
                )
                cursor += 64

    lifted_layer = lift_qsrt_atoms_v2_extent(
        payload,
        profile="k2_coupled_h512_h128",
        first_atom_slot=0,
        layer_index=1,
        hidden_size=hidden,
        global_intermediate_size=global_i,
        num_experts=experts,
        gate_suh=payloads.gate_suh,
        up_suh=payloads.up_suh,
        down_svh=payloads.down_svh,
        rotation_draws=payloads.rotation_draws,
    )
    lifted = prepare_btx_moe_weights(
        lifted_layer,
        activation="situ",
        device=device,
        tile_config=(128, 128, 128, 128),
    )

    assert torch.equal(lifted.w13, from_synth.w13)
    assert torch.equal(lifted.w2, from_synth.w2)
    assert torch.equal(
        lifted.intermediate_rotations, from_synth.intermediate_rotations
    )
    assert torch.equal(lifted.gate_suh, from_synth.gate_suh)
    assert lifted.coupled_hadamard and from_synth.coupled_hadamard


@requires_cuda
def test_v2_fixed_high_rate_lift_matches_synth_btx(tmp_path) -> None:
    from b12x.moe._shared.kernels.w4a16.btx import read_btx_layer
    from b12x.moe._shared.kernels.w4a16.btx_synth import write_btx_checkpoint

    hidden, experts, layer_index = 256, 3, 1
    hidden_tiles = hidden // 16
    first_atom_slot = 0
    physical_pair = 0
    expert_ids = torch.arange(experts, dtype=torch.int64)
    rotation = (5 * expert_ids + layer_index) % 12
    base_pair = (physical_pair - rotation) % 12
    modes = (base_pair == 0) | (base_pair == 6)
    fc_codes = torch.where(modes, rate_code(4, 3), rate_code(3, 3)).to(
        torch.uint8
    )
    config = BtxSynthConfig(
        codebook="sqg_e4m3",
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=256,
        moe_layer_indices=(layer_index,),
        bits=None,
        rate_tables={
            layer_index: (
                fc_codes.reshape(1, -1).clone(),
                fc_codes.reshape(1, -1).clone(),
            )
        },
        extent_alignment_slots=8,
        seed=33,
    )
    manifest = write_btx_checkpoint(tmp_path, config)
    btx_layer = read_btx_layer(
        tmp_path, manifest, layer_index, first_slot=0, slot_count=8
    )
    device = _device()
    from_synth = prepare_btx_moe_weights(
        btx_layer, activation="situ", device=device
    )

    # Pack the identical planes as a grouped-by-rate-class atoms-v2 row set.
    payloads = synth_layer_payloads(config, layer_index)
    p33_ids = torch.nonzero(~modes, as_tuple=False).flatten().tolist()
    p43_ids = torch.nonzero(modes, as_tuple=False).flatten().tolist()
    rows = []
    for slot in range(8):
        chunks = []
        for expert in p33_ids + p43_ids:
            for matrix in range(3):
                low, high = payloads.planes[(expert, slot, matrix)]
                chunks.append(low.contiguous().view(torch.uint8).reshape(-1))
                chunks.append(high.contiguous().view(torch.uint8).reshape(-1))
            for matrix in range(3):
                chunks.append(
                    payloads.rotations[slot, expert, matrix]
                    .contiguous()
                    .view(torch.uint8)
                )
        rows.append(torch.cat(chunks))
    payload = torch.stack(rows)

    lifted_layer = lift_qsrt_atoms_v2_extent(
        payload,
        profile="k3x22_k4x2",
        first_atom_slot=first_atom_slot,
        layer_index=layer_index,
        hidden_size=hidden,
        global_intermediate_size=256,
        num_experts=experts,
        gate_suh=payloads.gate_suh,
        up_suh=payloads.up_suh,
        down_svh=payloads.down_svh,
    )
    lifted = prepare_btx_moe_weights(
        lifted_layer, activation="situ", device=device
    )

    assert torch.equal(lifted.w13, from_synth.w13)
    assert torch.equal(lifted.w2, from_synth.w2)
    assert torch.equal(
        lifted.intermediate_rotations, from_synth.intermediate_rotations
    )
    assert lifted.fc1_trellis_pair_kind == "P33_P43"
    modes_tensor = lifted.fc1_trellis_pair_modes
    assert modes_tensor is not None
    assert torch.equal(
        (modes_tensor & 1).to(torch.bool).cpu(), modes
    )


def test_coupled_high_rate_profile_has_no_lift() -> None:
    with pytest.raises(ValueError, match="no BTX lift"):
        lift_qsrt_atoms_v2_extent(
            torch.zeros((8, 16), dtype=torch.uint8),
            profile="k3x22_k4x2_coupled_h512_h128",
            first_atom_slot=0,
            layer_index=1,
            hidden_size=3584,
            global_intermediate_size=3072,
            num_experts=896,
            gate_suh=torch.ones(3584, dtype=torch.float16),
            up_suh=torch.ones(3584, dtype=torch.float16),
            down_svh=torch.ones(3584, dtype=torch.float16),
        )
