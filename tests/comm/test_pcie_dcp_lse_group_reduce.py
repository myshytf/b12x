"""Group-reduce LSE reduce-scatter (CPU checks): staging layout regions, kernel
phase structure and combine order, runtime opt-in and mode wiring."""

from __future__ import annotations

import inspect

import pytest

from b12x.comm.pcie import _dcp_a2a_cute as cute_mod
from b12x.comm.pcie import pcie_dcp_a2a as runtime_mod
from b12x.comm.pcie._twoshot_bf16_cute import GROUP_REDUCE_MAX_SLOTS


def test_layout_without_group_reduce_is_unchanged() -> None:
    base = runtime_mod._staging_layout(signal_bytes=4096, world_size=9, max_batch_size=32, total_heads=99, head_dim=512)
    assert base.inbox_offset == 0 and base.partial_offset == 0
    assert base.slot_bytes == runtime_mod._align_up(base.lse_offset + base.lse_capacity * 4, runtime_mod.IPC_SLAB_ALIGNMENT)


def test_layout_with_group_reduce_adds_inbox_and_partial_regions() -> None:
    base = runtime_mod._staging_layout(signal_bytes=4096, world_size=9, max_batch_size=32, total_heads=99, head_dim=512)
    grp = runtime_mod._staging_layout(signal_bytes=4096, world_size=9, max_batch_size=32, total_heads=99, head_dim=512, lse_group_reduce=True)
    rows = 32 * 11
    assert grp.inbox_offset == base.slot_bytes
    assert grp.inbox_lse_offset >= grp.inbox_offset + GROUP_REDUCE_MAX_SLOTS * 9 * rows * 512 * 2
    assert grp.partial_offset >= grp.inbox_lse_offset + GROUP_REDUCE_MAX_SLOTS * 9 * rows * 4
    assert grp.partial_lse_offset >= grp.partial_offset + rows * 512 * 4
    assert grp.slot_bytes >= grp.partial_lse_offset + rows * 8
    for v in (grp.inbox_offset, grp.inbox_lse_offset, grp.partial_offset, grp.partial_lse_offset, grp.slot_bytes):
        assert v % runtime_mod.IPC_SLAB_ALIGNMENT == 0
    assert grp.slab_bytes == grp.staging1_offset + grp.slot_bytes
    # The unchanged regions keep their offsets so the other collectives are untouched.
    assert (grp.lse_offset, grp.lse_capacity, grp.output_capacity_elems) == (base.lse_offset, base.lse_capacity, base.output_capacity_elems)


def test_kernel_phases_and_combine_order() -> None:
    kernel = inspect.getsource(cute_mod._LseGroupReduceLaunch.kernel)
    assert kernel.count("block_pair_barrier(") == 2
    first = kernel.index("block_pair_barrier(")
    second = kernel.index("block_pair_barrier(", first + 1)
    inbox_push = kernel.index("group_reduce_slot(reducer, destination) * self._world_size + self._rank")
    reducer = kernel.index("for reduced_index in cutlass.range_constexpr(len(self._reduced_shards))")
    partial_store = kernel.index("f32_as_u32(accum[0])")
    group_lse = kernel.index("group_lse = group_max + cute.math.log(")
    owner = kernel.index("# Phase two: owner combine")
    partial_scale = kernel.index("partial_scale = self._lse_weight(group_max_value, max_lse, natural_log) * inv_weight_sum")
    assert inbox_push < first < reducer < group_lse < partial_store < second < owner < partial_scale
    # The reducer weights its own row first, then the group peers (ring order).
    reducer_phase = kernel[first:second]
    assert reducer_phase.index("own_weight = weights[0]") < reducer_phase.index("peer_weight = weights[peer_index + 1]")
    # The owner normalizes by the total weight that includes the group's exp(lse_g - m).
    owner_phase = kernel[second:]
    assert "weight_sum += weight" in owner_phase and "inv_weight_sum = Float32(1.0) / fmax_f32(" in owner_phase
    assert "u32_as_f32(low[element])" in owner_phase


def test_group_launch_requires_tp9_and_push() -> None:
    with pytest.raises(ValueError, match="TP9 only"):
        cute_mod._LseGroupReduceLaunch(8, 0, "bf16", 256, True)
    launch = cute_mod._LseGroupReduceLaunch(9, 4, "bf16", 256, True)
    assert launch._push and launch._reduced_shards == (0, 8) and launch._group_peers == (5, 6, 7)


def test_group_launcher_key_and_prepared_registry() -> None:
    key = cute_mod._lse_group_launcher_key(9, 0, "bf16", 256, True)
    assert key[-1] == "group" and key[:2] == (9, 0)
    assert not cute_mod.is_lse_group_reduce_prepared(9, 0, "bf16", 256, True)


def test_env_opt_in_requires_push_transport(monkeypatch) -> None:
    monkeypatch.setenv("B12X_PCIE_DCP_LSE_GROUP_REDUCE", "1")
    monkeypatch.setenv("B12X_PCIE_DCP_A2A_TRANSPORT", "push")
    assert runtime_mod.lse_group_reduce_requested(9)
    assert not runtime_mod.lse_group_reduce_requested(8)
    monkeypatch.setenv("B12X_PCIE_DCP_A2A_TRANSPORT", "pull")
    assert not runtime_mod.lse_group_reduce_requested(9)
    monkeypatch.delenv("B12X_PCIE_DCP_LSE_GROUP_REDUCE")
    monkeypatch.setenv("B12X_PCIE_DCP_A2A_TRANSPORT", "push")
    assert not runtime_mod.lse_group_reduce_requested(9)


def test_runtime_launch_selects_the_group_kernel_when_regions_exist() -> None:
    src = inspect.getsource(runtime_mod.PCIeDCPA2A._launch_lse_reduce_scatter)
    assert "if self.lse_group_reduce:" in src and "lse_group_reduce(" in src
    prep = inspect.getsource(runtime_mod.PCIeDCPA2A.prepare_graph_lse_reduce_scatter)
    assert "_get_compiled_lse_group_reduce(" in prep
