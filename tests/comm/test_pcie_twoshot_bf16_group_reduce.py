"""Group-reduce reduce-scatter of the TP9 push all-reduce (CPU checks).

Role tables (every shard has one reducer in the other switch group, reducer load
is balanced, rank 8 reduces nothing), the staging layout's inbox and partial
regions, the kernel's phase structure (three barriers, inbox pushes, fp32
partial exchange, owner sum order), mode selection and the switch-group
topology check.
"""

from __future__ import annotations

import inspect

import pytest

from b12x.comm.pcie import _twoshot_bf16_cute as cute_mod
from b12x.comm.pcie import pcie_twoshot_bf16 as runtime_mod
from b12x.comm.pcie._twoshot_bf16_cute import (
    GROUP_REDUCE_MAX_SLOTS,
    GROUP_REDUCE_WORLD_SIZE,
    get_twoshot_bf16_allreduce_launcher,
    group_reduce_group,
    group_reduce_peers,
    group_reduce_reducer,
    group_reduce_shards,
    group_reduce_slot,
)

RANKS = range(GROUP_REDUCE_WORLD_SIZE)


def test_groups_follow_the_switch_placement() -> None:
    assert [group_reduce_group(r) for r in RANKS] == [0, 0, 0, 0, 1, 1, 1, 1, 0]
    with pytest.raises(ValueError):
        group_reduce_group(9)


def test_every_shard_has_one_reducer_in_the_other_group() -> None:
    for shard in RANKS:
        reducer = group_reduce_reducer(shard)
        assert group_reduce_group(reducer) != group_reduce_group(shard), shard
        assert shard in group_reduce_shards(reducer)
        assert sum(shard in group_reduce_shards(r) for r in RANKS) == 1
    assert [group_reduce_reducer(s) for s in RANKS] == [4, 5, 6, 7, 0, 1, 2, 3, 4]


def test_reducer_load_is_balanced_and_rank_eight_is_spared() -> None:
    loads = {r: len(group_reduce_shards(r)) for r in RANKS}
    assert loads == {0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 1, 6: 1, 7: 1, 8: 0}
    assert max(loads.values()) <= GROUP_REDUCE_MAX_SLOTS
    assert group_reduce_shards(4) == (0, 8)
    assert group_reduce_slot(4, 8) == 1 and group_reduce_slot(4, 0) == 0
    with pytest.raises(ValueError):
        group_reduce_slot(8, 0)


def test_group_peers_keep_the_ring_order_without_the_other_group() -> None:
    assert group_reduce_peers(0) == (1, 2, 3, 8)
    assert group_reduce_peers(8) == (0, 1, 2, 3)
    assert group_reduce_peers(4) == (5, 6, 7)
    assert group_reduce_peers(7) == (4, 5, 6)
    for r in RANKS:
        peers = group_reduce_peers(r)
        assert r not in peers
        assert all(group_reduce_group(p) == group_reduce_group(r) for p in peers)
        # own + group peers + the other group's partial cover all nine contributions
        assert 1 + len(peers) + (GROUP_REDUCE_WORLD_SIZE - 1 - len(peers)) == GROUP_REDUCE_WORLD_SIZE


def test_layout_adds_inbox_and_partial_regions_inside_the_slot() -> None:
    layout = runtime_mod._make_layout(max_rows=49149, row_elems=8, world_size=9)
    stride = layout.pack_stride
    assert layout.reduced_offset >= 9 * stride * 16
    assert layout.inbox_offset >= layout.reduced_offset + 9 * stride * 16
    assert layout.partial_offset >= layout.inbox_offset + GROUP_REDUCE_MAX_SLOTS * 9 * stride * 16
    assert layout.slot_bytes >= layout.partial_offset + stride * 32
    for offset in (layout.reduced_offset, layout.inbox_offset, layout.partial_offset, layout.slot_bytes):
        assert offset % runtime_mod.IPC_SLAB_ALIGNMENT == 0
    assert layout.slab_bytes == layout.signal_bytes + 2 * layout.slot_bytes


def test_kernel_phases_and_sum_order() -> None:
    kernel = inspect.getsource(cute_mod._TwoShotGroupReduceLaunch.kernel)
    assert kernel.count("self._barrier(signals, local_rank)") == 3
    first = kernel.index("self._barrier(signals, local_rank)")
    second = kernel.index("self._barrier(signals, local_rank)", first + 1)
    third = kernel.index("self._barrier(signals, local_rank)", second + 1)
    inbox_push = kernel.index("group_reduce_slot(reducer, destination) * self._world_size + self._rank")
    reducer_sum = kernel.index("for reduced_index in cutlass.range_constexpr(len(self._reduced_shards))")
    partial_store = kernel.index("f32_as_u32(accumulator[0])")
    owner_partial = kernel.index("self._accumulate_f32_half(")
    publish = kernel.index("self._publish_reduced_pack(")
    copy_out = kernel.index("# Phase three")
    assert inbox_push < first < reducer_sum < partial_store < second < owner_partial < publish < third < copy_out
    # Owner order: own contribution, then group peers (ring order), then the fp32 partial.
    owner_phase = kernel[second:third]
    own = owner_phase.index("self._accumulate_words(accumulator, local_words)")
    peers = owner_phase.index("for peer_index in cutlass.range_constexpr(len(self._group_peers))")
    partial = owner_phase.index("self._accumulate_f32_half(")
    assert own < peers < partial
    # Reducer order: own contribution first, then group peers.
    reducer_phase = kernel[first:second]
    assert reducer_phase.index("self._accumulate_words(accumulator, own_words)") < reducer_phase.index(
        "for peer_index in cutlass.range_constexpr(len(self._group_peers))"
    )
    assert "static_peers=True" in inspect.getsource(cute_mod._TwoShotGroupReduceLaunch.__init__)


def test_launcher_modes_and_rejections() -> None:
    assert "push_group" in cute_mod._ALL_REDUCE_MODES and "push_group_relay" in cute_mod._ALL_REDUCE_MODES
    with pytest.raises(ValueError, match="TP9 only"):
        get_twoshot_bf16_allreduce_launcher(8, 0, True, 0, 512, 8, 0, mode="push_group")
    with pytest.raises(ValueError, match="TP9 only"):
        cute_mod._TwoShotGroupReduceLaunch(8, 0, True, 0, 512, 8)
    launch = cute_mod._TwoShotGroupReduceLaunch(9, 4, True, 0, 512, 8, pair_relay=True)
    assert launch._static_peers and launch._pair_relay and launch._reduced_shards == (0, 8)


def _runtime(*, static: bool, relay: bool, group: bool, min_packs: int = 7168):
    runtime = object.__new__(runtime_mod.PCIeTwoShotBF16)
    runtime.all_reduce_mode = "push"
    runtime.world_size = 9
    runtime._static_peers_enabled = static
    runtime._pair_relay_enabled = relay
    runtime._pair_relay_min_packs = 7168
    runtime._group_reduce_enabled = group
    runtime._group_reduce_min_packs = min_packs
    return runtime


def test_mode_selection_prefers_group_reduce_above_the_threshold() -> None:
    r = _runtime(static=True, relay=True, group=True)
    assert r._all_reduce_kernel_mode(100, 0) == "push_static"  # 900 packs: below both thresholds
    assert r._all_reduce_kernel_mode(796, 4) == "push_group_relay"  # 7168 packs
    assert r._all_reduce_kernel_mode(2000, 0) == "push_group_relay"
    assert _runtime(static=True, relay=False, group=True)._all_reduce_kernel_mode(2000, 0) == "push_group"
    assert _runtime(static=True, relay=True, group=False)._all_reduce_kernel_mode(2000, 0) == "push_relay"
    assert r.all_reduce_kernel_modes() == ("push", "push_static", "push_relay", "push_static_relay", "push_group_relay")
    assert _runtime(static=False, relay=False, group=True).all_reduce_kernel_modes() == ("push", "push_group")


def test_group_reduce_opt_in_reads_the_environment(monkeypatch) -> None:
    monkeypatch.delenv("B12X_PCIE_TP9_GROUP_REDUCE", raising=False)
    assert not runtime_mod.group_reduce_requested(9, 8)
    monkeypatch.setenv("B12X_PCIE_TP9_GROUP_REDUCE", "1")
    assert runtime_mod.group_reduce_requested(9, 8)
    assert not runtime_mod.group_reduce_requested(8, 8)
    monkeypatch.setenv("B12X_PCIE_TP9_GROUP_REDUCE_MIN_PACKS", "0")
    assert runtime_mod.group_reduce_min_packs() == 0


def _chains(tmp_path, bridges_by_rank):
    """bridges_by_rank: rank -> list of bridge names below a shared 'root/01:00.0'."""
    devices = tmp_path / "devices"
    devices.mkdir()
    identities = []
    for rank, chain in bridges_by_rank.items():
        ident = (0, 0x10 + rank, 0)
        identities.append(ident)
        target = tmp_path / "sys" / "root" / "01:00.0"
        for part in chain:
            target = target / part
        name = f"{ident[0]:04x}:{ident[1]:02x}:{ident[2]:02x}.0"
        (target / name).mkdir(parents=True)
        (devices / name).symlink_to(target / name)
    return str(devices), identities


def _served_like_chains():
    # Switch #2 (behind 02:00.0/03:00.0) hosts pairs (0,1) (2,3) and rank 8;
    # switch #1's ports 02:04.0 / 02:08.0 host pairs (4,5) (6,7).
    chains = {}
    for rank in range(4):
        pair = ["02:00.0", "03:00.0", ["06:04.0", "06:08.0"][rank // 2], ["0a:00.0", "0e:00.0"][rank // 2]]
        chains[rank] = pair + [f"0b:{rank % 2}0.0"]
    chains[8] = ["02:00.0", "03:00.0", "06:00.0", "07:00.0", "08:10.0"]
    for rank in range(4, 8):
        port = ["02:04.0", "02:08.0"][(rank - 4) // 2]
        bridge = ["17:00.0", "1b:00.0"][(rank - 4) // 2]
        chains[rank] = [port, bridge, f"18:{(rank - 4) % 2}0.0"]
    return chains


def test_group_topology_check_accepts_the_served_placement(tmp_path) -> None:
    root, identities = _chains(tmp_path, _served_like_chains())
    identities.sort(key=lambda i: i[1])
    assert runtime_mod.check_group_reduce_topology(identities, root) is True


def test_group_topology_check_rejects_a_group_b_rank_behind_the_cascade(tmp_path) -> None:
    chains = _served_like_chains()
    chains[6] = ["02:00.0", "03:00.0", "06:0c.0", "12:00.0", "13:00.0"]  # rank 6 moved behind switch #2
    root, identities = _chains(tmp_path, chains)
    identities.sort(key=lambda i: i[1])
    with pytest.raises(ValueError, match="rank 6 sits behind the cascaded switch"):
        runtime_mod.check_group_reduce_topology(identities, root)
    assert runtime_mod.check_group_reduce_topology(identities, str(tmp_path / "missing")) is False
