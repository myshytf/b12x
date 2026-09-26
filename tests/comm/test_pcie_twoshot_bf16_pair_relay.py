"""Pair-relay publish phase of the TP9 push all-reduce (CPU checks).

The relay table must hand every owner's reduced shard to every rank exactly
once (directly or through the rank's PIX partner), the kernel must gate its
publish stores and its phase-three source on that table, and the runtime must
select the relay kernels only when opted in and above the payload threshold.
"""

from __future__ import annotations

import inspect

import pytest

from b12x.comm.pcie import _twoshot_bf16_cute as cute_mod
from b12x.comm.pcie import pcie_twoshot_bf16 as runtime_mod
from b12x.comm.pcie._twoshot_bf16_cute import (
    PAIR_RELAY_SINGLE_RANK,
    PAIR_RELAY_WORLD_SIZE,
    get_twoshot_bf16_allreduce_launcher,
    pair_relay_direct,
    pair_relay_partner,
)

RANKS = range(PAIR_RELAY_WORLD_SIZE)


def test_pair_relay_partner_table() -> None:
    assert [pair_relay_partner(r) for r in range(8)] == [1, 0, 3, 2, 5, 4, 7, 6]
    assert pair_relay_partner(PAIR_RELAY_SINGLE_RANK) == PAIR_RELAY_WORLD_SIZE


def test_every_shard_reaches_every_rank_exactly_once() -> None:
    for owner in RANKS:
        for receiver in RANKS:
            if receiver == owner:
                continue
            direct = pair_relay_direct(owner, receiver)
            if receiver == PAIR_RELAY_SINGLE_RANK:
                assert direct, (owner, receiver)
                continue
            partner = pair_relay_partner(receiver)
            if partner == owner:
                # The owner's own partner always receives across the pair bridge.
                assert direct, (owner, receiver)
                continue
            via_partner = pair_relay_direct(owner, partner)
            # Exactly one member of the pair receives the shard over the fabric.
            assert direct != via_partner, (owner, receiver, direct, via_partner)


def test_relay_load_is_balanced_within_a_pair() -> None:
    for pair_base in range(0, PAIR_RELAY_SINGLE_RANK, 2):
        counts = []
        for member in (pair_base, pair_base + 1):
            owners = [
                o
                for o in RANKS
                if o not in (pair_base, pair_base + 1) and pair_relay_direct(o, member)
            ]
            counts.append(len(owners))
        assert sorted(counts) == [3, 4], (pair_base, counts)


def test_owner_publishes_to_five_ranks_and_rank_eight_to_four() -> None:
    for owner in RANKS:
        targets = [r for r in RANKS if r != owner and pair_relay_direct(owner, r)]
        expected = 4 if owner == PAIR_RELAY_SINGLE_RANK else 5
        assert len(targets) == expected, (owner, targets)


def test_device_predicate_mirrors_the_python_table() -> None:
    source = inspect.getsource(cute_mod._pair_relay_direct)
    # Same three clauses: rank 8, the owner's own partner, the chosen member.
    assert "PAIR_RELAY_SINGLE_RANK" in source
    assert "owner == partner" in source
    assert "chosen == receiver" in source
    assert "(owner + pair_index) % Int32(2)" in source


def test_push_kernel_gates_publish_and_phase_three_on_the_relay_table() -> None:
    kernel = inspect.getsource(cute_mod._TwoShotPushAllReduceLaunch.kernel)
    publish = kernel.index("_pair_relay_direct(local_rank, destination) == Int32(1)")
    second_barrier = kernel.index("self._barrier(signals, local_rank)", publish)
    await_flags = kernel.index("self._await_partner_flags(signals, local_rank)", second_barrier)
    partner_source = kernel.index("_pair_relay_direct(source_rank, local_rank) == Int32(0)", await_flags)
    assert publish < second_barrier < await_flags < partner_source
    # The relayed source is the partner's staging at the owner's slot.
    assert "_select_address(\n                            staging, _pair_relay_partner(local_rank)\n                        )" in kernel
    assert kernel.count("self._barrier(signals, local_rank)") == 2
    # The non-relay build keeps the served publish path (compile-time branch).
    assert "if cutlass.const_expr(self._pair_relay):" in kernel


def test_partner_flag_wait_polls_the_partner_signal_of_each_relayed_owner() -> None:
    source = inspect.getsource(cute_mod._TwoShotPushAllReduceLaunch._await_partner_flags)
    assert "_ld_relaxed_sys_u32(flag_address)" in source
    assert "self._select_address(\n                        signals, _pair_relay_partner(local_rank)\n                    )" in source
    assert "Int64(source) * Int64(_FLAG_STRIDE)" in source
    assert "value % Uint32(2)" in source
    assert source.rstrip().endswith("cute.arch.barrier()")


def test_launcher_rejects_relay_outside_tp9() -> None:
    with pytest.raises(ValueError, match="TP9 only"):
        get_twoshot_bf16_allreduce_launcher(8, 0, True, 0, 512, 8, 0, mode="push_relay")
    with pytest.raises(ValueError, match="invalid all-reduce mode"):
        get_twoshot_bf16_allreduce_launcher(9, 0, True, 0, 512, 8, 0, mode="relay")
    with pytest.raises(ValueError, match="TP9 only"):
        cute_mod._TwoShotPushAllReduceLaunch(8, 0, True, 0, 512, 8, pair_relay=True)


def _runtime(*, mode: str, static: bool, relay: bool, min_packs: int):
    runtime = object.__new__(runtime_mod.PCIeTwoShotBF16)
    runtime.all_reduce_mode = mode
    runtime.world_size = 9
    runtime._static_peers_enabled = static
    runtime._pair_relay_enabled = relay
    runtime._pair_relay_min_packs = min_packs
    return runtime


@pytest.mark.parametrize(
    ("static", "relay", "rows_per_rank", "expected"),
    [
        (False, False, 100, "push"),
        (True, False, 100, "push_static"),
        (True, False, 1000, "push"),
        (False, True, 100, "push"),  # 900 packs < 7168
        (False, True, 1000, "push_relay"),
        (True, True, 796, "push_static_relay"),  # 796 x 9 + 4 = 7168 packs: static and relay
    ],
)
def test_mode_selection(static, relay, rows_per_rank, expected) -> None:
    runtime = _runtime(mode="push", static=static, relay=relay, min_packs=7168)
    remainder = 4 if rows_per_rank == 796 else 0  # 796 * 9 + 4 = 7168 exactly
    assert runtime._all_reduce_kernel_mode(rows_per_rank, remainder) == expected


def test_mode_selection_keeps_pull_and_lists_every_compiled_mode() -> None:
    assert _runtime(mode="pull", static=True, relay=True, min_packs=0)._all_reduce_kernel_mode(1, 0) == "pull"
    assert _runtime(mode="push", static=True, relay=True, min_packs=0).all_reduce_kernel_modes() == (
        "push",
        "push_static",
        "push_relay",
        "push_static_relay",
    )
    assert _runtime(mode="push", static=False, relay=True, min_packs=0).all_reduce_kernel_modes() == (
        "push",
        "push_relay",
    )
    assert _runtime(mode="pull", static=True, relay=True, min_packs=0).all_reduce_kernel_modes() == ("pull",)


def test_relay_opt_in_reads_the_environment(monkeypatch) -> None:
    monkeypatch.delenv("B12X_PCIE_TP9_PAIR_RELAY", raising=False)
    assert not runtime_mod.pair_relay_requested(9, 8)
    monkeypatch.setenv("B12X_PCIE_TP9_PAIR_RELAY", "1")
    assert runtime_mod.pair_relay_requested(9, 8)
    assert not runtime_mod.pair_relay_requested(8, 8)
    assert not runtime_mod.pair_relay_requested(9, 7168)
    monkeypatch.delenv("B12X_PCIE_TP9_PAIR_RELAY_MIN_PACKS", raising=False)
    assert runtime_mod.pair_relay_min_packs() == 7168
    monkeypatch.setenv("B12X_PCIE_TP9_PAIR_RELAY_MIN_PACKS", "0")
    assert runtime_mod.pair_relay_min_packs() == 0
    monkeypatch.setenv("B12X_PCIE_TP9_PAIR_RELAY_MIN_PACKS", "-1")
    with pytest.raises(ValueError):
        runtime_mod.pair_relay_min_packs()


def _fake_sysfs(tmp_path, chains):
    """chains: identity -> list of bridge names from the root; creates realpath targets."""
    devices = tmp_path / "devices"
    devices.mkdir()
    for (domain, bus, dev), chain in chains.items():
        target = tmp_path / "sys"
        for part in chain:
            target = target / part
        name = f"{domain:04x}:{bus:02x}:{dev:02x}.0"
        target = target / name
        target.mkdir(parents=True)
        (devices / name).symlink_to(target)
    return str(devices)


def test_topology_check_accepts_the_served_placement(tmp_path) -> None:
    # Ranks 0..7: pairs behind bridges a, e, 17, 1b (grandparent of the GPU); rank 8 own bridge 07.
    chains = {}
    pair_bridges = ["0a", "0e", "17", "1b"]
    for rank in range(8):
        bridge = pair_bridges[rank // 2]
        chains[(0, 0x10 + rank, 0)] = ["root", f"{bridge}:00.0", f"{bridge}b:{rank % 2}0.0"]
    chains[(0, 0x09, 0)] = ["root", "07:00.0", "08:10.0"]
    root = _fake_sysfs(tmp_path, chains)
    identities = [(0, 0x10 + r, 0) for r in range(8)] + [(0, 0x09, 0)]
    assert runtime_mod.check_pair_relay_topology(identities, root) is True


def test_topology_check_rejects_a_split_pair_and_skips_without_sysfs(tmp_path) -> None:
    chains = {}
    for rank in range(8):
        bridge = ["0a", "0e", "17", "1b"][rank // 2]
        if rank == 3:
            bridge = "zz"  # rank 3 behind a different bridge than rank 2
        chains[(0, 0x10 + rank, 0)] = ["root", f"{bridge}:00.0", f"{bridge}b:{rank % 2}0.0"]
    chains[(0, 0x09, 0)] = ["root", "07:00.0", "08:10.0"]
    root = _fake_sysfs(tmp_path, chains)
    identities = [(0, 0x10 + r, 0) for r in range(8)] + [(0, 0x09, 0)]
    with pytest.raises(ValueError, match="ranks 2 and 3 are not a PCIe pair"):
        runtime_mod.check_pair_relay_topology(identities, root)
    assert runtime_mod.check_pair_relay_topology(identities, str(tmp_path / "missing")) is False
    with pytest.raises(ValueError, match="expects 9 ranks"):
        runtime_mod.check_pair_relay_topology(identities[:8], root)
