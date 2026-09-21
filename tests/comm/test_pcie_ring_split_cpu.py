"""Split all-reduce of the lossless DMA ring (reduce-scatter, the caller's
in-place work on its owned rows, all-gather), proven on the CPU emulation:
the phase boundary adds only a join/fork, the owned rows are the fully
reduced chunk of the served schedule, the all-gather distributes what the
hook wrote, replay uses two graphs, and the schedule model finds no
unordered overlap."""
from __future__ import annotations

import pytest
import torch

from b12x.comm.pcie import pcie_dma
from b12x.comm.pcie.pcie_dma_reference import (
    lossless_access_plan,
    owned_chunk,
    reduce_scatter_op_count,
    ring_schedule,
    scratch_step_widths,
)
from tests.comm.pcie_dma_emulation import (
    EmulatedRing,
    install_cuda_fakes,
    ring_all_reduce_reference,
)

WORLD = 9
GRANULE_ROWS = 4
WIDTH = 64
ROWS = WORLD * GRANULE_ROWS * 2  # two granules per chunk
MAX_BYTES = 4 * ROWS * WIDTH * 2


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    install_cuda_fakes(monkeypatch)


def _inputs(seed: int) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [
        (torch.randn(ROWS, WIDTH, generator=gen) * (1 + rank)).to(torch.bfloat16)
        for rank in range(WORLD)
    ]


def _hook_for(rank: int, ring: pcie_dma.PCIeDmaAllReduce, seen: dict, expected: torch.Tensor):
    """Doubles the rank's owned rows (exact in bf16) after checking that
    they hold the fully reduced values; records the rows it touched."""
    blocks = ring.split_owned_rows(expected)
    assert blocks is not None

    def between(out: torch.Tensor) -> None:
        touched = []
        for row0, rows in blocks:
            assert torch.equal(out[row0 : row0 + rows], expected[row0 : row0 + rows]), (
                f"rank {rank} rows {row0}:{row0 + rows} are not the reduced values"
            )
            out[row0 : row0 + rows] *= 2
            touched.append((row0, rows))
        seen[rank] = touched

    return between


def test_owned_chunk_matches_the_schedule_and_the_granule_mapping() -> None:
    for world in (2, 4, 8, 9, 16):
        last = ring_schedule(world)[world - 2]
        assert last.reduce and last.step == world - 2
        for rank in range(world):
            assert owned_chunk(rank, world) == (rank + last.recv_offset) % world
            assert owned_chunk(rank, world) == (rank + 1) % world
    emu = EmulatedRing(WORLD, MAX_BYTES, graph_replay=False, granule_rows=GRANULE_ROWS)
    x = torch.zeros(ROWS, WIDTH, dtype=torch.bfloat16)
    for rank, ring in enumerate(emu.rings):
        blocks = ring.split_owned_rows(x)
        assert blocks == [
            (b * GRANULE_ROWS, GRANULE_ROWS)
            for b in range(ROWS // GRANULE_ROWS)
            if b % WORLD == (rank + 1) % WORLD
        ]
    # Every row is owned by exactly one rank.
    owners = torch.zeros(ROWS, dtype=torch.int64)
    for ring in emu.rings:
        for row0, rows in ring.split_owned_rows(x):
            owners[row0 : row0 + rows] += 1
    assert torch.equal(owners, torch.ones(ROWS, dtype=torch.int64))


def test_plan_splits_at_the_reduce_scatter_op_count() -> None:
    for pieces in (1, 2):
        for fp32_hops in (0, 1):
            offsets, _ = pcie_dma.PCIeDmaAllReduce._scratch_layout(1 << 16, WORLD, fp32_hops)
            plan = lossless_access_plan(
                world=WORLD, rank=3, shard_elems=8 * 72 * pieces, pieces=pieces,
                granule=True, fp32_hops=fp32_hops, elem_size=2, scratch_offsets=offsets,
            )
            split = reduce_scatter_op_count(WORLD, pieces)
            assert all(op.step < WORLD - 1 for op in plan[:split])
            assert all(op.step >= WORLD - 1 for op in plan[split:])
            assert len(plan) == 2 * split


@pytest.mark.parametrize("fp32_hops", [0, 1])
def test_eager_split_equals_all_reduce_then_hook(fp32_hops: int) -> None:
    emu = EmulatedRing(
        WORLD, MAX_BYTES, graph_replay=False, granule_rows=GRANULE_ROWS,
        fp32_hops=fp32_hops, model=True,
    )
    inputs = _inputs(seed=11)
    plain = EmulatedRing(WORLD, MAX_BYTES, graph_replay=False, granule_rows=GRANULE_ROWS, fp32_hops=fp32_hops)
    expected = plain.run(lambda ring, rank: ring.all_reduce(inputs[rank]))[0]
    seen: dict = {}

    def call(ring, rank):
        assert ring.can_all_reduce_split(inputs[rank])
        return ring.all_reduce_in_place_split(inputs[rank], _hook_for(rank, ring, seen, expected))

    outs = emu.run(call)
    assert emu.conflicts == [], emu.conflicts
    doubled = expected * 2
    for rank, out in enumerate(outs):
        assert torch.equal(out, doubled), f"rank {rank}"
    assert sorted(len(v) for v in seen.values()) == [2] * WORLD


def test_replay_split_uses_two_graphs_and_matches_eager() -> None:
    emu = EmulatedRing(WORLD, MAX_BYTES, granule_rows=GRANULE_ROWS, fp32_hops=1, model=True)
    inputs = _inputs(seed=12)
    plain = EmulatedRing(WORLD, MAX_BYTES, graph_replay=False, granule_rows=GRANULE_ROWS, fp32_hops=1)
    expected = plain.run(lambda ring, rank: ring.all_reduce(inputs[rank]))[0]
    seen: dict = {}

    def calls(ring, rank):
        hook = _hook_for(rank, ring, seen, expected)
        first = ring.all_reduce_in_place_split(inputs[rank], hook)
        entries_after_first = len(ring._replay_entries)
        second = ring.all_reduce_in_place_split(inputs[rank], hook)
        borrowed = ring.all_reduce_in_place_split(inputs[rank], hook, borrow_output=True)
        return first, entries_after_first, second, borrowed

    results = emu.run(calls)
    assert emu.conflicts == [], emu.conflicts
    doubled = expected * 2
    for rank, (first, entries_after_first, second, borrowed) in enumerate(results):
        assert entries_after_first == 0
        assert torch.equal(first, doubled), f"rank {rank} eager"
        assert torch.equal(second, doubled), f"rank {rank} replay"
        assert torch.equal(borrowed, doubled), f"rank {rank} borrowed"
        ring = emu.rings[rank]
        assert ring.is_ring_storage(borrowed)
        assert not ring.is_ring_storage(second)
        key = ("ars", inputs[rank].numel(), torch.bfloat16, GRANULE_ROWS * WIDTH)
        assert list(ring._replay_entries) == [key]
        entry = ring._replay_entries[key]
        assert entry.graph is not None and entry.graph_ag is not None
        assert entry.inp is entry.out


def test_split_and_plain_entries_do_not_share_keys() -> None:
    emu = EmulatedRing(WORLD, MAX_BYTES, granule_rows=GRANULE_ROWS, fp32_hops=1)
    inputs = _inputs(seed=13)
    plain = EmulatedRing(WORLD, MAX_BYTES, graph_replay=False, granule_rows=GRANULE_ROWS, fp32_hops=1)
    expected = plain.run(lambda ring, rank: ring.all_reduce(inputs[rank]))[0]

    def calls(ring, rank):
        hook = _hook_for(rank, ring, {}, expected)
        ring.all_reduce(inputs[rank]); ring.all_reduce(inputs[rank])
        ring.all_reduce_in_place_split(inputs[rank], hook)
        split = ring.all_reduce_in_place_split(inputs[rank], hook)
        whole = ring.all_reduce(inputs[rank])
        return split, whole

    results = emu.run(calls)
    for rank, (split, whole) in enumerate(results):
        assert torch.equal(split, expected * 2)
        assert torch.equal(whole, expected)
        keys = list(emu.rings[rank]._replay_entries)
        assert sorted(k[0] for k in keys) == ["ar", "ars"]


def test_split_rejects_unsupported_inputs() -> None:
    served = EmulatedRing(WORLD, MAX_BYTES, graph_replay=False)  # contiguous mapping, no fp32 hops
    x = torch.ones(ROWS, WIDTH, dtype=torch.bfloat16)
    assert served.rings[0].split_owned_rows(x) is None
    assert not served.rings[0].can_all_reduce_split(x)
    with pytest.raises(ValueError):
        served.rings[0].all_reduce_in_place_split(x, lambda out: None)
    granule = EmulatedRing(WORLD, MAX_BYTES, graph_replay=False, granule_rows=GRANULE_ROWS)
    # Row count not a multiple of world * granule rows: the ring falls back to
    # the contiguous mapping, which the split refuses.
    y = torch.ones(ROWS + GRANULE_ROWS, WIDTH, dtype=torch.bfloat16)
    assert granule.rings[0].split_owned_rows(y) is None
    assert not granule.rings[0].can_all_reduce_split(y)
    # 1-D inputs have no rows to own.
    assert granule.rings[0].split_owned_rows(x.reshape(-1)) is None
