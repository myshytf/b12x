"""Research-only plan for sharing weight tiles without changing K sums.

The deployed small-M schedule stripes the tail of the logical MN tile grid
across resident CTAs. A route may share an M8 tile with another route only if
their expert, N tile and ordered K intervals match. This module describes
those groups; it is not a GPU builder or a serving implementation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteGroup:
    expert: int
    n_tile: int
    routes: tuple[int, ...]
    intervals: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class GemmTask:
    group_index: int
    k_begin: int
    k_count: int
    partial_index: int
    partial_count: int
    lock_slot: int


def plan_tasks(groups: tuple[RouteGroup, ...]) -> tuple[GemmTask, ...]:
    """Emit predecessor-first tasks with a distinct lock for every group."""
    return tuple(
        GemmTask(
            group_index, begin, end - begin, partial, len(group.intervals), group_index
        )
        for group_index, group in enumerate(groups)
        for partial, (begin, end) in enumerate(group.intervals)
    )


def reference_intervals(
    tile: int, total_tiles: int, k_tiles: int, resident_ctas: int
) -> tuple[tuple[int, int], ...]:
    """Return one MN tile's K intervals in deployed partial-merge order."""
    if not (0 <= tile < total_tiles and k_tiles > 0 and resident_ctas > 0):
        raise ValueError("invalid reference tile geometry")
    tail = total_tiles
    full = 0
    if total_tiles > resident_ctas:
        tail = total_tiles % resident_ctas
        if tail * 3 <= resident_ctas:
            tail += resident_ctas
        full = total_tiles - tail
    if tile < full:
        return ((0, k_tiles),)
    stripe = (tail * k_tiles + resident_ctas - 1) // resident_ctas
    begin = (tile - full) * k_tiles
    end = begin + k_tiles
    intervals = tuple(
        (max(begin, cta * stripe) - begin, min(end, (cta + 1) * stripe) - begin)
        for cta in range(begin // stripe, (end + stripe - 1) // stripe)
    )
    # The first K slice has reduce_slice_idx == reduce_slice_count - 1.
    return tuple(reversed(intervals))


def plan_groups(
    expert_ids: list[int],
    *,
    n_tiles: int,
    k_tiles: int,
    resident_ctas: int = 186,
    num_experts: int = 896,
    block_rows: int = 8,
) -> tuple[RouteGroup, ...]:
    """Group valid routes by their complete numerical reduction contract.

    Original flattened route IDs remain the output indices. Negative IDs are
    padding; the existing ordered top-k reducer ignores them. FP32 partials
    must be merged in the listed interval order, with the deployed intra-CTA
    K slices, MMA shape, scale application and FP16 store rounding retained.
    """
    if n_tiles < 1 or block_rows < 1 or num_experts < 1:
        raise ValueError("invalid grouping geometry")
    buckets = defaultdict(list)
    total_tiles = len(expert_ids) * n_tiles
    for route, expert in enumerate(expert_ids):
        if not 0 <= expert < num_experts:
            continue
        for n_tile in range(n_tiles):
            intervals = reference_intervals(
                route * n_tiles + n_tile, total_tiles, k_tiles, resident_ctas
            )
            buckets[expert, n_tile, intervals].append(route)
    groups = []
    for (expert, n_tile, intervals), routes in buckets.items():
        for start in range(0, len(routes), block_rows):
            groups.append(
                RouteGroup(
                    expert, n_tile, tuple(routes[start : start + block_rows]), intervals
                )
            )
    return tuple(groups)


def validate_groups(
    groups,
    expert_ids,
    *,
    n_tiles,
    k_tiles,
    resident_ctas=186,
    num_experts=896,
    block_rows=8,
):
    """Check coverage and K association independently of GPU launch order."""
    observed = set()
    for group in groups:
        assert 1 <= len(group.routes) <= block_rows
        for route in group.routes:
            key = (route, group.n_tile)
            assert key not in observed
            observed.add(key)
            assert expert_ids[route] == group.expert
            assert group.intervals == reference_intervals(
                route * n_tiles + group.n_tile,
                len(expert_ids) * n_tiles,
                k_tiles,
                resident_ctas,
            )
    expected = {
        (route, n)
        for route, expert in enumerate(expert_ids)
        if 0 <= expert < num_experts
        for n in range(n_tiles)
    }
    assert observed == expected
