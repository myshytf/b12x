"""CPU specification tests for an unserved K2 grouping experiment."""

from collections import defaultdict

from benchmarks.plan_kimi_reference_partition_groups import (
    plan_groups,
    plan_tasks,
    reference_intervals,
    validate_groups,
)


def test_intervals_match_independent_cta_stripe_simulation():
    # Exercise every served small decode count and both TP9 shard widths.
    grid = 186
    for tokens in range(1, 9):
        for n_tiles, k_tiles in ((6, 28), (4, 28), (28, 3), (28, 2)):
            total = tokens * 16 * n_tiles
            tail = total
            if total > grid:
                tail = total % grid
                if tail * 3 <= grid:
                    tail += grid
            full = total - tail
            stripe = (tail * k_tiles + grid - 1) // grid
            scheduled = defaultdict(list)
            for cta in range(grid):
                for tile in range(cta, full, grid):
                    scheduled[tile].append((0, k_tiles))
                # Simulate each CTA's linear K work, without using the
                # per-tile interval formula being tested.
                for linear in range(
                    cta * stripe, min((cta + 1) * stripe, tail * k_tiles)
                ):
                    tile, k = divmod(linear, k_tiles)
                    spans = scheduled[tile + full]
                    if spans and spans[-1][0] == cta:
                        spans[-1][2] = k + 1
                    else:
                        spans.append([cta, k, k + 1])
            for tile in range(total):
                if tile < full:
                    expected = tuple(scheduled[tile])
                else:
                    first_boundary = stripe * (
                        ((tile - full) * k_tiles + stripe - 1) // stripe
                    )
                    boundary_offset = first_boundary - (tile - full) * k_tiles
                    count = (k_tiles - boundary_offset + stripe - 1) // stripe + (
                        boundary_offset > 0
                    )
                    ordered = [None] * count
                    for cta, begin, end in scheduled[tile]:
                        delta = stripe * cta - first_boundary
                        if delta < 0 or (boundary_offset == 0 and delta == 0):
                            index = count - 1
                        else:
                            index = count - 1 - delta // stripe - (boundary_offset > 0)
                        assert ordered[index] is None
                        ordered[index] = (begin, end)
                    expected = tuple(ordered)
                assert reference_intervals(tile, total, k_tiles, grid) == expected


def test_task_order_cannot_block_its_own_predecessor():
    for n_tiles, k_tiles in ((6, 28), (4, 28), (28, 3), (28, 2)):
        groups = plan_groups(list(range(16)) * 4, n_tiles=n_tiles, k_tiles=k_tiles)
        tasks = plan_tasks(groups)
        for grid in (1, 4, 186):
            cursor = list(range(grid))
            completed = [0] * len(groups)
            remaining = len(tasks)
            while remaining:
                progress = 0
                for cta, index in enumerate(cursor):
                    if index >= len(tasks):
                        continue
                    task = tasks[index]
                    assert task.lock_slot == task.group_index
                    if completed[task.lock_slot] == task.partial_index:
                        completed[task.lock_slot] += 1
                        cursor[cta] += grid
                        progress += 1
                assert progress, (
                    "resident CTAs are blocked by an unscheduled predecessor"
                )
                remaining -= progress
            assert completed == [len(g.intervals) for g in groups]


def test_grouping_preserves_route_coverage_padding_and_association():
    for tokens in range(2, 9):
        for expert_ids in (
            list(range(16)) * tokens,
            list(range(tokens * 16)),
            list(range(16)) * (tokens - 1) + [-1] * 16,
        ):
            for n_tiles, k_tiles in ((6, 28), (4, 28), (28, 3), (28, 2)):
                groups = plan_groups(expert_ids, n_tiles=n_tiles, k_tiles=k_tiles)
                validate_groups(groups, expert_ids, n_tiles=n_tiles, k_tiles=k_tiles)
                assert (
                    sum(len(g.routes) for g in groups)
                    == sum(i >= 0 for i in expert_ids) * n_tiles
                )
