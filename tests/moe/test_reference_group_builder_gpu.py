"""GPU metadata must reproduce the deployed partition contract after replay."""

import os

import pytest
import torch

from benchmarks.plan_kimi_reference_partition_groups import plan_groups
from b12x._lib.utils import current_cuda_stream
from b12x.moe._shared.kernels.w4a16.reference_grouped import (
    HEADER_WORDS,
    MAX_PARTIALS,
    MAX_ROUTES,
    TASK_WORDS,
    build_reference_groups,
    workspace_layout,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("B12X_RUN_REFERENCE_GROUP_TEST") != "1",
    reason="set B12X_RUN_REFERENCE_GROUP_TEST=1 on an isolated CUDA device",
)


@pytest.mark.parametrize("rows", range(2, 9))
@pytest.mark.parametrize("width", (256, 384))
@pytest.mark.parametrize("include_fc1", (False, True))
@pytest.mark.parametrize("backend", ("standalone", "inline-token", "inline-route"))
def test_builder_replays_exact_reference_partitions(rows, width, include_fc1, backend):
    if include_fc1 and backend != "standalone":
        pytest.skip("The fused candidate prepares FC2 only")
    torch.manual_seed(20260914 + rows + width)
    layout = workspace_layout(width)
    workspace = torch.full((layout.words,), 123456789, dtype=torch.int32, device="cuda")
    ids = torch.zeros((rows * 16,), dtype=torch.int32, device="cuda")
    mapping = torch.arange(896, dtype=torch.int32, device="cuda")
    addresses = ids.data_ptr(), mapping.data_ptr(), workspace.data_ptr()
    if backend != "standalone":
        from benchmarks.inline_group_metadata_probe import LUT_WORDS, get_inline_probe

        lut = torch.randint(
            -(2**31), 2**31 - 1, (LUT_WORDS,), dtype=torch.int32, device="cuda"
        )
        copied_lut = torch.empty((28 * LUT_WORDS,), dtype=torch.int32, device="cuda")
        probe = get_inline_probe(rows, width, 7 if backend == "inline-token" else 112)

        # The in-kernel body must leave the resident-grid header untouched.
        probe(ids, mapping, workspace, lut, copied_lut, current_cuda_stream())
        torch.accelerator.synchronize()
        assert (workspace[:HEADER_WORDS] == 123456789).all()
        assert torch.equal(copied_lut.view(28, LUT_WORDS), lut.expand(28, LUT_WORDS))

    def run():
        if backend != "standalone":
            workspace[:HEADER_WORDS].zero_()
            probe(ids, mapping, workspace, lut, copied_lut, current_cuda_stream())
            return
        build_reference_groups(
            ids,
            mapping,
            workspace,
            rows=rows,
            width=width,
            grid=186,
            stream=current_cuda_stream(),
            include_fc1=include_fc1,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # Includes repeated IDs beyond one M8 group, distinct experts, shared
    # experts, invalid routes and changed global-to-local mapping.
    cases = [
        torch.zeros(rows * 16, dtype=torch.int32),
        torch.arange(rows * 16, dtype=torch.int32),
        torch.arange(16, dtype=torch.int32).repeat(rows),
        torch.randint(0, 24, (rows * 16,), dtype=torch.int32),
    ]
    invalid = cases[-1].clone()
    invalid[::7] = -1
    invalid[1::11] = 900
    cases.append(invalid)
    for index, case in enumerate(cases):
        local_map = torch.arange(896, dtype=torch.int32)
        if index == len(cases) - 1:
            local_map[::3] = -1
            local_map[1::3] //= 2
        ids.copy_(case)
        mapping.copy_(local_map)
        workspace.fill_(123456789)
        graph.replay()
        torch.accelerator.synchronize()
        if backend != "standalone":
            assert torch.equal(
                copied_lut.view(28, LUT_WORDS), lut.expand(28, LUT_WORDS)
            )
        local_ids = [int(local_map[x]) if 0 <= x < 896 else -1 for x in case.tolist()]
        for n_tiles, k_tiles, route_offset, task_offset, count_offset in (
            (
                width * 2 // 128,
                28,
                layout.fc1_routes,
                layout.fc1_tasks,
                layout.fc1_counts,
            ),
            (28, width // 128, layout.fc2_routes, layout.fc2_tasks, layout.fc2_counts),
        ):
            if not include_fc1 and route_offset == layout.fc1_routes:
                assert (
                    workspace[layout.fc1_routes : layout.fc2_routes] == 123456789
                ).all()
                continue
            group_capacity = rows * 16 * n_tiles
            raw_routes = (
                workspace[route_offset : route_offset + group_capacity * 8]
                .cpu()
                .view(group_capacity, 8)
            )
            counts = workspace[count_offset : count_offset + n_tiles].cpu().tolist()
            by_group = {}
            for n_tile, count in enumerate(counts):
                assert 0 <= count <= rows * 16 * MAX_PARTIALS
                start = task_offset + n_tile * MAX_ROUTES * MAX_PARTIALS * TASK_WORDS
                tasks = (
                    workspace[start : start + count * TASK_WORDS]
                    .cpu()
                    .view(count, TASK_WORDS)
                    .tolist()
                )
                for task in tasks:
                    assert task[0] >= 0 and task[2] == n_tile
                    by_group.setdefault(task[1], []).append(task)
            actual = set()
            for group, parts in by_group.items():
                first = parts[0]
                count = first[6]
                assert 1 <= count <= MAX_PARTIALS and len(parts) == count
                intervals = []
                for part, task in enumerate(parts):
                    assert task[:3] == first[:3]
                    assert task[5:8] == [part, count, group]
                    intervals.append((task[3], task[3] + task[4]))
                members = tuple(r for r in raw_routes[group].tolist() if r < rows * 16)
                actual.add((first[0], first[2], members, tuple(intervals)))
            expected = {
                (g.expert, g.n_tile, g.routes, g.intervals)
                for g in plan_groups(local_ids, n_tiles=n_tiles, k_tiles=k_tiles)
            }
            assert actual == expected
        assert (workspace[:HEADER_WORDS] == 0).all()
        assert (workspace[layout.locks : layout.locks + rows * 16 * 28] == 0).all()
        assert (workspace[layout.partials : layout.partials + 32] == 123456789).all()
        assert addresses == (ids.data_ptr(), mapping.data_ptr(), workspace.data_ptr())
