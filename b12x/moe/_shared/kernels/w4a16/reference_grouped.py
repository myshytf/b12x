"""Research-only GPU tasks that retain the direct route K partitions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream

ABI_VERSION = 4
MAX_ROUTES = 128
MAX_PARTIALS = 4
TASK_WORDS = 8
HEADER_WORDS = 1024
PARTIAL_WORDS_PER_GROUP = 16 * 128


def requested() -> bool:
    return os.environ.get("B12X_W4A16_REFERENCE_GROUPED", "0") == "1"


@dataclass(frozen=True)
class WorkspaceLayout:
    fc1_routes: int
    fc1_tasks: int
    fc1_counts: int
    fc2_routes: int
    fc2_tasks: int
    fc2_counts: int
    locks: int
    partials: int
    words: int


@cache
def workspace_layout(intermediate_size: int) -> WorkspaceLayout:
    if intermediate_size not in (256, 384):
        raise ValueError("reference grouping requires a qualified TP9 shard width")
    first_groups = MAX_ROUTES * (intermediate_size * 2 // 128)
    second_groups = MAX_ROUTES * 28
    first_routes = HEADER_WORDS
    first_tasks = first_routes + first_groups * 8
    first_counts = first_tasks + first_groups * MAX_PARTIALS * TASK_WORDS
    second_routes = (first_counts + intermediate_size * 2 // 128 + 3) // 4 * 4
    second_tasks = second_routes + second_groups * 8
    second_counts = second_tasks + second_groups * MAX_PARTIALS * TASK_WORDS
    locks = (second_counts + 28 + 3) // 4 * 4
    partials = (locks + second_groups + 3) // 4 * 4
    return WorkspaceLayout(
        first_routes,
        first_tasks,
        first_counts,
        second_routes,
        second_tasks,
        second_counts,
        locks,
        partials,
        partials + second_groups * PARTIAL_WORDS_PER_GROUP,
    )


@cute.jit
def emit_reference_group_tile(
    ids: cute.Tensor,
    expert_map: cute.Tensor,
    workspace: cute.Tensor,
    scratch: cute.Tensor,
    rows: Int32,
    resident_ctas: Int32,
    block: Int32,
    tid: Int32,
    width: cutlass.Constexpr,
):
    """Build one logical N tile with CTA-local temporary storage.

    Every thread in the containing CTA must call this function. Only the first
    128 threads own route records, allowing the identical integer algorithm
    in both the standalone 128-thread and fused 256-thread launches. The
    resident-grid header belongs to the caller and is never cleared here.
    """
    n1 = width * 2 // 128
    layout = workspace_layout(width)
    routes = rows * Int32(16)
    n_tiles = Int32(n1)
    n_tile = Int32(block)
    k_tiles = Int32(28)
    route_offset = Int64(layout.fc1_routes)
    task_offset = Int64(layout.fc1_tasks)
    count_offset = Int64(layout.fc1_counts)
    if block >= Int32(n1):
        n_tiles = Int32(28)
        n_tile = Int32(block) - Int32(n1)
        k_tiles = Int32(width // 128)
        route_offset = Int64(layout.fc2_routes)
        task_offset = Int64(layout.fc2_tasks)
        count_offset = Int64(layout.fc2_counts)

    total = routes * n_tiles
    tail = total
    if total > resident_ctas:
        tail = total % resident_ctas
        if tail * Int32(3) <= resident_ctas:
            tail += resident_ctas
    full = total - tail
    stripe = (tail * k_tiles + (resident_ctas - Int32(1))) // resident_ctas
    expert = Int32(-1)
    first_end = k_tiles
    if tid < routes:
        global_expert = ids[Int64(tid)].to(Int32)
        if global_expert >= Int32(0) and global_expert < Int32(896):
            expert = expert_map[Int64(global_expert)].to(Int32)
        if expert < Int32(0) or expert >= Int32(896):
            expert = Int32(-1)
        tile = Int32(tid) * n_tiles + n_tile
        if tile >= full:
            begin = (tile - full) * k_tiles
            first_end = cutlass.min(
                k_tiles,
                (begin // stripe + Int32(1)) * stripe - begin,
            )
        group = Int32(tid) * n_tiles + n_tile
        # FC2's group-index range contains FC1's; one writer per lock.
        if block >= Int32(n1):
            workspace[Int64(layout.locks) + Int64(group)] = Int32(0)
    if tid < Int32(MAX_ROUTES):
        scratch[tid] = expert
        scratch[Int32(MAX_ROUTES) + tid] = first_end
    cute.arch.sync_threads()

    parts = Int32(0)
    if tid < routes and expert >= Int32(0):
        predecessors = Int32(0)
        previous = Int32(0)
        while previous < Int32(tid):
            if (
                scratch[previous] == expert
                and scratch[Int32(128) + previous] == first_end
            ):
                predecessors += Int32(1)
            previous += Int32(1)
        if predecessors % Int32(8) == Int32(0):
            parts = Int32(1)
            if first_end < k_tiles:
                parts += (k_tiles - first_end + stripe - Int32(1)) // stripe
    if tid < Int32(MAX_ROUTES):
        scratch[Int32(2 * MAX_ROUTES) + tid] = parts
    cute.arch.sync_threads()
    if tid == Int32(0):
        total_tasks = Int32(0)
        row = Int32(0)
        while row < routes:
            total_tasks += scratch[Int32(256) + row]
            row += Int32(1)
        workspace[count_offset + Int64(n_tile)] = total_tasks
    if parts > Int32(0):
        prefix = Int32(0)
        previous = Int32(0)
        while previous < Int32(tid):
            prefix += scratch[Int32(256) + previous]
            previous += Int32(1)
        group = Int32(tid) * n_tiles + n_tile
        group_start = route_offset + Int64(group) * Int64(8)
        for slot in cutlass.range_constexpr(8):
            workspace[group_start + Int64(slot)] = routes
        count = Int32(0)
        other = Int32(tid)
        while other < routes and count < Int32(8):
            if scratch[other] == expert and scratch[Int32(128) + other] == first_end:
                workspace[group_start + Int64(count)] = other
                count += Int32(1)
            other += Int32(1)
        begin = Int32(0)
        end = first_end
        for part in cutlass.range_constexpr(MAX_PARTIALS):
            if Int32(part) < parts:
                merge_index = parts - Int32(1) - Int32(part)
                offset = task_offset + (
                    Int64(n_tile) * Int64(MAX_ROUTES * MAX_PARTIALS)
                    + Int64(prefix)
                    + Int64(merge_index)
                ) * Int64(TASK_WORDS)
                workspace[offset] = expert
                workspace[offset + Int64(1)] = group
                workspace[offset + Int64(2)] = n_tile
                workspace[offset + Int64(3)] = begin
                workspace[offset + Int64(4)] = end - begin
                workspace[offset + Int64(5)] = merge_index
                workspace[offset + Int64(6)] = parts
                workspace[offset + Int64(7)] = group
                begin = end
                end = cutlass.min(k_tiles, end + stripe)


class ReferenceGroupBuilder:
    def __init__(
        self, rows: int, intermediate_size: int, grid: int, include_fc1: bool = True
    ):
        if not 2 <= rows <= 8 or not 1 <= grid <= 255:
            raise ValueError("unsupported reference grouping geometry")
        self.rows = int(rows)
        self.width = int(intermediate_size)
        self.grid = int(grid)
        self.include_fc1 = bool(include_fc1)
        self.n1 = self.width * 2 // 128
        self.layout = workspace_layout(self.width)

    @cute.jit
    def __call__(
        self,
        ids: cute.Tensor,
        expert_map: cute.Tensor,
        workspace: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(ids, expert_map, workspace).launch(
            grid=((self.n1 if self.include_fc1 else 0) + 28, 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, ids: cute.Tensor, expert_map: cute.Tensor, workspace: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        # This launch precedes every MoE CTA on the same stream. Its disjoint
        # header stores therefore initialize the resident-grid barrier safely.
        if block == Int32(0):
            index = Int32(tid)
            while index < Int32(HEADER_WORDS):
                workspace[Int64(index)] = Int32(0)
                index += Int32(128)
        if cutlass.const_expr(not self.include_fc1):
            block += Int32(self.n1)
        alloc = cutlass.utils.SmemAllocator()
        scratch = alloc.allocate_tensor(Int32, cute.make_layout((3 * MAX_ROUTES,)))
        emit_reference_group_tile(
            ids,
            expert_map,
            workspace,
            scratch,
            Int32(self.rows),
            Int32(self.grid),
            Int32(block),
            Int32(tid),
            self.width,
        )


@cache
def _get_builder(rows: int, width: int, grid: int, include_fc1: bool = True):
    kernel = ReferenceGroupBuilder(rows, width, grid, include_fc1)
    key = (rows, width, grid, bool(include_fc1))
    raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=key)

    def fake(size):
        return cute.runtime.make_fake_compact_tensor(Int32, (size,), assumed_align=16)

    return b12x_compile(
        kernel,
        fake(rows * 16),
        fake(896),
        fake(kernel.layout.words),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "moe.w4a16.reference_group_builder",
            ABI_VERSION,
            key,
            labels=("rows", "width", "resident_ctas", "include_fc1"),
        ),
    )


def build_reference_groups(
    ids, expert_map, workspace, *, rows, width, grid, stream, include_fc1=True
):
    layout = workspace_layout(width)
    if workspace.numel() < layout.words:
        raise ValueError(
            "reference grouping workspace is smaller than the planned layout"
        )
    _get_builder(int(rows), int(width), int(grid), bool(include_fc1))(
        ids.view(-1),
        expert_map.view(-1),
        workspace.view(-1),
        cuda.CUstream(stream),
    )
