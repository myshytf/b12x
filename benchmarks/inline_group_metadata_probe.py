"""Validation fixture for metadata construction concurrent with a direct LUT copy."""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.intrinsics import (
    cp_async_bulk_g2s_mbar,
    get_ptr_as_int64,
    shared_ptr_to_u32,
)
from b12x._lib.utils import current_cuda_stream
from b12x.moe._shared.kernels.w4a16.reference_grouped import (
    ABI_VERSION,
    MAX_ROUTES,
    emit_reference_group_tile,
    workspace_layout,
)

LUT_OFFSET_BYTES = 35824
LUT_WORDS = 16384


class InlineMetadataProbe:
    def __init__(self, rows, width, first_cta):
        self.rows, self.width, self.first_cta = rows, width, first_cta

    @cute.jit
    def __call__(self, ids, mapping, workspace, lut, copied_lut, stream: cuda.CUstream):
        self.kernel(ids, mapping, workspace, lut, copied_lut).launch(
            grid=(186, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, ids, mapping, workspace, lut, copied_lut):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        alloc = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, LUT_OFFSET_BYTES // 4 + LUT_WORDS],
                1024,
            ]
            barrier: cute.struct.Align[cute.struct.MemRange[cutlass.Uint64, 1], 16]

        storage = alloc.allocate(Storage)
        shared_base = shared_ptr_to_u32(storage.words.data_ptr())
        barrier = storage.barrier.data_ptr()
        if tid == 0:
            cute.arch.mbarrier_init(barrier, Int32(1))
        cute.arch.barrier()
        if tid == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, Int32(LUT_WORDS * 4))
            cp_async_bulk_g2s_mbar(
                shared_base + Int32(LUT_OFFSET_BYTES),
                get_ptr_as_int64(lut, Int32(0)),
                Int32(LUT_WORDS * 4),
                shared_ptr_to_u32(barrier),
            )
        if block >= self.first_cta and block < self.first_cta + 28:
            scratch = cute.make_tensor(
                cute.recast_ptr(storage.words.data_ptr(), dtype=Int32),
                cute.make_layout((3 * MAX_ROUTES,), stride=(1,)),
            )
            logical = Int32(block - self.first_cta + self.width * 2 // 128)
            emit_reference_group_tile(
                ids,
                mapping,
                workspace,
                scratch,
                Int32(self.rows),
                Int32(186),
                logical,
                Int32(tid),
                self.width,
            )
        cute.arch.mbarrier_wait(barrier, phase=0)
        if block >= self.first_cta and block < self.first_cta + 28:
            table = cute.make_tensor(
                storage.words.data_ptr() + Int32(LUT_OFFSET_BYTES // 4),
                cute.make_layout((LUT_WORDS,), stride=(1,)),
            )
            index = Int32(tid)
            while index < Int32(LUT_WORDS):
                offset = Int64(block - self.first_cta) * Int64(LUT_WORDS) + Int64(index)
                copied_lut[offset] = table[index].to(Int32)
                index += Int32(256)


@cache
def get_inline_probe(rows, width, first_cta):
    probe = InlineMetadataProbe(rows, width, first_cta)

    def fake(size):
        return cute.runtime.make_fake_compact_tensor(Int32, (size,), assumed_align=16)

    return b12x_compile(
        probe,
        fake(rows * 16),
        fake(896),
        fake(workspace_layout(width).words),
        fake(LUT_WORDS),
        fake(28 * LUT_WORDS),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "validation.moe.inline_reference_groups",
            ABI_VERSION,
            (rows, width, first_cta),
        ),
    )
