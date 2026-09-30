"""Bit-equality of the direct state-table decode against the T12 decode.

The rate-indexed direct table precomposes the frozen XOR-Cheb rank map with
the modal T12 staircase (``direct[state] = t12[rank(state) >> 4]``), so the
two decode intrinsics must produce identical bytes for every window. The
probe drives both intrinsics on the same random ring windows and compares
the packed results.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass import Int32, Uint32
from cutlass.cute.runtime import from_dlpack

from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import (
    packed_decode_sqg_xor_cheb_t12_to_e4m3x8,
    packed_decode_trellis_sqg_direct_lut_smem_to_e4m3x8,
    packed_decode_trellis_sqg_direct_lut_to_e4m3x8,
    shared_ptr_to_u32,
)
from b12x._lib.quant.sqg_e4m3 import (
    sqg_xor_cheb_t12_direct_lut_cpu,
    sqg_xor_cheb_t12_lut_cpu,
)
from b12x._lib.utils import current_cuda_stream
from tests._reference.helpers import require_b12x

require_b12x()


class _DecodeProbe:
    def __init__(self, bits: int):
        self.bits = int(bits)

    @cute.jit
    def __call__(
        self,
        wins: cute.Tensor,
        t12: cute.Tensor,
        direct: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(wins, t12, direct, out).launch(
            grid=(cute.size(wins) // 64, 1, 1), block=[32, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        wins: cute.Tensor,
        t12: cute.Tensor,
        direct: cute.Tensor,
        out: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = Int32(tidx) + Int32(bidx) * 32
        t12_addr = t12.iterator.toint()
        dir_addr = direct.iterator.toint()
        wa = Uint32(wins[2 * lane])
        wb = Uint32(wins[2 * lane + 1])
        lo_t, hi_t = packed_decode_sqg_xor_cheb_t12_to_e4m3x8(
            wa, wb, t12_addr, self.bits, t12_in_shared=False
        )
        lo_d, hi_d = packed_decode_trellis_sqg_direct_lut_to_e4m3x8(
            wa, wb, dir_addr, self.bits, rate_indexed=True
        )
        out[4 * lane] = Int32(lo_t)
        out[4 * lane + 1] = Int32(hi_t)
        out[4 * lane + 2] = Int32(lo_d)
        out[4 * lane + 3] = Int32(hi_d)


class _SmemDecodeProbe:
    """The production shared-memory direct-table variant against the T12 decode.

    Stages the rate slice for the probe's bitrate into shared memory once,
    then decodes with the same intrinsic the fused W4A16 kernel uses.
    """

    def __init__(self, bits: int):
        self.bits = int(bits)

    def _shared_storage_cls(self):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "table": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, (1 << 14)], 16
            ],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        wins: cute.Tensor,
        t12: cute.Tensor,
        direct: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(wins, t12, direct, out).launch(
            grid=(cute.size(wins) // 64, 1, 1), block=[32, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        wins: cute.Tensor,
        t12: cute.Tensor,
        direct: cute.Tensor,
        out: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = Int32(tidx) + Int32(bidx) * 32
        storage = cutlass.utils.SmemAllocator().allocate(self._shared_storage_cls())
        table = storage.table.get_tensor(cute.make_layout((1 << 14,)))
        # Stage this bitrate's 64 KiB rate slice (u32 units) from the global
        # rate-indexed direct table.
        slice_base = Int32((self.bits - 2) << 14)
        i = Int32(tidx)
        while i < Int32(1 << 14):
            table[i] = direct[slice_base + i]
            i += Int32(32)
        cute.arch.sync_threads()
        smem_addr = shared_ptr_to_u32(table.iterator)

        t12_addr = t12.iterator.toint()
        wa = Uint32(wins[2 * lane])
        wb = Uint32(wins[2 * lane + 1])
        lo_t, hi_t = packed_decode_sqg_xor_cheb_t12_to_e4m3x8(
            wa, wb, t12_addr, self.bits, t12_in_shared=False
        )
        lo_d, hi_d = packed_decode_trellis_sqg_direct_lut_smem_to_e4m3x8(
            wa, wb, smem_addr, self.bits
        )
        out[4 * lane] = Int32(lo_t)
        out[4 * lane + 1] = Int32(hi_t)
        out[4 * lane + 2] = Int32(lo_d)
        out[4 * lane + 3] = Int32(hi_d)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_direct_lut_decode_bit_equals_t12(bits: int) -> None:
    device = torch.device("cuda")
    torch.manual_seed(20260812 + bits)
    wins = torch.zeros(64, dtype=torch.int32, device=device)
    t12 = sqg_xor_cheb_t12_lut_cpu().to(device)
    direct = sqg_xor_cheb_t12_direct_lut_cpu().to(device)
    out = torch.zeros(128, dtype=torch.int32, device=device)

    def args():
        return (
            from_dlpack(wins, assumed_align=16),
            from_dlpack(t12, assumed_align=16),
            from_dlpack(direct, assumed_align=16),
            from_dlpack(out, assumed_align=16),
            current_cuda_stream(),
        )

    compiled = b12x_compile(_DecodeProbe(bits), *args())
    mismatched = 0
    for _ in range(512):
        wins.copy_(
            torch.randint(
                -(2**31), 2**31 - 1, (64,), dtype=torch.int32, device=device
            )
        )
        compiled(*args())
        torch.cuda.synchronize()
        o = out.view(32, 4)
        mismatched += int((o[:, 0] != o[:, 2]).sum())
        mismatched += int((o[:, 1] != o[:, 3]).sum())
    assert mismatched == 0, mismatched


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_direct_lut_smem_decode_bit_equals_t12(bits: int) -> None:
    device = torch.device("cuda")
    torch.manual_seed(20260925 + bits)
    wins = torch.zeros(64, dtype=torch.int32, device=device)
    t12 = sqg_xor_cheb_t12_lut_cpu().to(device)
    direct_bytes = sqg_xor_cheb_t12_direct_lut_cpu()
    # The staging loop copies u32 units: view the byte table as little-endian
    # u32 so byte order survives the vectorized copy (196608 = 3 * 65536).
    direct = direct_bytes.view(torch.int32).to(device)
    out = torch.zeros(128, dtype=torch.int32, device=device)

    def args():
        return (
            from_dlpack(wins, assumed_align=16),
            from_dlpack(t12, assumed_align=16),
            from_dlpack(direct, assumed_align=16),
            from_dlpack(out, assumed_align=16),
            current_cuda_stream(),
        )

    compiled = b12x_compile(_SmemDecodeProbe(bits), *args())
    mismatched = 0
    for _ in range(512):
        wins.copy_(
            torch.randint(
                -(2**31), 2**31 - 1, (64,), dtype=torch.int32, device=device
            )
        )
        compiled(*args())
        torch.cuda.synchronize()
        o = out.view(32, 4)
        mismatched += int((o[:, 0] != o[:, 2]).sum())
        mismatched += int((o[:, 1] != o[:, 3]).sum())
    assert mismatched == 0, mismatched


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("shared", [False, True])
def test_direct_lut_all_codewords_and_graph_replay(bits: int, shared: bool) -> None:
    """Exhaust the codeword domain, then replay with changed crossing windows."""
    torch.manual_seed(731 + bits)
    device = torch.device("cuda")
    states = torch.arange(65536, device=device, dtype=torch.int32)
    upper = torch.randint(0, 65536, (65536, 2), device=device, dtype=torch.int32)
    wins = ((upper << 16) | states[:, None]).flatten()
    t12 = sqg_xor_cheb_t12_lut_cpu().to(device)
    direct = sqg_xor_cheb_t12_direct_lut_cpu().to(device)
    if shared:
        direct = direct.view(torch.int32)
    out = torch.empty(65536 * 4, device=device, dtype=torch.int32)
    args = tuple(
        from_dlpack(tensor, assumed_align=16) for tensor in (wins, t12, direct, out)
    ) + (current_cuda_stream(),)
    probe = _SmemDecodeProbe(bits) if shared else _DecodeProbe(bits)
    compiled = b12x_compile(probe, *args)
    compiled(*args)
    torch.cuda.synchronize()

    def check():
        result = out.view(-1, 4)
        assert torch.count_nonzero(result[:, :2]) > 0
        torch.testing.assert_close(result[:, :2], result[:, 2:], rtol=0, atol=0)

    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(*args)
    pointers = [tensor.data_ptr() for tensor in (wins, t12, direct, out)]
    for _ in range(3):
        upper.random_(0, 65536)
        wins.copy_(((upper << 16) | states[:, None]).flatten())
        out.fill_(0x5A5A5A5A)
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        assert pointers == [tensor.data_ptr() for tensor in (wins, t12, direct, out)]
        check()
