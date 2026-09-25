"""Tests for the opt-in tensor-core bf16 GEMV (``B12X_BF16_GEMV_TC=1``).

The tc backend (``_tensor_core.py``) streams the weight once for up to eight
live rows: FP32 accumulation, a fixed-order cross-warp reduction and one
BF16 rounding, so the output is deterministic. It is not bit-identical to
cuBLAS or to the SIMT small-N kernel (different reduction order); the
contract is "matches the f64 reference within BF16 rounding", the same
precision class as the rest of this file's family.
"""
from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@pytest.fixture()
def tc_on(monkeypatch):
    monkeypatch.setenv("B12X_BF16_GEMV_TC", "1")
    from b12x.gemm.bf16_gemv import _kernel

    _kernel._tc_gemv_enabled.cache_clear()
    yield _kernel
    _kernel._tc_gemv_enabled.cache_clear()


def _op():
    from b12x.gemm import bf16_gemv

    bf16_gemv.bf16_gemv_small_n  # noqa: B018
    return torch.ops.b12x.bf16_gemv_small_n


def _assert_matches_f64_ref(y, x, w):
    expected = (x.double() @ w.double().T).float()
    assert y.dtype == torch.bfloat16
    assert y.shape == (x.shape[0], w.shape[0])
    torch.testing.assert_close(y.float(), expected, rtol=1e-2, atol=1e-2)


@cuda_required
@pytest.mark.parametrize("m", [1, 2, 3, 8])
@pytest.mark.parametrize(
    "n,k",
    [
        (4096, 2048),
        (2560, 4096),
        (1040, 512),
        # Kimi-K3 decode projections per TP9 rank (router / down / up tail):
        (104, 7168),
        (400, 7168),
        (796, 3584),
    ],
)
def test_tc_gemv_matches_f64_reference(tc_on, m, n, k):
    torch.manual_seed(m * 131 + n + k)
    op = _op()
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    y = op(x, w)
    _assert_matches_f64_ref(y, x, w)
    again = op(x, w)
    assert torch.equal(y, again), "tensor-core GEMV must be deterministic"


@cuda_required
def test_tc_gemv_live_rows_share_one_compile(tc_on):
    """One (n, k) compile serves m=1..8 dynamically; rows past the live count
    must be left untouched (CUDA-graph capture at m=8 replays with fewer)."""
    torch.manual_seed(7)
    k, n = 3584, 796
    from b12x.gemm.bf16_gemv import _kernel

    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    x8 = torch.randn(8, k, device="cuda", dtype=torch.bfloat16)
    out = torch.full((8, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    launch = _kernel.compile_bf16_gemv_tc(n, k)
    for rows in (8, 1, 3):
        out.fill_(float("nan"))
        launch(x8[:rows], w, out[:rows])
        torch.cuda.synchronize()
        _assert_matches_f64_ref(out[:rows], x8[:rows], w)
        assert torch.isnan(out[rows:]).all(), "rows past live count must be untouched"
    assert _kernel.get_cached_bf16_gemv_tc(n, k) is launch


@cuda_required
def test_tc_gemv_default_off_uses_simt(monkeypatch):
    """Without the opt-in the op keeps the SIMT serving path (no tc compile)."""
    monkeypatch.delenv("B12X_BF16_GEMV_TC", raising=False)
    from b12x.gemm.bf16_gemv import _kernel

    _kernel._tc_gemv_enabled.cache_clear()
    _kernel._KERNEL_CACHE.clear()
    torch.manual_seed(3)
    op = _op()
    x = torch.randn(4, 7168, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(104, 7168, device="cuda", dtype=torch.bfloat16)
    y = op(x, w)
    _assert_matches_f64_ref(y, x, w)
    assert _kernel.get_cached_bf16_gemv_tc(104, 7168) is None
    assert _kernel.get_cached_bf16_gemv_small_n(4, 104, 7168) is not None
    _kernel._KERNEL_CACHE.clear()


@cuda_required
def test_tc_gemv_k_gate(tc_on):
    """K not a multiple of 512 stays off the tc path (SIMT or cuBLAS)."""
    torch.manual_seed(11)
    from b12x.gemm.bf16_gemv import _kernel

    op = _op()
    x = torch.randn(2, 384, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(96, 384, device="cuda", dtype=torch.bfloat16)
    y = op(x, w)
    _assert_matches_f64_ref(y, x, w)
    assert _kernel.get_cached_bf16_gemv_tc(96, 384) is None
