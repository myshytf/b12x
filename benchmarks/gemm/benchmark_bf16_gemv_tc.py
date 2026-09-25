"""Decode-size BF16 GEMV benchmark: cuBLAS vs SIMT small-N vs tensor-core.

Times the three backends on Kimi-K3 TP9 decode projection shapes with CUDA
graph replay (steady-state, launch overhead excluded from the loop body) and
an L2 flush between variants so the weight reads are cold. Reports us/call
and the DRAM weight-read floor for reference.

Run inside the serving image with the candidate source on PYTHONPATH:
    B12X_BF16_GEMV_TC=1 /opt/venv/bin/python \
        benchmarks/gemm/benchmark_bf16_gemv_tc.py
"""

from __future__ import annotations

import os

import torch

SHAPES = [
    # (n, k): Kimi-K3 TP9 decode projections
    (104, 7168),   # router gate per rank (bf16 out variant; router itself is fp32-out)
    (400, 7168),   # routed_expert_down_proj per rank
    (796, 3584),   # latent up_proj shard (tier-2 tail shape)
    (7168, 3584),  # tier-1 replicated latent up_proj
]
ROWS = (1, 4, 8)
CALLS_PER_GRAPH = 50
REPLAYS = 200
L2_FLUSH_MB = 256


def _time_graph(fn, calls, flush_buf):
    """Graph-capture ``calls`` invocations of fn, time replay with L2 flushed."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    best_ms = float("inf")
    for _ in range(REPLAYS):
        flush_buf.normal_()  # flush L2 so weight reads hit DRAM
        torch.cuda.synchronize()
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        best_ms = min(best_ms, start.elapsed_time(end))
    return best_ms * 1000.0 / calls  # us per call


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    flush_buf = torch.empty(L2_FLUSH_MB * 1024 * 1024 // 4, dtype=torch.float32, device=device)

    from b12x.gemm.bf16_gemv import _kernel

    os.environ.setdefault("B12X_BF16_GEMV_TC", "1")
    _kernel._tc_gemv_enabled.cache_clear()

    header = f"{'shape':>18} {'m':>3} {'cuBLAS':>9} {'SIMT':>9} {'tc':>9} {'floor':>7}"
    print(header)
    print("-" * len(header))
    for n, k in SHAPES:
        floor_us = n * k * 2 / 1.8e12 * 1e6  # bf16 weight bytes at 1.8 TB/s
        w = torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.05
        _kernel.precompile_bf16_gemv_small_n(w)
        for m in ROWS:
            x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
            y = torch.empty(m, n, device=device, dtype=torch.bfloat16)

            t_cublas = _time_graph(lambda: torch.mm(x, w.t(), out=y), CALLS_PER_GRAPH, flush_buf)

            simt = _kernel.compile_bf16_gemv_small_n(m, n, k)
            t_simt = _time_graph(lambda: simt(x, w, y), CALLS_PER_GRAPH, flush_buf)

            tc = _kernel.compile_bf16_gemv_tc(n, k)
            t_tc = _time_graph(lambda: tc(x, w, y), CALLS_PER_GRAPH, flush_buf)

            # spot-check numerics agree within bf16 rounding class
            ref = (x.double() @ w.double().T).float()
            for name, launch in (("simt", simt), ("tc", tc)):
                launch(x, w, y)
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    y.float(), ref, rtol=1e-2, atol=1e-2
                ), f"{name} mismatch at m={m} n={n} k={k}"

            print(
                f"{n:>6}x{k:<6} {m:>3} {t_cublas:>8.2f}u {t_simt:>8.2f}u "
                f"{t_tc:>8.2f}u {floor_us:>6.2f}u"
            )


if __name__ == "__main__":
    main()
