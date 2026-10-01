"""Nine-GPU proof of the fused RMSNorm-shard BF16 all-reduce.

Gate: ``B12X_RUN_PCIE_TP9_TEST=1`` and nine visible GPUs. For rows of
1/4/8/16 tokens of the Kimi-K3 latent width (3,584) the fused launch must
(1) reduce bit-identically to ``all_reduce``, (2) write this rank's packed
column block (400 columns, the last rank's 16 padding columns zero) whose
values are closer to the float64 RMSNorm than the served CUDA RMSNorm's
(``vllm._C.rms_norm`` when importable, else its torch fp32 emulation), and
(3) replay under CUDA graph capture with new inputs. Per-rank JSON stage
lines attribute a hang or mismatch to one check.
"""

from __future__ import annotations

import datetime
import json
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_TP9_TEST") != "1",
    reason="set B12X_RUN_PCIE_TP9_TEST=1 with nine GPUs to run",
)

HIDDEN = 3584
SHARD = 400
EPS = 1e-6
ROWS = (1, 4, 8, 16)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _served_rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The served kernel when available (bitwise), else its fp32 emulation."""
    try:
        import vllm._custom_ops as ops  # noqa: F401

        out = torch.empty_like(x)
        torch.ops._C.rms_norm(out, x, weight, EPS)
        return out
    except Exception:  # noqa: BLE001
        xf = x.float()
        var = (xf * xf).sum(dim=-1, keepdim=True) / HIDDEN
        return ((xf * torch.rsqrt(var + EPS)) * weight.float()).to(torch.bfloat16)


def _reference_rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    xd = x.double()
    var = (xd * xd).sum(dim=-1, keepdim=True) / HIDDEN
    return (xd * torch.rsqrt(var + EPS)) * weight.double()


def _worker(rank: int, port: int) -> None:
    from b12x.comm.pcie.pcie_twoshot_bf16 import PCIeTwoShotBF16

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=9,
        timeout=datetime.timedelta(seconds=300),
        device_id=device,
    )
    group = dist.group.WORLD

    def stage(name: str) -> None:
        torch.cuda.synchronize(device)
        print(json.dumps({"stage": name, "rank": rank}), flush=True)

    try:
        twoshot = PCIeTwoShotBF16.from_exchange_group(
            exchange_group=group, device=device, max_rows=49149, row_elems=8
        )
        twoshot.all_reduce_mode = "push"
        twoshot.norm_shard_enabled = True
        twoshot.prepare_graph()
        stage("prepared")
        generator = torch.Generator(device="cpu").manual_seed(4242)
        weight = (1.0 + 0.1 * torch.randn(HIDDEN, generator=generator)).to(
            device=device, dtype=torch.bfloat16
        )
        col0 = rank * SHARD
        logical = min(SHARD, HIDDEN - col0)
        worse = 0
        for rows in ROWS:
            inp = (torch.randn(rows, HIDDEN, generator=generator) * (0.5 + rank * 0.1)).to(
                device=device, dtype=torch.bfloat16
            )
            reduced_ref = torch.empty_like(inp)
            twoshot.all_reduce(inp, out=reduced_ref)
            reduced, block = twoshot.all_reduce_rms_norm_shard(inp, weight, EPS, col0, SHARD)
            torch.cuda.synchronize(device)
            assert torch.equal(reduced, reduced_ref), f"rank {rank} rows {rows}: reduced rows differ"
            served = _served_rms_norm(reduced_ref, weight)[:, col0 : col0 + logical]
            exact = _reference_rms_norm(reduced_ref, weight)[:, col0 : col0 + logical]
            fused = block[:, :logical]
            assert torch.all(block[:, logical:] == 0), f"rank {rank}: padding columns not zero"
            err_fused = (fused.double() - exact).abs()
            err_served = (served.double() - exact).abs()
            max_bf16_gap = (fused.float() - served.float()).abs().max().item()
            print(json.dumps({
                "rank": rank, "rows": rows,
                "fused_max": err_fused.max().item(), "fused_mean": err_fused.mean().item(),
                "served_max": err_served.max().item(), "served_mean": err_served.mean().item(),
                "max_abs_gap_fused_vs_served": max_bf16_gap,
            }), flush=True)
            if err_fused.mean().item() > err_served.mean().item():
                worse += 1
            stage(f"rows{rows}_checked")
        assert worse == 0, f"rank {rank}: the fused norm was less precise than the served kernel in {worse} cases"

        # Graph capture and replay with new inputs (device slot selection).
        rows = 4
        inp = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device)
        out = torch.empty_like(inp)
        block = torch.empty(rows, SHARD, dtype=torch.bfloat16, device=device)
        graph = torch.cuda.CUDAGraph()
        with twoshot.capture(), torch.cuda.graph(graph):
            twoshot.all_reduce_rms_norm_shard(inp, weight, EPS, col0, SHARD, out=out, shard_out=block)
        stage("captured")
        for iteration in range(3):
            fresh = (torch.randn(rows, HIDDEN, generator=generator) * (1 + iteration)).to(
                device=device, dtype=torch.bfloat16
            )
            inp.copy_(fresh)
            torch.cuda.synchronize(device)
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize(device)
            reduced_ref = torch.empty_like(inp)
            twoshot.all_reduce(inp, out=reduced_ref)
            exact = _reference_rms_norm(reduced_ref, weight)[:, col0 : col0 + logical]
            torch.cuda.synchronize(device)
            assert torch.equal(out, reduced_ref), f"replay {iteration}: reduced rows differ"
            gap = (block[:, :logical].double() - exact).abs().max().item()
            assert gap < 2e-2, f"replay {iteration}: block far from the float64 norm ({gap})"
            stage(f"replay{iteration}_ok")
        del graph
        twoshot.close()
        stage("complete")
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        os._exit(1)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_norm_shard_all_reduce_matches_all_reduce_and_beats_the_served_norm() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 9:
        pytest.skip("nine CUDA devices are required")
    mp.spawn(_worker, args=(_free_port(),), nprocs=9, join=True)
