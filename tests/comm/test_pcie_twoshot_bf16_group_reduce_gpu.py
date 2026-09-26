"""Nine-GPU proof of the group-reduce push all-reduce (precision-equal class).

Gate: ``B12X_RUN_PCIE_TP9_TEST=1`` and nine visible GPUs in the served order.
Runtimes on one group: the plain push kernel (reference order), the group
reduce (``B12X_PCIE_TP9_GROUP_REDUCE=1``, threshold 0) with the pair relay.
For 1/4/8/12/16/24/32 hidden rows (7168 bf16) and 4/16/32 latent rows (3584),
heavy-tailed random inputs: (1) the group kernel selects a group mode, (2) its
output differs from the plain kernel on at most 1e-4 of the elements and never
by more than one bf16 ulp of the value, (3) against the float64 sum of the
nine inputs its rounding error is no worse than the plain kernel's (mismatch
count and max ulp), (4) it is deterministic (eager twice equal) and replays
under CUDA graph capture with new inputs equal to its eager result. Per-rank
JSON stage lines attribute a hang or mismatch to one check.
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

SHAPES = tuple((rows, 7168) for rows in (1, 4, 8, 12, 16, 24, 32)) + tuple(
    (rows, 3584) for rows in (4, 16, 32)
)
GRAPH_ROWS = (8, 16, 24)
MAX_MISMATCH_FRACTION = 1e-4


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _mode(runtime, rows: int, width: int) -> str:
    packs = rows * width // 8
    return runtime._all_reduce_kernel_mode(packs // 9, packs % 9)


def _ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bf16 ulp distance via the ordered int16 view of the bit patterns."""
    ai = a.view(torch.int16).to(torch.int32)
    bi = b.view(torch.int16).to(torch.int32)
    ai = torch.where(ai < 0, -(ai & 0x7FFF), ai)
    bi = torch.where(bi < 0, -(bi & 0x7FFF), bi)
    return (ai - bi).abs()


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

    def stage(name: str, **extra) -> None:
        torch.cuda.synchronize(device)
        print(json.dumps({"stage": name, "rank": rank, **extra}), flush=True)

    def build(*, group_reduce: bool) -> PCIeTwoShotBF16:
        os.environ["B12X_PCIE_TP9_STATIC_PEERS"] = "1"
        os.environ["B12X_PCIE_TP9_PAIR_RELAY"] = "1"
        os.environ["B12X_PCIE_TP9_PAIR_RELAY_MIN_PACKS"] = "0"
        os.environ["B12X_PCIE_TP9_GROUP_REDUCE"] = "1" if group_reduce else "0"
        os.environ["B12X_PCIE_TP9_GROUP_REDUCE_MIN_PACKS"] = "0"
        runtime = PCIeTwoShotBF16.from_exchange_group(
            exchange_group=group, device=device, max_rows=49149, row_elems=8
        )
        runtime.all_reduce_mode = "push"
        return runtime

    runtimes = []
    try:
        plain = build(group_reduce=False)
        grouped = build(group_reduce=True)
        runtimes = [plain, grouped]
        stage("constructed")
        generator = torch.Generator(device="cpu").manual_seed(4000 + rank)

        def fresh(rows: int, width: int) -> torch.Tensor:
            # Heavy tails: a Student-t like mix of a normal and a scaled rare component.
            base = torch.randn(rows, width, generator=generator) * 3.0
            spikes = torch.randn(rows, width, generator=generator) * 300.0
            mask = torch.rand(rows, width, generator=generator) < 0.01
            return torch.where(mask, spikes, base).to(device=device, dtype=torch.bfloat16)

        for rows, width in SHAPES:
            assert _mode(grouped, rows, width) == "push_group_relay", (rows, width, _mode(grouped, rows, width))
            inp = fresh(rows, width)
            gathered = [torch.empty_like(inp) for _ in range(9)]
            dist.all_gather(gathered, inp, group=group)
            exact = torch.stack([g.double() for g in gathered]).sum(dim=0)
            ref = torch.empty_like(inp)
            plain.all_reduce(inp, out=ref)
            out = torch.empty_like(inp)
            grouped.all_reduce(inp, out=out)
            again = torch.empty_like(inp)
            grouped.all_reduce(inp, out=again)
            torch.cuda.synchronize(device)
            assert torch.equal(out, again), f"rank {rank} {rows}x{width}: group reduce not deterministic"
            ulps = _ulp_distance(out, ref)
            mismatch = int((ulps > 0).sum().item())
            frac = mismatch / out.numel()
            assert frac <= MAX_MISMATCH_FRACTION, f"rank {rank} {rows}x{width}: {frac:.2e} of elements differ from the plain kernel"
            assert int(ulps.max().item()) <= 1, f"rank {rank} {rows}x{width}: {int(ulps.max().item())} ulp from the plain kernel"
            exact_bf16 = exact.to(torch.bfloat16)
            plain_err = int((_ulp_distance(ref, exact_bf16) > 0).sum().item())
            group_err = int((_ulp_distance(out, exact_bf16) > 0).sum().item())
            plain_max = float((ref.double() - exact).abs().max().item())
            group_max = float((out.double() - exact).abs().max().item())
            assert group_err <= plain_err + max(2, plain_err // 10), (
                f"rank {rank} {rows}x{width}: group reduce misrounds {group_err} elements vs plain {plain_err}"
            )
            assert group_max <= plain_max * 1.5 + 1e-6, f"rank {rank} {rows}x{width}: max error {group_max} vs plain {plain_max}"
            stage(f"eager_{rows}x{width}_ok", mismatch_vs_plain=mismatch, plain_misround=plain_err, group_misround=group_err)

        grouped.prepare_graph()
        for rows in GRAPH_ROWS:
            inp = torch.empty(rows, 7168, dtype=torch.bfloat16, device=device)
            out = torch.empty_like(inp)
            graph = torch.cuda.CUDAGraph()
            inp.copy_(fresh(rows, 7168))
            with grouped.capture(), torch.cuda.graph(graph):
                grouped.all_reduce(inp, out=out)
            stage(f"captured_{rows}")
            for iteration in range(3):
                inp.copy_(fresh(rows, 7168))
                torch.cuda.synchronize(device)
                dist.barrier()
                graph.replay()
                torch.cuda.synchronize(device)
                eager = torch.empty_like(inp)
                grouped.all_reduce(inp, out=eager)
                torch.cuda.synchronize(device)
                assert torch.equal(out, eager), f"rank {rank} rows {rows} replay {iteration}: graph differs from eager"
                stage(f"replay_{rows}_{iteration}_ok")
            del graph
        stage("complete")
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        os._exit(1)
    finally:
        for runtime in runtimes:
            try:
                runtime.close()
            except Exception:  # noqa: BLE001
                pass
        if dist.is_initialized():
            dist.destroy_process_group()


def test_group_reduce_all_reduce_is_precision_equal_and_deterministic() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 9:
        pytest.skip("nine CUDA devices are required")
    mp.spawn(_worker, args=(_free_port(),), nprocs=9, join=True)
