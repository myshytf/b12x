"""Nine-GPU proof of the pair-relay publish phase of the push all-reduce.

Gate: ``B12X_RUN_PCIE_TP9_TEST=1`` and nine visible GPUs in the served order
(ranks (0,1) (2,3) (4,5) (6,7) PIX pairs, rank 8 single; the runtime verifies
the placement from sysfs). Three runtimes on the same group: the plain push
kernel, push with the pair relay (``B12X_PCIE_TP9_PAIR_RELAY=1``, threshold 0)
and static-peer push with the relay. For 1/4/8/12/16/24/32 hidden rows (7168
bf16) and 4/16/32 latent rows (3584) every relayed all-reduce must (1) select
a relay kernel, (2) equal the plain kernel bit for bit, eager, and stay within
one bf16 step of the float64 sum, (3) replay under CUDA graph capture with new
inputs, bit-identical to the plain kernel on the same inputs. Per-rank JSON
stage lines attribute a hang or mismatch to one check.
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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _relay_mode(runtime, rows: int, width: int) -> str:
    packs = rows * width // 8
    return runtime._all_reduce_kernel_mode(packs // 9, packs % 9)


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

    def build(*, relay: bool, static: bool) -> PCIeTwoShotBF16:
        os.environ["B12X_PCIE_TP9_PAIR_RELAY"] = "1" if relay else "0"
        os.environ["B12X_PCIE_TP9_PAIR_RELAY_MIN_PACKS"] = "0"
        os.environ["B12X_PCIE_TP9_STATIC_PEERS"] = "1" if static else "0"
        runtime = PCIeTwoShotBF16.from_exchange_group(
            exchange_group=group, device=device, max_rows=49149, row_elems=8
        )
        runtime.all_reduce_mode = "push"
        return runtime

    runtimes = []
    try:
        plain = build(relay=False, static=False)
        relay = build(relay=True, static=False)
        static_relay = build(relay=True, static=True)
        runtimes = [plain, relay, static_relay]
        assert relay._pair_relay_enabled and static_relay._pair_relay_enabled
        assert not plain._pair_relay_enabled
        stage("constructed")
        generator = torch.Generator(device="cpu").manual_seed(1000 + rank)

        def fresh(rows: int, width: int, scale: float = 4.0) -> torch.Tensor:
            return (torch.randn(rows, width, generator=generator) * scale).to(
                device=device, dtype=torch.bfloat16
            )

        for rows, width in SHAPES:
            assert _relay_mode(relay, rows, width) == "push_relay", (rows, width)
            expected_static = "push_static_relay" if rows * width // 8 <= 7168 else "push_relay"
            assert _relay_mode(static_relay, rows, width) == expected_static, (rows, width)
            inp = fresh(rows, width)
            reference = torch.empty_like(inp)
            plain.all_reduce(inp, out=reference)
            gathered = [torch.empty_like(inp) for _ in range(9)]
            dist.all_gather(gathered, inp, group=group)
            exact = torch.stack([g.double() for g in gathered]).sum(dim=0)
            for label, runtime in (("relay", relay), ("static_relay", static_relay)):
                out = torch.empty_like(inp)
                runtime.all_reduce(inp, out=out)
                torch.cuda.synchronize(device)
                assert torch.equal(out, reference), f"rank {rank} {label} {rows}x{width}: differs from the plain push kernel"
                gap = (out.double() - exact).abs().max().item()
                step = exact.abs().max().item() * 2 ** -7 + 1e-3
                assert gap <= step, f"rank {rank} {label} {rows}x{width}: {gap} from the float64 sum"
            stage(f"eager_{rows}x{width}_ok")

        for runtime, label in ((relay, "relay"), (static_relay, "static_relay")):
            runtime.prepare_graph()
            for rows in GRAPH_ROWS:
                inp = torch.empty(rows, 7168, dtype=torch.bfloat16, device=device)
                out = torch.empty_like(inp)
                graph = torch.cuda.CUDAGraph()
                inp.copy_(fresh(rows, 7168))
                with runtime.capture(), torch.cuda.graph(graph):
                    runtime.all_reduce(inp, out=out)
                stage(f"{label}_captured_{rows}")
                for iteration in range(3):
                    inp.copy_(fresh(rows, 7168, scale=1.0 + iteration))
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    graph.replay()
                    torch.cuda.synchronize(device)
                    reference = torch.empty_like(inp)
                    plain.all_reduce(inp, out=reference)
                    torch.cuda.synchronize(device)
                    assert torch.equal(out, reference), f"rank {rank} {label} rows {rows} replay {iteration}: differs"
                    stage(f"{label}_replay_{rows}_{iteration}_ok")
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


def test_pair_relay_all_reduce_matches_the_plain_push_kernel() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 9:
        pytest.skip("nine CUDA devices are required")
    mp.spawn(_worker, args=(_free_port(),), nprocs=9, join=True)
