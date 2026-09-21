from __future__ import annotations

import ctypes
import datetime
import json
import os
import socket
import sys
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import b12x.comm.pcie.pcie_dcp_a2a as pcie_dcp_a2a
from b12x.comm.pcie.pcie_oneshot import PCIeOneshotAllReduce
from b12x.comm.pcie.pcie_dcp_a2a import (
    PCIeDCPA2A,
    PCIeDCPA2APool,
    _staging_layout,
    lse_reduce_scatter_reference,
)


pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_DCP_A2A_TEST") != "1",
    reason="set B12X_RUN_PCIE_DCP_A2A_TEST=1 to run PCIe DCP A2A GPU tests",
)

# 16 heads serve world sizes 2/4/8/16; a multiple of 9 (18, 99) serves 9.
TOTAL_HEADS = int(os.getenv("B12X_PCIE_DCP_A2A_TEST_TOTAL_HEADS", "16"))
HEAD_DIM = 512
QUERY_HEAD_DIM = 576
MAX_BATCH = 64
TEST_BATCHES = (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64)
# Paired projection gather rows: a bf16 row and an fp32 row whose bytes sum to
# the query head dimension the channel is laid out for (512 + 64 = 576 B).
PAIR_FIRST_WIDTH = 256
PAIR_SECOND_WIDTH = 16
PAIR_BATCHES = (1, 2, 4, 8, 16, 64)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _rank_inputs(
    step: int,
    source_rank: int,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(10000 * step + source_rank)
    output = torch.randn(
        batch,
        TOTAL_HEADS,
        HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    lse = torch.randn(
        batch,
        TOTAL_HEADS,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device)
    if batch > 0:
        lse[0, 0] = -torch.inf
        lse[0, 1] = torch.nan
        if source_rank == 0:
            lse[0, 2] = -torch.inf
    return output, lse


def _rank_query(
    step: int,
    source_rank: int,
    world_size: int,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(20000 * step + source_rank)
    return torch.randn(
        batch,
        TOTAL_HEADS // world_size,
        QUERY_HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)


def _reference(
    step: int,
    rank: int,
    world_size: int,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    inputs = [
        _rank_inputs(step, source_rank, batch, dtype, device)
        for source_rank in range(world_size)
    ]
    return lse_reduce_scatter_reference(
        torch.stack([item[0] for item in inputs]),
        torch.stack([item[1] for item in inputs]),
        rank,
    )


def _rank_pair(
    step: int,
    source_rank: int,
    batch: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(30000 * step + source_rank)
    first = torch.randn(
        batch, PAIR_FIRST_WIDTH, generator=generator, dtype=torch.float32
    ).to(device=device, dtype=torch.bfloat16)
    second = torch.randn(
        batch, PAIR_SECOND_WIDTH, generator=generator, dtype=torch.float32
    ).to(device=device)
    return first, second


def _expected_pair(
    step: int,
    world_size: int,
    batch: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [_rank_pair(step, source, batch, device) for source in range(world_size)]
    return (
        torch.cat([row[0] for row in rows], dim=1),
        torch.cat([row[1] for row in rows], dim=1),
    )


def _check_pair_eager(
    pool: PCIeDCPA2APool,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    """Paired projection gather (latent + router rows) against torch.cat."""
    for step, batch in enumerate(PAIR_BATCHES, start=300):
        first, second = _rank_pair(step, rank, batch, device)
        out_first, out_second = pool.all_gather_pair(
            first, second, channel_id="eager:dcp"
        )
        torch.cuda.synchronize(device)
        expected_first, expected_second = _expected_pair(step, world_size, batch, device)
        assert torch.equal(out_first, expected_first), f"pair first rows batch {batch}"
        assert torch.equal(out_second, expected_second), f"pair second rows batch {batch}"
        _stage(rank, f"pair_eager_batch{batch}_ok")
    # Caller-owned outputs clipped to a logical width (the last rank's tail
    # packs dropped): 8 bf16 columns and 4 fp32 columns fewer than the full
    # gathered rows, both 16-byte multiples.
    for step, batch in enumerate((1, 4, 8), start=400):
        first, second = _rank_pair(step, rank, batch, device)
        first_width = world_size * PAIR_FIRST_WIDTH - 8
        second_width = world_size * PAIR_SECOND_WIDTH - 4
        out_first = torch.full((batch, first_width), 7.0, dtype=torch.bfloat16, device=device)
        out_second = torch.full((batch, second_width), 7.0, dtype=torch.float32, device=device)
        pool.all_gather_pair(first, second, out_first, out_second, channel_id="eager:dcp")
        torch.cuda.synchronize(device)
        expected_first, expected_second = _expected_pair(step, world_size, batch, device)
        assert torch.equal(out_first, expected_first[:, :first_width]), f"clipped first rows batch {batch}"
        assert torch.equal(out_second, expected_second[:, :second_width]), f"clipped second rows batch {batch}"
        _stage(rank, f"pair_eager_clipped_batch{batch}_ok")


def _check_pair_graph(
    pool: PCIeDCPA2APool,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    """Paired projection gather captured once and replayed with new inputs.

    The capture alternates the two staging slots across layers; every replay
    rewrites the captured inputs and checks the gathered rows of every layer.
    """
    layers = 3
    batch = 4
    stream = torch.cuda.Stream(device=device)
    firsts = [
        torch.empty(batch, PAIR_FIRST_WIDTH, dtype=torch.bfloat16, device=device)
        for _ in range(layers)
    ]
    seconds = [
        torch.empty(batch, PAIR_SECOND_WIDTH, dtype=torch.float32, device=device)
        for _ in range(layers)
    ]
    out_firsts = [
        torch.empty(
            batch, world_size * PAIR_FIRST_WIDTH, dtype=torch.bfloat16, device=device
        )
        for _ in range(layers)
    ]
    out_seconds = [
        torch.empty(
            batch, world_size * PAIR_SECOND_WIDTH, dtype=torch.float32, device=device
        )
        for _ in range(layers)
    ]
    # The eager paired checks already prepared the graph (device-slot) variant
    # of the launcher (channels are stream-affine, so no prepare call on an
    # existing channel here); `capture` prepares the new logical channel
    # collectively on this stream.
    graph = torch.cuda.CUDAGraph()
    _stage(rank, "pair_graph_capture")
    with pool.capture(stream, channel_id="graph:pair") as graph_channel, torch.cuda.graph(
        graph, stream=stream
    ):
        for layer in range(layers):
            graph_channel.all_gather_pair(
                firsts[layer], seconds[layer], out_firsts[layer], out_seconds[layer]
            )
    stream.synchronize()
    _stage(rank, "pair_graph_captured")
    for replay in range(4):
        for layer in range(layers):
            step = 4000 + 10 * replay + layer
            first, second = _rank_pair(step, rank, batch, device)
            firsts[layer].copy_(first)
            seconds[layer].copy_(second)
        torch.cuda.synchronize(device)
        dist.barrier()
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        for layer in range(layers):
            step = 4000 + 10 * replay + layer
            expected_first, expected_second = _expected_pair(
                step, world_size, batch, device
            )
            assert torch.equal(out_firsts[layer], expected_first), (
                f"pair graph first rows replay {replay} layer {layer}"
            )
            assert torch.equal(out_seconds[layer], expected_second), (
                f"pair graph second rows replay {replay} layer {layer}"
            )
        _stage(rank, f"pair_graph_replay{replay}_ok")
    del graph
    torch.cuda.synchronize(device)


def _local_staging_words(channel, stream: torch.cuda.Stream) -> tuple[int, int]:
    """Sample one gather staging word per slot that a gather call rewrites.

    Pull transport: a rank stages its own rows compactly from offset 0 of
    its slot, so the first word is its own row 0. Push transport: a rank's
    slot holds only its peers' rows, at their output positions, so the
    sample is the first word of the next rank's row 0.
    """
    assert channel._ipc is not None
    assert len(channel._owned_buffers) == 1
    layout = _staging_layout(
        signal_bytes=pcie_dcp_a2a._SIGNAL_BYTES,
        world_size=channel.world_size,
        max_batch_size=channel.max_batch_size,
        total_heads=channel.total_heads,
        head_dim=channel.head_dim,
        query_head_dim=channel.query_head_dim,
    )
    row_offset = 0
    if channel.push_transport:
        peer = (channel.rank + 1) % channel.world_size
        row_offset = peer * channel.heads_per_rank * channel.query_head_dim * 2
    words = (ctypes.c_uint16(), ctypes.c_uint16())
    local_ptr = channel._owned_buffers[0].local_ptr
    for word, offset in zip(
        words,
        (layout.staging0_offset + row_offset, layout.staging1_offset + row_offset),
        strict=True,
    ):
        channel._ipc.cudaMemcpyAsync(
            ctypes.addressof(word),
            local_ptr + offset,
            ctypes.sizeof(word),
            int(stream.cuda_stream),
        )
    stream.synchronize()
    return words[0].value, words[1].value


def _check_eager(
    pool: PCIeDCPA2APool,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    for dtype in (torch.bfloat16, torch.float16):
        for step, batch in enumerate(TEST_BATCHES, start=1):
            local_q = _rank_query(
                step + 100,
                rank,
                world_size,
                batch,
                dtype,
                device,
            )
            wrong_device = (rank + 1) % world_size
            guard_gather = dtype == torch.bfloat16 and step == 1
            if guard_gather:
                torch.cuda.set_device(wrong_device)
            gathered_q = pool.all_gather_heads(
                local_q, channel_id="eager:dcp"
            )
            if guard_gather:
                assert torch.cuda.current_device() == wrong_device
                torch.cuda.set_device(rank)
            expected_q = torch.cat(
                [
                    _rank_query(
                        step + 100,
                        source,
                        world_size,
                        batch,
                        dtype,
                        device,
                    )
                    for source in range(world_size)
                ],
                dim=1,
            )
            torch.testing.assert_close(gathered_q, expected_q, rtol=0, atol=0)

            partial_output, partial_lse = _rank_inputs(step, rank, batch, dtype, device)
            guard_reduce = dtype == torch.bfloat16 and step == 2
            if guard_reduce:
                torch.cuda.set_device(wrong_device)
            out = pool.lse_reduce_scatter(
                partial_output, partial_lse, channel_id="eager:dcp"
            )
            if guard_reduce:
                assert torch.cuda.current_device() == wrong_device
                torch.cuda.set_device(rank)
            torch.cuda.synchronize(device)
            expected = _reference(
                step,
                rank,
                world_size,
                batch,
                dtype,
                device,
            )
            torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)

            input_storage = torch.empty(
                TOTAL_HEADS,
                MAX_BATCH,
                HEAD_DIM,
                dtype=dtype,
                device=device,
            )
            head_major_input = input_storage.transpose(0, 1)[:batch]
            head_major_input.copy_(partial_output)
            output_storage = torch.empty(
                TOTAL_HEADS // world_size,
                MAX_BATCH,
                HEAD_DIM,
                dtype=dtype,
                device=device,
            )
            head_major_output = output_storage.transpose(0, 1)[:batch]
            actual = pool.lse_reduce_scatter(
                head_major_input,
                partial_lse,
                out=head_major_output,
                channel_id="eager:dcp",
            )
            torch.cuda.synchronize(device)
            assert actual is head_major_output
            assert actual.movedim(0, 1).stride(0) == MAX_BATCH * HEAD_DIM
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    for step, batch in enumerate(TEST_BATCHES, start=100):
        local_q = _rank_query(
            step,
            rank,
            world_size,
            batch,
            torch.float8_e4m3fn,
            device,
        )
        gathered_q = pool.all_gather_heads(local_q, channel_id="eager:dcp")
        expected_q = torch.cat(
            [
                _rank_query(
                    step,
                    source,
                    world_size,
                    batch,
                    torch.float8_e4m3fn,
                    device,
                )
                for source in range(world_size)
            ],
            dim=1,
        )
        torch.testing.assert_close(
            gathered_q.view(torch.uint8),
            expected_q.view(torch.uint8),
            rtol=0,
            atol=0,
        )


def _check_eager_adjacency(
    pool: PCIeDCPA2APool,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    channel = pool.for_stream(channel_id="eager:dcp")
    first_query = _rank_query(700, rank, world_size, MAX_BATCH, torch.bfloat16, device)
    second_query = _rank_query(701, rank, world_size, MAX_BATCH, torch.bfloat16, device)
    partial_output, partial_lse = _rank_inputs(702, rank, 1, torch.bfloat16, device)
    first_gather = torch.empty(
        MAX_BATCH,
        TOTAL_HEADS,
        QUERY_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    second_gather = torch.empty_like(first_gather)
    reduced = torch.empty(
        1,
        TOTAL_HEADS // world_size,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )

    # Issue large-grid AG -> small-grid RS -> large-grid AG without a host
    # synchronization so adjacent eager executions exercise slot advancement.
    channel.all_gather_heads(first_query, first_gather)
    channel.lse_reduce_scatter(partial_output, partial_lse, reduced)
    channel.all_gather_heads(second_query, second_gather)
    torch.cuda.synchronize(device)

    expected_first = torch.cat(
        [
            _rank_query(700, source, world_size, MAX_BATCH, torch.bfloat16, device)
            for source in range(world_size)
        ],
        dim=1,
    )
    expected_second = torch.cat(
        [
            _rank_query(701, source, world_size, MAX_BATCH, torch.bfloat16, device)
            for source in range(world_size)
        ],
        dim=1,
    )
    torch.testing.assert_close(first_gather, expected_first, rtol=0, atol=0)
    torch.testing.assert_close(second_gather, expected_second, rtol=0, atol=0)
    torch.testing.assert_close(
        reduced,
        _reference(702, rank, world_size, 1, torch.bfloat16, device),
        rtol=2e-2,
        atol=2e-2,
    )


def _check_graph(
    pool: PCIeDCPA2APool,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    stream = torch.cuda.Stream(device=device)
    pool.prepare_channels(("graph",))
    channel = pool.for_stream(stream, channel_id="graph")
    layers = 7
    input_storages = [
        torch.empty(
            TOTAL_HEADS,
            MAX_BATCH,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(layers)
    ]
    inputs = [storage.transpose(0, 1) for storage in input_storages]
    lses = [
        torch.empty(MAX_BATCH, TOTAL_HEADS, dtype=torch.float32, device=device)
        for _ in range(layers)
    ]
    output_storages = [
        torch.empty(
            TOTAL_HEADS // world_size,
            MAX_BATCH,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(layers)
    ]
    outputs = [storage.transpose(0, 1) for storage in output_storages]
    assert all(tensor.stride(1) == MAX_BATCH * HEAD_DIM for tensor in inputs)
    assert all(tensor.stride(1) == MAX_BATCH * HEAD_DIM for tensor in outputs)
    local_queries = [
        torch.empty(
            1,
            TOTAL_HEADS // world_size,
            QUERY_HEAD_DIM,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        for _ in range(layers)
    ]
    gathered_queries = [
        torch.empty(
            1,
            TOTAL_HEADS,
            QUERY_HEAD_DIM,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        for _ in range(layers)
    ]

    with torch.cuda.stream(stream):
        channel.all_gather_heads(local_queries[0], gathered_queries[0])
        channel.lse_reduce_scatter(inputs[0], lses[0], outputs[0])
    stream.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with pool.capture(
        stream, channel_id="graph:layer-stack"
    ) as graph_channel, torch.cuda.graph(graph, stream=stream):
        for layer in range(layers):
            graph_channel.all_gather_heads(
                local_queries[layer], gathered_queries[layer]
            )
            graph_channel.lse_reduce_scatter(
                inputs[layer], lses[layer], outputs[layer]
            )
    stream.synchronize()

    replay_count = int(os.getenv("B12X_PCIE_DCP_GRAPH_REPLAYS", "8"))
    if replay_count <= 0:
        raise ValueError("B12X_PCIE_DCP_GRAPH_REPLAYS must be positive")
    for replay in range(replay_count):
        expected = []
        expected_queries = []
        for layer in range(layers):
            step = 1000 * replay + layer + 100
            partial_output, partial_lse = _rank_inputs(
                step,
                rank,
                MAX_BATCH,
                torch.bfloat16,
                device,
            )
            inputs[layer].copy_(partial_output)
            lses[layer].copy_(partial_lse)
            local_queries[layer].copy_(
                _rank_query(
                    step,
                    rank,
                    world_size,
                    1,
                    torch.float8_e4m3fn,
                    device,
                )
            )
            expected_queries.append(
                torch.cat(
                    [
                        _rank_query(
                            step,
                            source,
                            world_size,
                            1,
                            torch.float8_e4m3fn,
                            device,
                        )
                        for source in range(world_size)
                    ],
                    dim=1,
                )
            )
            expected.append(
                _reference(
                    step,
                    rank,
                    world_size,
                    MAX_BATCH,
                    torch.bfloat16,
                    device,
                )
            )
        stream.wait_stream(torch.cuda.current_stream(device))
        graph.replay()
        stream.synchronize()
        for out, reference in zip(outputs, expected, strict=True):
            torch.testing.assert_close(out, reference, rtol=2e-2, atol=2e-2)
        for out, reference in zip(gathered_queries, expected_queries, strict=True):
            torch.testing.assert_close(
                out.view(torch.uint8),
                reference.view(torch.uint8),
                rtol=0,
                atol=0,
            )

    # One captured operation has odd capture-time parity. Adjacent A -> A
    # replays must advance the device-owned slot at execution time.
    odd_input = torch.empty(
        1,
        TOTAL_HEADS // world_size,
        QUERY_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    odd_output = torch.empty(
        1,
        TOTAL_HEADS,
        QUERY_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    with torch.cuda.stream(stream):
        channel.all_gather_heads(odd_input, odd_output)
    stream.synchronize()
    dist.barrier()

    odd_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(odd_graph, stream=stream):
        channel.all_gather_heads(odd_input, odd_output)
    stream.synchronize()

    # Prime both slots after capture, then observe them read-only. A graph with
    # a host-baked slot changes the same word on both replays; execution-owned
    # parity changes alternate words.
    with torch.cuda.stream(stream):
        for value in (11.0, 12.0):
            odd_input.fill_(value)
            channel.all_gather_heads(odd_input, odd_output)
    # Under the push transport the sampled word is written by a peer, and a
    # peer's next launch pushes into the other slot before its barrier, so
    # every rank must have finished a launch before any rank samples or
    # launches again; the rank barrier after each sample closes that window.
    stream.synchronize()
    dist.barrier()
    snapshots = [_local_staging_words(channel, stream)]
    dist.barrier()
    for value in (1.0, 2.0):
        with torch.cuda.stream(stream):
            odd_input.fill_(value)
            odd_graph.replay()
        stream.synchronize()
        dist.barrier()
        snapshots.append(_local_staging_words(channel, stream))
        torch.testing.assert_close(
            odd_output, torch.full_like(odd_output, value), rtol=0, atol=0
        )
        dist.barrier()

    changed_slots = [
        {
            slot
            for slot, (before, after) in enumerate(
                zip(snapshots[index], snapshots[index + 1], strict=True)
            )
            if before != after
        }
        for index in range(2)
    ]
    assert all(len(changed) == 1 for changed in changed_slots)
    assert changed_slots[0] != changed_slots[1]


def _check_semantic_capture_warmup(
    pool: PCIeDCPA2APool,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    """Match vLLM's eager DCP warmup inside a graph-owner scope."""
    stream = torch.cuda.Stream(device=device)
    local_query = _rank_query(
        900,
        rank,
        world_size,
        1,
        torch.bfloat16,
        device,
    )
    gathered = torch.empty(
        1,
        TOTAL_HEADS,
        QUERY_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    expected = torch.cat(
        [
            _rank_query(
                900,
                source,
                world_size,
                1,
                torch.bfloat16,
                device,
            )
            for source in range(world_size)
        ],
        dim=1,
    )
    pool.prepare_channels(("graph:warmup",))

    graph = torch.cuda.CUDAGraph()
    with (
        torch.cuda.stream(stream),
        pool.capture(stream, channel_id="graph:warmup") as graph_channel,
    ):
        warmup_channel = pool.for_stream(stream, channel_id="eager:dcp")
        assert warmup_channel is graph_channel
        warmup_channel.all_gather_heads(local_query, gathered)
        stream.synchronize()
        torch.testing.assert_close(gathered, expected, rtol=0, atol=0)

        with torch.cuda.graph(graph, stream=stream):
            graph_channel.all_gather_heads(local_query, gathered)

    # The eager DCP channel was already bound to vLLM's default stream. Its
    # original mapping must be usable again after the graph-owner scope exits.
    eager_channel = pool.for_stream(channel_id="eager:dcp")
    assert eager_channel is not graph_channel
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    torch.testing.assert_close(gathered, expected, rtol=0, atol=0)


def _check_teardown_retry(
    pool: PCIeDCPA2APool,
    rank: int,
    device: torch.device,
) -> None:
    """Inject one local unmap failure and prove every rank retries coherently."""

    assert pool._all_channels
    channel = pool._all_channels[0]
    assert channel._ipc is not None
    original_close = channel._ipc.cudaIpcCloseMemHandle
    injected = False

    if rank == 0:

        def fail_once(ptr):
            nonlocal injected
            if not injected:
                injected = True
                raise RuntimeError("injected DCP IPC unmap failure")
            return original_close(ptr)

        channel._ipc.cudaIpcCloseMemHandle = fail_once

    failed = False
    try:
        pool.close()
    except RuntimeError as exc:
        failed = "IPC unmap" in str(exc)
    verdict = torch.tensor(int(failed), device=device, dtype=torch.int32)
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
    assert int(verdict.item()) == 1
    assert not pool._closed

    if rank == 0:
        channel._ipc.cudaIpcCloseMemHandle = original_close
    pool.close()
    assert pool._closed


def _check_queued_mixed_grid_graph(
    pool: PCIeDCPA2APool,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    """Stress slot retirement across odd, mixed-grid graph replays.

    The graph intentionally contains 2-block gather, 16-block LSE, then
    2-block gather.  Its odd transport-node count flips the starting slot on
    every replay.  Replays are queued without a rank barrier or device sync,
    and rank zero pauses after its first enqueue so peer hosts can enqueue the
    next replay before rank zero enqueues its matching launch.
    """

    batch = MAX_BATCH
    local_heads = TOTAL_HEADS // world_size
    query_a = torch.empty(
        1,
        local_heads,
        QUERY_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    query_b = torch.empty_like(query_a)
    gathered_a = torch.empty(
        1,
        TOTAL_HEADS,
        QUERY_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    gathered_b = torch.empty_like(gathered_a)
    partial_output = torch.empty(
        batch,
        TOTAL_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    partial_lse = torch.empty(
        batch,
        TOTAL_HEADS,
        dtype=torch.float32,
        device=device,
    )
    reduced = torch.empty(
        batch,
        local_heads,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )

    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with pool.capture(channel_id="graph:mixed-grid") as channel, torch.cuda.graph(
        graph
    ):
        channel.all_gather_heads(query_a, gathered_a)
        channel.lse_reduce_scatter(partial_output, partial_lse, reduced)
        channel.all_gather_heads(query_b, gathered_b)

    replay_steps = (4100, 4200, 4300)
    local_payloads = []
    local_lses = []
    local_queries_a = []
    local_queries_b = []
    for step in replay_steps:
        payload, lse = _rank_inputs(
            step,
            rank,
            batch,
            torch.bfloat16,
            device,
        )
        local_payloads.append(payload)
        local_lses.append(lse)
        local_queries_a.append(
            _rank_query(
                step + 100,
                rank,
                world_size,
                1,
                torch.bfloat16,
                device,
            )
        )
        local_queries_b.append(
            _rank_query(
                step + 200,
                rank,
                world_size,
                1,
                torch.bfloat16,
                device,
            )
        )

    reduced_snapshots = [torch.empty_like(reduced) for _ in replay_steps]
    gathered_a_snapshots = [torch.empty_like(gathered_a) for _ in replay_steps]
    gathered_b_snapshots = [torch.empty_like(gathered_b) for _ in replay_steps]
    torch.cuda.synchronize(device)
    for replay_index in range(len(replay_steps)):
        partial_output.copy_(local_payloads[replay_index])
        partial_lse.copy_(local_lses[replay_index])
        query_a.copy_(local_queries_a[replay_index])
        query_b.copy_(local_queries_b[replay_index])
        graph.replay()
        reduced_snapshots[replay_index].copy_(reduced)
        gathered_a_snapshots[replay_index].copy_(gathered_a)
        gathered_b_snapshots[replay_index].copy_(gathered_b)
        if replay_index == 0 and rank == 0:
            time.sleep(0.02)
    torch.cuda.synchronize(device)
    dist.barrier()

    for replay_index, step in enumerate(replay_steps):
        expected_reduced = _reference(
            step,
            rank,
            world_size,
            batch,
            torch.bfloat16,
            device,
        )
        expected_a = torch.cat(
            [
                _rank_query(
                    step + 100,
                    source,
                    world_size,
                    1,
                    torch.bfloat16,
                    device,
                )
                for source in range(world_size)
            ],
            dim=1,
        )
        expected_b = torch.cat(
            [
                _rank_query(
                    step + 200,
                    source,
                    world_size,
                    1,
                    torch.bfloat16,
                    device,
                )
                for source in range(world_size)
            ],
            dim=1,
        )
        torch.testing.assert_close(
            reduced_snapshots[replay_index],
            expected_reduced,
            rtol=2e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            gathered_a_snapshots[replay_index], expected_a, rtol=0, atol=0
        )
        torch.testing.assert_close(
            gathered_b_snapshots[replay_index], expected_b, rtol=0, atol=0
        )


KIMI_LATENT_WIDTH = 3584
KIMI_ROUTER_WIDTH = 896
KIMI_PAIR_BATCHES = (1, 2, 3, 4, 5, 8)


def _kimi_shard_width(width: int, world_size: int) -> int:
    """This rank's share of a Kimi projection padded to whole 16-byte packs
    (the model pads every shard to a multiple of eight elements, so nine
    ranks hold 400 latent columns and 104 experts each)."""
    return -(-width // world_size // 8) * 8


def _kimi_rank_rows(
    step: int, source_rank: int, batch: int, world_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kimi decode projection shards with router logits that exercise the
    selection's tie and non-finite handling: a few distinct values repeated
    across experts, one row of all-equal logits, and +-inf / NaN entries."""
    generator = torch.Generator(device="cpu").manual_seed(50000 * step + source_rank)
    down_width = _kimi_shard_width(KIMI_LATENT_WIDTH, world_size)
    router_width = _kimi_shard_width(KIMI_ROUTER_WIDTH, world_size)
    down = torch.randn(batch, down_width, generator=generator, dtype=torch.float32)
    router = torch.randn(batch, router_width, generator=generator, dtype=torch.float32)
    # Ties: quantize half of the rows to sixteen distinct logit values.
    quantized = (router * 2).round() / 2
    router[: (batch + 1) // 2] = quantized[: (batch + 1) // 2]
    if batch >= 2:
        router[1] = 0.25  # every expert equal: ids must follow the index order
    if batch >= 3:
        router[2, ::7] = float("inf")
        router[2, 3::11] = float("-inf")
        router[2, 5::13] = float("nan")
    # Padding columns beyond the logical width are never selected: poison them.
    return (
        down.to(device=device, dtype=torch.bfloat16),
        router.to(device=device),
    )


def _kimi_correction_bias(world_size: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(777 + world_size)
    bias = torch.randn(KIMI_ROUTER_WIDTH, generator=generator, dtype=torch.float32) * 0.1
    bias[::5] = 0.0  # exact-zero selections must canonicalize the same way
    return bias.to(device=device)


def _check_pair_kimi_topk(rank: int, world_size: int, device: torch.device) -> None:
    """The fused pair gather + expert selection equals the served two-launch
    path (paired gather clipped to the logical widths, then the batched
    selection kernel) bit for bit: gathered latent rows, weights and ids,
    eagerly and under CUDA graph replay, for padded shards and batches of
    one to eight rows."""
    down_width = _kimi_shard_width(KIMI_LATENT_WIDTH, world_size)
    router_width = _kimi_shard_width(KIMI_ROUTER_WIDTH, world_size)
    combined = down_width * 2 + router_width * 4
    pool = PCIeDCPA2APool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_batch_size=8,
        total_heads=world_size,
        head_dim=combined,
        query_head_dim=combined,
        max_concurrent_channels=2,
    )
    try:
        pool.prepare_channels(("eager:kimi", "graph:kimi"))
        bias = _kimi_correction_bias(world_size, device)
        for step, batch in enumerate(KIMI_PAIR_BATCHES, start=600):
            down, router = _kimi_rank_rows(step, rank, batch, world_size, device)
            expected_down = torch.empty(
                batch, KIMI_LATENT_WIDTH, dtype=torch.bfloat16, device=device
            )
            expected_router = torch.empty(
                batch, KIMI_ROUTER_WIDTH, dtype=torch.float32, device=device
            )
            pool.all_gather_pair(
                down, router, expected_down, expected_router, channel_id="eager:kimi"
            )
            expected_weights, expected_ids = pool.kimi_topk16(
                expected_router, bias, channel_id="eager:kimi"
            )
            fused_down, fused_weights, fused_ids = pool.all_gather_pair_kimi_topk(
                down, router, bias, channel_id="eager:kimi"
            )
            torch.cuda.synchronize(device)
            assert torch.equal(fused_down, expected_down), f"fused latent rows batch {batch}"
            assert torch.equal(fused_ids, expected_ids), f"fused expert ids batch {batch}"
            assert torch.equal(
                fused_weights.view(torch.int32), expected_weights.view(torch.int32)
            ), f"fused expert weights batch {batch}"
            _stage(rank, f"pair_kimi_topk_eager_batch{batch}_ok")

        # Graph replay: two layers captured once, replayed with new inputs.
        layers = 2
        batch = 4
        stream = torch.cuda.Stream(device=device)
        downs = [
            torch.empty(batch, down_width, dtype=torch.bfloat16, device=device)
            for _ in range(layers)
        ]
        routers = [
            torch.empty(batch, router_width, dtype=torch.float32, device=device)
            for _ in range(layers)
        ]
        out_downs = [
            torch.empty(batch, KIMI_LATENT_WIDTH, dtype=torch.bfloat16, device=device)
            for _ in range(layers)
        ]
        out_weights = [
            torch.empty(batch, 16, dtype=torch.float32, device=device)
            for _ in range(layers)
        ]
        out_ids = [
            torch.empty(batch, 16, dtype=torch.int32, device=device)
            for _ in range(layers)
        ]
        graph = torch.cuda.CUDAGraph()
        _stage(rank, "pair_kimi_topk_graph_capture")
        with pool.capture(stream, channel_id="graph:kimi") as graph_channel, torch.cuda.graph(
            graph, stream=stream
        ):
            for layer in range(layers):
                graph_channel.all_gather_pair_kimi_topk(
                    downs[layer],
                    routers[layer],
                    bias,
                    out_downs[layer],
                    out_weights[layer],
                    out_ids[layer],
                )
        stream.synchronize()
        _stage(rank, "pair_kimi_topk_graph_captured")
        for replay in range(3):
            for layer in range(layers):
                step = 7000 + 10 * replay + layer
                down, router = _kimi_rank_rows(step, rank, batch, world_size, device)
                downs[layer].copy_(down)
                routers[layer].copy_(router)
            torch.cuda.synchronize(device)
            dist.barrier()
            with torch.cuda.stream(stream):
                graph.replay()
            stream.synchronize()
            for layer in range(layers):
                step = 7000 + 10 * replay + layer
                down, router = _kimi_rank_rows(step, rank, batch, world_size, device)
                expected_down = torch.empty(
                    batch, KIMI_LATENT_WIDTH, dtype=torch.bfloat16, device=device
                )
                expected_router = torch.empty(
                    batch, KIMI_ROUTER_WIDTH, dtype=torch.float32, device=device
                )
                pool.all_gather_pair(
                    down, router, expected_down, expected_router, channel_id="eager:kimi"
                )
                expected_weights, expected_ids = pool.kimi_topk16(
                    expected_router, bias, channel_id="eager:kimi"
                )
                torch.cuda.synchronize(device)
                assert torch.equal(out_downs[layer], expected_down), f"replay {replay} layer {layer} latent"
                assert torch.equal(out_ids[layer], expected_ids), f"replay {replay} layer {layer} ids"
                assert torch.equal(
                    out_weights[layer].view(torch.int32), expected_weights.view(torch.int32)
                ), f"replay {replay} layer {layer} weights"
            _stage(rank, f"pair_kimi_topk_graph_replay{replay}_ok")
        torch.cuda.synchronize(device)
        dist.barrier()
    finally:
        pool.close()



def _time_pair_kimi_topk(rank: int, world_size: int, device: torch.device) -> None:
    """Replay timing of the served two-launch selection (paired gather, then
    the batched top-16 kernel) against the fused launch, for batches of one to
    eight rows, ``B12X_PCIE_DCP_A2A_TIME_LAYERS`` layers per graph (default 30)
    and ``B12X_PCIE_DCP_A2A_TIME_REPLAYS`` replays (default 20). Every rank
    prints the median per-layer time; the collective's time is the slowest
    rank's. Enabled by ``B12X_PCIE_DCP_A2A_TIME=1``."""
    import json
    import statistics

    layers = int(os.getenv("B12X_PCIE_DCP_A2A_TIME_LAYERS", "30"))
    replays = int(os.getenv("B12X_PCIE_DCP_A2A_TIME_REPLAYS", "20"))
    down_width = _kimi_shard_width(KIMI_LATENT_WIDTH, world_size)
    router_width = _kimi_shard_width(KIMI_ROUTER_WIDTH, world_size)
    combined = down_width * 2 + router_width * 4
    pool = PCIeDCPA2APool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_batch_size=8,
        total_heads=world_size,
        head_dim=combined,
        query_head_dim=combined,
        max_concurrent_channels=3,
    )
    try:
        pool.prepare_channels(("eager:time", "graph:time-legacy", "graph:time-fused"))
        bias = _kimi_correction_bias(world_size, device)
        stream = torch.cuda.Stream(device=device)
        results = []
        for batch in (1, 2, 4, 8):
            down, router = _kimi_rank_rows(9000 + batch, rank, batch, world_size, device)
            # Eager launches prepare both launcher variants on this stream.
            with torch.cuda.stream(stream):
                out_down = torch.empty(batch, KIMI_LATENT_WIDTH, dtype=torch.bfloat16, device=device)
                out_router = torch.empty(batch, KIMI_ROUTER_WIDTH, dtype=torch.float32, device=device)
                pool.all_gather_pair(down, router, out_down, out_router, channel_id="eager:time")
                pool.kimi_topk16(out_router, bias, channel_id="eager:time")
                pool.all_gather_pair_kimi_topk(down, router, bias, channel_id="eager:time")
            stream.synchronize()
            dist.barrier()
            timings = {}
            for variant in ("legacy", "fused"):
                out_downs = [torch.empty(batch, KIMI_LATENT_WIDTH, dtype=torch.bfloat16, device=device) for _ in range(layers)]
                out_routers = [torch.empty(batch, KIMI_ROUTER_WIDTH, dtype=torch.float32, device=device) for _ in range(layers)]
                out_weights = [torch.empty(batch, 16, dtype=torch.float32, device=device) for _ in range(layers)]
                out_ids = [torch.empty(batch, 16, dtype=torch.int32, device=device) for _ in range(layers)]
                graph = torch.cuda.CUDAGraph()
                with pool.capture(stream, channel_id=f"graph:time-{variant}") as channel, torch.cuda.graph(graph, stream=stream):
                    for layer in range(layers):
                        if variant == "legacy":
                            channel.all_gather_pair(down, router, out_downs[layer], out_routers[layer])
                            channel.kimi_topk16(out_routers[layer], bias, out_weights[layer], out_ids[layer])
                        else:
                            channel.all_gather_pair_kimi_topk(down, router, bias, out_downs[layer], out_weights[layer], out_ids[layer])
                stream.synchronize()
                samples = []
                for _ in range(replays):
                    dist.barrier()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(stream):
                        start.record(stream)
                        graph.replay()
                        end.record(stream)
                    stream.synchronize()
                    samples.append(start.elapsed_time(end) * 1000.0 / layers)
                timings[variant] = statistics.median(samples)
                del graph
                torch.cuda.synchronize(device)
                dist.barrier()
            results.append({"rank": rank, "batch": batch, "legacy_us": timings["legacy"], "fused_us": timings["fused"]})
            print(json.dumps({"stage": "pair_kimi_topk_timing", **results[-1]}), flush=True)
        torch.cuda.synchronize(device)
        dist.barrier()
    finally:
        pool.close()

def _stage(rank: int, name: str) -> None:
    """One JSON line per rank and check boundary, so a hang or a collective
    sequence mismatch can be attributed to the check a rank was in."""
    print(json.dumps({"stage": name, "rank": rank}), flush=True)


def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        # A collective that never completes fails the run after this long
        # instead of NCCL's ten-minute default.
        timeout=datetime.timedelta(
            seconds=int(os.getenv("B12X_PCIE_DCP_A2A_TEST_NCCL_TIMEOUT_S", "600"))
        ),
    )
    pool = PCIeDCPA2APool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_batch_size=MAX_BATCH,
        total_heads=TOTAL_HEADS,
        head_dim=HEAD_DIM,
        query_head_dim=QUERY_HEAD_DIM,
        max_concurrent_channels=2,
    )
    closed = False
    try:
        pool.prepare_channels(("eager:dcp", "graph"))
        if rank == 0:
            print("A2A GPU gate: eager", flush=True)
        _stage(rank, "eager")
        _check_eager(pool, rank, world_size, device)
        dist.barrier()
        _stage(rank, "eager_adjacency")
        _check_eager_adjacency(pool, rank, world_size, device)
        dist.barrier()
        _stage(rank, "semantic_capture_warmup")
        _check_semantic_capture_warmup(pool, rank, world_size, device)
        dist.barrier()
        if rank == 0:
            print("A2A GPU gate: graph replay", flush=True)
        _stage(rank, "graph")
        _check_graph(pool, rank, world_size, device)
        dist.barrier()
        if rank == 0:
            print("A2A GPU gate: queued mixed-grid skew", flush=True)
        _stage(rank, "queued_mixed_grid_graph")
        _check_queued_mixed_grid_graph(pool, rank, world_size, device)
        dist.barrier()
        # The paired projection gather runs after the head/LSE checks so a
        # failure in either family is attributable to it alone.
        _stage(rank, "pair_eager")
        _check_pair_eager(pool, rank, world_size, device)
        dist.barrier()
        _stage(rank, "pair_graph")
        _check_pair_graph(pool, rank, world_size, device)
        dist.barrier()
        # The fused Kimi router path needs its own pool (Kimi row widths).
        _stage(rank, "pair_kimi_topk")
        _check_pair_kimi_topk(rank, world_size, device)
        if os.getenv("B12X_PCIE_DCP_A2A_TIME", "0") == "1":
            _stage(rank, "pair_kimi_topk_timing")
            _time_pair_kimi_topk(rank, world_size, device)
        _stage(rank, "complete")
        if rank == 0:
            print("A2A GPU gate: complete", flush=True)
        torch.cuda.synchronize(device)
        if os.getenv("B12X_PCIE_DCP_TEST_TEARDOWN_RETRY", "0") == "1":
            _check_teardown_retry(pool, rank, device)
            closed = True
    except BaseException as exc:  # noqa: BLE001
        # A rank that raises must fail loudly and at once: its teardown
        # collectives would otherwise pair with the other ranks' next
        # collectives and every rank hangs until the NCCL timeout, hiding
        # the exception. Print the traceback and end the process.
        import traceback

        print(
            json.dumps({"stage": "exception", "rank": rank, "error": repr(exc)[:400]}),
            flush=True,
        )
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(3)
    finally:
        if not closed:
            pool.close()
        dist.destroy_process_group()


def _residency_rejection_worker(rank: int, world_size: int, port: int) -> None:
    if rank != 0:
        # Simulate one constrained/MIG-like rank in an otherwise full device
        # group; every peer must reject before any IPC allocation.
        os.environ.pop("B12X_PCIE_TEST_VISIBLE_SM_COUNT", None)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    original_cuda_rt = pcie_dcp_a2a.CudaRTLibrary

    def unexpected_cuda_rt():
        raise AssertionError("CUDA IPC allocation started before residency rejection")

    pcie_dcp_a2a.CudaRTLibrary = unexpected_cuda_rt
    try:
        with pytest.raises(RuntimeError, match="requires at least 64 visible SMs"):
            PCIeDCPA2A.from_process_group(
                process_group=dist.group.WORLD,
                device=device,
                max_batch_size=MAX_BATCH,
                total_heads=TOTAL_HEADS,
                head_dim=HEAD_DIM,
                query_head_dim=QUERY_HEAD_DIM,
            )
        dist.barrier()
    finally:
        pcie_dcp_a2a.CudaRTLibrary = original_cuda_rt
        dist.destroy_process_group()


def _preflight_rejection_worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    original_cuda_rt = pcie_dcp_a2a.CudaRTLibrary
    original_allocate = PCIeOneshotAllReduce._allocate_shared_buffer

    class FakeExt:
        @staticmethod
        def meta_size():
            return 256

    def fail_cuda_rt():
        raise RuntimeError("injected rank-zero pre-allocation failure")

    def unexpected_allocate(*args, **kwargs):
        raise AssertionError(
            "CUDA IPC allocation started after a peer preflight failure"
        )

    if rank == 0:
        pcie_dcp_a2a.CudaRTLibrary = fail_cuda_rt
    PCIeOneshotAllReduce._allocate_shared_buffer = staticmethod(unexpected_allocate)
    try:
        with pytest.raises(RuntimeError, match="pre-allocation setup"):
            PCIeDCPA2A.from_process_group(
                process_group=dist.group.WORLD,
                device=device,
                max_batch_size=MAX_BATCH,
                total_heads=TOTAL_HEADS,
                head_dim=HEAD_DIM,
                query_head_dim=QUERY_HEAD_DIM,
                ext_module=FakeExt(),
            )
        dist.barrier()
    finally:
        pcie_dcp_a2a.CudaRTLibrary = original_cuda_rt
        PCIeOneshotAllReduce._allocate_shared_buffer = original_allocate
        dist.destroy_process_group()


def test_pcie_dcp_a2a_eager_and_cuda_graph_correctness():
    if (
        os.getenv("B12X_PCIE_DCP_TEST_EXPECT_RESIDENCY_REJECTION") == "1"
        or os.getenv("B12X_PCIE_DCP_TEST_EXPECT_PREFLIGHT_REJECTION") == "1"
    ):
        pytest.skip("running the reduced-SM residency rejection gate")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    world_size = int(os.getenv("B12X_PCIE_DCP_A2A_WORLD_SIZE", "2"))
    if world_size not in (2, 4, 8, 9, 16):
        pytest.skip("PCIe DCP A2A supports world sizes 2, 4, 8, 9, and 16")
    if TOTAL_HEADS % world_size:
        pytest.skip(
            f"B12X_PCIE_DCP_A2A_TEST_TOTAL_HEADS={TOTAL_HEADS} is not divisible "
            f"by world size {world_size}"
        )
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            f"need {world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    mp.spawn(_worker, args=(world_size, _free_port()), nprocs=world_size, join=True)


def test_pcie_dcp_a2a_rejects_reduced_sm_slice_before_ipc_allocation():
    if os.getenv("B12X_PCIE_DCP_TEST_EXPECT_RESIDENCY_REJECTION") != "1":
        pytest.skip("set the reduced-SM residency rejection gate to run this test")
    visible_sms = int(os.getenv("B12X_PCIE_TEST_VISIBLE_SM_COUNT", "0") or "0")
    assert 0 < visible_sms < pcie_dcp_a2a.DCP_A2A_REQUIRED_SMS, (
        "set B12X_PCIE_TEST_VISIBLE_SM_COUNT below "
        f"{pcie_dcp_a2a.DCP_A2A_REQUIRED_SMS} to exercise the residency gate"
    )
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    world_size = int(os.getenv("B12X_PCIE_DCP_A2A_WORLD_SIZE", "2"))
    if world_size not in (2, 4, 8, 16):
        pytest.skip("PCIe DCP A2A supports world sizes 2, 4, 8, and 16")
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            f"need {world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    mp.spawn(
        _residency_rejection_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
        join=True,
    )


def test_pcie_dcp_a2a_coordinates_one_rank_preflight_failure_before_ipc():
    if os.getenv("B12X_PCIE_DCP_TEST_EXPECT_PREFLIGHT_REJECTION") != "1":
        pytest.skip("set the one-rank preflight rejection gate to run this test")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    world_size = int(os.getenv("B12X_PCIE_DCP_A2A_WORLD_SIZE", "2"))
    if world_size not in (2, 4, 8):
        pytest.skip("PCIe DCP A2A supports world sizes 2, 4, and 8")
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            f"need {world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    mp.spawn(
        _preflight_rejection_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
        join=True,
    )
