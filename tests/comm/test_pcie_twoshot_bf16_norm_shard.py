"""Host-side contract of the fused RMSNorm-shard BF16 all-reduce (no GPU)."""

from __future__ import annotations

import inspect

import pytest
import torch

from b12x.comm.pcie import _twoshot_bf16_norm_cute as norm_cute
from b12x.comm.pcie.pcie_twoshot_bf16 import PCIeTwoShotBF16


def _runtime(world_size: int = 9) -> PCIeTwoShotBF16:
    return PCIeTwoShotBF16._from_prepared_factory(
        rank=0,
        world_size=world_size,
        device=torch.device("cuda", 0),
        signal_ptrs=(0,) * 9,
        staging_ptrs=((0,) * 9, (0,) * 9),
        owned_buffers=(),
        ipc=None,
        exchange_group=None,
        max_rows=16 * 3584 // 8,
        row_elems=8,
        pack_stride=1,
        reduced_offset=256,
        slot_bytes=768,
    )


def test_norm_shard_is_opt_in_and_needs_the_push_transport() -> None:
    runtime = _runtime()
    inp = torch.zeros((4, 3584), dtype=torch.bfloat16)
    weight = torch.zeros((3584,), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="not enabled"):
        runtime.all_reduce_rms_norm_shard(inp, weight, 1e-6, 0, 400)
    runtime.norm_shard_enabled = True
    with pytest.raises(ValueError, match="push transport"):
        runtime._norm_shard_mode()
    runtime.all_reduce_mode = "push"
    assert runtime._norm_shard_mode() == "push_norm_shard"
    runtime._static_peers_enabled = True
    assert runtime._norm_shard_mode() == "push_static_norm_shard"


def test_norm_shard_launcher_rejects_unsupported_geometry() -> None:
    with pytest.raises(ValueError, match="invalid norm-shard"):
        norm_cute.get_twoshot_bf16_allreduce_norm_shard_launcher(9, 0, True, 0, 512, 8, 0, "pull")
    with pytest.raises(ValueError, match="TP9 only"):
        norm_cute.get_twoshot_bf16_allreduce_norm_shard_launcher(
            8, 0, True, 0, 512, 8, 0, "push_static_norm_shard"
        )
    with pytest.raises(ValueError, match="512 threads"):
        norm_cute.get_twoshot_bf16_allreduce_norm_shard_launcher(9, 0, True, 0, 256, 8, 0)
    with pytest.raises(ValueError, match="single-pack rows"):
        norm_cute.get_twoshot_bf16_allreduce_norm_shard_launcher(9, 0, True, 0, 512, 16, 0)


def test_norm_shard_epilogue_accumulates_in_float64_and_keeps_the_served_order() -> None:
    source = inspect.getsource(norm_cute._TwoShotPushAllReduceNormShardLaunch.kernel)
    epilogue = source.split("# Epilogue:", 1)[1]
    # Sum of squares and the scale in float64; one rounding of the scale to
    # fp32; the served kernel's (x * scale) * weight fp32 order; one bf16
    # rounding through the pack helper; zero fill past the logical width.
    assert "acc = Float64(0.0)" in epilogue
    assert "acc = acc + lo64 * lo64" in epilogue
    assert "cute.math.sqrt(variance, fastmath=False)" in epilogue
    assert "scales[group] = Float32(scale64)" in epilogue
    assert "(lo * scale) * wlo, (hi * scale) * whi" in epilogue
    assert "out_words[word_index] = Uint32(0)" in epilogue
    # The all-reduce phases precede the epilogue unchanged: three phases,
    # two barriers, the output packs stored by phases two and three.
    phases = source.split("# Epilogue:", 1)[0]
    assert phases.count("self._barrier(signals, local_rank)") == 2
    assert "self._store_pack(" in phases


def test_norm_shard_launcher_key_carries_its_operation() -> None:
    key = norm_cute._bf16_process_key("all_reduce_push_norm_shard", 9, 0, True, 0, 512, 8, 0)
    assert key[1] == "all_reduce_push_norm_shard"
    assert not norm_cute.is_twoshot_bf16_allreduce_norm_shard_launcher_prepared(
        9, 0, True, 0, 512, 8, 0
    )
