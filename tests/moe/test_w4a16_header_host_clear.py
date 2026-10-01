"""Host-side contract of the resident-grid header clear (D3, no GPU)."""

from __future__ import annotations

import torch

from b12x.moe._shared.kernels.w4a16 import kernel as w4a16_kernel
from b12x.moe._shared.kernels.w4a16 import reference_grouped

HEADER = reference_grouped.HEADER_WORDS


def _dirty(words: int = 4096) -> torch.Tensor:
    return torch.full((words,), 7, dtype=torch.int32)


def test_default_clears_the_header_before_every_launch(monkeypatch) -> None:
    monkeypatch.delenv("B12X_W4A16_HEADER_HOST_CLEAR", raising=False)
    w4a16_kernel._RESIDENT_HEADER_CLEARED.clear()
    workspace = _dirty()
    w4a16_kernel._clear_resident_header(workspace)
    assert torch.all(workspace[:HEADER] == 0)
    assert torch.all(workspace[HEADER:] == 7)
    workspace[:HEADER] = 5
    w4a16_kernel._clear_resident_header(workspace)
    assert torch.all(workspace[:HEADER] == 0)
    assert not w4a16_kernel._RESIDENT_HEADER_CLEARED


def test_opt_out_clears_once_per_workspace_storage(monkeypatch) -> None:
    monkeypatch.setenv("B12X_W4A16_HEADER_HOST_CLEAR", "0")
    w4a16_kernel._RESIDENT_HEADER_CLEARED.clear()
    first = _dirty()
    second = _dirty()
    w4a16_kernel._clear_resident_header(first)
    assert torch.all(first[:HEADER] == 0)
    # A completed launch leaves the header self-reset; the host must not
    # touch it again (a graph captured now carries no fill node).
    first[:HEADER] = 3
    w4a16_kernel._clear_resident_header(first)
    assert torch.all(first[:HEADER] == 3)
    # Another storage (another arena) still gets its first clear.
    w4a16_kernel._clear_resident_header(second)
    assert torch.all(second[:HEADER] == 0)
    assert len(w4a16_kernel._RESIDENT_HEADER_CLEARED) == 2
    w4a16_kernel._RESIDENT_HEADER_CLEARED.clear()


def test_kernel_resets_every_header_word_it_arrives_on() -> None:
    """The split-K lock is reset by its last contributor and the grid
    barrier's count by the last arriver, so no header word except the
    monotonic sense survives a completed launch."""
    import inspect

    gemm_cls = next(
        cls for name, cls in vars(w4a16_kernel).items()
        if isinstance(cls, type) and hasattr(cls, "_publish_reduction_turn")
    )
    publish = inspect.getsource(gemm_cls._publish_reduction_turn)
    assert "if reset:" in publish and "st_global_i32(lock_addr, Int32(0))" in publish
    barrier = inspect.getsource(w4a16_kernel.W4A16FusedMoeKernel._grid_barrier)
    assert "st_global_i32(count_addr, Int32(0))" in barrier
    assert "red_add_global_release_i32(sense_addr, Int32(1))" in barrier
