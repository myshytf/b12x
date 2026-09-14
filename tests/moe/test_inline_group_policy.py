"""Host policy regression: physical SM count differs from the reserved budget.

This test constructs the actual kernel policy with device metadata only. It
does not compile CUDA or qualify native arithmetic.
"""

from types import SimpleNamespace

import pytest

from b12x.moe._shared.kernels.w4a16 import kernel


@pytest.mark.parametrize("width", (256, 384))
@pytest.mark.parametrize(
    "rows,reserve,enabled,direct,phases,expected",
    (
        (4, 2, True, True, "fc2", True),
        (4, 0, True, True, "fc2", False),
        (4, 2, False, True, "fc2", False),
        (4, 2, True, False, "fc2", False),
        (4, 2, True, True, "both", False),
        (1, 2, True, True, "fc2", False),
        (16, 2, True, True, "fc2", False),
    ),
)
def test_inline_policy_uses_reserved_sm_budget(
    monkeypatch, width, rows, reserve, enabled, direct, phases, expected
):
    properties = SimpleNamespace(
        multi_processor_count=188, shared_memory_per_block_optin=101376
    )
    monkeypatch.setattr(kernel.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(kernel.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(kernel.torch.cuda, "get_device_properties", lambda _: properties)
    for name, value in {
        "B12X_W4A16_REFERENCE_GROUPED": "1",
        "B12X_W4A16_REFERENCE_GROUPED_PHASES": phases,
        "B12X_W4A16_GROUP_BUILDER_INLINE": str(int(enabled)),
        "B12X_W4A16_SMALL_M_SPLITK": "1",
        "B12X_W4A16_FUSED_SM_RESERVE": str(reserve),
        "B12X_SQG_XOR_CHEB_T12_SMEM": "1",
        "B12X_SQG_XOR_CHEB_T12_DIRECT_SMEM": "1",
    }.items():
        monkeypatch.setenv(name, value)
    selected = kernel.W4A16FusedMoeKernel(
        size_m=rows,
        hidden_size=3584,
        intermediate_size=width,
        num_experts=896,
        top_k=16,
        activation="situ",
        apply_router_weight_on_input=False,
        zero_fc2_output=True,
        fc1_tile_n=128,
        fc1_tile_k=128,
        fc2_tile_n=128,
        fc2_tile_k=128,
        moe_block_size=8,
        max_m_blocks=128,
        element_dtype="fp16",
        weight_layout="trellis3_t256",
        scale_format="e4m3_k32",
        w13_layout="trellis3_t256_proj",
        trellis_bits=2,
        direct_topk_routes=direct,
        use_expert_map=direct,
        intermediate_rotation=True,
        full_rotation=True,
        coupled_hadamard=True,
        rotation_input_dtype="bf16",
        broadcast_suh=True,
    )
    assert selected.sms == 188 - reserve
    assert selected.reference_grouped_inline == expected
    assert selected.cta_threads == 256
    if expected:
        assert selected.sqg_xor_cheb_t12_smem_off >= 1536
