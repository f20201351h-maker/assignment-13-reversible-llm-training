from dataclasses import replace

import pytest
import torch

from revllm.backend import configure_attention_backend, profile_attention_operator
from revllm.model import ModelConfig


@pytest.fixture(autouse=True)
def restore_auto_backend():
    yield
    configure_attention_backend("auto", torch.device("cpu"))


def test_efficient_policy_disables_every_fallback(monkeypatch):
    # Policy configuration is testable on a CUDA device descriptor without
    # allocating CUDA memory; actual support is verified by the operator probe.
    result = configure_attention_backend("efficient", torch.device("cuda", 0))
    assert result["effective_policy"] == "efficient"
    assert result["flags"] == {
        "flash_sdp_enabled": False,
        "mem_efficient_sdp_enabled": True,
        "math_sdp_enabled": False,
        "cudnn_sdp_enabled": False,
    }


def test_efficient_policy_rejects_cpu_unless_evaluation_records_fallback():
    with pytest.raises(RuntimeError, match="requires CUDA"):
        configure_attention_backend("efficient", torch.device("cpu"))
    result = configure_attention_backend(
        "efficient", torch.device("cpu"), allow_cpu_efficient_fallback=True
    )
    assert result["requested_policy"] == "efficient"
    assert result["effective_policy"] == "math"
    assert result["cpu_fallback"] is True
    assert "evaluation is running on CPU" in result["fallback_reason"]
    assert result["flags"]["math_sdp_enabled"] is True


def test_cpu_math_operator_probe_preserves_rng_state():
    configure_attention_backend("math", torch.device("cpu"))
    cfg = ModelConfig(vocab_size=32, block_size=4, n_layer=1, n_head=2, n_embd=8)
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    result = profile_attention_operator(
        cfg, torch.device("cpu"), batch_size=1, amp=False, expected_policy="math"
    )
    assert torch.equal(torch.get_rng_state(), before)
    assert "aten::scaled_dot_product_attention" in result["operators"]
    assert any("attention_math" in name for name in result["operators"])


@pytest.mark.parametrize("policy", ["flash", "", "EFFICIENT"])
def test_unknown_backend_is_rejected(policy):
    with pytest.raises(ValueError, match="attention_backend"):
        configure_attention_backend(policy, torch.device("cpu"))
