"""Explicit, auditable scaled-dot-product-attention backend selection."""

from __future__ import annotations

import gc
from typing import Any

import torch

from .model import CausalAttention, ModelConfig


ATTENTION_BACKENDS = frozenset({"auto", "efficient", "math"})


def attention_backend_flags() -> dict[str, bool]:
    """Return the process-wide CUDA SDPA switches used by PyTorch dispatch."""
    return {
        "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        "cudnn_sdp_enabled": bool(torch.backends.cuda.cudnn_sdp_enabled()),
    }


def configure_attention_backend(
    policy: str,
    device: torch.device,
    *,
    allow_cpu_efficient_fallback: bool = False,
) -> dict[str, Any]:
    """Apply a process-wide SDPA policy and return an audit record.

    Efficient attention is a CUDA-only experimental condition. Training and
    profiling callers leave ``allow_cpu_efficient_fallback`` false so an
    unavailable kernel cannot silently change the experiment. Checkpoint
    evaluation may explicitly opt into a recorded CPU math fallback.
    """
    if policy not in ATTENTION_BACKENDS:
        choices = ", ".join(sorted(ATTENTION_BACKENDS))
        raise ValueError(f"attention_backend must be one of: {choices}")

    requested = policy
    fallback_reason = None
    if policy == "efficient" and device.type != "cuda":
        if not allow_cpu_efficient_fallback:
            raise RuntimeError("The efficient attention backend requires CUDA")
        policy = "math"
        fallback_reason = (
            "Checkpoint requested CUDA efficient attention, but evaluation is "
            "running on CPU; explicitly using the CPU math backend."
        )

    enabled = {
        "auto": (True, True, True, True),
        "efficient": (False, True, False, False),
        "math": (False, False, True, False),
    }[policy]
    torch.backends.cuda.enable_flash_sdp(enabled[0])
    torch.backends.cuda.enable_mem_efficient_sdp(enabled[1])
    torch.backends.cuda.enable_math_sdp(enabled[2])
    torch.backends.cuda.enable_cudnn_sdp(enabled[3])
    return {
        "requested_policy": requested,
        "effective_policy": policy,
        "device_type": device.type,
        "cpu_fallback": fallback_reason is not None,
        "fallback_reason": fallback_reason,
        "flags": attention_backend_flags(),
    }


def profile_attention_operator(
    cfg: ModelConfig,
    device: torch.device,
    *,
    batch_size: int,
    amp: bool,
    expected_policy: str,
) -> dict[str, Any]:
    """Profile one real CausalAttention forward/backward without advancing RNG.

    CPU profiler activity captures CUDA operator names while avoiding profiler
    GPU instrumentation. The tensor shape matches the run's physical batch and
    model attention shape; only one attention layer is instantiated.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == "cuda" else []
    operators: list[str] = []
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        attention = CausalAttention(cfg).to(device).train()
        x = torch.randn(
            batch_size, cfg.block_size, cfg.n_embd, device=device, requires_grad=True
        )
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as trace:
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=amp and device.type == "cuda"):
                attention(x).sum().backward()
        operators = sorted({
            event.key for event in trace.key_averages()
            if any(token in event.key for token in (
                "scaled_dot", "efficient_attention", "flash_attention", "cudnn_attention"
            ))
        })
        del x, attention
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    efficient_seen = (any("efficient_attention" in name and "backward" not in name for name in operators)
                      and any("efficient_attention" in name and "backward" in name for name in operators))
    if expected_policy == "efficient" and not efficient_seen:
        raise RuntimeError(
            "Efficient attention was forced, but the profiled CausalAttention "
            f"forward/backward did not execute it: {operators}"
        )
    return {
        "profile_activity": "CPU",
        "module": "revllm.model.CausalAttention",
        "batch_size": batch_size,
        "sequence_length": cfg.block_size,
        "heads": cfg.n_head,
        "head_dimension": cfg.n_embd // cfg.n_head,
        "amp_fp16": bool(amp and device.type == "cuda"),
        "operators": operators,
        "efficient_operator_observed": efficient_seen,
    }


def prepare_attention_backend(
    policy: str,
    cfg: ModelConfig,
    device: torch.device,
    *,
    batch_size: int,
    amp: bool,
    allow_cpu_efficient_fallback: bool = False,
) -> dict[str, Any]:
    """Configure dispatch and, on CUDA, verify the operator before a run."""
    record = configure_attention_backend(
        policy, device,
        allow_cpu_efficient_fallback=allow_cpu_efficient_fallback,
    )
    if device.type == "cuda":
        record["operator_probe"] = profile_attention_operator(
            cfg,
            device,
            batch_size=batch_size,
            amp=amp,
            expected_policy=record["effective_policy"],
        )
    else:
        record["operator_probe"] = {
            "status": "NOT_RUN",
            "reason": "CUDA operator verification is not applicable on CPU",
        }
    return record
