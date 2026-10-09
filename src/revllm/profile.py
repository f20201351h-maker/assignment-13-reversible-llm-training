"""Fresh-process OOM trials and steady-state step measurements."""

from __future__ import annotations

import gc
import math
import statistics
import time
from dataclasses import replace
from typing import Callable

import torch

from .backend import prepare_attention_backend
from .model import TinyGPT, count_unique_parameters
from .train import RunConfig, make_optimizer


DEFAULT_BENCHMARK_REPETITIONS = 3
DEFAULT_BENCHMARK_WARMUP_UPDATES = 20
DEFAULT_BENCHMARK_MEASURED_UPDATES = 100
MAX_OVERFLOW_RETRIES = 10


def _one_step(model, optimizer, scaler, x, y, amp: bool,
              physical_batch: int | None = None) -> bool:
    """Run one attempted update and return whether the optimizer committed it."""
    if physical_batch is None:
        physical_batch = x.shape[0]
    if not 0 < physical_batch <= x.shape[0]:
        raise ValueError("Physical batch must fit within the effective batch")
    optimizer.zero_grad(set_to_none=True)
    losses_finite = torch.ones((), device=x.device, dtype=torch.bool)
    for start in range(0, x.shape[0], physical_batch):
        stop = start + physical_batch
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = model(x[start:stop], y[start:stop]) / y.numel()
        losses_finite &= torch.isfinite(loss.detach())
        scaler.scale(loss).backward()
    if not bool(losses_finite):
        raise RuntimeError("Non-finite loss during GPU measurement")
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    overflow = amp and scaler.get_scale() < scale_before
    if not torch.isfinite(grad_norm) and not overflow:
        raise RuntimeError("Non-finite gradient without an AMP overflow")
    return not overflow


def _run_successful_updates(
    step: Callable[[], bool],
    updates: int,
    *,
    max_overflow_retries: int = MAX_OVERFLOW_RETRIES,
) -> tuple[int, int]:
    """Run exactly ``updates`` committed steps, retrying bounded AMP overflows."""
    if updates < 0:
        raise ValueError("updates must be non-negative")
    if max_overflow_retries < 0:
        raise ValueError("max_overflow_retries must be non-negative")
    committed = attempted = overflows = 0
    while committed < updates:
        attempted += 1
        if step():
            committed += 1
            continue
        overflows += 1
        if overflows > max_overflow_retries:
            raise RuntimeError(
                f"Exceeded {max_overflow_retries} AMP overflow retries while measuring {updates} updates"
            )
    return attempted, overflows


def _cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU measurement requires CUDA")
    return torch.device("cuda", torch.cuda.current_device())


def _cuda_metadata(device: torch.device, *, amp: bool) -> dict:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    capability = torch.cuda.get_device_capability(device)
    properties = torch.cuda.get_device_properties(device)
    allocator_backend = torch.cuda.memory.get_allocator_backend()
    device_uuid = getattr(properties, "uuid", None)
    return {
        "device": str(device),
        "device_index": device.index,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_uuid": str(device_uuid) if device_uuid is not None else None,
        "compute_capability": list(capability),
        "free_memory_bytes": int(free_bytes),
        "total_memory_bytes": int(total_bytes),
        "allocator_backend": allocator_backend,
        "torch_version": torch.__version__,
        "pytorch_version": torch.__version__,
        "backend": device.type,
        "device_backend": device.type,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "precision": "mixed_float16" if amp else "float32",
        "autocast_dtype": "float16" if amp else None,
        "amp_fp16": amp,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn_allowed": bool(torch.backends.cudnn.allow_tf32),
    }


def _elapsed_and_throughput(started: float, measured_tokens: int) -> tuple[float, float]:
    seconds = time.perf_counter() - started
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError(f"Invalid elapsed benchmark time: {seconds!r}")
    tokens_per_second = measured_tokens / seconds
    if not math.isfinite(tokens_per_second):
        raise RuntimeError(f"Invalid measured throughput: {tokens_per_second!r}")
    return seconds, tokens_per_second


def probe_once(cfg: RunConfig, physical_batch: int, updates: int = 20) -> dict:
    if physical_batch <= 0:
        raise ValueError("physical_batch must be positive")
    if updates <= 0:
        raise ValueError("updates must be positive")
    device = _cuda_device()
    attention_backend = prepare_attention_backend(
        cfg.attention_backend, cfg.model, device,
        batch_size=physical_batch, amp=cfg.use_amp,
    )
    device_metadata = _cuda_metadata(device, amp=cfg.use_amp)
    trial = replace(cfg, physical_batch=physical_batch, effective_batch=physical_batch)
    torch.manual_seed(trial.seed)
    model = TinyGPT(trial.model).to(device).train()
    optimizer = make_optimizer(model, trial)
    scaler = torch.amp.GradScaler("cuda", enabled=trial.use_amp)
    x = torch.randint(
        2, trial.model.vocab_size, (physical_batch, trial.model.block_size), device=device
    )
    y = torch.randint(
        2, trial.model.vocab_size, (physical_batch, trial.model.block_size), device=device
    )
    step = lambda: _one_step(model, optimizer, scaler, x, y, trial.use_amp)
    warmup_attempts, warmup_overflows = _run_successful_updates(step, 1)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    attempted_updates, measured_overflows = _run_successful_updates(step, updates)
    torch.cuda.synchronize(device)
    measured_tokens = updates * x.numel()
    seconds, tokens_per_second = _elapsed_and_throughput(started, measured_tokens)
    free_after_bytes, total_after_bytes = torch.cuda.mem_get_info(device)
    return {
        "status": "PASS",
        "physical_batch": physical_batch,
        "updates": updates,
        "successful_updates": updates,
        "attempted_updates": attempted_updates,
        "overflow_retries": measured_overflows,
        "warmup_updates": 1,
        "warmup_attempts": warmup_attempts,
        "warmup_overflow_retries": warmup_overflows,
        "measured_tokens": measured_tokens,
        "model": trial.model.integrator,
        "backward_mode": trial.model.backward_mode,
        "step_size": trial.model.step_size,
        "blend": trial.model.blend,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "free_memory_after_bytes": int(free_after_bytes),
        "total_memory_after_bytes": int(total_after_bytes),
        "seconds": seconds,
        "tokens_per_second": tokens_per_second,
        "parameter_count": count_unique_parameters(model),
        "attention_backend": attention_backend,
        **device_metadata,
    }


def _benchmark_repetition(
    cfg: RunConfig,
    device: torch.device,
    repetition: int,
    warmup: int,
    measured: int,
) -> dict:
    free_before_bytes, total_bytes = torch.cuda.mem_get_info(device)
    torch.manual_seed(cfg.seed + repetition)
    model = TinyGPT(cfg.model).to(device).train()
    optimizer = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)
    x = torch.randint(
        2, cfg.model.vocab_size, (cfg.effective_batch, cfg.model.block_size), device=device
    )
    y = torch.randint(
        2, cfg.model.vocab_size, (cfg.effective_batch, cfg.model.block_size), device=device
    )
    step = lambda: _one_step(model, optimizer, scaler, x, y, cfg.use_amp, cfg.physical_batch)
    warmup_attempts, warmup_overflows = _run_successful_updates(step, warmup)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    attempted_updates, measured_overflows = _run_successful_updates(step, measured)
    torch.cuda.synchronize(device)
    measured_tokens = measured * x.numel()
    seconds, tokens_per_second = _elapsed_and_throughput(started, measured_tokens)
    free_after_bytes, total_after_bytes = torch.cuda.mem_get_info(device)
    return {
        "repetition": repetition,
        "seconds": seconds,
        "tokens_per_second": tokens_per_second,
        "measured_tokens": measured_tokens,
        "successful_updates": measured,
        "attempted_updates": attempted_updates,
        "overflow_retries": measured_overflows,
        "warmup_updates": warmup,
        "warmup_attempts": warmup_attempts,
        "warmup_overflow_retries": warmup_overflows,
        "microbatches_per_update": math.ceil(cfg.effective_batch / cfg.physical_batch),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "free_memory_before_bytes": int(free_before_bytes),
        "free_memory_after_bytes": int(free_after_bytes),
        "total_memory_bytes": int(total_bytes),
        "total_memory_after_bytes": int(total_after_bytes),
    }


def benchmark(
    cfg: RunConfig,
    repetitions: int = DEFAULT_BENCHMARK_REPETITIONS,
    warmup: int = DEFAULT_BENCHMARK_WARMUP_UPDATES,
    measured: int = DEFAULT_BENCHMARK_MEASURED_UPDATES,
) -> dict:
    if repetitions <= 0 or warmup < 0 or measured <= 0:
        raise ValueError("Require repetitions > 0, warmup >= 0, and measured > 0")
    if not 0 < cfg.physical_batch <= cfg.effective_batch:
        raise ValueError("Require 0 < physical_batch <= effective_batch")
    device = _cuda_device()
    attention_backend = prepare_attention_backend(
        cfg.attention_backend, cfg.model, device,
        batch_size=cfg.physical_batch, amp=cfg.use_amp,
    )
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    device_metadata = _cuda_metadata(device, amp=cfg.use_amp)
    raw = []
    for repetition in range(repetitions):
        if repetition:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
        raw.append(_benchmark_repetition(cfg, device, repetition, warmup, measured))
    speeds = [record["tokens_per_second"] for record in raw]
    return {
        "status": "COMPLETE",
        "warmup_updates": warmup,
        "measured_updates": measured,
        "repetition_count": repetitions,
        "repetitions": raw,
        "total_measured_tokens": sum(record["measured_tokens"] for record in raw),
        "total_attempted_updates": sum(record["attempted_updates"] for record in raw),
        "total_overflow_retries": sum(record["overflow_retries"] for record in raw),
        "median_tokens_per_second": statistics.median(speeds),
        "min_tokens_per_second": min(speeds),
        "max_tokens_per_second": max(speeds),
        "physical_batch": cfg.physical_batch,
        "effective_batch": cfg.effective_batch,
        "microbatches_per_update": math.ceil(cfg.effective_batch / cfg.physical_batch),
        "sequence_length": cfg.model.block_size,
        "compile": False,
        "attention_backend": attention_backend,
        **device_metadata,
    }
