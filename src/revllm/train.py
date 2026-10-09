"""Training, accounting, evaluation, and atomic checkpoint/resume."""

from __future__ import annotations

import json
import hashlib
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import TokenTape, digest_file, load_manifest
from .model import ModelConfig, TinyGPT, count_unique_parameters
from .backend import ATTENTION_BACKENDS, prepare_attention_backend


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    model: ModelConfig
    data_dir: Path
    out_dir: Path
    physical_batch: int
    effective_batch: int
    seed: int = 1337
    target_tokens: int = 50_000_000
    warmup_tokens: int = 1_000_000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    eval_every_tokens: int = 5_000_000
    checkpoint_every_tokens: int = 5_000_000
    use_amp: bool = True
    evaluate_holdout: bool = False
    attention_backend: str = "auto"

    def validate(self) -> None:
        self.model.validate()
        if not (0 < self.physical_batch <= self.effective_batch):
            raise ValueError("Require 0 < physical_batch <= effective_batch")
        if self.target_tokens <= 0 or self.learning_rate <= 0:
            raise ValueError("Token budget and learning rate must be positive")
        if self.attention_backend not in ATTENTION_BACKENDS:
            choices = ", ".join(sorted(ATTENTION_BACKENDS))
            raise ValueError(f"attention_backend must be one of: {choices}")
        if self.use_amp and not torch.cuda.is_available():
            raise RuntimeError("FP16 autocast requires CUDA; use --no-amp on CPU")


def _jsonable(cfg: RunConfig) -> dict:
    d = asdict(cfg)
    # Keep the historical default byte-for-byte compatible with frozen configs
    # and checkpoints created before attention_backend was introduced.
    if d["attention_backend"] == "auto":
        d.pop("attention_backend")
    d["data_dir"], d["out_dir"] = str(cfg.data_dir), str(cfg.out_dir)
    return json.loads(json.dumps(d))


def learning_rate(tokens_committed: int, cfg: RunConfig) -> float:
    progress = max(1, tokens_committed)
    if progress < cfg.warmup_tokens:
        return cfg.learning_rate * progress / cfg.warmup_tokens
    fraction = min(1.0, (progress - cfg.warmup_tokens) / max(1, cfg.target_tokens - cfg.warmup_tokens))
    return cfg.min_learning_rate + 0.5 * (cfg.learning_rate - cfg.min_learning_rate) * (1 + math.cos(math.pi * fraction))


def make_optimizer(model: TinyGPT, cfg: RunConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim == 1 or "embedding" in name else decay).append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.learning_rate, betas=cfg.adam_betas, eps=cfg.adam_eps,
        fused=torch.cuda.is_available(),
    )


def _save_checkpoint(path: Path, state: dict) -> None:
    temp = path.with_suffix(path.suffix + ".pending")
    torch.save(state, temp)
    os.replace(temp, path)


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    # CPU staging avoids retaining a second full model/optimizer copy on CUDA.
    return torch.load(path, map_location="cpu", weights_only=False)


def evaluate(model: TinyGPT, tape: TokenTape, device: torch.device, batch_size: int = 8, amp: bool = False) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    cursor = 0
    with torch.no_grad():
        while cursor < tape.target_count:
            x, y, count = tape.batch(cursor, batch_size)
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss_sum = model(x, y)
            total += float(loss_sum)
            cursor += count
    model.train(was_training)
    return total / tape.target_count


def _write_jsonl(path: Path, item: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, allow_nan=False) + "\n")


def _log_tail(path: Path, offset: int) -> bytes:
    """Read bytes written after the checkpoint's durable log watermark."""
    if not path.exists():
        if offset:
            raise RuntimeError(f"Checkpoint expects missing log: {path}")
        return b""
    size = path.stat().st_size
    if size < offset:
        raise RuntimeError(f"Log shorter than checkpoint watermark: {path}")
    if size == offset:
        return b""
    with path.open("rb") as stream:
        stream.seek(offset)
        return stream.read()


def _tail_events(discarded_bytes: bytes) -> list[dict]:
    events = []
    for line in discarded_bytes.decode("utf-8", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return events


def _atomic_json(path: Path, item: dict) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(item, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def _archive_and_truncate(path: Path, offset: int, tail: bytes, archive_dir: Path) -> None:
    if not tail:
        return
    archive_dir.mkdir(exist_ok=True)
    digest = hashlib.sha256(tail).hexdigest()
    archive = archive_dir / f"{path.stem}-{digest}.jsonl"
    if not archive.exists():
        pending = archive.with_suffix(".pending")
        with pending.open("wb") as stream:
            stream.write(tail)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, archive)
    with path.open("r+b") as stream:
        stream.truncate(offset)


def train(cfg: RunConfig, *, resume: bool = False, max_updates: int | None = None) -> dict:
    cfg.validate()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(cfg.data_dir / "manifest.json")
    if cfg.target_tokens > manifest["tapes"]["train"]["target_count"]:
        raise ValueError("Requested more training targets than the prepared tape contains")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention_backend = prepare_attention_backend(
        cfg.attention_backend, cfg.model, device,
        batch_size=cfg.physical_batch, amp=cfg.use_amp,
    )
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    model = TinyGPT(cfg.model).to(device)
    if count_unique_parameters(model) != 19_969_152 and cfg.model.vocab_size == 10_000 and cfg.model.n_embd == 384:
        raise RuntimeError("Principal model parameter count mismatch")
    optimizer = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)
    tape = TokenTape(cfg.data_dir / "train.bin", cfg.target_tokens, cfg.model.block_size, manifest["pad_id"])
    config_path = cfg.out_dir / "run_config.json"
    frozen_config = _jsonable(cfg)
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != frozen_config:
            raise RuntimeError("Existing run configuration differs; use a new run ID")
    else:
        config_path.write_text(json.dumps(frozen_config, indent=2), encoding="utf-8")
    checkpoint_path = cfg.out_dir / "latest.pt"
    metrics_path = cfg.out_dir / "metrics.jsonl"
    eval_path = cfg.out_dir / "eval.jsonl"
    committed = attempted = updates = retries = 0
    consecutive_retries = 0
    elapsed_training = trailing_loss_sum = prior_job_seconds = last_update_loss = 0.0
    peak_train_allocated = peak_train_reserved = 0
    last_eval_at = last_checkpoint_at = 0
    if resume:
        state = _load_checkpoint(checkpoint_path, device)
        if state["config"] != frozen_config or state["data_sha256"] != manifest["tapes"]["train"]["sha256"]:
            raise RuntimeError("Checkpoint configuration or dataset hash mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        committed, attempted, updates, retries = (state[k] for k in ("committed", "attempted", "updates", "retries"))
        elapsed_training = state["elapsed_training"]
        trailing_loss_sum = state["trailing_loss_sum"]
        last_eval_at = state["last_eval_at"]
        last_checkpoint_at = state["last_checkpoint_at"]
        last_update_loss = state["last_update_loss"]
        prior_job_seconds = state["job_seconds"]
        peak_train_allocated = state.get("peak_train_allocated", 0)
        peak_train_reserved = state.get("peak_train_reserved", 0)
        if "metrics_offset_bytes" not in state or "eval_offset_bytes" not in state:
            raise RuntimeError("Checkpoint predates transactional log watermarks; cannot safely resume")
        checkpoint_id = digest_file(checkpoint_path)
        wal_path = cfg.out_dir / "recovery_wal.json"
        wal = json.loads(wal_path.read_text(encoding="utf-8")) if wal_path.exists() else None
        if wal is None or wal.get("checkpoint_sha256") != checkpoint_id:
            wal = {"checkpoint_sha256": checkpoint_id, "rolled_back_attempted_targets": 0,
                   "rolled_back_retries": 0, "rolled_back_training_seconds": 0.0,
                   "last_tail_sha256": None}
        metrics_tail = _log_tail(metrics_path, state["metrics_offset_bytes"])
        eval_tail = _log_tail(eval_path, state["eval_offset_bytes"])
        tail_digest = hashlib.sha256(metrics_tail).hexdigest() if metrics_tail else None
        if metrics_tail and tail_digest != wal["last_tail_sha256"]:
            rolled_back = _tail_events(metrics_tail)
            base_attempted = attempted + wal["rolled_back_attempted_targets"]
            last_attempted = max((record.get("attempted_targets", base_attempted)
                                  for record in rolled_back), default=base_attempted)
            wal["rolled_back_attempted_targets"] += max(0, last_attempted - base_attempted)
            wal["rolled_back_retries"] += sum(record.get("status") == "AMP_OVERFLOW_REPLAY"
                                              for record in rolled_back)
            wal["rolled_back_training_seconds"] += sum(float(record.get("step_seconds", 0.0))
                                                        for record in rolled_back)
            wal["last_tail_sha256"] = tail_digest
            _atomic_json(wal_path, wal)
        archive_dir = cfg.out_dir / "rollbacks"
        _archive_and_truncate(metrics_path, state["metrics_offset_bytes"], metrics_tail, archive_dir)
        _archive_and_truncate(eval_path, state["eval_offset_bytes"], eval_tail, archive_dir)
        attempted += wal["rolled_back_attempted_targets"]
        retries += wal["rolled_back_retries"]
        elapsed_training += wal["rolled_back_training_seconds"]
        prior_job_seconds += wal["rolled_back_training_seconds"]
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state
        if device.type == "cuda":
            torch.cuda.empty_cache()
    elif checkpoint_path.exists():
        raise RuntimeError("Checkpoint exists. Resume explicitly or use a new output directory")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    started = time.perf_counter()
    while committed < cfg.target_tokens:
        update_start = time.perf_counter()
        planned = min(cfg.effective_batch * cfg.model.block_size, cfg.target_tokens - committed)
        step_lr = learning_rate(committed + planned, cfg)
        for group in optimizer.param_groups:
            group["lr"] = step_lr
        optimizer.zero_grad(set_to_none=True)
        offset = 0
        loss_total_device = torch.zeros((), device=device)
        trailing_pending_device = torch.zeros((), device=device)
        separator_targets_device = torch.zeros((), device=device, dtype=torch.int64)
        losses_finite_device = torch.ones((), device=device, dtype=torch.bool)
        micro_count = 0
        padding_slots = 0
        while offset < planned:
            local_sequences = min(cfg.physical_batch, math.ceil((planned - offset) / cfg.model.block_size))
            x, y, count = tape.batch(committed + offset, local_sequences, planned - offset)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.use_amp):
                losses = model(x, y, return_token_losses=True)
                loss_sum = losses.sum()
            losses_finite_device &= torch.isfinite(loss_sum.detach())
            scorer = losses.detach().flatten()[:count]
            separator_targets_device += ((y != -100) & (y == manifest["eos_id"])).sum()
            padding_slots += y.numel() - count
            loss_total_device += scorer.sum()
            tail_start = max(0, cfg.target_tokens - 1_000_000 - (committed + offset))
            if tail_start < count:
                trailing_pending_device += scorer[tail_start:count].sum()
            scaler.scale(loss_sum / planned).backward()
            offset += count
            micro_count += 1
        if not bool(losses_finite_device):
            raise RuntimeError(f"Non-finite loss at committed target {committed}")
        attempted += planned
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=False)
        if not torch.isfinite(grad_norm) and not cfg.use_amp:
            raise RuntimeError(f"Non-finite gradient at committed target {committed}")
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        overflow = scaler.get_scale() < scale_before
        if not torch.isfinite(grad_norm) and not overflow:
            raise RuntimeError(f"Non-finite gradient without AMP replay at committed target {committed}")
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_train_allocated = max(peak_train_allocated, torch.cuda.max_memory_allocated())
            peak_train_reserved = max(peak_train_reserved, torch.cuda.max_memory_reserved())
        step_seconds = time.perf_counter() - update_start
        elapsed_training += step_seconds
        if overflow:
            retries += 1
            consecutive_retries += 1
            _write_jsonl(metrics_path, {
                "status": "AMP_OVERFLOW_REPLAY", "attempted_targets": attempted,
                "committed_targets": committed, "retries": retries, "step_seconds": step_seconds,
            })
            if consecutive_retries > 10:
                raise RuntimeError("Too many consecutive AMP overflow retries")
            continue
        consecutive_retries = 0
        loss_total = float(loss_total_device)
        trailing_pending = float(trailing_pending_device)
        separator_targets = int(separator_targets_device)
        committed += planned
        updates += 1
        trailing_loss_sum += trailing_pending
        record = {
            "status": "COMMITTED", "update": updates, "committed_targets": committed,
            "attempted_targets": attempted, "valid_targets": planned,
            "loss_nats": loss_total / planned, "lr": step_lr,
            "physical_batch": cfg.physical_batch, "effective_batch": cfg.effective_batch,
            "microbatches": micro_count, "grad_norm_before_clip": float(grad_norm),
            "separator_targets": separator_targets, "padding_slots": padding_slots,
            "step_seconds": step_seconds, "training_seconds": elapsed_training,
            "tokens_per_second": planned / step_seconds,
            "scaler_scale": scaler.get_scale(),
        }
        last_update_loss = record["loss_nats"]
        if device.type == "cuda":
            record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        _write_jsonl(metrics_path, record)
        if committed - last_eval_at >= cfg.eval_every_tokens or committed == cfg.target_tokens:
            dev = TokenTape(cfg.data_dir / "dev.bin", manifest["tapes"]["dev"]["target_count"], cfg.model.block_size, manifest["pad_id"])
            dev_loss = evaluate(model, dev, device, min(8, cfg.physical_batch), cfg.use_amp)
            _write_jsonl(eval_path, {
                "committed_targets": committed, "dev_loss_nats": dev_loss,
                "dev_perplexity": math.exp(dev_loss),
            })
            last_eval_at = committed
        stopping_early = max_updates is not None and updates >= max_updates
        if committed - last_checkpoint_at >= cfg.checkpoint_every_tokens or committed == cfg.target_tokens or stopping_early:
            state = {
                "config": frozen_config, "data_sha256": manifest["tapes"]["train"]["sha256"],
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "committed": committed, "attempted": attempted,
                "updates": updates, "retries": retries, "elapsed_training": elapsed_training,
                "trailing_loss_sum": trailing_loss_sum, "last_eval_at": last_eval_at,
                "last_checkpoint_at": committed,
                "last_update_loss": last_update_loss,
                "job_seconds": prior_job_seconds + time.perf_counter() - started,
                "metrics_offset_bytes": metrics_path.stat().st_size,
                "eval_offset_bytes": eval_path.stat().st_size if eval_path.exists() else 0,
                "peak_train_allocated": peak_train_allocated,
                "peak_train_reserved": peak_train_reserved,
                "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            }
            _save_checkpoint(checkpoint_path, state)
            last_checkpoint_at = committed
        if stopping_early:
            break

    if committed < cfg.target_tokens:
        partial = {
            "run_id": cfg.run_id, "status": "INCOMPLETE", "committed_targets": committed,
            "attempted_targets": attempted, "updates": updates, "checkpoint_sha256": digest_file(checkpoint_path),
            "attention_backend": attention_backend,
        }
        (cfg.out_dir / "partial.json").write_text(json.dumps(partial, indent=2), encoding="utf-8")
        return partial

    holdout_loss = None
    if cfg.evaluate_holdout:
        holdout = TokenTape(cfg.data_dir / "holdout.bin", manifest["tapes"]["holdout"]["target_count"], cfg.model.block_size, manifest["pad_id"])
        holdout_loss = evaluate(model, holdout, device, min(8, cfg.physical_batch), cfg.use_amp)
    train_eval = TokenTape(cfg.data_dir / "train.bin", min(131_072, cfg.target_tokens), cfg.model.block_size, manifest["pad_id"])
    train_eval_loss = evaluate(model, train_eval, device, min(8, cfg.physical_batch), cfg.use_amp)
    final = {
        "run_id": cfg.run_id, "status": "COMPLETE", "committed_targets": committed,
        "attempted_targets": attempted, "updates": updates, "retries": retries,
        "parameter_count": count_unique_parameters(model),
        "final_update_loss_nats": last_update_loss,
        "last_1m_training_loss_nats": trailing_loss_sum / min(1_000_000, cfg.target_tokens),
        "fixed_training_subset_loss_nats": train_eval_loss,
        "holdout_loss_nats": holdout_loss,
        "holdout_perplexity": math.exp(holdout_loss) if holdout_loss is not None else None,
        "training_seconds": elapsed_training,
        "job_seconds": prior_job_seconds + time.perf_counter() - started,
        "training_tokens_per_second": committed / elapsed_training,
        "physical_batch": cfg.physical_batch, "global_batch": cfg.physical_batch,
        "effective_batch": cfg.effective_batch, "sequence_length": cfg.model.block_size,
        "world_size": 1, "device": str(device),
        "gpu_name": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "torch_version": torch.__version__, "amp_fp16": cfg.use_amp,
        "attention_backend": attention_backend,
        "compile": False, "seed": cfg.seed,
        "peak_allocated_bytes": peak_train_allocated if device.type == "cuda" else None,
        "peak_reserved_bytes": peak_train_reserved if device.type == "cuda" else None,
        "checkpoint_sha256": digest_file(checkpoint_path),
        "data_sha256": manifest["tapes"]["train"]["sha256"],
        "tokenizer_sha256": manifest["tokenizer_sha256"],
    }
    (cfg.out_dir / "final.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    (cfg.out_dir / "partial.json").unlink(missing_ok=True)
    return final
