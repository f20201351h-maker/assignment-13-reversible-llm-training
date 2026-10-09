"""Evaluate a saved run checkpoint against locally regenerated public data."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .backend import prepare_attention_backend
from .data import TokenTape, digest_file, load_manifest
from .model import ModelConfig, TinyGPT, count_unique_parameters
from .train import evaluate


def evaluate_checkpoint(config_path: Path, checkpoint_path: Path, data_dir: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = load_manifest(data_dir / "manifest.json")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state["config"] != config:
        raise RuntimeError("Checkpoint and frozen run configuration differ")
    if state["data_sha256"] != manifest["tapes"]["train"]["sha256"]:
        raise RuntimeError("Checkpoint was trained on a different token tape")
    model_config = ModelConfig(**config["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config.get("use_amp", True)) and device.type == "cuda"
    attention_backend = prepare_attention_backend(
        config.get("attention_backend", "auto"), model_config, device,
        batch_size=min(8, config["physical_batch"]), amp=use_amp,
        allow_cpu_efficient_fallback=True,
    )
    model = TinyGPT(model_config)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    losses = {}
    for name, targets in (("dev", manifest["tapes"]["dev"]["target_count"]),
                          ("holdout", manifest["tapes"]["holdout"]["target_count"]),
                          ("train_subset", min(131_072, config["target_tokens"]))):
        tape_name = "train" if name == "train_subset" else name
        tape = TokenTape(data_dir / f"{tape_name}.bin", targets, model.cfg.block_size,
                         manifest["pad_id"])
        value = evaluate(model, tape, device, min(8, config["physical_batch"]), use_amp)
        losses[name] = {"loss_nats": value, "perplexity": math.exp(value), "targets": targets}
    return {
        "run_id": config["run_id"], "checkpoint_sha256": digest_file(checkpoint_path),
        "data_sha256": manifest["tapes"]["train"]["sha256"],
        "parameter_count": count_unique_parameters(model),
        "committed_targets_at_checkpoint": state["committed"],
        "device": str(device), "amp_fp16": use_amp,
        "attention_backend": attention_backend, "evaluation": losses,
    }
