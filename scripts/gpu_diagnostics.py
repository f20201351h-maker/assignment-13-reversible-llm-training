"""Exploratory post-training precision and actual SDPA-operator diagnostics."""
from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path

import torch

from revllm.correctness import reconstruction_probe
from revllm.backend import configure_attention_backend
from revllm.data import digest_file
from revllm.model import CausalAttention, ModelConfig, TinyGPT
from revllm.profile import _cuda_metadata


def attention_operators(batch: int) -> dict:
    torch.manual_seed(993)
    module = CausalAttention(ModelConfig()).cuda()
    x = torch.randn(batch, 512, 384, device="cuda", requires_grad=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        with torch.autocast("cuda", dtype=torch.float16):
            loss = module(x).float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
    names = sorted({event.key for event in profile.key_averages()
                    if any(word in event.key for word in ("scaled_dot", "efficient_attention", "flash_attention"))})
    if not any("backward" in name for name in names):
        raise RuntimeError(f"Attention backward operator was not captured: {names}")
    return {"physical_batch": batch, "sequence_length": 512, "heads": 6,
            "head_dimension": 64, "operators": names,
            "policy": "automatic; historical campaign did not explicitly force a backend",
            "flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_enabled": torch.backends.cuda.math_sdp_enabled()}


def trained_probe(checkpoint: Path) -> dict:
    final = json.loads((checkpoint.parent / "final.json").read_text())
    digest = digest_file(checkpoint)
    if digest != final["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**state["config"]["model"])
    if cfg.integrator == "conventional":
        return {"run_id": final["run_id"], "status": "NOT_REVERSIBLE", "checkpoint_sha256": digest}
    backend = configure_attention_backend(state["config"].get("attention_backend", "auto"), torch.device("cuda", 0))
    torch.manual_seed(993)
    ids = torch.randint(2, cfg.vocab_size, (1, cfg.block_size), device="cuda")
    targets = torch.randint(2, cfg.vocab_size, ids.shape, device="cuda")
    gradients, losses = [], []
    for mode in ("stored", "reconstructed"):
        model = TinyGPT(replace(cfg, backward_mode=mode)).cuda().train()
        model.load_state_dict(state["model"])
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(ids, targets) / targets.numel()
        (loss * 1024).backward()
        gradients.append({name: p.grad.detach().float().cpu() / 1024
                          for name, p in model.named_parameters()})
        losses.append(float(loss.detach()))
        if mode == "reconstructed":
            inverse = reconstruction_probe(model, amp=True, device="cuda")
        del model, loss
        gc.collect()
        torch.cuda.empty_cache()
    per_parameter, numerator, denominator = {}, 0.0, 0.0
    finite = True
    for name, reference in gradients[0].items():
        actual = gradients[1][name]
        finite &= bool(torch.isfinite(reference).all() and torch.isfinite(actual).all())
        diff = (actual - reference).double()
        norm = float(reference.double().norm())
        absolute = float(diff.norm())
        numerator += absolute ** 2
        denominator += norm ** 2
        per_parameter[name] = {"relative_error": absolute / max(norm, 1e-12),
                               "absolute_error_l2": absolute, "reference_norm_l2": norm}
    error = (numerator / max(denominator, 1e-30)) ** 0.5
    passed = finite and error <= 1e-2 and inverse["status"] == "PASS" and losses[0] == losses[1]
    return {"run_id": final["run_id"], "checkpoint_sha256": digest,
            "attention_backend": backend,
            "committed_targets_at_probe": state["committed"], "amp_fp16": True,
            "input_shape": list(ids.shape), "loss_scale": 1024,
            "global_relative_gradient_error": error, "threshold": 1e-2,
            "loss_difference": abs(losses[0] - losses[1]), "finite_gradients": finite,
            "per_parameter": per_parameter, "reconstruction": inverse,
            "status": "PASS" if passed else "FAIL"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-run-ids", nargs="+")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = _cuda_metadata(torch.device("cuda", torch.cuda.current_device()), amp=True)
    operators = []
    for batch in (1, 110, 181, 192):
        operators.append(attention_operators(batch))
        gc.collect()
        torch.cuda.empty_cache()
    (args.out / "attention_operators.json").write_text(json.dumps({"hardware": metadata, "probes": operators}, indent=2))
    results = []
    for checkpoint in sorted(args.input_root.rglob("latest.pt")):
        if checkpoint.parent.parent.name != "runs":
            continue
        result = trained_probe(checkpoint)
        results.append(result)
        (args.out / "trained_mixed_precision.json").write_text(json.dumps(results, indent=2))
        print(f"{result['run_id']}: {result['status']}", flush=True)
    if args.expected_run_ids and ({r["run_id"] for r in results} != set(args.expected_run_ids) or len(results) != len(args.expected_run_ids)):
        raise RuntimeError("Diagnostic checkpoint coverage differs from frozen campaign")
    if not results or any(row["status"] == "FAIL" for row in results):
        raise RuntimeError("Trained precision diagnostic incomplete or failed; retained evidence must be investigated")


if __name__ == "__main__":
    main()
