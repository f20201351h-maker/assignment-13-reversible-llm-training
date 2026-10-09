"""Numerical gates for stored versus reconstructive backward."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from .model import ModelConfig, TinyGPT


def reconstruction_probe(model: TinyGPT, *, amp: bool = False, device: str = "cpu") -> dict:
    """Measure end-to-end inverse drift across every trained reversible block."""
    stack = model.stack
    if stack is None:
        raise ValueError("Conventional residual stack has no explicit inverse")
    x = torch.randn(2, min(16, model.cfg.block_size), model.cfg.n_embd, device=device,
                    dtype=next(model.parameters()).dtype)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        if model.cfg.integrator == "coupled_euler":
            u, v = x, x
            for block in stack.blocks:
                u = u + stack.h * block.attn_update(v)
                v = v + stack.h * block.mlp_update(u)
            for block in reversed(stack.blocks):
                v = v - stack.h * block.mlp_update(u)
                u = u - stack.h * block.attn_update(v)
            error = max(float((u - x).norm() / x.norm()), float((v - x).norm() / x.norm()))
        else:
            prev, cur = x, x + 0.5 * stack.h * stack.blocks[0].force(x)
            for block in stack.blocks[1:]:
                prev, cur = cur, stack.a * prev + (1 - stack.a) * cur + 2 * stack.h * block.force(cur)
            for block in reversed(stack.blocks[1:]):
                older = (cur - (1 - stack.a) * prev - 2 * stack.h * block.force(prev)) / stack.a
                prev, cur = older, prev
            error = float((prev - x).norm() / x.norm())
    threshold = 1e-3 if amp else (1e-9 if x.dtype == torch.float64 else 1e-5)
    return {"relative_reconstruction_error": error, "threshold": threshold,
            "status": "PASS" if error <= threshold else "FAIL"}


def gradient_probe(integrator: str, h: float, blend: float = 1.0,
                   *, amp: bool = False, device: str = "cpu", depth: int = 4,
                   dtype: torch.dtype = torch.float64) -> dict:
    cfg = ModelConfig(vocab_size=80, block_size=16, n_layer=depth, n_head=3, n_embd=48,
                      integrator=integrator, backward_mode="stored", step_size=h, blend=blend)
    torch.manual_seed(314159)
    stored = TinyGPT(cfg).to(device)
    reconstructed = TinyGPT(replace(cfg, backward_mode="reconstructed")).to(device)
    reconstructed.load_state_dict(stored.state_dict())
    if device == "cpu":
        stored.to(dtype=dtype)
        reconstructed.to(dtype=dtype)
    tokens = torch.randint(2, cfg.vocab_size, (2, cfg.block_size), device=device)
    targets = torch.randint(2, cfg.vocab_size, tokens.shape, device=device)
    results = []
    for model in (stored, reconstructed):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = model(tokens, targets) / targets.numel()
        loss.backward()
        results.append((float(loss.detach()), {name: p.grad.detach().float().clone()
                                      for name, p in model.named_parameters()}))
    numerator = sum((results[0][1][k] - results[1][1][k]).square().sum()
                    for k in results[0][1])
    denominator = sum(results[0][1][k].square().sum() for k in results[0][1])
    global_error = float(torch.sqrt(numerator / denominator))
    per_parameter = {k: float((results[0][1][k] - results[1][1][k]).norm() /
                              results[0][1][k].norm().clamp_min(1e-12))
                     for k in results[0][1]}
    gate = 1e-2 if amp else (1e-7 if dtype == torch.float64 and device == "cpu" else 1e-4)
    return {
        "integrator": integrator, "step_size": h, "blend": blend,
        "device": device, "amp_fp16": amp, "depth": depth,
        "dtype": "fp16_autocast" if amp else str(dtype),
        "global_relative_gradient_error": global_error,
        "loss_difference": abs(results[0][0] - results[1][0]),
        "max_per_parameter_relative_error": max(per_parameter.values()),
        "per_parameter_relative_error": per_parameter,
        "threshold": gate, "status": "PASS" if global_error <= gate else "FAIL",
    }


def run_gpu_gates(out: Path) -> list[dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU correctness gate requires CUDA")
    cases = [("midpoint", 0.25, 1.0), ("midpoint", 0.5, 1.0),
             ("midpoint", 1.0, 1.0), ("coupled_euler", 0.25, 1.0),
             ("coupled_euler", 0.5, 1.0), ("coupled_euler", 1.0, 1.0),
             ("blended_midpoint", 0.25, 0.5)]
    result = [gradient_probe(*case, amp=True, device="cuda") for case in cases]
    for record in result:
        cfg = ModelConfig(vocab_size=80, block_size=16, n_layer=9, n_head=3, n_embd=48,
                          integrator=record["integrator"], backward_mode="reconstructed",
                          step_size=record["step_size"], blend=record["blend"])
        torch.manual_seed(314159)
        model = TinyGPT(cfg).cuda()
        record["reconstruction"] = reconstruction_probe(model, amp=True, device="cuda")
        if record["reconstruction"]["status"] != "PASS":
            record["status"] = "FAIL"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if any(case["status"] != "PASS" for case in result):
        raise RuntimeError(f"Mixed-precision gradient gate failed; inspect {out}")
    return result
