"""Repeat stored/reconstructed gradient and inverse checks at trained weights."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from revllm.model import ModelConfig, TinyGPT
from revllm.correctness import reconstruction_probe
from revllm.data import digest_file


ROOT = Path(__file__).resolve().parents[1]
PILOTS = ROOT / "experiments" / "kaggle-pilots-v3" / "runs"


def probe(checkpoint: Path) -> dict:
    final_path = checkpoint.parent / "final.json"
    if final_path.exists():
        final = json.loads(final_path.read_text(encoding="utf-8"))
        if digest_file(checkpoint) != final["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint hash disagrees with final summary: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = ModelConfig(**state["config"]["model"])
    if model_config.integrator == "conventional":
        return {"run_id": state["config"]["run_id"], "status": "NOT_REVERSIBLE"}
    stored = TinyGPT(replace(model_config, backward_mode="stored"))
    reconstructed = TinyGPT(replace(model_config, backward_mode="reconstructed"))
    stored.load_state_dict(state["model"])
    reconstructed.load_state_dict(state["model"])
    torch.manual_seed(993)
    ids = torch.randint(2, model_config.vocab_size, (1, 16))
    targets = torch.randint(2, model_config.vocab_size, ids.shape)
    losses = []
    for model in (stored, reconstructed):
        model.train()
        loss = model(ids, targets) / targets.numel()
        loss.backward()
        losses.append((loss.detach(), [p.grad.detach().clone() for p in model.parameters()]))
    sum_difference = sum((a - b).square().sum() for a, b in zip(losses[0][1], losses[1][1]))
    sum_reference = sum(a.square().sum() for a in losses[0][1])
    error = float((sum_difference / sum_reference).sqrt())
    reconstruction = reconstruction_probe(reconstructed)
    result = {
        "run_id": state["config"]["run_id"], "integrator": model_config.integrator,
        "step_size": model_config.step_size, "blend": model_config.blend,
        "committed_targets_at_probe": state["committed"],
        "loss_difference": abs(float(losses[0][0] - losses[1][0])),
        "global_relative_gradient_error_fp32": error,
        "reconstruction": reconstruction,
        "threshold": 1e-4,
        "status": "PASS" if error <= 1e-4 and reconstruction["status"] == "PASS" else "FAIL",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=PILOTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.runs_root.rglob("latest.pt"))
    results = [probe(path) for path in paths]
    output = args.output or args.runs_root.parent / "trained_checkpoint_probes.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoints_found": len(paths), "pass": sum(x["status"] == "PASS" for x in results),
                      "fail": sum(x["status"] == "FAIL" for x in results), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
