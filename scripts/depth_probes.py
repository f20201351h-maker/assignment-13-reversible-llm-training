"""Generate real depth-sweep numerical probes for analysis plots."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from revllm.correctness import gradient_probe, reconstruction_probe
from revllm.model import ModelConfig, TinyGPT


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "depth-sweep" / "probes.json"


def main() -> None:
    records = []
    for kind, h, blend in (("midpoint", 0.5, 1.0), ("coupled_euler", 0.5, 1.0)):
        for depth in (1, 2, 4, 6, 9):
            for dtype in (torch.float64, torch.float32):
                torch.manual_seed(314159)
                cfg = ModelConfig(vocab_size=80, block_size=16, n_layer=depth, n_head=3,
                                  n_embd=48, integrator=kind, backward_mode="reconstructed",
                                  step_size=h, blend=blend)
                model = TinyGPT(cfg).to(dtype=dtype)
                reconstruction = reconstruction_probe(model)
                gradient = gradient_probe(kind, h, blend, depth=depth, dtype=dtype)
                record = {
                    "integrator": kind, "depth": depth, "dtype": str(dtype),
                    "relative_reconstruction_error": reconstruction["relative_reconstruction_error"],
                    "relative_gradient_error": gradient["global_relative_gradient_error"],
                    "reconstruction_threshold": reconstruction["threshold"],
                    "gradient_threshold": gradient["threshold"],
                    "status": "PASS" if reconstruction["status"] == gradient["status"] == "PASS" else "FAIL",
                }
                records.append(record)
                print(record, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
