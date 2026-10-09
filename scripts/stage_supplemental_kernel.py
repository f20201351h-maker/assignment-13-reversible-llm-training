"""Stage preregistered extra formulation, checkpointing, and third-seed runs."""

from __future__ import annotations

import json
from pathlib import Path

from stage_full_kernel import ENTRY

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "staging" / "kaggle-supplemental"
CONFIGS = ROOT / "configs" / "supplemental"


def main() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    CONFIGS.mkdir(parents=True, exist_ok=True)
    conditions = [
        ("F-midpoint-1337", "midpoint", "reconstructed", 110, 110, 1337),
        ("G-checkpointed-1337", "conventional", "checkpointed", 110, 110, 1337),
        ("A-3141", "conventional", "stored", 110, 110, 3141),
        ("B-3141", "coupled_euler", "reconstructed", 110, 110, 3141),
        ("C-3141", "coupled_euler", "reconstructed", 192, 192, 3141),
    ]
    names = []
    run_configs = {}
    for name, integrator, backward, physical, effective, seed in conditions:
        cfg = {
            "run_id": name,
            "model": {"integrator": integrator, "backward_mode": backward,
                      "step_size": 0.5, "blend": 1.0},
            "data_dir": "/kaggle/input/datasets/KAGGLE_USER/revllm-session13-assets",
            "out_dir": f"/kaggle/working/runs/{name}",
            "physical_batch": physical, "effective_batch": effective,
            "seed": seed, "target_tokens": 50_000_000,
            "warmup_tokens": 1_000_000, "use_amp": True,
            "evaluate_holdout": True,
            "eval_every_tokens": 5_000_000, "checkpoint_every_tokens": 5_000_000,
        }
        (CONFIGS / f"{name}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        names.append(name)
        run_configs[name] = cfg
    entry = ENTRY.replace("__RUN_NAMES__", repr(names)).replace("__RUN_CONFIGS__", repr(run_configs))
    (STAGE / "supplemental.py").write_text(entry, encoding="utf-8")
    metadata = {
        "id": "KAGGLE_USER/reversible-llm-session-13-supplemental-runs",
        "title": "Reversible LLM Session 13 supplemental runs",
        "code_file": "supplemental.py", "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_gpu": "true", "enable_internet": "false",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["KAGGLE_USER/revllm-session13-assets"],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    (STAGE / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"stage": str(STAGE), "run_ids": names}, indent=2))


if __name__ == "__main__":
    main()
