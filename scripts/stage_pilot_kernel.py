"""Create a private Kaggle pilot script kernel without exposing course material."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "staging" / "kaggle-pilots"


def main() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "scripts" / "kaggle_job.py", STAGE / "pilot.py")
    meta = {
        "id": "KAGGLE_USER/reversible-llm-session-13-pilots",
        "title": "Reversible LLM Session 13 pilots",
        "code_file": "pilot.py", "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_gpu": "true", "enable_internet": "false",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["KAGGLE_USER/revllm-session13-assets"],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    (STAGE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(STAGE)


if __name__ == "__main__":
    main()
