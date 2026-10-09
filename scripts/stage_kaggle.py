"""Stage private, versioned Kaggle input assets from locally verified artifacts."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from revllm.data import load_manifest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "staging" / "kaggle-assets"


def main() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(ROOT / "data" / "manifest.json")
    for name in ("train.bin", "dev.bin", "holdout.bin", "tokenizer.json", "manifest.json"):
        shutil.copy2(ROOT / "data" / name, STAGE / name)
    with zipfile.ZipFile(STAGE / "revllm_source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted((ROOT / "src" / "revllm").glob("*.py")):
            archive.write(file, f"revllm/{file.name}")
    metadata = {
        "title": "Session 13 reversible LLM private research assets",
        "id": "KAGGLE_USER/revllm-session13-assets",
        "licenses": [{"name": "other"}],
        "description": "Private reproducibility artifacts for the Session 13 reversible language-model study; TinyStories CDLA-Sharing-1.0.",
    }
    (STAGE / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"stage": str(STAGE), "tape_hashes": {k: v["sha256"] for k, v in manifest["tapes"].items()}}, indent=2))


if __name__ == "__main__":
    main()
