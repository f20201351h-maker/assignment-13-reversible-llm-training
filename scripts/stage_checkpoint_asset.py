"""Package verified locked-run checkpoints for a private Kaggle input dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_PATH = ROOT / "configs" / "locked_campaign.json"
STAGING = ROOT / "staging" / "kaggle-locked-checkpoints"
TARGETS = 50_000_000
DATASET_ID = "KAGGLE_USER/revllm-session13-locked-checkpoints"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_run(run_id: str, roots: list[str]) -> Path:
    matches = [ROOT / relative_root / run_id for relative_root in roots
               if (ROOT / relative_root / run_id).is_dir()]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one local copy of {run_id}, found {len(matches)}: "
            f"{[str(path.relative_to(ROOT)) for path in matches]}"
        )
    return matches[0]


def main() -> None:
    campaign = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
    run_ids = campaign["run_ids"]
    roots = campaign["local_roots"]
    if len(run_ids) != 8 or len(set(run_ids)) != 8:
        raise RuntimeError(f"Expected exactly eight unique campaign IDs, got {run_ids}")

    entries = []
    sources: dict[str, dict[str, Path]] = {}
    for run_id in run_ids:
        run_dir = locate_run(run_id, roots)
        files = {name: run_dir / name for name in ("latest.pt", "run_config.json", "final.json")}
        missing = [name for name, path in files.items() if not path.is_file()]
        if missing:
            raise RuntimeError(f"{run_id} is missing required artifacts: {missing}")

        final = json.loads(files["final.json"].read_text(encoding="utf-8"))
        config = json.loads(files["run_config.json"].read_text(encoding="utf-8"))
        if final.get("run_id") != run_id or config.get("run_id") != run_id:
            raise RuntimeError(f"Run ID mismatch in artifacts for {run_id}")
        if final.get("committed_targets") != TARGETS:
            raise RuntimeError(
                f"{run_id} committed {final.get('committed_targets')} targets; expected {TARGETS}"
            )
        actual_hash = sha256(files["latest.pt"])
        if final.get("checkpoint_sha256") != actual_hash:
            raise RuntimeError(
                f"Checkpoint hash mismatch for {run_id}: final.json="
                f"{final.get('checkpoint_sha256')}, actual={actual_hash}"
            )

        entries.append({
            "run_id": run_id,
            "source_run_directory": str(run_dir.relative_to(ROOT)).replace("\\", "/"),
            "committed_targets": TARGETS,
            "checkpoint_sha256": actual_hash,
            "files": {
                name: {"archive_path": f"runs/{run_id}/{name}", "sha256": sha256(path),
                       "size_bytes": path.stat().st_size}
                for name, path in files.items()
            },
        })
        sources[run_id] = files

    STAGING.mkdir(parents=True, exist_ok=True)
    inventory = {
        "dataset_id": DATASET_ID,
        "campaign_config": "configs/locked_campaign.json",
        "campaign_config_sha256": sha256(CAMPAIGN_PATH),
        "run_count": len(entries),
        "runs": entries,
    }
    inventory_path = STAGING / "asset_inventory.json"
    zip_path = STAGING / "locked_checkpoints.zip"
    inventory_bytes = (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode("utf-8")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for run_id in run_ids:
            for name, source in sources[run_id].items():
                archive.write(source, f"runs/{run_id}/{name}")
        archive.writestr("asset_inventory.json", inventory_bytes)

    inventory_path.write_bytes(inventory_bytes)
    metadata = {
        "id": DATASET_ID,
        "title": "Session 13 locked checkpoints",
        "licenses": [{"name": "other"}],
        "description": (
            "Private research checkpoints for the eight verified 50M-target locked "
            "Session 13 runs. Contains checkpoint files, run configurations, final "
            "summaries, and SHA-256 inventory; contains no token tapes or credentials."
        ),
    }
    (STAGING / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "dataset_id": DATASET_ID,
        "run_count": len(entries),
        "zip_path": str(zip_path),
        "zip_size_bytes": zip_path.stat().st_size,
        "inventory_path": str(inventory_path),
    }, indent=2))


if __name__ == "__main__":
    main()
