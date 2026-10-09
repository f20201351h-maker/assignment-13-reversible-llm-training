"""Fresh-process OOM-boundary search with retained pass and failure evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def trial(config: Path, batch: int, updates: int, log: Path, env: dict) -> dict:
    command = [sys.executable, "-m", "revllm.cli", "probe-once",
               "--config", str(config), "--batch", str(batch), "--updates", str(updates)]
    result = subprocess.run(command, text=True, capture_output=True, env=env)
    record = {
        "batch": batch, "updates_requested": updates, "returncode": result.returncode,
        "stdout": result.stdout[-12000:], "stderr": result.stderr[-12000:],
    }
    if result.returncode == 0:
        try:
            record["measurement"] = json.loads(result.stdout)
            record["status"] = "PASS"
        except json.JSONDecodeError:
            record["status"] = "INVALID_OUTPUT"
    elif "out of memory" in (result.stderr + result.stdout).lower():
        record["status"] = "OOM"
    else:
        record["status"] = "FAIL"
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    return record


def search(config: Path, output: Path, *, max_batch: int = 512) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    log = output / "trials.jsonl"
    if log.exists():
        raise RuntimeError("Search output already exists; use a fresh experiment ID")
    env = os.environ.copy()
    # Callers supply PYTHONPATH (or install revllm); preserve their chosen GPU mapping.
    low, high = 0, None
    batch = 1
    while batch <= max_batch:
        result = trial(config, batch, 20, log, env)
        print(result["status"], batch, flush=True)
        if result["status"] == "PASS":
            low = batch
            batch *= 2
        elif result["status"] == "OOM":
            high = batch
            break
        else:
            raise RuntimeError(f"Non-OOM failure at batch {batch}; inspect {log}")
    if high is not None:
        while high - low > 1:
            mid = (low + high) // 2
            result = trial(config, mid, 20, log, env)
            print(result["status"], mid, flush=True)
            if result["status"] == "PASS":
                low = mid
            elif result["status"] == "OOM":
                high = mid
            else:
                raise RuntimeError(f"Non-OOM failure at batch {mid}; inspect {log}")
    if not low:
        raise RuntimeError(f"Batch 1 failed; inspect {log}")
    confirmations = [trial(config, low, 20, log, env) for _ in range(3)]
    sustained = trial(config, low, 200, log, env)
    status = "VALIDATED" if all(r["status"] == "PASS" for r in confirmations + [sustained]) else "UNSTABLE"
    summary = {
        "status": status, "largest_stable_batch": low if status == "VALIDATED" else None,
        "smallest_observed_oom_batch": high,
        "boundary_type": "observed_oom" if high is not None else "tested_lower_bound",
        "trials_log": str(log),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batch", type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(search(args.config, args.output, max_batch=args.max_batch), indent=2))
