"""Freeze principal run configs and stage the private two-worker Kaggle job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "staging" / "kaggle-full"
CONFIGS = ROOT / "configs" / "runs"
ASSETS = "/kaggle/input/datasets/KAGGLE_USER/revllm-session13-assets"
ENTRY = r'''
"""Independent single-T4 principal runs. The two GPUs do not form one model."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/kaggle/working")
MOUNT = list(Path("/kaggle/input").rglob("manifest.json"))
if len(MOUNT) != 1:
    raise RuntimeError(f"Expected one private asset mount, got {MOUNT}")
ASSETS = MOUNT[0].parent
RUN_CONFIGS = __RUN_CONFIGS__

def worker(name, preflight=False):
    sys.path.insert(0, str(ASSETS / "revllm_source"))
    from revllm.model import ModelConfig
    from revllm.train import RunConfig, train
    raw = dict(RUN_CONFIGS[name])
    raw["model"] = ModelConfig(**raw["model"])
    raw["data_dir"] = ASSETS
    raw["out_dir"] = ROOT / ("preflight" if preflight else "runs") / name
    if preflight:
        raw["run_id"] = f"preflight-{name}"
    raw["adam_betas"] = tuple(raw.get("adam_betas", (0.9, 0.95)))
    result = train(RunConfig(**raw), max_updates=1 if preflight else None)
    print(json.dumps(result, indent=2), flush=True)

def main():
    if len(sys.argv) == 3 and sys.argv[1] in {"worker", "preflight"}:
        worker(sys.argv[2], preflight=sys.argv[1] == "preflight")
        return
    gate_env = os.environ.copy()
    gate_env["PYTHONPATH"] = str(ASSETS / "revllm_source")
    gate_env["CUDA_VISIBLE_DEVICES"] = "0"
    gate = subprocess.run([sys.executable, "-c",
        "from pathlib import Path; from revllm.correctness import run_gpu_gates; run_gpu_gates(Path('/kaggle/working/full_gpu_correctness.json'))"],
        env=gate_env, text=True, capture_output=True)
    print(f"Full-run numerical gate exit={gate.returncode}: {gate.stdout[-1000:]} {gate.stderr[-1000:]}", flush=True)
    if gate.returncode:
        raise RuntimeError("Numerical gate failed before full runs")
    for name in __RUN_NAMES__:
        test = subprocess.run([sys.executable, __file__, "preflight", name],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}, text=True, capture_output=True)
        print(f"Exact training-path preflight {name} exit={test.returncode}: {test.stdout[-1200:]} {test.stderr[-1200:]}", flush=True)
        if test.returncode:
            raise RuntimeError(f"Training-path capacity preflight failed: {name}")
    names = __RUN_NAMES__
    for start in range(0, len(names), 2):
        running = []
        for gpu, name in enumerate(names[start:start + 2]):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            path = ROOT / f"{name}.stdout.log"
            stream = path.open("w", encoding="utf-8")
            process = subprocess.Popen([sys.executable, __file__, "worker", name],
                                       env=env, stdout=stream, stderr=subprocess.STDOUT)
            running.append((name, process, stream, path))
            print(f"Started {name} on physical GPU {gpu}", flush=True)
        monitor = ROOT / 'gpu_monitor.jsonl'
        while any(process.poll() is None for _, process, _, _ in running):
            try:
                sample = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used',
                     '--format=csv,noheader,nounits'], text=True, capture_output=True)
                apps = subprocess.run(
                    ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                     '--format=csv,noheader,nounits'], text=True, capture_output=True)
                snapshot = {'gpu_query_exit': sample.returncode, 'gpu_query': sample.stdout,
                            'compute_apps_exit': apps.returncode, 'compute_apps': apps.stdout}
            except OSError as error:
                snapshot = {'monitor_error': str(error)}
            with monitor.open('a', encoding='utf-8') as record:
                record.write(json.dumps({'timestamp_unix': time.time(),
                    'workers': [{'run_id': name, 'pid': process.pid,
                                 'running': process.poll() is None}
                                for name, process, _, _ in running],
                    **snapshot}) + '\n')
            time.sleep(10)
        failed = []
        for name, process, stream, path in running:
            code = process.wait()
            stream.close()
            print(f"{name} exit={code}; tail={path.read_text(encoding='utf-8')[-2000:]}", flush=True)
            if code:
                failed.append(name)
        if failed:
            raise RuntimeError(f"Principal runs failed; retained outputs: {failed}")

if __name__ == "__main__":
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0", type=int, required=True)
    parser.add_argument("--br", type=int, required=True)
    args = parser.parse_args()
    if args.b0 < 1 or args.br < args.b0:
        raise ValueError("Require 1 <= B0 <= BR")
    STAGE.mkdir(parents=True, exist_ok=True)
    CONFIGS.mkdir(parents=True, exist_ok=True)
    conditions = [
        ("A-1337", "conventional", "stored", args.b0, args.b0, 1337),
        ("B-1337", "coupled_euler", "reconstructed", args.b0, args.b0, 1337),
        ("C-1337", "coupled_euler", "reconstructed", args.br, args.br, 1337),
        ("D-1337", "coupled_euler", "stored", args.b0, args.b0, 1337),
        ("E-1337", "coupled_euler", "reconstructed", args.b0, args.br, 1337),
        ("A-2027", "conventional", "stored", args.b0, args.b0, 2027),
        ("B-2027", "coupled_euler", "reconstructed", args.b0, args.b0, 2027),
        ("C-2027", "coupled_euler", "reconstructed", args.br, args.br, 2027),
    ]
    if args.br == args.b0:
        # Identical C/E conditions are not submitted as independent evidence.
        conditions = [c for c in conditions if c[0] not in {"C-1337", "E-1337", "C-2027"}]
    names = []
    run_configs = {}
    for name, integrator, backward, physical, effective, seed in conditions:
        cfg = {
            "run_id": name,
            "model": {"integrator": integrator, "backward_mode": backward,
                      "step_size": 0.5, "blend": 1.0},
            "data_dir": ASSETS, "out_dir": f"/kaggle/working/runs/{name}",
            "physical_batch": physical, "effective_batch": effective,
            "seed": seed, "target_tokens": 50_000_000,
            "warmup_tokens": 1_000_000, "use_amp": True,
            "evaluate_holdout": True,
            "eval_every_tokens": 5_000_000, "checkpoint_every_tokens": 5_000_000,
        }
        body = json.dumps(cfg, indent=2)
        (CONFIGS / f"{name}.json").write_text(body, encoding="utf-8")
        (STAGE / f"{name}.json").write_text(body, encoding="utf-8")
        names.append(name)
        run_configs[name] = cfg
    entry = ENTRY.replace("__RUN_NAMES__", repr(names)).replace("__RUN_CONFIGS__", repr(run_configs))
    (STAGE / "full.py").write_text(entry, encoding="utf-8")
    metadata = {
        "id": "KAGGLE_USER/reversible-llm-session-13-full-runs",
        "title": "Reversible LLM Session 13 full runs",
        "code_file": "full.py", "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_gpu": "true", "enable_internet": "false",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["KAGGLE_USER/revllm-session13-assets"],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    (STAGE / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"stage": str(STAGE), "run_ids": names,
                      "B0": args.b0, "BR": args.br}, indent=2))


if __name__ == "__main__":
    main()
