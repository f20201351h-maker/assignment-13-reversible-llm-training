"""Private Kaggle entrypoint; runs independent single-GPU research workers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


INPUT_ROOT = Path("/kaggle/input")
ASSETS = Path("/kaggle/input/revllm-session13-assets")
if not ASSETS.exists():
    matches = list(INPUT_ROOT.rglob("manifest.json"))
    if len(matches) == 1:
        ASSETS = matches[0].parent
OUT = Path("/kaggle/working")
PILOTS = {
    "conventional": ("conventional", "stored", 0.25, 1.0),
    "midpoint025": ("midpoint", "reconstructed", 0.25, 1.0),
    "midpoint050": ("midpoint", "reconstructed", 0.5, 1.0),
    "midpoint100": ("midpoint", "reconstructed", 1.0, 1.0),
    "coupled025": ("coupled_euler", "reconstructed", 0.25, 1.0),
    "coupled050": ("coupled_euler", "reconstructed", 0.5, 1.0),
    "coupled100": ("coupled_euler", "reconstructed", 1.0, 1.0),
    "blended025": ("blended_midpoint", "reconstructed", 0.25, 0.5),
}


def worker(name: str, batch: int, targets: int) -> None:
    sys.path.insert(0, str(ASSETS / "revllm_source"))
    from revllm.model import ModelConfig
    from revllm.train import RunConfig, train

    integrator, backward, h, a = PILOTS[name]
    cfg = RunConfig(
        run_id=f"pilot-{name}", data_dir=ASSETS,
        out_dir=OUT / "runs" / f"pilot-{name}",
        model=ModelConfig(integrator=integrator, backward_mode=backward, step_size=h, blend=a),
        physical_batch=batch, effective_batch=batch, seed=1337,
        target_tokens=targets, eval_every_tokens=targets,
        checkpoint_every_tokens=targets, evaluate_holdout=False,
    )
    result = train(cfg)
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["pilots", "worker", "correctness"])
    parser.add_argument("--name", choices=PILOTS)
    parser.add_argument("--names", default=",".join(PILOTS))
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--targets", type=int, default=2_000_000)
    args = parser.parse_args(["pilots"] if len(sys.argv) == 1 else None)
    if args.action == "correctness":
        sys.path.insert(0, str(ASSETS / "revllm_source"))
        from revllm.correctness import run_gpu_gates
        print(json.dumps(run_gpu_gates(OUT / "gpu_correctness.json"), indent=2), flush=True)
        return
    if args.action == "worker":
        worker(args.name, args.batch, args.targets)
        return
    names = args.names.split(",")
    if not ASSETS.exists():
        raise RuntimeError(f"Missing private Kaggle dataset {ASSETS}; mounts={list(INPUT_ROOT.iterdir()) if INPUT_ROOT.exists() else 'no input root'}")
    sys.path.insert(0, str(ASSETS / "revllm_source"))
    from revllm.correctness import run_gpu_gates
    print(json.dumps(run_gpu_gates(OUT / "gpu_correctness.json"), indent=2), flush=True)
    for start in range(0, len(names), 2):
        processes = []
        for gpu, name in enumerate(names[start:start + 2]):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log = OUT / f"{name}.stdout.log"
            handle = log.open("w", encoding="utf-8")
            cmd = [sys.executable, __file__, "worker", "--name", name,
                   "--batch", str(args.batch), "--targets", str(args.targets)]
            print(f"Launching {name} on physical GPU {gpu}: {' '.join(cmd)}", flush=True)
            process = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT)
            processes.append((name, process, handle, log))
        for name, process, handle, log in processes:
            code = process.wait()
            handle.close()
            print(f"{name} exit={code} log={log}", flush=True)
            if code:
                print(log.read_text(encoding="utf-8")[-4000:], flush=True)
                raise RuntimeError(f"Pilot failed: {name}")


if __name__ == "__main__":
    main()
