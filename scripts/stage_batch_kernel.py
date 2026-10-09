"""Stage four private T4 batch-boundary searches with independent GPU workers."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "staging" / "kaggle-batch-search"
SEARCH = (ROOT / "scripts" / "batch_search.py").read_text(encoding="utf-8").split('if __name__ == "__main__":')[0]
ENTRY = r'''
import os

CONDITIONS = {
    "conventional": ("conventional", "stored"),
    "checkpointed": ("conventional", "checkpointed"),
    "coupled_stored": ("coupled_euler", "stored"),
    "coupled_reconstructed": ("coupled_euler", "reconstructed"),
}

def assets_dir():
    matches = list(Path("/kaggle/input").rglob("manifest.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one manifest mount, found {matches}")
    return matches[0].parent

def worker(name):
    assets = assets_dir()
    integrator, backward = CONDITIONS[name]
    out = Path("/kaggle/working") / f"batch-search-{name}"
    cfg = {
        "run_id": f"batch-search-{name}",
        "model": {"integrator": integrator, "backward_mode": backward, "step_size": 0.5},
        "data_dir": str(assets), "out_dir": str(out),
        "physical_batch": 1, "effective_batch": 1, "seed": 1337,
        "use_amp": True,
    }
    out.mkdir(parents=True)
    config = out / "config.json"
    config.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    result = search(config, out / "search", max_batch=256)
    print(json.dumps(result, indent=2), flush=True)

def main():
    if len(sys.argv) == 3 and sys.argv[1] == "worker":
        worker(sys.argv[2])
        return
    assets = assets_dir()
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = str(assets / "revllm_source")
    names = list(CONDITIONS)
    for start in range(0, len(names), 2):
        running = []
        for gpu, name in enumerate(names[start:start+2]):
            env = env_base.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            path = Path("/kaggle/working") / f"{name}.log"
            stream = path.open("w", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, __file__, "worker", name],
                                    env=env, stdout=stream, stderr=subprocess.STDOUT)
            running.append((name, proc, stream, path))
            print(f"Started {name} on physical GPU {gpu}", flush=True)
        for name, proc, stream, path in running:
            code = proc.wait()
            stream.close()
            print(f"{name} exit={code}: {path.read_text(encoding='utf-8')[-1500:]}", flush=True)
            if code:
                raise RuntimeError(f"Batch search failed: {name}")

if __name__ == "__main__":
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--locked', action='store_true')
    args = parser.parse_args()
    stage = ROOT/'staging'/'kaggle-locked-batch-search' if args.locked else STAGE
    stage.mkdir(parents=True, exist_ok=True)
    entry = ENTRY
    if args.locked:
        entry = entry.replace('"use_amp": True,', '"use_amp": True, "attention_backend": "efficient",')
    # The generated entry embeds the same search() implementation used locally.
    (stage / "batch.py").write_text(SEARCH + "\n" + entry, encoding="utf-8")
    meta = {
        "id": "KAGGLE_USER/reversible-llm-session-13-batch-search",
        "title": "Reversible LLM Session 13 batch search",
        "code_file": "batch.py", "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_gpu": "true", "enable_internet": "false",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["KAGGLE_USER/revllm-session13-assets"],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    if args.locked:
        meta['id'] = 'KAGGLE_USER/reversible-llm-session-13-locked-batch-search'
        meta['title'] = 'Reversible LLM Session 13 locked batch search'
    (stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(stage)


if __name__ == "__main__":
    main()
