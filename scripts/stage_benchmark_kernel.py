"""Stage an isolated private Kaggle benchmark with GPU-assignment swaps."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "staging" / "kaggle-benchmark"
CONFIGS = ROOT / "configs" / "runs"
ENTRY = r'''
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path('/kaggle/working')
MOUNTS = list(Path('/kaggle/input').rglob('manifest.json'))
if len(MOUNTS) != 1:
    raise RuntimeError(f'Expected one private asset mount, found {MOUNTS}')
ASSETS = MOUNTS[0].parent
CONFIGS = __CONFIGS__

def worker(name, assigned_gpu):
    sys.path.insert(0, str(ASSETS / 'revllm_source'))
    from revllm.model import ModelConfig
    from revllm.profile import benchmark
    from revllm.train import RunConfig
    raw = dict(CONFIGS[name])
    raw['model'] = ModelConfig(**raw['model'])
    raw['data_dir'] = ASSETS
    raw['out_dir'] = ROOT / 'benchmark' / name
    raw['adam_betas'] = tuple(raw.get('adam_betas', (0.9, 0.95)))
    result = benchmark(RunConfig(**raw), repetitions=1, warmup=20, measured=100)
    result['condition'] = name
    result['assigned_physical_gpu'] = assigned_gpu
    result['source_asset_path'] = str(ASSETS)
    print(json.dumps(result), flush=True)

def main():
    if len(sys.argv) == 4 and sys.argv[1] == 'worker':
        worker(sys.argv[2], int(sys.argv[3]))
        return
    out = ROOT / 'benchmark'
    out.mkdir(exist_ok=True)
    hardware = subprocess.run(['nvidia-smi', '-L'], text=True, capture_output=True)
    (out / 'hardware.txt').write_text(hardware.stdout + hardware.stderr)
    summary = {}
    for name in CONFIGS:
        records = []
        for repetition, gpu in enumerate((0, 1, 0)):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            completed = subprocess.run([sys.executable, __file__, 'worker', name, str(gpu)],
                                       env=env, text=True, capture_output=True)
            log = out / f'{name}-rep{repetition}.stdout.log'
            log.write_text(completed.stdout + completed.stderr)
            if completed.returncode:
                failure = {'condition': name, 'repetition': repetition, 'assigned_physical_gpu': gpu,
                           'returncode': completed.returncode, 'log': str(log)}
                (out / f'{name}-rep{repetition}-failure.json').write_text(json.dumps(failure, indent=2))
                raise RuntimeError(f'Benchmark failure: {failure}')
            record = json.loads(completed.stdout)
            record['repetition_index'] = repetition
            (out / f'{name}-rep{repetition}.json').write_text(json.dumps(record, indent=2))
            records.append(record)
            print(f'{name} repetition {repetition} GPU {gpu}: '
                  f"{record['median_tokens_per_second']:.1f} targets/s", flush=True)
        rates = [record['median_tokens_per_second'] for record in records]
        summary[name] = {'condition': name, 'physical_batch': records[0]['physical_batch'],
                         'effective_batch': records[0]['effective_batch'],
                         'median_targets_per_second': statistics.median(rates),
                         'min_targets_per_second': min(rates),
                         'max_targets_per_second': max(rates),
                         'peak_allocated_bytes': max(record['repetitions'][0]['peak_allocated_bytes']
                                                     for record in records),
                         'peak_reserved_bytes': max(record['repetitions'][0]['peak_reserved_bytes']
                                                    for record in records),
                         'raw_repetitions': [f'{name}-rep{i}.json' for i in range(3)]}
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--capacity-repair', action='store_true')
    args = parser.parse_args()
    STAGE.mkdir(parents=True, exist_ok=True)
    configs = {name: json.loads((CONFIGS / f"{name}-1337.json").read_text(encoding="utf-8"))
               for name in "ABCDE"}
    for raw in configs.values():
        raw["attention_backend"] = "efficient"
    maximum_checkpointed = 181
    if args.capacity_repair:
        campaign = json.loads((ROOT/'configs/locked_campaign.json').read_text())
        configs['C']['physical_batch'] = campaign['BR']
        configs['C']['effective_batch'] = configs['E']['effective_batch'] = campaign['BR']
        maximum_checkpointed = campaign['checkpointed_max']
    for name, batch in (("checkpointed-B0", 110), ("checkpointed-max", maximum_checkpointed)):
        raw = json.loads(json.dumps(configs["A"]))
        raw["run_id"] = name
        raw["model"]["backward_mode"] = "checkpointed"
        raw["physical_batch"] = raw["effective_batch"] = batch
        configs[name] = raw
    (STAGE / "benchmark.py").write_text(ENTRY.replace("__CONFIGS__", repr(configs)), encoding="utf-8")
    metadata = {
        "id": "KAGGLE_USER/reversible-llm-session-13-isolated-benchmarks",
        "title": "Reversible LLM Session 13 isolated benchmarks",
        "code_file": "benchmark.py", "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_gpu": "true", "enable_internet": "false",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["KAGGLE_USER/revllm-session13-assets"],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    (STAGE / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"stage": str(STAGE), "conditions": list(configs)}, indent=2))


if __name__ == "__main__":
    main()
