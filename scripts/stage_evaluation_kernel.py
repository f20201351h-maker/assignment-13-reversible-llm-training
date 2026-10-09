"""Stage independent checkpoint reevaluation from the completed private run outputs."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "staging" / "kaggle-evaluation"
ENTRY = r'''
"""Reevaluate saved checkpoints in fresh processes against fixed token tapes."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('/kaggle/working')
RUN_NAMES = __RUN_NAMES__

def one(name, assigned_gpu):
    manifests = list(Path('/kaggle/input').rglob('manifest.json'))
    if len(manifests) != 1:
        raise RuntimeError(f'Expected one private data manifest, found {manifests}')
    data = manifests[0].parent
    sys.path.insert(0, str(data / 'revllm_source'))
    from revllm.evaluate import evaluate_checkpoint
    checkpoints = [path for path in Path('/kaggle/input').rglob(f'{name}/latest.pt')
                   if path.parent.parent.name == 'runs']
    if len(checkpoints) != 1:
        raise RuntimeError(f'Expected one saved checkpoint for {name}, found {checkpoints}')
    checkpoint = checkpoints[0]
    config = checkpoint.parent / 'run_config.json'
    final_path = checkpoint.parent / 'final.json'
    if not config.exists() or not final_path.exists():
        raise RuntimeError(f'Missing frozen config/final summary for {name}')
    final = json.loads(final_path.read_text())
    result = evaluate_checkpoint(config, checkpoint, data)
    if result['checkpoint_sha256'] != final['checkpoint_sha256']:
        raise RuntimeError(f'Checkpoint hash mismatch for {name}')
    result['assigned_physical_gpu'] = assigned_gpu
    result['original_holdout_loss_nats'] = final['holdout_loss_nats']
    result['holdout_recheck_absolute_difference'] = abs(
        result['evaluation']['holdout']['loss_nats'] - final['holdout_loss_nats'])
    if result['holdout_recheck_absolute_difference'] > 1e-4:
        raise RuntimeError(f'Holdout mismatch for {name}: {result}')
    out = ROOT / 'evaluations'
    out.mkdir(exist_ok=True)
    (out / f'{name}-evaluation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'run_id': name, 'holdout_recheck_absolute_difference':
                      result['holdout_recheck_absolute_difference']}), flush=True)

def main():
    if len(sys.argv) == 4 and sys.argv[1] == 'worker':
        one(sys.argv[2], int(sys.argv[3]))
        return
    out = ROOT / 'evaluations'
    out.mkdir(exist_ok=True)
    failures = []
    for start in range(0, len(RUN_NAMES), 2):
        processes = []
        for gpu, name in enumerate(RUN_NAMES[start:start + 2]):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            path = out / f'{name}.stdout.log'
            stream = path.open('w', encoding='utf-8')
            process = subprocess.Popen([sys.executable, __file__, 'worker', name, str(gpu)],
                                       env=env, stdout=stream, stderr=subprocess.STDOUT)
            processes.append((name, process, stream, path))
        for name, process, stream, path in processes:
            code = process.wait()
            stream.close()
            print(f'{name} evaluation exit={code}; tail={path.read_text()[-1200:]}', flush=True)
            if code:
                failures.append(name)
    if failures:
        raise RuntimeError(f'Checkpoint reevaluation failures: {failures}')

if __name__ == '__main__':
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', choices=('original', 'locked', 'supplemental'), default='original')
    parser.add_argument('--checkpoint-asset', action='store_true')
    args = parser.parse_args()
    stage = STAGE if args.campaign == 'original' else ROOT/'staging'/f'kaggle-{args.campaign}-evaluation'
    stage.mkdir(parents=True, exist_ok=True)
    names = [f"{letter}-1337" for letter in "ABCDE"] + [f"{letter}-2027" for letter in "ABC"]
    source = 'KAGGLE_USER/reversible-llm-session-13-full-runs'
    title = 'Reversible LLM Session 13 checkpoint evaluation'
    slug = 'reversible-llm-session-13-checkpoint-evaluation'
    if args.campaign == 'locked':
        campaign = json.loads((ROOT/'configs/locked_campaign.json').read_text())
        names = campaign['run_ids']
        source = campaign['kernel_sources']
    elif args.campaign == 'supplemental':
        names = ['A-3141', 'B-3141', 'C-3141', 'F-midpoint-1337', 'G-checkpointed-1337']
        source = 'KAGGLE_USER/reversible-llm-session-13-supplemental-runs'
    if args.campaign != 'original':
        title = f'Reversible LLM Session 13 {args.campaign} evaluation'
        slug = f'reversible-llm-session-13-{args.campaign}-evaluation'
    (stage / "evaluation.py").write_text(ENTRY.replace("__RUN_NAMES__", repr(names)), encoding="utf-8")
    metadata = {
        "id": f"KAGGLE_USER/{slug}",
        "title": title,
        "code_file": "evaluation.py", "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_gpu": "true", "enable_internet": "false",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["KAGGLE_USER/revllm-session13-assets"],
        "competition_sources": [],
        "kernel_sources": source if isinstance(source, list) else [source],
        "model_sources": [],
    }
    if args.checkpoint_asset:
        if args.campaign != 'locked':
            raise ValueError('Checkpoint asset contains only the locked campaign')
        metadata['dataset_sources'].append('KAGGLE_USER/revllm-session13-locked-checkpoints')
        metadata['kernel_sources'] = []
    (stage / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"stage": str(stage), "run_ids": names}, indent=2))


if __name__ == "__main__":
    main()
