"""Stage the preregistered private GPU diagnostic without changing training."""
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--campaign', choices=('original','locked'), default='original')
parser.add_argument('--checkpoint-asset', action='store_true')
args = parser.parse_args()
STAGE = ROOT / "staging" / ('kaggle-locked-diagnostics' if args.campaign == 'locked' else 'kaggle-diagnostics')
STAGE.mkdir(parents=True, exist_ok=True)
source = (ROOT / "scripts" / "gpu_diagnostics.py").read_text()
expected = json.loads((ROOT/"configs/locked_campaign.json").read_text())["run_ids"] if args.campaign == "locked" else []
entry = '''from pathlib import Path
import sys
manifests = list(Path('/kaggle/input').rglob('manifest.json'))
assert len(manifests) == 1, manifests
sys.path.insert(0, str(manifests[0].parent / 'revllm_source'))
sys.argv = ['gpu_diagnostics.py', '--input-root', '/kaggle/input', '--out', '/kaggle/working/diagnostics']
sys.argv += __EXPECTED_ARGS__
exec(compile(__SOURCE__, 'gpu_diagnostics.py', 'exec'))
'''.replace("__SOURCE__", repr(source)).replace("__EXPECTED_ARGS__", repr(["--expected-run-ids"] + expected if expected else []))
(STAGE / "diagnostics.py").write_text(entry, encoding="utf-8")
metadata = {
    "id": "KAGGLE_USER/reversible-llm-session-13-gpu-diagnostics",
    "title": "Reversible LLM Session 13 GPU diagnostics",
    "code_file": "diagnostics.py", "language": "python", "kernel_type": "script",
    "is_private": "true", "enable_gpu": "true", "enable_internet": "false",
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": ["KAGGLE_USER/revllm-session13-assets"],
    "competition_sources": [], "model_sources": [],
    "kernel_sources": ["KAGGLE_USER/reversible-llm-session-13-full-runs"],
}
if args.campaign == 'locked':
    metadata['id'] = 'KAGGLE_USER/reversible-llm-session-13-locked-diagnostics'
    metadata['title'] = 'Reversible LLM Session 13 locked diagnostics'
    metadata['kernel_sources'] = json.loads((ROOT/'configs/locked_campaign.json').read_text())['kernel_sources']
if args.checkpoint_asset:
    if args.campaign != 'locked':
        raise ValueError('Checkpoint asset contains only the locked campaign')
    metadata['dataset_sources'].append('KAGGLE_USER/revllm-session13-locked-checkpoints')
    metadata['kernel_sources'] = []
(STAGE / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
print(STAGE)
