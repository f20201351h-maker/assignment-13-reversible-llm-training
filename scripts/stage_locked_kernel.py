"""Stage protocol 06 confirmations with an explicitly locked SDPA backend."""
import json
from pathlib import Path

from stage_full_kernel import ENTRY

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "staging" / "kaggle-locked"
STAGE.mkdir(parents=True, exist_ok=True)
frozen = ROOT / "configs" / "locked"
frozen.mkdir(parents=True, exist_ok=True)
names = [f"{letter}-{seed}" for seed, letters in ((1337, "ABCDE"), (2027, "ABC")) for letter in letters]
configs = {}
for original in names:
    cfg = json.loads((ROOT / "configs" / "runs" / f"{original}.json").read_text())
    letter, seed = original.split("-")
    name = f"{letter}-L-{seed}"
    cfg.update(run_id=name, out_dir=f"/kaggle/working/runs/{name}", attention_backend="efficient")
    configs[name] = cfg
    (frozen / f"{name}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
entry = ENTRY.replace("__RUN_NAMES__", repr(list(configs))).replace("__RUN_CONFIGS__", repr(configs))
entry = entry.replace('    gate_env = os.environ.copy()', '''    import hashlib
    import importlib.metadata
    hardware = subprocess.run(['nvidia-smi', '-q'], text=True, capture_output=True)
    (ROOT / 'hardware.txt').write_text(hardware.stdout + hardware.stderr)
    environment = {'packages': {dist.metadata['Name']: dist.version
                               for dist in importlib.metadata.distributions() if dist.metadata['Name']},
                   'source_sha256': {str(path.relative_to(ASSETS)): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in sorted((ASSETS / 'revllm_source').rglob('*.py'))},
                   'run_configs': RUN_CONFIGS,
                   'campaign': 'protocol-06-locked-backend',
                   'started_unix': time.time()}
    (ROOT / 'environment.json').write_text(json.dumps(environment, indent=2))
    gate_env = os.environ.copy()''')
entry = entry.replace(
    "from pathlib import Path; from revllm.correctness import run_gpu_gates;",
    "from pathlib import Path; import torch; from revllm.backend import configure_attention_backend; "
    "configure_attention_backend('efficient', torch.device('cuda', 0)); "
    "from revllm.correctness import run_gpu_gates;")
(STAGE / "locked.py").write_text(entry, encoding="utf-8")
metadata = {
    "id": "KAGGLE_USER/reversible-llm-session-13-locked-confirmation",
    "title": "Reversible LLM Session 13 locked confirmation",
    "code_file": "locked.py", "language": "python", "kernel_type": "script",
    "is_private": "true", "enable_gpu": "true", "enable_internet": "false",
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": ["KAGGLE_USER/revllm-session13-assets"],
    "competition_sources": [], "kernel_sources": [], "model_sources": [],
}
(STAGE / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(json.dumps({"stage": str(STAGE), "runs": list(configs)}))
