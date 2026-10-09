"""Generate and execute the five evidence-backed notebooks from shared modules."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
import hashlib
import json


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
SETUP = """from pathlib import Path
import json, csv, sys
ROOT = Path.cwd()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
assert (ROOT / 'src' / 'revllm').exists(), 'Execute with repository root as working directory'
sys.path.insert(0, str(ROOT / 'src'))
from revllm.data import digest_file
def read_final(path):
    result = json.loads(path.read_text())
    updates = [json.loads(line) for line in (path.parent/'metrics.jsonl').read_text().splitlines()]
    assert sum(row['valid_targets'] for row in updates if row['status']=='COMMITTED') == result['committed_targets']
    return result
"""
DEMO = """# Real two-update execution; excluded from assignment token totals.
from tempfile import TemporaryDirectory
from revllm.model import ModelConfig
from revllm.train import RunConfig, train
with TemporaryDirectory(prefix='revllm-notebook-') as scratch:
    cfg = RunConfig(run_id='notebook-demo',
                    model=ModelConfig(n_layer=2, n_embd=16, n_head=2, block_size=16,
                                      integrator=INTEGRATOR, backward_mode=BACKWARD, step_size=0.5),
                    data_dir=ROOT/'data', out_dir=Path(scratch), physical_batch=2, effective_batch=2,
                    target_tokens=256, warmup_tokens=128, use_amp=False)
    measured = train(cfg, max_updates=2)
    assert measured['committed_targets'] == 64
    print({'demonstration': INTEGRATOR, 'successful_updates': measured['updates'],
           'committed_targets': measured['committed_targets'], 'assignment_run': False})
"""


def make(title: str, purpose: str, cells: list[str]) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{purpose}"),
                      nbf.v4.new_code_cell(SETUP)] + [nbf.v4.new_code_cell(cell) for cell in cells]
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return notebook


def main() -> None:
    NOTEBOOKS.mkdir(exist_ok=True)
    specs = {
        "00_setup_and_correctness.ipynb": make(
            "Setup and correctness", "Reads the prepared manifest and executes a small real inverse test.", [
                "from revllm.model import ModelConfig, TinyGPT, count_unique_parameters\nfrom revllm.correctness import reconstruction_probe\nprint({kind: count_unique_parameters(TinyGPT(ModelConfig(integrator=kind))) for kind in ('conventional','midpoint','coupled_euler','blended_midpoint')})",
                "model = TinyGPT(ModelConfig(vocab_size=32, block_size=8, n_layer=4, n_head=2, n_embd=16, integrator='midpoint', backward_mode='reconstructed')).double()\nprint(reconstruction_probe(model))",
                "manifest = json.loads((ROOT/'data'/'manifest.json').read_text())\nprint({'revision': manifest['dataset_revision'], 'tokenizer_sha256': manifest['tokenizer_sha256'], 'targets': {k:v['target_count'] for k,v in manifest['tapes'].items()}})",
            ]),
        "01_baseline.ipynb": make(
            "Conventional baseline", "Inputs: prepared data tapes and retained Kaggle logs. Executes two real small-model updates through revllm.train, then checks the full-run evidence. The demonstration is not a 50M-token assignment run.", [
                DEMO.replace('INTEGRATOR', repr('conventional')).replace('BACKWARD', repr('stored')),
                "path = ROOT/'experiments'/'kaggle-pilots-v3'/'runs'/'pilot-conventional'\nfinal = json.loads((path/'final.json').read_text())\nupdates = [json.loads(x) for x in (path/'metrics.jsonl').read_text().splitlines()]\nprint({'run_id': final['run_id'], 'committed_targets': final['committed_targets'], 'updates': final['updates'], 'dev_loss_nats': json.loads((path/'eval.jsonl').read_text().splitlines()[-1])['dev_loss_nats'], 'raw_update_records': len(updates)})",
                "for campaign in ('kaggle-full-v1','kaggle-locked-v1','kaggle-locked-repair-v1'):\n    for path in sorted((ROOT/'experiments'/campaign/'runs').glob('A-*/final.json')):\n        r=read_final(path)\n        print({k:r[k] for k in ('run_id','committed_targets','updates','holdout_loss_nats','training_tokens_per_second')})",
            ]),
        "02_reversible_variants.ipynb": make(
            "Reversible candidates", "Shows measured pilot development losses and trained-checkpoint numerical gates.", [
                DEMO.replace('INTEGRATOR', repr('coupled_euler')).replace('BACKWARD', repr('reconstructed')),
                "base = ROOT/'experiments'/'kaggle-pilots-v3'\nrows = []\nfor path in sorted((base/'runs').glob('pilot-*/final.json')):\n    final = json.loads(path.read_text())\n    ev = json.loads((path.parent/'eval.jsonl').read_text().splitlines()[-1])\n    rows.append((final['run_id'], ev['dev_loss_nats'], final['committed_targets']))\nfor row in sorted(rows, key=lambda x:x[1]): print(row)",
                "checks = json.loads((base/'trained_checkpoint_probes.json').read_text())\nprint([(c['run_id'],c['status'],c.get('global_relative_gradient_error_fp32'),c.get('reconstruction',{}).get('relative_reconstruction_error')) for c in checks])",
                "for campaign in ('kaggle-full-v1','kaggle-locked-v1','kaggle-locked-repair-v1'):\n    for path in sorted((ROOT/'experiments'/campaign/'runs').glob('[BCDE]-*/final.json')):\n        r=read_final(path)\n        print(r['run_id'], r['committed_targets'], r['holdout_loss_nats'])",
            ]),
        "03_maximum_batch.ipynb": make(
            "Maximum physical batch", "Reads fresh-process search results; never infers an OOM boundary from a missing trial.", [
                "files = sorted((ROOT/'experiments').rglob('batch-search*/search/summary.json'))\nprint([(str(p.relative_to(ROOT)),json.loads(p.read_text())) for p in files] if files else 'Batch searches pending')",
            ]),
        "04_results_and_audit.ipynb": make(
            "Results and evidence audit", "Verifies the raw inputs behind the figures and displays regenerated results. The external audit/ledger.md is the authoritative final audit; its status can change after this notebook executes.", [
                "table = ROOT/'results'/'runs.csv'\nrows = list(csv.DictReader(table.open(newline='',encoding='utf-8'))) if table.exists() else []\nfull = [r for r in rows if r['run_id'][:1] in 'ABCDE' and r['committed_targets']=='50000000']\nprint({'verified_50m_runs':len(full), 'run_ids':[r['run_id'] for r in full]})",
                "provenance=json.loads((ROOT/'results/figures/figure_provenance.json').read_text())\nfor item in provenance['inputs']:\n    assert digest_file(ROOT/item['path']) == item['sha256'], item['path']\nprint({'verified_figure_inputs':len(provenance['inputs']), 'main_campaign':provenance['main_campaign'], 'gaps':provenance['gaps']})",
                "from IPython.display import Image, display\nfor name in ('development_loss_curves','causal_loss_chain_seed1337','memory_throughput','seed_variation_holdout'):\n    display(Image(filename=str(ROOT/'results/figures'/f'{name}.png'),width=900))",
            ]),
    }
    execution = []
    for filename, notebook in specs.items():
        path = NOTEBOOKS / filename
        executed = NotebookClient(notebook, timeout=180, kernel_name="python3",
                                  resources={"metadata": {"path": str(ROOT)}}).execute()
        nbf.write(executed, path)
        execution.append({'path': str(path.relative_to(ROOT)),
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'code_cells': sum(cell.cell_type == 'code' for cell in executed.cells),
                          'status': 'PASS'})
        print(f"Executed {path.name}")
    (ROOT/'audit'/'notebook_execution.json').write_text(json.dumps(execution, indent=2))


if __name__ == "__main__":
    main()
