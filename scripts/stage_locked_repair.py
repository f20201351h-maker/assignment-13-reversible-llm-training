"""Stage protocol 08's five remaining confirmations; retain the failed campaign."""
import json
from pathlib import Path
import stage_locked_kernel as original

ROOT = Path(__file__).resolve().parents[1]
stage = ROOT/'staging'/'kaggle-locked-repair'
stage.mkdir(parents=True, exist_ok=True)
configs = {}
for old in ('C-L-1337', 'E-L-1337', 'A-L-2027', 'B-L-2027', 'C-L-2027'):
    cfg = dict(original.configs[old])
    name = old.replace('-L-', '-L191-') if old[0] in 'CE' else old
    if old[0] in 'CE':
        cfg['effective_batch'] = 191
        if old[0] == 'C':
            cfg['physical_batch'] = 191
    cfg.update(run_id=name, out_dir=f'/kaggle/working/runs/{name}')
    configs[name] = cfg
    (ROOT/'configs'/'locked'/f'{name}.json').write_text(json.dumps(cfg, indent=2))
entry = original.entry.replace(repr(original.configs), repr(configs))
entry = entry.replace(repr(list(original.configs)), repr(list(configs)))
entry = entry.replace('max_updates=1 if preflight else None',
                      "max_updates=(200 if '-L191-' in name else 1) if preflight else None")
entry = entry.replace('protocol-06-locked-backend', 'protocol-08-capacity-repair')
(stage/'repair.py').write_text(entry, encoding='utf-8')
metadata = dict(original.metadata)
metadata.update(id='KAGGLE_USER/reversible-llm-session-13-locked-capacity-repair',
                title='Reversible LLM Session 13 locked capacity repair', code_file='repair.py')
(stage/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
manifest = {'protocol':'protocols/08_locked_capacity_repair.md', 'B0':110, 'BR':191,
            'checkpointed_max':182,
            'run_ids':['A-L-1337','B-L-1337','D-L-1337',*configs],
            'kernel_sources':['KAGGLE_USER/reversible-llm-session-13-locked-confirmation',
                              metadata['id']],
            'local_roots':['experiments/kaggle-locked-v1/runs',
                           'experiments/kaggle-locked-repair-v1/runs'],
            'benchmark_root':'experiments/kaggle-benchmark-v2'}
(ROOT/'configs'/'locked_campaign.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps({'stage':str(stage),'pending_runs':list(configs)}))
