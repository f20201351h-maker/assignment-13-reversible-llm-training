"""Compare same-assignment repetitions without concealing device/order variation."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'experiments/kaggle-benchmark-v2/benchmark'
rows=[]
inputs=[]
for rep in range(3):
    records={}
    for condition in ('A','B','C','D','E','checkpointed-B0','checkpointed-max'):
        path=base/f'{condition}-rep{rep}.json'
        record=json.loads(path.read_text())
        measurement=record['repetitions'][0]
        assert record['status']=='COMPLETE' and measurement['successful_updates']==100
        assert measurement['warmup_updates']==20
        records[condition]=record
        inputs.append({'path':path.relative_to(ROOT).as_posix(),
                       'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    for left,right,label in [('D','B','reconstruction'),('E','C','physical_batch'),
                              ('B','C','capacity_combined'),('A','checkpointed-B0','checkpointing')]:
        a,b=records[left],records[right]
        assert a['gpu_uuid']==b['gpu_uuid'] and a['assigned_physical_gpu']==b['assigned_physical_gpu']
        rows.append({'contrast':label,'repetition':rep,'assigned_gpu':a['assigned_physical_gpu'],
                     'gpu_uuid':a['gpu_uuid'],'from_condition':left,'to_condition':right,
                     'throughput_ratio':b['median_tokens_per_second']/a['median_tokens_per_second']})
out=ROOT/'results/benchmark_comparisons.json'
out.write_text(json.dumps({'comparisons':rows,'inputs':inputs,
                          'limitation':'Same GPU assignment and repetition index; separate sequential trials, not simultaneous pairs. GPU identity and repetition order are confounded.'},indent=2))
with out.with_suffix('.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
print(json.dumps(rows))
