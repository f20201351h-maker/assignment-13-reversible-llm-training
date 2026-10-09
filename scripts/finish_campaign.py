"""Bounded private-job continuation. Stops on failures; never retries a push.

Run once after the protocol-08 repair launch. Persistent dispatch records prevent
duplicate launches if the local process is interrupted. This does not publish,
submit, or mark the study complete; final evidence review remains separate.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/final-continuation'
OUT.mkdir(exist_ok=True)
DEADLINE = time.monotonic() + 4*3600
PY = sys.executable
OWNER = 'KAGGLE_USER/'
REPAIR = OWNER+'reversible-llm-session-13-locked-capacity-repair'
BENCH = OWNER+'reversible-llm-session-13-isolated-benchmarks'
EVAL = OWNER+'reversible-llm-session-13-locked-evaluation'
DIAG = OWNER+'reversible-llm-session-13-locked-diagnostics'

def event(phase, **fields):
    record = {'time_unix':time.time(), 'pid':os.getpid(), 'phase':phase, **fields}
    with (OUT/'events.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n')
    (OUT/'status.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)

def command(args, *, label=None):
    result = subprocess.run(args,cwd=ROOT,text=True,capture_output=True,timeout=1800)
    if label:
        (OUT/(label+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'{args[:3]} failed: {(result.stdout+result.stderr)[-1500:]}')
    return result.stdout

def status(slug):
    output = command(['kaggle','kernels','status',slug])
    match = re.search(r'KernelWorkerStatus\.(\w+)',output)
    if not match:
        raise RuntimeError(f'Cannot establish job state: {output}')
    return match.group(1)

def wait_complete(slug):
    previous = None
    while time.monotonic()<DEADLINE:
        current=status(slug)
        if current!=previous:
            event('job-status',job=slug,status=current)
            previous=current
        if current=='COMPLETE':
            return
        if current not in {'RUNNING','QUEUED'}:
            raise RuntimeError(f'{slug} ended {current}; preserve failure and investigate')
        time.sleep(45)
    raise TimeoutError('Four-hour continuation deadline reached')

def push_once(stage,slug,version):
    marker=OUT/(slug.rsplit('/',1)[-1]+f'-v{version}-dispatch.json')
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved.get('status')!='CONFIRMED':
            raise RuntimeError(f'Uncertain previous dispatch: inspect {marker}; do not retry automatically')
        return
    quota=command(['kaggle','quota'],label=f'quota-before-{version}-{Path(stage).name}')
    row=next((line for line in quota.splitlines() if line.startswith('GPU ')),None)
    if row is None or float(row.split()[2].rstrip('h'))<1:
        raise RuntimeError('Insufficient or unverified quota for the next bounded job')
    # Exclusive creation records intent before the external side effect.
    with marker.open('x') as stream:
        json.dump({'status':'DISPATCHING','slug':slug,'version':version,'time_unix':time.time()},stream)
    output=command(['kaggle','kernels','push','-p',str(Path(stage))],label=f'push-{Path(stage).name}')
    if 'not valid' in output.lower() or 'could not be added' in output.lower():
        raise RuntimeError(f'Kaggle rejected an input dependency; do not accept the launch: {output}')
    if f'Kernel version {version} successfully pushed' not in output or slug not in output:
        raise RuntimeError(f'Unexpected push receipt; inspect saved log: {output}')
    marker.write_text(json.dumps({'status':'CONFIRMED','slug':slug,'version':version,'receipt':output},indent=2))
    event('launched',job=slug,version=version)

def download(slug,version,folder,exclude_preflight=False):
    args=['kaggle','kernels','output',slug+f'/{version}','-p',str(Path(folder)),'--page-size','200']
    if exclude_preflight:
        args.extend(['--file-pattern',r'^(?!preflight/).*'])
    command(args,label='download-'+Path(folder).name)
    event('downloaded',job=slug,folder=folder)

def main():
    event('started')
    if '--evaluations-only' not in sys.argv:
        wait_complete(REPAIR)
        command([PY,'scripts/stage_benchmark_kernel.py','--capacity-repair'],label='stage-benchmark-v2')
        push_once('staging/kaggle-benchmark',BENCH,2)
        download(REPAIR,1,'experiments/kaggle-locked-repair-v1',True)
        command([PY,'scripts/analyze.py'],label='verify-repaired-runs')
        command([PY,'scripts/probe_pilot_checkpoints.py','--runs-root','experiments/kaggle-locked-repair-v1/runs'],label='trained-cpu-probes')
        wait_complete(BENCH)
        download(BENCH,2,'experiments/kaggle-benchmark-v2')
        command([PY,'scripts/stage_evaluation_kernel.py','--campaign','locked'],label='stage-evaluation')
        command([PY,'scripts/stage_diagnostics_kernel.py','--campaign','locked'],label='stage-diagnostics')
        push_once('staging/kaggle-locked-evaluation',EVAL,1)
        push_once('staging/kaggle-locked-diagnostics',DIAG,1)
    wait_complete(EVAL)
    verification_version = 2 if '--verification-v2' in sys.argv else 1
    download(EVAL,verification_version,f'experiments/kaggle-locked-evaluation-v{verification_version}')
    wait_complete(DIAG)
    download(DIAG,verification_version,f'experiments/kaggle-locked-diagnostics-v{verification_version}')
    command(['kaggle','quota'],label='quota-after-campaign')
    event('ready-for-final-review')

if __name__=='__main__':
    try:
        main()
    except Exception as error:
        event('STOPPED_ERROR',error=str(error))
        raise
