"""One bounded transport-repair dispatch; never retries a notebook launch."""
import json, subprocess, sys, time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
from finish_campaign import ROOT, OUT, EVAL, DIAG, event, push_once
api=KaggleApi(); api.authenticate()
asset='KAGGLE_USER/revllm-session13-locked-checkpoints'
repair=ROOT/'experiments/verification-input-repair'
try:
    event('waiting-checkpoint-asset')
    deadline=time.monotonic()+1800
    while True:
        try: state=json.loads(api.dataset_status(asset,format='json'))
        except Exception: state={'status':'upload-not-visible'}
        if state['status']=='ready': break
        if state['status'] in ('error','failed'): raise RuntimeError(state)
        if time.monotonic()>deadline: raise TimeoutError('Checkpoint asset not ready within 30 minutes')
        time.sleep(45)
    (repair/'dataset-status.json').write_text(json.dumps(state,indent=2))
    meta_path=api.dataset_metadata(asset,str(repair/'dataset-metadata'))
    metadata=json.loads(Path(meta_path).read_text()); info=metadata.get('info',metadata)
    assert info.get('isPrivate') is True, metadata
    for slug,stage in ((EVAL,'staging/kaggle-locked-evaluation'),(DIAG,'staging/kaggle-locked-diagnostics')):
        request=ApiGetKernelRequest(); request.user_name='KAGGLE_USER'; request.kernel_slug=slug.split('/')[1]
        with api.build_kaggle_client() as client:
            before=client.kernels.kernels_api_client.get_kernel(request).to_dict()['metadata']
        if before['currentVersionNumber'] != 1: raise RuntimeError('Existing version changed: inspect before any dispatch')
        status=api.kernels_status(slug).to_dict()
        if status.get('status') not in ('COMPLETE','ERROR'): raise RuntimeError(f'Existing job still active: {status}')
        (repair/(slug.split('/')[-1]+'-v1-final-status.json')).write_text(json.dumps(status,indent=2))
        push_once(stage,slug,2)
        with api.build_kaggle_client() as client:
            after=client.kernels.kernels_api_client.get_kernel(request).to_dict()['metadata']
        (repair/(slug.split('/')[-1]+'-v2-metadata.json')).write_text(json.dumps(after,indent=2))
        assert after['currentVersionNumber']==2 and after['isPrivate'] is True, after
        assert set(after['datasetDataSources'])=={asset,'KAGGLE_USER/revllm-session13-assets'}, after
        assert not after.get('kernelDataSources'), after
    event('verification-v2-inputs-confirmed')
    subprocess.run([sys.executable,str(ROOT/'scripts/finish_campaign.py'),'--evaluations-only','--verification-v2'],cwd=ROOT,check=True)
except Exception as exc:
    event('STOPPED_ERROR',error=str(exc)); raise
