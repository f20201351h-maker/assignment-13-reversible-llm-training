"""Measure unique saved storage for the reversible stack, excluding parameters."""
import json
from pathlib import Path
import torch
from revllm.model import ModelConfig, TinyGPT

ROOT=Path(__file__).resolve().parents[1]
records=[]
for integrator in ('midpoint','coupled_euler'):
    for depth in (2,4,9,18):
        for backward in ('stored','reconstructed'):
            torch.manual_seed(314159)
            cfg=ModelConfig(vocab_size=80,block_size=16,n_layer=depth,n_head=3,n_embd=48,
                            integrator=integrator,backward_mode=backward,step_size=0.5)
            model=TinyGPT(cfg)
            parameters={p.untyped_storage().data_ptr() for p in model.parameters()}
            storages={}
            def pack(tensor):
                storage=tensor.untyped_storage()
                if storage.data_ptr() not in parameters:
                    storages[storage.data_ptr()]=storage.nbytes()
                return tensor
            x=torch.randn(2,16,48,requires_grad=True)
            with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
                loss=model.stack(x).square().sum()
            records.append({'integrator':integrator,'depth':depth,'backward_mode':backward,
                            'saved_unique_storage_bytes':sum(storages.values()),
                            'saved_unique_storages':len(storages),'dtype':'float32','device':'cpu',
                            'input_shape':[2,16,48],
                            'scope':'stack forward plus squared-output reduction; unique retained storage, parameter aliases excluded; not CUDA peak memory'})
            del loss, model, x
for kind in ('midpoint','coupled_euler'):
    rec=[r['saved_unique_storage_bytes'] for r in records if r['integrator']==kind and r['backward_mode']=='reconstructed']
    stored=[r['saved_unique_storage_bytes'] for r in records if r['integrator']==kind and r['backward_mode']=='stored']
    assert max(rec)==min(rec), (kind,rec)
    assert all(a<b for a,b in zip(stored,stored[1:])), (kind,stored)
out=ROOT/'experiments/storage-depth-v1/measurements.json'
out.parent.mkdir(parents=True,exist_ok=True)
if out.exists() and json.loads(out.read_text())!=records:
    raise RuntimeError('Answered diagnostic differs; retain it and use a new experiment ID')
out.write_text(json.dumps(records,indent=2))
print(json.dumps({'measurements':len(records),'depth_constant_reconstructed_storage':'PASS','output':str(out)}))
