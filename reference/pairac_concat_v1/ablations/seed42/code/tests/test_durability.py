import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from duplexfm_joint.io_utils import seed_all, load_checkpoint
from duplexfm_mechanism import parallel as p
from duplexfm_mechanism.common import read, sha, FileLock, backup
from duplexfm_mechanism.durable_io import durable_json as write, save_checkpoint, valid_checkpoint, recover_checkpoint
from duplexfm_mechanism.efficiency import choose, backward
from duplexfm_mechanism.training_hourly import run_engine




class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Dropout(.2), torch.nn.Linear(8, 1))
    def forward(self, batch):
        return self.fusion(batch["x"]).squeeze(-1)

class ToyStore:
    def __init__(self):
        self.x = torch.arange(60).reshape(20, 3).float() / 30
        self.y = torch.tensor([0., 1.] * 10)
    def collate(self, ids):
        return {"x": self.x[ids].clone(), "labels": self.y[ids].clone(), "row_indices": torch.as_tensor(ids)}

def tiny(step=0):
    return {'signature':{'synthetic':'checkpoint'}, 'state':{'step':step,'last_validation_step':0},
            'optimizer':{},'rng':{},'trainable_parameters':{'w':torch.tensor([float(step)])},'fusion_buffers':{}}

def test_atomic_checkpoint_previous_and_corrupt_latest(tmp_path):
    folder=tmp_path/'runs/full_real';folder.mkdir(parents=True)
    write(folder/'signature.json',tiny()['signature'])
    save_checkpoint(folder/'initial.pt',tiny())
    save_checkpoint(folder/'last.pt',tiny(1));save_checkpoint(folder/'last.pt',tiny(2))
    assert valid_checkpoint(folder/'last.previous.pt')['state']['step']==1
    assert (folder/'last.pt').stat().st_nlink==1
    (folder/'last.pt').write_bytes(b'torn')
    restored=recover_checkpoint(folder)
    assert restored['step']==1 and restored['selected']=='last.previous.pt'
    assert list(folder.glob('last.pt.retained_*'))

def test_checkpoint_torn_receipt_falls_back(tmp_path):
    write(tmp_path/'signature.json',tiny()['signature'])
    save_checkpoint(tmp_path/'initial.pt',tiny())
    save_checkpoint(tmp_path/'last.pt',tiny(1));save_checkpoint(tmp_path/'last.pt',tiny(2))
    (tmp_path/'last.commit.json').write_text('{')
    assert recover_checkpoint(tmp_path)['step']==1

def test_hourly_independent_copy_recovery(tmp_path):
    folder=tmp_path/'runs/full_real';folder.mkdir(parents=True)
    write(folder/'signature.json',tiny()['signature'])
    save_checkpoint(folder/'initial.pt',tiny());save_checkpoint(folder/'last.pt',tiny(3))
    receipt=backup(tmp_path,list(folder.glob('*')),'hour')
    write(folder/'latest_backup.json',receipt)
    (folder/'last.pt').write_bytes(b'torn')
    assert recover_checkpoint(folder)['step']==3

def test_checkpoint_signature_and_no_verified_fallback_rejected(tmp_path):
    write(tmp_path/'signature.json',{'different':True})
    save_checkpoint(tmp_path/'initial.pt',tiny());save_checkpoint(tmp_path/'last.pt',tiny(1))
    with pytest.raises(ValueError):recover_checkpoint(tmp_path)

@pytest.mark.parametrize('micro',[1,2,3,8,512])
def test_weighted_micro_gradient_global_and_tail(micro):
    seed_all(42)
    model=ToyModel(); model.eval()
    data=ToyStore().collate(np.arange(5)); labels=data['labels']
    loss=torch.nn.functional.binary_cross_entropy_with_logits(model(data),labels)
    loss.backward();expected={n:v.grad.clone() for n,v in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    total=backward(model,data,labels,micro,torch.device('cpu'),'fp32')
    assert torch.allclose(loss.detach(),total,atol=1e-7)
    for n,v in model.named_parameters():assert torch.allclose(v.grad,expected[n],atol=1e-7,rtol=1e-6),n

@pytest.mark.parametrize('stop_step',[1,2,3,5])
def test_hourly_engine_native_exact_resume(tmp_path,stop_step):
    opts={'reference_rows':8,'batch_size':3,'eval_batch_size':2,'max_updates':6,'seed':42,'amp':'fp32',
          'checkpoint_steps':2,'backup_seconds':3600,'micro_batch_size':2}
    fit,val=np.arange(8),np.arange(8,14)
    seed_all(42);run_engine(ToyModel(),ToyStore(),fit,val,tmp_path/'a',opts,{'test':'hourly'})
    seed_all(42);r=run_engine(ToyModel(),ToyStore(),fit,val,tmp_path/'b',opts,{'test':'hourly'},stop_after_step=stop_step)
    assert r['status']=='interrupted'
    recover_checkpoint(tmp_path/'b')
    seed_all(999);run_engine(ToyModel(),ToyStore(),fit,val,tmp_path/'b',opts,{'test':'hourly'},resume=True)
    aa,bb=load_checkpoint(tmp_path/'a/last.pt'),load_checkpoint(tmp_path/'b/last.pt')
    for key in aa['trainable_parameters']:assert torch.equal(aa['trainable_parameters'][key],bb['trainable_parameters'][key])
    assert torch.equal(aa['rng']['torch'],bb['rng']['torch'])
    assert aa['state']['history']==bb['state']['history']
    for key,items in aa['optimizer']['state'].items():
        for n,value in items.items():
            other=bb['optimizer']['state'][key][n]
            assert torch.equal(value,other) if isinstance(value,torch.Tensor) else value==other

def test_snapshot_during_validation_preserves_training_rng(tmp_path):
    opts={'reference_rows':8,'batch_size':3,'eval_batch_size':1,'max_updates':6,'seed':42,'amp':'fp32',
          'checkpoint_steps':2,'backup_seconds':0,'micro_batch_size':2}
    fit,val=np.arange(8),np.arange(8,14)
    seed_all(42);run_engine(ToyModel(),ToyStore(),fit,val,tmp_path/'runs/a',opts,{'synthetic':True},backup_root=tmp_path)
    seed_all(42);run_engine(ToyModel(),ToyStore(),fit,val,tmp_path/'runs/b',opts,{'synthetic':True})
    aa,bb=load_checkpoint(tmp_path/'runs/a/last.pt'),load_checkpoint(tmp_path/'runs/b/last.pt')
    assert aa['state']['history']==bb['state']['history']
    for n in aa['trainable_parameters']:assert torch.equal(aa['trainable_parameters'][n],bb['trainable_parameters'][n])

def inventory():
    return [(str(i),f'GPU-{i}','NVIDIA H100 80GB','0','81920','0') for i in range(4)],[]

def test_two_unique_idle_only():
    assert len(p.select_cards('0,1',inventory()))==2
    for names in ['0','0,0','0,1,2','0,GPU-0']:
        with pytest.raises(ValueError):p.select_cards(names,inventory())
    g,proc=inventory();proc=[('GPU-1','1234','400')]
    with pytest.raises(RuntimeError):p.select_cards('0,1',(g,proc))

def ledger():
    return {'version':1,'active':None,'sessions':[], 'experiment_limit_gpu_seconds':460800,'platform_limit_gpu_seconds':460800}
