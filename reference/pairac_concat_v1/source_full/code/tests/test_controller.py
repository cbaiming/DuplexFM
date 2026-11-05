import contextlib
from pathlib import Path
import pytest
from duplexfm_mechanism import parallel as p
from duplexfm_mechanism import efficiency as e
from duplexfm_mechanism.common import read
from duplexfm_mechanism.durable_io import durable_json as write

def test_three_seeds_then_evaluation_and_repeat_is_noop(tmp_path,monkeypatch):
    write(tmp_path/'parallel_ledger.json',{'active':None,'sessions':[],
        'experiment_limit_gpu_seconds':128*3600,'platform_limit_gpu_seconds':128*3600})
    write(tmp_path/'registration.json',{'synthetic':True})
    write(tmp_path/'parallel_execution.json',p.EXECUTION)
    write(tmp_path/'parallel_registration.json',{'synthetic':True})
    completed=set();launched=[]
    gpus=[(str(i),f'GPU-{i}','NVIDIA H100 80GB','0','81920','0') for i in (0,1)]
    cards=[{'index':i,'uuid':f'GPU-{i}','name':'NVIDIA H100 80GB','memory_mib':81920} for i in (0,1)]
    monkeypatch.setattr(p,'FileLock',lambda *_:contextlib.nullcontext())
    monkeypatch.setattr(p,'verify',lambda *_:{})
    monkeypatch.setattr(p,'check_disk',lambda *_:None)
    monkeypatch.setattr(p,'gpu_inventory',lambda:(gpus,[]))
    monkeypatch.setattr(p,'dispatch_gpu',lambda uuid,*_:cards[int(uuid[-1])])
    monkeypatch.setattr(p,'process_identity',lambda pid:{'pid':pid})
    monkeypatch.setattr(p,'backup',lambda *_:{})
    monkeypatch.setattr(p,'check_complete',lambda _,v:v in completed)
    monkeypatch.setattr(e,'load_plan',lambda *_:{'cpu_threads':1,'hardware_name':'NVIDIA H100 80GB','memory_mib':81920})
    class Child:
        def __init__(self,args,**kwargs):
            self.lease=read(args[-1]);self.pid=100+len(launched);self.done=False
            self.variant=self.lease['variant'];launched.append(self.variant)
            assert kwargs['env']['CUDA_VISIBLE_DEVICES']==self.lease['gpu']['uuid']
            if self.variant=='evaluation':assert set(p.VARIANTS)<=completed
        def poll(self):return 0 if self.done else None
        def wait(self,timeout=None):
            self.done=True
            if self.variant=='efficiency':write(tmp_path/'calibration/complete.json',{'synthetic':True})
            else:completed.add(self.variant)
            return 0
    monkeypatch.setattr(p.subprocess,'Popen',Child)
    result=p.controller(tmp_path,'/synthetic')
    assert result['status']=='source_training_complete'
    assert launched==['efficiency','seed42','seed43','seed44','evaluation']
    ledger=read(tmp_path/'parallel_ledger.json')
    assert ledger['active'] is None and len(ledger['sessions'])==1
    assert ledger['sessions'][0]['allocation_gpu_seconds']>=0
    again=p.controller(tmp_path,'/synthetic')
    assert again['status']=='source_training_complete' and len(launched)==5

def test_live_ledger_and_wrong_gpu_rejected():
    with pytest.raises(RuntimeError):p.charged_seconds({'active':{'id':'x'}})
    g=[('0','GPU-a','NVIDIA GeForce RTX 5090','0','32768','0'),('1','GPU-b','NVIDIA H100','0','81920','0')]
    with pytest.raises(ValueError):p.select_cards('0,1',(g,[]))
