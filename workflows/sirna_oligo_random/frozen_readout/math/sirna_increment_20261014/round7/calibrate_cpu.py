"""One synthetic size/timing qualification, never a biological parameter search."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
from kernel import fit_transform,fit_heads,configs,transform
out=R/'CPU_QUALIFICATION.json';assert not out.exists()
start=time.monotonic();rng=np.random.default_rng(42);x=rng.normal(size=(1600,216));y=rng.normal(size=1600);v=rng.normal(size=(320,216))
z,t=fit_transform(x,'joint');mapped=time.monotonic();b,c=fit_heads(z,y,configs());p=transform(v,t)@b.T+c
assert p.shape==(320,25) and np.isfinite(p).all()
result=dict(synthetic_only=True,training_rows=1600,prediction_rows=320,landmarks=64,head_width=z.shape[1],
    map_seconds=mapped-start,total_seconds=time.monotonic()-start,GPU_hours=0,CPU_threads=1,
    kernel_code_sha256=hashlib.sha256((R/'kernel.py').read_bytes()).hexdigest(),no_scientific_setting_change=True)
with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
