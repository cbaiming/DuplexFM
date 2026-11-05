"""Trace-norm shrinkage of the native six-pool by 32-channel coefficients."""
import importlib.util
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
prior=load('_r25_coordinate',R.parent/'round21/coordinate_sparse.py')
prox=load('_r25_nuclear',R.parent/'round17/nuclear.py')
transform=prior.transform
deployment=prior.deployment
SHAPE=(6,32)
RATIO=.1


def fit(z,y,ratio=RATIO,deadline=None):
    if not np.isfinite(ratio) or ratio<0:raise ValueError('Nonnegative finite fixed ratio')
    h,q,td0,coupling,center=prior.profile(z,y)
    critical=float(np.linalg.norm(q.reshape(SHAPE),2));lam=ratio*critical
    source,info=prox.solve(h,q,SHAPE,lam,deadline=deadline)
    beta=np.r_[td0-coupling@source,source]
    info.update(mode='native_trace',alpha=100.,lambda_ratio=float(ratio),lambda_critical=critical,
                lambda_value=lam,matrix_shape=SHAPE,train_MSE=float(np.mean((z@beta+center-y)**2)))
    return beta,center,info


def qualify(records,all_numeric):
    core=[r for r in records if r['source']=='core_real']
    if len(core)!=3:raise ValueError('Three primary pools required')
    performance=(sum(r['delta_PCC_vs_R4']>=.005 and r['MSE_reduction_vs_R4_pct']>=0 for r in core)>=2
                 and all(r['delta_PCC_vs_R4']>=-.005 for r in core))
    paired=[dict(setting=r['setting'],delta_PCC=r['metrics']['PCC']-r['dense_metrics']['PCC'],
                 MSE_reduction_pct=100*(1-r['metrics']['MSE']/r['dense_metrics']['MSE'])) for r in core]
    paired_gate=sum(r['MSE_reduction_pct']>=0 for r in paired)>=2 and np.mean([r['MSE_reduction_pct'] for r in paired])>=0
    return dict(qualified=bool(performance and paired_gate and all_numeric),performance_gate=bool(performance),
                paired_gate=bool(paired_gate),all_numerically_qualified=bool(all_numeric),paired_core_records=paired)
