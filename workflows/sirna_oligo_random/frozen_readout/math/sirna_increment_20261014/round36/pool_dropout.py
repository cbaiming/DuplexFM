"""Exact expected squared loss under six source-pool dropout masks."""
import importlib.util
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_r36_prior25',R.parent/'round25/native_trace.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
transform=prior.transform;deployment=prior.deployment
KEEP=.9;SHAPE=(6,32)


def penalty(source,keep=KEEP):
    source=np.asarray(source,float)
    if source.ndim!=2 or source.shape[1]!=192 or not np.isfinite(source).all() or not 0<keep<=1:
        raise ValueError('Finite native192 and keep in(0,1] required')
    tau=(1-keep)/keep;out=np.zeros((192,192))
    for j in range(6):
        sl=slice(j*32,(j+1)*32);out[sl,sl]=tau*(source[:,sl].T@source[:,sl])
    return out


def fit(z,y,keep=KEEP,deadline=None):
    h,q,td0,coupling,center=prior.prior.profile(z,y)
    extra=penalty(z[:,24:],keep);system=h+extra;system=(system+system.T)/2
    critical=float(np.linalg.norm(q.reshape(SHAPE),2));lam=.1*critical
    source,cert=prior.prox.solve(system,q,SHAPE,lam,deadline=deadline)
    beta=np.r_[td0-coupling@source,source]
    outputs=np.stack([z[:,24+j*32:24+(j+1)*32]@source[j*32:(j+1)*32] for j in range(6)],axis=1)
    variance=float((1-keep)/keep*np.sum(outputs**2));mse=float(np.mean((z@beta+center-y)**2))
    cert.update(mode='native_trace_pool_dropout',alpha=100.,lambda_ratio=.1,lambda_critical=critical,lambda_value=lam,
        matrix_shape=SHAPE,keep_probability=keep,dropout_penalty_trace=float(np.trace(extra)),
        train_MSE=mse,dropout_extra_SSE=variance,expected_train_MSE=mse+variance/len(y),
        block_output_RMS=np.sqrt(np.mean(outputs**2,axis=0)))
    return beta,center,cert


def qualify(records,numeric):
    core=[r for r in records if r['source']=='core_real']
    if len(core)!=3:raise ValueError('Three primary training pools required')
    performance=sum(r['delta_PCC_vs_R25']>=.005 and r['MSE_reduction_vs_R25_pct']>=0 for r in core)>=2 and all(r['delta_PCC_vs_R25']>=-.005 for r in core)
    pairs=[dict(setting=r['setting'],delta_PCC=r['metrics']['PCC']-r['TD_metrics']['PCC'],MSE_reduction_pct=100*(1-r['metrics']['MSE']/r['TD_metrics']['MSE'])) for r in core]
    increment=sum(r['delta_PCC']>=0 for r in pairs)>=2 and np.mean([r['MSE_reduction_pct'] for r in pairs])>=0
    return dict(qualified=bool(performance and increment and numeric),performance_gate=bool(performance),
        source_increment_gate=bool(increment),all_numerically_qualified=bool(numeric),paired_TD=pairs)
