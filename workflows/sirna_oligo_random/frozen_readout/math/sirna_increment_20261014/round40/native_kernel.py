"""Uncompressed source coordinates for the fixed R33 arc-cosine kernel."""
import importlib.util,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_r40_kernel33',R.parent/'round33/relu_kernel.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
arc=base.arc;ALPHA=100.;AUX_TRACE=16.


def transform(raw,state):
    raw=np.asarray(raw,float)
    if raw.ndim!=2 or raw.shape[1]!=216 or not np.isfinite(raw).all():raise ValueError('Finite TD24 plus native192 required')
    return np.c_[(raw[:,:24]-state['td_mean'])/state['td_scale'],
                 (raw[:,24:]-state['src_mean'])/state['src_scale']]


def checked(z):
    z=np.asarray(z,float)
    if z.ndim!=2 or z.shape[1]!=216 or not np.isfinite(z).all():raise ValueError('Finite 216-dimensional native inputs required')
    return z


def embedding(z):
    z=checked(z)
    return np.c_[z[:,:24]/np.sqrt(24.),z[:,24:]/np.sqrt(192.)]


def fit(z,y,deadline=None):
    z=checked(z);y=np.asarray(y,float)
    if not 20<=len(z)<=2000 or y.shape!=(len(z),) or not np.isfinite(y).all() or np.ptp(y)==0:raise ValueError('Training contract')
    if abs(z.mean(0)).max()>1e-8:raise ValueError('Training inputs must be centered')
    if deadline is not None and time.monotonic()>deadline:raise TimeoutError('CPU deadline')
    u=embedding(z);raw=arc(u,u);raw=(raw+raw.T)*.5;np.fill_diagonal(raw,np.sum(u*u,axis=1))
    rowmean=raw.mean(1);grand=float(rowmean.mean());centered=raw-rowmean[:,None]-rowmean[None,:]+grand
    variance=float(np.trace(centered)/len(z));scale=AUX_TRACE/variance if variance>1e-12 else 0.
    gram=z[:,:24]@z[:,:24].T+scale*centered;gram=(gram+gram.T)*.5
    if deadline is not None and time.monotonic()>deadline:raise TimeoutError('CPU deadline')
    dual,center,error,limit=base.ridge_from_gram(gram,y)
    residual=gram@dual+center-y;norm=float(dual@gram@dual)
    return dict(train_u=u.copy(),train_td=z[:,:24].copy(),dual=dual,td_coef=z[:,:24].T@dual,
        intercept=center,raw_row_mean=rowmean,raw_grand_mean=grand,aux_scale=scale,aux_variance=variance,
        normal_error=error,normal_limit=limit,numerically_qualified=bool(error<=limit),
        centered_kernel_row_mean_max=float(abs(centered.mean(1)).max()),mean_aux_diagonal=float(scale*variance),
        sum_dual=float(dual.sum()),train_MSE=float(np.mean(residual**2)),RKHS_squared_norm=norm,
        objective=float(.5*(residual@residual+ALPHA*norm)),alpha=ALPHA,training_rows=len(z),
        embedding_dim=216,new_linear_solves=1)


def predict(z,m):
    z=checked(z);raw=arc(embedding(z),m['train_u'])
    centered=raw-raw.mean(1)[:,None]-m['raw_row_mean'][None,:]+m['raw_grand_mean']
    return z[:,:24]@m['td_coef']+m['aux_scale']*centered@m['dual']+m['intercept']


def qualify(records,numeric,td_records):
    core=[r for r in records if r['source']=='core_real']
    if len(core)!=3 or len(td_records)!=3:raise ValueError('Three primary and unchanged TD pools required')
    performance=sum(r['delta_PCC_vs_R4']>=.005 and r['MSE_reduction_vs_R4_pct']>=0 for r in core)>=2 and all(r['delta_PCC_vs_R4']>=-.005 for r in core)
    compression=sum(r['delta_PCC_vs_R33']>=0 and r['MSE_reduction_vs_R33_pct']>=0 for r in core)>=2 and all(r['delta_PCC_vs_R33']>=-.005 for r in core)
    native=sum(r['MSE_reduction_vs_R25_pct']>=0 for r in core)>=2 and np.mean([r['MSE_reduction_vs_R25_pct'] for r in core])>=0 and all(r['delta_PCC_vs_R25']>=-.005 for r in core)
    pairs=[]
    for r in core:
        td=next(t for t in td_records if t['setting']==r['setting'])
        pairs.append(dict(setting=r['setting'],delta_PCC=r['metrics']['PCC']-td['metrics']['PCC'],
            MSE_reduction_pct=100*(1-r['metrics']['MSE']/td['metrics']['MSE'])))
    inc=sum(r['delta_PCC']>=0 for r in pairs)>=2 and np.mean([r['MSE_reduction_pct'] for r in pairs])>=0
    return dict(qualified=bool(performance and compression and native and inc and numeric),performance_gate=bool(performance),
        compression_pair_gate=bool(compression),native_reference_gate=bool(native),source_increment_gate=bool(inc),
        all_numerically_qualified=bool(numeric),paired_kernel_TD=pairs)
