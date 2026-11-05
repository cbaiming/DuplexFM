"""Exact first-order arc-cosine kernel plus a shared linear TD24 component."""
import time
import numpy as np
from scipy.linalg import cho_factor,cho_solve
ALPHA=100.
AUX_TRACE=16.
MAX_TRAIN_ROWS=2000


def checked(z):
    z=np.asarray(z,dtype=float)
    if z.ndim!=2 or z.shape[1] not in (24,40) or not np.isfinite(z).all():raise ValueError('Finite TD24 or TD24+PCA16')
    return z


def embedding(z):
    z=checked(z);td=z[:,:24]/np.sqrt(24.)
    return td if z.shape[1]==24 else np.c_[td,z[:,24:]/4.]


def arc(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if x.ndim!=2 or y.ndim!=2 or x.shape[1]!=y.shape[1] or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite matching kernel dimensions')
    norms=np.sqrt(np.sum(x*x,axis=1))[:,None]*np.sqrt(np.sum(y*y,axis=1))[None,:]
    dots=x@y.T;cosine=np.divide(dots,norms,out=np.zeros_like(dots),where=norms>0)
    angle=np.arccos(np.clip(cosine,-1.,1.))
    return norms*(np.sin(angle)+(np.pi-angle)*np.cos(angle))/np.pi


def ridge_from_gram(gram,y):
    y=np.asarray(y,float);gram=np.asarray(gram,float)
    if gram.shape!=(len(y),len(y)) or not np.isfinite(gram).all() or not np.isfinite(y).all():raise ValueError('Kernel/response contract')
    center=float(y.mean());yc=y-center
    dual=cho_solve(cho_factor(gram+ALPHA*np.eye(len(y)),lower=True),yc)
    error=float(abs((gram+ALPHA*np.eye(len(y)))@dual-yc).max())
    limit=float(1e-9*(1+abs(yc).max()))
    return dual,center,error,limit


def fit(z,y,deadline=None):
    z=checked(z);y=np.asarray(y,float)
    if not 20<=len(z)<=MAX_TRAIN_ROWS or y.shape!=(len(z),) or not np.isfinite(y).all() or np.ptp(y)==0:raise ValueError('Training contract')
    if abs(z.mean(0)).max()>1e-8:raise ValueError('Training inputs must be centered')
    if deadline is not None and time.monotonic()>deadline:raise TimeoutError('CPU deadline')
    u=embedding(z);raw=arc(u,u);raw=(raw+raw.T)*.5
    np.fill_diagonal(raw,np.sum(u*u,axis=1))
    rowmean=raw.mean(1);grand=float(rowmean.mean());centered=raw-rowmean[:,None]-rowmean[None,:]+grand
    variance=float(np.trace(centered)/len(z));scale=AUX_TRACE/variance if variance>1e-12 else 0.
    gram=z[:,:24]@z[:,:24].T+scale*centered;gram=(gram+gram.T)*.5
    if deadline is not None and time.monotonic()>deadline:raise TimeoutError('CPU deadline')
    dual,center,error,limit=ridge_from_gram(gram,y)
    residual=gram@dual+center-y;regnorm=float(dual@gram@dual)
    objective=float(.5*(residual@residual+ALPHA*regnorm))
    return dict(train_u=u.copy(),train_td=z[:,:24].copy(),dual=dual,td_coef=z[:,:24].T@dual,
        intercept=center,raw_row_mean=rowmean,raw_grand_mean=grand,aux_scale=scale,aux_variance=variance,
        normal_error=error,normal_limit=limit,numerically_qualified=bool(error<=limit),
        centered_kernel_row_mean_max=float(abs(centered.mean(1)).max()),
        mean_aux_diagonal=float(scale*variance),sum_dual=float(dual.sum()),
        train_MSE=float(np.mean(residual**2)),RKHS_squared_norm=regnorm,objective=objective,alpha=ALPHA,
        training_rows=len(z),embedding_dim=u.shape[1],new_linear_solves=1)


def predict(z,m):
    z=checked(z);u=embedding(z)
    if u.shape[1]!=m['train_u'].shape[1]:raise ValueError('Matching source/TD interface')
    raw=arc(u,m['train_u']);centered=raw-raw.mean(1)[:,None]-m['raw_row_mean'][None,:]+m['raw_grand_mean']
    return z[:,:24]@m['td_coef']+m['aux_scale']*centered@m['dual']+m['intercept']


def qualify(records,numeric):
    core=[r for r in records if r['source']=='core_real']
    if len(core)!=3:raise ValueError('Three primary pools')
    performance=sum(r['delta_PCC_vs_R4']>=.005 and r['MSE_reduction_vs_R4_pct']>=0 for r in core)>=2 and all(r['delta_PCC_vs_R4']>=-.005 for r in core)
    pairs=[]
    for r in core:
        td=next(t for t in records if t['setting']==r['setting'] and t['source']=='TD_only')
        pairs.append(dict(setting=r['setting'],delta_PCC=r['metrics']['PCC']-td['metrics']['PCC'],
            MSE_reduction_pct=100*(1-r['metrics']['MSE']/td['metrics']['MSE'])))
    incremental=sum(r['delta_PCC']>=0 for r in pairs)>=2 and np.mean([r['MSE_reduction_pct'] for r in pairs])>=0
    return dict(qualified=bool(performance and incremental and numeric),performance_gate=bool(performance),
        source_increment_gate=bool(incremental),all_numerically_qualified=bool(numeric),paired_kernel_TD=pairs)
