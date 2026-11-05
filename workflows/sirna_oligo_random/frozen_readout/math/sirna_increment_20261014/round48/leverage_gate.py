"""Input-only ridge geometry radius clipping; no predictive uncertainty claim."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_r48_native25',R.parent/'round25/native_trace.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
transform=native.transform
QUANTILE=.9;METRIC_RIDGE=100.


def data(x):
    x=np.asarray(x,float)
    if x.ndim!=2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():raise ValueError('Nonempty finite matrix required')
    return x


def leverage(x,precision):
    x=data(x);p=np.asarray(precision,float)
    if p.shape!=(x.shape[1],x.shape[1]) or not np.isfinite(p).all():raise ValueError('Finite matching precision required')
    h=np.einsum('ni,ni->n',x,x@p)
    if h.min()<-1e-10:raise ValueError('Negative metric energy')
    return np.maximum(h,0.)


def weights(h,threshold):
    h=np.asarray(h,float)
    if h.ndim!=1 or not np.isfinite(h).all() or np.any(h<0) or not np.isfinite(threshold) or threshold<0:raise ValueError('Nonnegative finite energies required')
    w=np.ones(len(h));outside=h>threshold
    w[outside]=np.sqrt(threshold/h[outside])
    return w


def fit_gate(x):
    x=data(x);n,p=x.shape
    gram=x.T@x+METRIC_RIDGE*np.eye(p)
    precision=cho_solve(cho_factor(gram,lower=True),np.eye(p));precision=(precision+precision.T)/2
    h=leverage(x,precision);index=int(np.ceil(QUANTILE*(n-1)));threshold=float(np.sort(h)[index])
    w=weights(h,threshold);mean=(w[:,None]*x).mean(0)
    return dict(precision=precision,mean=mean,threshold=threshold,quantile=QUANTILE,quantile_method='higher',quantile_index=index,
        metric_ridge=METRIC_RIDGE,training_n=n,feature_dim=p,training_leverage_sum=float(h.sum()),
        training_leverage_min=float(h.min()),training_leverage_max=float(h.max()),training_clipped_count=int(np.count_nonzero(w<1)),
        training_weight_min=float(w.min()),training_weight_mean=float(w.mean()),input_only=True,
        calibrated_predictive_uncertainty=False)


def apply_gate(x,gate):
    x=data(x)
    if x.shape[1]!=gate['feature_dim']:raise ValueError('Feature dimension mismatch')
    h=leverage(x,gate['precision']);w=weights(h,gate['threshold'])
    mean=np.asarray(gate['mean'],float)
    if mean.shape!=(x.shape[1],) or not np.isfinite(mean).all():raise ValueError('Finite stored training mean required')
    return w[:,None]*x-mean,h,w


def apply_design(z,gate,source):
    z=data(z)
    if z.shape[1]!=216:raise ValueError('Native216 input required')
    out,h,w=apply_gate(z[:,24:] if source else z[:,:24],gate)
    return (np.c_[z[:,:24],out] if source else out),h,w


def fit_head(z,y,gate,source,deadline=None):
    zz,_,_=apply_design(z,gate,source);y=np.asarray(y,float)
    if y.shape!=(len(zz),) or not np.isfinite(y).all() or np.ptp(y)==0:raise ValueError('Finite nonconstant response required')
    if np.max(abs(zz.mean(0)))>1e-8:raise ValueError('Training-centered design required')
    if source:return native.fit(zz,y,deadline=deadline)
    center=float(y.mean());g=zz.T@zz+100*np.eye(24);rhs=zz.T@(y-center)
    beta=cho_solve(cho_factor(g,lower=True),rhs);error=float(np.max(abs(g@beta-rhs)))
    threshold=1e-8*(1+float(np.max(abs(rhs))))
    return beta,center,dict(mode='TD_leverage_gated_Ridge',alpha=100.,normal_max=error,normal_threshold=threshold,
        train_MSE=float(np.mean((zz@beta+center-y)**2)),certified=bool(error<=threshold))


def qualify(records,numeric):
    core=[r for r in records if r['source']=='core_real']
    if len(core)!=3 or len({r['setting'] for r in core})!=3:raise ValueError('Three primary pools required')
    perf=sum(r['delta_PCC_vs_reference']>=.005 and r['MSE_reduction_vs_reference_pct']>=0 for r in core)>=2 and all(r['delta_PCC_vs_reference']>=-.005 for r in core)
    pairs=[]
    for r in core:
        td=next(a for a in records if a['setting']==r['setting'] and a['source']=='TD_only')
        pairs.append(dict(setting=r['setting'],delta_PCC=r['metrics']['PCC']-td['metrics']['PCC'],
            MSE_reduction_pct=100*(1-r['metrics']['MSE']/td['metrics']['MSE']),
            delta_gain_above_TD_gate=r['delta_PCC_vs_reference']-td['delta_PCC_vs_reference']))
    incr=sum(p['delta_PCC']>=0 for p in pairs)>=2 and np.mean([p['MSE_reduction_pct'] for p in pairs])>=0
    return dict(qualified=bool(perf and incr and numeric),performance_gate=bool(perf),
        source_increment_vs_gated_TD=bool(incr),all_numerically_qualified=bool(numeric),paired_gated_TD=pairs)
