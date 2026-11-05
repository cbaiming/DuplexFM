"""One joint native trace-norm head with a fixed TD coefficient trust term."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
R=Path(__file__).resolve().parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


prior=load('_r29_coordinate',R.parent/'round21/coordinate_sparse.py')
prox=load('_r29_nuclear',R.parent/'round17/nuclear.py')
transform=prior.transform
deployment=prior.deployment
ALPHA=100.;KAPPA=100.;RATIO=.1;SHAPE=(6,32)


def profile(z,y,td_reference,kappa=KAPPA):
    z=np.asarray(z,float);y=np.asarray(y,float);b0=np.asarray(td_reference,float)
    if z.ndim!=2 or z.shape[1]!=216 or y.shape!=(len(z),) or b0.shape!=(24,):raise ValueError('Input/anchor shape')
    if not np.isfinite(z).all() or not np.isfinite(y).all() or not np.isfinite(b0).all():raise ValueError('Finite inputs required')
    if not np.isfinite(kappa) or kappa<0 or abs(z.mean(0)).max()>1e-8 or np.ptp(y)==0:raise ValueError('Centered/nonconstant/fixed penalty contract')
    center=float(y.mean());g=z.T@z;rhs=z.T@(y-center)
    reference_error=float(abs((g[:24,:24]+ALPHA*np.eye(24))@b0-rhs[:24]).max())
    if reference_error>1e-9*(1+float(abs(rhs[:24]).max())):raise ValueError('TD reference is not the training alpha100 solution')
    factor=cho_factor(g[:24,:24]+(ALPHA+kappa)*np.eye(24),lower=True)
    coupling=cho_solve(factor,g[:24,24:])
    h=g[24:,24:]+ALPHA*np.eye(192)-g[24:,:24]@coupling;h=(h+h.T)*.5
    q=rhs[24:]-g[24:,:24]@b0
    return h,q,b0,coupling,center,reference_error


def fit(z,y,td_reference,kappa=KAPPA,ratio=RATIO,deadline=None):
    if not np.isfinite(ratio) or ratio<0:raise ValueError('Finite nonnegative trace ratio')
    h,q,b0,coupling,center,reference_error=profile(z,y,td_reference,kappa)
    critical=float(np.linalg.norm(q.reshape(SHAPE),2));lam=ratio*critical
    source,info=prox.solve(h,q,SHAPE,lam,deadline=deadline)
    beta=np.r_[b0-coupling@source,source]
    info.update(mode='TD_anchored_native_trace',alpha=ALPHA,kappa=float(kappa),lambda_ratio=float(ratio),
                lambda_critical=critical,lambda_value=lam,matrix_shape=SHAPE,TD_reference_error=reference_error,
                TD_shift_norm=float(np.linalg.norm(beta[:24]-b0)),
                anchor_penalty=float(.5*kappa*np.sum((beta[:24]-b0)**2)),
                train_MSE=float(np.mean((np.asarray(z)@beta+center-y)**2)))
    return beta,center,info


def qualify(records,numeric):
    core=[r for r in records if r['source']=='core_real']
    if len(core)!=3:raise ValueError('Three primary pools')
    performance=(sum(r['delta_PCC_vs_R25']>=.005 and r['MSE_reduction_vs_R25_pct']>=0 for r in core)>=2
                 and all(r['delta_PCC_vs_R25']>=-.005 for r in core))
    paired=[dict(setting=r['setting'],delta_PCC=r['metrics']['PCC']-r['TD_metrics']['PCC'],
                 MSE_reduction_pct=100*(1-r['metrics']['MSE']/r['TD_metrics']['MSE'])) for r in core]
    incremental=sum(r['delta_PCC']>=0 for r in paired)>=2 and np.mean([r['MSE_reduction_pct'] for r in paired])>=0
    return dict(qualified=bool(performance and incremental and numeric),performance_gate=bool(performance),
                source_increment_gate=bool(incremental),all_numerically_qualified=bool(numeric),paired_TD=paired)
