"""Fixed channel groups across the six native pooled blocks, one joint head."""
import importlib.util,time
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_r30_prior21',R.parent/'round21/coordinate_sparse.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
transform=prior.transform;deployment=prior.deployment
SHAPE=(6,32);ALPHA=100.;RATIO=.1;TOL=1e-9;MAXITER=5000


def group_norms(vector):
    x=np.asarray(vector,float)
    if x.shape!=(192,) or not np.isfinite(x).all():raise ValueError('Finite source192 required')
    return np.linalg.norm(x.reshape(SHAPE),axis=0)


def prox(vector,tau):
    if not np.isfinite(tau) or tau<0:raise ValueError('Finite nonnegative threshold')
    norms=group_norms(vector)
    scale=np.divide(np.maximum(norms-tau,0.),norms,out=np.zeros_like(norms),where=norms>0)
    return (np.asarray(vector,float).reshape(SHAPE)*scale).ravel()


def certificate(h,q,b,lam,factor):
    norms=group_norms(b);dual=q-h@b;dn=group_norms(dual)
    factors=np.minimum(1.,np.divide(lam,dn,out=np.ones_like(dn),where=dn>0))
    dual=(dual.reshape(SHAPE)*factors).ravel()
    primal=.5*float(b@h@b)-float(q@b)+lam*float(norms.sum())
    lower=-.5*float((q-dual)@cho_solve(factor,q-dual));gap=primal-lower
    if gap < -1e-8:raise ArithmeticError('Negative dual gap')
    return dict(primal_profile=primal,dual_profile=lower,gap=max(0.,gap),dual=dual,
        group_norms=norms,active_channels=int(sum(norms>0)),effective_channels=int(sum(norms>1e-8)))


def solve(h,q,lam,deadline=None):
    h,q=np.asarray(h,float),np.asarray(q,float)
    if h.shape!=(192,192) or q.shape!=(192,) or not np.isfinite(h).all() or not np.isfinite(q).all():raise ValueError('Profile shape/finite contract')
    if not np.isfinite(lam) or lam<0 or not np.allclose(h,h.T,atol=1e-10,rtol=1e-10):raise ValueError('Penalty/symmetry contract')
    factor=cho_factor(h,lower=True);ridge=cho_solve(factor,q);gapscale=max(1.,.5*float(q@ridge))
    critical=float(group_norms(q).max())
    if lam==0 or lam>=critical:
        beta=ridge if lam==0 else np.zeros(192)
        cert=certificate(h,q,beta,lam,factor)
        cert.update(iterations=0,certified=cert['gap']<=TOL*gapscale,gap_scale=gapscale,history=[])
        return beta,cert
    lip=float(np.linalg.eigvalsh(h)[-1]);b=np.zeros(192);look=b.copy();momentum=1.;history=[]
    for step in range(1,MAXITER+1):
        new=prox(look-(h@look-q)/lip,lam/lip)
        next_momentum=(1+np.sqrt(1+4*momentum*momentum))/2
        look=new+((momentum-1)/next_momentum)*(new-b);b=new;momentum=next_momentum
        if deadline is not None and time.monotonic()>deadline:raise TimeoutError('CPU deadline')
        if step%10==0 or step==MAXITER:
            cert=certificate(h,q,b,lam,factor);history.append([step,cert['gap']])
            if cert['gap']<=TOL*gapscale:break
    cert.update(iterations=step,certified=bool(cert['gap']<=TOL*gapscale),gap_scale=gapscale,L=lip,history=history)
    return b,cert


def fit(z,y,ratio=RATIO,deadline=None):
    if not np.isfinite(ratio) or ratio<0:raise ValueError('Finite nonnegative ratio')
    h,q,td0,coupling,center=prior.profile(z,y)
    critical=float(group_norms(q).max());lam=ratio*critical
    source,cert=solve(h,q,lam,deadline);beta=np.r_[td0-coupling@source,source]
    cert.update(mode='native_channel_group',alpha=ALPHA,lambda_ratio=float(ratio),lambda_critical=critical,
        lambda_value=lam,matrix_shape=SHAPE,group_axis='columns',group_size=6,group_count=32,
        TD_shift_norm=float(np.linalg.norm(beta[:24]-td0)),
        train_MSE=float(np.mean((np.asarray(z)@beta+center-y)**2)))
    return beta,center,cert


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
