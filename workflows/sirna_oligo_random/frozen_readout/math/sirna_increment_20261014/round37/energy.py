"""Add verified matched energies to the existing native trace-norm head."""
import importlib.util
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_r37_trace',R.parent/'round25/native_trace.py')
prior=importlib.util.module_from_spec(s);s.loader.exec_module(prior)


def check_energy(e):
    e=np.asarray(e,float)
    if e.ndim!=2 or e.shape[1]!=4 or not np.isfinite(e).all() or not np.isin(e[:,3],[0.,1.]).all():
        raise ValueError('Finite three energies plus binary validity required')
    return e


def fit_energy(e):
    e=check_energy(e)
    if not len(e):raise ValueError('Empty training energy')
    valid=e[:,3]==1;count=int(valid.sum())
    if count:
        mean=e[valid,:3].mean(0);sd=e[valid,:3].std(0);scale=np.where(sd>=1e-12,sd,1.)
    else:mean=np.zeros(3);scale=np.ones(3)
    flagmean=float(valid.mean());flagsd=float(valid.std())
    return dict(enabled=count>0,valid_training_rows=count,mean=mean,scale=scale,
        flag_mean=flagmean,flag_scale=flagsd if flagsd>=1e-12 else 1.,flag_enabled=flagsd>=1e-12)


def map_energy(e,t):
    e=check_energy(e);z=np.zeros_like(e)
    if not t['enabled']:return z
    valid=e[:,3]==1;z[valid,:3]=(e[valid,:3]-t['mean'])/t['scale']
    if t['flag_enabled']:z[:,3]=(e[:,3]-t['flag_mean'])/t['flag_scale']
    return z


def canonical(raw,e):
    raw=np.asarray(raw,float);e=check_energy(e)
    if raw.ndim!=2 or raw.shape!=(len(e),216) or not np.isfinite(raw).all():raise ValueError('TD24+source192 identity')
    return np.c_[raw[:,:24],e[:,:3]*e[:,3,None],e[:,3],raw[:,24:]]


def transform(raw,e,oldstate,state):
    z=prior.transform(raw,oldstate)
    if len(z)!=len(e):raise ValueError('Row alignment')
    return np.c_[z[:,:24],map_energy(e,state),z[:,24:]]


def fit(z,y,ratio=.1,deadline=None):
    z,y=np.asarray(z,float),np.asarray(y,float)
    if z.ndim!=2 or z.shape[1] not in (28,220) or y.shape!=(len(z),):raise ValueError('Shape')
    if not np.isfinite(z).all() or not np.isfinite(y).all() or np.ptp(y)==0 or abs(z.mean(0)).max()>1e-8:raise ValueError('Finite centered design and nonconstant y')
    if not np.isfinite(ratio) or ratio<0:raise ValueError('Finite nonnegative ratio')
    center=float(y.mean());g=z.T@z+100*np.eye(z.shape[1]);rhs=z.T@(y-center)
    if z.shape[1]==28:
        beta=np.linalg.solve(g,rhs)
        cert=dict(mode='TD_energy28',alpha=100.,normal_error=float(abs(g@beta-rhs).max()),certified=True)
    else:
        tt=g[:28,:28];stat0=np.linalg.solve(tt,rhs[:28]);coupling=np.linalg.solve(tt,g[:28,28:])
        h=g[28:,28:]-g[28:,:28]@coupling;h=(h+h.T)/2;q=rhs[28:]-g[28:,:28]@stat0
        critical=float(np.linalg.norm(q.reshape(6,32),2));lam=ratio*critical
        b,cert=prior.prox.solve(h,q,(6,32),lam,deadline=deadline);beta=np.r_[stat0-coupling@b,b]
        cert.update(mode='native_trace_energy',alpha=100.,lambda_ratio=ratio,lambda_critical=critical,lambda_value=lam,matrix_shape=(6,32))
    cert['train_MSE']=float(np.mean((z@beta+center-y)**2))
    return beta,center,cert


def deployment(oldstate,state,beta,center):
    beta=np.asarray(beta,float)
    if beta.shape not in ((28,),(220,)):raise ValueError('Coefficient shape')
    coef=np.zeros_like(beta);coef[:24]=beta[:24]/oldstate['td_scale']
    offset=float(center-np.asarray(oldstate['td_mean'])@coef[:24])
    if state['enabled']:
        coef[24:27]=beta[24:27]/state['scale'];coef[27]=-np.asarray(state['mean'])@coef[24:27]
        if state['flag_enabled']:
            coef[27]+=beta[27]/state['flag_scale'];offset-=state['flag_mean']*beta[27]/state['flag_scale']
    if len(beta)==220:
        coef[28:]=beta[28:]/np.asarray(oldstate['src_scale'])/np.sqrt(12.)
        offset-=np.asarray(oldstate['src_mean'])@coef[28:]
    return coef,float(offset)


def qualify(records,numeric):
    core=[r for r in records if r['source']=='core_real']
    if len(core)!=3:raise ValueError('Three primary pools required')
    performance=sum(r['delta_PCC_vs_R25']>=.005 and r['MSE_reduction_vs_R25_pct']>=0 for r in core)>=2 and all(r['delta_PCC_vs_R25']>=-.005 for r in core)
    pairs=[dict(setting=r['setting'],delta_PCC=r['metrics']['PCC']-r['TD28_metrics']['PCC'],
        MSE_reduction_pct=100*(1-r['metrics']['MSE']/r['TD28_metrics']['MSE'])) for r in core]
    increment=sum(r['delta_PCC']>=0 for r in pairs)>=2 and np.mean([r['MSE_reduction_pct'] for r in pairs])>=0
    return dict(qualified=bool(performance and increment and numeric),performance_gate=bool(performance),source_increment_gate=bool(increment),all_numerically_qualified=bool(numeric),paired_TD28=pairs)
