"""One convex model: native linear coefficients plus an even nonlinear RKHS."""
import importlib.util,time
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
R=Path(__file__).resolve().parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
native=load('_r41_native25',R.parent/'round25/native_trace.py')
arcmod=load('_r41_arc33',R.parent/'round33/relu_kernel.py')
transform=native.transform
ALPHA=100.;AUX_TRACE=16.;RATIO=.1


def checked(z):
    z=np.asarray(z,float)
    if z.ndim!=2 or z.shape[1] not in (24,216) or not np.isfinite(z).all():raise ValueError('Finite native216 or TD24 input')
    return z


def embed(z):
    z=checked(z)
    return z/np.sqrt(24.) if z.shape[1]==24 else np.c_[z[:,:24]/np.sqrt(24.),z[:,24:]/4.]


def even_kernel(u,v):
    # 2 E[ReLU(g.u) ReLU(g.v)] - .5 u.v = .5 E[|g.u||g.v|], PSD.
    return arcmod.arc(u,v)-.5*(u@v.T)


def kernel_state(z,aux_trace=AUX_TRACE):
    u=embed(z);raw=even_kernel(u,u);raw=(raw+raw.T)/2
    np.fill_diagonal(raw,.5*np.sum(u*u,axis=1))
    mean=raw.mean(1);grand=float(mean.mean());centered=raw-mean[:,None]-mean[None,:]+grand
    variance=float(np.trace(centered)/len(z));scale=aux_trace/variance if variance>1e-12 else 0.
    gram=scale*centered;gram=(gram+gram.T)/2
    return gram,dict(train_u=u,raw_row_mean=mean,raw_grand_mean=grand,aux_scale=scale,aux_variance=variance,aux_trace=aux_trace)


def fit(z,y,deadline=None,aux_trace=AUX_TRACE,ratio=RATIO):
    z=checked(z);y=np.asarray(y,float)
    if y.shape!=(len(z),) or not 20<=len(z)<=2000 or not np.isfinite(y).all() or np.ptp(y)==0 or abs(z.mean(0)).max()>1e-8:
        raise ValueError('Centered inputs and nonconstant finite training labels')
    if not np.isfinite(aux_trace) or aux_trace<0 or not np.isfinite(ratio) or ratio<0:raise ValueError('Finite nonnegative fixed constants')
    if deadline is not None and time.monotonic()>deadline:raise TimeoutError('CPU deadline')
    center=float(y.mean());yc=y-center;n,p=z.shape;gram,state=kernel_state(z,aux_trace)
    factor=cho_factor(gram+ALPHA*np.eye(n),lower=True)
    solved=cho_solve(factor,np.c_[z,yc]);px=ALPHA*solved[:,:p];py=ALPHA*solved[:,p]
    g=z.T@px+ALPHA*np.eye(p);g=(g+g.T)/2;q=z.T@py
    critical=lam=0.
    if p==216:
        _,q0,_,_,_=native.prior.profile(z,y)
        critical=float(np.linalg.norm(q0.reshape(6,32),2));lam=ratio*critical
        td_factor=cho_factor(g[:24,:24],lower=True)
        td0=cho_solve(td_factor,q[:24]);coupling=cho_solve(td_factor,g[:24,24:])
        h=g[24:,24:]-g[24:,:24]@coupling;h=(h+h.T)/2;rhs=q[24:]-g[24:,:24]@td0
        source,info=native.prox.solve(h,rhs,(6,32),lam,deadline=deadline)
        beta=np.r_[td0-coupling@source,source];feasible=np.r_[np.zeros(24),info['dual']]
        penalty=lam*float(np.linalg.svd(source.reshape(6,32),compute_uv=False).sum())
        gap_scale=float(info['gap_scale']);profile=info
    else:
        beta=cho_solve(cho_factor(g,lower=True),q);feasible=np.zeros(p);penalty=0.
        gap_scale=max(1.,.5*float(q@beta));profile=dict(iterations=0,certified=True)
    dual=solved[:,p]-solved[:,:p]@beta
    linear=z@beta;nonlinear=gram@dual;residual=linear+nonlinear-yc
    primal=float(.5*beta@g@beta-q@beta+penalty)
    dv=q-feasible;dual_value=float(-.5*dv@cho_solve(cho_factor(g,lower=True),dv));gap=primal-dual_value
    normal=float(abs((g@beta-q)[:24]).max());normal_limit=1e-8*(1+abs(q[:24]).max())
    kernel_error=float(abs((gram+ALPHA*np.eye(n))@dual-(yc-linear)).max());kernel_limit=1e-9*(1+abs(yc-linear).max())
    norm=float(dual@gram@dual);full_objective=float(.5*(residual@residual+ALPHA*(beta@beta+norm))+penalty)
    numeric=bool(profile['certified'] and -1e-8<=gap<=1e-9*gap_scale+1e-11 and normal<=normal_limit and kernel_error<=kernel_limit)
    if deadline is not None and time.monotonic()>deadline:raise TimeoutError('CPU deadline')
    certificate=dict(alpha=ALPHA,lambda_ratio=ratio if p==216 else 0.,lambda_critical=critical,lambda_value=lam,
        matrix_shape=[6,32] if p==216 else None,profile=profile,profiled_joint_primal=primal,profiled_joint_dual=dual_value,
        full_gap=gap,gap_scale=gap_scale,TD_normal_error=normal,TD_normal_limit=normal_limit,
        kernel_stationarity_error=kernel_error,kernel_stationarity_limit=kernel_limit,
        full_objective=full_objective,linear_squared_norm=float(beta@beta),nonlinear_RKHS_squared_norm=norm,
        train_MSE=float(np.mean(residual**2)),training_prediction_mean=float((linear+nonlinear+center).mean()),
        sum_kernel_dual=float(dual.sum()),numerically_qualified=numeric,training_rows=n,input_dim=p,
        joint_fitting=True,other_model_predictions_used=False)
    return dict(**state,beta=beta,dual=dual,intercept=center,certificate=certificate)


def predict(z,m):
    z=checked(z)
    if z.shape[1]!=len(m['beta']):raise ValueError('Matching deployment dimension')
    raw=even_kernel(embed(z),m['train_u'])
    centered=raw-raw.mean(1)[:,None]-m['raw_row_mean'][None,:]+m['raw_grand_mean']
    return z@m['beta']+m['aux_scale']*centered@m['dual']+m['intercept']


def qualify(records,numeric):
    core=[r for r in records if r['source']=='core_real']
    if len(core)!=3:raise ValueError('Three primary pools required')
    primary=sum(r['delta_PCC_vs_R25']>=.005 and r['MSE_reduction_vs_R25_pct']>=0 for r in core)>=2 and all(r['delta_PCC_vs_R25']>=-.005 for r in core)
    paired=sum(r['MSE_reduction_vs_R40_pct']>=0 for r in core)>=2 and np.mean([r['MSE_reduction_vs_R40_pct'] for r in core])>=0 and all(r['delta_PCC_vs_R40']>=-.005 for r in core)
    pairs=[]
    for r in core:
        td=next(t for t in records if t['setting']==r['setting'] and t['source']=='TD_only')
        pairs.append(dict(setting=r['setting'],delta_PCC=r['metrics']['PCC']-td['metrics']['PCC'],MSE_reduction_pct=100*(1-r['metrics']['MSE']/td['metrics']['MSE'])))
    inc=sum(p['delta_PCC']>=0 for p in pairs)>=2 and np.mean([p['MSE_reduction_pct'] for p in pairs])>=0
    return dict(qualified=bool(primary and paired and inc and numeric),primary_native_gate=bool(primary),paired_kernel_gate=bool(paired),
        source_increment_gate=bool(inc),all_numerically_qualified=bool(numeric),paired_TD=pairs)
