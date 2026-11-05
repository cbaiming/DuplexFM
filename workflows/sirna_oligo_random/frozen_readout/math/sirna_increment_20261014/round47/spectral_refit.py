"""Fixed singular directions, nonnegative amplitude refit, retained L2 penalty."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_r47_native25',R.parent/'round25/native_trace.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
transform,deployment=native.transform,native.deployment
ALPHA=100.;RANK_EPS=1e-8


def basis_from_previous(beta):
    beta=np.asarray(beta,float)
    if beta.shape!=(216,) or not np.isfinite(beta).all():raise ValueError('Finite native216 coefficients required')
    u,s,vh=np.linalg.svd(beta[24:].reshape(6,32),full_matrices=False)
    keep=s>RANK_EPS;u,s,vh=u[:,keep],s[keep],vh[keep];rank=len(s)
    basis=np.column_stack([np.outer(u[:,j],vh[j]).ravel() for j in range(rank)]) if rank else np.empty((192,0))
    return basis,u,s,vh


def nonnegative_quadratic(h,q):
    h,q=np.asarray(h,float),np.asarray(q,float);r=len(q)
    if h.shape!=(r,r) or r>6 or not np.isfinite(h).all() or not np.isfinite(q).all():raise ValueError('Finite SPD problem of dimension at most6 required')
    if r and (np.max(abs(h-h.T))>1e-9 or np.linalg.eigvalsh(h)[0]<=0):raise ValueError('SPD Hessian required')
    best=np.zeros(r);value=0.;feasible=0;solves=0
    for mask in range(1<<r):
        active=np.array([j for j in range(r) if mask&(1<<j)],dtype=int);d=np.zeros(r)
        if len(active):
            d[active]=np.linalg.solve(h[np.ix_(active,active)],q[active]);solves+=1
        if np.any(d < -1e-12):continue
        d=np.maximum(d,0);feasible+=1;v=float(.5*d@h@d-q@d)
        if v<value:best,value=d,v
    g=h@best-q
    minimum=np.where(best>0,g,np.minimum(g,0))
    return best,dict(enumerated_active_sets=1<<r,feasible_face_solutions=feasible,active_system_solves=solves,
                    profile_objective=value,profile_KKT_max=float(np.max(abs(minimum))) if r else 0.)


def fit(z,y,previous_beta):
    z,y=np.asarray(z,float),np.asarray(y,float)
    if z.ndim!=2 or z.shape[1]!=216 or y.shape!=(len(z),) or not np.isfinite(z).all() or not np.isfinite(y).all():raise ValueError('Finite training design required')
    if len(y)<2 or np.ptp(y)==0 or np.max(abs(z.mean(0)))>1e-8:raise ValueError('Centered input and nonconstant response required')
    basis,u,s,vh=basis_from_previous(previous_beta);rank=len(s)
    x=np.c_[z[:,:24],z[:,24:]@basis];center=float(y.mean());yc=y-center
    g=x.T@x+ALPHA*np.eye(x.shape[1]);q=x.T@yc
    factor=cho_factor(g[:24,:24],lower=True);td0=cho_solve(factor,q[:24]);coupling=cho_solve(factor,g[:24,24:])
    h=g[24:,24:]-g[24:,:24]@coupling;h=(h+h.T)/2;rhs=q[24:]-g[24:,:24]@td0
    amp,info=nonnegative_quadratic(h,rhs);w=np.r_[td0-coupling@amp,amp]
    beta=np.r_[w[:24],basis@amp];gradient=g@w-q
    stationarity=np.r_[gradient[:24],np.where(amp>0,gradient[24:],np.minimum(gradient[24:],0))]
    error=float(np.max(abs(stationarity)));threshold=1e-8*(1+float(np.max(abs(q))))
    obj=float(.5*np.sum((x@w-yc)**2)+.5*ALPHA*(w@w))
    oldw=np.r_[np.asarray(previous_beta)[:24],s]
    oldobj=float(.5*np.sum((x@oldw-yc)**2)+.5*ALPHA*(oldw@oldw))
    valid=error<=threshold and bool(np.all(amp>=0)) and obj<=oldobj+1e-9*max(1.,oldobj)
    cert=dict(**info,mode='fixed_singular_amplitude_refit',alpha=ALPHA,rank_threshold=RANK_EPS,
        selected_rank=rank,selected_parameter_count=24+rank+1,deployment_parameter_count=217,
        supervised_basis_from_old_training_head=True,new_direction_search=False,
        minimum_amplitude=float(amp.min()) if rank else None,active_amplitudes=int(np.count_nonzero(amp)),
        KKT_max=error,KKT_threshold=threshold,conditional_suboptimality_upper=float(stationarity@stationarity/(2*ALPHA)),
        objective_without_trace=obj,reference_objective_without_trace=oldobj,
        train_MSE=float(np.mean((z@beta+center-y)**2)),
        old_basis_reconstruction_error=float(np.max(abs(basis@s-np.asarray(previous_beta)[24:]))),
        certified=bool(valid))
    return dict(beta=beta,intercept=np.array(center),basis=basis,left_vectors=u,right_rows=vh,
                old_singular_values=s,amplitudes=amp,reduced_coef=w),cert


def qualify(records,numeric):
    core=[r for r in records if r['source']=='core_real']
    if len(core)!=3 or len({r['setting'] for r in core})!=3:raise ValueError('Three distinct primary pools required')
    perf=sum(r['delta_PCC_vs_R25']>=.005 and r['MSE_reduction_vs_R25_pct']>=0 for r in core)>=2 and all(r['delta_PCC_vs_R25']>=-.005 for r in core)
    pairs=[dict(setting=r['setting'],delta_PCC=r['metrics']['PCC']-r['TD_metrics']['PCC'],MSE_reduction_pct=100*(1-r['metrics']['MSE']/r['TD_metrics']['MSE'])) for r in core]
    incr=sum(p['delta_PCC']>=0 for p in pairs)>=2 and np.mean([p['MSE_reduction_pct'] for p in pairs])>=0
    return dict(qualified=bool(perf and incr and numeric),performance_gate=bool(perf),source_increment_gate=bool(incr),all_numerically_qualified=bool(numeric),paired_TD=pairs)
