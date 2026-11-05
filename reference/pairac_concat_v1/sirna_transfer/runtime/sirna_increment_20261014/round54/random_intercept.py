"""Fixed common-penalty target intercepts; deployment never reads a group."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
R=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_r54_prior49',R.parent/'round49/group_nuisance.py')
prior=importlib.util.module_from_spec(s);s.loader.exec_module(prior)
transform,deployment,qualify=prior.transform,prior.deployment,prior.qualify
ALPHA=100.;RATIO=.1


def geometry(labels):
    g=np.asarray(labels)
    if g.ndim!=1 or len(g)<2 or not np.issubdtype(g.dtype,np.integer): raise ValueError('Integer training group ids required')
    if g.min()!=0 or set(g.tolist())!=set(range(int(g.max())+1)): raise ValueError('Consecutive group ids required')
    n=len(g); sizes=np.bincount(g); k=len(sizes); rho=n/k
    gamma=rho/(rho+sizes)
    trace=float(n-k+gamma.sum()-np.sum(sizes*gamma*gamma)/np.sum(sizes*gamma))
    normalizer=(n-1)/trace
    return dict(n=n,K=k,sizes=sizes.tolist(),common_group_penalty=float(rho),group_gamma=gamma.tolist(),
        centered_metric_trace=trace,normalizer=float(normalizer),reference_equal=bool(np.all(sizes==sizes[0])),
        gamma_min=float(gamma.min()),gamma_max=float(gamma.max()),training_inputs_only=True,
        variance_estimated=False,confirmed_gene_or_assay=False)


def sqrt_action(x,labels,meta):
    x=np.asarray(x,float);g=np.asarray(labels,int);squeeze=x.ndim==1
    if squeeze:x=x[:,None]
    if x.ndim!=2 or len(x)!=len(g) or not np.isfinite(x).all():raise ValueError('Finite aligned matrix required')
    sizes=np.asarray(meta['sizes']);gamma=np.asarray(meta['group_gamma'])
    total=np.zeros((len(sizes),x.shape[1]));np.add.at(total,g,x)
    out=x-(1-np.sqrt(gamma[g]))[:,None]*(total[g]/sizes[g,None])
    return out[:,0] if squeeze else out


def design(z,y,labels):
    z=np.asarray(z,float);y=np.asarray(y,float);g=np.asarray(labels)
    if z.ndim!=2 or z.shape[1] not in (24,216) or y.shape!=(len(z),) or len(g)!=len(z):raise ValueError('Aligned native design/response required')
    if not np.isfinite(z).all() or not np.isfinite(y).all() or np.ptp(y)==0:raise ValueError('Finite nonconstant response required')
    meta=geometry(g);w=np.asarray(meta['group_gamma'])[g]
    zm=np.average(z,axis=0,weights=w);ym=float(np.average(y,weights=w))
    root=np.sqrt(meta['normalizer'])
    return root*sqrt_action(z-zm,g,meta),root*sqrt_action(y-ym,g,meta),zm,ym,meta


def fit(z,y,labels,source,deadline=None):
    z=np.asarray(z,float)
    if z.ndim!=2 or z.shape[1]!=216:raise ValueError('Native216 required')
    x=z if source else z[:,:24]
    a,t,zm,ym,meta=design(x,y,labels)
    gram=a.T@a+ALPHA*np.eye(x.shape[1]);rhs=a.T@t
    if source:
        tf=cho_factor(gram[:24,:24],lower=True);td0=cho_solve(tf,rhs[:24]);coupling=cho_solve(tf,gram[:24,24:])
        h=gram[24:,24:]-gram[24:,:24]@coupling;h=(h+h.T)*.5;q=rhs[24:]-gram[24:,:24]@td0
        critical=float(np.linalg.norm(q.reshape(6,32),2));lam=RATIO*critical
        sb,cert=prior.native.prox.solve(h,q,(6,32),lam,deadline=deadline)
        beta=np.r_[td0-coupling@sb,sb]
        cert.update(mode='native_trace_common_target_intercept',alpha=ALPHA,lambda_ratio=RATIO,
            lambda_critical=critical,lambda_value=lam,matrix_shape=[6,32])
    else:
        beta=cho_solve(cho_factor(gram,lower=True),rhs)
        normal=float(np.max(abs(gram@beta-rhs)));threshold=1e-8*(1+float(np.max(abs(rhs))))
        cert=dict(mode='TD_common_target_intercept_Ridge',alpha=ALPHA,normal_max=normal,normal_threshold=threshold,certified=bool(normal<=threshold))
    center=float(ym-zm@beta)
    cert['train_MSE']=float(np.mean((a@beta-t)**2))
    return beta,center,cert


def residual_stats(z,y,beta,center,labels,source):
    x=z if source else z[:,:24];r=np.asarray(y)-x@beta-center;g=np.asarray(labels,int);m=geometry(g)
    sizes=np.asarray(m['sizes']);rho=m['common_group_penalty'];means=np.bincount(g,weights=r)/sizes
    u=sizes/(sizes+rho)*means
    raw=float(r@r);profile=m['normalizer']*(raw-float(np.sum(sizes*sizes/(sizes+rho)*means*means)))
    return dict(raw_training_MSE=raw/len(r),normalized_profiled_SSE=profile,
        normalized_augmented_SSE=m['normalizer']*float(np.sum((r-u[g])**2)+rho*np.sum(u*u)),
        weighted_intercept_normal=float(np.sum(np.asarray(m['group_gamma'])[g]*r)),
        training_group_effect_norm=float(np.linalg.norm(u)),group_effects_deployed=False)
