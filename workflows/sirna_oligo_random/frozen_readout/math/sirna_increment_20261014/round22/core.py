"""Same R21 convex objective, with fixed five-alpha selection and native CD."""
import warnings
import numpy as np
from scipy.linalg import cho_factor,cho_solve,solve_triangular
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning
ALPHAS=(.1,1.,10.,100.,1000.)
MODES=('dense','sparse')

def transform(raw,t):
    raw=np.asarray(raw,float)
    if raw.ndim!=2 or raw.shape[1]!=216 or not np.isfinite(raw).all():raise ValueError('Input contract')
    return np.c_[(raw[:,:24]-t['td_mean'])/t['td_scale'],(raw[:,24:]-t['src_mean'])/t['src_scale']/np.sqrt(12.)]

def fit(z,y,alpha,mode):
    if alpha not in ALPHAS or mode not in MODES:raise ValueError('Fixed family only')
    if z.shape[1]!=216 or len(y)!=len(z) or not np.isfinite(z).all() or not np.isfinite(y).all():raise ValueError('Shape/finite')
    if max(abs(z.mean(0)))>1e-8 or np.ptp(y)==0:raise ValueError('Centered input/nonconstant response')
    center=float(y.mean());gram=z.T@z;q=z.T@(y-center);tt=cho_factor(gram[:24,:24]+alpha*np.eye(24),lower=True)
    td0=cho_solve(tt,q[:24]);coupling=cho_solve(tt,gram[:24,24:])
    h=gram[24:,24:]+alpha*np.eye(192)-gram[24:,:24]@coupling;h=(h+h.T)*.5
    cq=q[24:]-gram[24:,:24]@td0;lmax=float(max(abs(cq)));lam=.1*lmax if mode=='sparse' else 0.
    niter=0;dual=0.;messages=[]
    if lam==0:b=cho_solve(cho_factor(h,lower=True),cq)
    else:
        lower=np.linalg.cholesky(h);design=np.asfortranarray(lower.T);response=solve_triangular(lower,cq,lower=True)
        model=Lasso(alpha=lam/192,fit_intercept=False,precompute=True,max_iter=100000,tol=1e-14,selection='cyclic',warm_start=False)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always',ConvergenceWarning);model.fit(design,response)
        b=model.coef_.copy();niter=int(model.n_iter_);dual=float(model.dual_gap_);messages=[str(w.message) for w in ws]
    beta=np.r_[td0-coupling@b,b];g=(gram+alpha*np.eye(216))@beta-q
    sg=np.where(b!=0,g[24:]+lam*np.sign(b),np.sign(g[24:])*np.maximum(abs(g[24:])-lam,0.))
    err=float(max(abs(sg)));limit=1e-8*(1+lmax)
    valid=bool(err<=limit and max(abs(g[:24]))<=limit and niter<100000 and not messages)
    cert=dict(alpha=alpha,mode=mode,lambda_max=lmax,lambda_value=lam,source_nonzero=int(np.count_nonzero(b)),
        iterations=niter,dual_gap=dual,warnings=messages,source_KKT=err,TD_KKT=float(max(abs(g[:24]))),KKT_threshold=limit,
        suboptimality_upper_bound=float(np.r_[g[:24],sg]@np.r_[g[:24],sg]/(2*alpha)),numerically_qualified=valid)
    return beta,center,cert

def choose(pcc,mse):
    return min(range(5),key=lambda k:(-float(pcc[k]),float(mse[k]),-ALPHAS[k],k))

def deployment(t,b,c):
    mean=np.r_[t['td_mean'],t['src_mean']];scale=np.r_[t['td_scale'],np.asarray(t['src_scale'])*np.sqrt(12.)]
    return b/scale,float(c-mean@(b/scale))
