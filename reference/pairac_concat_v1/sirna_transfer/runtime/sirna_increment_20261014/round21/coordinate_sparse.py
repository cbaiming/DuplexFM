"""Source-coordinate L1 with L2 and analytically profiled TD24."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
ALPHA=100.; RATIO=.1; MAX_ITER=5000; TOL=1e-8; SOURCE_SCALE=np.sqrt(192./16.)

def transform(raw,state):
    raw=np.asarray(raw,float)
    if raw.ndim!=2 or raw.shape[1]!=216 or not np.isfinite(raw).all():raise ValueError('Finite TD24+source192 required')
    return np.c_[(raw[:,:24]-state['td_mean'])/state['td_scale'],
                 (raw[:,24:]-state['src_mean'])/state['src_scale']/SOURCE_SCALE]

def profile(z,y):
    z,y=np.asarray(z,float),np.asarray(y,float)
    if z.ndim!=2 or z.shape[1]!=216 or y.shape!=(len(z),):raise ValueError('Shape contract')
    if not np.isfinite(z).all() or not np.isfinite(y).all():raise ValueError('Nonfinite input')
    if np.max(abs(z.mean(0)))>1e-8 or np.ptp(y)==0:raise ValueError('Centered input and nonconstant response required')
    center=float(y.mean());g=z.T@z;q=z.T@(y-center)
    tt=cho_factor(g[:24,:24]+ALPHA*np.eye(24),lower=True)
    td0=cho_solve(tt,q[:24]);coupling=cho_solve(tt,g[:24,24:])
    h=g[24:,24:]+ALPHA*np.eye(192)-g[24:,:24]@coupling;h=(h+h.T)*.5
    return h,q[24:]-g[24:,:24]@td0,td0,coupling,center

def minimum_subgradient(h,q,b,lam):
    g=h@b-q
    return np.where(b!=0,g+lam*np.sign(b),np.sign(g)*np.maximum(abs(g)-lam,0.))

def solve_profile(h,q,lam):
    if lam<0 or not np.isfinite(lam):raise ValueError('Nonnegative finite L1 penalty required')
    if lam==0:return cho_solve(cho_factor(h,lower=True),q),0,True
    if lam>=np.max(abs(q)):return np.zeros(len(q)),0,True
    lip=float(np.linalg.eigvalsh(h)[-1])
    if lip<ALPHA*(1-1e-10):raise ValueError('Hessian must be at least alpha I')
    momentum=(np.sqrt(lip)-np.sqrt(ALPHA))/(np.sqrt(lip)+np.sqrt(ALPHA))
    b=np.zeros(len(q));v=b.copy();threshold=TOL*(1+np.max(abs(q)))
    for step in range(1,MAX_ITER+1):
        u=v-(h@v-q)/lip;bn=np.sign(u)*np.maximum(abs(u)-lam/lip,0.)
        if np.max(abs(minimum_subgradient(h,q,bn,lam)))<=threshold:return bn,step,True
        v=bn+momentum*(bn-b);b=bn
    return b,MAX_ITER,False

def fit_head(z,y,mode):
    if mode not in ('dense','sparse'):raise ValueError('Unknown fixed mode')
    h,q,td0,coupling,center=profile(z,y);lambda_max=float(np.max(abs(q)))
    lam=RATIO*lambda_max if mode=='sparse' else 0.
    source,steps,converged=solve_profile(h,q,lam);beta=np.r_[td0-coupling@source,source]
    sg=minimum_subgradient(h,q,source,lam)
    certificate=dict(mode=mode,alpha=ALPHA,lambda_ratio=RATIO if mode=='sparse' else 0.,
        lambda_max=lambda_max,lambda_value=lam,iterations=steps,converged=bool(converged),
        profiled_KKT_max=float(np.max(abs(sg))),KKT_threshold=float(TOL*(1+lambda_max)),
        profiled_suboptimality_upper_bound=float(sg@sg/(2*ALPHA)),
        source_nonzero=int(np.count_nonzero(source)),source_coordinates=192,
        train_MSE=float(np.mean((z@beta+center-y)**2)))
    return beta,center,certificate

def deployment(state,beta,center):
    coef=np.r_[beta[:24]/state['td_scale'],beta[24:]/np.asarray(state['src_scale'])/SOURCE_SCALE]
    offset=float(center-np.asarray(state['td_mean'])@coef[:24]-np.asarray(state['src_mean'])@coef[24:])
    return coef,offset
