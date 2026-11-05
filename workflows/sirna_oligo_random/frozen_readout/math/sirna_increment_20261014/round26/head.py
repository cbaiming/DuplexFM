"""Same convex native trace objective, generalized to the original five alphas."""
import importlib.util,time
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
R=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('_r26_prox',R.parent/'round17/nuclear.py');prox=importlib.util.module_from_spec(sp);sp.loader.exec_module(prox)
ALPHAS=(.1,1.,10.,100.,1000.);SHAPE=(6,32);RATIO=.1;MAXITER=20000;TOL=1e-9


def transform(raw,t):
    raw=np.asarray(raw,float)
    if raw.ndim!=2 or raw.shape[1]!=216 or not np.isfinite(raw).all():raise ValueError('Finite TD24+native192 input required')
    return np.c_[(raw[:,:24]-t['td_mean'])/t['td_scale'],(raw[:,24:]-t['src_mean'])/t['src_scale']/np.sqrt(12.)]


def profile(z,y,alpha):
    z,y=np.asarray(z,float),np.asarray(y,float)
    if alpha not in ALPHAS or z.ndim!=2 or z.shape[1]!=216 or y.shape!=(len(z),):raise ValueError('Fixed family/shape contract')
    if not np.isfinite(z).all() or not np.isfinite(y).all() or max(abs(z.mean(0)))>1e-8 or np.ptp(y)==0:raise ValueError('Finite centered features/nonconstant response')
    c=float(y.mean());g=z.T@z;q=z.T@(y-c);factor=cho_factor(g[:24,:24]+alpha*np.eye(24),lower=True)
    td0=cho_solve(factor,q[:24]);coupling=cho_solve(factor,g[:24,24:])
    h=g[24:,24:]+alpha*np.eye(192)-g[24:,:24]@coupling;h=(h+h.T)*.5
    return h,q[24:]-g[24:,:24]@td0,td0,coupling,c


def fit(z,y,alpha,deadline=None):
    h,q,td0,coupling,center=profile(z,y,alpha);critical=float(np.linalg.norm(q.reshape(SHAPE),2));lam=RATIO*critical
    factor=cho_factor(h,lower=True);dense=cho_solve(factor,q);gap_scale=max(1.,.5*float(q@dense))
    eigen=np.linalg.eigvalsh(h);lip=float(eigen[-1]);assert eigen[0]>=alpha*(1-1e-8)
    momentum=(np.sqrt(lip)-np.sqrt(alpha))/(np.sqrt(lip)+np.sqrt(alpha))
    b=np.zeros(192);look=b.copy();history=[]
    if critical==0:
        info=prox.certificate(h,q,b,lam,SHAPE,factor);step=0
    else:
        for step in range(1,MAXITER+1):
            nxt=prox.shrink((look-(h@look-q)/lip).reshape(SHAPE),lam/lip).ravel()
            look=nxt+momentum*(nxt-b);b=nxt
            if deadline is not None and time.monotonic()>deadline:raise TimeoutError('CPU stage deadline')
            if step%10==0 or step==MAXITER:
                info=prox.certificate(h,q,b,lam,SHAPE,factor);history.append([step,info['gap']])
                if info['gap']<=TOL*gap_scale:break
    info.update(alpha=float(alpha),mode='native_trace',lambda_ratio=RATIO,lambda_critical=critical,lambda_value=lam,
        matrix_shape=SHAPE,iterations=step,certified=bool(info['gap']<=TOL*gap_scale),gap_scale=gap_scale,
        L=lip,history=np.asarray(history),algorithm='strongly_convex_proximal_gradient',maximum_iterations=MAXITER)
    beta=np.r_[td0-coupling@b,b];info['train_MSE']=float(np.mean((z@beta+center-y)**2))
    return beta,center,info


def choose(pc,ms):return min(range(5),key=lambda k:(-float(pc[k]),float(ms[k]),-ALPHAS[k],k))


def deployment(t,b,c):
    mean=np.r_[t['td_mean'],t['src_mean']];scale=np.r_[t['td_scale'],np.array(t['src_scale'])*np.sqrt(12.)]
    return b/scale,float(c-mean@(b/scale))
