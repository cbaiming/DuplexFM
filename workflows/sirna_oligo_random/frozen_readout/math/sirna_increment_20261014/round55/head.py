"""R54 target random intercepts, using the unchanged five-alpha convex solver."""
import importlib.util,time
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
R=Path(__file__).resolve().parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
prior=load('_r55_common54',R.parent/'round54/random_intercept.py')
native=load('_r55_native26',R.parent/'round26/head.py')
ALPHAS=native.ALPHAS;geometry=prior.geometry;transform=native.transform;deployment=native.deployment;choose=native.choose


def fit(z,y,labels,alpha,source,deadline=None):
    if alpha not in ALPHAS:raise ValueError('Original five alphas only')
    z=np.asarray(z,float);y=np.asarray(y,float)
    if z.ndim!=2 or z.shape[1]!=216:raise ValueError('Native216 required')
    x=z if source else z[:,:24];a,t,zm,ym,meta=prior.design(x,y,labels)
    gram=a.T@a+alpha*np.eye(x.shape[1]);rhs=a.T@t
    if not source:
        b=cho_solve(cho_factor(gram,lower=True),rhs);err=float(np.max(abs(gram@b-rhs)));threshold=1e-8*(1+float(np.max(abs(rhs))))
        cert=dict(mode='TD_common_target_intercept_Ridge',alpha=float(alpha),normal_max=err,normal_threshold=threshold,certified=bool(err<=threshold))
    else:
        tf=cho_factor(gram[:24,:24],lower=True);td0=cho_solve(tf,rhs[:24]);coupling=cho_solve(tf,gram[:24,24:])
        h=gram[24:,24:]-gram[24:,:24]@coupling;h=(h+h.T)*.5;q=rhs[24:]-gram[24:,:24]@td0
        critical=float(np.linalg.norm(q.reshape(6,32),2));lam=.1*critical;factor=cho_factor(h,lower=True)
        dense=cho_solve(factor,q);gap_scale=max(1.,.5*float(q@dense));eigen=np.linalg.eigvalsh(h);lip=float(eigen[-1])
        assert eigen[0]>=alpha*(1-1e-8);momentum=(np.sqrt(lip)-np.sqrt(alpha))/(np.sqrt(lip)+np.sqrt(alpha))
        sb=np.zeros(192);look=sb.copy();history=[]
        if critical==0:cert=native.prox.certificate(h,q,sb,lam,(6,32),factor);step=0
        else:
            for step in range(1,native.MAXITER+1):
                nxt=native.prox.shrink((look-(h@look-q)/lip).reshape(6,32),lam/lip).ravel()
                look=nxt+momentum*(nxt-sb);sb=nxt
                if deadline is not None and time.monotonic()>deadline:raise TimeoutError('CPU stage deadline')
                if step%10==0 or step==native.MAXITER:
                    cert=native.prox.certificate(h,q,sb,lam,(6,32),factor);history.append([step,cert['gap']])
                    if cert['gap']<=native.TOL*gap_scale:break
        cert.update(alpha=float(alpha),mode='native_trace_common_target_intercept',lambda_ratio=.1,lambda_critical=critical,lambda_value=lam,
            matrix_shape=[6,32],iterations=step,certified=bool(cert['gap']<=native.TOL*gap_scale),gap_scale=gap_scale,L=lip,history=np.asarray(history),
            algorithm='strongly_convex_proximal_gradient',maximum_iterations=native.MAXITER)
        b=np.r_[td0-coupling@sb,sb]
    center=float(ym-zm@b);cert['train_MSE']=float(np.mean((a@b-t)**2))
    return b,center,cert


def reuse_kind(setting,fold,inner,alpha,equal,old_final_alpha=None):
    if inner is not None and fold==0 and setting in ('Huesken','Mixset','Takayuki') and alpha==100.:return 'R54'
    if equal and (inner is not None or alpha==old_final_alpha):return 'R50_equal'
    return 'new'


def relative_gain(pcc,base):
    pcc,base=np.asarray(pcc,float),np.asarray(base,float)
    if pcc.shape!=(4,) or base.shape!=(4,) or not np.isfinite(pcc).all() or not np.isfinite(base).all() or np.any(base<=0):raise ValueError('Four valid fixed settings required')
    return float(np.mean(100*(pcc/base-1)))
