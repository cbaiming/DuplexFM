"""Fixed R49 training projection with the already fixed five-alpha family."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
R=Path(__file__).resolve().parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
prior=load('_r50_group49',R.parent/'round49/group_nuisance.py')
native=load('_r50_native26',R.parent/'round26/head.py')
ALPHAS=native.ALPHAS;groups=prior.groups;project=prior.project;transform=native.transform;deployment=native.deployment;choose=native.choose


def td_cert(zz,yc,b,alpha):
    rhs=zz.T@yc;normal=float(np.max(abs((zz.T@zz+alpha*np.eye(24))@b-rhs)));threshold=1e-8*(1+float(max(abs(rhs))))
    return dict(mode='TD_partial_target_group_Ridge',alpha=float(alpha),normal_max=normal,normal_threshold=threshold,
        train_MSE=float(np.mean((zz@b-yc)**2)),certified=bool(normal<=threshold))


def fit(z,y,labels,alpha,source,deadline=None):
    if alpha not in ALPHAS:raise ValueError('Original five alphas required')
    x=np.asarray(z,float) if source else np.asarray(z,float)[:,:24];y=np.asarray(y,float)
    zz=project(x,labels);yc=project(y-y.mean(),labels);center=float(y.mean())
    if source:return native.fit(zz,yc+center,alpha,deadline=deadline)
    beta=cho_solve(cho_factor(zz.T@zz+alpha*np.eye(24),lower=True),zz.T@yc)
    return beta,center,td_cert(zz,yc,beta,alpha)


def reuse_kind(setting,fold,inner,alpha,identity,source,old_final_alpha=None):
    if inner is not None and fold==0 and setting in ('Huesken','Mixset','Takayuki') and alpha==100.:return 'R49'
    if identity and (inner is not None or old_final_alpha==alpha):return 'R26_identity' if source else 'TD_identity'
    return 'new'


def relative_gain(pcc,base):
    pcc,base=np.asarray(pcc,float),np.asarray(base,float)
    if pcc.shape!=(4,) or base.shape!=(4,) or not np.isfinite(pcc).all() or not np.isfinite(base).all() or np.any(base<=0):raise ValueError('Four valid fixed settings required')
    return float(np.mean(100*(pcc/base-1)))
