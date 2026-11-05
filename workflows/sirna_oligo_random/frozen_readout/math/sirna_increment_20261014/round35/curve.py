"""Deterministic nested training subsets and unchanged native convex readout."""
import hashlib,importlib.util
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_r35_head26',R.parent/'round26/head.py')
head=importlib.util.module_from_spec(spec);spec.loader.exec_module(head)
FRACTIONS=(.5,.75);DRAWS=(42,43,44)


def subset(ids,fraction,draw):
    if len(ids)!=len(set(ids)) or not all(isinstance(s,str) for s in ids):raise ValueError('Unique string IDs')
    if fraction not in FRACTIONS or draw not in DRAWS or len(ids)<40:raise ValueError('Fixed sample budget')
    order=sorted(ids,key=lambda s:(hashlib.sha256((str(draw)+'\x00'+s).encode()).digest(),s))
    chosen=set(order[:int(np.ceil(fraction*len(ids)))]);return [i for i,s in enumerate(ids) if s in chosen]


def scaler(x,floor):
    mean=x.mean(0);sd=x.std(0)
    if floor:
        constant=np.ptp(x,axis=0)==0;positive=sd[(~constant)&(sd>0)]
        lo=.01*(float(np.median(positive)) if len(positive) else 1.)
        sd=np.where(constant,1.,np.maximum(sd,lo))
    else:sd[sd<1e-12]=1.
    return mean,sd


def fit_transform(raw):
    raw=np.asarray(raw,float)
    if raw.ndim!=2 or raw.shape[1]!=216 or len(raw)<20 or not np.isfinite(raw).all():raise ValueError('Finite native216 training inputs')
    tm,ts=scaler(raw[:,:24],False);sm,ss=scaler(raw[:,24:],True)
    state=dict(td_mean=tm,td_scale=ts,src_mean=sm,src_scale=ss)
    return head.transform(raw,state),state


def fit(z,y,full_n,kind,deadline=None):
    z,y=np.asarray(z,float),np.asarray(y,float)
    if kind not in ('native','TD') or z.shape!=(len(y),216) or full_n<len(y) or len(y)<20:raise ValueError('Curve fit contract')
    weight=full_n/len(y);s=np.sqrt(weight)
    if kind=='native':
        beta,center,certificate=head.fit(s*z,s*y,100.,deadline=deadline)
        center/=s
    else:
        center=float(y.mean());zt=s*z[:,:24]
        beta=np.zeros(216);beta[:24]=cho_solve(cho_factor(zt.T@zt+100*np.eye(24),lower=True),zt.T@(s*(y-center)))
        normal=float(abs((zt.T@zt+100*np.eye(24))@beta[:24]-zt.T@(s*(y-center))).max()/len(y))
        certificate=dict(mode='TD',alpha=100.,normal_error=normal,certified=bool(normal<1e-8))
    certificate=dict(certificate,row_weight=weight,equivalent_unweighted_alpha=100./weight,full_training_rows=full_n,training_rows=len(y))
    return beta,center,certificate


def metrics(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float)
    return dict(PCC=float(np.corrcoef(y,p)[0,1]),MSE=float(np.mean((y-p)**2)))


def perturbation_variance(predictions):
    p=np.asarray(predictions,float)
    if p.ndim!=2 or p.shape[0]!=3 or not np.isfinite(p).all():raise ValueError('Three subset draws required')
    # Pairwise identity: population variance across draws, no averaged prediction is deployed or scored.
    return float(sum(np.mean((p[i]-p[j])**2) for i in range(3) for j in range(i+1,3))/9.)
