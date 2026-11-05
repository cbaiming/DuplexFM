"""Label-independent fixed random subspace with training covariance whitening."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve

MAP_SEEDS=(42,43,44)


def make_projection(seed):
    if seed not in MAP_SEEDS:raise ValueError('Unregistered map seed')
    g=np.random.default_rng(seed).standard_normal((192,16))
    q,r=np.linalg.qr(g,mode='reduced')
    return q*np.where(np.diag(r)<0,-1.,1.)


def fit_transform(x,reference,projection):
    x=np.asarray(x,float);p=np.asarray(projection,float)
    if x.ndim!=2 or x.shape[1]!=216 or len(x)<20 or not np.isfinite(x).all():raise ValueError('Input contract')
    if p.shape!=(192,16) or not np.allclose(p.T@p,np.eye(16),rtol=1e-12,atol=1e-12):raise ValueError('Projection contract')
    s=(x[:,24:]-reference['src_mean'])/reference['src_scale'];mean=s.mean(0);z=(s-mean)@p
    cov=z.T@z/len(z);e,v=np.linalg.eigh(cov);e=np.maximum(e,0);sd=np.sqrt(e)
    positive=sd[sd>0];floor=.01*(float(np.median(positive)) if len(positive) else 1.)
    scales=np.where(sd>0,np.maximum(sd,floor),1.);white=(v/scales)@v.T
    state={k:np.asarray(reference[k]) for k in ('td_mean','td_scale','src_mean','src_scale')}
    state.update(source_center=mean,projection=p,covariance=cov,eigenvalues=e,whitening=white,
        floor=floor,rank16_valid=bool(e[-1]>0 and e[0]>1e-10*e[-1]),floor_active=bool(np.any((sd>0)&(sd<floor))))
    return transform(x,state),state


def transform(x,state):
    x=np.asarray(x,float)
    td=(x[:,:24]-state['td_mean'])/state['td_scale'];src=(x[:,24:]-state['src_mean'])/state['src_scale']
    return np.c_[td,(src-state['source_center'])@np.asarray(state['projection'])@np.asarray(state['whitening'])]


def fit_head(z,y):
    center=float(np.mean(y));beta=cho_solve(cho_factor(z.T@z+100*np.eye(40),lower=True),z.T@(y-center))
    return beta,center


def deployment(state,beta,center):
    raw_source=np.asarray(state['projection'])@np.asarray(state['whitening'])@beta[24:]
    coefficient=np.r_[beta[:24]/state['td_scale'],raw_source/state['src_scale']]
    intercept=float(center-np.asarray(state['td_mean'])@coefficient[:24]-np.asarray(state['src_mean'])@coefficient[24:]
        -np.asarray(state['source_center'])@raw_source)
    return coefficient,intercept
