"""Fixed low-rank joint-RBF feature map, fitted only on the training partition."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
p=Path(__file__).resolve().parents[1]/'round4/core.py'
spec=importlib.util.spec_from_file_location('_r4_fixed_projection',p)
projection=importlib.util.module_from_spec(spec);spec.loader.exec_module(projection)
configs=projection.configs;score=projection.score;choose=projection.choose

def embedding(z,kind):
    if kind=='joint':
        if z.shape[1]!=40:raise ValueError('Joint width')
        return np.c_[z[:,:24]/np.sqrt(24.),z[:,24:]/np.sqrt(16.)]
    if kind=='td':
        if z.shape[1]!=24:raise ValueError('TD width')
        return z/np.sqrt(24.)
    raise ValueError('Map kind')

def kernel_map(z,kind):
    u=embedding(z,kind);n=len(u);indices=np.sort(np.random.default_rng(42).choice(n,min(64,n),replace=False));landmarks=u[indices].copy()
    distances=cdist(u,u,'sqeuclidean');positive=distances[np.triu_indices(n,1)];positive=positive[positive>1e-12]
    bandwidth=float(np.median(positive)) if len(positive) else 1.
    w=np.exp(-cdist(landmarks,landmarks,'sqeuclidean')/bandwidth);e,v=np.linalg.eigh(w)
    threshold=float(e.max()*1e-10);keep=e>threshold;inverse=np.zeros_like(e);inverse[keep]=1./np.sqrt(e[keep])
    b=(v*inverse)@v.T;b=(b+b.T)/2
    phi=np.exp(-cdist(u,landmarks,'sqeuclidean')/bandwidth)@b;mean=phi.mean(0)
    variance=float(np.mean(np.sum((phi-mean)**2,axis=1)));gain=float(np.sqrt(16./variance)) if variance>1e-12 else 0.
    state=dict(kind=kind,indices=indices,landmarks=landmarks,bandwidth=bandwidth,inverse_root=b,
        eigenvalues=e,threshold=threshold,effective_rank=int(keep.sum()),feature_mean=mean,feature_gain=gain,feature_variance=variance)
    return (phi-mean)*gain,state

def fit_transform(x,kind):
    x=np.asarray(x,float)
    if x.ndim!=2 or len(x)<20 or not np.isfinite(x).all():raise ValueError('Input contract')
    if kind=='joint':z,p=projection.fit_transform(x)
    elif kind=='td':
        if x.shape[1]!=24:raise ValueError('TD-only source contamination')
        m,s=projection.scaler(x);p=dict(td_mean=m,td_scale=s);z=(x-m)/s
    else:raise ValueError('Map kind')
    phi,state=kernel_map(z,kind)
    return np.c_[z[:,:24],phi],dict(projection=p,kernel=state)

def transform(x,t):
    x=np.asarray(x,float);state=t['kernel'];p=t['projection']
    if state['kind']=='joint':z=projection.transform(x,p)
    else:z=(x-np.asarray(p['td_mean']))/np.asarray(p['td_scale'])
    u=embedding(z,state['kind']);phi=np.exp(-cdist(u,np.asarray(state['landmarks']),'sqeuclidean')/state['bandwidth'])@np.asarray(state['inverse_root'])
    return np.c_[z[:,:24],(phi-np.asarray(state['feature_mean']))*state['feature_gain']]

def fit_heads(z,y,config):
    n=z.shape[1]
    if n<25 or n>88:raise ValueError('Head width')
    c=float(y.mean());g=z.T@z;q=z.T@(y-c);b=np.zeros((len(config),n))
    for k,cf in enumerate(config):
        width=n if cf['weight']>0 else 24
        pen=np.r_[np.full(24,cf['alpha']),np.full(width-24,cf['alpha']/cf['weight'] if cf['weight'] else 0)]
        b[k,:width]=cho_solve(cho_factor(g[:width,:width]+np.diag(pen),lower=True),q[:width])
    return b,c
