"""Fixed train-only bilinear map followed by one group-penalized linear head."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
_p=Path(__file__).resolve().parents[1]/'round4/core.py'
_spec=importlib.util.spec_from_file_location('_r4_projection',_p)
projection=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(projection)
configs=projection.configs
score=projection.score
choose=projection.choose

def second_order(z,kind):
    if kind=='bilinear':
        if z.shape[1]!=40:raise ValueError('Bilinear input width')
        return np.einsum('ni,nj->nij',z[:,:24],z[:,24:]).reshape(len(z),384)
    if kind=='td_quadratic':
        if z.shape[1]!=24:raise ValueError('TD input width')
        i,j=np.triu_indices(24)
        return z[:,i]*z[:,j]
    raise ValueError('Unknown map')

def assemble(z,q,t):
    q=(q-np.asarray(t['second_mean']))/np.asarray(t['second_scale'])
    if t['kind']=='bilinear':return np.c_[z[:,:24],z[:,24:]/np.sqrt(2.),q/np.sqrt(48.)]
    return np.c_[z,q*np.sqrt(16./300.)]

def fit_transform(x,kind):
    x=np.asarray(x,float)
    if x.ndim!=2 or len(x)<20 or not np.isfinite(x).all():raise ValueError('Input contract')
    if kind=='bilinear':
        z,p=projection.fit_transform(x)
    elif kind=='td_quadratic':
        if x.shape[1]!=24:raise ValueError('TD-only source contamination')
        m,s=projection.scaler(x);p={'td_mean':m,'td_scale':s};z=(x-m)/s
    else:raise ValueError('Unknown map')
    q=second_order(z,kind);m,s=projection.scaler(q)
    t={'kind':kind,'projection':p,'second_mean':m,'second_scale':s}
    return assemble(z,q,t),t

def transform(x,t):
    x=np.asarray(x,float)
    if t['kind']=='bilinear':z=projection.transform(x,t['projection'])
    else:z=(x-np.asarray(t['projection']['td_mean']))/np.asarray(t['projection']['td_scale'])
    return assemble(z,second_order(z,t['kind']),t)

def fit_heads(z,y,config):
    if z.shape[1] not in (324,424):raise ValueError('Expanded width')
    c=float(y.mean());g=z.T@z;q=z.T@(y-c);b=np.zeros((len(config),z.shape[1]))
    for k,cf in enumerate(config):
        width=z.shape[1] if cf['weight']>0 else 24
        penalty=np.r_[np.full(24,cf['alpha']),np.full(width-24,cf['alpha']/cf['weight'] if cf['weight'] else 0)]
        b[k,:width]=cho_solve(cho_factor(g[:width,:width]+np.diag(penalty),lower=True),q[:width])
    return b,c
