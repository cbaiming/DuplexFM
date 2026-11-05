"""Input-only orthogonalization and one joint efficiency readout."""
import importlib.util
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parents[1]/'round4/core.py'
spec=importlib.util.spec_from_file_location('_r8_projection',p)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
configs=base.configs;fit_heads=base.fit_heads;score=base.score;choose=base.choose

def fit_transform(x):
    z,p=base.fit_transform(x);td,source=z[:,:24],z[:,24:]
    a,_,rank,singular=np.linalg.lstsq(td,source,rcond=1e-10)
    rest=source-td@a;mean=rest.mean(0)
    state=dict(projection=p,td_to_source=a,residual_mean=mean,TD_rank=int(rank),TD_singular=singular,rcond=1e-10)
    return np.c_[td,rest-mean],state

def transform(x,t):
    z=base.transform(x,t['projection'])
    return np.c_[z[:,:24],z[:,24:]-z[:,:24]@np.asarray(t['td_to_source'])-np.asarray(t['residual_mean'])]

def collapse(t,b,c):
    """Fold the entire preprocessing/head into one affine function of input216."""
    p=t['projection'];b=np.asarray(b);bt=b[:,:24]-b[:,24:]@np.asarray(t['td_to_source']).T
    bs=b[:,24:]/np.asarray(p['pc_scale'])
    td=bt/np.asarray(p['td_scale']);src=(bs@np.asarray(p['components']))/np.asarray(p['src_scale'])
    offset=float(c)-b[:,24:]@np.asarray(t['residual_mean'])-td@np.asarray(p['td_mean'])-src@np.asarray(p['src_mean'])
    offset-=bs@(np.asarray(p['components'])@np.asarray(p['pca_mean'])+np.asarray(p['pc_mean']))
    return np.c_[td,src],offset
