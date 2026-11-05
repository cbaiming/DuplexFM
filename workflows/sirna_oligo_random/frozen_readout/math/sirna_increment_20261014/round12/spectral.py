"""One common source scale; the locked PCA axes and function space are unchanged."""
import importlib.util
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parents[1]/'round4/core.py'
spec=importlib.util.spec_from_file_location('_r12_ridge',p);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
configs=base.configs;fit_heads=base.fit_heads;score=base.score;choose=base.choose

def raw_projection(x,p):
    x=np.asarray(x,float)
    if x.ndim!=2 or x.shape[1]!=216 or not np.isfinite(x).all():raise ValueError('Input contract')
    td=(x[:,:24]-p['td_mean'])/p['td_scale']
    z=(x[:,24:]-p['src_mean'])/p['src_scale']
    source=(z-p['pca_mean'])@np.asarray(p['components']).T-p['pc_mean']
    return td,source

def fit_transform(x,projection):
    if len(x)<20:raise ValueError('Training pool too small')
    td,source=raw_projection(x,projection);variance=source.var(0)
    common=float(np.sqrt(variance.mean()));common=common if common>1e-12 else 1.
    state=dict(projection=projection,raw_pc_variance=variance,common_pc_scale=common,
        source_variance_trace=float(variance.sum()/common**2),new_PCA_fit=False)
    return np.c_[td,source/common],state

def transform(x,state):
    td,source=raw_projection(x,state['projection'])
    return np.c_[td,source/state['common_pc_scale']]

def collapse(state,b,c):
    p=state['projection'];b=np.asarray(b)
    td=b[:,:24]/np.asarray(p['td_scale']);pc=b[:,24:]/state['common_pc_scale']
    src=(pc@np.asarray(p['components']))/np.asarray(p['src_scale'])
    offset=float(c)-td@np.asarray(p['td_mean'])-src@np.asarray(p['src_mean'])
    offset-=pc@(np.asarray(p['components'])@np.asarray(p['pca_mean'])+np.asarray(p['pc_mean']))
    return np.c_[td,src],offset
