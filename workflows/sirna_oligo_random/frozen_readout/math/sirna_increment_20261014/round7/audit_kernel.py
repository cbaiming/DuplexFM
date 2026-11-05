"""Independent Nyström reconstruction; imports no R7 fitting implementation."""
import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('_r7_projection_audit',Path(__file__).resolve().parents[1]/'round4/verify.py')
audit4=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit4)
close=audit4.close


def distances(a,b):
    out=np.zeros((len(a),len(b)),dtype=np.float64)
    for j in range(a.shape[1]):out+=(a[:,j,None]-b[None,:,j])**2
    return out


def expand(train,other,t,view):
    p=t['projection'];state=t['kernel']
    if view=='TDKernel':
        assert train.shape[1]==24 and state['kind']=='td'
        mu=train.mean(0);sd=train.std(0);sd[sd<1e-12]=1.
        close(mu,p['td_mean'],1e-11);close(sd,p['td_scale'],1e-11)
        z=(train-mu)/sd;zv=(other-mu)/sd;u=z/np.sqrt(24.);uv=zv/np.sqrt(24.)
    else:
        assert train.shape[1]==216 and state['kind']=='joint'
        z=audit4.project(train,train,p);zv=audit4.project(train,other,p)
        u=np.c_[z[:,:24]/np.sqrt(24.),z[:,24:]/np.sqrt(16.)]
        uv=np.c_[zv[:,:24]/np.sqrt(24.),zv[:,24:]/np.sqrt(16.)]
    indices=np.sort(np.random.default_rng(42).choice(len(u),min(64,len(u)),replace=False))
    np.testing.assert_array_equal(indices,state['indices'])
    landmarks=u[indices];close(landmarks,state['landmarks'],1e-11)
    d=distances(u,u);positive=d[np.triu_indices(len(u),1)];positive=positive[positive>1e-12]
    bandwidth=float(np.median(positive)) if len(positive) else 1.
    close(bandwidth,state['bandwidth'],1e-11)
    W=np.exp(-distances(landmarks,landmarks)/bandwidth)
    e,v=np.linalg.eigh(W);threshold=float(e.max()*1e-10);keep=e>threshold
    close(e,state['eigenvalues'],1e-10);close(threshold,state['threshold'],1e-12)
    assert int(keep.sum())==state['effective_rank']
    inverse=np.zeros_like(e);inverse[keep]=1/np.sqrt(e[keep]);expected=(v*inverse)@v.T
    B=np.asarray(state['inverse_root'],dtype=float)
    close(B,B.T,1e-12);close(B,expected,1e-8)
    assert np.linalg.eigvalsh(B).min()>-1e-8
    close(B@W@B,v[:,keep]@v[:,keep].T,1e-8)
    phi=np.exp(-distances(u,landmarks)/bandwidth)@B
    mean=phi.mean(0);var=float(np.mean(np.sum((phi-mean)**2,axis=1)))
    gain=float(np.sqrt(16/var)) if var>1e-12 else 0.
    close(mean,state['feature_mean'],1e-10);close(var,state['feature_variance'],1e-10)
    close(gain,state['feature_gain'],1e-10)
    result=np.exp(-distances(uv,landmarks)/bandwidth)@B
    return np.c_[zv[:,:24],(result-mean)*gain]
