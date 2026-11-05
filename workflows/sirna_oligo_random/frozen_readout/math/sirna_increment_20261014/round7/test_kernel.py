import os,sys
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from kernel import configs,fit_transform,transform,fit_heads,kernel_map,embedding,choose

def sample(n=90):
    rng=np.random.default_rng(42);return rng.normal(size=(n,216)),rng.normal(size=n)

@pytest.mark.parametrize('kind',['joint','td'])
def test_primal_dual_and_td_equivalence(kind):
    x,y=sample();x=x if kind=='joint' else x[:,:24];z,t=fit_transform(x,kind);assert z.shape==(90,88)
    cfg=configs();b,c=fit_heads(z,y,cfg)
    for k,cf in enumerate(cfg):
        if cf['weight']==0:
            ref=np.linalg.solve(z[:,:24].T@z[:,:24]+cf['alpha']*np.eye(24),z[:,:24].T@(y-y.mean()))
            np.testing.assert_allclose(b[k,:24],ref,atol=1e-10);assert np.count_nonzero(b[k,24:])==0
        else:
            K=z[:,:24]@z[:,:24].T+cf['weight']*z[:,24:]@z[:,24:].T
            pred=K@np.linalg.solve(K+cf['alpha']*np.eye(len(y)),y-y.mean())+y.mean()
            np.testing.assert_allclose(z@b[k]+c,pred,atol=1e-9)

def test_nystrom_identity_and_variance():
    rng=np.random.default_rng(42);z=rng.normal(size=(90,40));phi,t=kernel_map(z,'joint');u=embedding(z,'joint');L=u[t['indices']]
    W=np.exp(-cdist(L,L,'sqeuclidean')/t['bandwidth']);C=np.exp(-cdist(u,L,'sqeuclidean')/t['bandwidth'])
    approximate=C@np.linalg.pinv(W,rcond=1e-10)@C.T;H=np.eye(90)-np.ones((90,90))/90
    np.testing.assert_allclose(phi@phi.T,H@approximate@H*t['feature_gain']**2,atol=1e-8)
    np.testing.assert_allclose(phi.mean(0),0,atol=1e-12);np.testing.assert_allclose(phi.var(0).sum(),16.,atol=1e-10)

def test_test_data_cannot_change_state():
    x,y=sample();z,t=fit_transform(x,'joint');before=t['kernel']['landmarks'].copy();b=t['kernel']['bandwidth']
    transformed=transform(x+100,t);assert np.isfinite(transformed).all()
    np.testing.assert_array_equal(before,t['kernel']['landmarks']);assert b==t['kernel']['bandwidth']
    np.testing.assert_allclose(transform(x,t),z,atol=1e-12)

def test_constant_and_duplicate_inputs_are_finite():
    z,t=fit_transform(np.ones((90,24)),'td');assert np.isfinite(z).all() and t['kernel']['feature_gain']==0
    x,y=sample();x[10:20]=x[0];z,t=fit_transform(x,'joint');b,c=fit_heads(z,y,configs());assert np.isfinite(b).all()

def test_shapes_and_fixed_choices():
    x,y=sample()
    with pytest.raises(ValueError):fit_transform(x,'td')
    with pytest.raises(ValueError):fit_transform(x[:10],'joint')
    cfg=configs();assert choose(np.ones(25),np.ones(25),cfg,range(25))==4
    assert choose(np.ones(25),np.ones(25),cfg,range(15,20))==19
