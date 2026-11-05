import numpy as np
import pytest
from spectral import base,fit_transform,transform,fit_heads,configs,collapse

def setup():
    rng=np.random.default_rng(42);x=rng.normal(size=(100,216))
    x[:,24:]+=rng.normal(size=(100,5))@rng.normal(size=(5,192))
    old,p=base.fit_transform(x);new,t=fit_transform(x,p)
    y=.4+.08*new[:,0]+.03*new[:,25]+rng.normal(0,.05,len(x))
    return x,y,old,p,new,t

def test_trace_and_relative_variances():
    x,y,old,p,z,t=setup()
    np.testing.assert_allclose(z.mean(0),0,atol=1e-12)
    np.testing.assert_allclose(z[:,24:].var(0).sum(),16,atol=1e-11)
    np.testing.assert_allclose(z[:,:24],old[:,:24],atol=0,rtol=0)
    v=z[:,24:].var(0);e=np.asarray(p['eigenvalues'])
    np.testing.assert_allclose(v/v.sum(),e/e.sum(),atol=1e-12)

def test_same_TD_candidates():
    x,y,old,p,z,t=setup();a,c=fit_heads(old,y,configs());b,d=fit_heads(z,y,configs())
    np.testing.assert_allclose(a[:5],b[:5],atol=1e-12);assert c==d

def test_equivalent_anisotropic_penalty():
    x,y,old,p,z,t=setup();cfg=[dict(alpha=10.,weight=.1)];b,c=fit_heads(z,y,cfg)
    scale=np.r_[np.ones(24),np.asarray(p['pc_scale'])/t['common_pc_scale']]
    penalty=np.r_[np.full(24,10.),np.full(16,100.)]/scale**2
    equivalent=np.linalg.solve(old.T@old+np.diag(penalty),old.T@(y-y.mean()))
    np.testing.assert_allclose(equivalent,b[0]*scale,atol=1e-11)

def test_deployment_and_heldout_scale():
    x,y,old,p,z,t=setup();b,c=fit_heads(z,y,configs()[15:20]);coef,intercept=collapse(t,b,c)
    unseen=x[:4]*17+8;scale=t['common_pc_scale']
    np.testing.assert_allclose(unseen@coef.T+intercept,transform(unseen,t)@b.T+c,atol=1e-10)
    assert t['common_pc_scale']==scale

def test_zero_source_safe():
    x=np.zeros((40,216));_,p=base.fit_transform(x);z,t=fit_transform(x,p)
    assert t['common_pc_scale']==1 and np.isfinite(z).all() and t['source_variance_trace']==0

def test_bad_shape_rejected():
    x,y,old,p,z,t=setup()
    with pytest.raises(ValueError):fit_transform(x[:5],p)
    with pytest.raises(ValueError):transform(x[:,:215],t)
