import copy
import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from additive import base,hinge,curvature_fit,curvature_transform,fit_transform,transform,fit_heads

def fixture():
    r=np.random.default_rng(42);x=r.normal(size=(140,216));x[:,24:]+=r.normal(size=(140,4))@r.normal(size=(4,192))
    _,p=base.fit_transform(x);z,t=fit_transform(x,p,'core_real');y=r.normal(size=140)
    return x,p,z,t,y

def test_center_variance_linear_orthogonality():
    z=np.random.default_rng(42).normal(size=(160,16));q,t=curvature_fit(z)
    np.testing.assert_allclose(q.mean(0),0,atol=1e-12)
    np.testing.assert_allclose(q.var(0),1,atol=1e-12)
    np.testing.assert_allclose(np.mean((z-z.mean(0))*q,0),0,atol=1e-12)

def test_natural_cubic_and_linear_tails():
    k=np.array([[-2.],[-.3],[3.]]);a=(k[1,0]-k[0,0])/(k[2,0]-k[0,0]);u=np.linspace(-2,3,301)
    spline=CubicSpline([0,a,1],[0,a**3,2*a-a*a],bc_type='natural')
    ref=spline(np.clip(u,0,1));ref[u<0]=0;ref[u>1]=spline(1)+spline(1,1)*(u[u>1]-1)
    got=hinge((u*5-2)[:,None],k,[True])[:,0]
    np.testing.assert_allclose(got,ref,atol=1e-12)
    np.testing.assert_allclose(spline([0,1],2),0,atol=1e-12)

def test_TD_candidates_and_joint_normal_equations():
    x,p,z,t,y=fixture();cfg=base.configs();b,c=fit_heads(z,y,cfg);old=base.transform(x,p);bb,cc=base.fit_heads(old,y,cfg)
    np.testing.assert_allclose(b[:5,:24],bb[:5,:24],atol=1e-12);assert c==cc
    for row,config in zip(b,cfg):
        n=56 if config['weight'] else 24;pen=np.r_[np.full(24,config['alpha']),np.full(n-24,config['alpha']/config['weight'] if config['weight'] else 0)]
        np.testing.assert_allclose(z[:,:n].T@(z@row+c-y)+pen*row[:n],0,atol=1e-9)

def test_heldout_does_not_refit():
    x,p,z,t,y=fixture();saved=copy.deepcopy(t);q=transform(x[:7]*12,t)
    assert q.shape==(7,56)
    for key in t['curve']:np.testing.assert_array_equal(t['curve'][key],saved['curve'][key])
    np.testing.assert_allclose(transform(x,t),z,atol=0,rtol=0)

def test_constant_and_tied_knots_disabled():
    z=np.zeros((40,3));z[-1,1]=1;z[:,2]=np.arange(40)
    q,t=curvature_fit(z);assert t['active'].tolist()==[False,False,True]
    assert not q[:,:2].any() and np.isfinite(curvature_transform(z*1e4,t)).all()

def test_TD_control_shape():
    x,p,z,t,y=fixture();zz,tt=fit_transform(x,p,'TD24');assert zz.shape==(140,48)
    np.testing.assert_allclose(zz[:,:24],z[:,:24],atol=0,rtol=0)

def test_invalid_data_rejected():
    with pytest.raises(ValueError):curvature_fit(np.zeros((3,16)))
    with pytest.raises(ValueError):curvature_fit(np.full((40,16),np.nan))

def test_curvature_model_nested_contains_linear_solution():
    x,p,z,t,y=fixture();b,c=base.fit_heads(base.transform(x,p),y,[dict(alpha=100.,weight=1.)])
    extended=np.r_[b[0],np.zeros(16)]
    np.testing.assert_allclose(z@extended+c,base.transform(x,p)@b[0]+c,atol=1e-12)
