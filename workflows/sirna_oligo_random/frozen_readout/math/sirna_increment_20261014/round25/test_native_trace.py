import numpy as np
import pytest
from native_trace import fit,transform,deployment,SHAPE


def data():
    r=np.random.default_rng(42);z=r.normal(size=(80,216));z-=z.mean(0);z[:,24:]/=np.sqrt(12)
    c=np.outer(np.arange(1.,7.),r.normal(size=32)).ravel()/30
    y=.4+z[:,:24]@r.normal(size=24)/30+z[:,24:]@c+r.normal(size=80)*.1
    return z,y


def independent_certificate(z,y,b,center,s):
    g=z.T@z+100*np.eye(216);q=z.T@(y-center);lam=s['lambda_value'];u=q-g@b
    np.testing.assert_allclose(u[:24],np.zeros(24),atol=1e-9)
    sv=np.linalg.svd(b[24:].reshape(SHAPE),compute_uv=False)
    dual=u[24:].copy();norm=np.linalg.norm(dual.reshape(SHAPE),2)
    if norm>lam:dual*=lam/norm
    v=np.r_[np.zeros(24),dual];primal=.5*b@g@b-q@b+lam*sv.sum()
    dualobj=-.5*(q-v)@np.linalg.solve(g,q-v)
    return primal-dualobj


def test_zero_penalty_matches_full_ridge():
    z,y=data();b,c,s=fit(z,y,ratio=0)
    direct=np.linalg.solve(z.T@z+100*np.eye(216),z.T@(y-y.mean()))
    np.testing.assert_allclose(b,direct,atol=1e-12)
    assert c==y.mean() and s['certified']


def test_critical_spectral_penalty_discards_source_only():
    z,y=data();b,c,s=fit(z,y,ratio=1.)
    np.testing.assert_allclose(b[24:],np.zeros(192),atol=1e-10)
    expected=np.linalg.solve(z[:,:24].T@z[:,:24]+100*np.eye(24),z[:,:24].T@(y-y.mean()))
    np.testing.assert_allclose(b[:24],expected,atol=1e-10)
    assert s['certified']


def test_full_joint_global_certificate():
    z,y=data();b,c,s=fit(z,y)
    gap=independent_certificate(z,y,b,c,s)
    assert -1e-8<=gap<=1e-9*s['gap_scale']+1e-10 and s['certified']
    assert s['matrix_shape']==SHAPE and len(s['singular_values'])==6


def test_six_by_32_inner_product_contract():
    z,y=data();b,c,s=fit(z,y)
    from_matrix=z[:,:24]@b[:24]+np.einsum('nrc,rc->n',z[:,24:].reshape(-1,6,32),b[24:].reshape(6,32))+c
    np.testing.assert_allclose(from_matrix,z@b+c,atol=1e-12)


def test_shared_channel_basis_rotation():
    z,y=data();r=np.random.default_rng(43);q=np.linalg.qr(r.normal(size=(32,32)))[0]
    zr=np.c_[z[:,:24],(z[:,24:].reshape(-1,6,32)@q).reshape(-1,192)]
    b,c,s=fit(z,y);br,cr,sr=fit(zr,y)
    np.testing.assert_allclose(z@b+c,zr@br+cr,atol=1e-7)
    np.testing.assert_allclose(s['singular_values'],sr['singular_values'],atol=1e-7)


def test_pool_order_permutation_equivariance():
    z,y=data();order=[4,1,5,0,3,2];zz=np.c_[z[:,:24],z[:,24:].reshape(-1,6,32)[:,order,:].reshape(-1,192)]
    b,c,_=fit(z,y);bb,cc,_=fit(zz,y)
    np.testing.assert_allclose(z@b+c,zz@bb+cc,atol=1e-7)


def test_raw_deployment_and_input_state():
    r=np.random.default_rng(42);raw=r.normal(size=(80,216));m=raw.mean(0);sd=raw.std(0)
    t=dict(td_mean=m[:24],td_scale=sd[:24],src_mean=m[24:],src_scale=sd[24:])
    z=transform(raw,t);y=r.normal(size=80);b,c,_=fit(z,y);coef,offset=deployment(t,b,c)
    other=r.normal(size=(7,216));np.testing.assert_allclose(transform(other,t)@b+c,other@coef+offset,atol=1e-11)
    np.testing.assert_array_equal(t['td_mean'],m[:24]);np.testing.assert_array_equal(t['src_scale'],sd[24:])


def test_invalid_input():
    z,y=data()
    with pytest.raises(ValueError):fit(z+1,y)
    with pytest.raises(ValueError):fit(z,y,ratio=-1.)
    with pytest.raises(ValueError):fit(z[:,:40],y)
