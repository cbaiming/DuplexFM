import numpy as np
import pytest
from channel_sparse import fit,solve,group_norms,prox,prior


def sample():
    rng=np.random.default_rng(42);z=rng.normal(size=(120,216))
    z[:,24:]+=.5*z[:,:24].repeat(8,axis=1);z-=z.mean(0)
    y=.2*z[:,2]+.1*z[:,26]+rng.normal(size=len(z))*.03
    return z,y


def test_group_identity_and_exact_shrink():
    x=np.zeros(192);x[3]=3;x[32+3]=4;x[4]=.2
    p=prox(x,1);assert p[3]==pytest.approx(2.4) and p[35]==pytest.approx(3.2)
    assert p[4]==0 and np.count_nonzero(p)==2
    assert group_norms(x)[3]==5


def test_zero_penalty_is_augmented_joint_ridge():
    z,y=sample();b,c,cert=fit(z,y,ratio=0)
    expected=np.linalg.lstsq(np.r_[z,10*np.eye(216)],np.r_[y-y.mean(),np.zeros(216)],rcond=None)[0]
    np.testing.assert_allclose(b,expected,atol=1e-12,rtol=1e-12);assert cert['certified']


def test_critical_returns_exact_td():
    z,y=sample();b,c,cert=fit(z,y,ratio=1)
    td=np.linalg.solve(z[:,:24].T@z[:,:24]+100*np.eye(24),z[:,:24].T@(y-y.mean()))
    np.testing.assert_allclose(b[:24],td,atol=1e-12,rtol=1e-12)
    assert not np.count_nonzero(b[24:]) and cert['active_channels']==0 and cert['certified']


def test_isotropic_profile_closed_form():
    rng=np.random.default_rng(3);q=rng.normal(size=192);lam=1.9
    b,cert=solve(7*np.eye(192),q,lam)
    expected=np.zeros(192)
    for channel in range(32):
        indices=np.arange(channel,192,32);v=q[indices];n=np.linalg.norm(v)
        expected[indices]=v*max(0.,1-lam/n)/7
    np.testing.assert_allclose(b,expected,atol=1e-12,rtol=1e-12);assert cert['certified']


def test_full_joint_global_certificate():
    z,y=sample();b,c,cert=fit(z,y);g=z.T@z+100*np.eye(216);r=z.T@(y-c)
    group_penalty=sum(np.linalg.norm(b[24+channel:216:32]) for channel in range(32))
    dv=np.r_[np.zeros(24),cert['dual']]
    primal=.5*b@g@b-r@b+cert['lambda_value']*group_penalty
    lower=-.5*(r-dv)@np.linalg.solve(g,r-dv)
    assert -1e-9<=primal-lower<=1e-9*cert['gap_scale']+1e-10
    assert max(np.linalg.norm(dv[24+channel:216:32]) for channel in range(32))<=cert['lambda_value']+1e-12
    assert abs((g@b-r)[:24]).max()<1e-10


def test_shared_channel_permutation_equivariance():
    z,y=sample();b,c,cert=fit(z,y)
    permutation=np.random.default_rng(1).permutation(32)
    order=np.r_[np.arange(24),(24+np.arange(192).reshape(6,32)[:,permutation]).ravel()]
    bp,cp,certp=fit(z[:,order],y)
    np.testing.assert_allclose(bp,b[order],atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(z[:,order]@bp+cp,z@b+c,atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(certp['group_norms'],np.asarray(cert['group_norms'])[permutation],atol=1e-10)


def test_zero_source_returns_td_and_inputs_unchanged():
    z,y=sample();z[:,24:]=0;zz=z.copy();yy=y.copy();b,c,cert=fit(z,y)
    td=np.linalg.solve(z[:,:24].T@z[:,:24]+100*np.eye(24),z[:,:24].T@(y-y.mean()))
    np.testing.assert_allclose(b[:24],td,atol=1e-12);assert np.count_nonzero(b[24:])==0
    np.testing.assert_array_equal(z,zz);np.testing.assert_array_equal(y,yy);assert cert['certified']


def test_invalid_inputs_rejected():
    z,y=sample()
    with pytest.raises(ValueError):fit(z+1,y)
    with pytest.raises(ValueError):fit(z,y,ratio=-1)
    with pytest.raises(ValueError):prox(np.zeros(192),-1)
    with pytest.raises(ValueError):group_norms(np.zeros(32))
    with pytest.raises(ValueError):solve(np.eye(192),np.ones(192),float('nan'))
