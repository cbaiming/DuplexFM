import numpy as np
import pytest
from anchored import profile,fit,prior,prox


def sample():
    rng=np.random.default_rng(42);z=rng.normal(size=(120,216));z[:,24:]+=.7*z[:,:24].repeat(8,axis=1)
    z-=z.mean(0);y=.1*z[:,1]+.1*z[:,35]+rng.normal(size=120)*.02
    b0=np.linalg.solve(z[:,:24].T@z[:,:24]+100*np.eye(24),z[:,:24].T@(y-y.mean()))
    return z,y,b0


def test_no_source_returns_exact_td():
    z,y,b0=sample();z[:,24:]=0;b,c,cert=fit(z,y,b0)
    np.testing.assert_array_equal(b[:24],b0);assert np.count_nonzero(b[24:])==0
    assert c==float(y.mean()) and cert['certified']


def test_kappa_zero_matches_original_profile():
    z,y,b0=sample();h,q,base,coupling,c,_=profile(z,y,b0,0)
    old=prior.profile(z,y)
    for a,b in zip((h,q,base,coupling,c),old):np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)


def test_dense_equivalent_augmented_joint_solve():
    z,y,b0=sample();b,c,cert=fit(z,y,b0,ratio=0)
    anchor=np.c_[10*np.eye(24),np.zeros((24,192))]
    augmented=np.r_[z,10*np.eye(216),anchor]
    target=np.r_[y-y.mean(),np.zeros(216),10*b0]
    expected=np.linalg.lstsq(augmented,target,rcond=None)[0]
    np.testing.assert_allclose(b,expected,atol=1e-12,rtol=1e-12)
    assert cert['certified']


def test_profile_gradient_unchanged_hessian_strengthened():
    z,y,b0=sample();new=profile(z,y,b0,100);old=profile(z,y,b0,0)
    np.testing.assert_array_equal(new[1],old[1])
    assert np.linalg.eigvalsh(new[0]-old[0]).min()>-1e-10
    assert np.linalg.eigvalsh(new[0]).min()>99.999999


def test_trace_full_system_dual_certificate():
    z,y,b0=sample();b,c,cert=fit(z,y,b0)
    g=z.T@z+100*np.eye(216);g[:24,:24]+=100*np.eye(24)
    rhs=z.T@(y-c);rhs[:24]+=100*b0
    dual=np.r_[np.zeros(24),cert['dual']]
    singular=np.linalg.svd(b[24:].reshape(6,32),compute_uv=False)
    primal=.5*b@g@b-rhs@b+cert['lambda_value']*singular.sum()
    lower=-.5*(rhs-dual)@np.linalg.solve(g,rhs-dual)
    assert -1e-9<=primal-lower<=1e-9*cert['gap_scale']+1e-10
    assert np.max(abs((g@b-rhs)[:24]))<1e-10


def test_fixed_source_zero_respects_base_normal_equation():
    z,y,b0=sample();h,q,_,coupling,c,_=profile(z,y,b0)
    source,cert=prox.solve(h,q,(6,32),float(np.linalg.norm(q.reshape(6,32),2)))
    assert np.linalg.norm(source)<1e-10
    np.testing.assert_allclose(b0-coupling@source,b0,atol=1e-12)


def test_nonzero_anchor_not_test_data_fit():
    z,y,b0=sample()
    with pytest.raises(ValueError):fit(z,y,b0+.1)
    with pytest.raises(ValueError):fit(z+1,y,b0)
    with pytest.raises(ValueError):fit(z,y,b0,kappa=-1)
    with pytest.raises(ValueError):fit(z,np.ones(len(y)),b0)


def test_input_anchor_readonly():
    z,y,b0=sample();before=[a.copy() for a in (z,y,b0)];fit(z,y,b0)
    for a,b in zip((z,y,b0),before):np.testing.assert_array_equal(a,b)
