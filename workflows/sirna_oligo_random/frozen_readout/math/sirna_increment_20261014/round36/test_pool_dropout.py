import itertools
import numpy as np
import pytest
from pool_dropout import penalty,fit,prior,deployment,transform


def sample():
    rng=np.random.default_rng(42);z=rng.normal(size=(48,216));z-=z.mean(0)
    z[:,24:]/=np.sqrt(12);y=rng.normal(size=48)
    return z,y,rng


def test_exact_expected_squared_loss_all64_masks():
    z,y,rng=sample();b=rng.normal(size=216);keep=.9
    pools=np.stack([z[:,24+32*j:24+32*(j+1)]@b[24+32*j:24+32*(j+1)] for j in range(6)],1)
    loss=probability=0.
    for bits in itertools.product((0,1),repeat=6):
        p=keep**sum(bits)*(1-keep)**(6-sum(bits));probability+=p
        prediction=.2+z[:,:24]@b[:24]+pools@np.array(bits)/keep
        loss+=p*np.sum((prediction-y)**2)
    expected=np.sum((z@b+.2-y)**2)+b[24:]@penalty(z[:,24:])@b[24:]
    np.testing.assert_allclose(probability,1.,atol=1e-13)
    np.testing.assert_allclose(loss,expected,rtol=1e-12)


def test_penalty_blocks_psd_and_keep_one():
    z,_,_=sample();p=penalty(z[:,24:])
    assert np.linalg.eigvalsh(p).min()>-1e-12
    for a in range(6):
        for b in range(6):
            if a!=b:assert np.count_nonzero(p[32*a:32*(a+1),32*b:32*(b+1)])==0
    assert np.count_nonzero(penalty(z[:,24:],1.))==0


def test_no_dropout_matches_original_science():
    z,y,_=sample();b,c,cert=fit(z,y,keep=1.)
    ob,oc,old=prior.fit(z,y)
    np.testing.assert_allclose(b,ob,rtol=1e-10,atol=1e-10);np.testing.assert_allclose(c,oc)
    assert cert['certified'] and cert['dropout_extra_SSE']==0


def test_zero_source_matches_independent_TD_LS():
    z,y,_=sample();z[:,24:]=0;b,c,cert=fit(z,y)
    reference=np.linalg.lstsq(np.r_[z[:,:24],10*np.eye(24)],np.r_[y-y.mean(),np.zeros(24)],rcond=None)[0]
    np.testing.assert_allclose(b[:24],reference,atol=1e-12)
    assert np.count_nonzero(b[24:])==0 and cert['certified']


def test_fixed_block_rotation_preserves_expected_variance():
    z,_,rng=sample();x=z[:,24:];b=rng.normal(size=(6,32));q,_=np.linalg.qr(rng.normal(size=(32,32)))
    newx=np.concatenate([x[:,j*32:(j+1)*32]@q for j in range(6)],axis=1);newb=b@q
    np.testing.assert_allclose(b.ravel()@penalty(x)@b.ravel(),newb.ravel()@penalty(newx)@newb.ravel(),atol=1e-10)


def test_added_quadratic_gradient_finite_difference():
    z,_,rng=sample();p=penalty(z[:,24:]);b=rng.normal(size=192);v=rng.normal(size=192);v/=np.linalg.norm(v);eps=1e-5
    objective=lambda a:.5*a@p@a
    finite=(objective(b+eps*v)-objective(b-eps*v))/(2*eps)
    np.testing.assert_allclose(finite,v@(p@b),rtol=1e-7,atol=1e-8)


def test_deployment_remains_single_affine_predictor():
    z,y,rng=sample();b,c,cert=fit(z,y)
    state=dict(td_mean=rng.normal(size=24),td_scale=np.ones(24)*2,src_mean=rng.normal(size=192),src_scale=np.ones(192)*3)
    raw=rng.normal(size=(15,216));coef,offset=deployment(state,b,c)
    np.testing.assert_allclose(raw@coef+offset,transform(raw,state)@b+c,atol=1e-12)
    assert cert['certified']


def test_contracts_reject_invalid_probabilities_or_inputs():
    z,_,_=sample()
    for keep in (0,-.1,1.1,np.nan):
        with pytest.raises(ValueError):penalty(z[:,24:],keep)
    with pytest.raises(ValueError):penalty(z[:,:30])
    with pytest.raises(ValueError):penalty(np.full((3,192),np.inf))
