import inspect
import numpy as np
import pytest
from random_subspace import make_projection,fit_transform,transform,fit_head,deployment


def fixture():
    r=np.random.default_rng(19);x=r.normal(size=(160,216))+np.arange(216)/10
    ref=dict(td_mean=x[:,:24].mean(0),td_scale=x[:,:24].std(0),src_mean=x[:,24:].mean(0),src_scale=x[:,24:].std(0))
    y=.3*x[:,0]+.2*x[:,40]+r.normal(size=160)*.1
    return x,ref,y


def test_map_orthogonal_reproducible_and_distinct():
    q=make_projection(42)
    np.testing.assert_allclose(q.T@q,np.eye(16),atol=1e-12)
    np.testing.assert_array_equal(q,make_projection(42))
    assert not np.allclose(q,make_projection(43)) and not np.allclose(q,make_projection(44))


def test_train_whitening_covariance():
    x,r,y=fixture();z,s=fit_transform(x,r,make_projection(42))
    np.testing.assert_allclose(z[:,24:].mean(0),0,atol=1e-12)
    np.testing.assert_allclose(z[:,24:].T@z[:,24:]/len(x),np.eye(16),atol=1e-11)
    assert s['rank16_valid'] and not s['floor_active']


def test_whitening_independent_svd():
    x,r,y=fixture();z,s=fit_transform(x,r,make_projection(43))
    raw=((x[:,24:]-r['src_mean'])/r['src_scale']-s['source_center'])@s['projection']
    _,e,v=np.linalg.svd(raw,full_matrices=False);independent=(v.T/(e/np.sqrt(len(x))))@v
    np.testing.assert_allclose(s['whitening'],independent,atol=1e-10)


def test_ridge_and_deployment():
    x,r,y=fixture();z,s=fit_transform(x,r,make_projection(42));b,c=fit_head(z,y);coef,offset=deployment(s,b,c)
    np.testing.assert_allclose((z.T@z+100*np.eye(40))@b,z.T@(y-y.mean()),atol=1e-10)
    np.testing.assert_allclose(x@coef+offset,z@b+c,atol=1e-10)


def test_subspace_basis_rotation_preserves_predictions():
    x,r,y=fixture();p=make_projection(42);v=np.linalg.qr(np.random.default_rng(50).normal(size=(16,16)))[0]
    z,a=fit_transform(x,r,p);zz,b=fit_transform(x,r,p@v)
    ba,ca=fit_head(z,y);bb,cb=fit_head(zz,y)
    np.testing.assert_allclose(z@ba+ca,zz@bb+cb,atol=1e-10)


def test_no_label_argument_or_heldout_refit():
    assert tuple(inspect.signature(fit_transform).parameters)==('x','reference','projection')
    x,r,y=fixture();z,s=fit_transform(x,r,make_projection(42));before=s['whitening'].copy()
    transform(np.full((5,216),1e6),s)
    np.testing.assert_array_equal(before,s['whitening']);np.testing.assert_array_equal(z,transform(x,s))


def test_zero_source_rank_is_reported_not_fabricated():
    x,r,y=fixture();x[:,24:]=0;r['src_mean']=np.zeros(192);r['src_scale']=np.ones(192)
    z,s=fit_transform(x,r,make_projection(42));assert not s['rank16_valid']
    np.testing.assert_array_equal(z[:,24:],np.zeros((len(x),16)))


def test_invalid_dimensions_and_seed_rejected():
    x,r,y=fixture()
    with pytest.raises(ValueError):make_projection(45)
    with pytest.raises(ValueError):fit_transform(x[:,:200],r,make_projection(42))
    with pytest.raises(ValueError):fit_transform(x,r,np.ones((192,16)))
