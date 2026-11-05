import copy
import numpy as np
import pytest
from quadratic import raw_terms,fit_map,transform,fit,direct_predict,PAIRS,FACTOR


def sample(n=100):
    z=np.random.default_rng(42).normal(size=(n,40));return z-z.mean(0)


def test_layout_and_only_source():
    z=sample();q=raw_terms(z)
    assert q.shape==(100,136) and len(PAIRS)==136
    np.testing.assert_array_equal(q,np.stack([z[:,24+i]*z[:,24+j] for i,j in PAIRS],axis=1))
    other=z.copy();other[:,:24]+=100
    np.testing.assert_array_equal(q,raw_terms(other))


def test_scaled_variance_and_preserved_base():
    z=sample();x,t=fit_map(z)
    np.testing.assert_array_equal(x[:,:40],z)
    np.testing.assert_allclose(x[:,40:].mean(0),0.,atol=1e-15)
    np.testing.assert_allclose(x[:,40:].var(0).sum(),16.,atol=1e-12)


def test_readonly_validation_state():
    z=sample();_,t=fit_map(z);before=copy.deepcopy(t)
    other=z[:7]*3+5;actual=transform(other,t)
    np.testing.assert_array_equal(actual[:,40:],(raw_terms(other)-t['mean'])/t['scale']*FACTOR)
    np.testing.assert_array_equal(t['mean'],before['mean']);np.testing.assert_array_equal(t['scale'],before['scale'])


def test_constant_products_safe():
    z=np.zeros((30,40));x,t=fit_map(z)
    assert np.isfinite(x).all() and np.count_nonzero(x)==0
    np.testing.assert_array_equal(t['scale'],np.ones(136))


def test_symmetric_deployment():
    z=sample();x,t=fit_map(z);b=np.random.default_rng(43).normal(size=176)
    other=z[:8]*1.2
    np.testing.assert_allclose(direct_predict(other,t,b,.3),transform(other,t)@b+.3,atol=1e-12,rtol=1e-12)


def test_unique_joint_ridge_solution():
    z=sample();x,t=fit_map(z);y=z[:,1]+z[:,25]*z[:,26]
    b,c,cert=fit(x,y)
    independent=np.linalg.lstsq(np.r_[x,10*np.eye(176)],np.r_[y-y.mean(),np.zeros(176)],rcond=None)[0]
    np.testing.assert_allclose(b,independent,atol=1e-12,rtol=1e-12)
    assert cert['numerically_qualified'] and c==float(y.mean())


def test_bilinear_source_signal_is_in_map():
    z=sample(300);x,t=fit_map(z);k=PAIRS.index((2,6));b=np.zeros(176);b[40+k]=t['scale'][k]/FACTOR
    np.testing.assert_allclose(x@b+t['mean'][k],z[:,26]*z[:,30],atol=1e-12)


def test_invalid_contracts():
    z=sample();x,t=fit_map(z)
    with pytest.raises(ValueError):raw_terms(np.zeros((5,24)))
    with pytest.raises(ValueError):fit_map(z[:19])
    with pytest.raises(ValueError):fit(x,np.ones(len(x)))
    with pytest.raises(ValueError):fit(x+1,np.arange(len(x)))
    bad=copy.deepcopy(t);bad['scale'][0]=0
    with pytest.raises(ValueError):transform(z,bad)
