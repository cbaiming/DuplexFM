import numpy as np
import pytest
from scipy.optimize import approx_fprime
from mlp import initialize,unpack,loss_grad,forward,fit,predict


def test_full_gradient_finite_difference():
    rng=np.random.default_rng(7); z=rng.normal(size=(21,40)); y=rng.normal(size=21); p=initialize(40)
    analytic=loss_grad(p,z,y)[1]
    numeric=approx_fprime(p,lambda q:loss_grad(q,z,y)[0],1e-7)
    np.testing.assert_allclose(analytic,numeric,rtol=2e-5,atol=3e-7)


def test_shared_initialization_and_counts():
    a=unpack(initialize(40),40); b=unpack(initialize(24),24)
    assert len(initialize(40))==673 and len(initialize(24))==417
    np.testing.assert_array_equal(a[0][:24],b[0]); np.testing.assert_array_equal(a[1],b[1]); np.testing.assert_array_equal(a[2],b[2])
    np.testing.assert_array_equal(initialize(40),initialize(40))


def test_forward_and_batch_independence():
    z=np.random.default_rng(1).normal(size=(10,40)); p=initialize(40); w,b,a,c=unpack(p,40)
    direct=np.array([sum(np.tanh(row@w+b)*a)+c for row in z])
    np.testing.assert_allclose(forward(z,p),direct,atol=1e-14)
    np.testing.assert_allclose(forward(z[:3],p),forward(z,p)[:3],atol=1e-14)


def test_training_reproducible_and_target_scaling():
    rng=np.random.default_rng(10); z=rng.normal(size=(50,24)); y=.4+.1*np.tanh(z[:,0]-z[:,1]); saved=[]
    a=fit(z,y,save_initial=lambda t,m,s:saved.append((t.copy(),m,s)))
    b=fit(z,2*y+3)
    assert a['qualified_numerically'] and b['qualified_numerically'] and len(saved)==1
    np.testing.assert_array_equal(saved[0][0],a['initial'])
    np.testing.assert_allclose(predict(z,b),2*predict(z,a)+3,rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(a['y_mean'],y.mean()); np.testing.assert_allclose(a['y_scale'],y.std())


def test_invalid_inputs():
    with pytest.raises(ValueError): fit(np.ones((8,25)),np.ones(8))
    z=np.ones((8,24)); z[0,0]=np.nan
    with pytest.raises(ValueError): fit(z,np.ones(8))


def test_constant_target_finite():
    z=np.random.default_rng(8).normal(size=(50,24)); y=np.full(50,.42)
    a=fit(z,y)
    assert a['qualified_numerically'] and a['y_scale']==1.
    np.testing.assert_allclose(predict(z,a),.42,atol=1e-5)
