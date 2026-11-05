import numpy as np
import pytest
from scipy.linalg import null_space
from evidence import prepare, evaluate, fit, predict, projected_gradient, LOW, HIGH


def fixture():
    r = np.random.default_rng(42)
    x = r.normal(size=(48, 7)) + np.arange(7)
    y = .2*x[:, 0]-.5*x[:, 4]+r.normal(size=48)*.2+.7
    return x, y


def test_independent_contrast_covariance():
    x, y = fixture(); d = prepare(x, y, (3, 4)); theta = np.log([2., 17.])
    s = evaluate(theta, d)
    h = null_space(np.ones((1, len(x))))
    z, yc = h.T @ x, h.T @ y
    precision = np.repeat(np.exp(theta), (3, 4))
    covariance = np.eye(len(x)-1) + (z/precision) @ z.T
    sign, logdet = np.linalg.slogdet(covariance); assert sign == 1
    rss = yc @ np.linalg.solve(covariance, yc)
    f = .5*((len(x)-1)*np.log(rss/(len(x)-1))+logdet)
    np.testing.assert_allclose(s['objective'], f, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(s['rss'], rss, atol=1e-10, rtol=1e-10)


def test_analytic_gradient():
    x, y = fixture(); d = prepare(x, y, (3, 4)); theta = np.log([10., 100.])
    s = evaluate(theta, d)
    for j in range(2):
        step = np.eye(2)[j]*1e-5
        fd = (evaluate(theta+step, d)['objective']-evaluate(theta-step, d)['objective'])/2e-5
        np.testing.assert_allclose(s['gradient'][j], fd, atol=1e-7, rtol=1e-7)


def test_one_block_normal_equation():
    x, y = fixture(); d = prepare(x, y, (7,)); s = evaluate(np.log([100.]), d)
    expected = np.linalg.solve(d['z'].T @ d['z']+100*np.eye(7), d['z'].T @ d['yc'])
    np.testing.assert_allclose(s['beta'], expected, atol=1e-12, rtol=1e-12)


def test_fit_and_initial_saved_before_update():
    x, y = fixture(); calls=[]
    s = fit(x, y, (3,4), initial=lambda *args:calls.append(args))
    assert len(calls)==1 and s['optimizer_success'] and s['numerically_qualified']
    np.testing.assert_array_equal(calls[0][0], np.full(2, np.log(100.)))
    assert s['objective'] <= evaluate(calls[0][0],prepare(x,y,(3,4)))['objective']+1e-10
    assert s['iterations'] <= 1000


def test_affine_response_and_translation_invariance():
    x,y = fixture(); a = fit(x,y,(3,4)); b = fit(x+8,3*y-2,(3,4))
    np.testing.assert_allclose(a['theta'],b['theta'],atol=2e-6,rtol=2e-6)
    np.testing.assert_allclose(predict(x+8,b),3*predict(x,a)-2,atol=1e-7,rtol=1e-7)


def test_block_rotation_invariance():
    x,y=fixture(); q=np.linalg.qr(np.random.default_rng(19).normal(size=(4,4)))[0]
    z=x.copy();z[:,3:]=x[:,3:]@q
    a=evaluate(np.log([3.,40.]),prepare(x,y,(3,4)));b=evaluate(np.log([3.,40.]),prepare(z,y,(3,4)))
    np.testing.assert_allclose(a['objective'],b['objective'],atol=1e-10)
    np.testing.assert_allclose(a['gradient'],b['gradient'],atol=1e-10)


def test_bound_kkt_direction():
    np.testing.assert_array_equal(projected_gradient([LOW, HIGH, LOW, HIGH, 0],[1.,-1.,-2.,2.,3.]),[0,0,-2,2,3])


def test_prediction_does_not_fit_or_mutate():
    x,y=fixture(); s=fit(x,y,(3,4)); before=s['beta'].copy(); p=predict(x[:2],s)
    predict(np.full((10,7),1e6),s)
    np.testing.assert_array_equal(s['beta'],before);np.testing.assert_array_equal(p,predict(x[:2],s))


def test_invalid_or_constant_labels_rejected():
    x,y=fixture()
    with pytest.raises(ValueError):prepare(x,np.ones_like(y),(3,4))
    with pytest.raises(ValueError):prepare(x,y,(3,3))
