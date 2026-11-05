import time
import numpy as np
import pytest
from scipy.optimize import minimize
from epsilon_kernel import centered_soft, soft, solve, certificate, qualify


def data():
    rng = np.random.default_rng(20261015)
    x = rng.normal(size=(37, 6)); x -= x.mean(0)
    y = .3 + x @ rng.normal(size=6) + rng.normal(size=len(x)) * .2
    return x, y


def test_zero_sum_prox_independent_constrained_optimization():
    v = np.array([-1.2, .7, 2.1, -.3, .9, 1.])
    u, t = centered_soft(v, .23)
    np.testing.assert_allclose(u, soft(v-t, .23), atol=1e-14)
    assert abs(u.sum()) < 1e-13
    n = len(v)
    objective = lambda a: .5*np.sum((a[:n]-v)**2)+.23*np.sum(a[n:])
    constraints = [dict(type='eq', fun=lambda a: a[:n].sum()),
        dict(type='ineq', fun=lambda a: a[n:]-a[:n]), dict(type='ineq', fun=lambda a: a[n:]+a[:n])]
    result = minimize(objective, np.r_[u, abs(u)], constraints=constraints, method='SLSQP', options={'ftol':1e-12,'maxiter':1000})
    assert result.success
    np.testing.assert_allclose(result.x[:n], u, atol=1e-6)


def test_epsilon_zero_recovers_kernel_ridge():
    x, y = data(); k = x@x.T
    u, c = solve(k, y, 0.)
    exact = 100*np.linalg.solve(k+100*np.eye(len(y)), y-y.mean())
    assert c['numerically_qualified']
    np.testing.assert_allclose(u, exact, atol=2e-5)
    assert abs(c['intercept']-y.mean()) < 1e-12


def test_independent_primal_bfgs_and_duality():
    x, y = data(); eps = .1*np.std(y)
    u, c = solve(x@x.T, y, eps)
    def fg(a):
        r = y-x@a[:-1]-a[-1]; s = soft(r, eps)
        return .5*s@s+50*a[:-1]@a[:-1], np.r_[-x.T@s+100*a[:-1], -s.sum()]
    opt = minimize(fg, np.r_[np.zeros(x.shape[1]), y.mean()], method='BFGS', jac=True, options={'gtol':1e-9,'maxiter':1000})
    assert np.max(abs(fg(opt.x)[1])) < 1e-6
    np.testing.assert_allclose(x.T@u/100, opt.x[:-1], atol=3e-5)
    assert abs(c['intercept']-opt.x[-1]) < 3e-5
    assert c['numerically_qualified'] and c['gap'] >= -1e-10


def test_response_affine_and_sign_equivariance():
    x,y=data(); k=x@x.T; u,c=solve(k,y,.1*np.std(y))
    for scale,shift in [(2.,7.),(-1.,.5)]:
        v,d=solve(k,scale*y+shift,.1*np.std(scale*y+shift))
        np.testing.assert_allclose(v,scale*u,atol=3e-5)
        assert abs(d['intercept']-(scale*c['intercept']+shift))<3e-5


def test_row_permutation_and_no_validation_argument():
    x,y=data(); k=x@x.T; eps=.1*np.std(y); u,c=solve(k,y,eps)
    p=np.random.default_rng(42).permutation(len(y)); v,d=solve(k[np.ix_(p,p)],y[p],eps)
    np.testing.assert_allclose(v,u[p],atol=1e-10)
    assert abs(d['intercept']-c['intercept'])<1e-10
    # Out-of-training labels/inputs have no place in the solver signature.
    with pytest.raises(TypeError): solve(k,y,eps,validation_y=y)


def test_zero_kernel_and_constant_response():
    y=np.array([-.3,.1,.2,.9]); k=np.zeros((4,4)); u,c=solve(k,y,.04)
    assert c['numerically_qualified']
    np.testing.assert_allclose(u,soft(y-c['intercept'],.04),atol=1e-12)
    u,c=solve(k,np.ones(4)*.7,.1)
    assert c['numerically_qualified'] and np.max(abs(u))==0
    assert np.max(abs(.7-c['intercept']))<=.1+1e-12


def test_certificate_rejects_nonoptimal_and_deadline_contracts():
    x,y=data(); k=x@x.T
    assert not certificate(k,y,np.zeros(len(y)),.1)['numerically_qualified']
    with pytest.raises(TimeoutError): solve(k,y,.1,deadline=time.monotonic()-1)
    for eps in [-1.,np.nan]:
        with pytest.raises(ValueError): solve(k,y,eps)
    with pytest.raises(ValueError): solve(k[:,1:],y,.1)
    with pytest.raises(ValueError): solve(k+np.triu(np.ones_like(k)),y,.1)


def test_gate_keeps_loss_control_and_all_primary_pools():
    rows=[]
    for s in ('Huesken','Mixset','Takayuki'):
        rows += [dict(source='core_real',setting=s,delta_PCC_vs_R4=.006,MSE_reduction_vs_R4_pct=.1,
            delta_PCC_vs_R33=.001,MSE_reduction_vs_R33_pct=.1,metrics=dict(PCC=.6,MSE=.09)),
            dict(source='TD_only',setting=s,metrics=dict(PCC=.59,MSE=.1))]
    assert qualify(rows,True)['qualified']
    for r in rows:
        if r['source']=='core_real':r['MSE_reduction_vs_R33_pct']=-.01
    assert not qualify(rows,True)['qualified']
