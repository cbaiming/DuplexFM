import numpy as np
from nuclear import shrink,solve,fit_map,transform,fit_pair,ALPHA

def test_singular_threshold():
    matrix=np.diag([5.,2.,.1]); np.testing.assert_allclose(shrink(matrix,1.),np.diag([4.,1.,0.]),atol=1e-13)

def test_identity_hessian_exact_solution():
    rng=np.random.default_rng(1); q=rng.normal(size=12); Q=3*np.eye(12); lam=.3
    c,r=solve(Q,q,(4,3),lam)
    np.testing.assert_allclose(c.reshape(4,3),shrink(q.reshape(4,3)/3,lam/3),atol=1e-12)
    assert r['certified'] and r['gap']<1e-12

def test_critical_zero_solution():
    q=np.arange(1.,13); Q=np.diag(np.arange(1.,13)); lam=np.linalg.norm(q.reshape(4,3),2)
    c,r=solve(Q,q,(4,3),lam)
    np.testing.assert_allclose(c,0.,atol=1e-12); assert r['certified']

def test_lambda_zero_joint_ridge_and_gap():
    rng=np.random.default_rng(4); z=rng.normal(size=(50,40)); y=rng.normal(size=50)
    (x,j),t=fit_map(z,'source'); outputs=fit_pair(x,j,y,t['matrix_shape']); a=np.c_[x,j]
    expected=np.linalg.solve(a.T@a+ALPHA*np.eye(a.shape[1]),a.T@(y-y.mean()))
    np.testing.assert_allclose(np.r_[outputs['Ridge']['b'],outputs['Ridge']['c']],expected,atol=1e-11)
    assert outputs['Nuclear']['certified']

def test_map_center_scale_and_external_independence():
    z=np.random.default_rng(8).normal(size=(60,40)); (x,j),t=fit_map(z,'source')
    np.testing.assert_allclose(j.var(0).sum(),16.,atol=1e-12); np.testing.assert_allclose(j.mean(0),0.,atol=1e-12)
    v=z[:4]*2; jv=transform(v,t)[1]; transform(np.r_[v,z*10],t)
    np.testing.assert_array_equal(jv,transform(v,t)[1])

def test_symmetric_td_matrix():
    rng=np.random.default_rng(9); z=rng.normal(size=(50,24)); y=rng.normal(size=50); (x,j),t=fit_map(z,'TD24')
    results=fit_pair(x,j,y,t['matrix_shape'])
    for r in results.values():
        np.testing.assert_allclose(r['c'].reshape(24,24),r['c'].reshape(24,24).T,atol=1e-10)
        assert r['certified']

def test_independent_dual_gap_and_feasibility():
    rng=np.random.default_rng(31); a=rng.normal(size=(12,12)); Q=a.T@a+2*np.eye(12); q=rng.normal(size=12); lam=.2
    c,r=solve(Q,q,(4,3),lam); dual=r['dual']; assert np.linalg.norm(dual.reshape(4,3),2)<=lam+1e-12
    primal=.5*c@Q@c-q@c+lam*np.linalg.svd(c.reshape(4,3),compute_uv=False).sum()
    gap=primal+.5*(q-dual)@np.linalg.solve(Q,q-dual)
    np.testing.assert_allclose(gap,r['gap'],atol=1e-12); assert r['certified']
