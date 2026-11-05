import numpy as np
import pytest
from soft_route import initial_gate,design,profile,fit,predict,branch_predict


def sample(dim=40):
    rng=np.random.default_rng(5);z=rng.normal(size=(150,dim));z-=z.mean(0)
    y=.2*z[:,0]+.3*np.tanh(z[:,1]-z[:,2])*z[:,-1]+rng.normal(size=len(z))*.03
    return z,y


def test_profile_matches_augmented_lstsq():
    z,y=sample();a=initial_gate();value,gradient,m=profile(a,z,y);x,_,_,_=design(z,a)
    penalty=np.r_[np.full(40,100.),np.full(16,400.)]
    expected=np.linalg.lstsq(np.r_[x,np.diag(np.sqrt(penalty))],np.r_[y-y.mean(),np.zeros(56)],rcond=None)[0]
    np.testing.assert_allclose(m['beta'],expected,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(x.mean(0),0,atol=1e-12)


def test_profile_gradient_all_coordinates():
    z,y=sample();a=initial_gate();_,gradient,_=profile(a,z,y);numeric=[]
    for k in range(25):
        step=np.zeros(25);step[k]=1e-5
        numeric.append((profile(a+step,z,y)[0]-profile(a-step,z,y)[0])/2e-5)
    np.testing.assert_allclose(gradient,numeric,atol=1e-8,rtol=1e-6)


def test_zero_gate_is_original_ridge():
    z,y=sample();_,g,m=profile(np.zeros(25),z,y)
    expected=np.linalg.solve(z.T@z+100*np.eye(40),z.T@(y-y.mean()))
    np.testing.assert_allclose(m['beta'][:40],expected,atol=1e-12)
    assert not np.count_nonzero(m['beta'][40:]) and not np.count_nonzero(g)


def test_sign_symmetry_and_branch_equivalence():
    z,y=sample();a=initial_gate();v,grad,m=profile(a,z,y);v2,g2,m2=profile(-a,z,y)
    m.update(gate=a);m2.update(gate=-a)
    np.testing.assert_allclose(v,v2,atol=1e-14)
    np.testing.assert_allclose(grad,-g2,atol=1e-12)
    np.testing.assert_allclose(predict(z,m)[0],predict(z,m2)[0],atol=1e-12)
    np.testing.assert_allclose(predict(z,m)[0],branch_predict(z,m)[0],atol=1e-12)


def test_affine_response_invariance():
    z,y=sample();a=initial_gate();v,g,m=profile(a,z,y);v2,g2,m2=profile(a,z,3*y+2)
    np.testing.assert_allclose(v,v2,atol=1e-12);np.testing.assert_allclose(g,g2,atol=1e-12)
    np.testing.assert_allclose(m2['beta'],3*m['beta'],atol=1e-12)
    assert m2['intercept']==pytest.approx(3*m['intercept']+2)


def test_deterministic_local_convergence_and_saved_initial():
    z,y=sample();saved=[];m=fit(z,y,save_initial=saved.append);m2=fit(z,y)
    assert m['numerically_qualified'] and m2['numerically_qualified']
    assert m['objective']<=m['initial_objective']+1e-12
    np.testing.assert_array_equal(saved[0],initial_gate());np.testing.assert_array_equal(m['gate'],m2['gate'])
    assert m['new_linear_solves']==m['function_evaluations']+2


def test_td_control_layout_and_frozen_inference():
    z,y=sample(24);a=initial_gate();_,_,m=profile(a,z,y);m.update(gate=a)
    assert m['beta'].shape==(48,)
    before={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in m.items()}
    one=predict(z[:1],m)[0];batch=predict(np.r_[z[:1],100*z[1:]],m)[0]
    np.testing.assert_allclose(one,batch[:1],atol=1e-12)
    np.testing.assert_allclose(predict(z,m)[0],branch_predict(z,m)[0],atol=1e-12)
    for k,v in m.items():np.testing.assert_array_equal(v,before[k])


def test_input_contract():
    z,y=sample()
    with pytest.raises(ValueError):profile(initial_gate(),z+1,y)
    with pytest.raises(ValueError):profile(initial_gate(),z,np.ones(len(y)))
    with pytest.raises(ValueError):design(z,np.zeros(24))
    with pytest.raises(ValueError):design(z[:,:20],initial_gate())
