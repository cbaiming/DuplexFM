import numpy as np
import pytest
from scipy.optimize import lsq_linear
from spectral_refit import basis_from_previous,nonnegative_quadratic,fit,qualify,ALPHA


def fixture(seed=17):
    rng=np.random.default_rng(seed);z=rng.normal(size=(90,216));z-=z.mean(0)
    y=rng.normal(size=len(z));b=rng.normal(size=216)*.01
    return z,y,b


def test_basis_reconstruction_and_orthogonality():
    _,_,b=fixture();d,u,s,vh=basis_from_previous(b)
    np.testing.assert_allclose(d.T@d,np.eye(6),atol=1e-12)
    np.testing.assert_allclose(d@s,b[24:],atol=1e-12)
    np.testing.assert_allclose(u@np.diag(s)@vh,b[24:].reshape(6,32),atol=1e-12)


def test_rank_and_pair_sign_invariance():
    b=np.zeros(216);b[24]=3.;b[24+33]=1.;b[24+66]=1e-10
    d,u,s,vh=basis_from_previous(b);assert len(s)==2
    sign=np.array([-1.,1.]);dd=np.column_stack([np.outer((u*sign)[:,j],(vh*sign[:,None])[j]).ravel() for j in range(2)])
    np.testing.assert_allclose(d,dd,atol=0)


def test_diagonal_constrained_solution():
    h=np.diag([2.,3.,4.]);q=np.array([2.,-3.,8.]);d,info=nonnegative_quadratic(h,q)
    np.testing.assert_allclose(d,[1.,0.,2.]);assert info['enumerated_active_sets']==8 and info['active_system_solves']==7


def test_correlated_solution_KKT():
    h=np.array([[2.,1.],[1.,2.]]);q=np.array([1.,-.5]);d,info=nonnegative_quadratic(h,q)
    np.testing.assert_allclose(d,[.5,0]);assert info['profile_KKT_max']<1e-12


def test_zero_source_is_TD_Ridge():
    z,y,_=fixture();b=np.zeros(216);m,c=fit(z,y,b)
    expected=np.linalg.solve(z[:,:24].T@z[:,:24]+100*np.eye(24),z[:,:24].T@(y-y.mean()))
    np.testing.assert_allclose(m['beta'][:24],expected,atol=1e-12)
    assert not np.any(m['beta'][24:]) and c['selected_rank']==0 and c['certified']


def test_augmented_BVLS_independence():
    z,y,b=fixture();m,c=fit(z,y,b);x=np.c_[z[:,:24],z[:,24:]@m['basis']];p=x.shape[1]
    result=lsq_linear(np.r_[x,10*np.eye(p)],np.r_[y-y.mean(),np.zeros(p)],bounds=(np.r_[np.full(24,-np.inf),np.zeros(p-24)],np.full(p,np.inf)),method='bvls',tol=1e-12)
    assert result.success;np.testing.assert_allclose(m['reduced_coef'],result.x,atol=1e-10);assert c['certified']


def test_fixed_direction_objective_not_worse():
    z,y,b=fixture(72);m,c=fit(z,y,b)
    assert c['objective_without_trace']<=c['reference_objective_without_trace']+1e-10
    assert c['conditional_suboptimality_upper']<1e-14 and (m['amplitudes']>=0).all()


def test_native_and_reduced_predictions_match():
    z,y,b=fixture();m,c=fit(z,y,b);x=np.c_[z[:,:24],z[:,24:]@m['basis']]
    np.testing.assert_allclose(z@m['beta'],x@m['reduced_coef'],atol=1e-12)
    assert c['deployment_parameter_count']==217 and c['supervised_basis_from_old_training_head']


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):basis_from_previous(np.zeros(192))
    with pytest.raises(ValueError):nonnegative_quadratic(-np.eye(2),np.ones(2))
    with pytest.raises(ValueError):nonnegative_quadratic(np.eye(7),np.ones(7))
    z,y,b=fixture()
    with pytest.raises(ValueError):fit(z+2,y,b)
    with pytest.raises(ValueError):fit(z,np.ones(len(y)),b)


def test_gate_fixed():
    records=[dict(setting=s,source='core_real',delta_PCC_vs_R25=.006,MSE_reduction_vs_R25_pct=.1,
                  metrics=dict(PCC=.6,MSE=.02),TD_metrics=dict(PCC=.5,MSE=.03)) for s in ('Hu','Mix','Tak')]
    assert qualify(records,True)['qualified']
    records[0]['delta_PCC_vs_R25']=-.006;assert not qualify(records,True)['qualified']
    assert not qualify(records,False)['qualified']
