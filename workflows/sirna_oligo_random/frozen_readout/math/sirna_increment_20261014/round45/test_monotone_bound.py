import numpy as np
import pytest
from scipy.optimize import minimize,isotonic_regression
from monotone_bound import project,qualify


def example():
    rng=np.random.default_rng(42);s=np.arange(24,dtype=float)
    y=.2+.6*(s/23.)**2+rng.normal(size=len(s))*.12
    return s,y


def test_known_pool_adjacent_violators():
    p,state=project([1,2,3,4],[3,1,4,2])
    np.testing.assert_allclose(p,[2,2,3,3],atol=1e-12)
    assert state['certificate']['certified'] and not state['deployable'] and state['uses_evaluation_labels']


def test_independent_quadratic_program():
    s,y=example();p,state=project(s,y)
    d=np.diff(np.eye(len(s)),axis=0)
    result=minimize(lambda x:.5*np.sum((x-y)**2),np.full(len(y),y.mean()),jac=lambda x:x-y,
        constraints={'type':'ineq','fun':lambda x:d@x,'jac':lambda x:d},
        method='SLSQP',options={'ftol':1e-12,'maxiter':500})
    assert result.success
    np.testing.assert_allclose(p,result.x,atol=1e-8)
    np.testing.assert_allclose(p,isotonic_regression(y,weights=np.ones(len(y)),increasing=True).x,atol=1e-12)


def test_full_kkt_and_moreau_identity():
    s,y=example();p,t=project(s,y);c=t['certificate']
    assert -1e-10<=c['gap']<=c['threshold'] and c['stationarity_max']<1e-10
    pc=p-p.mean();yc=y-y.mean()
    np.testing.assert_allclose((yc-pc)@pc,0,atol=1e-11)
    np.testing.assert_allclose(np.corrcoef(y,p)[0,1],t['metrics']['upper_PCC'],atol=1e-12)


def test_ties_and_row_permutation():
    s=np.array([3,1,2,1,3,2.]);y=np.array([.7,.3,.2,.1,.5,.4])
    p,t=project(s,y);order=np.array([2,5,0,4,1,3]);pp,tt=project(s[order],y[order])
    np.testing.assert_allclose(p[order],pp,atol=1e-12)
    assert p[0]==p[4] and p[1]==p[3] and p[2]==p[5]
    assert t['metrics']['unique_score_count']==3


def test_positive_affine_score_invariance():
    s,y=example();p,t=project(s,y);pp,tt=project(7*s-19,y)
    np.testing.assert_allclose(pp,p,atol=1e-12)
    np.testing.assert_allclose(tt['metrics']['upper_PCC'],t['metrics']['upper_PCC'],atol=1e-12)


def test_nonlinear_monotone_exact_response_has_ceiling_one():
    s=np.linspace(-1,1,31);y=(s+1)**3
    p,t=project(s,y)
    np.testing.assert_allclose(p,y,atol=1e-12)
    assert t['metrics']['upper_PCC']==pytest.approx(1) and t['metrics']['raw_PCC']<.95


def test_degenerate_and_invalid_inputs():
    p,t=project([1,2,3],[3,2,1]);np.testing.assert_allclose(p,[2,2,2])
    assert t['metrics']['upper_PCC']==0 and not t['metrics']['upper_attained']
    p,t=project([1,1,1],[1,2,3]);assert t['metrics']['raw_PCC'] is None
    for s,y in (([1,2],[1,2]),([1,2,3],[1,1,1]),([1,2,np.nan],[1,2,3]),([[1,2,3]],[1,2,3])):
        with pytest.raises(ValueError):project(s,y)


def test_ceiling_for_random_monotone_scores_and_affine_mse():
    s,y=example();_,t=project(s,y);rng=np.random.default_rng(43)
    for _ in range(100):
        alternative=np.cumsum(rng.exponential(size=len(s)))
        assert np.corrcoef(alternative,y)[0,1]<=t['metrics']['upper_PCC']+1e-12
    expected=np.var(y)*(1-max(t['metrics']['raw_PCC'],0)**2)
    np.testing.assert_allclose(expected,t['metrics']['positive_affine_MSE'],atol=1e-12)


def test_fixed_qualification_semantics():
    records=[]
    for i in range(3):
        records.append(dict(setting=str(i),condition='core_real',metrics=dict(upper_PCC=.65,upper_attained=True,absolute_PCC_room=.02,optimistic_MSE_reduction_vs_affine_pct=2.)))
        records.append(dict(setting=str(i),condition='TD_only',metrics=dict(upper_PCC=.64)))
    assert qualify(records,True)['qualified'] and not qualify(records,False)['qualified']
    records[0]['metrics']['upper_PCC']=.63
    assert not qualify(records,True)['qualified']
    with pytest.raises(ValueError):qualify(records[:2],True)
