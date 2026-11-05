import copy
import numpy as np
import pytest
from stump import gates, design, fit, predict, branch_predict, choose


def sample(p=40):
    rng = np.random.default_rng(42)
    z = rng.normal(size=(120, p)); z -= z.mean(0)
    y = .2*z[:, 1] + (z[:, 0] > 0)*(.8*z[:, -1]) + .01*rng.normal(size=len(z))
    return z, y


def test_gate_support_thresholds():
    z, _ = sample(); candidates = gates(z)
    assert len(candidates) == 73 and candidates[0]['feature'] == -1
    for g in candidates[1:]:
        assert min(g['counts']) >= 24
        assert g['threshold'] == np.quantile(z[:, g['feature']], (.25,.5,.75)[g['quantile_index']])


def test_constants_and_quantile_deduplication():
    z = np.zeros((100, 24)); z[50:, 0] = 1; z -= z.mean(0)
    gs = gates(z)
    assert len(gs) == 3 and gs[1]['threshold'] != gs[2]['threshold']
    assert all(g['feature'] in (-1, 0) for g in gs)


@pytest.mark.parametrize('p', [24, 40])
def test_normal_and_independent_augmented_solve(p):
    z, y = sample(p); m = fit(z, y)
    assert m['all_certified']
    k = m['selected']; x, _ = design(z, m['candidates'][k]); q = x.shape[1]-p
    penalty = np.r_[np.full(p,100.),np.full(q,400.)]
    b = np.linalg.lstsq(np.r_[x,np.diag(np.sqrt(penalty))],np.r_[y-y.mean(),np.zeros(x.shape[1])],rcond=None)[0]
    np.testing.assert_allclose(b,m['beta'][k],atol=1e-12,rtol=1e-12)


def test_global_split_choice_training_objective():
    z,y=sample();m=fit(z,y)
    assert m['selected']==choose(m['candidates'])
    assert m['candidates'][m['selected']]['objective'] <= m['candidates'][0]['objective']+1e-10


def test_exact_reference_reuse():
    z,y=sample();b=np.linalg.solve(z.T@z+100*np.eye(40),z.T@(y-y.mean()))
    m=fit(z,y,reference_beta=b)
    assert m['reference_reused']==1 and m['new_linear_solves']==len(m['candidates'])-1
    np.testing.assert_array_equal(m['beta'][0,:40],b)
    assert np.count_nonzero(m['beta'][0,40:])==0


def test_onehot_deployment_and_readonly_state():
    z,y=sample();m=fit(z,y);original=copy.deepcopy(m);new=z[:15]*2+.3
    p,route=branch_predict(new,m)
    np.testing.assert_allclose(p,predict(new,m),atol=1e-12,rtol=1e-12)
    assert set(route).issubset({0,1})
    np.testing.assert_array_equal(m['interaction_mean'],original['interaction_mean'])
    assert m['candidates']==original['candidates']


def test_tie_prefers_null_and_index():
    c=[dict(objective=1.,feature=-1,quantile_index=-1),dict(objective=1.+1e-12,feature=0,quantile_index=0)]
    assert choose(c)==0


def test_null_is_plain_ridge_with_center():
    z=np.zeros((40,24));y=np.arange(40,dtype=float);m=fit(z,y)
    assert len(m['candidates'])==1 and m['selected']==0
    np.testing.assert_array_equal(predict(z,m),np.full(40,y.mean()))


def test_reject_bad_contracts():
    z,y=sample()
    with pytest.raises(ValueError): fit(z+1,y)
    with pytest.raises(ValueError): fit(z,np.ones(len(y)))
    with pytest.raises(ValueError): fit(z,y,reference_beta=np.zeros(3))
    with pytest.raises(ValueError): gates(np.zeros((19,24)))
    with pytest.raises(ValueError): predict(np.full((2,40),np.nan),fit(z,y))
