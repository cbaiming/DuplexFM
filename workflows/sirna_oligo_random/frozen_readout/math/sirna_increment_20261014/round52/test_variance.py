import numpy as np
import pytest
from variance import halves,sample_stats,direction,diagnose,gate

def test_half_coverage_and_disjoint():
    ids=[f'id{i}' for i in range(25)];g=np.r_[np.zeros(10,int),np.ones(9,int),np.full(6,2)]
    b=halves(ids,g,0);assert len(b)==2
    for x in b:
        assert set(x['A']).isdisjoint(x['B']) and min(len(x['A']),len(x['B']))>=4
        assert sorted(x['A']+x['B'])==np.flatnonzero(g==x['group']).tolist()

def test_half_identity_order_invariant():
    ids=[f'id{i}' for i in range(24)];g=np.repeat(np.arange(3),8);p=np.random.default_rng(42).permutation(24)
    a=halves(ids,g,1);b=halves([ids[i] for i in p],g[p],1)
    for x,y in zip(a,b):
        for key in ('A','B'):assert [ids[i] for i in x[key]]==[ids[p[i]] for i in y[key]]

def test_variance_formula():
    x=np.array([1,2,8,10,4,3.]);s=sample_stats(x,[0,1,2,3]);assert s['df']==3
    np.testing.assert_allclose(s['variance'],np.var(x[:4],ddof=1),atol=1e-13)

def test_prediction_does_not_use_validation():
    a=[dict(SSE=3.,df=3),dict(SSE=30.,df=3)];b=[dict(variance=1.,df=3),dict(variance=10.,df=3)]
    p=direction(a,b);q=direction(a,[dict(variance=100.,df=3),dict(variance=1.,df=3)])
    assert p['predicted_variances']==q['predicted_variances'] and p['global_variance']==q['global_variance']
    assert p['score_improvement']>0

def test_zero_variance_and_empty():
    assert not direction([],[])['defined']
    s=direction([dict(SSE=0.,df=3)],[dict(variance=0.,df=3)]);assert not s['defined']

def test_no_small_group_claim():
    ids=[str(i) for i in range(8)];blocks=halves(ids,[0]*8,0)
    r=diagnose(np.arange(8.),blocks,8);assert not r['structurally_eligible'] and not r['qualified']

def test_gate_requires_two_pools_both_layouts():
    rec=[dict(setting=s,source='core_real',layout=l,qualified=(s=='Huesken')) for s in ('Huesken','Mixset','Takayuki') for l in (0,1)]
    assert not gate(rec)['qualified']
    for r in rec:
        if r['setting']=='Mixset':r['qualified']=True
    assert gate(rec)['qualified'];rec[3]['qualified']=False;assert not gate(rec)['qualified']

def test_invalid_input_rejected():
    with pytest.raises(ValueError):halves(['a']*8,[0]*8,0)
    with pytest.raises(ValueError):halves([str(i) for i in range(8)],[0]*8,2)
    with pytest.raises(ValueError):sample_stats(np.ones(3),[0,1,2])
