import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sample_graph import graph,energies,pool,qualify,K


def test_original_adjacency_contract():
    path=Path(__file__).resolve().parents[2]/'sirna_benchmark_20261007/benchmark_core.py'
    spec=importlib.util.spec_from_file_location('_r46_test_benchmark',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    rng=np.random.default_rng(17);alphabet=np.array(list('ACGU'))
    guides=[''.join(rng.choice(alphabet,19)) for _ in range(9)]
    targets=[''.join(rng.choice(alphabet,57)) for _ in range(9)]
    guides[1]=guides[0][:-2]+''.join('A' if v!='A' else 'C' for v in guides[0][-2:])
    targets[3]=targets[3][:12]+mod.reverse_complement(targets[2][7:32])+targets[3][37:]
    f=pd.DataFrame(dict(guide=guides,target57=targets));actual=mod.build_adjacency(f)
    rc=lambda s:s.translate(str.maketrans('ACGU','UGCA'))[::-1]
    tokens=[{min(t[k:k+25],rc(t[k:k+25])) for k in range(33)} for t in targets]
    expected=np.array([[i!=j and (sum(a!=b for a,b in zip(guides[i],guides[j]))<=2 or bool(tokens[i]&tokens[j])) for j in range(9)] for i in range(9)])
    np.testing.assert_array_equal(actual,expected);assert actual[0,1] and actual[2,3]


def test_ties_and_forbidden():
    n=15;e=~np.eye(n,dtype=bool);e[0,1]=e[1,0]=False;ids=[f'id{i:02}' for i in range(n)]
    g=graph(np.zeros((n,2)),e,ids)
    assert g[0].tolist()==list(range(2,12));assert g[1].tolist()==list(range(2,12))


def test_permutation_equivariance():
    rng=np.random.default_rng(4);x=rng.normal(size=(24,3));ids=np.array([f'r{i:03}' for i in range(24)])
    e=~np.eye(24,dtype=bool);g=graph(x,e,ids);p=rng.permutation(24);gp=graph(x[p],e[np.ix_(p,p)],ids[p])
    for i,old in enumerate(p):assert ids[p][gp[i]].tolist()==ids[g[old]].tolist()


def test_graph_ignores_responses():
    rng=np.random.default_rng(6);x=rng.normal(size=(30,5));e=~np.eye(30,dtype=bool);ids=[str(i) for i in range(30)]
    g=graph(x,e,ids);a=g.copy()
    energies(rng.normal(size=30),e,g);energies(rng.normal(size=30)*20,e,g)
    np.testing.assert_array_equal(g,a);np.testing.assert_array_equal(graph(x,e,ids),a)


def test_energy_explicit():
    x=np.arange(15.)[:,None];e=~np.eye(15,dtype=bool);g=graph(x,e,[str(i) for i in range(15)]);y=x[:,0]**2
    s,kn,nu=energies(y,e,g)
    expected=[sum((y[i]-y[j])**2 for j in g[i])/K for i in range(15)]
    np.testing.assert_allclose(kn,expected)
    np.testing.assert_allclose(nu,[sum((y[i]-y[j])**2 for j in range(15) if j!=i)/14 for i in range(15)])
    assert s['active_rows']==15 and s['ratio']==pytest.approx(sum(expected)/sum(nu))


def test_empty_graph():
    e=np.zeros((3,3),bool);g=graph(np.ones((3,2)),e,['a','b','c']);s,k,n=energies(np.arange(3.),e,g)
    assert (g==-1).all() and s['ratio'] is None and s['active_rows']==0 and np.isnan(k).all() and np.isnan(n).all()


def test_constant_response():
    e=~np.eye(12,dtype=bool);g=graph(np.arange(12.)[:,None],e,[str(i) for i in range(12)])
    s,_,_=energies(np.ones(12),e,g);assert s['ratio'] is None and s['null_sum']==0


def test_pool_ratio_not_mean_ratios():
    a=dict(n=10,active_rows=10,knn_sum=2.,null_sum=10.,coverage=1.,coverage_ge5=1.,ratio=.2)
    b=dict(n=20,active_rows=20,knn_sum=18.,null_sum=20.,coverage=1.,coverage_ge5=1.,ratio=.9)
    p=pool([dict(n=10,residual=a,response=a),dict(n=20,residual=b,response=b)])
    assert p['residual']['ratio']==pytest.approx(2/3) and p['residual']['knn_mean']==pytest.approx(2/3)


def test_qualification_thresholds():
    records=[]
    for setting,r in zip(('Hu','Mix','Tak'),(.80,.89,1.1)):
        for condition,ratio in (('core_real',r),('TD_only',1.),('full_real',.8),('initial_core',.7)):
            records.append(dict(setting=setting,condition=condition,aggregate=dict(residual=dict(ratio=ratio),blocks_below_null=4,coverage_qualified=True)))
    assert qualify(records)['qualified']
    records[4]['aggregate']['blocks_below_null']=3;assert not qualify(records)['qualified']
    records[4]['aggregate']['blocks_below_null']=4;records[-1]['aggregate']['coverage_qualified']=False
    assert not qualify(records)['qualified']


def test_invalid_inputs():
    e=~np.eye(3,dtype=bool)
    with pytest.raises(ValueError):graph(np.ones((3,2)),e,['x','x','z'])
    with pytest.raises(ValueError):graph(np.full((3,2),np.nan),e,['x','y','z'])
    with pytest.raises(ValueError):graph(np.ones((3,2)),np.ones((3,3),bool),['x','y','z'])
