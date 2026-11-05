import numpy as np
import pytest
from route import fit,predict,summary,qualify

def test_obvious_clusters_and_determinism():
    rng=np.random.default_rng(42);x=np.r_[rng.normal(-3,.1,(30,24)),rng.normal(3,.1,(30,24))]
    a,b=fit(x),fit(x);np.testing.assert_array_equal(a['centers'],b['centers']);assert a['training_counts'].tolist()==[30,30]
    assert predict(x,a['centers']).tolist()==[0]*30+[1]*30

def test_predict_is_rowwise_and_tie_is_zero():
    c=np.stack([-np.ones(24),np.ones(24)]);x=np.zeros((1,24));assert predict(x,c).item()==0
    np.testing.assert_array_equal(predict(x,c),predict(np.r_[x,np.ones((3,24))*100],c)[:1])

def test_known_conditional_gain():
    y=np.zeros(4);pt=np.ones(4);ps=np.array([0,0,2,2]);g=np.array([0,0,1,1]);s=summary(y,pt,ps,g)
    assert s['groups'][0]['mean_gain']==1 and s['groups'][1]['mean_gain']==-3 and s['normalized_gain_gap']==-4

def test_missing_route_cannot_qualify():
    s=summary(np.arange(4),np.ones(4),np.zeros(4),np.zeros(4,dtype=int));assert s['normalized_gain_gap'] is None
    assert not qualify({'summary':s,'inner_blocks':[]},{})

def test_predeclared_qualification_boundaries():
    r={'summary':{'normalized_gain_gap':.05},'inner_blocks':[{'summary':{'normalized_gain_gap':.01}}]*4+[{'summary':{'normalized_gain_gap':-.01}}]}
    meta={'balanced':True,'median_ARI':.5,'all_converged':True};assert qualify(r,meta)
    meta['median_ARI']=.499;assert not qualify(r,meta)
    meta['median_ARI']=.5;r['summary']['normalized_gain_gap']=.049;assert not qualify(r,meta)

def test_nonfinite_or_wrong_inputs_rejected():
    with pytest.raises(ValueError):predict(np.zeros((2,24)),np.zeros((3,24)))
    with pytest.raises(ValueError):fit(np.ones((30,24)))
    with pytest.raises(ValueError):summary(np.arange(3),np.ones(3),np.ones(3),np.array([0,1,2]))

def test_independent_verifier_nested_values_and_ari():
    import importlib.util
    from pathlib import Path
    p=Path(__file__).resolve().parent/'verify.py';s=importlib.util.spec_from_file_location('_verify23_test',p);v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
    a={'setting':'H','groups':[{'n':3,'PCC':None}],'balanced':True};v.compare(a,a)
    assert v.ari(np.array([0,0,1,1]),np.array([1,1,0,0]))==1.
    np.testing.assert_allclose(v.ari(np.array([0,0,1,1]),np.array([0,1,0,1])),-.5,atol=1e-15)
