import numpy as np
import pytest
from stability import components,statistics,paired,qualification,interval,macro

def random_seq(rng,n):return ''.join(rng.choice(list('ACGT'),n))

def test_hamming_chain():
    gs=['A'*19,'CC'+'A'*17,'CCCC'+'A'*15];r=np.random.default_rng(3)
    assert components(gs,[random_seq(r,57) for _ in gs]).tolist()==[0,0,0]

def test_target_reverse_complement():
    r=np.random.default_rng(4);t=random_seq(r,57);rc=t.translate(str.maketrans('ACGT','TGCA'))[::-1]
    assert components(['A'*19,'C'*19],[t,rc]).tolist()==[0,0]

def test_distant_separate():
    r=np.random.default_rng(5)
    assert components(['A'*19,'C'*19],[random_seq(r,57),random_seq(r,57)]).tolist()==[0,1]

def test_unknown_rejected():
    with pytest.raises(ValueError):components(['N'*19],['A'*57])

def test_statistics_repeated_rows():
    r=np.random.default_rng(6);y=r.normal(size=12);p=r.normal(size=(12,6));w=r.integers(0,5,12)
    pc,mse=statistics(y,p,w);ix=np.repeat(np.arange(12),w)
    np.testing.assert_allclose(pc,[np.corrcoef(y[ix],p[ix,j])[0,1] for j in range(6)],atol=1e-13)
    np.testing.assert_allclose(mse,np.mean((p[ix]-y[ix,None])**2,axis=0),atol=1e-13)

def test_empty_or_constant():
    pc,ms=statistics(np.ones(5),np.ones((5,6)));assert np.isnan(pc).all() and (ms==0).all()
    pc,ms=statistics(np.arange(5),np.ones((5,6)),np.zeros(5));assert np.isnan(pc).all() and np.isnan(ms).all()

def test_bad_weights():
    with pytest.raises(ValueError):statistics(np.arange(5),np.ones((5,6)),[-1,1,1,1,1])

def test_component_eligibility():
    assert qualification(np.arange(12))['eligible']
    assert not qualification(np.r_[np.zeros(100,int),np.arange(1,12)])['eligible']
    assert not qualification(np.zeros(10,int))['eligible']

def test_interval_missingness():
    a=np.arange(2000,dtype=float);a[:101]=np.nan;assert interval(a,True)['percentile95'] is None
    assert interval(np.arange(2000),False)['percentile95'] is None
    np.testing.assert_allclose(interval(np.arange(2000),True)['percentile95'],[49.975,1949.025])

def test_relative_and_macro():
    pc=np.array([.5,.6,.4,.55,.45,.58]);ms=np.ones(6);assert abs(paired(pc,ms)[0,1]-20)<1e-12
    pc[0]=0;assert np.isnan(paired(pc,ms)[0,1])
    assert macro(np.array([1,2,4,9]))==4
    with pytest.raises(ValueError):macro([1,2,3])
