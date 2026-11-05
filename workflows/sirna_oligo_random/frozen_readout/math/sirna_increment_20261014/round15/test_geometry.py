import numpy as np
import pytest
from geometry import direction, source_norm, statistics, qualification


def test_positive_scale_invariance_and_unchanged_td():
    rng=np.random.default_rng(42); x=rng.normal(size=(30,216)); before=x.copy()
    other=x.copy(); other[:,24:]*=np.linspace(.01,100,30)[:,None]
    np.testing.assert_allclose(direction(x),direction(other),rtol=1e-13,atol=1e-13)
    np.testing.assert_array_equal(direction(x)[:,:24],x[:,:24])
    np.testing.assert_array_equal(x,before)
    np.testing.assert_allclose(np.linalg.norm(direction(x)[:,24:],axis=1),1.,atol=1e-14)


def test_zero_norm_and_tiny_rows_are_zero():
    x=np.ones((3,216)); x[0,24:]=0; x[1,24:]=1e-20
    z=direction(x)
    assert np.count_nonzero(z[:2,24:])==0 and np.isfinite(z).all()
    np.testing.assert_array_equal(z[:,:24],x[:,:24])


def test_large_norm_stability():
    x=np.ones((2,216)); x[:,24:]*=1e200
    np.testing.assert_allclose(source_norm(x),np.sqrt(192)*1e200)
    np.testing.assert_allclose(direction(x)[:,24:],1/np.sqrt(192))


def test_no_cross_sample_dependence():
    x=np.random.default_rng(4).normal(size=(40,216)); one=direction(x[:3]); x[3:]*=17
    np.testing.assert_array_equal(one,direction(x)[:3])


def test_invalid_and_constant_statistics():
    with pytest.raises(ValueError): direction(np.ones((3,215)))
    x=np.ones((3,216)); x[0,2]=np.nan
    with pytest.raises(ValueError): direction(x)
    assert not statistics(np.ones(5),np.arange(5))['pool_qualified']


def test_fixed_qualification_counts_core_only():
    records=[dict(setting=s,source='core_real',pool_qualified=i<2,all_nonzero=True)
             for i,s in enumerate(('Huesken','Mixset','Takayuki'))]
    assert qualification(records)
    records[1]['pool_qualified']=False
    records.append(dict(setting='Huesken',source='full_real',pool_qualified=True,all_nonzero=True))
    assert not qualification(records)
    records[1]['pool_qualified']=True; records[2]['all_nonzero']=False
    assert not qualification(records)
