import numpy as np
import pytest
from tails import inspect,qualify


def test_exact_single_extreme():
    x=np.zeros((100,192));x[-1]=100
    r=inspect(x)
    np.testing.assert_allclose(r['top_fraction_squared_deviation'],.99,atol=1e-14)
    np.testing.assert_allclose(r['max_abs_z'],np.sqrt(99),atol=1e-14)
    assert r['top_rows']==1 and r['flagged_columns']==192 and r['required_columns']==20 and r['pool_qualified']


def test_constant_columns_and_floor():
    x=np.zeros((100,192));x[-1,:20]=100;x[-1,20]=1e-10
    r=inspect(x)
    assert r['eligible_columns']==20 and r['flagged_columns']==20
    assert not r['eligible'][20] and not np.any(r['eligible'][21:])
    assert np.all(np.isfinite(r['raw_kurtosis']))


def test_affine_and_row_permutation():
    rng=np.random.default_rng(42);x=rng.normal(size=(320,192));x[-1,:30]+=20
    a=inspect(x);b=inspect(3*x[rng.permutation(len(x))]+5)
    for k in ('top_fraction_squared_deviation','max_abs_z','raw_kurtosis'):
        np.testing.assert_allclose(a[k],b[k],atol=1e-12,rtol=1e-12)
    np.testing.assert_array_equal(a['flagged'],b['flagged'])


def test_normal_reference_not_heavy_by_rule():
    x=np.random.default_rng(42).normal(size=(10000,24));r=inspect(x)
    assert r['flagged_columns']==0 and not r['pool_qualified']
    assert .07<r['median_tail_share']<.10


def test_ceil_top_rows_and_all_constant():
    r=inspect(np.ones((101,24)))
    assert r['top_rows']==2 and r['eligible_columns']==0 and not r['pool_qualified']
    assert r['median_tail_share']==r['maximum_abs_z']==0


def test_gate_requires_two_primary_pools():
    records=[dict(setting=s,source='core_real',statistics=dict(pool_qualified=v)) for s,v in zip('abc',[True,False,False])]
    records.append(dict(setting='a',source='full_real',statistics=dict(pool_qualified=True)))
    assert not qualify(records)
    records[1]['statistics']['pool_qualified']=True
    assert qualify(records)
    with pytest.raises(ValueError):qualify(records[:2])


def test_contract():
    for x in (np.zeros((99,24)),np.zeros((100,40)),np.full((100,192),np.nan)):
        with pytest.raises(ValueError):inspect(x)
