import importlib.util
from pathlib import Path
import numpy as np
import pytest
s=importlib.util.spec_from_file_location('_energy_test',Path(__file__).with_name('energy.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def fixture():
    rng=np.random.default_rng(37);raw=rng.normal(size=(72,216));y=rng.normal(size=72)
    st=dict(td_mean=raw[:,:24].mean(0),td_scale=raw[:,:24].std(0),src_mean=raw[:,24:].mean(0),src_scale=raw[:,24:].std(0))
    e=np.c_[rng.normal(size=(72,3)),(rng.random(72)>.2).astype(float)]
    return raw,y,st,e


def test_valid_only_scaling_and_missing_values():
    raw,y,st,e=fixture();t=m.fit_energy(e);z=m.map_energy(e,t)
    np.testing.assert_allclose(z.mean(0),0,atol=1e-15);np.testing.assert_allclose(z[e[:,3]==1,:3].std(0),1)
    assert np.all(z[e[:,3]==0,:3]==0)
    altered=e.copy();altered[e[:,3]==0,:3]=1e8
    np.testing.assert_array_equal(m.map_energy(altered,t),z)


def test_no_training_validity_disables_entire_block():
    raw,y,st,e=fixture();e[:,3]=0;t=m.fit_energy(e);future=e.copy();future[:,3]=1
    assert not t['enabled'];assert not m.map_energy(future,t).any()
    z=m.transform(raw,e,st,t);b,c,_=m.fit(z,y);oldb,oldc,_=m.prior.fit(m.prior.transform(raw,st),y)
    np.testing.assert_allclose(b[:24],oldb[:24],atol=1e-13);np.testing.assert_allclose(b[28:],oldb[24:],atol=1e-13)
    np.testing.assert_allclose(c,oldc);assert not b[24:28].any()


def test_constant_energy_and_constant_flag():
    raw,y,st,e=fixture();e[:,:3]=[2,3,4];e[:,3]=1;t=m.fit_energy(e)
    assert not t['flag_enabled'];assert not m.map_energy(e,t).any()


def test_validation_cannot_change_training_scale():
    raw,y,st,e=fixture();t=m.fit_energy(e[:45]);a=m.map_energy(e[45:],t)
    revised=e[45:].copy();revised[:,:3]*=1000;m.map_energy(revised,t)
    np.testing.assert_array_equal(a,m.map_energy(e[45:],t));np.testing.assert_array_equal(t['mean'],e[:45][e[:45,3]==1,:3].mean(0))


def test_zero_source_equals_TD28():
    raw,y,st,e=fixture();z=m.transform(raw,e,st,m.fit_energy(e));z[:,28:]=0
    b,c,_=m.fit(z,y);tb,tc,_=m.fit(z[:,:28],y)
    np.testing.assert_allclose(b[:28],tb,atol=1e-14);assert not b[28:].any();assert c==tc


def test_zero_nuclear_penalty_independent_augmented_lstsq():
    raw,y,st,e=fixture();z=m.transform(raw,e,st,m.fit_energy(e));b,c,cert=m.fit(z,y,ratio=0)
    expected=np.linalg.lstsq(np.r_[z,10*np.eye(220)],np.r_[y-y.mean(),np.zeros(220)],rcond=None)[0]
    np.testing.assert_allclose(b,expected,atol=1e-13);assert cert['certified']


def test_positive_energy_units_invariant():
    raw,y,st,e=fixture();ee=e.copy();ee[:,:3]=ee[:,:3]*[2,3,4]+[1,-5,7]
    np.testing.assert_allclose(m.map_energy(e,m.fit_energy(e)),m.map_energy(ee,m.fit_energy(ee)),atol=1e-14)


def test_single_affine_deployment_including_missing_and_disabled():
    raw,y,st,e=fixture()
    for state in (m.fit_energy(e),m.fit_energy(np.c_[e[:,:3],np.zeros(len(e))])):
        z=m.transform(raw,e,st,state)
        for size in (28,220):
            b,c,_=m.fit(z[:,:size],y);coef,offset=m.deployment(st,state,b,c)
            np.testing.assert_allclose(m.canonical(raw,e)[:,:size]@coef+offset,z[:,:size]@b+c,atol=1e-13)


def test_invalid_contracts():
    with pytest.raises(ValueError):m.fit_energy(np.ones((3,3)))
    with pytest.raises(ValueError):m.fit_energy(np.zeros((0,4)))
    with pytest.raises(ValueError):m.fit_energy(np.full((2,4),.5))
    with pytest.raises(ValueError):m.fit_energy(np.full((2,4),np.nan))
    with pytest.raises(ValueError):m.fit(np.zeros((4,28)),np.ones(4))
