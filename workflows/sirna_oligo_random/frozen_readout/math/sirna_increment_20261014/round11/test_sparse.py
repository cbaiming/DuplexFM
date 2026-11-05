import importlib.util
from pathlib import Path
import numpy as np
import pytest
from sparse import fit,predict,choose,collapse,PARAMETERS

def data(p=40):
    rng=np.random.default_rng(42);z=rng.normal(size=(130,p));z-=z.mean(0)
    y=.4+.1*z[:,0]-.08*z[:,min(26,p-1)]+rng.normal(0,.03,len(z))
    return z,y

def test_native_posterior():
    z,y=data();m=fit(z,y);a=np.asarray(m['active_indices']);x=z-z.mean(0);noise=m['noise_precision']
    precision=noise*x[:,a].T@x[:,a]+np.diag(np.asarray(m['weight_precision'])[a])
    np.testing.assert_allclose(precision@np.asarray(m['posterior_covariance']),np.eye(len(a)),atol=1e-9)
    np.testing.assert_allclose(precision@np.asarray(m['coef'])[a],noise*x[:,a].T@(y-y.mean()),atol=1e-8)
    assert m['active_source']<16 and m['active_TD']<24

def test_td_and_constant_column():
    z,y=data(24);z[:,3]=1.;m=fit(z,y)
    assert m['input_width']==24 and m['active_source']==0 and abs(m['coef'][3])<1e-10
    assert np.isfinite(predict(z,m)).all()

def test_deterministic():
    z,y=data();a=fit(z,y);b=fit(z.copy(),y.copy())
    for key in ('coef','weight_precision','posterior_covariance'):np.testing.assert_array_equal(a[key],b[key])
    assert a['intercept']==b['intercept'] and a['noise_precision']==b['noise_precision']

def test_choose_and_locked_fallback():
    p=np.array([.1,.2,.3,.4,.5,.45]);e=np.ones(6)
    assert choose(p,e)==choose(p,e,range(5))==4
    p[5]=.6;assert choose(p,e)==5
    assert choose(np.ones(6),e)==4

def test_raw_collapse():
    path=Path(__file__).resolve().parent.parent/'round4/core.py'
    spec=importlib.util.spec_from_file_location('_r11_test_old',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    rng=np.random.default_rng(45);train=rng.normal(size=(100,216));unseen=rng.normal(size=(7,216))
    z,t=old.fit_transform(train);y=.5+.08*z[:,0]+.03*z[:,25]+rng.normal(0,.02,len(z))
    for width in (24,40):
        m=fit(z[:,:width],y);beta,b=collapse(t,m)
        np.testing.assert_allclose(unseen@beta+b,predict(old.transform(unseen,t)[:,:width],m),atol=1e-12)

def test_bad_response_fails():
    z,y=data()
    with pytest.raises(ValueError):fit(z,np.ones(len(z)))
    y[0]=np.nan
    with pytest.raises(ValueError):fit(z,y)

def test_training_arrays_preserved():
    z,y=data();original_z=z.copy();original_y=y.copy()
    fit(z,y)
    np.testing.assert_array_equal(z,original_z)
    np.testing.assert_array_equal(y,original_y)
