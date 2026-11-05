import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
sys.path.insert(0,str(Path(__file__).resolve().parent))
from core import configs,columns,fit_transform,transform,fit_heads,score,choose

def fixture():
    rng=np.random.default_rng(512);x=rng.normal(size=(90,184));x[:,3]=1
    y=.4+.3*x[:,30]-.1*x[:,1]+rng.normal(size=90)*.03
    return x,y

def test_pca_only_training_and_exact_projection():
    x,_=fixture();z,t=fit_transform(x[:70]);held=x[70:].copy();held[:]=1e9
    _,u=fit_transform(x[:70])
    for k in t:np.testing.assert_array_equal(t[k],u[k])
    assert transform(held,t).shape==(20,44)
    np.testing.assert_allclose(z,transform(x[:70],t),atol=1e-12)
    for j in range(5):
        np.testing.assert_allclose(t['components'][j]@t['components'][j].T,np.eye(4),atol=1e-12)
        np.testing.assert_allclose(z[:,24+4*j:28+4*j].std(0),1,atol=1e-12)

def test_all_routes_normal_equations_and_reference():
    x,y=fixture();z,t=fit_transform(x);cfg=configs();b,c=fit_heads(z,y,cfg)
    assert b.shape==(40,44)
    for k,conf in enumerate(cfg):
        keep=columns(conf);off=np.setdiff1d(np.arange(44),keep)
        assert np.count_nonzero(b[k,off])==0
        np.testing.assert_allclose((z[:,keep].T@z[:,keep]+conf['alpha']*np.eye(len(keep)))@b[k,keep],z[:,keep].T@(y-c),atol=1e-9)
        ref=Ridge(alpha=conf['alpha'],fit_intercept=True).fit(z[:,keep],y)
        np.testing.assert_allclose(z@b[k]+c,ref.predict(z[:,keep]),atol=1e-10)

def test_masks_and_source_only_are_distinct():
    cfg=configs();assert len(cfg)==40
    assert np.array_equal(columns(cfg[30]),np.arange(24))
    assert np.array_equal(columns(cfg[35]),np.arange(24,44))
    assert len(columns(cfg[5]))==40 and not set(range(24,28))&set(columns(cfg[5]))

def test_select_uses_training_oof_only_and_stronger_tie():
    cfg=configs();p=np.ones(40);m=np.ones(40)
    assert choose(p,m,cfg,range(35))==34
    assert choose(p,m,cfg,range(5))==4
    y=np.arange(10.);pred=np.c_[y,np.ones(10),-y]
    r,_=score(y,pred);np.testing.assert_allclose(r,[1,-2,-1])
