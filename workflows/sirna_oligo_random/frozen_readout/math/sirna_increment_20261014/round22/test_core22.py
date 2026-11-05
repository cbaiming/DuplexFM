import importlib.util
from pathlib import Path
import numpy as np
import pytest
from core import ALPHAS,fit,transform,choose,deployment

def data():
    rng=np.random.default_rng(42);z=rng.normal(size=(82,216));z-=z.mean(0);z[:,24:]/=np.sqrt(12.)
    return z,.2*z[:,0]+z[:,35]+rng.normal(size=82)*.1+.5

@pytest.mark.parametrize('alpha',ALPHAS)
def test_all_alpha_convex_certificates(alpha):
    z,y=data()
    for mode in ('dense','sparse'):
        b,c,s=fit(z,y,alpha,mode);assert s['numerically_qualified'] and s['iterations']<100000
        g=(z.T@z+alpha*np.eye(216))@b-z.T@(y-c)
        tail=np.where(b[24:]!=0,g[24:]+s['lambda_value']*np.sign(b[24:]),np.sign(g[24:])*np.maximum(abs(g[24:])-s['lambda_value'],0))
        assert max(abs(np.r_[g[:24],tail]))<=s['KKT_threshold']
        if mode=='dense':np.testing.assert_allclose(b,np.linalg.solve(z.T@z+alpha*np.eye(216),z.T@(y-c)),atol=1e-11)

def test_r21_alpha100_parity():
    p=Path(__file__).resolve().parent.parent/'round21/coordinate_sparse.py';s=importlib.util.spec_from_file_location('pilot21',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    z,y=data();b,c,_=fit(z,y,100.,'sparse');bp,cp,_=m.fit_head(z,y,'sparse')
    np.testing.assert_allclose(b,bp,atol=1e-8);assert c==cp

def test_tie_rule_matches_td24():
    assert choose([.5]*5,[.1]*5)==4
    assert choose([.5,.51,.5,.5,.5],[.1]*5)==1
    assert choose([.5]*5,[.1,.09,.1,.1,.1])==1

def test_deployment_and_holdout_transform():
    z,y=data();t=dict(td_mean=np.zeros(24),td_scale=np.ones(24),src_mean=np.zeros(192),src_scale=np.ones(192))
    x=transform(z,t);b,c,_=fit(x,y,100.,'sparse');coef,offset=deployment(t,b,c)
    np.testing.assert_allclose(z@coef+offset,x@b+c,atol=1e-12)
    np.testing.assert_array_equal(transform(z[:1],t),transform(np.r_[z[:1],np.ones((1,216))*9999],t)[:1])

def test_contract_failure():
    z,y=data()
    with pytest.raises(ValueError):fit(z,y,3.,'sparse')
    with pytest.raises(ValueError):fit(z,np.ones(len(y)),100.,'sparse')
