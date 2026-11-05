import importlib.util
from pathlib import Path
import numpy as np
import pytest
from head import fit,profile,ALPHAS,choose,transform


def data(n=140):
    r=np.random.default_rng(42);td=r.normal(size=(n,24));src=r.normal(size=(n,8))@r.normal(size=(8,192))/np.sqrt(8)
    z=np.c_[td,src/np.sqrt(12)];z-=z.mean(0);y=.4+z@r.normal(size=216)/25+r.normal(size=n)*.1
    return z,y


def certificate(z,y,b,c,s,alpha):
    g=z.T@z+alpha*np.eye(216);rhs=z.T@(y-c);v=np.r_[np.zeros(24),s['dual']]
    singular=np.linalg.svd(b[24:].reshape(6,32),compute_uv=False)
    primal=.5*b@g@b-rhs@b+s['lambda_value']*singular.sum()
    dual=-.5*(rhs-v)@np.linalg.solve(g,rhs-v)
    np.testing.assert_allclose((g@b-rhs)[:24],np.zeros(24),atol=1e-8)
    assert np.linalg.norm(v[24:].reshape(6,32),2)<=s['lambda_value']+1e-9
    assert -1e-8<=primal-dual<=1e-9*s['gap_scale']+1e-10 and s['certified']


@pytest.mark.parametrize('alpha',ALPHAS)
def test_five_alpha_global_certificates(alpha):
    z,y=data();b,c,s=fit(z,y,alpha);certificate(z,y,b,c,s,alpha)


def test_alpha100_matches_pilot_objective():
    p=Path(__file__).resolve().parents[1]/'round25/native_trace.py'
    sp=importlib.util.spec_from_file_location('_r26_pilot_test',p);old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
    z,y=data();b,c,s=fit(z,y,100.);bb,cc,ss=old.fit(z,y)
    bound=np.sqrt(2*s['gap']/100)+np.sqrt(2*ss['gap']/100)+1e-9
    assert np.linalg.norm(b-bb)<=bound and c==cc
    h,q,*_=profile(z,y,100.);hh,qq,*_=old.prior.profile(z,y)
    np.testing.assert_allclose(h,hh,atol=1e-12);np.testing.assert_allclose(q,qq,atol=1e-12)


def test_zero_source_exact_td():
    z,y=data();z[:,24:]=0;b,c,s=fit(z,y,.1)
    ref=np.linalg.solve(z[:,:24].T@z[:,:24]+.1*np.eye(24),z[:,:24].T@(y-y.mean()))
    np.testing.assert_allclose(b[:24],ref,atol=1e-12);assert np.count_nonzero(b[24:])==0 and s['rank']==0 and s['certified']


def test_low_rank_ill_conditioned_weak_alpha():
    r=np.random.default_rng(44);n=500;td=r.normal(size=(n,24));src=r.normal(size=(n,2))@r.normal(size=(2,192))
    z=np.c_[td,src];z-=z.mean(0);z[:,24:]/=z[:,24:].std(0)*np.sqrt(12.)
    y=z[:,0]+z[:,26]*.7+r.normal(size=n)*.1;b,c,s=fit(z,y,.1);certificate(z,y,b,c,s,.1)


def test_fixed_choice_ties():
    assert choose(np.ones(5),np.ones(5))==4
    assert choose(np.array([0,1,1,0,0]),np.array([1,.2,.3,1,1]))==1


def test_reject_invalid_and_readonly_transform():
    z,y=data()
    with pytest.raises(ValueError):fit(z,y,2.)
    with pytest.raises(ValueError):fit(z+1,y,100.)
    r=np.random.default_rng(1);raw=r.normal(size=(30,216));m=raw.mean(0);sd=raw.std(0)
    t=dict(td_mean=m[:24],td_scale=sd[:24],src_mean=m[24:],src_scale=sd[24:]);a=transform(raw,t)
    transform(raw+100,t);np.testing.assert_array_equal(a,transform(raw,t))
