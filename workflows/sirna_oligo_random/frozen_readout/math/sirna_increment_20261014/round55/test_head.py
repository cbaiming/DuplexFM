import numpy as np
import pytest
import head
import audit_math as audit


def fixture():
    rng=np.random.default_rng(55);g=np.repeat(np.arange(6),[1,2,4,8,15,50])
    z=rng.normal(size=(80,216))+.3*rng.normal(size=(6,216))[g];z-=z.mean(0)
    y=.03*z[:,1]+rng.normal(size=80)*.3+np.linspace(-1,2,6)[g]
    return z,y,g


def test_original_alpha_family_TD_explicit_nuisance_lstsq():
    z,y,g=fixture()
    for alpha in head.ALPHAS:
        b,c,cert=head.fit(z,y,g,alpha,False)
        assert audit.audit_TD(z[:,:24],y,g,b,c,cert,alpha)[1]<1e-8


def test_original_alpha_family_source_full_joint_certificate():
    z,y,g=fixture()
    for alpha in head.ALPHAS:
        b,c,cert=head.fit(z,y,g,alpha,True)
        assert audit.check_source(z,y,g,b,c,cert,alpha)[1]<1.1e-9


def test_alpha100_matches_original_pilot_objective():
    z,y,g=fixture();b,c,cert=head.fit(z,y,g,100.,True);bb,cc,old=head.prior.fit(z,y,g,True)
    np.testing.assert_allclose(z@b+c,z@bb+cc,rtol=1e-4,atol=1e-5)
    assert cert['lambda_ratio']==old['lambda_ratio']==.1


def test_equal_groups_same_as_R50_and_deployment():
    prev=head.load('_r55_test50',head.R.parent/'round50/head.py');z,y,_=fixture();g=np.arange(80)%8
    for source in (False,True):
        b,c,cert=head.fit(z,y,g,100.,source);bb,cc,old=prev.fit(z,y,g,100.,source)
        np.testing.assert_allclose(b,bb,atol=1e-11);assert c==pytest.approx(cc)
        if source:assert audit.check_source(z,y,g,bb,cc,old,100.)[1]<1.1e-9
        else:assert audit.audit_TD(z[:,:24],y,g,bb,cc,old,100.)[1]<1e-8
    rng=np.random.default_rng(7);state=dict(td_mean=rng.normal(size=24),td_scale=np.exp(rng.normal(size=24)),src_mean=rng.normal(size=192),src_scale=np.exp(rng.normal(size=192)))
    raw=rng.normal(size=(17,216));coef,offset=head.deployment(state,b,c)
    np.testing.assert_allclose(raw@coef+offset,head.transform(raw,state)@b+c,atol=1e-11)


def test_weighted_intercept_and_group_equivariance():
    z,y,g=fixture();b,c,_=head.fit(z,y,g,10.,False);ix=np.random.default_rng(9).permutation(len(y))
    bb,cc,_=head.fit(z[ix],y[ix],5-g[ix],10.,False)
    np.testing.assert_allclose(b,bb,atol=1e-11);assert c==pytest.approx(cc)
    zz=np.zeros_like(z);yy=(g==5).astype(float);b,c,_=head.fit(zz,yy,g,10.,False)
    w=np.array(head.geometry(g)['group_gamma'])[g]
    assert c==pytest.approx(np.average(yy,weights=w)) and abs(c-yy.mean())>.1


def test_selection_and_original_alpha_boundary():
    assert head.choose([.5]*5,[.2]*5)==4
    assert head.choose([.5,.6,.5,.5,.5],[.1]*5)==1
    assert head.choose([.5]*5,[.2,.1,.2,.2,.2])==1
    z,y,g=fixture()
    with pytest.raises(ValueError):head.fit(z,y,g,50.,True)


def test_exact_reuse_dispatch():
    assert head.reuse_kind('Huesken',0,1,100.,True)=='R54'
    assert head.reuse_kind('Huesken',1,1,10.,True)=='R50_equal'
    assert head.reuse_kind('Mixset',1,2,10.,False)=='new'
    assert head.reuse_kind('Huesken',0,None,100.,True,100.)=='R50_equal'
    assert head.reuse_kind('Huesken',0,None,100.,True,10.)=='new'
    assert head.reuse_kind('Huesken',0,None,100.,False,100.)=='new'


def test_four_setting_relative_gain():
    b=np.array([.4,.5,.6,.7]);p=b*np.array([1.1,1.04,.98,1.08])
    assert head.relative_gain(p,b)==pytest.approx(5.)
    assert head.relative_gain(b,b)==0.
    with pytest.raises(ValueError):head.relative_gain(p[:3],b[:3])
