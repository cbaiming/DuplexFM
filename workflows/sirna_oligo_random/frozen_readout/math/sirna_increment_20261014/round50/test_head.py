import numpy as np
import pytest
import head


def fixture():
    rng=np.random.default_rng(42);z=rng.normal(size=(80,216));z-=z.mean(0);y=rng.normal(size=80);return z,y,np.arange(80)%8


def test_original_alpha_family_TD():
    from verify import audit_TD,group
    z,y,g=fixture()
    for alpha in head.ALPHAS:
        b,c,cert=head.fit(z,y,g,alpha,False);zz=group.weighted_design(z[:,:24],g);yy=group.weighted_design(y-y.mean(),g)+y.mean()
        assert audit_TD(zz,yy,b,c,cert,alpha)[0]<1e-8


def test_original_alpha_family_source():
    from verify import prev,group
    z,y,g=fixture()
    for alpha in head.ALPHAS:
        b,c,cert=head.fit(z,y,g,alpha,True);zz=group.weighted_design(z,g);yy=group.weighted_design(y-y.mean(),g)+y.mean()
        cc=dict(cert,matrix_shape=list(cert['matrix_shape']))
        assert prev.check_head(zz,yy,b,c,cc,alpha,zz.T@zz,zz.T@(yy-yy.mean()))[1]<1.1e-9


def test_100_same_objective_as_pilot():
    z,y,g=fixture();b,c,cert=head.fit(z,y,g,100.,True);bb,cc,old=head.prior.fit(z,y,g,True)
    np.testing.assert_allclose(z@b+c,z@bb+cc,rtol=1e-4,atol=1e-5);assert cert['lambda_ratio']==old['lambda_ratio']==.1


def test_identity_projection_matches_original():
    z,y,g=fixture();g[:]=0;b,c,cert=head.fit(z,y,g,100.,True);bb,cc,_=head.native.fit(z,y,100.)
    np.testing.assert_allclose(b,bb,atol=1e-12);assert c==pytest.approx(cc)


def test_fixed_selection_ties():
    assert head.choose([.5]*5,[.2]*5)==4
    assert head.choose([.5,.6,.5,.5,.5],[.1]*5)==1
    assert head.choose([.5]*5,[.2,.1,.2,.2,.2])==1


def test_reuse_dispatch():
    assert head.reuse_kind('Huesken',0,1,100.,True,True)=='R49'
    assert head.reuse_kind('Huesken',1,1,10.,True,True)=='R26_identity'
    assert head.reuse_kind('Mixset',1,2,10.,True,False)=='TD_identity'
    assert head.reuse_kind('Huesken',0,None,100.,True,True,100.)=='R26_identity'
    assert head.reuse_kind('Huesken',0,None,100.,True,True,10.)=='new'
    assert head.reuse_kind('Huesken',0,None,100.,False,True,100.)=='new'


def test_reject_extra_alpha():
    z,y,g=fixture()
    with pytest.raises(ValueError):head.fit(z,y,g,50.,True)


def test_four_setting_relative_gain():
    b=np.array([.4,.5,.6,.7]);p=b*np.array([1.1,1.04,.98,1.08])
    assert head.relative_gain(p,b)==pytest.approx(5.)
    assert head.relative_gain(b,b)==0.
    with pytest.raises(ValueError):head.relative_gain(p[:3],b[:3])
