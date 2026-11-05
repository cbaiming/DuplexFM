import numpy as np
import pytest
from random_intercept import geometry,design,fit,residual_stats,deployment,prior
from verify import indicator_geometry,profile,audit_TD,certify_source


def fixture():
    rng=np.random.default_rng(54);g=np.repeat(np.arange(6),[1,2,4,8,15,50]);n=len(g)
    z=rng.normal(size=(n,216))+.7*rng.normal(size=(6,216))[g];z-=z.mean(0)
    y=.03*z[:,1]+rng.normal(size=n)*.3+np.linspace(-1,2,6)[g]
    return z,y,g


def test_geometry_profiled_trace_and_positive_semidefiniteness():
    z,y,g=fixture();m=geometry(g);G,inv,w,ind=indicator_geometry(g)
    for key in ('common_group_penalty','normalizer','centered_metric_trace','gamma_min','gamma_max'):assert m[key]==pytest.approx(ind[key])
    M=np.eye(len(g))-G@np.diag(inv)@G.T;C=M-np.outer(w,w)/w.sum()
    assert np.linalg.eigvalsh(C).min()>-1e-12
    assert np.trace(C*m['normalizer'])==pytest.approx(len(g)-1)
    np.testing.assert_allclose(C@np.ones(len(g)),0,atol=1e-12)


def test_transformed_gram_rhs_and_weighted_centers():
    z,y,g=fixture();x,t,zm,ym,m=design(z,y,g);gram,rhs,quad,zz,yy,G,w,mm=profile(z,y,g)
    np.testing.assert_allclose(x.T@x,gram,atol=1e-10);np.testing.assert_allclose(x.T@t,rhs,atol=1e-10)
    assert t@t==pytest.approx(quad);np.testing.assert_allclose(zm,zz,atol=1e-12);assert ym==pytest.approx(yy)
    assert np.max(abs(x.mean(0)))>1e-3  # Cannot silently pass this through a second ordinary-centering step.


def test_TD_explicit_augmented_nuisance_solution():
    z,y,g=fixture();b,c,cert=fit(z,y,g,False)
    assert audit_TD(z[:,:24],y,g,b,c,cert)[0]


def test_intercept_is_not_unweighted_response_mean():
    z,y,g=fixture();z[:]=0;y=(g==5).astype(float)
    b,c,cert=fit(z,y,g,False);m=geometry(g);w=np.array(m['group_gamma'])[g]
    assert c==pytest.approx(np.average(y,weights=w)) and abs(c-y.mean())>.1
    assert cert['certified'];np.testing.assert_array_equal(b,np.zeros(24))


def test_equal_size_groups_reduce_to_R49():
    z,y,_=fixture()
    for g in (np.zeros(len(y),int),np.arange(len(y)),np.arange(len(y))%8):
        assert geometry(g)['reference_equal']
        b,c,cert=fit(z,y,g,False);bb,cc,_=prior.fit(z,y,g,False)
        np.testing.assert_allclose(b,bb,atol=1e-11);assert c==pytest.approx(cc)


def test_group_name_and_training_order_equivariance():
    z,y,g=fixture();b,c,_=fit(z,y,g,False);ix=np.random.default_rng(9).permutation(len(y))
    bb,cc,_=fit(z[ix],y[ix],5-g[ix],False)
    np.testing.assert_allclose(b,bb,atol=1e-11);assert c==pytest.approx(cc)


def test_source_full_joint_certificate():
    z,y,g=fixture();b,c,cert=fit(z,y,g,True)
    assert certify_source(z,y,g,b,c,cert)[0]


def test_deployed_model_needs_no_group_at_prediction():
    z,y,g=fixture();b,c,cert=fit(z,y,g,True);rng=np.random.default_rng(7)
    state=dict(td_mean=rng.normal(size=24),td_scale=np.exp(rng.normal(size=24)),src_mean=rng.normal(size=192),src_scale=np.exp(rng.normal(size=192)))
    raw=rng.normal(size=(17,216));zz=prior.transform(raw,state);coef,offset=deployment(state,b,c)
    np.testing.assert_allclose(raw@coef+offset,zz@b+c,atol=1e-11)
    np.testing.assert_allclose(np.array([row@coef+offset for row in raw]),raw@coef+offset,atol=1e-12)


def test_profile_equals_explicit_penalized_group_residual():
    z,y,g=fixture();rng=np.random.default_rng(2);beta=rng.normal(size=216);c=.5;m=geometry(g)
    stats=residual_stats(z,y,beta,c,g,True);G,inv,w,_=indicator_geometry(g);r=y-z@beta-c;u=inv*(G.T@r)
    expected=m['normalizer']*((r-G@u)@(r-G@u)+m['common_group_penalty']*(u@u))
    assert stats['normalized_profiled_SSE']==pytest.approx(expected)
    assert stats['normalized_augmented_SSE']==pytest.approx(expected) and not stats['group_effects_deployed']


def test_invalid_training_contracts_fail():
    z,y,g=fixture()
    with pytest.raises(ValueError):geometry(g.astype(float))
    with pytest.raises(ValueError):geometry(g+1)
    with pytest.raises(ValueError):geometry(np.array([0]))
    with pytest.raises(ValueError):fit(z,np.ones(len(y)),g,True)
    with pytest.raises(ValueError):fit(z[:,:24],y,g,False)
    with pytest.raises(ValueError):design(z,np.full(len(y),np.nan),g)
