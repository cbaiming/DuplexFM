import numpy as np
import pytest
from metric_scale import fit_scaling, apply_scaling, raw_deployment, fit, prior, transform
from verify import independent_scaling, compare, ref


def fixture():
    rng=np.random.default_rng(53); n=80; g=np.arange(n)%9
    z=rng.normal(size=(n,216))+rng.normal(size=(9,216))[g]*np.linspace(0,2,216)
    return z-z.mean(0),g,rng.normal(size=n)


def test_quadratic_form_and_block_energy():
    z,g,_=fixture(); f,meta=fit_scaling(z,g); ff,mm=independent_scaling(z,g)
    np.testing.assert_allclose(f,ff,atol=1e-12); compare(meta,mm)
    n=len(z); q=np.eye(9)[g]; p=q@np.diag(1/np.bincount(g))@q.T
    metric=79/(79-.5*8)*(np.eye(n)-.5*(p-np.ones((n,n))/n))
    for a,b in ((0,24),(24,216)):
        original=np.einsum('ij,ij->j',z[:,a:b],metric@z[:,a:b])
        adjusted=original*f[a:b]**2
        assert adjusted.sum()==pytest.approx(original.sum(),rel=1e-12)
        ratio=adjusted/np.sum(z[:,a:b]**2,axis=0)
        np.testing.assert_allclose(ratio,ratio[0],atol=1e-12)


def test_single_group_and_singletons_exact_identity():
    z,_,_=fixture()
    for g in (np.zeros(len(z),int),np.arange(len(z))):
        f,meta=fit_scaling(z,g); np.testing.assert_array_equal(f,np.ones(216)); assert meta['identity']
        np.testing.assert_array_equal(apply_scaling(z,f),z)


def test_zero_columns_and_zero_block_are_explicit():
    z,g,_=fixture(); z[:,:24]=0; z[:,30]=0; f,meta=fit_scaling(z,g)
    np.testing.assert_array_equal(f[:24],np.ones(24)); assert f[30]==1
    assert meta['blocks'][0]['block_metric_ratio'] is None
    assert meta['blocks'][0]['active_columns']==0 and meta['blocks'][1]['active_columns']==191
    ff,_=fit_scaling(np.zeros_like(z),g); np.testing.assert_array_equal(ff,np.ones(216))


def test_row_order_labels_and_block_unit_equivariance():
    z,g,_=fixture(); f,_=fit_scaling(z,g); ix=np.random.default_rng(2).permutation(len(z))
    ff,_=fit_scaling(z[ix],8-g[ix]); np.testing.assert_allclose(f,ff,atol=1e-12)
    zz=z.copy(); zz[:,:24]*=7; zz[:,24:]*=.2; ff,_=fit_scaling(zz,g)
    np.testing.assert_allclose(f,ff,atol=1e-12)


def test_TD_independence_and_fixed_validation_transform():
    z,g,_=fixture(); f,_=fit_scaling(z,g); zz=z.copy(); zz[:,24:]=0
    ff,_=fit_scaling(zz,g); np.testing.assert_array_equal(f[:24],ff[:24])
    valid=np.random.default_rng(3).normal(size=(12,216))*100+500
    expected=valid*f
    np.testing.assert_array_equal(apply_scaling(valid,f),expected)
    np.testing.assert_array_equal(np.r_[apply_scaling(valid[:3],f),apply_scaling(valid[3:],f)],expected)


def test_invalid_shapes_nonfinite_and_noncentered_fail():
    z,g,_=fixture()
    for bad in (z[:,:24],z+1,np.full_like(z,np.nan)):
        with pytest.raises(ValueError): fit_scaling(bad,g)
    with pytest.raises(ValueError): fit_scaling(z,np.ones(len(z),int))
    with pytest.raises(ValueError): apply_scaling(z,np.zeros(216))
    with pytest.raises(ValueError): apply_scaling(z,np.ones(24))


def test_TD_fit_independent_augmented_lstsq():
    z,g,y=fixture(); f,_=fit_scaling(z,g); x=z*f
    b,c,cert=fit(x,y,g,False); tx=ref.weighted_design(x[:,:24],g); ty=ref.weighted_design(y-y.mean(),g)
    independent=np.linalg.lstsq(np.r_[tx,10*np.eye(24)],np.r_[ty,np.zeros(24)],rcond=None)[0]
    np.testing.assert_allclose(b,independent,atol=1e-12); assert c==pytest.approx(y.mean())
    assert ref.audit_TD(tx,ty,b,cert)[0]


def test_native_joint_convex_certificate():
    z,g,y=fixture(); f,_=fit_scaling(z,g); x=z*f; b,c,cert=fit(x,y,g,True)
    cert=dict(cert,matrix_shape=list(cert['matrix_shape']))
    assert ref.certify(ref.weighted_design(x,g),ref.weighted_design(y-y.mean(),g)+y.mean(),b,c,cert,False)[0]


def test_raw_deployment_requires_no_group_or_scaling_at_prediction():
    z,g,_=fixture(); f,_=fit_scaling(z,g); rng=np.random.default_rng(5)
    state=dict(td_mean=rng.normal(size=24),td_scale=np.exp(rng.normal(size=24)),
        src_mean=rng.normal(size=192),src_scale=np.exp(rng.normal(size=192)))
    valid=rng.normal(size=(11,216)); x=transform(valid,state)*f
    for source in (False,True):
        beta=rng.normal(size=216 if source else 24); center=.4
        coef,offset=raw_deployment(state,f,beta,center,source)
        pp=(x if source else x[:,:24])@beta+center
        np.testing.assert_allclose(valid@coef+offset,pp,atol=1e-11)
        if not source: np.testing.assert_array_equal(coef[24:],np.zeros(192))


def test_this_changes_the_penalty_prior_not_just_a_solver_preconditioner():
    z,g,y=fixture(); f,_=fit_scaling(z,g); beta=np.random.default_rng(7).normal(size=216); raw_beta=f*beta
    assert np.max(abs(f-1))>.01
    np.testing.assert_allclose(prior.project(z*f,g)@beta,prior.project(z,g)@raw_beta,atol=1e-11)
    original_penalty=np.dot(raw_beta,raw_beta)
    new_penalty=np.dot(raw_beta/f,raw_beta/f)
    assert new_penalty==pytest.approx(np.dot(beta,beta)) and abs(new_penalty-original_penalty)>1e-3
