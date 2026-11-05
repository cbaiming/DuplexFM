import numpy as np
import pytest
from curve import subset,fit_transform,fit,head,perturbation_variance


def test_subset_nested_exact_size():
    ids=[f'id{i}' for i in range(101)]
    for seed in (42,43,44):
        small,large=subset(ids,.5,seed),subset(ids,.75,seed)
        assert len(small)==51 and len(large)==76 and set(small)<=set(large)


def test_subset_identity_order_invariance_and_draw_difference():
    ids=[f'id{i}' for i in range(101)];rev=ids[::-1]
    assert {ids[i] for i in subset(ids,.5,42)}=={rev[i] for i in subset(rev,.5,42)}
    assert subset(ids,.5,42)!=subset(ids,.5,43)


def test_subset_contracts():
    with pytest.raises(ValueError):subset(['x']*60,.5,42)
    with pytest.raises(ValueError):subset([str(i) for i in range(60)],.25,42)
    with pytest.raises(ValueError):subset([str(i) for i in range(60)],.5,45)


def test_training_scaler_only_and_affine_deployment():
    rng=np.random.default_rng(3);raw=rng.normal(size=(60,216));z,t=fit_transform(raw)
    np.testing.assert_allclose(z.mean(0),0,atol=1e-12)
    np.testing.assert_allclose(z[:,24:].var(0),1/12,atol=1e-12)
    beta=rng.normal(size=216);coef,c=head.deployment(t,beta,.3)
    other=rng.normal(size=(15,216))+4
    np.testing.assert_allclose(other@coef+c,head.transform(other,t)@beta+.3,atol=1e-11)


def test_row_weight_keeps_mean_loss_regularization_fixed():
    rng=np.random.default_rng(5);z=rng.normal(size=(50,216));y=rng.normal(size=50);b=rng.normal(size=216)
    full_n=100;w=full_n/len(y)
    left=(w*np.sum((z@b-y)**2)+100*np.sum(b*b))/full_n
    right=np.mean((z@b-y)**2)+100/full_n*np.sum(b*b)
    np.testing.assert_allclose(left,right)


def test_native_zero_source_equals_TD_and_replicated_weight():
    rng=np.random.default_rng(6);raw=np.c_[rng.normal(size=(40,24)),np.zeros((40,192))];z,t=fit_transform(raw);y=rng.normal(size=40)
    b,c,cert=fit(z,y,80,'native');bt,ct,td=fit(z,y,80,'TD')
    np.testing.assert_allclose(b,bt,atol=1e-12);np.testing.assert_allclose(c,ct)
    repeated=np.repeat(z,2,axis=0);yy=np.repeat(y,2)
    br,cr,_=fit(repeated,yy,80,'TD')
    np.testing.assert_allclose(bt,br,atol=1e-12);np.testing.assert_allclose(ct,cr)
    assert cert['certified'] and td['certified'] and cert['row_weight']==2


def test_variance_pairwise_identity_and_finite_sample_error_decomposition():
    rng=np.random.default_rng(7);p=rng.normal(size=(3,30));y=rng.normal(size=30)
    value=perturbation_variance(p)
    np.testing.assert_allclose(value,np.mean(np.var(p,axis=0)))
    np.testing.assert_allclose(np.mean((p-y)**2),np.mean((p.mean(0)-y)**2)+value)
    assert perturbation_variance(np.ones((3,30)))==0
    with pytest.raises(ValueError):perturbation_variance(p[:2])
