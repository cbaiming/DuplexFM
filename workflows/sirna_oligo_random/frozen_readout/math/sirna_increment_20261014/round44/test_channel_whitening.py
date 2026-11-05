import copy
import numpy as np
import pytest
from channel_whitening import fit_whitener, transform_channels, fold_coefficients, deployment, transform, fit, qualify


def data():
    rng=np.random.default_rng(42)
    raw=rng.normal(size=(96,216)); mixing=rng.normal(size=(32,32))/6
    raw[:,24:]=(raw[:,24:].reshape(-1,6,32)@mixing).reshape(-1,192)
    mean=raw.mean(0); sd=raw.std(0)
    scaler=dict(td_mean=mean[:24],td_scale=sd[:24],src_mean=mean[24:],src_scale=sd[24:])
    z=transform(raw,scaler); y=.4+z[:,:24]@rng.normal(size=24)/30+rng.normal(size=96)*.1
    return raw,scaler,z,y


def test_covariance_and_regularized_inverse_root():
    _,_,z,_=data(); t=fit_whitener(z); s=z[:,24:].reshape(-1,6,32)*np.sqrt(12)
    independent=sum(s[:,j,:].T@s[:,j,:] for j in range(6))/(6*len(z))
    np.testing.assert_allclose(t['covariance'],independent,atol=1e-13)
    np.testing.assert_allclose(t['inverse_root']@t['shrunk_covariance']@t['inverse_root'],np.eye(32),atol=1e-12)
    assert np.linalg.eigvalsh(t['weight'])[0]>0


def test_energy_preservation_isotropic_and_zero_cases():
    _,_,z,_=data(); t=fit_whitener(z); zz=transform_channels(z,t)
    np.testing.assert_allclose(np.sum(z[:,24:]**2),np.sum(zz[:,24:]**2),rtol=1e-12)
    s=np.r_[np.eye(32),-np.eye(32)]*np.sqrt(32.)
    iso=np.c_[np.zeros((64,24)),np.tile(s,(1,6))/np.sqrt(12)]
    np.testing.assert_allclose(fit_whitener(iso)['weight'],np.eye(32),atol=1e-12)
    zero=fit_whitener(np.zeros((4,216)))
    np.testing.assert_array_equal(zero['weight'],np.eye(32)); assert zero['zero_covariance']


def test_shared_orthogonal_basis_equivariance():
    _,_,z,_=data(); t=fit_whitener(z)
    q=np.linalg.qr(np.random.default_rng(43).normal(size=(32,32)))[0]
    zr=np.c_[z[:,:24],(z[:,24:].reshape(-1,6,32)@q).reshape(-1,192)]
    tr=fit_whitener(zr)
    np.testing.assert_allclose(tr['weight'],q.T@t['weight']@q,atol=1e-12)
    expected=transform_channels(z,t)[:,24:].reshape(-1,6,32)@q
    np.testing.assert_allclose(transform_channels(zr,tr)[:,24:].reshape(-1,6,32),expected,atol=1e-12)


def test_training_only_state_and_batch_invariance():
    _,_,z,_=data(); original=z.copy(); t=fit_whitener(z); saved=copy.deepcopy(t)
    other=np.random.default_rng(44).normal(size=(11,216))*9
    one=transform_channels(other,t); chunks=np.r_[transform_channels(other[:3],t),transform_channels(other[3:],t)]
    np.testing.assert_array_equal(z,original); np.testing.assert_allclose(one,chunks,atol=1e-12)
    for k in t:
        if isinstance(t[k],np.ndarray):np.testing.assert_array_equal(t[k],saved[k])
        else:assert t[k]==saved[k]
    np.testing.assert_array_equal(one[:,:24],other[:,:24])


def test_folded_raw_deployment():
    _,scaler,z,y=data(); t=fit_whitener(z); zz=transform_channels(z,t); b,c,_=fit(zz,y)
    coef,offset=deployment(scaler,b,c,t); rawother=np.random.default_rng(45).normal(size=(7,216))
    expected=transform_channels(transform(rawother,scaler),t)@b+c
    np.testing.assert_allclose(rawother@coef+offset,expected,atol=1e-11)


def test_zero_nuclear_penalty_matches_generalized_ridge():
    _,_,z,y=data(); t=fit_whitener(z); zz=transform_channels(z,t); b,c,s=fit(zz,y,ratio=0.)
    direct=np.linalg.lstsq(np.r_[zz,10*np.eye(216)],np.r_[y-y.mean(),np.zeros(216)],rcond=None)[0]
    np.testing.assert_allclose(b,direct,atol=1e-12)
    base=fold_coefficients(b,t); inverse=np.linalg.inv(t['weight']); penalty=np.eye(216)
    for j in range(6):penalty[24+32*j:56+32*j,24+32*j:56+32*j]=inverse.T@inverse
    rawsolve=np.linalg.solve(z.T@z+100*penalty,z.T@(y-y.mean()))
    np.testing.assert_allclose(base,rawsolve,atol=1e-11)
    np.testing.assert_allclose(base@penalty@base,b@b,atol=1e-12); assert s['certified']


def test_full_convex_global_certificate():
    _,_,z,y=data(); t=fit_whitener(z); zz=transform_channels(z,t); b,c,s=fit(zz,y)
    gram=zz.T@zz+100*np.eye(216); rhs=zz.T@(y-y.mean()); lam=s['lambda_value']
    u=rhs-gram@b; np.testing.assert_allclose(u[:24],np.zeros(24),atol=1e-10)
    dual=u[24:]; norm=np.linalg.norm(dual.reshape(6,32),2)
    if norm>lam:dual*=lam/norm
    dv=np.r_[np.zeros(24),dual]
    primal=.5*b@gram@b-rhs@b+lam*np.linalg.svd(b[24:].reshape(6,32),compute_uv=False).sum()
    gap=primal+.5*(rhs-dv)@np.linalg.solve(gram,rhs-dv)
    assert s['certified'] and -1e-8<=gap<=1e-9*s['gap_scale']+1e-10
    folded=fold_coefficients(b,t)[24:].reshape(6,32)
    np.testing.assert_allclose(folded@np.linalg.inv(t['weight']).T,b[24:].reshape(6,32),atol=1e-12)


def test_bad_input_and_center_contract():
    _,_,z,y=data()
    for bad in (z[:,:40],z[:0],z+1,np.full_like(z,np.nan)):
        with pytest.raises(ValueError):fit_whitener(bad)
    with pytest.raises(ValueError):transform_channels(z,dict(weight=np.eye(6)))
    with pytest.raises(ValueError):fold_coefficients(np.zeros(40),fit_whitener(z))
    with pytest.raises(ValueError):fit(transform_channels(z,fit_whitener(z)),np.ones(len(z)))


def test_fixed_gate_semantics():
    records=[dict(source='core_real',setting=str(i),delta_PCC_vs_R25=.006,MSE_reduction_vs_R25_pct=1.,
                  metrics=dict(PCC=.6,MSE=.1),TD_metrics=dict(PCC=.59,MSE=.11)) for i in range(3)]
    assert qualify(records,True)['qualified']; assert not qualify(records,False)['qualified']
    records[0]['delta_PCC_vs_R25']=-.006
    assert not qualify(records,True)['qualified']
    with pytest.raises(ValueError):qualify(records[:2],True)
