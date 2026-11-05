import numpy as np
import pytest
from contrast import subspace, fitted_transform, transform, fit, deployment, scale


def fixture(seed=42):
    rng = np.random.default_rng(seed)
    td = rng.normal(size=(240,24))
    f = rng.normal(size=(240,192))
    b = rng.normal(size=(240,192)) * np.r_[np.full(48,.1),np.full(144,2.)]
    mu, sd = scale(f)
    old = dict(td_mean=td.mean(0),td_scale=td.std(0),src_mean=mu,src_scale=sd)
    return np.c_[td,f],np.c_[td,b],old


def test_contrast_finds_enriched_directions_and_trace_matches():
    f,b,t = fixture()
    c = subspace(f,b,t,'contrast')
    assert c['spectral_qualified'] and c['positive_eigenvalues'] >= 16
    assert np.sum(c['components'][:,:48]**2) > 8
    np.testing.assert_allclose(c['gamma']*c['trace_background'],c['trace_foreground'])
    assert abs(c['covariance_trace']) < 1e-10


def test_background_positive_scale_and_translation_invariance():
    f,b,t = fixture()
    c = subspace(f,b,t,'contrast')
    other = b.copy();other[:,24:] = b[:,24:]*7.3+5.
    q = subspace(f,other,t,'contrast')
    np.testing.assert_allclose(c['components'].T@c['components'],q['components'].T@q['components'],atol=1e-11)


def test_identical_background_is_not_qualified():
    f,_,t = fixture()
    c = subspace(f,f,t,'contrast')
    assert not c['spectral_qualified'] and c['positive_eigenvalues'] == 0


def test_component_spectrum_and_rayleigh_equations():
    f,b,t = fixture()
    for mode,sign in [('contrast',-1),('pooled',1)]:
        c = subspace(f,b,t,mode)
        xf = (f[:,24:]-f[:,24:].mean(0))/t['src_scale']
        xb = (b[:,24:]-b[:,24:].mean(0))/t['src_scale']
        cov = np.cov(xf,rowvar=False,bias=True)+sign*c['gamma']*np.cov(xb,rowvar=False,bias=True)
        v = c['components'].T
        np.testing.assert_allclose(cov@v,v*c['eigenvalues'],atol=1e-10)
        np.testing.assert_allclose(v.T@v,np.eye(16),atol=1e-11)


def test_batch_independent_transform_and_affine_deployment():
    f,b,t = fixture()
    c = subspace(f[:200],b[:200],dict(t,src_mean=f[:200,24:].mean(0)), 'contrast')
    # The numerical method requires caller-verified original train scaler; synthetic scalar reuse is harmless here.
    z,state = fitted_transform(f[:200],t,c)
    y = np.linspace(-1,1,200);m = fit(z,y)
    coef,intercept = deployment(state,m['beta'],m['intercept'])
    expected = transform(f[200:],state)@m['beta']+m['intercept']
    np.testing.assert_allclose(f[200:]@coef+intercept,expected,atol=1e-12)
    np.testing.assert_allclose(np.r_[transform(f[200:213],state)@m['beta'],transform(f[213:],state)@m['beta']]+m['intercept'],expected)


def test_ridge_matches_augmented_lstsq_and_response_affinity():
    f,b,t = fixture();z,state = fitted_transform(f,t,subspace(f,b,t,'contrast'))
    y = np.sin(np.arange(len(f)))
    m = fit(z,y);reference = np.linalg.lstsq(np.r_[z,10*np.eye(40)],np.r_[y-y.mean(),np.zeros(40)],rcond=None)[0]
    np.testing.assert_allclose(m['beta'],reference,atol=1e-12)
    assert m['numerically_qualified']
    other = fit(z,2*y+3)
    np.testing.assert_allclose(other['beta'],2*m['beta'],atol=1e-12)
    np.testing.assert_allclose(other['intercept'],2*m['intercept']+3)


def test_shared_projection_initialization_does_not_require_foreground_at_prediction():
    f,b,t = fixture();c = subspace(f,b,t,'contrast')
    z,state = fitted_transform(b,t,c)
    np.testing.assert_allclose(z.mean(0),0,atol=1e-12)
    np.testing.assert_allclose(z.std(0),1,atol=1e-10)
    assert transform(b[:5],state).shape == (5,40)
    assert not any(k in state for k in ('background','foreground','labels'))


def test_contracts_reject_misaligned_nonfinite_and_degenerate_background():
    f,b,t = fixture()
    for wrong in [b[:-1],np.c_[b[:,:24]+1,b[:,24:]],np.full(b.shape,np.nan)]:
        with pytest.raises(ValueError):subspace(f,wrong,t,'contrast')
    with pytest.raises(ValueError):subspace(f,np.c_[b[:,:24],np.zeros((len(b),192))],t,'contrast')
    with pytest.raises(ValueError):subspace(f,b,t,'adaptive')
    with pytest.raises(ValueError):fit(np.zeros((4,40)),np.full(4,np.nan))
