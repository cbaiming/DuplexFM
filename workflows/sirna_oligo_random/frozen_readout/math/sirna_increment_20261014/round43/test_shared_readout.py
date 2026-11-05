import numpy as np
import pytest
from shared_readout import original, fit_transform, transform, fit, deployment, qualify


def test_exact_existing_192_readout_parity():
    x = np.random.default_rng(1).normal(size=(50, 216))
    z, t = fit_transform(x)
    a, old = original.fit_transform(x)
    assert np.array_equal(a, z)
    assert all(np.array_equal(t[k], old[k]) for k in t)


def test_160_pca_matches_independent_svd_subspace():
    x = np.random.default_rng(11).normal(size=(55, 184))
    z, t = fit_transform(x)
    src = (x[:, 24:] - t['src_mean']) / t['src_scale'] - t['pca_mean']
    _, s, vt = np.linalg.svd(src, full_matrices=False)
    assert np.allclose(t['eigenvalues'], s[:16] ** 2 / len(x))
    assert np.allclose(t['components'].T @ t['components'], vt[:16].T @ vt[:16])
    assert np.max(abs(z.mean(0))) < 1e-12


def test_head_matches_independent_augmented_lstsq():
    rng = np.random.default_rng(5)
    z, _ = fit_transform(rng.normal(size=(60, 184)))
    y = rng.normal(size=60)
    b, c = fit(z, y)
    expected = np.linalg.lstsq(np.r_[z, 10 * np.eye(40)], np.r_[y - y.mean(), np.zeros(40)], rcond=None)[0]
    assert np.allclose(b, expected) and c == y.mean()


def test_raw_deployment_and_batch_invariance():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 184)); y = rng.normal(size=50)
    z, t = fit_transform(x[:50]); b, c = fit(z, y)
    coef, offset = deployment(t, b, c)
    p = transform(x[50:], t) @ b + c
    assert np.allclose(p, x[50:] @ coef + offset)
    assert np.allclose(transform(x[50:51], t) @ b + c, p[:1])


def test_transform_is_training_only_and_zero_padding_equivalent():
    rng = np.random.default_rng(4); x = rng.normal(size=(80, 184))
    z, t = fit_transform(x[:50]); copy = {k: np.array(v).copy() for k, v in t.items()}
    transform(x[50:] + 1e4, t)
    assert all(np.array_equal(t[k], copy[k]) for k in t)
    padded = np.c_[x[:50], np.zeros((50, 32))]
    zp, tp = original.fit_transform(padded)
    # Coefficients can change sign with eigensolvers; prediction-space Gram cannot.
    assert np.allclose(z @ z.T, zp @ zp.T, atol=1e-9)


def test_constant_source_coordinates_are_safe():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(50, 184)); x[:, -20:] = 4
    z, t = fit_transform(x)
    assert np.isfinite(z).all() and np.all(np.asarray(t['src_scale'])[-20:] == 1)
    assert np.all(np.asarray(t['src_mean'])[-20:] == 4)


def test_bad_shapes_nonfinite_and_response_rejected():
    for x in [np.zeros((50, 160)), np.zeros((19, 184)), np.full((50, 184), np.nan)]:
        with pytest.raises(ValueError):
            fit_transform(x)
    with pytest.raises(ValueError):
        fit(np.ones((30, 40)), np.ones(29))


def test_qualification_requires_same_readout_strong_head_and_TD():
    records = [dict(setting=s, delta_PCC_vs_R4_core=.006, MSE_reduction_vs_R4_core_pct=1,
                    delta_PCC_vs_R25_core=.001, MSE_reduction_vs_R25_core_pct=1,
                    delta_PCC_vs_TD=.001, MSE_reduction_vs_TD_pct=1) for s in ('A', 'B', 'C')]
    assert qualify(records, True)['qualified']
    assert not qualify(records, False)['qualified']
    records[0]['delta_PCC_vs_R25_core'] = -.006
    assert not qualify(records, True)['qualified']
    records[0]['delta_PCC_vs_R25_core'] = 0
    records[1]['MSE_reduction_vs_TD_pct'] = -10
    assert not qualify(records, True)['qualified']
