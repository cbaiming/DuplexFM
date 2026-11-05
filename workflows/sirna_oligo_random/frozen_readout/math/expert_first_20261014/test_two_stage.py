import numpy as np
import pytest
from sklearn.linear_model import Ridge
from two_stage import standardize_fit, expert_path, select_experts, fixed_mask_ridge, predict_fixed, key


def data(seed=42, n=220):
    r = np.random.default_rng(seed)
    x = r.normal(size=(n, 160))
    y = .4 * x[:, 0] - .2 * x[:, 37] + r.normal(size=n) * .1
    return x, y


def test_endpoints_and_ridge_equivalence():
    x, y = data()
    z, _, _ = standardize_fit(x)
    path = expert_path(z, y, 10., ratios=(1., 0.))
    assert np.count_nonzero(path[0]['coef']) == 0
    ref = Ridge(alpha=10, solver='svd').fit(z, y)
    np.testing.assert_allclose(path[-1]['coef'], ref.coef_, atol=1e-10)


def test_orthogonal_block_soft_threshold_closed_form():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(220, 160)); x -= x.mean(0)
    q, _ = np.linalg.qr(x)
    z = q * np.sqrt(len(x))
    y = z[:, 0] + .2 * z[:, 35]
    row = expert_path(z, y, 10., ratios=(.1,))[0]
    corr = (z.T @ (y-y.mean()) / len(z)).reshape(5, 32)
    norm = np.linalg.norm(corr, axis=1)
    expected = (corr * np.maximum(0, 1-row['lambda_']*np.sqrt(32)/np.maximum(norm, 1e-300))[:, None] / (1+10/len(z))).ravel()
    np.testing.assert_allclose(row['coef'], expected, atol=2e-7, rtol=1e-6)


def test_collinear_and_constant_columns_certified():
    x, y = data(n=100)
    x[:, 50:80] = x[:, :30]; x[:, -1] = 3
    z, _, _ = standardize_fit(x)
    for row in expert_path(z, y, .1):
        assert row['kkt'] <= row['tolerance']


def test_first_stage_has_no_td_or_test_input():
    import inspect
    assert list(inspect.signature(select_experts).parameters) == ['source_fit', 'y_fit', 'source_validation', 'y_validation']
    x, y = data(n=100)
    stage = select_experts(x[:80], y[:80], x[80:], y[80:])
    np.testing.assert_allclose(stage['mean'], x[:80].mean(0))
    assert stage['mask'] == [v > 0 for v in stage['selected']['group_norms']]
    assert len(stage['trials']) == 30


@pytest.mark.parametrize('mask,with_td', [([False]*5, False), ([False]*5, True), ([True, False, True, False, False], False), ([True, False, True, False, False], True)])
def test_second_stage_mask_and_td_contract(mask, with_td):
    x, y = data(n=120)
    full = np.c_[x[:, :24]*2, x]
    stage = fixed_mask_ridge(full[:90], y[:90], full[90:105], y[90:105], mask, with_td)
    cols = stage['columns']
    assert stage['mask'] == mask
    assert (0 in cols) == with_td
    assert all((24+32*j in cols) == active for j, active in enumerate(mask))
    altered = full[105:].copy()
    omitted = sorted(set(range(184)) - set(cols)); altered[:, omitted] += 1000
    np.testing.assert_array_equal(predict_fixed(stage, full[105:]), predict_fixed(stage, altered))
    if not cols:
        np.testing.assert_allclose(predict_fixed(stage, altered), y[:90].mean())


def test_tie_rule_and_validation_no_scaling_leak():
    rows = [dict(val_mse=1., active_groups=n, alpha=a, ratio=r) for n,a,r in [(2,10,.1),(1,1,.3),(1,100,.3),(1,100,1.)]]
    assert min(rows, key=key) is rows[-1]
    x, y = data(n=100); full=np.c_[x[:, :24], x]
    st = fixed_mask_ridge(full[:80], y[:80], full[80:]+999, y[80:], [True]*5, True)
    np.testing.assert_allclose(st['mean'], full[:80].mean(0))


def test_nonconvergence_is_an_error():
    x, y = data(); z, _, _ = standardize_fit(x)
    with pytest.raises(RuntimeError, match='Nonconvergence'):
        expert_path(z, y, .1, ratios=(.03,), max_iter=0)
