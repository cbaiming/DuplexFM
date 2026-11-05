"""One supervised hard split with shared regression and shrunken slopes."""
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve

ALPHA = 100.
LOCAL_MULTIPLIER = 4.
QUANTILES = (.25, .5, .75)


def _input(z):
    z = np.asarray(z, float)
    if z.ndim != 2 or z.shape[1] not in (24, 40) or not np.isfinite(z).all():
        raise ValueError('Finite TD24 or TD24+PCA16 input required')
    return z


def gates(z):
    z = _input(z)
    if len(z) < 20:
        raise ValueError('At least 20 training rows')
    minimum = max(10, int(np.ceil(.2 * len(z))))
    out = [dict(feature=-1, quantile_index=-1, threshold=0., right_fraction=0., counts=[len(z), 0])]
    for j in range(24):
        seen = set()
        for k, q in enumerate(QUANTILES):
            t = float(np.quantile(z[:, j], q, method='linear'))
            if t in seen:
                continue
            seen.add(t)
            g = z[:, j] > t
            counts = [int((~g).sum()), int(g.sum())]
            if min(counts) < minimum:
                continue
            out.append(dict(feature=j, quantile_index=k, threshold=t,
                            right_fraction=float(g.mean()), counts=counts))
    return out


def local_columns(z):
    return z if z.shape[1] == 24 else z[:, 24:]


def design(z, gate, mean=None):
    z = _input(z)
    w = local_columns(z)
    if gate['feature'] == -1:
        interaction = np.zeros_like(w)
    else:
        g = (z[:, gate['feature']] > gate['threshold']).astype(float) - gate['right_fraction']
        interaction = g[:, None] * w
    mean = interaction.mean(0) if mean is None else np.asarray(mean, float)
    if mean.shape != (w.shape[1],) or not np.isfinite(mean).all():
        raise ValueError('Interaction mean contract')
    return np.c_[z, interaction - mean], mean.copy()


def choose(candidates):
    return min(range(len(candidates)), key=lambda k: (
        round(candidates[k]['objective'], 10), candidates[k]['feature'],
        candidates[k]['quantile_index'], k))


def fit(z, y, reference_beta=None, deadline=None):
    z = _input(z)
    y = np.asarray(y, float)
    if y.shape != (len(z),) or not np.isfinite(y).all() or np.ptp(y) == 0:
        raise ValueError('Finite nonconstant training response required')
    if abs(z.mean(0)).max() > 1e-8:
        raise ValueError('Training features must be centered')
    p, q = z.shape[1], local_columns(z).shape[1]
    penalty = np.r_[np.full(p, ALPHA), np.full(q, ALPHA * LOCAL_MULTIPLIER)]
    center = float(y.mean())
    candidates, coefficients, means = [], [], []
    solves = 0
    for gate in gates(z):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError('No new split fit after CPU stage deadline')
        x, m = design(z, gate)
        h = x.T @ x + np.diag(penalty)
        target = x.T @ (y - center)
        reused = gate['feature'] == -1 and reference_beta is not None
        if reused:
            ref = np.asarray(reference_beta, float)
            if ref.shape != (p,) or not np.isfinite(ref).all():
                raise ValueError('Reference coefficient contract')
            b = np.r_[ref, np.zeros(q)]
        else:
            b = cho_solve(cho_factor(h, lower=True), target)
            solves += 1
        residual = y - center - x @ b
        obj = float(residual @ residual + np.dot(penalty, b * b))
        error = float(abs(h @ b - target).max())
        limit = 1e-9 * (1 + float(abs(target).max()))
        candidates.append(dict(**gate, objective=obj, normal_error=error,
                               normal_limit=limit, certified=error <= limit,
                               reused_reference=reused))
        coefficients.append(b)
        means.append(m)
    selected = choose(candidates)
    return dict(selected=selected, candidates=candidates, beta=np.array(coefficients),
                interaction_mean=np.array(means), intercept=center, new_linear_solves=solves,
                reference_reused=int(reference_beta is not None),
                all_certified=all(c['certified'] for c in candidates))


def predict(z, model):
    k = model['selected']
    x, _ = design(z, model['candidates'][k], model['interaction_mean'][k])
    return x @ model['beta'][k] + model['intercept']


def branch_predict(z, model):
    """Equivalent one-hot deployment: one local slope is used per sample."""
    z = _input(z)
    k = model['selected']
    gate = model['candidates'][k]
    b = model['beta'][k]
    p = z.shape[1]
    route = np.zeros(len(z), int) if gate['feature'] == -1 else (z[:, gate['feature']] > gate['threshold']).astype(int)
    left = -gate['right_fraction'] * b[p:]
    right = (1 - gate['right_fraction']) * b[p:]
    local = np.where(route[:, None] == 0, left[None, :], right[None, :])
    offset = model['intercept'] - model['interaction_mean'][k] @ b[p:]
    return z @ b[:p] + np.sum(local_columns(z) * local, axis=1) + offset, route


def qualify(records, numeric):
    core = [r for r in records if r['source'] == 'core_real']
    if len(core) != 3:
        raise ValueError('Three primary pools required')
    performance = (sum(r['delta_PCC_vs_R4'] >= .005 and r['MSE_reduction_vs_R4_pct'] >= 0 for r in core) >= 2
                   and all(r['delta_PCC_vs_R4'] >= -.005 for r in core))
    paired = []
    for r in core:
        td = next(s for s in records if s['setting'] == r['setting'] and s['source'] == 'TD_only')
        paired.append(dict(setting=r['setting'], delta_PCC=r['metrics']['PCC']-td['metrics']['PCC'],
                           MSE_reduction_pct=100*(1-r['metrics']['MSE']/td['metrics']['MSE'])))
    incremental = sum(r['delta_PCC'] >= 0 for r in paired) >= 2 and np.mean([r['MSE_reduction_pct'] for r in paired]) >= 0
    return dict(qualified=bool(performance and incremental and numeric), performance_gate=bool(performance),
                source_increment_gate=bool(incremental), all_numerically_qualified=bool(numeric),
                paired_gated_TD=paired)
