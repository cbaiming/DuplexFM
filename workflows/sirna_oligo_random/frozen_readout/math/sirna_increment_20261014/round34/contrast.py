"""A fixed, label-free contrastive subspace followed by one linear Ridge."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

DIM = 16
ALPHA = 100.


def checked(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 216 or len(x) < 1 or not np.isfinite(x).all():
        raise ValueError('Finite TD24 + native source192 required')
    return x


def scale(x):
    mean = x.mean(0)
    sd = x.std(0)
    constant = np.ptp(x, axis=0) == 0
    positive = sd[(~constant) & (sd > 0)]
    floor = .01 * (float(np.median(positive)) if len(positive) else 1.)
    return mean, np.where(constant, 1., np.maximum(sd, floor))


def subspace(foreground, background, old_state, mode):
    f, b = checked(foreground), checked(background)
    if f.shape != b.shape or len(f) < 20 or not np.array_equal(f[:, :24], b[:, :24]):
        raise ValueError('Paired inputs and at least20 training rows required')
    if mode not in ('contrast', 'pooled'):
        raise ValueError('Fixed contrast or pooled map only')
    mu = np.asarray(old_state['src_mean'], float)
    sd = np.asarray(old_state['src_scale'], float)
    if mu.shape != (192,) or sd.shape != (192,) or not np.all(sd > 0):
        raise ValueError('Original foreground scaler contract')
    xf = (f[:, 24:] - mu) / sd
    xb = (b[:, 24:] - b[:, 24:].mean(0)) / sd
    xf -= xf.mean(0)
    xb -= xb.mean(0)
    cf, cb = xf.T @ xf / len(f), xb.T @ xb / len(b)
    tf, tb = float(np.trace(cf)), float(np.trace(cb))
    if min(tf, tb) <= 1e-12:
        raise ValueError('Foreground/background covariance must be nonzero')
    gamma = tf / tb
    matrix = cf + (-1. if mode == 'contrast' else 1.) * gamma * cb
    matrix = (matrix + matrix.T) / 2
    eigen, vectors = np.linalg.eigh(matrix)
    component = vectors[:, -DIM:][:, ::-1].T.copy()
    for row in component:
        if row[np.argmax(abs(row))] < 0:
            row *= -1
    values = eigen[-DIM:][::-1].copy()
    limit = 1e-10 * max(float(abs(eigen).max()), 1.)
    positive = int(np.count_nonzero(eigen > limit))
    return dict(mode=mode, components=component, eigenvalues=values, trace_foreground=tf,
                trace_background=tb, gamma=gamma, positive_eigenvalues=positive,
                positive_limit=limit, spectral_qualified=bool(values[-1] > limit),
                covariance_trace=float(np.trace(matrix)), foreground_scale=sd.copy())


def fitted_transform(train, old_state, mapping):
    x = checked(train)
    component = np.asarray(mapping['components'])
    sd = np.asarray(mapping['foreground_scale'])
    mean = x[:, 24:].mean(0)
    z = (x[:, 24:] - mean) / sd
    pm = z.mean(0)
    pcmean, pcscale = scale((z - pm) @ component.T)
    state = dict(td_mean=np.asarray(old_state['td_mean']), td_scale=np.asarray(old_state['td_scale']),
                 src_mean=mean, src_scale=sd, pca_mean=pm, components=component,
                 pc_mean=pcmean, pc_scale=pcscale)
    return transform(x, state), state


def transform(x, state):
    x = checked(x)
    td = (x[:, :24] - state['td_mean']) / state['td_scale']
    z = (x[:, 24:] - state['src_mean']) / state['src_scale'] - state['pca_mean']
    return np.c_[td, (z @ np.asarray(state['components']).T - state['pc_mean']) / state['pc_scale']]


def fit(z, y):
    z, y = np.asarray(z, float), np.asarray(y, float)
    if z.ndim != 2 or z.shape[1] != 40 or y.shape != (len(z),) or not np.isfinite(z).all() or not np.isfinite(y).all():
        raise ValueError('Finite40D single-response training contract')
    center = float(y.mean())
    matrix = z.T @ z + ALPHA * np.eye(40)
    q = z.T @ (y - center)
    beta = cho_solve(cho_factor(matrix, lower=True), q)
    error = float(abs(matrix @ beta - q).max() / len(y))
    residual = z @ beta + center - y
    return dict(beta=beta, intercept=center, alpha=ALPHA, normal_error=error,
                numerically_qualified=bool(error < 1e-8),
                objective=float(.5 * (residual @ residual + ALPHA * (beta @ beta))))


def deployment(state, beta, center):
    beta = np.asarray(beta)
    head = np.asarray(state['components']).T @ (beta[24:] / state['pc_scale'])
    coef = np.r_[beta[:24] / state['td_scale'], head / state['src_scale']]
    intercept = float(center - np.asarray(state['td_mean']) @ coef[:24]
                      - np.asarray(state['src_mean']) @ coef[24:]
                      - np.asarray(state['pca_mean']) @ head
                      - np.asarray(state['pc_mean']) @ (beta[24:] / state['pc_scale']))
    return coef, intercept


def qualify(records, numeric):
    primary = [r for r in records if r['condition'] == 'core_contrast']
    if len(primary) != 3:
        raise ValueError('Three primary training pools required')
    performance = sum(r['delta_PCC_vs_R4'] >= .005 and r['MSE_reduction_vs_R4_pct'] >= 0 for r in primary) >= 2 and all(r['delta_PCC_vs_R4'] >= -.005 for r in primary)
    paired = []
    for r in primary:
        control = next(s for s in records if s['setting'] == r['setting'] and s['condition'] == 'core_pooled')
        paired.append(dict(setting=r['setting'], MSE_reduction_vs_pooled_pct=100*(1-r['metrics']['MSE']/control['metrics']['MSE']),
                           delta_PCC_vs_pooled=r['metrics']['PCC']-control['metrics']['PCC'],
                           delta_PCC_vs_TD=r['metrics']['PCC']-r['TD_reference']['PCC'],
                           MSE_reduction_vs_TD_pct=100*(1-r['metrics']['MSE']/r['TD_reference']['MSE'])))
    contrastive = sum(r['MSE_reduction_vs_pooled_pct'] >= 0 for r in paired) >= 2 and np.mean([r['MSE_reduction_vs_pooled_pct'] for r in paired]) >= 0
    incremental = sum(r['delta_PCC_vs_TD'] >= 0 for r in paired) >= 2 and np.mean([r['MSE_reduction_vs_TD_pct'] for r in paired]) >= 0
    return dict(qualified=bool(performance and contrastive and incremental and numeric), performance_gate=bool(performance),
                contrastive_gate=bool(contrastive), source_increment_gate=bool(incremental), all_numerically_qualified=bool(numeric), pairs=paired)
