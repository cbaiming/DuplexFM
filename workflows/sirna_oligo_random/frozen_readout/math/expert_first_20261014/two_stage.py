"""Expert-only selection followed by a fixed-mask Ridge readout.

Training functions accept fit/validation arrays only, never test arrays.
The first-stage API has no TD24 argument.
"""
import numpy as np
from sklearn.linear_model import Ridge

EXPERTS = ('accessibility', 'utr_cross_attention', 'score_matrix_attention',
           'biological_priors', 'intarna_energy_geometry')
ALPHAS = (0.1, 1., 10., 100., 1000.)
RATIOS = (1., .3, .1, .03, .01, 0.)


def standardize_fit(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Invalid training features')
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1.
    return (x - mean) / scale, mean, scale


def certificate(h, q, b, lam):
    grad = h @ b - q
    errors = []
    for j in range(5):
        s = slice(j * 32, (j + 1) * 32)
        norm = np.linalg.norm(b[s])
        penalty = lam * np.sqrt(32)
        errors.append(float(np.linalg.norm(grad[s] + penalty * b[s] / norm))
                      if norm > 0 else float(max(0, np.linalg.norm(grad[s]) - penalty)))
    return max(errors)


def expert_path(z, y, alpha, ratios=RATIOS, max_iter=100000):
    """Strongly convex group Ridge; z must already be fit-centered/scaled."""
    z, y = np.asarray(z, np.float64), np.asarray(y, np.float64)
    if (z.ndim != 2 or z.shape[1] != 160 or y.shape != (len(z),)
            or len(z) < 2 or alpha <= 0 or not np.isfinite(z).all()
            or not np.isfinite(y).all() or np.max(np.abs(z.mean(0))) > 1e-8):
        raise ValueError('Expected centered 160D fit data')
    if any(r < 0 or r > 1 for r in ratios):
        raise ValueError('Invalid penalty ratio')
    n = len(z)
    intercept = float(y.mean())
    h = (z.T @ z + alpha * np.eye(160)) / n
    q = z.T @ (y - intercept) / n
    lm = max(np.linalg.norm(q[j * 32:(j + 1) * 32]) / np.sqrt(32) for j in range(5))
    eig = np.linalg.eigvalsh(h)
    lipschitz = float(eig[-1])
    minimum = max(float(eig[0]), alpha / n * .999)
    momentum = (np.sqrt(lipschitz) - np.sqrt(minimum)) / (np.sqrt(lipschitz) + np.sqrt(minimum))
    tolerance = float(1e-7 * (1 + np.max(np.abs(q))))
    current = np.zeros(160)
    output = []
    for ratio in ratios:
        lam = float(ratio * lm)
        if ratio == 1:
            current = np.zeros(160)
            iterations = 0
        elif ratio == 0:
            current = np.linalg.solve(h, q)
            iterations = 0
        else:
            accelerated = current.copy()
            for iterations in range(1, max_iter + 1):
                v = (accelerated - (h @ accelerated - q) / lipschitz).reshape(5, 32)
                norms = np.linalg.norm(v, axis=1)
                shrink = np.maximum(0, 1 - lam * np.sqrt(32) / lipschitz / np.maximum(norms, 1e-300))
                nxt = (v * shrink[:, None]).ravel()
                accelerated = nxt + momentum * (nxt - current)
                current = nxt
                if iterations == 1 or iterations % 20 == 0:
                    if certificate(h, q, current, lam) <= tolerance:
                        break
            else:
                raise RuntimeError(f'Nonconvergence alpha={alpha}, ratio={ratio}')
        error = certificate(h, q, current, lam)
        if not np.isfinite(current).all() or error > tolerance:
            raise RuntimeError('Uncertified solution')
        norms = [float(np.linalg.norm(current[j * 32:(j + 1) * 32])) for j in range(5)]
        output.append(dict(alpha=float(alpha), ratio=float(ratio), lambda_=lam,
                           lambda_max=float(lm), coef=current.copy(), intercept=intercept,
                           kkt=error, tolerance=tolerance, iterations=iterations,
                           group_norms=norms, active_groups=sum(v > 0 for v in norms)))
    return output


def key(row):
    return (row['val_mse'], row['active_groups'], -row['alpha'], -row['ratio'])


def select_experts(source_fit, y_fit, source_validation, y_validation):
    z, mean, scale = standardize_fit(source_fit)
    zv = (np.asarray(source_validation, np.float64) - mean) / scale
    yv = np.asarray(y_validation, np.float64)
    if zv.shape != (len(yv), 160) or not np.isfinite(zv).all() or not np.isfinite(yv).all():
        raise ValueError('Invalid validation data')
    trials = []
    for alpha in ALPHAS:
        for row in expert_path(z, y_fit, alpha):
            row['val_mse'] = float(np.mean((zv @ row['coef'] + row['intercept'] - yv) ** 2))
            trials.append(row)
    selected = min(trials, key=key)
    dense = min((r for r in trials if r['ratio'] == 0), key=key)
    mask = [v > 0 for v in selected['group_norms']]
    return dict(mean=mean, scale=scale, trials=trials, selected=selected,
                dense=dense, mask=mask)


def fixed_mask_ridge(x_fit, y_fit, x_validation, y_validation, mask, with_td):
    """Columns are always [TD24, 5x32D]. Source mask is never reselected."""
    mask = tuple(bool(v) for v in mask)
    if len(mask) != 5:
        raise ValueError('Five source switches required')
    x_fit, x_validation = np.asarray(x_fit, np.float64), np.asarray(x_validation, np.float64)
    if x_fit.shape[1] != 184 or x_validation.shape[1] != 184:
        raise ValueError('Expected 184D aligned arrays')
    columns = list(range(24)) if with_td else []
    columns += [k for j, active in enumerate(mask) if active for k in range(24 + 32*j, 24 + 32*(j+1))]
    z, mean, scale = standardize_fit(x_fit[:, columns])
    zv = (x_validation[:, columns] - mean) / scale
    trials = []
    for alpha in ALPHAS:
        if columns:
            model = Ridge(alpha=alpha, solver='svd').fit(z, y_fit)
            coef, intercept = model.coef_, float(model.intercept_)
        else:
            coef, intercept = np.zeros(0), float(np.mean(y_fit))
        trials.append(dict(alpha=float(alpha), coef=coef, intercept=intercept,
                           val_mse=float(np.mean((zv @ coef + intercept - y_validation) ** 2))))
    selected = min(trials, key=lambda r: (r['val_mse'], -r['alpha']))
    return dict(mask=list(mask), with_td=bool(with_td), columns=columns,
                mean=mean, scale=scale, trials=trials, selected=selected)


def predict_source(stage, row, source):
    return ((source - stage['mean']) / stage['scale']) @ row['coef'] + row['intercept']


def predict_fixed(stage, x):
    z = (x[:, stage['columns']] - stage['mean']) / stage['scale']
    return z @ stage['selected']['coef'] + stage['selected']['intercept']
