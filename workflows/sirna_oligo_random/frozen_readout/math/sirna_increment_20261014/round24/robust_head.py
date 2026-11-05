"""One shared ridge head: balanced risk or worst-of-two training group risk."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq

ALPHA = 100.0


def fit(z, y, group, mode):
    z, y, group = np.asarray(z, float), np.asarray(y, float), np.asarray(group)
    if (z.ndim != 2 or y.shape != (len(z),) or group.shape != y.shape
            or not np.isfinite(z).all() or not np.isfinite(y).all()
            or set(np.unique(group)) != {0, 1} or mode not in ('balanced', 'robust')):
        raise ValueError('Training input contract')
    n, p = z.shape
    x = np.c_[np.ones(n), z]
    penalty = np.r_[0., np.full(p, ALPHA)]
    counts = np.bincount(group.astype(int), minlength=2)
    grams = [n / counts[k] * x[group == k].T @ x[group == k] for k in (0, 1)]
    rhs = [n / counts[k] * x[group == k].T @ y[group == k] for k in (0, 1)]
    cache = {}

    def solve(q):
        q = float(q)
        if q not in cache:
            h = q * grams[0] + (1 - q) * grams[1] + np.diag(penalty)
            v = q * rhs[0] + (1 - q) * rhs[1]
            b = cho_solve(cho_factor(h, lower=True), v)
            residual = y - x @ b
            losses = np.array([n * np.mean(residual[group == k] ** 2) for k in (0, 1)])
            cache[q] = (b, losses, h, v)
        return cache[q]

    if mode == 'balanced':
        q = .5
        location = 'fixed_equal_group_weight'
        iterations = 0
    else:
        def derivative(q):
            losses = solve(q)[1]
            return float(losses[0] - losses[1])
        d0, d1 = derivative(0.), derivative(1.)
        if d0 <= 0:
            q, location, iterations = 0., 'lower_boundary', 0
        elif d1 >= 0:
            q, location, iterations = 1., 'upper_boundary', 0
        else:
            q, result = brentq(derivative, 0., 1., xtol=1e-12, rtol=1e-14,
                               maxiter=100, full_output=True, disp=True)
            if not result.converged:
                raise RuntimeError('Unconverged dual solve')
            location, iterations = 'interior', int(result.iterations)
    b, losses, h, v = solve(q)
    regularizer = float(np.dot(penalty, b * b))
    weighted = float(q * losses[0] + (1 - q) * losses[1] + regularizer)
    worst = float(max(losses) + regularizer)
    gap = worst - weighted if mode == 'robust' else 0.
    normal_error = float(np.max(abs(h @ b - v)))
    normal_limit = 1e-9 * (1 + float(np.max(abs(v))))
    gap_limit = 1e-9 * (1 + abs(worst))
    qualified = bool(normal_error <= normal_limit and -gap_limit <= gap <= gap_limit)
    certificate = dict(mode=mode, alpha=ALPHA, n=n, features=p, training_counts=counts,
                       group0_weight=float(q), location=location, dual_iterations=iterations,
                       weighted_ridge_solves=len(cache), group_scaled_losses=losses,
                       regularizer=regularizer, weighted_objective=weighted, worst_objective=worst,
                       duality_gap=float(gap), gap_limit=gap_limit,
                       normal_equation_error=normal_error, normal_equation_limit=normal_limit,
                       numerically_qualified=qualified)
    return b[1:], float(b[0]), certificate


def qualify(records, all_numeric):
    core = [r for r in records if r['source'] == 'core_real' and r['mode'] == 'robust']
    if len(core) != 3:
        raise ValueError('Exactly three primary pools required')
    performance = (sum(r['delta_PCC_vs_R4'] >= .005 and r['MSE_reduction_vs_R4_pct'] >= 0
                       for r in core) >= 2
                   and all(r['delta_PCC_vs_R4'] >= -.005 for r in core))
    paired = []
    for a in core:
        b = next(r for r in records if r['source'] == 'core_real' and r['mode'] == 'balanced'
                 and r['setting'] == a['setting'])
        paired.append(dict(setting=a['setting'], delta_PCC=a['metrics']['PCC'] - b['metrics']['PCC'],
                           MSE_reduction_pct=100 * (1 - a['metrics']['MSE'] / b['metrics']['MSE'])))
    paired_gate = (sum(r['MSE_reduction_pct'] >= 0 for r in paired) >= 2
                   and np.mean([r['MSE_reduction_pct'] for r in paired]) >= 0)
    return dict(qualified=bool(performance and paired_gate and all_numeric),
                performance_gate=bool(performance), paired_gate=bool(paired_gate),
                all_numerically_qualified=bool(all_numeric), paired_core_records=paired)
