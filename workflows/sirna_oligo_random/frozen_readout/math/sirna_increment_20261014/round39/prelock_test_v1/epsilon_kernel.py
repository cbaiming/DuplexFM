"""Single fixed squared epsilon-insensitive readout, certified convex dual."""
import time
import numpy as np

ALPHA = 100.
EPSILON_FRACTION = .1
MAX_ITER = 5000
GAP_TOL = 1e-9


def soft(v, threshold):
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.)


def centered_soft(v, threshold):
    """prox of threshold*L1 plus the exact zero-sum constraint."""
    v = np.asarray(v, float)
    if v.ndim != 1 or not len(v) or not np.isfinite(v).all() or not np.isfinite(threshold) or threshold < 0:
        raise ValueError('Finite vector and nonnegative threshold required')
    if threshold == 0:
        center = float(v.mean())
        return v - center, center
    lo, hi = float(v.min() - threshold), float(v.max() + threshold)
    for _ in range(70):
        center = (lo + hi) * .5
        if float(soft(v - center, threshold).sum()) > 0: lo = center
        else: hi = center
    center = (lo + hi) * .5
    result = soft(v - center, threshold)
    result -= result.mean()  # Roundoff-sized correction makes the dual feasible.
    return result, center


def certificate(gram, y, u, epsilon, alpha=ALPHA):
    f = gram @ u / alpha
    _, intercept = centered_soft(y - f, epsilon)
    residual = y - intercept - f
    loss_gradient = soft(residual, epsilon)
    norm = float(u @ f)
    primal = .5 * (norm + float(loss_gradient @ loss_gradient))
    dual = float(y @ u - .5 * (u @ u) - epsilon * np.abs(u).sum() - .5 * norm)
    gap = primal - dual
    relative_gap = max(0., gap) / max(1., primal)
    feasibility = float(abs(u.sum()))
    return dict(intercept=intercept, primal=primal, dual_objective=dual, gap=gap,
        relative_gap=relative_gap, sum_dual=float(u.sum()),
        max_loss_gradient_difference=float(np.max(np.abs(u - loss_gradient))),
        intercept_gradient=float(loss_gradient.sum()),
        train_MSE=float(np.mean(residual ** 2)),
        RKHS_squared_norm=norm / alpha, epsilon=epsilon,
        inactive_fraction=float(np.mean(np.abs(residual) <= epsilon)),
        numerically_qualified=bool(relative_gap <= GAP_TOL and gap >= -1e-9 * max(1., primal)
            and feasibility <= 1e-10 * (1 + np.abs(u).sum())))


def solve(gram, y, epsilon, deadline=None, alpha=ALPHA):
    gram, y = np.asarray(gram, float), np.asarray(y, float)
    if y.ndim != 1 or len(y) < 2 or gram.shape != (len(y), len(y)) or not np.isfinite(gram).all() or not np.isfinite(y).all():
        raise ValueError('Finite matching kernel and labels required')
    if not np.isfinite(epsilon) or epsilon < 0 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Nonnegative epsilon and positive alpha required')
    if np.max(np.abs(gram - gram.T)) > 1e-9:
        raise ValueError('Symmetric PSD kernel required')
    # Native R33 Gram is PSD by construction. Gershgorin gives a safe Lipschitz
    # upper bound; the unit quadratic in the dual gives strong convexity >= 1.
    lipschitz = 1. + float(np.abs(gram).sum(1).max()) / alpha
    momentum = (np.sqrt(lipschitz) - 1.) / (np.sqrt(lipschitz) + 1.)
    u = np.zeros(len(y)); extrapolated = u.copy()
    info = None
    for iteration in range(1, MAX_ITER + 1):
        if deadline is not None and time.monotonic() > deadline: raise TimeoutError('CPU stage deadline')
        gradient = extrapolated + gram @ extrapolated / alpha - y
        updated, _ = centered_soft(extrapolated - gradient / lipschitz, epsilon / lipschitz)
        extrapolated = updated + momentum * (updated - u)
        u = updated
        if iteration % 10 == 0 or iteration == MAX_ITER:
            info = certificate(gram, y, u, epsilon, alpha)
            if info['numerically_qualified']: break
    return u, dict(**info, iterations=iteration, alpha=alpha, lipschitz_upper_bound=lipschitz,
        maximum_iterations=MAX_ITER, gap_tolerance=GAP_TOL, epsilon_fraction=EPSILON_FRACTION)


def qualify(records, numeric):
    core = [r for r in records if r['source'] == 'core_real']
    if len(core) != 3: raise ValueError('Three fixed primary pools required')
    performance = sum(r['delta_PCC_vs_R4'] >= .005 and r['MSE_reduction_vs_R4_pct'] >= 0 for r in core) >= 2 and all(r['delta_PCC_vs_R4'] >= -.005 for r in core)
    loss_factor = sum(r['delta_PCC_vs_R33'] >= 0 and r['MSE_reduction_vs_R33_pct'] >= 0 for r in core) >= 2 and all(r['delta_PCC_vs_R33'] >= -.005 for r in core)
    pairs = []
    for r in core:
        td = next(t for t in records if t['setting'] == r['setting'] and t['source'] == 'TD_only')
        pairs.append(dict(setting=r['setting'], delta_PCC=r['metrics']['PCC'] - td['metrics']['PCC'],
            MSE_reduction_pct=100 * (1 - r['metrics']['MSE'] / td['metrics']['MSE'])))
    incremental = sum(r['delta_PCC'] >= 0 for r in pairs) >= 2 and np.mean([r['MSE_reduction_pct'] for r in pairs]) >= 0
    return dict(qualified=bool(performance and loss_factor and incremental and numeric),
        performance_gate=bool(performance), loss_factor_gate=bool(loss_factor),
        source_increment_gate=bool(incremental), all_numerically_qualified=bool(numeric), paired_kernel_TD=pairs)
