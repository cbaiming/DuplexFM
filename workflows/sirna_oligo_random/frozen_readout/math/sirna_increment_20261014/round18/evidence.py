"""A single linear readout with one Gaussian precision per input block."""
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

LOW, HIGH = np.log(1e-6), np.log(1e8)
INIT = np.log(100.)


def prepare(x, y, widths):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.shape != (len(x),) or len(x) < 3:
        raise ValueError('Invalid training shape')
    if sum(widths) != x.shape[1] or any(w < 1 for w in widths):
        raise ValueError('Invalid blocks')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite input')
    xm, ym = x.mean(0), float(y.mean())
    z, yc = x - xm, y - ym
    if not float(yc @ yc) > 0:
        raise ValueError('Constant labels: no noise-profile optimum')
    groups = np.repeat(np.arange(len(widths)), widths)
    return dict(z=z, yc=yc, mean=xm, intercept=ym, groups=groups,
                widths=tuple(widths), gram=z.T @ z, rhs=z.T @ yc,
                nu=len(y)-1)


def evaluate(theta, data):
    theta = np.asarray(theta, float)
    if theta.shape != (len(data['widths']),):
        raise ValueError('Invalid precision shape')
    lam = np.exp(theta)
    penalty = lam[data['groups']]
    k = data['gram'] + np.diag(penalty)
    ch = cho_factor(k, lower=True, check_finite=False)
    beta = cho_solve(ch, data['rhs'], check_finite=False)
    inv = cho_solve(ch, np.eye(len(beta)), check_finite=False)
    residual = data['yc'] - data['z'] @ beta
    rss = float(residual @ residual + (penalty * beta) @ beta)
    if rss <= 0 or not np.isfinite(rss):
        raise FloatingPointError('Invalid profiled residual variance')
    nu = data['nu']
    logdet = float(2 * np.log(np.diag(ch[0])).sum() - np.log(penalty).sum())
    objective = .5 * (nu * np.log(rss / nu) + logdet)
    grad = []
    for j, width in enumerate(data['widths']):
        take = data['groups'] == j
        grad.append(.5 * (nu / rss * lam[j] * float(beta[take] @ beta[take])
                           + lam[j] * float(np.diag(inv)[take].sum()) - width))
    return dict(objective=float(objective), gradient=np.array(grad), beta=beta,
                lambdas=lam, noise_variance=rss/nu, rss=rss, logdet=logdet)


def projected_gradient(theta, gradient):
    theta, gradient = np.asarray(theta), np.asarray(gradient)
    # For minimization, positive gradient is stationary at the lower bound.
    return np.where(((theta <= LOW+1e-10) & (gradient > 0)) |
                    ((theta >= HIGH-1e-10) & (gradient < 0)), 0., gradient)


def fit(x, y, widths, initial=None, checkpoint=None, deadline=None):
    data = prepare(x, y, widths)
    theta0 = np.full(len(widths), INIT)
    state0 = evaluate(theta0, data)
    if initial is not None:
        initial(theta0.copy(), state0['beta'].copy(), data['mean'].copy(), data['intercept'])
    history = []

    def objective(theta):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError('Fixed CPU deadline')
        s = evaluate(theta, data)
        return s['objective'], s['gradient']

    def callback(theta):
        s = evaluate(theta, data)
        history.append(dict(theta=theta.copy(), objective=s['objective'], gradient=s['gradient'].copy()))
        if checkpoint is not None and len(history) % 500 == 0:
            checkpoint(len(history), theta.copy())

    result = minimize(objective, theta0, jac=True, method='L-BFGS-B',
                      bounds=[(LOW, HIGH)]*len(widths), callback=callback,
                      options=dict(maxiter=1000, maxfun=10000, maxls=30,
                                   ftol=1e-12, gtol=1e-9))
    state = evaluate(result.x, data)
    pg = projected_gradient(result.x, state['gradient'])
    state.update(theta=result.x, mean=data['mean'], intercept=data['intercept'],
                 widths=list(widths), nu=data['nu'], iterations=int(result.nit),
                 evaluations=int(result.nfev), status=int(result.status),
                 optimizer_success=bool(result.success), message=str(result.message),
                 projected_gradient=pg, history=history,
                 numerically_qualified=bool(result.success and np.max(abs(pg)) <= 1e-6))
    assert len(history) == result.nit
    return state


def predict(x, state):
    return (np.asarray(x) - state['mean']) @ state['beta'] + state['intercept']
