"""Training-loss-metric diagonal scaling; fixed block energy, no new strength."""
import importlib.util
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent
s = importlib.util.spec_from_file_location('_r53_partial49', R.parent/'round49/group_nuisance.py')
prior = importlib.util.module_from_spec(s); s.loader.exec_module(prior)
transform, deployment, fit, qualify = prior.transform, prior.deployment, prior.fit, prior.qualify
BLOCKS = ((0, 24), (24, 216))


def fit_scaling(z, labels):
    """Input is only the centered inner-training design, with no response."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape[1] != 216 or len(z) < 2 or not np.isfinite(z).all():
        raise ValueError('Finite native216 training matrix required')
    tz = prior.project(z, labels)
    g = np.asarray(labels, dtype=int)
    identity = len(np.unique(g)) in (1, len(z))
    v = np.sum(z*z, axis=0)
    m = np.sum(tz*tz, axis=0)
    factor = np.ones(216)
    blocks = []
    for first, last in BLOCKS:
        vv, mm = v[first:last], m[first:last]
        active = vv > 0
        if np.any(mm[active] <= 0) or np.any(mm[~active] != 0):
            raise ValueError('Positive definite metric or exact zero-column contract failed')
        block_ratio = float(mm.sum()/vv.sum()) if np.any(active) else None
        if not identity and np.any(active):
            factor[first:last][active] = np.sqrt(block_ratio*vv[active]/mm[active])
        # gamma=.5 implies factor in [sqrt(.5), sqrt(2)], without clipping.
        f = factor[first:last]
        if np.min(f) < np.sqrt(.5)-1e-12 or np.max(f) > np.sqrt(2)+1e-12:
            raise ValueError('Analytical factor bounds failed')
        adjusted = float(np.sum(mm*f*f))
        if not np.isclose(adjusted, mm.sum(), rtol=1e-12, atol=1e-12):
            raise ValueError('Block weighted-training energy changed')
        blocks.append(dict(first=first, last=last, active_columns=int(active.sum()),
            original_energy=float(vv.sum()), projected_energy=float(mm.sum()),
            adjusted_energy=adjusted, block_metric_ratio=block_ratio,
            factor_min=float(f.min()), factor_max=float(f.max()),
            factor_RMS=float(np.sqrt(np.mean(f*f)))))
    return factor, dict(identity=identity, blocks=blocks, finite=True,
        train_inputs_only=True, response_used=False, separate_TD_and_source=True)


def apply_scaling(z, factor):
    z = np.asarray(z, dtype=float); factor = np.asarray(factor, dtype=float)
    if z.ndim != 2 or z.shape[1] != 216 or factor.shape != (216,):
        raise ValueError('Native216 shape contract')
    if not np.isfinite(z).all() or not np.isfinite(factor).all() or np.any(factor <= 0):
        raise ValueError('Finite input and positive scale required')
    return z*factor


def raw_deployment(original_state, factor, beta, center, source):
    beta = np.asarray(beta, float)
    if beta.shape != ((216,) if source else (24,)):
        raise ValueError('Coefficient shape contract')
    full = beta if source else np.r_[beta, np.zeros(192)]
    return deployment(original_state, np.asarray(factor)*full, center)
