"""Per-sample source normalization; TD24 and source dimensionality unchanged."""
import numpy as np


def source_norm(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 216 or not np.isfinite(x).all():
        raise ValueError('Expected finite TD24 + source192')
    source = x[:, 24:]
    largest = np.max(np.abs(source), axis=1)
    scaled = np.divide(source, largest[:, None], out=np.zeros_like(source), where=largest[:, None] > 0)
    norm = largest * np.sqrt(np.einsum('ij,ij->i', scaled, scaled))
    if not np.isfinite(norm).all():
        raise ValueError('Source norm overflow')
    return norm


def direction(x):
    x = np.asarray(x, dtype=np.float64)
    norm = source_norm(x)
    src = np.divide(x[:, 24:], norm[:, None], out=np.zeros_like(x[:, 24:]), where=norm[:, None] > 1e-12)
    return np.concatenate([x[:, :24], src], axis=1)


def statistics(norm, pc1):
    norm = np.asarray(norm, float)
    pc1 = np.asarray(pc1, float)
    if norm.ndim != 1 or pc1.shape != norm.shape or len(norm) < 2:
        raise ValueError('Paired rows required')
    if not np.isfinite(norm).all() or not np.isfinite(pc1).all():
        raise ValueError('Finite values required')
    mean = float(norm.mean()); sd = float(norm.std())
    a, b = norm-mean, pc1-pc1.mean()
    denominator = float(np.sqrt((a@a)*(b@b)))
    corr = float(a@b/denominator) if denominator > 1e-14 else 0.
    cv = sd/mean if mean > 1e-12 else 0.
    return dict(norm_min=float(norm.min()), norm_max=float(norm.max()), norm_mean=mean, norm_sd=sd,
                norm_quantiles=np.quantile(norm, [.05,.5,.95]), norm_CV=cv,
                norm_PC1_PCC=corr, norm_PC1_r2=corr*corr,
                pool_qualified=bool(cv>=.10 and corr*corr>=.20), all_nonzero=bool(np.all(norm>1e-12)))


def qualification(records):
    core = [r for r in records if r['source']=='core_real']
    if len(core) != 3 or {r['setting'] for r in core} != {'Huesken','Mixset','Takayuki'}:
        raise ValueError('Exactly three Core training pools required')
    return bool(sum(r['pool_qualified'] for r in core)>=2 and all(r['all_nonzero'] for r in core))
