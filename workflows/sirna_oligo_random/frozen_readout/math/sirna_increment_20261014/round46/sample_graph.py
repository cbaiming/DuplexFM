"""Input-only neighbor graph and TD-residual smoothness qualification, no predictor."""
import numpy as np
from scipy.spatial.distance import cdist

K = 10


def graph(x, eligible, ids):
    x = np.asarray(x, float)
    eligible = np.asarray(eligible, bool)
    ids = np.asarray(ids, dtype=str)
    n = len(x)
    if x.ndim != 2 or eligible.shape != (n, n) or ids.shape != (n,):
        raise ValueError('Shape mismatch')
    if not np.isfinite(x).all() or len(set(ids)) != n:
        raise ValueError('Finite features and unique identities required')
    if not np.array_equal(eligible, eligible.T) or np.any(np.diag(eligible)):
        raise ValueError('Symmetric eligibility with no self edges required')
    distances = cdist(x, x, metric='sqeuclidean')
    selected = np.full((n, K), -1, dtype=np.int64)
    for i in range(n):
        candidates = np.flatnonzero(eligible[i])
        order = np.lexsort((ids[candidates], distances[i, candidates]))
        keep = candidates[order[:K]]
        selected[i, :len(keep)] = keep
    return selected


def energies(values, eligible, selected):
    values = np.asarray(values, float)
    n = len(values)
    if values.shape != (n,) or not np.isfinite(values).all():
        raise ValueError('Finite scalar values required')
    if eligible.shape != (n, n) or selected.shape != (n, K):
        raise ValueError('Shape mismatch')
    knn = np.full(n, np.nan)
    null = np.full(n, np.nan)
    counts = eligible.sum(axis=1)
    for i in range(n):
        js = selected[i][selected[i] >= 0]
        if len(js) != min(K, counts[i]) or len(set(js)) != len(js):
            raise ValueError('Incorrect neighbor count')
        if len(js):
            if not eligible[i, js].all():
                raise ValueError('Forbidden edge')
            knn[i] = np.mean((values[i] - values[js])**2)
            null[i] = np.mean((values[i] - values[eligible[i]])**2)
    active = counts > 0
    nullsum = float(np.nansum(null))
    return dict(n=n, active_rows=int(active.sum()), rows_ge5=int((counts >= 5).sum()),
                coverage=float(active.mean()), coverage_ge5=float((counts >= 5).mean()),
                knn_sum=float(np.nansum(knn)), null_sum=nullsum,
                ratio=float(np.nansum(knn)/nullsum) if nullsum > 1e-14 else None), knn, null


def pool(blocks):
    def combine(field):
        states = [b[field] for b in blocks]
        a = sum(s['active_rows'] for s in states)
        en = sum(s['knn_sum'] for s in states)
        e0 = sum(s['null_sum'] for s in states)
        return dict(active_rows=a, knn_sum=en, null_sum=e0,
                    knn_mean=en/a if a else None, null_mean=e0/a if a else None,
                    ratio=en/e0 if e0 > 1e-14 else None)
    residual = combine('residual')
    return dict(n=sum(b['n'] for b in blocks), residual=residual, response=combine('response'),
                blocks_below_null=sum(b['residual']['ratio'] is not None and b['residual']['ratio'] < 1 for b in blocks),
                coverage_qualified=all(b['residual']['coverage'] >= .9 and b['residual']['coverage_ge5'] >= .9 for b in blocks))


def qualify(records):
    decisions = []
    for c in (r for r in records if r['condition'] == 'core_real'):
        td = next(r for r in records if r['setting'] == c['setting'] and r['condition'] == 'TD_only')
        a, b = c['aggregate']['residual']['ratio'], td['aggregate']['residual']['ratio']
        defined = a is not None and b is not None
        passed = (defined and a <= .90 and a <= b - .05 and c['aggregate']['blocks_below_null'] >= 4
                  and c['aggregate']['coverage_qualified'])
        decisions.append(dict(setting=c['setting'],Core_ratio=a,TD_ratio=b,
                              Core_minus_TD=None if not defined else a-b,passed=bool(passed)))
    coverage = all(r['aggregate']['coverage_qualified'] for r in records)
    return dict(qualified=bool(coverage and sum(d['passed'] for d in decisions) >= 2),
                per_pool=decisions,coverage_qualified=coverage,qualification_only=True,goal_achieved=False)
