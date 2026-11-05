"""The R4 global PCA16/Ridge readout, accepting either saved source width."""
import importlib.util
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_r43_original_core', R.parent / 'round4/core.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
transform = original.transform


def fit_transform(x):
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] not in (184, 216) or len(x) < 20 or not np.isfinite(x).all():
        raise ValueError('TD24 plus 160/192 source coordinates, at least 20 train rows')
    tm, ts = original.scaler(x[:, :24])
    sm, ss = original.scaler(x[:, 24:], True)
    z = (x[:, 24:] - sm) / ss
    pm = z.mean(0); z = z - pm
    e, v = np.linalg.eigh(z.T @ z / len(z))
    c = v[:, -16:][:, ::-1].T
    ev = np.maximum(e[-16:][::-1], 0)
    for row in c:
        if row[np.argmax(abs(row))] < 0:
            row *= -1
    pcm, pcs = original.scaler(z @ c.T, True)
    state = dict(td_mean=tm, td_scale=ts, src_mean=sm, src_scale=ss, pca_mean=pm,
                 components=c, eigenvalues=ev, pc_mean=pcm, pc_scale=pcs)
    return transform(x, state), state


def fit(z, y):
    z, y = np.asarray(z, float), np.asarray(y, float)
    if z.ndim != 2 or z.shape[1] != 40 or y.shape != (len(z),) or not np.isfinite(z).all() or not np.isfinite(y).all():
        raise ValueError('Finite aligned 40D design and response required')
    b, intercept = original.fit_heads(z, y, [dict(weight=1., alpha=100.)])
    return b[0], intercept


def deployment(t, beta, intercept):
    b = np.asarray(beta, float)
    td = b[:24] / np.asarray(t['td_scale'])
    pc = b[24:] / np.asarray(t['pc_scale'])
    c = np.asarray(t['components'])
    source = (pc @ c) / np.asarray(t['src_scale'])
    offset = (float(intercept) - np.asarray(t['td_mean']) @ td - np.asarray(t['src_mean']) @ source
              - np.asarray(t['pca_mean']) @ c.T @ pc - np.asarray(t['pc_mean']) @ pc)
    return np.r_[td, source], float(offset)


def qualify(records, all_numeric):
    if len(records) != 3 or len({r['setting'] for r in records}) != 3:
        raise ValueError('Three predefined development pools required')
    matched = (sum(r['delta_PCC_vs_R4_core'] >= .005 and r['MSE_reduction_vs_R4_core_pct'] >= 0
                   for r in records) >= 2 and all(r['delta_PCC_vs_R4_core'] >= -.005 for r in records))
    native = (sum(r['MSE_reduction_vs_R25_core_pct'] >= 0 for r in records) >= 2
              and np.mean([r['MSE_reduction_vs_R25_core_pct'] for r in records]) >= 0
              and all(r['delta_PCC_vs_R25_core'] >= -.005 for r in records))
    td = (sum(r['delta_PCC_vs_TD'] >= 0 for r in records) >= 2
          and np.mean([r['MSE_reduction_vs_TD_pct'] for r in records]) >= 0)
    return dict(qualified=bool(matched and native and td and all_numeric),
                matched_readout_gate=bool(matched), strong_native_gate=bool(native),
                TD_increment_gate=bool(td), all_numerically_qualified=bool(all_numeric))
