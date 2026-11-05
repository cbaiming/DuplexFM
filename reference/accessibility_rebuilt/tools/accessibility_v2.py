"""Shared, versioned contracts for observed-label accessibility fine-tuning.

The seven regions are candidate 19-nt target coordinates on the mRNA strand.
They are not experimentally annotated guide binding sites in icSHAPE data.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np

SCHEMA = 'accessibility-observed-v2'
REGIONS = ('target_14_19', 'target_11_18', 'target_8_11', 'target_0_8',
           'target_0_19', 'upstream', 'downstream')
REQUIRED_ARRAYS = ('sequences', 'median_accessibility_profiles', 'lengths',
                   'splits', 'view_ids', 'transcript_components')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def identity(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with open(temp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def file_contract(paths):
    """Call on effective inputs, not merely on their declared metadata."""
    return {str(name): sha256(path) for name, path in sorted(paths.items())}


def require_match(saved, expected, what='contract'):
    if saved != expected:
        if isinstance(saved, dict) and isinstance(expected, dict):
            changed = sorted(k for k in set(saved) | set(expected)
                             if saved.get(k) != expected.get(k))
        else:
            changed = ['value']
        raise ValueError(f'{what} mismatch: {changed}; use a new output directory for a new run')


def validate_completion(path, contract):
    """A completed cache is reusable only with matching inputs AND outputs."""
    path = Path(path)
    done = json.loads(path.read_text())
    require_match(done['contract'], contract, 'completed run')
    if done.get('status') != 'complete' or not done.get('artifacts'):
        raise ValueError('Incomplete completion manifest')
    root = path.parent.resolve()
    for name, expected in done['artifacts'].items():
        artifact = (root / name).resolve()
        if not artifact.is_relative_to(root) or not artifact.is_file():
            raise ValueError(f'Invalid completed artifact: {name}')
        require_match(sha256(artifact), expected, f'artifact {name}')
    return done


def load_dataset(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('status') != 'pass':
        raise ValueError('Dataset manifest is not validated')
    paths = {f'{k}.npy': root / f'{k}.npy' for k in REQUIRED_ARRAYS}
    paths['manifest.json'] = root / 'manifest.json'
    # Hash all effective metadata, including optional explicit observed masks.
    paths.update({p.name: p for p in root.glob('*.npy')})
    data_hashes = file_contract(paths)
    arrays = {k: np.load(root / f'{k}.npy', mmap_mode='r', allow_pickle=False)
              for k in REQUIRED_ARRAYS}
    y = arrays['median_accessibility_profiles']
    n = len(arrays['sequences'])
    if n == 0 or y.shape != (n, 60):
        raise ValueError('Expected nonempty N by 60 profiles')
    for k in REQUIRED_ARRAYS:
        if k != 'median_accessibility_profiles' and arrays[k].shape != (n,):
            raise ValueError(f'Invalid shape for {k}')
    lengths = arrays['lengths']
    splits, views = arrays['splits'], arrays['view_ids']
    for key in ('lengths', 'splits', 'view_ids', 'transcript_components'):
        if not np.issubdtype(arrays[key].dtype, np.integer):
            raise ValueError(f'{key} must contain integers')
    if np.any((lengths < 50) | (lengths > 60)):
        raise ValueError('Sequence lengths must lie in 50..60')
    if not np.isin(splits, [0, 1, 2]).all() or set(splits.tolist()) != {0, 1, 2}:
        raise ValueError('Invalid or empty train/validation/test split')
    if not np.isin(views, [0, 1]).all() or np.any(views[splits != 0] != 0):
        raise ValueError('Shifted views must be training only')
    components = arrays['transcript_components']
    group_sets = [set(components[splits == k].tolist()) for k in range(3)]
    seq_sets = [set(arrays['sequences'][splits == k].tolist()) for k in range(3)]
    for i in range(3):
        for j in range(i):
            if group_sets[i] & group_sets[j] or seq_sets[i] & seq_sets[j]:
                raise ValueError('Group or exact-sequence leakage across splits')
    coverage = np.empty(n, dtype=np.float32)
    explicit = np.load(root / 'observed_mask.npy', mmap_mode='r', allow_pickle=False) \
        if (root / 'observed_mask.npy').is_file() else None
    if explicit is not None and (explicit.shape != y.shape or explicit.dtype != np.bool_):
        raise ValueError('Explicit observed mask must be boolean and match profiles')
    for start in range(0, n, 8192):
        stop = min(start + 8192, n)
        valid = np.arange(60)[None, :] < lengths[start:stop, None]
        values = np.asarray(y[start:stop])
        observed = valid & np.isfinite(values)
        if np.isinf(values).any() or np.any(np.isfinite(values) & ~valid):
            raise ValueError('Infinite labels or finite padding labels')
        if np.any((values[observed] < 0) | (values[observed] > 1)):
            raise ValueError('Observed labels must lie in [0,1]')
        if np.any(observed.sum(1) == 0):
            raise ValueError('Rows with no observations must be excluded')
        if explicit is not None and not np.array_equal(explicit[start:stop], observed):
            raise ValueError('Observed mask disagrees with finite real-position labels')
        for seq, length in zip(arrays['sequences'][start:stop], lengths[start:stop]):
            seq = bytes(seq).decode('ascii').rstrip('\0')
            if len(seq) != length or not set(seq) <= set('ACGU'):
                raise ValueError('Invalid sequence or sequence-length mismatch')
        coverage[start:stop] = observed.sum(1) / lengths[start:stop]
    arrays['coverage'] = coverage
    return arrays, data_hashes, manifest


def region_bounds(lengths, site_starts=None):
    """Half-open mRNA coordinates; an explicit target start overrides centering."""
    lengths = np.asarray(lengths, dtype=np.int64)
    starts = (lengths - 19) // 2 if site_starts is None else np.asarray(site_starts, dtype=np.int64)
    if starts.shape != lengths.shape or np.any(starts < 0) or np.any(starts + 19 > lengths):
        raise ValueError('Invalid 19-nt candidate target coordinates')
    zero = np.zeros_like(starts)
    lo = np.stack([starts + x for x in (14, 11, 8, 0, 0)] + [zero, starts + 19], 1)
    hi = np.stack([starts + x for x in (19, 18, 11, 8, 19)] + [starts, lengths], 1)
    return lo, hi


def observed_loss(prediction, target, lengths, region_weight=0.0, min_region_coverage=0.5,
                  site_starts=None):
    import torch
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError('Prediction and target shapes differ')
    pos = torch.arange(target.shape[1], device=target.device)
    valid = pos[None, :] < lengths[:, None]
    observed = valid & torch.isfinite(target)
    count = observed.sum(1)
    if torch.any(count == 0):
        raise ValueError('Cannot train on a row without observed labels')
    if not torch.isfinite(prediction[valid]).all():
        raise FloatingPointError('Nonfinite prediction on a real nucleotide')
    # Clear before arithmetic: NaN multiplied by zero is still NaN.
    p = prediction.float().masked_fill(~observed, 0)
    y = target.float().masked_fill(~observed, 0)
    base = (p - y).abs().sum(1) / count
    starts = (lengths - 19) // 2 if site_starts is None else site_starts
    if torch.any(starts < 0) or torch.any(starts + 19 > lengths):
        raise ValueError('Invalid target coordinates')
    bounds = [(starts + a, starts + b) for a, b in ((14,19),(11,18),(8,11),(0,8),(0,19))]
    bounds += [(torch.zeros_like(starts), starts), (starts + 19, lengths)]
    errors, eligible = [], []
    for lo, hi in bounds:
        mask = observed & (pos[None, :] >= lo[:, None]) & (pos[None, :] < hi[:, None])
        support = mask.sum(1)
        ok = (support > 0) & (support >= (hi - lo) * min_region_coverage)
        error = ((p - y).masked_fill(~mask, 0).sum(1) / support.clamp_min(1)).abs()
        errors.append(error.masked_fill(~ok, 0))
        eligible.append(ok)
    num_regions = torch.stack(eligible, 1).sum(1)
    regional = torch.stack(errors, 1).sum(1) / num_regions.clamp_min(1)
    # Equal sequence weighting, with unsupported region terms contributing zero.
    return (base + region_weight * regional).mean(), {
        'base_l1': base.mean().detach(), 'region_l1': regional.mean().detach(),
        'rows_with_region': (num_regions > 0).sum().detach(),
    }


def pearson(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return None
    x, y = x - x.mean(), y - y.mean()
    denominator = math.sqrt(float(x @ x) * float(y @ y))
    return float(np.clip((x @ y) / denominator, -1, 1)) if denominator > 1e-15 else None


def profile_metrics(pred, target, lengths, min_region_coverage=0.5, site_starts=None):
    """Finite-only metrics; undefined correlations are JSON null, with counts."""
    pred, target = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    lengths = np.asarray(lengths)
    valid = np.arange(target.shape[1])[None, :] < lengths[:, None]
    mask = valid & np.isfinite(target)
    if not len(target) or np.any(mask.sum(1) == 0):
        raise ValueError('Metrics require observations in every row')
    if not np.isfinite(pred[valid]).all():
        raise FloatingPointError('Nonfinite prediction')
    error = np.where(mask, pred - np.where(mask, target, 0), 0)
    counts = mask.sum(1)
    within = [pearson(p[m], y[m]) for p, y, m in zip(pred, target, mask)]
    finite_within = [x for x in within if x is not None]
    x, y = target[mask], pred[mask]
    vx = float(np.var(x))
    slope = float(np.mean((x - x.mean()) * (y - y.mean())) / vx) if vx > 1e-15 else None
    lo, hi = region_bounds(lengths, site_starts)
    region_maes, region_details = [], {}
    for r, name in enumerate(REGIONS):
        region_mask = mask & (np.arange(60)[None, :] >= lo[:, r, None]) & (np.arange(60)[None, :] < hi[:, r, None])
        n = region_mask.sum(1)
        ok = (n > 0) & (n >= min_region_coverage * (hi[:, r] - lo[:, r]))
        rp = np.where(region_mask, pred, 0).sum(1)[ok] / n[ok]
        ry = np.where(region_mask, target, 0).sum(1)[ok] / n[ok]
        mae = float(np.abs(rp - ry).mean()) if ok.any() else None
        region_details[name] = {'rows': int(ok.sum()), 'mae': mae, 'pearson': pearson(rp, ry)}
        if mae is not None:
            region_maes.append(mae)
    return {
        'rows': len(target), 'observations': int(mask.sum()),
        'coverage_mean': float((counts / lengths).mean()),
        'micro_pearson': pearson(x, y),
        'within_sequence_pearson': float(np.mean(finite_within)) if finite_within else None,
        'defined_within_sequences': len(finite_within),
        'undefined_within_sequences': len(within) - len(finite_within),
        'mae': float(np.abs(error).sum() / counts.sum()),
        'sequence_mae': float((np.abs(error).sum(1) / counts).mean()),
        'bias': float(error.sum() / counts.sum()), 'calibration_slope_pred_on_true': slope,
        'region_mae': float(np.mean(region_maes)) if region_maes else None,
        'regions': region_details,
    }


def selection_value(metrics, policy='corr_region_tie'):
    correlations = [metrics['micro_pearson'], metrics['within_sequence_pearson']]
    if all(v is not None and math.isfinite(v) for v in correlations):
        score = sum(correlations) / 2
        defined = True
    else:
        score, defined = -metrics['sequence_mae'], False
    return {'correlations_defined': defined, 'score': score,
            'region_mae': metrics['region_mae'], 'sequence_mae': metrics['sequence_mae'], 'policy': policy}


def is_better(candidate, best, tolerance=1e-4):
    if best is None:
        return True
    if candidate['correlations_defined'] != best['correlations_defined']:
        return candidate['correlations_defined']
    if candidate['score'] > best['score'] + tolerance:
        return True
    if abs(candidate['score'] - best['score']) <= tolerance:
        key = 'region_mae' if candidate['policy'] == 'corr_region_tie' and candidate['region_mae'] is not None and best['region_mae'] is not None else 'sequence_mae'
        return candidate[key] < best[key] - 1e-7
    return False


def capture_rng():
    import torch
    n = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [n[0], n[1].tolist(), int(n[2]), int(n[3]), float(n[4])],
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import torch
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda']:
        if len(state['torch_cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count changed during strict resume')
        torch.cuda.set_rng_state_all([x.cpu() for x in state['torch_cuda']])


def epoch_indices(ids, count, seed, epoch):
    """Explicit deterministic sampler, independent of dropout/global RNG."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 7301]))
    ids = np.asarray(ids, dtype=np.int64)
    if not len(ids) or count <= 0:
        raise ValueError('Empty sampler')
    pieces = []
    left = count
    while left:
        piece = rng.permutation(ids)[:left]
        pieces.append(piece)
        left -= len(piece)
    return np.concatenate(pieces)
