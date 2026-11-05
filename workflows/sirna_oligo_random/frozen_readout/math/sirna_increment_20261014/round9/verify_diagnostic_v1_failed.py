"""Independent, read-only numerical audit of the bounded R9 qualification."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / 'round4'
BENCH = ROOT.parents[2] / 'core_six_settings_20261011' / 'benchmark' / 'data'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def table(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def stats(residuals):
    n = len(residuals)
    center = statistics.median(residuals)
    sigma = max(1.4826 * statistics.median(abs(v - center) for v in residuals), 1e-6)
    squares = [v * v for v in residuals]
    total = math.fsum(squares)
    count = math.ceil(n * .05)
    return {'n': n, 'median_residual': center, 'MAD_scale': sigma, 'top5_count': count,
            'top5_SSE_fraction': math.fsum(sorted(squares, reverse=True)[:count]) / total if total else 0.,
            'tail3MAD_fraction': sum(abs(v - center) > 3 * sigma for v in residuals) / n,
            'MSE': total / n}

def compare(actual, expected):
    for key, value in expected.items():
        assert math.isclose(float(actual[key]), value, rel_tol=1e-12, abs_tol=1e-14), (key, actual[key], value)

def main():
    started = time.monotonic()
    assert not (ROOT / 'verification.json').exists(), 'already audited'
    lock = read(ROOT / 'diagnostic_lock.json')
    for name, digest in lock['files'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in lock['dependencies'].items():
        assert sha(Path(name)) == digest, name
    result = read(ROOT / 'diagnostic.json')
    assert result['lock_sha256'] == sha(ROOT / 'diagnostic_lock.json')
    assert result['diagnostic_rows_sha256'] == sha(ROOT / 'diagnostic_rows.csv')
    assert result['new_fits'] == result['new_source_forward'] == result['GPU_hours'] == 0
    assert result['outer_test_metrics_used'] is False
    details = table(ROOT / 'diagnostic_rows.csv')
    plan = read(PRIOR / 'plan.json')
    records = {(r['setting'], r['view']): r for r in result['statistics']}
    assert len(records) == 12
    expected_rows = 0
    core = []
    for setting in ('Huesken', 'Mixset', 'Takayuki'):
        item = next(r for r in plan['folds'] if r['setting'] == setting and r['fold'] == 0)
        ids = item['fit_ids']
        assert result['fit_ids'][setting] == ids
        labels = {r['sample_id']: float(r['label']) for r in table(BENCH / f'{setting}.csv')}
        block = {}
        for k, part in enumerate(item['inner']):
            assert set(part['train']).isdisjoint(part['valid'])
            for i in part['valid']:
                assert i not in block
                block[i] = k
        assert set(block) == set(range(len(ids)))
        for view in ('core_real', 'full_real', 'initial_core', 'TD24'):
            expected_rows += len(ids)
            rows = [r for r in details if r['setting'] == setting and r['view'] == view]
            assert [r['sample_id'] for r in rows] == ids
            path = PRIOR / 'results' / 'folds' / setting / 'fold0' / ('core_real' if view == 'TD24' else view) / 'arrays.npz'
            with np.load(path, allow_pickle=False) as saved:
                predicted = saved['inner_oof'][:, 3 if view == 'TD24' else 18].copy()
            residuals = []
            for i, row in enumerate(rows):
                label = labels[ids[i]]
                residual = label - float(predicted[i])
                assert float(row['label']) == label
                assert float(row['prediction']) == float(predicted[i])
                assert float(row['residual']) == residual
                assert int(row['inner_block']) == block[i]
                residuals.append(residual)
            independent = stats(residuals)
            record = records[setting, view]
            compare(record, independent)
            assert len(record['inner_blocks']) == 5
            for k in range(5):
                assert record['inner_blocks'][k]['block'] == k
                compare(record['inner_blocks'][k], stats([v for i, v in enumerate(residuals) if block[i] == k]))
            if view == 'core_real':
                core.append(dict(setting=setting, **independent))
    assert len(details) == expected_rows
    qualified = any(r['top5_SSE_fraction'] >= .35 or r['tail3MAD_fraction'] >= .015 for r in core)
    assert result['qualified'] == qualified
    receipts = list(ROOT.glob('diagnostic_execution_*.json'))
    assert len(receipts) == 1 and read(receipts[0])['returncode'] == 0
    report = dict(passed=True, qualified=qualified, groups=12, block_groups=60, rows=expected_rows,
                  core=core, new_fits=0, new_source_forward=0, GPU_hours=0,
                  input_SHA_verified=True, verifier_sha256=sha(Path(__file__)),
                  diagnostic_sha256=sha(ROOT / 'diagnostic.json'),
                  diagnostic_lock_sha256=sha(ROOT / 'diagnostic_lock.json'),
                  execution_sha256=sha(receipts[0]), CPU_wall_seconds=time.monotonic()-started)
    with (ROOT / 'verification.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
