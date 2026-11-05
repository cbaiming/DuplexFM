"""One fixed-width-readout comparison of an already exported source."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse, importlib.util, sys, time, traceback, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R = Path(__file__).resolve().parent; B = R.parent; RESEARCH = B.parent
sys.path.insert(0, str(R))
from shared_readout import fit_transform, transform, fit, deployment, qualify
spec = importlib.util.spec_from_file_location('_r43_old4', B / 'round4/run.py')
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
io = old.io
P = B / 'round4'; G = B / 'round1'; N = B / 'round25'
FEATURES = RESEARCH / 'expert_selection_20261014/results_download/verified/features'
SETTINGS = old.SETTINGS[:3]
FILES = ('run.py', 'shared_readout.py', 'verify.py', 'test_shared_readout.py', 'PROTOCOL.md', 'CODE_AUDIT.md', 'pytest.xml')


def inputs():
    receipt = io.read(FEATURES / 'complete.json')
    io.check_files(FEATURES, receipt['files_sha256'])
    assert receipt['source_seed'] == 42 and receipt['source_audit']['source_frozen']
    assert receipt['source_checkpoint_sha256'] == '6c97cd93a1b6089caa62e4ddadde88839ba1dae9fa49eeea18345317d114a8a4'
    rows = []
    for setting, n in zip(SETTINGS, (2196, 469, 664)):
        part = io.table(old.BENCH / 'data' / f'{setting}.csv')
        assert len(part) == n and all(r['cohort'] == setting for r in part)
        rows.extend(part)
    meta = io.table(FEATURES / 'row_identity.csv')
    idx = {r['sample_id']: i for i, r in enumerate(meta)}
    assert len(idx) == len(meta) == 3688
    src = np.load(FEATURES / 'slots.npy', allow_pickle=False)
    assert src.shape == (3688, 160)
    for row in rows:
        assert all(row[k] == meta[idx[row['sample_id']]][k] for k in ('sample_id', 'cohort', 'guide', 'target57'))
    td = np.array([[float(x) for x in row['td'].split(',')] for row in rows])
    raw = np.c_[td, src[[idx[row['sample_id']] for row in rows]].astype(float)]
    assert raw.shape == (3329, 184) and np.isfinite(raw).all()
    return rows, raw, receipt


def register():
    assert not (R / 'lock.json').exists()
    old.validate()
    suites = list(ET.parse(R / 'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests', 0)) for s in suites) == 8
    assert not any(int(s.get(k, 0)) for s in suites for k in ('errors', 'failures', 'skipped'))
    assert io.read(P / 'plan.json')['folds'] == io.read(G / 'plan.json')['folds']
    deps = {}
    for base in (P, G, N):
        lock = io.read(base / 'lock.json')
        io.check_files(base, lock['files']); deps.update(lock['dependencies'])
        for name in lock['files']:
            deps[str((base / name).resolve())] = io.sha(base / name)
        for p in (base / 'lock.json', base / 'verification.json'):
            deps[str(p.resolve())] = io.sha(p)
        assert io.read(base / 'verification.json')['passed']
    for base, scope in ((P, 'core_real'), (G, None)):
        manifest = io.read(base / 'results/complete.json')['files_sha256']
        for setting in SETTINGS:
            directory = base / 'results/folds' / setting / 'fold0'
            if scope is not None: directory /= scope
            for name in ('arrays.npz', 'transforms.json', 'complete.json'):
                p = directory / name
                assert io.sha(p) == manifest[p.relative_to(base / 'results').as_posix()]
                deps[str(p.resolve())] = io.sha(p)
        deps[str((base / 'results/complete.json').resolve())] = io.sha(base / 'results/complete.json')
    manifest = io.read(N / 'diagnostic.json')['pilot_files_sha256']
    for setting in SETTINGS:
        for j in range(5):
            for name in ('model.npz', 'state.json', 'complete.json'):
                p = N / 'pilot' / setting / 'core_real' / f'inner{j}' / name
                assert io.sha(p) == manifest[p.relative_to(N / 'pilot').as_posix()]
                deps[str(p.resolve())] = io.sha(p)
    previous = io.read(B / 'round42/closure_receipt.json')
    assert previous['status'] == 'closed' and previous['readback_passed']
    for name, h in previous['files_sha256'].items(): assert io.sha(B / name) == h
    for p in (B / 'round42/closure_receipt.json', FEATURES / 'complete.json', FEATURES / 'slots.npy', FEATURES / 'row_identity.csv'):
        deps[str(p.resolve())] = io.sha(p)
    for name, h in deps.items(): assert io.sha(Path(name)) == h, name
    io.write(R / 'lock.json', dict(files={n: io.sha(R / n) for n in FILES}, dependencies=deps,
        versions=io.versions(), settings=SETTINGS, source_seed=42, primary='released_slots42_global16',
        source_checkpoint_sha256='6c97cd93a1b6089caa62e4ddadde88839ba1dae9fa49eeea18345317d114a8a4',
        source_width=160, PCA_width=16, TD_width=24, alpha=100., source_weight=1.,
        new_heads=15, new_transform_bundles=15, new_PCA=15, new_internal_OOF_rows=2299,
        reused_R4_core_heads=15, reused_R25_core_heads=15, reused_R4_TD_heads=15, reused_R1_group_heads=15,
        new_source_forward=0, CPU_only=True, CPU_limit_seconds=900))
    print('REGISTERED', io.sha(R / 'lock.json'), flush=True)


def validate():
    lock = io.read(R / 'lock.json'); io.check_files(R, lock['files'])
    for name, h in lock['dependencies'].items(): assert io.sha(Path(name)) == h, name
    assert lock['versions'] == io.versions()
    return lock


def stats(y, p):
    pc, ms = old.score(y, np.asarray(p)[:, None])
    return dict(PCC=float(pc[0]), MSE=float(ms[0]))


def run(start):
    validate(); assert not (R / 'pilot').exists() and not (R / 'diagnostic.json').exists()
    threads = threadpool_info(); assert all(t['num_threads'] == 1 for t in threads)
    rows, raw, source = inputs(); lookup = {row['sample_id']: i for i, row in enumerate(rows)}
    plan = io.read(P / 'plan.json'); records = []; details = []; pools = {}; all_numeric = True; count = 0
    for setting in SETTINGS:
        item = next(f for f in plan['folds'] if f['setting'] == setting and f['fold'] == 0)
        ids = item['fit_ids']; pools[setting] = ids
        assert set(ids).isdisjoint(item['test_ids'])
        ii = np.array([lookup[s] for s in ids]); x = raw[ii]
        y = np.array([float(rows[i]['label']) for i in ii])
        with np.load(P / 'results/folds' / setting / 'fold0/core_real/arrays.npz', allow_pickle=False) as a:
            refs = dict(R4_core=a['inner_oof'][:, 18].copy(), TD=a['inner_oof'][:, 3].copy())
        with np.load(G / 'results/folds' / setting / 'fold0/arrays.npz', allow_pickle=False) as a:
            refs['R1_group'] = a['inner_oof'][:, 3].copy()
        refs['R25_core'] = np.full(len(y), np.nan)
        pred = np.full(len(y), np.nan); blocks = []
        for j, sp in enumerate(item['inner']):
            if time.monotonic() - start > 900: raise TimeoutError('CPU pilot cap')
            tr, va = np.array(sp['train']), np.array(sp['valid'])
            root = N / 'pilot' / setting / 'core_real' / f'inner{j}'
            st = io.read(root / 'state.json'); co = io.read(root / 'complete.json')
            assert st['train_indices'] == tr.tolist() and st['valid_indices'] == va.tolist()
            io.check_files(root, co['files_sha256'])
            with np.load(root / 'model.npz', allow_pickle=False) as a: refs['R25_core'][va] = a['prediction']
            before = time.monotonic(); z, t = fit_transform(x[tr]); b, c = fit(z, y[tr])
            pp = transform(x[va], t) @ b + c; coef, offset = deployment(t, b, c)
            normal = float(np.max(abs((z.T @ z + 100 * np.eye(40)) @ b - z.T @ (y[tr] - c))) / len(tr))
            valid = bool(normal < 1e-8 and np.isfinite(pp).all()); all_numeric &= valid
            pred[va] = pp; count += 1; out = R / 'pilot' / setting / f'inner{j}'
            old.npz_write(out / 'model.npz', beta=b, intercept=np.array(c), prediction=pp,
                          deployment_coef=coef, deployment_intercept=np.array(offset))
            io.write(out / 'transform.json', t)
            io.write(out / 'state.json', dict(setting=setting, inner=j, source='released_slots42', train_indices=tr,
                valid_indices=va, normal_residual=normal, numerically_qualified=valid,
                CPU_head_wall_seconds=time.monotonic() - before))
            io.write(out / 'complete.json', dict(lock_sha256=io.sha(R / 'lock.json'),
                files_sha256={p.name: io.sha(p) for p in out.iterdir() if p.is_file()}))
            blocks.append(dict(inner=j, n=len(va), metrics=stats(y[va], pp), normal_residual=normal,
                numerically_qualified=valid, references={k: stats(y[va], p[va]) for k, p in refs.items()}))
        assert np.isfinite(pred).all() and all(np.isfinite(p).all() for p in refs.values())
        metric = stats(y, pred); reference = {k: stats(y, p) for k, p in refs.items()}
        rec = dict(setting=setting, source='released_slots42_global16', n=len(y), metrics=metric,
                   references=reference, inner_blocks=blocks)
        for k, rr in reference.items():
            rec[f'delta_PCC_vs_{k}'] = metric['PCC'] - rr['PCC']
            rec[f'MSE_reduction_vs_{k}_pct'] = 100 * (1 - metric['MSE'] / rr['MSE'])
        records.append(rec)
        details.extend(dict(setting=setting, sample_id=s, label=float(y[i]), prediction=float(pred[i]),
                           **{k+'_prediction': float(p[i]) for k, p in refs.items()}) for i, s in enumerate(ids))
        print(setting, 'deltaPCCvsCore', rec['delta_PCC_vs_R4_core'], 'vsNative', rec['delta_PCC_vs_R25_core'], flush=True)
    assert count == 15 and len(details) == 2299
    io.write_csv(R / 'diagnostic_rows.csv', details); gates = qualify(records, all_numeric)
    io.write(R / 'diagnostic.json', dict(**gates, records=records, fit_ids=pools, source_identity=source,
        new_heads=15, new_transform_bundles=15, new_PCA=15, new_internal_OOF_rows=2299,
        reused_heads_each_reference=15, new_source_forward=0, GPU_hours=0, outer_test_metrics_used=False,
        threadpool=threads, lock_sha256=io.sha(R / 'lock.json'), diagnostic_rows_sha256=io.sha(R / 'diagnostic_rows.csv'),
        pilot_files_sha256={p.relative_to(R / 'pilot').as_posix(): io.sha(p) for p in sorted((R / 'pilot').rglob('*')) if p.is_file()}))
    validate(); print('QUALIFIED', gates['qualified'], flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=('register', 'run'))
    if ap.parse_args().mode == 'register': register()
    else:
        mutex = R / 'execution.mutex'
        with mutex.open('x') as f: f.write(str(os.getpid()))
        start = time.monotonic(); rc = 1; error = None
        try: run(start); rc = 0
        except Exception: error = traceback.format_exc(); raise
        finally:
            io.write(R / f'execution_{time.time_ns()}.json', dict(returncode=rc, error=error,
                CPU_wall_seconds=time.monotonic() - start, GPU_hours=0, new_source_forward=0))
            mutex.unlink()
