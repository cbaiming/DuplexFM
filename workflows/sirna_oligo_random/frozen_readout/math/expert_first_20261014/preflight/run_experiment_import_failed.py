"""CPU-only, locked two-stage experiment; prior source features are read-only."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import csv
import hashlib
import io
import json
import platform
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import scipy
import sklearn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from two_stage import EXPERTS, ALPHAS, RATIOS, select_experts, fixed_mask_ridge, predict_source, predict_fixed

HERE = Path(__file__).resolve().parent
COHORTS = ('Huesken', 'Mixset', 'Takayuki', 'wang2009', 'alas1_patent')
COUNTS = (2196, 469, 664, 131, 228)
SETTINGS = ('Huesken', 'Mixset', 'Takayuki', 'Inter_Hu_to_Mix', 'wang2009', 'alas1_patent')
NEW = ('ExpertsOnly_Ridge', 'ExpertsOnly_GroupSparse', 'SelectedExperts_Ridge', 'SelectedExperts_TD24_Ridge')
OLD = ('TD24_Ridge', 'Expert5_Ridge', 'Expert5_GroupSparse')
CODE = ('two_stage.py', 'run_experiment.py', 'verify_results.py', 'test_two_stage.py', 'PROTOCOL.md', 'CODE_AUDIT.md')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def atomic(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f'Refuse overwrite: {path}')
        return
    temp = path.with_name(path.name + '.tmp')
    with temp.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def write(path, value):
    atomic(path, (json.dumps(plain(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode())


def write_csv(path, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    atomic(path, stream.getvalue().encode())


def check_files(root, files):
    for name, digest in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != digest:
            raise ValueError(f'File identity mismatch: {name}')


def load_inputs(prior, benchmark):
    old_lock = read(prior/'readout_lock.json')
    check_files(benchmark, old_lock['benchmark'])
    check_files(prior/'results', read(prior/'results/complete.json')['files_sha256'])
    previous_audit = read(prior/'local_verification.json')
    if not previous_audit['passed'] or previous_audit['results_complete_sha256'] != sha(prior/'results/complete.json'):
        raise ValueError('Previous results not independently verified')
    fc = read(prior/'features/complete.json')
    check_files(prior/'features', fc['files_sha256'])
    if fc['experts'] != list(EXPERTS) or fc['source_seed'] != 42 or not fc['source_audit']['source_frozen']:
        raise ValueError('Wrong frozen source')
    rows = []
    for cohort, count in zip(COHORTS, COUNTS):
        part = table(benchmark/'data'/f'{cohort}.csv')
        if len(part) != count or any(r['cohort'] != cohort for r in part):
            raise ValueError('Dataset identity')
        rows.extend(part)
    identity = table(prior/'features/row_identity.csv')
    if len(rows) != 3688 or len(identity) != 3688 or len({r['sample_id'] for r in rows}) != 3688:
        raise ValueError('Duplicate or missing inputs')
    for a, b in zip(rows, identity):
        if any(a[k] != b[k] for k in ('sample_id', 'cohort', 'guide', 'target57')):
            raise ValueError('Feature ordering mismatch')
    source = np.load(prior/'features/slots.npy', allow_pickle=False).astype(np.float64)
    if source.shape != (3688, 160) or not np.isfinite(source).all():
        raise ValueError('Source dimensions')
    return rows, source


def versions():
    return dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                scipy=scipy.__version__, sklearn=sklearn.__version__)


def lock_content(root, prior, benchmark):
    return dict(code={n: sha(root/n) for n in CODE}, pytest_sha256=sha(root/'pytest.xml'),
                feature_complete_sha256=sha(prior/'features/complete.json'),
                prior_complete_sha256=sha(prior/'results/complete.json'),
                prior_verification_sha256=sha(prior/'local_verification.json'),
                prior_readout_lock_sha256=sha(prior/'readout_lock.json'),
                benchmark=read(prior/'readout_lock.json')['benchmark'],
                source_seed=42, alphas=ALPHAS, ratios=RATIOS, new_conditions=NEW,
                reused_conditions=OLD, execution='local CPU, one thread, CUDA hidden', versions=versions())


def metric(rows):
    y = np.array([float(r['label']) for r in rows]); p = np.array([float(r['prediction']) for r in rows])
    binary = np.array([int(r['binary']) for r in rows])
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Nonfinite prediction')
    return dict(n=len(y), PCC=float(pearsonr(y, p).statistic) if y.std()>0 and p.std()>0 else None,
                Spearman=float(spearmanr(y,p).statistic) if y.std()>0 and p.std()>0 else None,
                MSE=float(np.mean((y-p)**2)), MAE=float(np.mean(abs(y-p))),
                AP=float(average_precision_score(binary,p)) if len(set(binary))==2 else None,
                AUROC=float(roc_auc_score(binary,p)) if len(set(binary))==2 else None,
                F1=float(f1_score(binary,p>=.5,zero_division=0)))


def run(root, prior, benchmark):
    started = time.monotonic()
    lock = read(root/'experiment_lock.json')
    if plain(lock_content(root, prior, benchmark)) != lock:
        raise ValueError('Execution does not match registered lock')
    rows, source = load_inputs(prior, benchmark)
    locksha = sha(root/'experiment_lock.json'); outroot = root/'results'
    if (outroot/'complete.json').exists():
        raise FileExistsError('Results already complete; do not rerun')
    index = {r['sample_id']: i for i,r in enumerate(rows)}
    # TD values are deliberately not parsed until a first-stage decision is saved.
    predictions = []; selections = []; fold_metrics = []; candidates = 0; durations = []
    for setting in SETTINGS:
        splits = sorted((benchmark/'splits'/setting).glob('fold*.json'))
        if len(splits) != (1 if setting=='Inter_Hu_to_Mix' else 5):
            raise ValueError('Fold count')
        for path in splits:
            split = read(path); fold = split['fold']; out = outroot/'folds'/setting/f'fold{fold}'
            if not split['trainable']:
                raise ValueError('Untrainable split')
            for co, digest in split['data_sha256'].items():
                if sha(benchmark/'data'/f'{co}.csv') != digest:
                    raise ValueError('Split dataset SHA')
            fi, vi, ti = [[index[s] for s in split[k]] for k in ('fit_sample_ids','validation_sample_ids','test_sample_ids')]
            if any(len(s)!=len(set(s)) for s in (fi,vi,ti)) or any(set(a)&set(b) for a,b in ((fi,vi),(fi,ti),(vi,ti))):
                raise ValueError('Split overlap/duplicates')
            if (out/'complete.json').exists():
                receipt = read(out/'complete.json')
                if receipt['experiment_lock_sha256'] != locksha:
                    raise ValueError('Resume lock mismatch')
                check_files(out, receipt['files_sha256'])
            else:
                if out.exists() and any(out.iterdir()):
                    raise RuntimeError('Partial fold requires explicit audit; no silent overwrite')
                began = time.monotonic()
                yf = np.array([float(rows[i]['label']) for i in fi]); yv = np.array([float(rows[i]['label']) for i in vi])
                stage1 = select_experts(source[fi], yf, source[vi], yv)
                write(out/'stage1.json', stage1)
                write(out/'selection_lock.json', dict(stage1_sha256=sha(out/'stage1.json'), mask=stage1['mask'],
                      split_sha256=sha(path), experiment_lock_sha256=locksha,
                      fit_ids=split['fit_sample_ids'], validation_ids=split['validation_sample_ids'],
                      test_ids=split['test_sample_ids'], TD24_used=False))
                td = np.array([[float(v) for v in r['td'].split(',')] for r in rows])
                if td.shape!=(3688,24) or not np.isfinite(td).all():
                    raise ValueError('TD24 input')
                x = np.c_[td,source]
                stage2 = {name: fixed_mask_ridge(x[fi],yf,x[vi],yv,stage1['mask'],with_td)
                          for name,with_td in zip(NEW[2:],(False,True))}
                write(out/'stage2.json',stage2)
                # No selection below this line. Test labels enter only these prediction rows.
                predicted = {NEW[0]:predict_source(stage1,stage1['dense'],source[ti]),
                             NEW[1]:predict_source(stage1,stage1['selected'],source[ti])}
                predicted.update({name:predict_fixed(st,x[ti]) for name,st in stage2.items()})
                part=[]
                for name,values in predicted.items():
                    for i,p in zip(ti,values):
                        part.append(dict(condition=name,setting=setting,fold=fold,sample_id=rows[i]['sample_id'],
                                         cohort=rows[i]['cohort'],label=float(rows[i]['label']),binary=int(rows[i]['y']),prediction=float(p)))
                write_csv(out/'predictions.csv',part)
                files = ('stage1.json','selection_lock.json','stage2.json','predictions.csv')
                write(out/'complete.json',dict(experiment_lock_sha256=locksha,candidate_evaluations=40,
                      selected_heads=4,wall_seconds=time.monotonic()-began,
                      files_sha256={n:sha(out/n) for n in files}))
                print(json.dumps(dict(setting=setting,fold=fold,complete=True)),flush=True)
            receipt=read(out/'complete.json'); durations.append(receipt['wall_seconds']);candidates+=40
            part=table(out/'predictions.csv');predictions+=part
            stage1=read(out/'stage1.json');stage2=read(out/'stage2.json')
            for name in NEW:
                if name in NEW[:2]:
                    st=stage1['dense' if name==NEW[0] else 'selected']; mask=[v>0 for v in st['group_norms']]
                else:
                    st=stage2[name]['selected'];mask=stage2[name]['mask']
                selections.append(dict(condition=name,setting=setting,fold=fold,alpha=st['alpha'],
                    ratio=st.get('ratio',''),validation_mse=st['val_mse'],active_groups=sum(mask),
                    **dict(zip(EXPERTS,[int(v) for v in mask]))))
                fold_metrics.append(dict(condition=name,setting=setting,fold=fold,**metric([p for p in part if p['condition']==name])))
    if len(predictions)!=16628 or len(selections)!=104 or candidates!=1040:
        raise ValueError('New totals')
    oldpred=table(prior/'results/oof_predictions.csv')
    if len(oldpred)!=12471 or {r['condition'] for r in oldpred}!=set(OLD):
        raise ValueError('Prior prediction count')
    combined=oldpred+predictions; summary=[]
    for name in OLD+NEW:
        for setting in SETTINGS:
            part=[p for p in combined if p['condition']==name and p['setting']==setting]
            co='Mixset' if setting=='Inter_Hu_to_Mix' else setting
            expected={r['sample_id'] for r in rows if r['cohort']==co}
            if len(part)!=len(expected) or {p['sample_id'] for p in part}!=expected:
                raise ValueError('OOF identity/coverage')
            summary.append(dict(condition=name,setting=setting,origin='reused' if name in OLD else 'new',**metric(part)))
    write_csv(outroot/'new_oof_predictions.csv',predictions)
    write_csv(outroot/'combined_oof_predictions.csv',combined)
    write_csv(outroot/'summary.csv',summary)
    write_csv(outroot/'fold_metrics.csv',fold_metrics)
    write_csv(outroot/'selection.csv',selections)
    write(outroot/'verification.json',dict(passed=True,folds=26,new_selected_heads=104,candidate_evaluations=1040,
        new_OOF_rows=16628,reused_OOF_rows=12471,source_updates=0,new_GPU_hours=0,source_seed=42,
        sum_fold_wall_seconds=sum(durations),controller_wall_seconds=time.monotonic()-started,
        stage1_no_TD=True,stage2_mask_fixed=True,no_prediction_stacking=True))
    # Verify all original inputs/results again before declaring completion.
    load_inputs(prior,benchmark)
    write(outroot/'complete.json',dict(experiment_lock_sha256=locksha,
        files_sha256={p.relative_to(outroot).as_posix():sha(p) for p in sorted(outroot.rglob('*')) if p.is_file()}))
    print('TWO_STAGE_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('register','run'))
    p.add_argument('--root',type=Path,default=HERE)
    p.add_argument('--prior',type=Path,default=HERE.parent/'expert_selection_20261014/results_download/verified')
    p.add_argument('--benchmark',type=Path,default=HERE.parent/'core_six_settings_20261011/benchmark')
    a=p.parse_args();root=a.root.resolve();prior=a.prior.resolve();benchmark=a.benchmark.resolve()
    if a.mode=='register':
        if (root/'experiment_lock.json').exists():
            raise FileExistsError('Already registered')
        import xml.etree.ElementTree as ET
        report=ET.parse(root/'pytest.xml').getroot()
        suites=list(report.iter('testsuite'))
        if sum(int(s.get('tests',0)) for s in suites)!=10 or any(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0)) for s in suites):
            raise ValueError('Synthetic tests not passed')
        load_inputs(prior,benchmark)
        write(root/'experiment_lock.json',lock_content(root,prior,benchmark))
        print('REGISTERED',sha(root/'experiment_lock.json'))
        return
    mutex=root/'execution.mutex'
    with mutex.open('x') as f:
        f.write(str(os.getpid()))
    t=time.monotonic();error=None;rc=1
    try:
        run(root,prior,benchmark);rc=0
    except Exception:
        error=traceback.format_exc();raise
    finally:
        name=f'execution_{time.time_ns()}.json'
        write(root/name,dict(returncode=rc,wall_seconds=time.monotonic()-t,new_GPU_hours=0,
              source_inference_reused=True,device='CPU',threads=1,exception=error,
              experiment_lock_sha256=sha(root/'experiment_lock.json')))
        mutex.unlink()


if __name__=='__main__':
    main()
