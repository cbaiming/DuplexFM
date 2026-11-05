"""Registered four-setting CPU pilot. Every real run is receipted."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,csv,json,sys,time,traceback
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;RESEARCH=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(1,str(RESEARCH/'expert_first_20261014'))
import run_experiment as io
from core import configs,fit_transform,transform,fit_heads,score,choose
OLD=RESEARCH/'expert_banded_20261014';PRIOR=RESEARCH/'expert_selection_20261014/results_download/verified'
BENCH=RESEARCH/'core_six_settings_20261011/benchmark'
SETTINGS=('Huesken','Mixset','Takayuki','Inter_Hu_to_Mix')
METHODS=('PCA4_Full','PCA4_GlobalHard','PCA4_SourceOnly')
CODE=('core.py','run.py','verify.py','test_core.py','PROTOCOL.md','CODE_AUDIT.md')

def npz_write(p,**arrays):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    tmp=p.with_suffix('.tmp')
    with tmp.open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def inputs():
    fc=io.read(PRIOR/'features/complete.json');io.check_files(PRIOR/'features',fc['files_sha256'])
    assert fc['source_seed']==42 and fc['source_audit']['source_frozen']
    rows=[]
    for name,n in [('Huesken',2196),('Mixset',469),('Takayuki',664)]:
        part=io.table(BENCH/'data'/f'{name}.csv');assert len(part)==n and all(r['cohort']==name for r in part)
        rows.extend(part)
    identity=io.table(PRIOR/'features/row_identity.csv');ix={r['sample_id']:i for i,r in enumerate(identity)}
    src=np.load(PRIOR/'features/slots.npy',allow_pickle=False)
    assert src.shape==(3688,160) and len(ix)==3688
    for r in rows:
        assert all(r[k]==identity[ix[r['sample_id']]][k] for k in ('sample_id','cohort','guide','target57'))
    src=src[[ix[r['sample_id']] for r in rows]].astype(np.float64)
    x=np.c_[np.array([[float(z) for z in r['td'].split(',')] for r in rows]),src]
    assert x.shape==(3329,184) and np.isfinite(x).all()
    return rows,x

def register():
    if (HERE/'lock.json').exists():raise FileExistsError('Registered')
    import xml.etree.ElementTree as ET
    suites=list(ET.parse(HERE/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==4
    assert not any(int(s.get('errors',0))+int(s.get('failures',0))+int(s.get('skipped',0)) for s in suites)
    oldaudit=io.read(OLD/'local_verification.json');assert oldaudit['passed']
    assert oldaudit['result_complete_sha256']==io.sha(OLD/'results/complete.json')
    rows,_=inputs();ids={r['sample_id'] for r in rows}
    folds=[f for f in io.read(OLD/'plan.json')['folds'] if f['setting'] in SETTINGS]
    assert len(folds)==16 and all(set(f['fit_ids']+f['test_ids']+f['validation_ids'])<=ids for f in folds)
    io.write(HERE/'plan.json',dict(folds=folds,configs=configs(),excluded=['wang2009','alas1_patent']))
    deps=[OLD/'plan.json',OLD/'results/summary.csv',OLD/'results/oof_predictions.csv',OLD/'results/complete.json',
      OLD/'local_verification.json',PRIOR/'features/complete.json',PRIOR/'features/row_identity.csv',PRIOR/'features/slots.npy',
      RESEARCH/'expert_first_20261014/run_experiment.py',RESEARCH/'expert_first_20261014/two_stage.py']
    deps += [BENCH/'data'/f'{n}.csv' for n in SETTINGS[:3]]
    deps += [BENCH/'splits'/f['setting']/f"fold{f['fold']}.json" for f in folds]
    io.write(HERE/'lock.json',dict(files={n:io.sha(HERE/n) for n in CODE+('plan.json','pytest.xml')},
      dependencies={str(p.resolve()):io.sha(p) for p in deps},versions=io.versions(),source_seed=42,
      CPU_only=True,source_updates=0,GPU_hours=0,mean_relative_PCC_goal_pct=5))
    print('REGISTERED',io.sha(HERE/'lock.json'),flush=True)

def validate():
    lock=io.read(HERE/'lock.json');io.check_files(HERE,lock['files'])
    for name,digest in lock['dependencies'].items():assert io.sha(Path(name))==digest,name
    assert lock['versions']==io.versions()

def run():
    validate();rows,x=inputs();plan=io.read(HERE/'plan.json');cfg=plan['configs'];root=HERE/'results'
    if (root/'complete.json').exists():raise FileExistsError('Complete')
    y=np.array([float(r['label']) for r in rows]);index={r['sample_id']:i for i,r in enumerate(rows)}
    locksha=io.sha(HERE/'lock.json');allpred=[];selections=[]
    for item in plan['folds']:
        setting,fold=item['setting'],item['fold'];out=root/'folds'/setting/f'fold{fold}'
        fi=np.array([index[s] for s in item['fit_ids']]);ti=np.array([index[s] for s in item['test_ids']])
        if (out/'complete.json').exists():
            receipt=io.read(out/'complete.json');assert receipt['lock_sha256']==locksha;io.check_files(out,receipt['files_sha256'])
        else:
            if out.exists() and any(out.iterdir()):raise RuntimeError('Partial requires explicit audit')
            start=time.monotonic();oof=np.full((len(fi),40),np.nan);trans=[];coefs=[];intercepts=[]
            for sp in item['inner']:
                tr=np.array(sp['train']);va=np.array(sp['valid'])
                z,t=fit_transform(x[fi[tr]]);b,c=fit_heads(z,y[fi[tr]],cfg)
                oof[va]=transform(x[fi[va]],t)@b.T+c
                trans.append(t);coefs.append(b);intercepts.append(c)
            assert np.isfinite(oof).all()
            pc,ms=score(y[fi],oof)
            selected=dict(zip(METHODS,[choose(pc,ms,cfg,range(5)),choose(pc,ms,cfg,range(35)),choose(pc,ms,cfg,range(35,40))]))
            io.write(out/'selection.json',dict(selected=selected,PCC=pc,MSE=ms,fit_ids=item['fit_ids'],test_ids=item['test_ids'],
                outer_validation_unused=True,lock_sha256=locksha))
            z,t=fit_transform(x[fi]);final_cfg=[cfg[selected[m]] for m in METHODS];b,c=fit_heads(z,y[fi],final_cfg)
            pred=transform(x[ti],t)@b.T+c
            npz_write(out/'arrays.npz',inner_coef=np.array(coefs),inner_intercept=np.array(intercepts),inner_oof=oof,
                final_coef=b,final_intercept=np.array(c),test_pred=pred)
            io.write(out/'transforms.json',dict(inner=trans,final=t))
            part=[]
            for k,m in enumerate(METHODS):
                for pos,i in enumerate(ti):
                    part.append(dict(setting=setting,fold=fold,condition=m,sample_id=rows[i]['sample_id'],
                        label=y[i],binary=int(rows[i]['y']),prediction=pred[pos,k]))
            io.write_csv(out/'predictions.csv',part)
            io.write(out/'complete.json',dict(lock_sha256=locksha,wall_seconds=time.monotonic()-start,
                inner_fits=200,final_heads=3,GPU_hours=0,source_updates=0,
                files_sha256={n:io.sha(out/n) for n in ('selection.json','transforms.json','arrays.npz','predictions.csv')}))
            print(json.dumps(dict(setting=setting,fold=fold,complete=True)),flush=True)
        allpred+=io.table(out/'predictions.csv');sel=io.read(out/'selection.json')
        for name,k in sel['selected'].items():
            selections.append(dict(setting=setting,fold=fold,condition=name,candidate=k,**cfg[k],inner_PCC=sel['PCC'][k]))
    assert len(allpred)==11394 and len(selections)==48
    summary=[]
    baseline={(r['setting'],r['condition']):r for r in io.table(OLD/'results/summary.csv') if r['setting'] in SETTINGS}
    for s in SETTINGS:
        expected={r['sample_id'] for r in rows if r['cohort']==('Mixset' if s=='Inter_Hu_to_Mix' else s)}
        for m in METHODS:
            part=[r for r in allpred if r['setting']==s and r['condition']==m]
            assert len(part)==len(expected) and {r['sample_id'] for r in part}==expected
            metric=io.metric(part);base=baseline[s,'TD24_Ridge_CV']
            summary.append(dict(setting=s,condition=m,**metric,TD24_PCC=float(base['PCC']),
                PCC_delta=metric['PCC']-float(base['PCC']),PCC_relative_gain_pct=100*(metric['PCC']/float(base['PCC'])-1),
                MSE_relative_reduction_pct=100*(1-metric['MSE']/float(base['MSE']))))
    goal=[]
    for m in METHODS:
        part=[r for r in summary if r['condition']==m];g=float(np.mean([r['PCC_relative_gain_pct'] for r in part]))
        goal.append(dict(condition=m,mean_relative_PCC_gain_pct=g,reaches_development_5pct=g>=5,
            negative_settings=[r['setting'] for r in part if r['PCC_delta']<0],source_only_diagnostic=m==METHODS[2]))
    io.write_csv(root/'summary.csv',summary);io.write_csv(root/'oof_predictions.csv',allpred);io.write_csv(root/'selection.csv',selections)
    io.write(root/'goal_summary.json',goal)
    validate()
    io.write(root/'complete.json',dict(lock_sha256=locksha,outer_folds=16,inner_folds=80,inner_fits=3200,final_heads=48,
        rows=11394,GPU_hours=0,source_updates=0,excluded=['wang2009','alas1_patent'],
        files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    print('ROUND1_COMPLETE',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('register','run'));mode=p.parse_args().mode
    if mode=='register':register();return
    mutex=HERE/'execution.mutex'
    with mutex.open('x') as f:f.write(str(os.getpid()))
    start=time.monotonic();rc=1;error=None
    try:run();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        io.write(HERE/f'execution_{time.time_ns()}.json',dict(returncode=rc,error=error,wall_seconds=time.monotonic()-start,
            CPU_threads=1,GPU_hours=0,source_updates=0,lock_sha256=io.sha(HERE/'lock.json')))
        mutex.unlink()

if __name__=='__main__':main()
