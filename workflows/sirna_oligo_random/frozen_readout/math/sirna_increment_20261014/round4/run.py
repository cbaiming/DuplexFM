"""Registered CPU-only single-source comparison; no encoder or network calls."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;RESEARCH=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(1,str(RESEARCH/'expert_first_20261014'))
import run_experiment as io
from core import configs,fit_transform,transform,fit_heads,score,choose
PRIOR=HERE.parent/'round1';OLD=RESEARCH/'expert_banded_20261014'
SOURCE=RESEARCH/'core_six_settings_20261011/results'
BENCH=RESEARCH/'core_six_settings_20261011/benchmark'
PACKAGE=RESEARCH/'mechanism_transfer_20261010/results/h100_source_20261011/extracted'
SETTINGS=('Huesken','Mixset','Takayuki','Inter_Hu_to_Mix')
SOURCES=('core_real','full_real','initial_core');MODES=('Uniform40','Adaptive40')
CODE=('core.py','run.py','verify.py','test_core.py','PROTOCOL.md','CODE_AUDIT.md')

def npz_write(p,**arrays):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    tmp=p.with_suffix('.tmp')
    with tmp.open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def inputs():
    audit=io.read(SOURCE/'local_verification.json');assert audit['status']=='passed'
    assert audit['bundle_manifest_sha256']==io.sha(SOURCE/'bundle_manifest.json')
    manifest=io.read(SOURCE/'bundle_manifest.json')['files_sha256']
    rows=[]
    for name,n in zip(SETTINGS[:3],(2196,469,664)):
        p=BENCH/'data'/f'{name}.csv';assert io.sha(p)==audit['data_sha256'][name]
        part=io.table(p);assert len(part)==n;rows+=part
    td=np.array([[float(v) for v in r['td'].split(',')] for r in rows]);assert td.shape==(3329,24)
    xx={};identities={}
    for name in SOURCES:
        folder=SOURCE/'exports'/name;c=io.read(folder/'complete.json');assert c['status']=='complete'
        key=f'exports/{name}/complete.json';assert io.sha(folder/'complete.json')==manifest[key]
        io.check_files(folder,c['files_sha256']);identity=c['source_identity']
        assert identity['name']==name and identity['source_seed']==42
        path=PACKAGE/'runs'/identity['variant']/identity['checkpoint'];assert io.sha(path)==identity['checkpoint_sha256']
        assert io.sha(PACKAGE/'runs'/identity['variant']/'initial.pt')==identity['initial_sha256']
        meta=io.table(folder/'row_identity.csv');index={r['sample_id']:i for i,r in enumerate(meta)}
        assert len(index)==3688 and c['source_shape']==[3688,192]
        for r in rows:assert all(r[k]==meta[index[r['sample_id']]][k] for k in ('sample_id','cohort','guide','target57'))
        source=np.load(folder/'source.npy',allow_pickle=False)[[index[r['sample_id']] for r in rows]].astype(float)
        assert source.shape==(3329,192) and np.isfinite(source).all()
        xx[name]=np.c_[td,source];identities[name]=identity
    assert identities['initial_core']['checkpoint_sha256']==identities['core_real']['initial_sha256']
    return rows,xx,identities

def register():
    if (HERE/'lock.json').exists():raise FileExistsError('Registered')
    suites=list(ET.parse(HERE/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==5
    assert not any(int(s.get('errors',0))+int(s.get('failures',0))+int(s.get('skipped',0)) for s in suites)
    old=io.read(PRIOR/'lock.json');io.check_files(PRIOR,old['files'])
    for p,h in old['dependencies'].items():assert io.sha(Path(p))==h
    assert io.read(PRIOR/'verification.json')['passed']
    rows,_,identities=inputs();folds=io.read(PRIOR/'plan.json')['folds'];assert len(folds)==16
    io.write(HERE/'plan.json',dict(folds=folds,configs=configs(),sources=SOURCES,modes=MODES,source_identity=identities,
        primary_condition='core_real_Adaptive40',excluded=['Wang','ALAS1']))
    deps=[Path(p) for p in old['dependencies']]+[PRIOR/'lock.json',PRIOR/'plan.json',PRIOR/'verification.json',
        SOURCE/'bundle_manifest.json',SOURCE/'local_verification.json']
    for name in SOURCES:
        folder=SOURCE/'exports'/name;deps +=[folder/'complete.json']+[folder/p for p in io.read(folder/'complete.json')['files_sha256']]
        s=identities[name];deps +=[PACKAGE/'runs'/s['variant']/s['checkpoint'],PACKAGE/'runs'/s['variant']/'initial.pt']
    deps += [OLD/'results/folds'/f['setting']/f"fold{f['fold']}"/'arrays.npz' for f in folds]
    deps += [OLD/'results/folds'/f['setting']/f"fold{f['fold']}"/'selection.json' for f in folds]
    io.write(HERE/'lock.json',dict(files={n:io.sha(HERE/n) for n in CODE+('plan.json','pytest.xml')},
        dependencies={str(p.resolve()):io.sha(p) for p in deps},versions=io.versions(),CPU_only=True,GPU_hours=0,
        source_updates=0,new_source_forward=0,mean_relative_PCC_goal_pct=5))
    print('REGISTERED',io.sha(HERE/'lock.json'),flush=True)

def validate():
    lock=io.read(HERE/'lock.json');io.check_files(HERE,lock['files'])
    for p,h in lock['dependencies'].items():assert io.sha(Path(p))==h,p
    assert lock['versions']==io.versions()

def run():
    validate();rows,xx,_=inputs();root=HERE/'results';locksha=io.sha(HERE/'lock.json')
    if (root/'complete.json').exists():raise FileExistsError('Complete')
    plan=io.read(HERE/'plan.json');cfg=plan['configs'];y=np.array([float(r['label']) for r in rows]);idx={r['sample_id']:i for i,r in enumerate(rows)}
    predictions=[];selections=[]
    for item in plan['folds']:
        setting,fold=item['setting'],item['fold'];fi=np.array([idx[s] for s in item['fit_ids']]);ti=np.array([idx[s] for s in item['test_ids']])
        for source in SOURCES:
            x=xx[source];out=root/'folds'/setting/f'fold{fold}'/source
            if (out/'complete.json').exists():
                c=io.read(out/'complete.json');assert c['lock_sha256']==locksha;io.check_files(out,c['files_sha256'])
            else:
                if out.exists() and any(out.iterdir()):raise RuntimeError('Partial block requires explicit audit')
                oof=np.full((len(fi),25),np.nan);tt=[];bb=[];cc=[]
                for split in item['inner']:
                    tr=np.array(split['train']);va=np.array(split['valid'])
                    z,t=fit_transform(x[fi[tr]]);b,c=fit_heads(z,y[fi[tr]],cfg)
                    oof[va]=transform(x[fi[va]],t)@b.T+c;tt.append(t);bb.append(b);cc.append(c)
                assert np.isfinite(oof).all()
                with np.load(OLD/'results/folds'/setting/f'fold{fold}'/'arrays.npz',allow_pickle=False) as old:
                    np.testing.assert_allclose(oof[:,:5],old['inner_oof'][:,5:10],atol=1e-10,rtol=1e-10)
                pc,ms=score(y[fi],oof);selected=dict(zip(MODES,[choose(pc,ms,cfg,range(15,20)),choose(pc,ms,cfg,range(25))]))
                io.write(out/'selection.json',dict(selected=selected,PCC=pc,MSE=ms,fit_ids=item['fit_ids'],test_ids=item['test_ids'],
                    source=source,outer_validation_unused=True,TD24_inner_predictions_match_original=True))
                z,t=fit_transform(x[fi]);b,c=fit_heads(z,y[fi],[cfg[selected[n]] for n in MODES]);p=transform(x[ti],t)@b.T+c
                io.write(out/'transforms.json',dict(inner=tt,final=t))
                npz_write(out/'arrays.npz',inner_coef=np.array(bb),inner_intercept=np.array(cc),inner_oof=oof,
                    final_coef=b,final_intercept=np.array(c),test_pred=p)
                part=[dict(setting=setting,fold=fold,source=source,condition=source+'_'+m,sample_id=rows[i]['sample_id'],
                    label=rows[i]['label'],binary=rows[i]['y'],prediction=float(p[j,k])) for k,m in enumerate(MODES) for j,i in enumerate(ti)]
                io.write_csv(out/'predictions.csv',part)
                io.write(out/'complete.json',dict(lock_sha256=locksha,files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
            predictions+=io.table(out/'predictions.csv');sel=io.read(out/'selection.json')
            for mode,k in sel['selected'].items():selections.append(dict(setting=setting,fold=fold,condition=source+'_'+mode,
                source=source,candidate=k,**cfg[k],inner_PCC=sel['PCC'][k]))
        print('CLOSED',setting,fold,flush=True)
    assert len(predictions)==22788 and len(selections)==96
    baseline={(r['setting'],r['condition']):r for r in io.table(OLD/'results/summary.csv')};summary=[]
    for setting in SETTINGS:
        expected={r['sample_id'] for r in rows if r['cohort']==('Mixset' if setting=='Inter_Hu_to_Mix' else setting)}
        for source in SOURCES:
            for mode in MODES:
                condition=source+'_'+mode;part=[r for r in predictions if r['setting']==setting and r['condition']==condition]
                assert len(part)==len(expected) and {r['sample_id'] for r in part}==expected
                metric=io.metric(part);base=baseline[setting,'TD24_Ridge_CV'];bp=float(base['PCC'])
                summary.append(dict(setting=setting,source=source,condition=condition,**metric,TD24_PCC=bp,
                    PCC_delta=metric['PCC']-bp,PCC_relative_gain_pct=100*(metric['PCC']/bp-1),
                    MSE_relative_reduction_pct=100*(1-metric['MSE']/float(base['MSE']))))
    goals=[]
    for source in SOURCES:
        for mode in MODES:
            condition=source+'_'+mode;part=[r for r in summary if r['condition']==condition];g=float(np.mean([r['PCC_relative_gain_pct'] for r in part]))
            goals.append(dict(condition=condition,mean_relative_PCC_gain_pct=g,reaches_development_5pct=g>=5,
                primary=condition==plan['primary_condition'],initial_control=source=='initial_core',negative_settings=[r['setting'] for r in part if r['PCC_delta']<0]))
    io.write_csv(root/'summary.csv',summary);io.write_csv(root/'oof_predictions.csv',predictions);io.write_csv(root/'selection.csv',selections)
    io.write(root/'goal_summary.json',goals);validate()
    io.write(root/'complete.json',dict(lock_sha256=locksha,outer_folds=16,sources=3,inner_projection_blocks=240,
        inner_fits=6000,final_heads=96,rows=22788,GPU_hours=0,source_updates=0,new_source_forward=0,
        files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    print('ROUND4_COMPLETE',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('register','run'));mode=p.parse_args().mode
    if mode=='register':register();return
    mutex=HERE/'execution.mutex'
    with mutex.open('x') as f:f.write(str(os.getpid()))
    started=time.monotonic();rc=1;error=None
    try:run();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        io.write(HERE/f'execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-started,
            GPU_hours=0,source_updates=0,new_source_forward=0,lock_sha256=io.sha(HERE/'lock.json')))
        mutex.unlink()
if __name__=='__main__':main()
