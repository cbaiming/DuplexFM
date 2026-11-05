"""Locked CPU run. No original CLI, GPU, remote access or source inference."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
HERE=Path(__file__).resolve().parent;RESEARCH=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(1,str(RESEARCH/'expert_first_20261014'))
import numpy as np
import torch
import run_experiment as io
from native import P12,P13,FILES,model,normalize,predict
from train import fit,select,ENDPOINTS,bytes_torch
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
PRIOR=HERE.parent/'round1';OLD=RESEARCH/'expert_banded_20261014'
SOURCE=RESEARCH/'expert_selection_20261014/results_download/verified/features'
BENCH=RESEARCH/'core_six_settings_20261011/benchmark'
SETTINGS=('Huesken','Mixset','Takayuki','Inter_Hu_to_Mix');METHODS=('P12_native_head','P13_native_head')
CONDITIONS=dict(zip(METHODS,(P12,P13)))
CODE=('native.py','train.py','run.py','verify.py','test_native.py','PROTOCOL.md','CODE_AUDIT.md')

def inputs():
    fc=io.read(SOURCE/'complete.json');io.check_files(SOURCE,fc['files_sha256'])
    assert fc['source_seed']==42 and fc['source_audit']['source_frozen']
    rows=[]
    for name,n in zip(SETTINGS[:3],(2196,469,664)):
        part=io.table(BENCH/'data'/f'{name}.csv');assert len(part)==n;rows+=part
    identity=io.table(SOURCE/'row_identity.csv');index={r['sample_id']:i for i,r in enumerate(identity)}
    s=np.load(SOURCE/'slots.npy',allow_pickle=False)
    for r in rows:assert all(r[k]==identity[index[r['sample_id']]][k] for k in ('sample_id','cohort','guide','target57'))
    s=s[[index[r['sample_id']] for r in rows]]
    td=np.array([[float(v) for v in r['td'].split(',')] for r in rows])
    x=np.c_[s,td].astype(np.float32);assert x.shape==(3329,184) and np.isfinite(x).all()
    return rows,x

def register():
    assert not (HERE/'lock.json').exists()
    suites=list(ET.parse(HERE/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==5
    assert not any(int(s.get('errors',0))+int(s.get('failures',0))+int(s.get('skipped',0)) for s in suites)
    previous=io.read(PRIOR/'lock.json');io.check_files(PRIOR,previous['files'])
    for p,h in previous['dependencies'].items():assert io.sha(Path(p))==h
    assert io.read(PRIOR/'verification.json')['passed']
    folds=io.read(PRIOR/'plan.json')['folds'];assert len(folds)==16
    rows,_=inputs();assert len(rows)==3329
    plan=dict(folds=folds,conditions=CONDITIONS,endpoints=ENDPOINTS,source_seed=42,head_seed=42,excluded=['Wang','ALAS1'])
    io.write(HERE/'plan.json',plan)
    deps=[Path(p) for p in previous['dependencies']]+[PRIOR/'lock.json',PRIOR/'plan.json',PRIOR/'verification.json',*FILES]
    io.write(HERE/'lock.json',dict(files={n:io.sha(HERE/n) for n in CODE+('plan.json','pytest.xml')},
        dependencies={str(p.resolve()):io.sha(p) for p in deps},versions={**io.versions(),'torch':torch.__version__},
        CPU_only=True,source_updates=0,GPU_hours=0,primary='P13_native_head_vs_fixed_TD24_Ridge_CV',mean_relative_PCC_goal_pct=5))
    print('REGISTERED',io.sha(HERE/'lock.json'),flush=True)

def validate():
    lock=io.read(HERE/'lock.json');io.check_files(HERE,lock['files'])
    for p,h in lock['dependencies'].items():assert io.sha(Path(p))==h,p
    assert lock['versions']=={**io.versions(),'torch':torch.__version__}

def fit_block(path,x,y,condition,epochs,locksha,probe=None):
    if (path/'complete.json').exists():
        c=io.read(path/'complete.json');assert c['lock_sha256']==locksha
        io.check_files(path,c['files_sha256']);return torch.load(path/'result.pt',map_location='cpu',weights_only=False)
    if path.exists() and any(path.iterdir()):raise RuntimeError('Partial block requires explicit audit: '+str(path))
    def checkpoint(name,value):io.atomic(path/name,bytes_torch(value))
    result=fit(x,y,condition,epochs,probe=probe,checkpoint=checkpoint)
    io.atomic(path/'result.pt',bytes_torch(result))
    io.write(path/'complete.json',dict(lock_sha256=locksha,epochs=epochs,condition=condition,updates=result['updates'],
        initial_sha256=result['initial_sha256'],final_sha256=result['final_sha256'],wall_seconds=result['wall_seconds'],
        trainable_parameters=result['trainable_parameters'],files_sha256={p.name:io.sha(p) for p in path.iterdir() if p.is_file()}))
    return result

def run():
    validate();root=HERE/'results'
    if (root/'complete.json').exists():raise FileExistsError('Already complete')
    rows,x=inputs();ids={r['sample_id']:i for i,r in enumerate(rows)};y=np.array([float(r['label']) for r in rows])
    plan=io.read(HERE/'plan.json');locksha=io.sha(HERE/'lock.json');predictions=[];selections=[]
    for item in plan['folds']:
        setting,fold=item['setting'],item['fold'];out=root/'folds'/setting/f'fold{fold}'
        fi=np.array([ids[s] for s in item['fit_ids']]);ti=np.array([ids[s] for s in item['test_ids']])
        for method,condition in CONDITIONS.items():
            path=out/method
            if (path/'complete.json').exists():
                c=io.read(path/'complete.json');assert c['lock_sha256']==locksha;io.check_files(path,c['files_sha256'])
            else:
                oof=np.full((len(fi),5),np.nan)
                for j,sp in enumerate(item['inner']):
                    tr=np.array(sp['train']);va=np.array(sp['valid'])
                    r=fit_block(path/f'inner{j}',x[fi[tr]],y[fi[tr]],condition,100,locksha,probe=x[fi[va]])
                    oof[va]=r['predictions']
                assert np.isfinite(oof).all()
                chosen,pcc,mse=select(y[fi],oof);ep=ENDPOINTS[chosen]
                io.write(path/'selection.json',dict(condition=method,selected=chosen,epoch=ep,inner_PCC=pcc,inner_MSE=mse,
                    fit_ids=item['fit_ids'],test_ids=item['test_ids'],outer_validation_unused=True))
                r=fit_block(path/'final',x[fi],y[fi],condition,ep,locksha)
                m=model(condition,42);m.load_state_dict(r['state'],strict=True)
                pp=predict(m,normalize(x[ti],r['normalizer']))
                part=[dict(setting=setting,fold=fold,condition=method,sample_id=rows[i]['sample_id'],label=rows[i]['label'],
                    binary=rows[i]['y'],prediction=float(p)) for i,p in zip(ti,pp)]
                io.write_csv(path/'predictions.csv',part)
                io.write(path/'complete.json',dict(lock_sha256=locksha,
                    files_sha256={p.relative_to(path).as_posix():io.sha(p) for p in sorted(path.rglob('*')) if p.is_file()}))
            predictions+=io.table(path/'predictions.csv');sel=io.read(path/'selection.json')
            selections.append(dict(setting=setting,fold=fold,condition=method,epoch=sel['epoch'],inner_PCC=sel['inner_PCC'][sel['selected']]))
            print('CLOSED',setting,fold,method,flush=True)
    assert len(predictions)==7596 and len(selections)==32
    base={(r['setting'],r['condition']):r for r in io.table(OLD/'results/summary.csv')}
    summary=[]
    for s in SETTINGS:
        expected={r['sample_id'] for r in rows if r['cohort']==('Mixset' if s=='Inter_Hu_to_Mix' else s)}
        for method in METHODS:
            part=[r for r in predictions if r['setting']==s and r['condition']==method]
            assert len(part)==len(expected) and {r['sample_id'] for r in part}==expected
            metric=io.metric(part);baseline=base[s,'TD24_Ridge_CV'];bp=float(baseline['PCC'])
            summary.append(dict(setting=s,condition=method,**metric,TD24_PCC=bp,PCC_delta=metric['PCC']-bp,
                PCC_relative_gain_pct=100*(metric['PCC']/bp-1),MSE_relative_reduction_pct=100*(1-metric['MSE']/float(baseline['MSE']))))
    goal=[]
    for method in METHODS:
        part=[r for r in summary if r['condition']==method];g=float(np.mean([r['PCC_relative_gain_pct'] for r in part]))
        goal.append(dict(condition=method,mean_relative_PCC_gain_pct=g,reaches_development_5pct=g>=5,
            negative_settings=[r['setting'] for r in part if r['PCC_delta']<0],TD_only_control=method==METHODS[0]))
    io.write_csv(root/'oof_predictions.csv',predictions);io.write_csv(root/'selection.csv',selections);io.write_csv(root/'summary.csv',summary)
    io.write(root/'goal_summary.json',goal);validate()
    io.write(root/'complete.json',dict(lock_sha256=locksha,outer_folds=16,inner_training_runs=160,inner_endpoint_states=800,
        final_heads=32,OOF_rows=7596,CPU_only=True,GPU_hours=0,source_updates=0,
        files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    print('ROUND3_COMPLETE',flush=True)

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
            GPU_hours=0,source_updates=0,lock_sha256=io.sha(HERE/'lock.json')))
        mutex.unlink()
if __name__=='__main__':main()
