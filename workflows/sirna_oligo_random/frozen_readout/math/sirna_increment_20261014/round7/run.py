"""Bounded CPU experiment with frozen inputs and a single conditional readout."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;PRIOR=HERE.parent/'round4'
sys.path.insert(0,str(HERE))
from kernel import configs,fit_transform,transform,fit_heads,score,choose
spec=importlib.util.spec_from_file_location('_r4_io',PRIOR/'run.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
io=previous.io
SETTINGS=previous.SETTINGS
VIEWS=('core_real','full_real','initial_core','TDKernel')
MODES=('Uniform','Adaptive')
CODE=('run.py','kernel.py','verify.py','audit_kernel.py','test_kernel.py','test_execution.py','build_io.py','calibrate_cpu.py','PROTOCOL.md','CODE_AUDIT.md','CPU_QUALIFICATION.json','execution_pytest.xml')

def condition(view,mode):return view+'_'+mode+'Kernel'
def kind(view):return 'td' if view=='TDKernel' else 'joint'

def register():
    if (HERE/'lock.json').exists():raise FileExistsError('Already registered')
    suites=list(ET.parse(HERE/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==6
    assert not any(int(s.get('errors',0))+int(s.get('failures',0))+int(s.get('skipped',0)) for s in suites)
    extra=list(ET.parse(HERE/'execution_pytest.xml').getroot().iter('testsuite'))
    assert sum(int(x.get('tests',0)) for x in extra)==3
    assert not any(int(x.get('errors',0))+int(x.get('failures',0))+int(x.get('skipped',0)) for x in extra)
    qualification=io.read(HERE/'CPU_QUALIFICATION.json')
    assert qualification['synthetic_only'] and qualification['GPU_hours']==0 and qualification['landmarks']==64
    assert qualification['kernel_code_sha256']==io.sha(HERE/'kernel.py')
    previous.validate();v=io.read(PRIOR/'verification.json')
    assert v['passed'] and v['complete_sha256']==io.sha(PRIOR/'results/complete.json')
    io.check_files(PRIOR/'results',io.read(PRIOR/'results/complete.json')['files_sha256'])
    old=io.read(PRIOR/'lock.json');plan=io.read(PRIOR/'plan.json')
    io.write(HERE/'plan.json',dict(folds=plan['folds'],configs=configs(),views=VIEWS,modes=MODES,
        source_identity=plan['source_identity'],primary_condition='core_real_AdaptiveKernel',
        excluded=['Wang','ALAS1'],CPU_phase_limit_seconds=1800))
    dependencies=dict(old['dependencies'])
    for name in old['files']:dependencies[str((PRIOR/name).resolve())]=io.sha(PRIOR/name)
    for p in (PRIOR/'lock.json',PRIOR/'verification.json',PRIOR/'results/complete.json',
              HERE.parent/'round6/lock.json',HERE.parent/'round6/verification.json',HERE.parent/'round6/results/complete.json'):
        dependencies[str(p.resolve())]=io.sha(p)
    for name,h in io.read(PRIOR/'results/complete.json')['files_sha256'].items():dependencies[str((PRIOR/'results'/name).resolve())]=h
    io.write(HERE/'lock.json',dict(files={n:io.sha(HERE/n) for n in CODE+('plan.json','pytest.xml')},
        dependencies=dependencies,versions=io.versions(),CPU_only=True,GPU_hours=0,new_source_forward=0,
        mean_relative_PCC_goal_pct=5,CPU_phase_limit_seconds=1800))
    print('REGISTERED',io.sha(HERE/'lock.json'),flush=True)

def validate():
    lock=io.read(HERE/'lock.json');io.check_files(HERE,lock['files'])
    for p,h in lock['dependencies'].items():assert io.sha(Path(p))==h,p
    assert lock['versions']==io.versions()

def same_projection(t,ref):
    for key in ref:np.testing.assert_allclose(t[key],ref[key],atol=1e-12,rtol=1e-12)

def run(start):
    validate();rows,xx,_=previous.inputs();xx['TDKernel']=xx['core_real'][:,:24].copy()
    root=HERE/'results';locksha=io.sha(HERE/'lock.json');plan=io.read(HERE/'plan.json');cfg=plan['configs']
    if (root/'complete.json').exists():raise FileExistsError('Already complete')
    y=np.array([float(r['label']) for r in rows]);ids={r['sample_id']:i for i,r in enumerate(rows)}
    predictions=[];selections=[]
    for item in plan['folds']:
        s,fold=item['setting'],item['fold'];fi=np.array([ids[a] for a in item['fit_ids']]);ti=np.array([ids[a] for a in item['test_ids']])
        for view in VIEWS:
            if time.monotonic()-start>=plan['CPU_phase_limit_seconds']:raise TimeoutError('CPU phase cap; no new block started')
            out=root/'folds'/s/f'fold{fold}'/view;x=xx[view]
            if (out/'complete.json').exists():
                c=io.read(out/'complete.json');assert c['lock_sha256']==locksha;io.check_files(out,c['files_sha256'])
            else:
                if out.exists() and any(out.iterdir()):raise RuntimeError('Partial block requires explicit audit')
                ref=io.read(PRIOR/'results/folds'/s/f'fold{fold}'/('core_real' if view=='TDKernel' else view)/'transforms.json')
                oof=np.full((len(fi),25),np.nan);tt=[];bb=[];cc=[]
                for j,sp in enumerate(item['inner']):
                    tr=np.array(sp['train']);va=np.array(sp['valid']);z,t=fit_transform(x[fi[tr]],kind(view))
                    if view!='TDKernel':same_projection(t['projection'],ref['inner'][j])
                    b,c=fit_heads(z,y[fi[tr]],cfg);oof[va]=transform(x[fi[va]],t)@b.T+c
                    tt.append(t);bb.append(b);cc.append(c)
                assert np.isfinite(oof).all()
                with np.load(previous.OLD/'results/folds'/s/f'fold{fold}'/'arrays.npz',allow_pickle=False) as old:
                    np.testing.assert_allclose(oof[:,:5],old['inner_oof'][:,5:10],atol=1e-10,rtol=1e-10)
                pc,ms=score(y[fi],oof);selected=dict(zip(MODES,[choose(pc,ms,cfg,range(15,20)),choose(pc,ms,cfg,range(25))]))
                z,t=fit_transform(x[fi],kind(view))
                if view!='TDKernel':same_projection(t['projection'],ref['final'])
                b,c=fit_heads(z,y[fi],[cfg[selected[m]] for m in MODES]);p=transform(x[ti],t)@b.T+c
                io.write(out/'selection.json',dict(selected=selected,PCC=pc,MSE=ms,fit_ids=item['fit_ids'],test_ids=item['test_ids'],
                    view=view,outer_validation_unused=True,TD24_inner_predictions_match_original=True,feature_width=z.shape[1]))
                io.write(out/'transforms.json',dict(inner=tt,final=t))
                previous.npz_write(out/'arrays.npz',inner_coef=np.array(bb),inner_intercept=np.array(cc),inner_oof=oof,
                    final_coef=b,final_intercept=np.array(c),test_pred=p)
                part=[dict(setting=s,fold=fold,view=view,condition=condition(view,m),sample_id=rows[i]['sample_id'],
                    label=rows[i]['label'],binary=rows[i]['y'],prediction=float(p[j,k])) for k,m in enumerate(MODES) for j,i in enumerate(ti)]
                io.write_csv(out/'predictions.csv',part)
                io.write(out/'complete.json',dict(lock_sha256=locksha,files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
            predictions+=io.table(out/'predictions.csv');sel=io.read(out/'selection.json')
            for mode,k in sel['selected'].items():selections.append(dict(setting=s,fold=fold,view=view,condition=condition(view,mode),
                candidate=k,**cfg[k],inner_PCC=sel['PCC'][k]))
        print('CLOSED',s,fold,flush=True)
    assert len(predictions)==30384 and len(selections)==128
    baseline={(r['setting'],r['condition']):r for r in io.table(previous.OLD/'results/summary.csv')};summary=[]
    for s in SETTINGS:
        expected={r['sample_id'] for r in rows if r['cohort']==('Mixset' if s=='Inter_Hu_to_Mix' else s)}
        for view in VIEWS:
            for mode in MODES:
                cond=condition(view,mode);part=[r for r in predictions if r['setting']==s and r['condition']==cond]
                assert len(part)==len(expected) and {r['sample_id'] for r in part}==expected
                metric=io.metric(part);base=baseline[s,'TD24_Ridge_CV'];bp=float(base['PCC'])
                summary.append(dict(setting=s,view=view,condition=cond,**metric,TD24_PCC=bp,
                    PCC_delta=metric['PCC']-bp,PCC_relative_gain_pct=100*(metric['PCC']/bp-1),
                    MSE_relative_reduction_pct=100*(1-metric['MSE']/float(base['MSE']))))
    goals=[]
    for view in VIEWS:
        for mode in MODES:
            cond=condition(view,mode);part=[r for r in summary if r['condition']==cond];gain=float(np.mean([r['PCC_relative_gain_pct'] for r in part]))
            goals.append(dict(condition=cond,mean_relative_PCC_gain_pct=gain,reaches_development_5pct=gain>=5,
                primary=cond==plan['primary_condition'],no_source_control=view=='TDKernel',initial_control=view=='initial_core',
                negative_settings=[r['setting'] for r in part if r['PCC_delta']<0]))
    io.write_csv(root/'summary.csv',summary);io.write_csv(root/'oof_predictions.csv',predictions);io.write_csv(root/'selection.csv',selections)
    io.write(root/'goal_summary.json',goals);validate()
    io.write(root/'complete.json',dict(lock_sha256=locksha,outer_folds=16,source_models=3,views=4,inner_projection_blocks=320,
        inner_fits=8000,final_heads=128,rows=30384,GPU_hours=0,source_updates=0,new_source_forward=0,
        files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    print('ROUND7_COMPLETE',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('register','run'));mode=parser.parse_args().mode
    if mode=='register':register();return
    mutex=HERE/'execution.mutex'
    with mutex.open('x') as f:f.write(str(os.getpid()))
    start=time.monotonic();rc=1;error=None
    try:run(start);rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        io.write(HERE/f'execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,
            GPU_hours=0,source_updates=0,new_source_forward=0,lock_sha256=io.sha(HERE/'lock.json')))
        mutex.unlink()
if __name__=='__main__':main()
