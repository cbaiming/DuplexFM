"""Locked CPU-only ARD comparison, with original TD24 and PCA artifacts reused."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import sklearn.linear_model._bayes as native
R=Path(__file__).resolve().parent;P=R.parent/'round4'
sys.path.insert(0,str(R))
from sparse import PARAMETERS,fit,predict,choose,collapse
spec=importlib.util.spec_from_file_location('_r11_prior_io',P/'run.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
io=previous.io
VIEWS=('core_real','full_real','initial_core','TD_only');MODES=('AlwaysARD','AdaptiveARD')
SETTINGS=previous.SETTINGS
CODE=('run.py','sparse.py','verify.py','test_sparse.py','PROTOCOL.md','CODE_AUDIT.md','SYNTHETIC_REPAIR.md')

def condition(view,mode):return view+'_'+mode

def register():
    assert not (R/'lock.json').exists()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==7
    assert not any(int(s.get('errors',0))+int(s.get('failures',0))+int(s.get('skipped',0)) for s in suites)
    previous.validate();old=io.read(P/'lock.json');prior=io.read(P/'results/complete.json')
    io.check_files(P/'results',prior['files_sha256']);assert io.read(P/'verification.json')['passed']
    assert io.read(R.parent/'round10/verification.json')['passed']
    rows,_,identities=previous.inputs();plan=io.read(P/'plan.json')
    io.write(R/'plan.json',dict(folds=plan['folds'],views=VIEWS,modes=MODES,source_identity=identities,
        parameters=PARAMETERS,primary_condition='core_real_AdaptiveARD',excluded=['Wang','ALAS1'],
        CPU_phase_limit_seconds=1800,candidates=['TD_Ridge_0.1','TD_Ridge_1','TD_Ridge_10','TD_Ridge_100','TD_Ridge_1000','native_ARD']))
    deps=dict(old['dependencies'])
    for name in old['files']:deps[str((P/name).resolve())]=io.sha(P/name)
    for name,h in prior['files_sha256'].items():deps[str((P/'results'/name).resolve())]=h
    for p in (P/'lock.json',P/'verification.json',P/'results/complete.json',Path(native.__file__),
              R.parent/'round10/verification.json',R.parent/'round10/diagnostic.json'):
        deps[str(p.resolve())]=io.sha(p)
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in CODE+('plan.json','pytest.xml','pytest_v1_failed.xml','test_sparse_v1.py')},
        dependencies=deps,versions=io.versions(),native_ARD_sha256=io.sha(Path(native.__file__)),
        CPU_only=True,GPU_hours=0,source_updates=0,new_source_forward=0,goal_pct=5))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)

def validate():
    lock=io.read(R/'lock.json');io.check_files(R,lock['files'])
    for p,h in lock['dependencies'].items():assert io.sha(Path(p))==h,p
    assert lock['versions']==io.versions() and os.environ['CUDA_VISIBLE_DEVICES']==''

def run(start):
    validate();rows,xx,_=previous.inputs();plan=io.read(R/'plan.json');root=R/'results'
    assert not (root/'complete.json').exists()
    ids={r['sample_id']:i for i,r in enumerate(rows)};y=np.array([float(r['label']) for r in rows])
    predictions=[];selections=[];fit_audit=[];locksha=io.sha(R/'lock.json')
    for item in plan['folds']:
        setting,fold=item['setting'],item['fold'];fi=np.array([ids[s] for s in item['fit_ids']]);ti=np.array([ids[s] for s in item['test_ids']])
        oldpath=previous.OLD/'results/folds'/setting/f'fold{fold}'
        oldselection=io.read(oldpath/'selection.json');base_idx=oldselection['selected']['TD24_Ridge_CV']-5
        with np.load(oldpath/'arrays.npz',allow_pickle=False) as old:
            oldinner=old['inner_oof'][:,5:10].copy();oldtest=old['test_pred'][:,1].copy()
        for view in VIEWS:
            if time.monotonic()-start>=plan['CPU_phase_limit_seconds']:raise TimeoutError('No new block after CPU cap')
            out=root/'folds'/setting/f'fold{fold}'/view
            if (out/'complete.json').exists():
                saved=io.read(out/'complete.json');assert saved['lock_sha256']==locksha;io.check_files(out,saved['files_sha256'])
            else:
                if out.exists() and any(out.iterdir()):raise RuntimeError('Partial block requires explicit audit')
                source='core_real' if view=='TD_only' else view;width=24 if view=='TD_only' else 40;x=xx[source]
                transform_path=P/'results/folds'/setting/f'fold{fold}'/source/'transforms.json'
                transforms=io.read(transform_path);oof=np.c_[oldinner,np.full(len(fi),np.nan)];models=[]
                for j,part in enumerate(item['inner']):
                    tr=np.array(part['train']);va=np.array(part['valid']);t=transforms['inner'][j]
                    z=previous.transform(x[fi[tr]],t)[:,:width];zv=previous.transform(x[fi[va]],t)[:,:width]
                    model=fit(z,y[fi[tr]]);models.append(model);oof[va,5]=predict(zv,model)
                assert np.isfinite(oof).all();pc,ms=previous.score(y[fi],oof)
                assert choose(pc,ms,range(5))==base_idx
                selected={'AlwaysARD':5,'AdaptiveARD':choose(pc,ms)}
                if selected['AdaptiveARD']!=5:assert selected['AdaptiveARD']==base_idx
                t=transforms['final'];z=previous.transform(x[fi],t)[:,:width];zt=previous.transform(x[ti],t)[:,:width]
                final=fit(z,y[fi]);pp=predict(zt,final);beta,intercept=collapse(t,final)
                np.testing.assert_allclose(x[ti]@beta+intercept,pp,atol=1e-10,rtol=1e-10)
                p=np.c_[pp,pp if selected['AdaptiveARD']==5 else oldtest]
                io.write(out/'models.json',dict(inner=models,final=final,transform_path=str(transform_path.resolve()),
                    transform_sha256=io.sha(transform_path),source=source,input_width=width))
                io.write(out/'selection.json',dict(selected=selected,PCC=pc,MSE=ms,fit_ids=item['fit_ids'],test_ids=item['test_ids'],
                    view=view,outer_validation_unused=True,TD_original_candidate=base_idx,feature_width=width))
                previous.npz_write(out/'arrays.npz',inner_oof=oof,test_pred=p,collapsed_coef=beta,collapsed_intercept=np.array(intercept))
                part=[dict(setting=setting,fold=fold,view=view,condition=condition(view,mode),sample_id=rows[i]['sample_id'],
                    label=rows[i]['label'],binary=rows[i]['y'],prediction=float(p[j,k])) for k,mode in enumerate(MODES) for j,i in enumerate(ti)]
                io.write_csv(out/'predictions.csv',part)
                io.write(out/'complete.json',dict(lock_sha256=locksha,files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
            predictions+=io.table(out/'predictions.csv');sel=io.read(out/'selection.json');models=io.read(out/'models.json')
            for mode,k in sel['selected'].items():selections.append(dict(setting=setting,fold=fold,view=view,condition=condition(view,mode),
                candidate=k,uses_ARD=k==5,inner_PCC=sel['PCC'][k]))
            for j,m in enumerate(models['inner']+[models['final']]):fit_audit.append(dict(setting=setting,fold=fold,view=view,
                partition='final' if j==5 else f'inner{j}',iterations=m['iterations'],cap_reached=m['iteration_cap_reached'],
                active_TD=m['active_TD'],active_source=m['active_source'],warning_count=len(m['warnings'])))
        print('CLOSED',setting,fold,flush=True)
    assert len(predictions)==30384 and len(selections)==128 and len(fit_audit)==384
    baseline={(r['setting'],r['condition']):r for r in io.table(previous.OLD/'results/summary.csv')};summary=[]
    for setting in SETTINGS:
        expected={r['sample_id'] for r in rows if r['cohort']==('Mixset' if setting=='Inter_Hu_to_Mix' else setting)}
        for view in VIEWS:
            for mode in MODES:
                name=condition(view,mode);part=[r for r in predictions if r['setting']==setting and r['condition']==name]
                assert len(part)==len(expected) and {r['sample_id'] for r in part}==expected
                m=io.metric(part);base=baseline[setting,'TD24_Ridge_CV'];bp=float(base['PCC'])
                summary.append(dict(setting=setting,view=view,condition=name,**m,TD24_PCC=bp,
                    PCC_delta=m['PCC']-bp,PCC_relative_gain_pct=100*(m['PCC']/bp-1),
                    MSE_relative_reduction_pct=100*(1-m['MSE']/float(base['MSE']))))
    goals=[]
    for view in VIEWS:
        for mode in MODES:
            name=condition(view,mode);part=[r for r in summary if r['condition']==name]
            gain=float(np.mean([r['PCC_relative_gain_pct'] for r in part]))
            goals.append(dict(condition=name,mean_relative_PCC_gain_pct=gain,reaches_development_5pct=gain>=5,
                primary=name==plan['primary_condition'],negative_settings=[r['setting'] for r in part if r['PCC_delta']<0],
                source_free=view=='TD_only',initial_control=view=='initial_core'))
    io.write_csv(root/'summary.csv',summary);io.write_csv(root/'oof_predictions.csv',predictions)
    io.write_csv(root/'selection.csv',selections);io.write_csv(root/'fit_audit.csv',fit_audit);io.write(root/'goal_summary.json',goals)
    validate()
    io.write(root/'complete.json',dict(lock_sha256=locksha,outer_folds=16,views=4,sources=3,inner_fits=320,final_fits=64,
        output_conditions=8,rows=30384,selections=128,iteration_capped_fits=sum(r['cap_reached'] for r in fit_audit),
        GPU_hours=0,source_updates=0,new_source_forward=0,new_PCA_fits=0,
        files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    print('R11_COMPLETE',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('register','run'));args=ap.parse_args()
    if args.mode=='register':register();return
    mutex=R/'execution.mutex'
    with mutex.open('x') as f:f.write(str(os.getpid()))
    start=time.monotonic();rc=1;error=None
    try:run(start);rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        io.write(R/f'execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,
            GPU_hours=0,source_updates=0,new_source_forward=0,lock_sha256=io.sha(R/'lock.json')))
        mutex.unlink()

if __name__=='__main__':main()
