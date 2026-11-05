"""Consensus selection of existing candidates; at most changed final heads are fit."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round5';sys.path.insert(0,str(R))
from stability import select
spec=importlib.util.spec_from_file_location('_fixed_r5',P/'run.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
io=old.io;OLD=old.previous.OLD;VIEWS=old.VIEWS;MODES=old.MODES;SETTINGS=old.SETTINGS
CODE=('run.py','verify.py','stability.py','test_stability.py','PROTOCOL.md','CODE_AUDIT.md','DIAGNOSTIC_PROTOCOL.md','diagnose.py','diagnostic.json')
def condition(view,mode):return old.condition(view,mode)+'_Consensus'

def register():
    if (R/'lock.json').exists():raise FileExistsError('Registered')
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==5
    assert not any(int(s.get('errors',0))+int(s.get('failures',0))+int(s.get('skipped',0)) for s in suites)
    old.validate();v=io.read(P/'verification.json');assert v['passed'] and v['complete_sha256']==io.sha(P/'results/complete.json')
    c=io.read(P/'results/complete.json');io.check_files(P/'results',c['files_sha256'])
    compat=io.read(P/'artifact_metric_execution_lock.json');io.check_files(P,compat['files'])
    assert compat['original_lock_sha256']==io.sha(P/'lock.json') and compat['original_result_sha256']==io.sha(P/'results/complete.json')
    previous=io.read(P/'lock.json');plan=io.read(P/'plan.json')
    io.write(R/'plan.json',dict(folds=plan['folds'],configs=plan['configs'],views=VIEWS,modes=MODES,
        source_identity=plan['source_identity'],primary_condition=condition('core_real','Adaptive'),
        excluded=['Wang','ALAS1'],CPU_phase_limit_seconds=1800,matched_TD_condition='TD24_Consensus'))
    deps=dict(previous['dependencies'])
    for n,h in previous['files'].items():deps[str((P/n).resolve())]=h
    for n,h in c['files_sha256'].items():deps[str((P/'results'/n).resolve())]=h
    for n,h in compat['files'].items():deps[str((P/n).resolve())]=h
    for p in (P/'lock.json',P/'results/complete.json',P/'verification.json',P/'artifact_metric_execution_lock.json',P/'artifact_metric_execution_receipt.json'):
        deps[str(p.resolve())]=io.sha(p)
    for item in plan['folds']:
        p=OLD/'results/folds'/item['setting']/f"fold{item['fold']}"/'predictions.csv';deps[str(p.resolve())]=io.sha(p)
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in CODE+('plan.json','pytest.xml')},dependencies=deps,
        versions=io.versions(),CPU_only=True,GPU_hours=0,new_inner_fits=0,new_source_forward=0,mean_relative_PCC_goal_pct=5))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)

def validate():
    lock=io.read(R/'lock.json');io.check_files(R,lock['files'])
    for p,h in lock['dependencies'].items():assert io.sha(Path(p))==h,p
    assert lock['versions']==io.versions()

def close_block(folder,locksha):
    io.write(folder/'complete.json',dict(lock_sha256=locksha,files_sha256={p.name:io.sha(p) for p in folder.iterdir() if p.is_file()}))

def run(start):
    validate();rows,xx,_=old.previous.inputs();xx['TDQuadratic']=xx['core_real'][:,:24].copy()
    plan=io.read(R/'plan.json');cfg=plan['configs'];locksha=io.sha(R/'lock.json');root=R/'results'
    if (root/'complete.json').exists():raise FileExistsError('Complete')
    ids={r['sample_id']:i for i,r in enumerate(rows)};y=np.array([float(r['label']) for r in rows])
    predictions=[];selection=[];new_heads=0;reused_heads=0
    for item in plan['folds']:
        setting,fold=item['setting'],item['fold'];fi=np.array([ids[i] for i in item['fit_ids']]);ti=np.array([ids[i] for i in item['test_ids']]);blocks=[s['valid'] for s in item['inner']]
        for view in VIEWS:
            if time.monotonic()-start>=1800:raise TimeoutError('CPU phase cap at new block')
            parent=P/'results/folds'/setting/f'fold{fold}'/view;out=root/'folds'/setting/f'fold{fold}'/view
            if (out/'complete.json').exists():
                c=io.read(out/'complete.json');assert c['lock_sha256']==locksha;io.check_files(out,c['files_sha256'])
            else:
                if out.exists() and any(out.iterdir()):raise RuntimeError('Partial block needs explicit audit')
                with np.load(parent/'arrays.npz',allow_pickle=False) as a:
                    inner=a['inner_oof'].copy();oldcoef=a['final_coef'].copy();oldintercept=float(a['final_intercept']);oldpred=a['test_pred'].copy()
                parent_sel=io.read(parent/'selection.json');t=io.read(parent/'transforms.json')['final']
                diagnostics={m:select(y[fi],inner,cfg,blocks,range(15,20) if m=='Uniform' else range(25)) for m in MODES}
                coef=[];pred=[];origin={};z=zt=None
                for mode in MODES:
                    d=diagnostics[mode];assert d['original']==parent_sel['selected'][mode]
                    matches=[m for m in MODES if parent_sel['selected'][m]==d['selected']]
                    if matches:
                        m=matches[0];j=MODES.index(m);b=oldcoef[j].copy();p=oldpred[:,j].copy()
                        origin[mode]=dict(kind='reuse_R5_final',mode=m,parent_array_sha256=io.sha(parent/'arrays.npz'))
                    else:
                        if z is None:z=old.transform(xx[view][fi],t);zt=old.transform(xx[view][ti],t)
                        bb,c=old.fit_heads(z,y[fi],[cfg[d['selected']]]);assert abs(c-oldintercept)<1e-14
                        b=bb[0];p=zt@b+c;origin[mode]=dict(kind='new_R6_final',parent_transform_sha256=io.sha(parent/'transforms.json'))
                    coef.append(b);pred.append(p)
                pp=np.array(pred).T
                old.previous.npz_write(out/'arrays.npz',final_coef=np.array(coef),final_intercept=np.array(oldintercept),test_pred=pp)
                io.write(out/'selection.json',dict(diagnostics=diagnostics,origin=origin,fit_ids=item['fit_ids'],test_ids=item['test_ids'],
                    parent_relative=str(parent.relative_to(R.parent)),parent_arrays_sha256=io.sha(parent/'arrays.npz'),
                    parent_transforms_sha256=io.sha(parent/'transforms.json'),outer_validation_unused=True))
                part=[dict(setting=setting,fold=fold,view=view,condition=condition(view,mode),sample_id=rows[i]['sample_id'],
                    label=rows[i]['label'],binary=rows[i]['y'],prediction=float(pp[j,k])) for k,mode in enumerate(MODES) for j,i in enumerate(ti)]
                io.write_csv(out/'predictions.csv',part);close_block(out,locksha)
            meta=io.read(out/'selection.json');predictions+=io.table(out/'predictions.csv')
            for mode in MODES:
                d=meta['diagnostics'][mode];k=d['selected'];new=meta['origin'][mode]['kind']=='new_R6_final'
                new_heads+=int(new);reused_heads+=int(not new)
                selection.append(dict(setting=setting,fold=fold,view=view,condition=condition(view,mode),candidate=k,
                    original_candidate=d['original'],**cfg[k],original_support=d['original_support'],selected_support=d['votes'][k],new_fit=new))
        # Apply the identical consensus rule to TD24 and verify original baseline identity.
        out=root/'folds'/setting/f'fold{fold}'/'TD24_Consensus';parent=OLD/'results/folds'/setting/f'fold{fold}'
        if (out/'complete.json').exists():
            c=io.read(out/'complete.json');assert c['lock_sha256']==locksha;io.check_files(out,c['files_sha256'])
        else:
            if out.exists() and any(out.iterdir()):raise RuntimeError('Partial TD control')
            with np.load(P/'results/folds'/setting/f'fold{fold}'/'core_real/arrays.npz',allow_pickle=False) as a:d=select(y[fi],a['inner_oof'],cfg,blocks,range(5))
            oldselection=io.read(parent/'selection.json');original=oldselection['selected']['TD24_Ridge_CV']-5
            assert d['selected']==d['original']==original,'TD control differs: stop, do not alter baseline'
            source=[r for r in io.table(parent/'predictions.csv') if r['condition']=='TD24_Ridge_CV'];assert [r['sample_id'] for r in source]==item['test_ids']
            part=[dict(setting=setting,fold=fold,view='TD24',condition='TD24_Consensus',sample_id=r['sample_id'],
                label=r['label'],binary=r['binary'],prediction=float(r['prediction'])) for r in source]
            io.write(out/'selection.json',dict(diagnostics=d,parent_prediction_sha256=io.sha(parent/'predictions.csv'),
                original_candidate=original,fit_ids=item['fit_ids'],test_ids=item['test_ids'],exact_original_baseline=True))
            io.write_csv(out/'predictions.csv',part);close_block(out,locksha)
        predictions+=io.table(out/'predictions.csv');d=io.read(out/'selection.json')['diagnostics'];k=d['selected']
        selection.append(dict(setting=setting,fold=fold,view='TD24',condition='TD24_Consensus',candidate=k,
            original_candidate=d['original'],**cfg[k],original_support=d['original_support'],selected_support=d['votes'][k],new_fit=False))
        print('CLOSED',setting,fold,flush=True)
    assert new_heads+reused_heads==128 and len(selection)==144 and len(predictions)==34182
    baseline={(r['setting'],r['condition']):r for r in io.table(OLD/'results/summary.csv')};summary=[]
    conditions=[condition(v,m) for v in VIEWS for m in MODES]+['TD24_Consensus']
    for setting in SETTINGS:
        expected={r['sample_id'] for r in rows if r['cohort']==('Mixset' if setting=='Inter_Hu_to_Mix' else setting)}
        for cond in conditions:
            part=[r for r in predictions if r['setting']==setting and r['condition']==cond]
            assert len(part)==len(expected) and {r['sample_id'] for r in part}==expected
            metrics=io.metric(part);b=baseline[setting,'TD24_Ridge_CV'];bp=float(b['PCC'])
            if cond=='TD24_Consensus':
                for key,value in metrics.items():np.testing.assert_allclose(value,float(b[key]),atol=1e-12,rtol=1e-12)
            summary.append(dict(setting=setting,condition=cond,**metrics,TD24_PCC=bp,PCC_delta=metrics['PCC']-bp,
                PCC_relative_gain_pct=100*(metrics['PCC']/bp-1),MSE_relative_reduction_pct=100*(1-metrics['MSE']/float(b['MSE']))))
    goals=[]
    for cond in conditions:
        part=[r for r in summary if r['condition']==cond];gain=float(np.mean([r['PCC_relative_gain_pct'] for r in part]))
        goals.append(dict(condition=cond,mean_relative_PCC_gain_pct=gain,reaches_development_5pct=gain>=5,
            primary=cond==plan['primary_condition'],negative_settings=[r['setting'] for r in part if r['PCC_delta']<-1e-12]))
    io.write_csv(root/'summary.csv',summary);io.write_csv(root/'oof_predictions.csv',predictions);io.write_csv(root/'selection.csv',selection)
    io.write(root/'goal_summary.json',goals);validate()
    io.write(root/'complete.json',dict(lock_sha256=locksha,outer_folds=16,new_inner_fits=0,reused_R5_inner_candidates=8000,
        new_final_heads=new_heads,reused_R5_final_heads=reused_heads,reused_matched_TD_heads=16,total_final_conditions=144,
        rows=34182,GPU_hours=0,new_source_forward=0,source_updates=0,
        files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    print('ROUND6_COMPLETE new_final_heads',new_heads,'reused_R5_heads',reused_heads,flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('register','run'));mode=parser.parse_args().mode
    if mode=='register':register();return
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
