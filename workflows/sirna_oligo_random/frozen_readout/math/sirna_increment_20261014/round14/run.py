"""Fixed same-readout test of one pre-existing Core+Access source."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;PRIOR=HERE.parent/'round4';sys.path.insert(0,str(HERE))
from assets import inputs as asset_inputs,old
io=old.io;OLD=old.OLD;SETTINGS=old.SETTINGS
configs,fit_transform,transform,fit_heads,score,choose=(old.configs,old.fit_transform,old.transform,old.fit_heads,old.score,old.choose)
npz_write=old.npz_write
SOURCES=('core_access_real',);MODES=('Uniform40','Adaptive40')
CODE=('run.py','verify.py','assets.py','test_assets.py','build_io.py','build_io_v1_failed.py','BUILD_REPAIR.md','ASSET_PROTOCOL.md','PROTOCOL.md','CODE_AUDIT.md','pytest.xml')

def inputs():
    rows,xx,identity,_=asset_inputs();return rows,xx,{'core_access_real':identity}

def register():
    assert not (HERE/'lock.json').exists()
    tests=list(ET.parse(HERE/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(t.get('tests',0)) for t in tests)==4
    assert not any(int(t.get('errors',0))+int(t.get('failures',0))+int(t.get('skipped',0)) for t in tests)
    asset=io.read(HERE/'asset_verification.json');assert asset['passed'] and asset['rows']==3329
    assert asset['code_sha256']==io.sha(HERE/'assets.py') and asset['protocol_sha256']==io.sha(HERE/'ASSET_PROTOCOL.md')
    for p,h in asset['files_sha256'].items():assert io.sha(Path(p))==h,p
    rows,xx,identity,newdeps=asset_inputs();assert identity==asset['source_identity']
    old.validate();ref=io.read(PRIOR/'lock.json');deps=dict(ref['dependencies']);deps.update(newdeps)
    assert io.read(PRIOR/'verification.json')['passed']
    assert io.read(PRIOR/'verification.json')['complete_sha256']==io.sha(PRIOR/'results/complete.json')
    for n in ref['files']:deps[str((PRIOR/n).resolve())]=io.sha(PRIOR/n)
    for p in (PRIOR/'lock.json',PRIOR/'verification.json',PRIOR/'results/complete.json',HERE/'asset_verification.json',HERE.parent/'round13/closure_receipt.json'):
        deps[str(p.resolve())]=io.sha(p)
    for n,h in io.read(PRIOR/'results/complete.json')['files_sha256'].items():
        p=PRIOR/'results'/n;assert io.sha(p)==h;deps[str(p.resolve())]=h
    plan=io.read(PRIOR/'plan.json')
    io.write(HERE/'plan.json',dict(folds=plan['folds'],configs=configs(),sources=SOURCES,modes=MODES,
        source_identity={'core_access_real':identity},primary_condition='core_access_real_Adaptive40',
        excluded=['Wang','ALAS1'],CPU_phase_limit_seconds=1800,reference_R4_complete_sha256=io.sha(PRIOR/'results/complete.json')))
    io.write(HERE/'lock.json',dict(files={n:io.sha(HERE/n) for n in CODE+('plan.json',)},
        dependencies=deps,versions=io.versions(),CPU_only=True,GPU_hours=0,source_updates=0,new_source_forward=0,
        mean_relative_PCC_goal_pct=5,CPU_phase_limit_seconds=1800))
    print('REGISTERED',io.sha(HERE/'lock.json'),flush=True)

def validate():
    lock=io.read(HERE/'lock.json');io.check_files(HERE,lock['files'])
    for p,h in lock['dependencies'].items():assert io.sha(Path(p))==h,p
    assert lock['versions']==io.versions()

def run():
    phase_start=time.monotonic()
    validate();rows,xx,_=inputs();root=HERE/'results';locksha=io.sha(HERE/'lock.json')
    if (root/'complete.json').exists():raise FileExistsError('Complete')
    plan=io.read(HERE/'plan.json');cfg=plan['configs'];y=np.array([float(r['label']) for r in rows]);idx={r['sample_id']:i for i,r in enumerate(rows)}
    predictions=[];selections=[]
    for item in plan['folds']:
        setting,fold=item['setting'],item['fold'];fi=np.array([idx[s] for s in item['fit_ids']]);ti=np.array([idx[s] for s in item['test_ids']])
        for source in SOURCES:
            if time.monotonic()-phase_start>1800:raise TimeoutError('CPU phase cap; no new block')
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
    assert len(predictions)==7596 and len(selections)==32
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
    io.write(root/'complete.json',dict(lock_sha256=locksha,outer_folds=16,sources=1,inner_projection_blocks=80,final_projection_blocks=16,
        inner_fits=2000,final_heads=32,rows=7596,GPU_hours=0,source_updates=0,new_source_forward=0,
        files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    print('ROUND14_COMPLETE',flush=True)

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
