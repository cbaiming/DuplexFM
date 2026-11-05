"""Fixed four-setting evaluation after native trace pilot qualification."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;P=R.parent/'round4';Q=R.parent/'round25';D=R.parent/'round22/results_v2'
sys.path.insert(0,str(R));import head
sp=importlib.util.spec_from_file_location('_r26_old4',P/'run.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
io=old.io;SOURCES=old.SOURCES;SETTINGS=old.SETTINGS;ALPHAS=head.ALPHAS
FILES=('run.py','head.py','verify.py','test_head.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml')


def register():
    assert not (R/'lock.json').exists();old.validate()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==10
    assert not any(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0)) for s in suites)
    qualified=io.read(Q/'verification.json');assert qualified['passed'] and qualified['qualified']
    for parent in (Q,R.parent/'round22'):
        c=io.read(parent/'closure_receipt.json')
        for name,h in c['files_sha256'].items():assert io.sha(R.parent/name)==h,name
    qlock=io.read(Q/'lock.json');deps=dict(qlock['dependencies'])
    for name in qlock['files']:deps[str((Q/name).resolve())]=io.sha(Q/name)
    for p in (Q/'lock.json',Q/'closure_receipt.json',Q/'verification.json',Q/'diagnostic.json',D/'complete.json',
              R.parent/'round22/closure_receipt.json',R.parent/'round22/verification.json',old.OLD/'results/summary.csv'):
        deps[str(p.resolve())]=io.sha(p)
    qdiag=io.read(Q/'diagnostic.json')
    for name,h in qdiag['pilot_files_sha256'].items():
        p=Q/'pilot'/name;assert io.sha(p)==h;deps[str(p.resolve())]=h
    dcomplete=io.read(D/'complete.json');io.check_files(D,dcomplete['files_sha256'])
    for name in ('summary.csv','oof_predictions.csv','selection.csv','goal_summary.json'):
        p=D/name;deps[str(p.resolve())]=io.sha(p)
    baseplan=io.read(P/'plan.json')
    for item in baseplan['folds']:
        for source in SOURCES:
            for name in ('transforms.json','arrays.npz'):
                p=P/'results/folds'/item['setting']/f"fold{item['fold']}"/source/name;deps[str(p.resolve())]=io.sha(p)
            for name in ('arrays.npz','selection.json','predictions.csv','references.json','complete.json'):
                p=D/'folds'/item['setting']/f"fold{item['fold']}"/source/name;deps[str(p.resolve())]=io.sha(p)
    io.write(R/'plan.json',dict(folds=baseplan['folds'],sources=SOURCES,alphas=ALPHAS,primary='core_real_native_trace',
        lambda_ratio=.1,matrix_shape=[6,32],excluded=['Wang','ALAS1'],source_seed=42,
        reuse='R25 source-matched outer fold0, Huesken/Mixset/Takayuki, alpha100 inner heads only'))
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES+('plan.json',)},dependencies=deps,versions=io.versions(),
        CPU_only=True,CPU_limit_seconds=1800,total_inner_heads=1200,new_inner_heads=1155,reused_inner_heads=45,new_final_heads=48,
        new_scientific_heads=1203,new_OOF_rows=11394,source_forward=0,new_PCA=0,primary='core_real_native_trace',
        baseline='TD24_Ridge_CV',goal_mean_relative_PCC_gain_pct=5.))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)


def validate():
    lock=io.read(R/'lock.json');io.check_files(R,lock['files'])
    for name,h in lock['dependencies'].items():assert io.sha(Path(name))==h,name
    assert lock['versions']==io.versions();return lock


def run(start):
    validate();root=R/'results';assert not root.exists();plan=io.read(R/'plan.json');threads=threadpool_info();assert all(t['num_threads']==1 for t in threads)
    rows,xx,identities=old.inputs();idx={r['sample_id']:i for i,r in enumerate(rows)};locksha=io.sha(R/'lock.json')
    allpred=[];selections=[];nnew=nreuse=nfinal=0
    for item in plan['folds']:
        setting,fold=item['setting'],item['fold'];fi=np.array([idx[s] for s in item['fit_ids']]);ti=np.array([idx[s] for s in item['test_ids']])
        assert set(fi).isdisjoint(ti);yf=np.array([float(rows[i]['label']) for i in fi])
        for source in SOURCES:
            if time.monotonic()-start>1800:raise TimeoutError('No new block: CPU cap')
            x=xx[source];out=root/'folds'/setting/f'fold{fold}'/source;ref=P/'results/folds'/setting/f'fold{fold}'/source;ts=io.read(ref/'transforms.json')
            oof=np.full((len(fi),5),np.nan);bc=np.empty((5,5,216));ic=np.empty((5,5));reused=[]
            with np.load(ref/'arrays.npz',allow_pickle=False) as a:tdcoef=a['inner_coef'][:,:5,:24].copy();tdcenter=a['inner_intercept'].copy();tdoof=a['inner_oof'][:,:5].copy()
            for j,split in enumerate(item['inner']):
                tr,va=np.array(split['train']),np.array(split['valid']);z=head.transform(x[fi[tr]],ts['inner'][j]);zv=head.transform(x[fi[va]],ts['inner'][j])
                np.testing.assert_allclose(zv[:,:24]@tdcoef[j].T+tdcenter[j],tdoof[va],atol=1e-10,rtol=1e-10)
                for k,alpha in enumerate(ALPHAS):
                    if time.monotonic()-start>1800:raise TimeoutError('No new head: CPU cap')
                    if fold==0 and setting in SETTINGS[:3] and alpha==100.:
                        cached=Q/'pilot'/setting/source/f'inner{j}';state=io.read(cached/'state.json');cc=io.read(cached/'complete.json');io.check_files(cached,cc['files_sha256'])
                        assert cc['lock_sha256']==io.sha(Q/'lock.json') and state['train_indices']==tr.tolist() and state['valid_indices']==va.tolist() and state['source']==source
                        cert=state['certificate']
                        with np.load(cached/'model.npz',allow_pickle=False) as a:b=a['beta'].copy();c=float(a['intercept']);pp=a['prediction'].copy()
                        np.testing.assert_allclose(zv@b+c,pp,atol=1e-10,rtol=1e-10)
                        reused.append(dict(inner=j,candidate=k,alpha=alpha,origin=str(cached.resolve()),
                            model_sha256=io.sha(cached/'model.npz'),state_sha256=io.sha(cached/'state.json'),complete_sha256=io.sha(cached/'complete.json')));nreuse+=1
                    else:
                        b,c,cert=head.fit(z,yf[tr],alpha,deadline=start+1800);pp=zv@b+c;nnew+=1
                    io.write(out/f'inner{j}_alpha{k}_certificate.json',cert)
                    if not cert['certified'] or cert['gap']>1e-9*cert['gap_scale']:raise ArithmeticError(f'Unqualified {setting}/{fold}/{source}/{j}/{alpha}')
                    bc[j,k]=b;ic[j,k]=c;oof[va,k]=pp
            assert np.isfinite(oof).all();pc,ms=old.score(yf,oof);chosen=head.choose(pc,ms)
            io.write(out/'selection.json',dict(selected=chosen,alpha=ALPHAS[chosen],PCC=pc,MSE=ms,
                fit_ids=item['fit_ids'],test_ids=item['test_ids'],source=source,outer_validation_unused=True))
            io.write(out/'reuse.json',reused)
            selections.append(dict(setting=setting,fold=fold,source=source,condition=source+'_native_trace',candidate=chosen,alpha=ALPHAS[chosen],inner_PCC=float(pc[chosen])))
            z=head.transform(x[fi],ts['final']);zt=head.transform(x[ti],ts['final'])
            b,c,cert=head.fit(z,yf,ALPHAS[chosen],deadline=start+1800);nfinal+=1;io.write(out/'final_certificate.json',cert)
            if not cert['certified'] or cert['gap']>1e-9*cert['gap_scale']:raise ArithmeticError('Final unqualified')
            coef,offset=head.deployment(ts['final'],b,c);pred=zt@b+c
            old.npz_write(out/'arrays.npz',inner_coef=bc,inner_intercept=ic,inner_oof=oof,final_coef=b,final_intercept=np.array(c),
                test_pred=pred,deployment_coef=coef,deployment_intercept=np.array(offset))
            io.write(out/'references.json',dict(transform_sha256=io.sha(ref/'transforms.json'),source_identity=identities[source]))
            part=[dict(setting=setting,fold=fold,source=source,condition=source+'_native_trace',sample_id=rows[i]['sample_id'],label=rows[i]['label'],binary=rows[i]['y'],prediction=float(pred[j])) for j,i in enumerate(ti)]
            io.write_csv(out/'predictions.csv',part);allpred+=part
            io.write(out/'complete.json',dict(lock_sha256=locksha,total_inner_heads=25,new_inner_heads=25-len(reused),reused_inner_heads=len(reused),final_heads=1,
                files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
        print('CLOSED',setting,fold,'new_heads',nnew+nfinal,'reused',nreuse,flush=True)
    assert nnew==1155 and nreuse==45 and nfinal==48 and len(allpred)==11394
    baseline={(r['setting'],r['condition']):r for r in io.table(old.OLD/'results/summary.csv')}
    dense={(r['setting'],r['condition']):r for r in io.table(D/'summary.csv')};summary=[];goals=[];paired=[]
    for source in SOURCES:
        cond=source+'_native_trace'
        for setting in SETTINGS:
            part=[r for r in allpred if r['condition']==cond and r['setting']==setting];cohort='Mixset' if setting=='Inter_Hu_to_Mix' else setting
            expected={r['sample_id'] for r in rows if r['cohort']==cohort};assert len(part)==len(expected) and {r['sample_id'] for r in part}==expected
            m=io.metric(part);base=baseline[setting,'TD24_Ridge_CV'];bp=float(base['PCC'])
            summary.append(dict(setting=setting,source=source,condition=cond,**m,TD24_PCC=bp,PCC_delta=m['PCC']-bp,
                PCC_relative_gain_pct=100*(m['PCC']/bp-1),MSE_relative_reduction_pct=100*(1-m['MSE']/float(base['MSE']))))
            dd=dense[setting,source+'_dense'];paired.append(dict(setting=setting,source=source,new_PCC=m['PCC'],dense_PCC=float(dd['PCC']),
                delta_PCC=m['PCC']-float(dd['PCC']),MSE_reduction_pct=100*(1-m['MSE']/float(dd['MSE']))))
        part=[r for r in summary if r['condition']==cond];gain=float(np.mean([r['PCC_relative_gain_pct'] for r in part]))
        goals.append(dict(condition=cond,mean_relative_PCC_gain_pct=gain,primary=cond==plan['primary'],reaches_development_5pct=gain>=5,
            negative_settings=[r['setting'] for r in part if r['PCC_delta']<0]))
    io.write_csv(root/'summary.csv',summary);io.write_csv(root/'oof_predictions.csv',allpred);io.write_csv(root/'selection.csv',selections)
    io.write_csv(root/'paired_dense.csv',paired);io.write(root/'goal_summary.json',goals);validate()
    io.write(root/'complete.json',dict(lock_sha256=locksha,total_inner_heads=nnew+nreuse,new_inner_heads=nnew,reused_inner_heads=nreuse,new_final_heads=nfinal,
        new_scientific_heads=nnew+nfinal,rows=len(allpred),outer_folds=16,source_forward=0,new_PCA=0,GPU_hours=0,threadpool=threads,
        files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    print('COMPLETE',goals,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('register','run'));mode=ap.parse_args().mode
    if mode=='register':register()
    else:
        mutex=R/'execution.mutex'
        with mutex.open('x') as f:f.write(str(os.getpid()))
        start=time.monotonic();rc=1;error=None
        try:run(start);rc=0
        except Exception:error=traceback.format_exc();raise
        finally:
            io.write(R/f'execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,source_forward=0))
            mutex.unlink()
