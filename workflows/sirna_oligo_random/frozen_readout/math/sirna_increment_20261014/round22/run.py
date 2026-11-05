"""Locked four-setting CPU evaluation, full coordinate heads only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;P=R.parent/'round4'
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
new=load('_r22_core',R/'core.py');old=load('_r22_old4',P/'run.py');io=old.io
SOURCES=old.SOURCES;SETTINGS=old.SETTINGS;MODES=new.MODES;ALPHAS=new.ALPHAS
FILES=('run.py','core.py','verify.py','test_core22.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml')

def register():
    assert not (R/'lock.json').exists();old.validate()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==9
    assert not any(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0)) for s in suites)
    prior=io.read(P/'lock.json');deps=dict(prior['dependencies'])
    for n in prior['files']:deps[str((P/n).resolve())]=io.sha(P/n)
    qualified=io.read(R.parent/'round21/verification.json');assert qualified['passed'] and qualified['qualified']
    for p in (P/'lock.json',P/'results/complete.json',P/'verification.json',R.parent/'round21/closure_receipt.json',R.parent/'round21/verification.json',R.parent/'round21/coordinate_sparse.py',old.OLD/'results/summary.csv'):deps[str(p.resolve())]=io.sha(p)
    plan=io.read(P/'plan.json')
    for f in plan['folds']:
        for source in SOURCES:
            for n in ('transforms.json','arrays.npz'):
                p=P/'results/folds'/f['setting']/f"fold{f['fold']}"/source/n;deps[str(p.resolve())]=io.sha(p)
    io.write(R/'plan.json',dict(folds=plan['folds'],sources=SOURCES,modes=MODES,alphas=ALPHAS,
        primary='core_real_sparse',lambda_ratio=.1,excluded=['Wang','ALAS1'],source_seed=42))
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES+('plan.json',)},dependencies=deps,versions=io.versions(),
        CPU_only=True,CPU_limit_seconds=1800,inner_fits=2400,final_fits=96,rows=22788,source_forward=0,new_PCA=0,
        lambda_ratio=.1,primary='core_real_sparse',baseline='TD24_Ridge_CV',mean_relative_PCC_goal_pct=5))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)

def validate():
    lock=io.read(R/'lock.json');io.check_files(R,lock['files'])
    for p,h in lock['dependencies'].items():assert io.sha(Path(p))==h,p
    assert lock['versions']==io.versions();return lock

def run(start):
    validate();root=R/'results';assert not root.exists();plan=io.read(R/'plan.json');threads=threadpool_info()
    assert all(t['num_threads']==1 for t in threads)
    rows,xx,identities=old.inputs();idx={r['sample_id']:i for i,r in enumerate(rows)};locksha=io.sha(R/'lock.json')
    allpred=[];selections=[];ninner=0;nfinal=0
    for item in plan['folds']:
        setting,fold=item['setting'],item['fold'];fi=np.array([idx[s] for s in item['fit_ids']]);ti=np.array([idx[s] for s in item['test_ids']])
        yf=np.array([float(rows[i]['label']) for i in fi]);assert set(fi).isdisjoint(ti)
        for source in SOURCES:
            if time.monotonic()-start>1800:raise TimeoutError('No new source block')
            x=xx[source];out=root/'folds'/setting/f'fold{fold}'/source;refdir=P/'results/folds'/setting/f'fold{fold}'/source
            ts=io.read(refdir/'transforms.json');oof=np.full((len(fi),2,5),np.nan);bc=np.empty((5,2,5,216));ic=np.empty((5,2,5));certs=[]
            with np.load(refdir/'arrays.npz',allow_pickle=False) as a:
                tdcoef=a['inner_coef'][:,:5,:24].copy();tdcenter=a['inner_intercept'].copy();tdoof=a['inner_oof'][:,:5].copy()
            for j,sp in enumerate(item['inner']):
                tr=np.array(sp['train']);va=np.array(sp['valid']);z=new.transform(x[fi[tr]],ts['inner'][j]);zv=new.transform(x[fi[va]],ts['inner'][j])
                np.testing.assert_allclose(zv[:,:24]@tdcoef[j].T+tdcenter[j],tdoof[va],atol=1e-10,rtol=1e-10)
                for m,mode in enumerate(MODES):
                    for k,alpha in enumerate(ALPHAS):
                        if time.monotonic()-start>1800:raise TimeoutError('No new head')
                        b,c,cert=new.fit(z,yf[tr],alpha,mode);ninner+=1
                        io.write(out/f'inner{j}_{mode}_alpha{k}_certificate.json',cert)
                        if not cert['numerically_qualified']:raise ArithmeticError(f'Unqualified head {setting} {fold} {source} {j} {mode} {alpha}')
                        bc[j,m,k]=b;ic[j,m,k]=c;oof[va,m,k]=zv@b+c;certs.append(dict(inner=j,mode=mode,candidate=k,**cert))
            assert np.isfinite(oof).all();choices={};pcall=[];msall=[]
            for m,mode in enumerate(MODES):
                pc,ms=old.score(yf,oof[:,m,:]);choices[mode]=new.choose(pc,ms);pcall.append(pc);msall.append(ms)
                selections.append(dict(setting=setting,fold=fold,source=source,condition=source+'_'+mode,candidate=choices[mode],alpha=ALPHAS[choices[mode]],inner_PCC=float(pc[choices[mode]])))
            io.write(out/'selection.json',dict(selected=choices,PCC=pcall,MSE=msall,fit_ids=item['fit_ids'],test_ids=item['test_ids'],outer_validation_unused=True,source=source))
            z=new.transform(x[fi],ts['final']);zt=new.transform(x[ti],ts['final']);fb=[];fc=[];fp=[];db=[];dc=[];finalcert=[]
            for mode in MODES:
                b,c,cert=new.fit(z,yf,ALPHAS[choices[mode]],mode);nfinal+=1;io.write(out/f'final_{mode}_certificate.json',cert)
                if not cert['numerically_qualified']:raise ArithmeticError('Final head unqualified')
                coef,offset=new.deployment(ts['final'],b,c);fb.append(b);fc.append(c);fp.append(zt@b+c);db.append(coef);dc.append(offset);finalcert.append(cert)
            pred=np.array(fp).T
            old.npz_write(out/'arrays.npz',inner_coef=bc,inner_intercept=ic,inner_oof=oof,final_coef=np.array(fb),final_intercept=np.array(fc),
                test_pred=pred,deployment_coef=np.array(db),deployment_intercept=np.array(dc))
            io.write(out/'references.json',dict(transform_sha256=io.sha(refdir/'transforms.json'),source_identity=identities[source]))
            part=[dict(setting=setting,fold=fold,source=source,condition=source+'_'+mode,sample_id=rows[i]['sample_id'],
                label=rows[i]['label'],binary=rows[i]['y'],prediction=float(pred[j,m])) for m,mode in enumerate(MODES) for j,i in enumerate(ti)]
            io.write_csv(out/'predictions.csv',part);allpred+=part
            io.write(out/'complete.json',dict(lock_sha256=locksha,inner_heads=50,final_heads=2,files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
        print('CLOSED',setting,fold,'heads',ninner+nfinal,flush=True)
    assert ninner==2400 and nfinal==96 and len(allpred)==22788
    baseline={(r['setting'],r['condition']):r for r in io.table(old.OLD/'results/summary.csv')};summary=[];goals=[]
    for source in SOURCES:
        for mode in MODES:
            cond=source+'_'+mode
            for setting in SETTINGS:
                part=[r for r in allpred if r['condition']==cond and r['setting']==setting]
                cohort='Mixset' if setting=='Inter_Hu_to_Mix' else setting;expected={r['sample_id'] for r in rows if r['cohort']==cohort}
                assert len(part)==len(expected) and {r['sample_id'] for r in part}==expected
                metric=io.metric(part);base=baseline[setting,'TD24_Ridge_CV'];bp=float(base['PCC'])
                summary.append(dict(setting=setting,source=source,condition=cond,**metric,TD24_PCC=bp,PCC_delta=metric['PCC']-bp,
                    PCC_relative_gain_pct=100*(metric['PCC']/bp-1),MSE_relative_reduction_pct=100*(1-metric['MSE']/float(base['MSE']))))
            part=[r for r in summary if r['condition']==cond];gain=float(np.mean([r['PCC_relative_gain_pct'] for r in part]))
            goals.append(dict(condition=cond,mean_relative_PCC_gain_pct=gain,primary=cond==plan['primary'],reaches_development_5pct=gain>=5,
                negative_settings=[r['setting'] for r in part if r['PCC_delta']<0]))
    io.write_csv(root/'summary.csv',summary);io.write_csv(root/'oof_predictions.csv',allpred);io.write_csv(root/'selection.csv',selections)
    io.write(root/'goal_summary.json',goals);validate()
    io.write(root/'complete.json',dict(lock_sha256=locksha,inner_fits=ninner,final_heads=nfinal,rows=len(allpred),outer_folds=16,
        source_forward=0,new_PCA=0,GPU_hours=0,threadpool=threads,files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
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
