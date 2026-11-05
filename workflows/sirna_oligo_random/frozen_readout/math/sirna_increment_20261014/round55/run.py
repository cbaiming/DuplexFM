"""Four fixed settings after the R54 gate; no new scientific grid."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;B=R.parent;P=B/'round4';Q=B/'round54';N=B/'round50'
sys.path.insert(0,str(R));import head
sp=importlib.util.spec_from_file_location('_r55_prior4',P/'run.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
io=old.io;SETTINGS=old.SETTINGS;SOURCES=('TD_only',)+tuple(old.SOURCES);ALPHAS=head.ALPHAS
FILES=('run.py','head.py','verify.py','test_head.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml','close.py','build_controller.py','audit_math.py','build_verifier.py')


def register():
    assert not (R/'lock.json').exists();old.validate()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==8
    assert not any(int(s.get(k,0)) for s in suites for k in ('failures','errors','skipped'))
    assert io.read(Q/'verification.json')['passed'] and io.read(Q/'verification.json')['qualified']
    for parent in (Q,N):
        receipt=io.read(parent/'closure_receipt.json')
        for name,h in receipt['files_sha256'].items():assert io.sha(B/name)==h,name
    qlock=io.read(Q/'lock.json');deps=dict(qlock['dependencies'])
    for name in qlock['files']:deps[str((Q/name).resolve())]=io.sha(Q/name)
    for parent in (Q,N):
        for name in ('lock.json','closure_receipt.json','verification.json'):
            deps[str((parent/name).resolve())]=io.sha(parent/name)
    for p in (Q/'diagnostic.json',N/'head.py',N/'verify.py',B/'round26/head.py',B/'round26/verify.py',old.OLD/'results/summary.csv'):
        deps[str(p.resolve())]=io.sha(p)
    for name,h in io.read(Q/'diagnostic.json')['pilot_files_sha256'].items():
        p=Q/'pilot'/name;assert io.sha(p)==h;deps[str(p.resolve())]=h
    complete=io.read(N/'results/complete.json');io.check_files(N/'results',complete['files_sha256'])
    deps[str((N/'results/complete.json').resolve())]=io.sha(N/'results/complete.json')
    for name,h in complete['files_sha256'].items():deps[str((N/'results'/name).resolve())]=h
    base=io.read(P/'plan.json')
    for item in base['folds']:
        s,f=item['setting'],item['fold']
        for src in old.SOURCES:
            for name in ('arrays.npz','transforms.json'):
                p=P/'results/folds'/s/f'fold{f}'/src/name;deps[str(p.resolve())]=io.sha(p)
        for name in ('arrays.npz','selection.json','transforms.json','complete.json'):
            p=old.OLD/'results/folds'/s/f'fold{f}'/name;deps[str(p.resolve())]=io.sha(p)
    io.write(R/'plan.json',dict(folds=base['folds'],sources=SOURCES,alphas=ALPHAS,primary='core_real_common_intercept',
        common_group_penalty='n/K',lambda_ratio=.1,source_seed=42,excluded=['Wang','ALAS1'],baseline='TD24_Ridge_CV'))
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES+('plan.json',)},dependencies=deps,versions=io.versions(),
        CPU_only=True,CPU_limit_seconds=1800,total_inner_condition_heads=1600,total_final_condition_heads=64,OOF_rows=15192,
        total_training_partitions=96,reused_R50_partitions=96,new_training_partitions=0,reused_R54_geometries=15,new_group_geometries=81,reused_R54_inner_heads=60,
        equal_group_reuse=True,source_forward=0,new_PCA=0,new_original_scalers=0,goal_mean_relative_PCC_gain_pct=5.))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)


def validate():
    lock=io.read(R/'lock.json');io.check_files(R,lock['files'])
    for p,h in lock['dependencies'].items():assert io.sha(Path(p))==h,p
    assert io.versions()==lock['versions'];return lock


def group_block(item,rows,fi,root):
    result=[];s,f=item['setting'],item['fold'];out=root/'groups'/s/f'fold{f}'
    for j in range(6):
        tr=np.arange(len(fi)) if j==5 else np.array(item['inner'][j]['train']);targets=[rows[fi[i]]['target57'] for i in tr]
        name='final' if j==5 else f'inner{j}';origin=N/'results/groups'/s/f'fold{f}'/(name+'.json');saved=io.read(origin)
        assert saved['train_indices']==tr.tolist() and saved['train_ids']==[item['fit_ids'][i] for i in tr]
        assert saved['target_strings_sha256']==hashlib.sha256('\n'.join(targets).encode()).hexdigest()
        g=np.array(saved['labels'],int);pilot=None
        if f==0 and s in SETTINGS[:3] and j<5:
            pilot=Q/'pilot/geometry'/s/name/'geometry.json';oldmeta=io.read(pilot)
            assert oldmeta['train_indices']==tr.tolist() and oldmeta['train_ids']==[item['fit_ids'][i] for i in tr]
            meta=oldmeta['metadata']
        else:meta=head.geometry(g)
        state=dict(train_indices=tr,train_ids=[item['fit_ids'][i] for i in tr],labels=g,metadata=meta,
            source_group_metadata=saved['metadata'],target_strings_sha256=saved['target_strings_sha256'],
            reused_R50_sha256=io.sha(origin),reused_R54_geometry_sha256=io.sha(pilot) if pilot else None)
        io.write(out/(name+'.json'),state);result.append((g,meta,out/(name+'.json')))
    io.write(out/'complete.json',dict(lock_sha256=io.sha(R/'lock.json'),files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
    return result


def run(start):
    validate();root=R/'results';assert not root.exists();plan=io.read(R/'plan.json');threads=threadpool_info();assert all(t['num_threads']==1 for t in threads)
    rows,xx,identities=old.inputs();idx={r['sample_id']:i for i,r in enumerate(rows)};locksha=io.sha(R/'lock.json')
    allpred=[];selections=[];counts={kind:0 for kind in ('inner_new','inner_R54','inner_R50_equal','final_new','final_R50_equal')}
    for item in plan['folds']:
        s,f=item['setting'],item['fold'];fi=np.array([idx[v] for v in item['fit_ids']]);ti=np.array([idx[v] for v in item['test_ids']]);assert set(fi).isdisjoint(ti)
        yf=np.array([float(rows[i]['label']) for i in fi]);groups=group_block(item,rows,fi,root)
        for source in SOURCES:
            is_source=source!='TD_only';origin=source if is_source else 'core_real';x=xx[origin]
            out=root/'folds'/s/f'fold{f}'/source;ref=P/'results/folds'/s/f'fold{f}'/origin;nt=N/'results/folds'/s/f'fold{f}'/source
            ts=io.read(ref/'transforms.json');oof=np.full((len(fi),5),np.nan);bc=np.empty((5,5,216 if is_source else 24));ic=np.empty((5,5));reuse=[]
            for j,split in enumerate(item['inner']):
                tr,va=np.array(split['train']),np.array(split['valid']);labels,meta,_=groups[j]
                z=head.transform(x[fi[tr]],ts['inner'][j]);zv=head.transform(x[fi[va]],ts['inner'][j]);vdesign=zv if is_source else zv[:,:24]
                for k,alpha in enumerate(ALPHAS):
                    if time.monotonic()-start>1800:raise TimeoutError('CPU stage cap')
                    kind=head.reuse_kind(s,f,j,alpha,meta['reference_equal']);origin_files={}
                    if kind=='R54':
                        folder=Q/'pilot'/s/source/f'inner{j}';st=io.read(folder/'state.json');cert=st['certificate']
                        assert st['train_indices']==tr.tolist() and st['valid_indices']==va.tolist()
                        assert st['geometry_sha256']==io.sha(Q/'pilot/geometry'/s/f'inner{j}'/'geometry.json')
                        with np.load(folder/'model.npz',allow_pickle=False) as a:b=a['beta'].copy();c=float(a['intercept'])
                        origin_files={str((folder/n).resolve()):io.sha(folder/n) for n in ('model.npz','state.json','complete.json')}
                    elif kind=='R50_equal':
                        with np.load(nt/'arrays.npz',allow_pickle=False) as a:b=a['inner_coef'][j,k].copy();c=float(a['inner_intercept'][j,k])
                        cert=io.read(nt/f'inner{j}_alpha{k}_certificate.json')
                        origin_files={str((nt/n).resolve()):io.sha(nt/n) for n in ('arrays.npz',f'inner{j}_alpha{k}_certificate.json')}
                    else:b,c,cert=head.fit(z,yf[tr],labels,alpha,is_source,deadline=start+1800)
                    counts['inner_'+kind]+=1
                    if kind!='new':reuse.append(dict(stage='inner',inner=j,candidate=k,kind=kind,origin_files=origin_files))
                    assert cert['certified'];io.write(out/f'inner{j}_alpha{k}_certificate.json',cert)
                    bc[j,k]=b;ic[j,k]=c;oof[va,k]=vdesign@b+c
            assert np.isfinite(oof).all();pc,ms=old.score(yf,oof);chosen=head.choose(pc,ms);alpha=ALPHAS[chosen]
            io.write(out/'selection.json',dict(selected=chosen,alpha=alpha,PCC=pc,MSE=ms,fit_ids=item['fit_ids'],test_ids=item['test_ids'],source=source,outer_validation_unused=True))
            selections.append(dict(setting=s,fold=f,source=source,condition=source+'_common_intercept',candidate=chosen,alpha=alpha,inner_PCC=float(pc[chosen])))
            labels,meta,_=groups[5];z=head.transform(x[fi],ts['final']);zt=head.transform(x[ti],ts['final'])
            oldalpha=io.read(nt/'selection.json')['alpha']
            kind=head.reuse_kind(s,f,None,alpha,meta['reference_equal'],oldalpha);origin_files={}
            if kind=='R50_equal':
                with np.load(nt/'arrays.npz',allow_pickle=False) as a:b=a['final_coef'].copy();c=float(a['final_intercept'])
                cert=io.read(nt/'final_certificate.json')
                origin_files={str((nt/n).resolve()):io.sha(nt/n) for n in ('arrays.npz','final_certificate.json','selection.json')}
            else:b,c,cert=head.fit(z,yf,labels,alpha,is_source,deadline=start+1800)
            counts['final_'+kind]+=1
            if kind!='new':reuse.append(dict(stage='final',kind=kind,candidate=chosen,origin_files=origin_files))
            assert cert['certified'];io.write(out/'final_certificate.json',cert);io.write(out/'reuse.json',reuse)
            coef,offset=head.deployment(ts['final'],b if is_source else np.r_[b,np.zeros(192)],c);pred=(zt if is_source else zt[:,:24])@b+c
            old.npz_write(out/'arrays.npz',inner_coef=bc,inner_intercept=ic,inner_oof=oof,final_coef=b,final_intercept=np.array(c),test_pred=pred,deployment_coef=coef,deployment_intercept=np.array(offset))
            io.write(out/'references.json',dict(transform_sha256=io.sha(ref/'transforms.json'),source_identity=identities[origin] if is_source else None,
                groups_sha256={p.name:io.sha(p) for _,_,p in groups},final_equal_reused=kind!='new'))
            part=[dict(setting=s,fold=f,source=source,condition=source+'_common_intercept',sample_id=rows[i]['sample_id'],label=rows[i]['label'],binary=rows[i]['y'],prediction=float(pred[j])) for j,i in enumerate(ti)]
            io.write_csv(out/'predictions.csv',part);allpred+=part
            io.write(out/'complete.json',dict(lock_sha256=locksha,total_inner_heads=25,total_final_heads=1,
                files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
        print('CLOSED',s,f,'new',counts['inner_new']+counts['final_new'],flush=True)
    assert sum(v for k,v in counts.items() if k.startswith('inner_'))==1600 and sum(v for k,v in counts.items() if k.startswith('final_'))==64
    assert counts['inner_R54']==60 and len(allpred)==15192
    # All choices, final heads and predictions are saved before numeric test-label aggregation begins.
    io.write(root/'models_fixed.json',dict(lock_sha256=locksha,counts=counts,rows=len(allpred),all_models_fixed=True,
        model_complete_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted((root/'folds').rglob('complete.json'))}))
    base={(r['setting'],r['condition']):r for r in io.table(old.OLD/'results/summary.csv')};previous={(r['setting'],r['source']):r for r in io.table(N/'results/summary.csv')}
    metric={}
    for source in SOURCES:
        for s in SETTINGS:
            part=[r for r in allpred if r['source']==source and r['setting']==s];cohort='Mixset' if s=='Inter_Hu_to_Mix' else s
            expected={r['sample_id'] for r in rows if r['cohort']==cohort};assert len(part)==len(expected) and {r['sample_id'] for r in part}==expected
            metric[s,source]=io.metric(part)
    summary=[];goals=[];paired=[]
    for source in SOURCES:
        for s in SETTINGS:
            m=metric[s,source];oldtd=base[s,'TD24_Ridge_CV'];td=metric[s,'TD_only'];bp=float(oldtd['PCC'])
            summary.append(dict(setting=s,source=source,condition=source+'_common_intercept',**m,TD24_PCC=bp,PCC_delta=m['PCC']-bp,
                PCC_relative_gain_pct=100*(m['PCC']/bp-1),MSE_relative_reduction_pct=100*(1-m['MSE']/float(oldtd['MSE'])),
                same_treatment_TD_PCC=td['PCC'],PCC_delta_vs_same_TD=m['PCC']-td['PCC'],PCC_relative_gain_vs_same_TD_pct=100*(m['PCC']/td['PCC']-1),
                MSE_reduction_vs_same_TD_pct=100*(1-m['MSE']/td['MSE'])))
            if source!='TD_only':
                p=previous[s,source];paired.append(dict(setting=s,source=source,previous_PCC=float(p['PCC']),new_PCC=m['PCC'],delta_PCC=m['PCC']-float(p['PCC']),
                    MSE_reduction_pct=100*(1-m['MSE']/float(p['MSE']))))
        part=[r for r in summary if r['source']==source];gain=float(np.mean([r['PCC_relative_gain_pct'] for r in part]))
        goals.append(dict(condition=source+'_common_intercept',primary=source=='core_real',mean_relative_PCC_gain_pct=gain,reaches_development_5pct=gain>=5,
            mean_relative_gain_vs_same_TD_pct=float(np.mean([r['PCC_relative_gain_vs_same_TD_pct'] for r in part])),
            negative_settings=[r['setting'] for r in part if r['PCC_delta']<0],goal_complete=False))
    io.write_csv(root/'summary.csv',summary);io.write_csv(root/'oof_predictions.csv',allpred);io.write_csv(root/'selection.csv',selections)
    io.write_csv(root/'paired_R50.csv',paired);io.write(root/'goal_summary.json',goals);validate()
    io.write(root/'complete.json',dict(lock_sha256=locksha,counts=counts,total_inner_heads=1600,total_final_heads=64,rows=15192,
        total_training_partitions=96,reused_R50_partitions=96,new_training_partitions=0,reused_R54_geometries=15,new_group_geometries=81,outer_folds=16,
        source_forward=0,new_PCA=0,new_original_scalers=0,GPU_hours=0,threadpool=threads,
        files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    print('COMPLETE',goals,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('register','run'))
    if ap.parse_args().mode=='register':register()
    else:
        mutex=R/'execution.mutex'
        with mutex.open('x') as f:f.write(str(os.getpid()))
        start=time.monotonic();rc=1;error=None
        try:run(start);rc=0
        except Exception:error=traceback.format_exc();raise
        finally:
            io.write(R/f'execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,source_forward=0))
            mutex.unlink()
