"""All four settings: independent groups, objectives, selection and output audit."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;B=R.parent;P=B/'round4';Q=B/'round54';N=B/'round50'
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
prev=load('_r55_audit26',B/'round26/verify.py');group=load('_r55_audit49',B/'round49/verify.py');math=load('_r55_math',R/'audit_math.py')
old=prev.old;read,sha,table,check,close=old.read,old.sha,old.table,old.check,old.close
ALPHAS=(.1,1.,10.,100.,1000.);SOURCES=('TD_only',)+tuple(old.SOURCES)


def main():
    start=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for p,h in lock['dependencies'].items():assert sha(Path(p))==h,p
    root=R/'results';complete=read(root/'complete.json');check(root,complete['files_sha256']);assert complete['lock_sha256']==sha(R/'lock.json')
    assert complete['source_forward']==complete['new_PCA']==complete['new_original_scalers']==complete['GPU_hours']==0
    assert all(t['num_threads']==1 for t in complete['threadpool'])
    runs=[read(p) for p in R.glob('execution_*.json')];assert len(runs)==1 and runs[0]['returncode']==0
    plan=read(R/'plan.json');assert plan['folds']==read(P/'plan.json')['folds'] and plan['alphas']==list(ALPHAS) and plan['sources']==list(SOURCES)
    assert plan['common_group_penalty']=='n/K' and plan['lambda_ratio']==.1 and plan['source_seed']==42 and plan['excluded']==['Wang','ALAS1'] and plan['primary']=='core_real_common_intercept'
    rows=[]
    for name,n in zip(old.S[:3],(2196,469,664)):
        rr=table(old.BENCH/'data'/f'{name}.csv');assert len(rr)==n;rows+=rr
    idx={r['sample_id']:i for i,r in enumerate(rows)};assert len(idx)==3329
    td=np.array([[float(v) for v in r['td'].split(',')] for r in rows]);y=np.array([float(r['label']) for r in rows]);binary=np.array([int(r['y']) for r in rows]);xx={}
    for source in old.SOURCES:
        folder=old.EXPORT/source;identity=read(folder/'complete.json');check(folder,identity['files_sha256'])
        meta=table(folder/'row_identity.csv');imap={r['sample_id']:i for i,r in enumerate(meta)}
        for r in rows:assert all(r[k]==meta[imap[r['sample_id']]][k] for k in ('sample_id','cohort','guide','target57'))
        xx[source]=np.c_[td,np.load(folder/'source.npy',allow_pickle=False)[[imap[r['sample_id']] for r in rows]].astype(float)]
    predictions=table(root/'oof_predictions.csv');summary=table(root/'summary.csv');sels=table(root/'selection.csv');paired=table(root/'paired_R50.csv')
    assert len(predictions)==15192 and len(summary)==16 and len(sels)==64 and len(paired)==12
    fixed=read(root/'models_fixed.json');check(root,fixed['model_complete_sha256']);assert len(fixed['model_complete_sha256'])==64 and fixed['all_models_fixed']
    assert (root/'models_fixed.json').stat().st_mtime_ns<=(root/'summary.csv').stat().st_mtime_ns
    counts={kind:0 for kind in ('inner_new','inner_R54','inner_R50_equal','final_new','final_R50_equal')}
    nsource=ntd=ngroup=0;maxnormal=maxgap=maxpred=maxdeploy=maxTD=0.;canonical={};partition_meta=[]
    for item in plan['folds']:
        s,f=item['setting'],item['fold'];fi=np.array([idx[v] for v in item['fit_ids']]);ti=np.array([idx[v] for v in item['test_ids']]);assert set(fi).isdisjoint(ti)
        gdir=root/'groups'/s/f'fold{f}';gr=read(gdir/'complete.json');check(gdir,gr['files_sha256']);assert gr['lock_sha256']==sha(R/'lock.json');grouping=[]
        for j in range(6):
            tr=np.arange(len(fi)) if j==5 else np.array(item['inner'][j]['train']);name='final' if j==5 else f'inner{j}';state=read(gdir/(name+'.json'))
            targets=[rows[fi[i]]['target57'] for i in tr];labels,gm=group.independent_groups(targets)
            meta=math.indicator_geometry(labels)[3]
            assert state['train_indices']==tr.tolist() and state['train_ids']==[item['fit_ids'][i] for i in tr]
            assert state['target_strings_sha256']==hashlib.sha256('\n'.join(targets).encode()).hexdigest();close(state['labels'],labels,0)
            group.compare_mapping(state['metadata'],meta);group.compare_mapping(state['source_group_metadata'],gm)
            prior=N/'results/groups'/s/f'fold{f}'/(name+'.json');saved=read(prior)
            assert state['reused_R50_sha256']==sha(prior) and saved['train_indices']==tr.tolist() and saved['train_ids']==state['train_ids']
            close(saved['labels'],labels,0);group.compare_mapping(saved['metadata'],gm)
            assert saved['target_strings_sha256']==state['target_strings_sha256']
            if f==0 and s in old.S[:3] and j<5:
                pilot=Q/'pilot/geometry'/s/name/'geometry.json';p=read(pilot)
                assert state['reused_R54_geometry_sha256']==sha(pilot) and p['train_indices']==tr.tolist() and p['train_ids']==state['train_ids']
                group.compare_mapping(p['metadata'],meta)
            else:assert state['reused_R54_geometry_sha256'] is None
            grouping.append((labels,meta));partition_meta.append(dict(setting=s,fold=f,stage=name,**meta));ngroup+=1
        for source in SOURCES:
            if time.monotonic()-start>1800:raise TimeoutError('Audit cap')
            is_source=source!='TD_only';origin=source if is_source else 'core_real';x=xx[origin]
            out=root/'folds'/s/f'fold{f}'/source;ref=P/'results/folds'/s/f'fold{f}'/origin;nt=N/'results/folds'/s/f'fold{f}'/source
            receipt=read(out/'complete.json');check(out,receipt['files_sha256']);assert receipt['lock_sha256']==sha(R/'lock.json') and receipt['total_inner_heads']==25 and receipt['total_final_heads']==1
            ts=read(ref/'transforms.json');references=read(out/'references.json');sel=read(out/'selection.json');reuse=read(out/'reuse.json')
            assert references['transform_sha256']==sha(ref/'transforms.json')
            assert references['source_identity']==(read(old.EXPORT/origin/'complete.json')['source_identity'] if is_source else None)
            assert references['groups_sha256']=={('final' if j==5 else f'inner{j}')+'.json':sha(gdir/(('final' if j==5 else f'inner{j}')+'.json')) for j in range(6)}
            for r in reuse:
                for p,h in r['origin_files'].items():assert p in lock['dependencies'] and sha(Path(p))==h==lock['dependencies'][p]
            with np.load(out/'arrays.npz',allow_pickle=False) as aa:a={k:aa[k].copy() for k in aa.files}
            oof=np.full((len(fi),5),np.nan);coverage=[];expected_reuse=0
            for j,sp in enumerate(item['inner']):
                tr,va=np.array(sp['train']),np.array(sp['valid']);coverage+=va.tolist();labels,meta=grouping[j]
                old.project(x[fi[tr]],x[fi[tr]],ts['inner'][j]);z=prev.project(x[fi[tr]],ts['inner'][j]);zv=prev.project(x[fi[va]],ts['inner'][j])
                profiled=math.profile(z if is_source else z[:,:24],y[fi[tr]],labels)
                for k,alpha in enumerate(ALPHAS):
                    b,c=a['inner_coef'][j,k],float(a['inner_intercept'][j,k]);cert=read(out/f'inner{j}_alpha{k}_certificate.json')
                    if f==0 and s in old.S[:3] and alpha==100.:kind='R54'
                    elif meta['reference_equal']:kind='R50_equal'
                    else:kind='new'
                    counts['inner_'+kind]+=1
                    if kind!='new':
                        rr=next(r for r in reuse if r['stage']=='inner' and r['inner']==j and r['candidate']==k);assert rr['kind']==kind;expected_reuse+=1
                        if kind=='R54':
                            p=Q/'pilot'/s/source/f'inner{j}';st=read(p/'state.json');assert st['certificate']==cert
                            assert st['train_indices']==tr.tolist() and st['valid_indices']==va.tolist()
                            assert st['geometry_sha256']==sha(Q/'pilot/geometry'/s/f'inner{j}'/'geometry.json')
                            with np.load(p/'model.npz',allow_pickle=False) as cm:close(b,cm['beta'],0);close(c,cm['intercept'],0)
                        elif kind=='R50_equal':
                            with np.load(nt/'arrays.npz',allow_pickle=False) as cm:close(b,cm['inner_coef'][j,k],0);close(c,cm['inner_intercept'][j,k],0)
                            assert cert==read(nt/f'inner{j}_alpha{k}_certificate.json')
                    if is_source:
                        normal,gap=math.check_source(z,y[fi[tr]],labels,b,c,cert,alpha,profiled);maxgap=max(maxgap,gap);nsource+=1
                    else:normal,error=math.audit_TD(z[:,:24],y[fi[tr]],labels,b,c,cert,alpha,profiled);maxTD=max(maxTD,error);ntd+=1
                    maxnormal=max(maxnormal,normal);oof[va,k]=(zv if is_source else zv[:,:24])@b+c
            assert sorted(coverage)==list(range(len(fi)));close(oof,a['inner_oof'],1e-9)
            assert sel['fit_ids']==item['fit_ids'] and sel['test_ids']==item['test_ids'] and sel['outer_validation_unused'] and sel['source']==source
            yc=y[fi]-y[fi].mean();pc=a['inner_oof']-a['inner_oof'].mean(0);pcc=yc@pc/np.sqrt((yc@yc)*np.sum(pc*pc,axis=0));mse=np.mean((a['inner_oof']-y[fi,None])**2,axis=0)
            close(sel['PCC'],pcc,1e-10);close(sel['MSE'],mse,1e-11);k=min(range(5),key=lambda k:(-pcc[k],mse[k],-ALPHAS[k],k));alpha=ALPHAS[k]
            assert sel['selected']==k and sel['alpha']==alpha
            sr=next(r for r in sels if r['setting']==s and int(r['fold'])==f and r['source']==source);assert int(sr['candidate'])==k and float(sr['alpha'])==alpha;close(float(sr['inner_PCC']),pcc[k],1e-10)
            old.project(x[fi],x[fi],ts['final']);z=prev.project(x[fi],ts['final']);zt=prev.project(x[ti],ts['final']);labels,meta=grouping[5]
            profiled=math.profile(z if is_source else z[:,:24],y[fi],labels)
            b,c=a['final_coef'],float(a['final_intercept']);cert=read(out/'final_certificate.json');oldalpha=read(nt/'selection.json')['alpha']
            kind='R50_equal' if meta['reference_equal'] and alpha==oldalpha else 'new'
            assert references['final_equal_reused']==(kind!='new');counts['final_'+kind]+=1
            if kind!='new':
                rr=next(r for r in reuse if r['stage']=='final');assert rr['kind']==kind and rr['candidate']==k;expected_reuse+=1
                with np.load(nt/'arrays.npz',allow_pickle=False) as cm:close(b,cm['final_coef'],0);close(c,cm['final_intercept'],0)
                assert cert==read(nt/'final_certificate.json')
            assert len(reuse)==expected_reuse
            if is_source:
                normal,gap=math.check_source(z,y[fi],labels,b,c,cert,alpha,profiled);maxgap=max(maxgap,gap);nsource+=1
            else:normal,error=math.audit_TD(z[:,:24],y[fi],labels,b,c,cert,alpha,profiled);maxTD=max(maxTD,error);ntd+=1
            maxnormal=max(maxnormal,normal);pp=(zt if is_source else zt[:,:24])@b+c;close(pp,a['test_pred'],1e-9);maxpred=max(maxpred,float(abs(pp-a['test_pred']).max()))
            bfull=b if is_source else np.r_[b,np.zeros(192)];mean=np.r_[ts['final']['td_mean'],ts['final']['src_mean']];scale=np.r_[ts['final']['td_scale'],np.array(ts['final']['src_scale'])*np.sqrt(12)]
            close(a['deployment_coef'],bfull/scale,1e-11);close(a['deployment_intercept'],c-mean@(bfull/scale),1e-11)
            rawpred=x[ti]@a['deployment_coef']+a['deployment_intercept'];close(rawpred,pp,1e-9);maxdeploy=max(maxdeploy,float(abs(rawpred-pp).max()))
            assert (out/'selection.json').stat().st_mtime_ns<=(out/'final_certificate.json').stat().st_mtime_ns
            part=[r for r in predictions if r['setting']==s and int(r['fold'])==f and r['source']==source];assert [r['sample_id'] for r in part]==item['test_ids']
            close([float(r['prediction']) for r in part],pp,1e-9)
            for r,i in zip(part,ti):assert r['condition']==source+'_common_intercept' and float(r['label'])==y[i] and int(r['binary'])==binary[i]
            canonical.setdefault((s,source),[]).extend(zip(ti,[float(r['prediction']) for r in part]))
        print('VERIFIED',s,f,flush=True)
    assert nsource==1248 and ntd==416 and ngroup==96 and counts==complete['counts']==fixed['counts']
    assert sum(v for k,v in counts.items() if k.startswith('inner_'))==1600 and sum(v for k,v in counts.items() if k.startswith('final_'))==64 and counts['inner_R54']==60
    assert complete['total_inner_heads']==1600 and complete['total_final_heads']==64 and complete['rows']==15192 and complete['outer_folds']==16
    assert complete['total_training_partitions']==96 and complete['new_training_partitions']==0 and complete['reused_R50_partitions']==96 and complete['reused_R54_geometries']==15 and complete['new_group_geometries']==81
    baseline={(r['setting'],r['condition']):r for r in table(old.OLD/'results/summary.csv')};previous={(r['setting'],r['source']):r for r in table(N/'results/summary.csv')};metrics={}
    for (s,source),pairs in canonical.items():
        ii=np.array([i for i,p in pairs]);pp=np.array([p for i,p in pairs]);cohort='Mixset' if s=='Inter_Hu_to_Mix' else s
        expected={i for i,r in enumerate(rows) if r['cohort']==cohort};assert len(ii)==len(expected) and set(ii)==expected
        metrics[s,source]=old.metric(y[ii],pp,binary[ii])
    for r in summary:
        s,source=r['setting'],r['source'];m=metrics[s,source];td=metrics[s,'TD_only'];base=baseline[s,'TD24_Ridge_CV'];bp=float(base['PCC'])
        for k,v in m.items():close(float(r[k]),v,1e-9)
        expected=dict(TD24_PCC=bp,PCC_delta=m['PCC']-bp,PCC_relative_gain_pct=100*(m['PCC']/bp-1),MSE_relative_reduction_pct=100*(1-m['MSE']/float(base['MSE'])),
            same_treatment_TD_PCC=td['PCC'],PCC_delta_vs_same_TD=m['PCC']-td['PCC'],PCC_relative_gain_vs_same_TD_pct=100*(m['PCC']/td['PCC']-1),MSE_reduction_vs_same_TD_pct=100*(1-m['MSE']/td['MSE']))
        for k,v in expected.items():close(float(r[k]),v,1e-9)
        if source!='TD_only':
            p=previous[s,source];pair=next(v for v in paired if v['setting']==s and v['source']==source)
            for k,v in dict(previous_PCC=float(p['PCC']),new_PCC=m['PCC'],delta_PCC=m['PCC']-float(p['PCC']),MSE_reduction_pct=100*(1-m['MSE']/float(p['MSE']))).items():close(float(pair[k]),v,1e-9)
    goals=read(root/'goal_summary.json');assert len(goals)==4
    for r in goals:
        part=[p for p in summary if p['condition']==r['condition']];assert len(part)==4
        gain=float(np.mean([float(p['PCC_relative_gain_pct']) for p in part]));close(r['mean_relative_PCC_gain_pct'],gain,1e-9)
        close(r['mean_relative_gain_vs_same_TD_pct'],np.mean([float(p['PCC_relative_gain_vs_same_TD_pct']) for p in part]),1e-9)
        assert r['primary']==(r['condition']=='core_real_common_intercept') and r['reaches_development_5pct']==(gain>=5) and not r['goal_complete']
        assert r['negative_settings']==[p['setting'] for p in part if float(p['PCC_delta'])<0]
    result=dict(passed=True,counts=counts,source_full_joint_certificates=nsource,TD_independent_augmented_lstsq=ntd,
        independent_training_graphs=ngroup,OOF_rows=15192,source_models=3,auxiliary_TD_conditions=1,
        max_normal_error=maxnormal,max_relative_dual_gap=maxgap,max_independent_TD_coef_difference=maxTD,
        max_prediction_difference=maxpred,max_deployment_difference=maxdeploy,goal_summary=goals,partition_metadata=partition_meta,
        new_source_forward=0,new_PCA=0,new_original_scalers=0,GPU_hours=0,
        lock_sha256=sha(R/'lock.json'),complete_sha256=sha(root/'complete.json'),CPU_wall_seconds=time.monotonic()-start)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='partition_metadata'}),flush=True)


if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0),f,indent=2)
