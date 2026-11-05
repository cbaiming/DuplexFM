"""All four settings: independent groups, objectives, selection and output audit."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;B=R.parent;P=B/'round4';Q=B/'round49';N=B/'round26'
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
prev=load('_r50_audit26',N/'verify.py');group=load('_r50_audit49',Q/'verify.py')
old=prev.old;read,sha,table,check,close=old.read,old.sha,old.table,old.check,old.close
ALPHAS=(.1,1.,10.,100.,1000.);SOURCES=('TD_only',)+tuple(old.SOURCES)


def audit_TD(z,y,b,c,cert,alpha):
    close(c,y.mean(),1e-12);yc=y-y.mean();rhs=z.T@yc
    assert cert['alpha']==alpha and cert['mode']=='TD_partial_target_group_Ridge'
    error=float(abs((z.T@z+alpha*np.eye(24))@b-rhs).max());threshold=1e-8*(1+float(abs(rhs).max()))
    expected=np.linalg.lstsq(np.r_[z,np.sqrt(alpha)*np.eye(24)],np.r_[yc,np.zeros(24)],rcond=None)[0]
    close(b,expected,1e-8);close(cert['normal_max'],error,1e-8);close(cert['normal_threshold'],threshold,1e-9)
    assert cert['certified'] and error<=threshold;close(cert['train_MSE'],np.mean((z@b+c-y)**2),1e-10)
    return error,float(np.max(abs(b-expected)))


def main():
    start=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for p,h in lock['dependencies'].items():assert sha(Path(p))==h,p
    root=R/'results';complete=read(root/'complete.json');check(root,complete['files_sha256']);assert complete['lock_sha256']==sha(R/'lock.json')
    assert complete['source_forward']==complete['new_PCA']==complete['new_original_scalers']==complete['GPU_hours']==0
    assert all(t['num_threads']==1 for t in complete['threadpool'])
    runs=[read(p) for p in R.glob('execution_*.json')];assert len(runs)==1 and runs[0]['returncode']==0
    plan=read(R/'plan.json');assert plan['folds']==read(P/'plan.json')['folds'] and plan['alphas']==list(ALPHAS) and plan['sources']==list(SOURCES)
    assert plan['gamma']==.5 and plan['lambda_ratio']==.1 and plan['source_seed']==42 and plan['excluded']==['Wang','ALAS1'] and plan['primary']=='core_real_partial_group'
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
    predictions=table(root/'oof_predictions.csv');summary=table(root/'summary.csv');sels=table(root/'selection.csv');paired=table(root/'paired_R26.csv')
    assert len(predictions)==15192 and len(summary)==16 and len(sels)==64 and len(paired)==12
    fixed=read(root/'models_fixed.json');check(root,fixed['model_complete_sha256']);assert len(fixed['model_complete_sha256'])==64 and fixed['all_models_fixed']
    assert (root/'models_fixed.json').stat().st_mtime_ns<=(root/'summary.csv').stat().st_mtime_ns
    counts={kind:0 for kind in ('inner_new','inner_R49','inner_R26_identity','inner_TD_identity','final_new','final_R26_identity','final_TD_identity')}
    nsource=ntd=ngroup=0;maxnormal=maxgap=maxpred=maxdeploy=maxTD=0.;canonical={};partition_meta=[]
    for item in plan['folds']:
        s,f=item['setting'],item['fold'];fi=np.array([idx[v] for v in item['fit_ids']]);ti=np.array([idx[v] for v in item['test_ids']]);assert set(fi).isdisjoint(ti)
        gdir=root/'groups'/s/f'fold{f}';gr=read(gdir/'complete.json');check(gdir,gr['files_sha256']);assert gr['lock_sha256']==sha(R/'lock.json');grouping=[]
        for j in range(6):
            tr=np.arange(len(fi)) if j==5 else np.array(item['inner'][j]['train']);name='final' if j==5 else f'inner{j}';state=read(gdir/(name+'.json'))
            targets=[rows[fi[i]]['target57'] for i in tr];labels,meta=group.independent_groups(targets)
            assert state['train_indices']==tr.tolist() and state['train_ids']==[item['fit_ids'][i] for i in tr]
            assert state['target_strings_sha256']==hashlib.sha256('\n'.join(targets).encode()).hexdigest();close(state['labels'],labels,0);group.compare_mapping(state['metadata'],meta)
            if f==0 and s in old.S[:3] and j<5:assert state['reused_R49_sha256']==sha(Q/'pilot/groups'/s/name/'groups.json')
            else:assert state['reused_R49_sha256'] is None
            grouping.append((labels,meta));partition_meta.append(dict(setting=s,fold=f,stage=name,**meta));ngroup+=1
        basepath=old.OLD/'results/folds'/s/f'fold{f}';bs=read(basepath/'selection.json');bt=read(basepath/'transforms.json')['final']
        basealpha=bs['candidates'][bs['selected']['TD24_Ridge_CV']]['alpha']
        for source in SOURCES:
            if time.monotonic()-start>1800:raise TimeoutError('Audit cap')
            is_source=source!='TD_only';origin=source if is_source else 'core_real';x=xx[origin]
            out=root/'folds'/s/f'fold{f}'/source;ref=P/'results/folds'/s/f'fold{f}'/origin;nt=N/'results/folds'/s/f'fold{f}'/origin
            receipt=read(out/'complete.json');check(out,receipt['files_sha256']);assert receipt['lock_sha256']==sha(R/'lock.json') and receipt['total_inner_heads']==25 and receipt['total_final_heads']==1
            ts=read(ref/'transforms.json');references=read(out/'references.json');sel=read(out/'selection.json');reuse=read(out/'reuse.json')
            assert references['transform_sha256']==sha(ref/'transforms.json')
            assert references['source_identity']==(read(old.EXPORT/origin/'complete.json')['source_identity'] if is_source else None)
            assert references['groups_sha256']=={('final' if j==5 else f'inner{j}')+'.json':sha(gdir/(('final' if j==5 else f'inner{j}')+'.json')) for j in range(6)}
            for r in reuse:
                for p,h in r['origin_files'].items():assert p in lock['dependencies'] and sha(Path(p))==h==lock['dependencies'][p]
            with np.load(out/'arrays.npz',allow_pickle=False) as aa:a={k:aa[k].copy() for k in aa.files}
            with np.load(ref/'arrays.npz',allow_pickle=False) as aa:tdcoef=aa['inner_coef'][:,:5,:24].copy();tdcenter=aa['inner_intercept'].copy();tdoof=aa['inner_oof'][:,:5].copy()
            oof=np.full((len(fi),5),np.nan);coverage=[];expected_reuse=0
            for j,sp in enumerate(item['inner']):
                tr,va=np.array(sp['train']),np.array(sp['valid']);coverage+=va.tolist();labels,meta=grouping[j]
                old.project(x[fi[tr]],x[fi[tr]],ts['inner'][j]);z=prev.project(x[fi[tr]],ts['inner'][j]);zv=prev.project(x[fi[va]],ts['inner'][j])
                zz=group.weighted_design(z if is_source else z[:,:24],labels);yy=group.weighted_design(y[fi[tr]]-y[fi[tr]].mean(),labels)+y[fi[tr]].mean()
                gram=zz.T@zz;rhs=zz.T@(yy-yy.mean())
                close(zv[:,:24]@tdcoef[j].T+tdcenter[j],tdoof[va],1e-10)
                for k,alpha in enumerate(ALPHAS):
                    b,c=a['inner_coef'][j,k],float(a['inner_intercept'][j,k]);cert=read(out/f'inner{j}_alpha{k}_certificate.json')
                    if f==0 and s in old.S[:3] and alpha==100.:kind='R49'
                    elif meta['identity_on_centered_space']:kind='R26_identity' if is_source else 'TD_identity'
                    else:kind='new'
                    counts['inner_'+kind]+=1
                    if kind!='new':
                        rr=next(r for r in reuse if r['stage']=='inner' and r['inner']==j and r['candidate']==k);assert rr['kind']==kind;expected_reuse+=1
                        if kind=='R49':
                            p=Q/'pilot'/s/source/f'inner{j}';st=read(p/'state.json');assert st['certificate']==cert
                            assert st['train_indices']==tr.tolist() and st['valid_indices']==va.tolist()
                            with np.load(p/'model.npz',allow_pickle=False) as cm:close(b,cm['beta'],0);close(c,cm['intercept'],0)
                        elif kind=='R26_identity':
                            with np.load(nt/'arrays.npz',allow_pickle=False) as cm:close(b,cm['inner_coef'][j,k],0);close(c,cm['inner_intercept'][j,k],0)
                            assert cert==read(nt/f'inner{j}_alpha{k}_certificate.json')
                        else:close(b,tdcoef[j,k],0);close(c,tdcenter[j],0)
                    if is_source:
                        normal,gap=prev.check_head(zz,yy,b,c,cert,alpha,gram,rhs,reused='maximum_iterations' not in cert);maxgap=max(maxgap,gap);nsource+=1
                    else:normal,error=audit_TD(zz,yy,b,c,cert,alpha);maxTD=max(maxTD,error);ntd+=1
                    maxnormal=max(maxnormal,normal);oof[va,k]=(zv if is_source else zv[:,:24])@b+c
            assert sorted(coverage)==list(range(len(fi)));close(oof,a['inner_oof'],1e-9)
            assert sel['fit_ids']==item['fit_ids'] and sel['test_ids']==item['test_ids'] and sel['outer_validation_unused'] and sel['source']==source
            yc=y[fi]-y[fi].mean();pc=a['inner_oof']-a['inner_oof'].mean(0);pcc=yc@pc/np.sqrt((yc@yc)*np.sum(pc*pc,axis=0));mse=np.mean((a['inner_oof']-y[fi,None])**2,axis=0)
            close(sel['PCC'],pcc,1e-10);close(sel['MSE'],mse,1e-11);k=min(range(5),key=lambda k:(-pcc[k],mse[k],-ALPHAS[k],k));alpha=ALPHAS[k]
            assert sel['selected']==k and sel['alpha']==alpha
            sr=next(r for r in sels if r['setting']==s and int(r['fold'])==f and r['source']==source);assert int(sr['candidate'])==k and float(sr['alpha'])==alpha;close(float(sr['inner_PCC']),pcc[k],1e-10)
            old.project(x[fi],x[fi],ts['final']);z=prev.project(x[fi],ts['final']);zt=prev.project(x[ti],ts['final']);labels,meta=grouping[5]
            zz=group.weighted_design(z if is_source else z[:,:24],labels);yy=group.weighted_design(y[fi]-y[fi].mean(),labels)+y[fi].mean()
            b,c=a['final_coef'],float(a['final_intercept']);cert=read(out/'final_certificate.json');oldalpha=read(nt/'selection.json')['alpha'] if is_source else basealpha
            kind=('R26_identity' if is_source else 'TD_identity') if meta['identity_on_centered_space'] and alpha==oldalpha else 'new'
            assert references['final_identity_reused']==(kind!='new');counts['final_'+kind]+=1
            if kind!='new':
                rr=next(r for r in reuse if r['stage']=='final');assert rr['kind']==kind and rr['candidate']==k;expected_reuse+=1
                if is_source:
                    with np.load(nt/'arrays.npz',allow_pickle=False) as cm:close(b,cm['final_coef'],0);close(c,cm['final_intercept'],0)
                    assert cert==read(nt/'final_certificate.json')
                else:
                    close(bt['mean'][:24],ts['final']['td_mean'],1e-11);close(bt['scale'][:24],ts['final']['td_scale'],1e-11)
                    with np.load(basepath/'arrays.npz',allow_pickle=False) as cm:close(b,cm['final_coef'][1,:24],0);assert not np.any(cm['final_coef'][1,24:])
                    close(c,bt['intercept'],0)
            assert len(reuse)==expected_reuse
            if is_source:
                normal,gap=prev.check_head(zz,yy,b,c,cert,alpha,zz.T@zz,zz.T@(yy-yy.mean()),reused='maximum_iterations' not in cert);maxgap=max(maxgap,gap);nsource+=1
            else:normal,error=audit_TD(zz,yy,b,c,cert,alpha);maxTD=max(maxTD,error);ntd+=1
            maxnormal=max(maxnormal,normal);pp=(zt if is_source else zt[:,:24])@b+c;close(pp,a['test_pred'],1e-9);maxpred=max(maxpred,float(abs(pp-a['test_pred']).max()))
            bfull=b if is_source else np.r_[b,np.zeros(192)];mean=np.r_[ts['final']['td_mean'],ts['final']['src_mean']];scale=np.r_[ts['final']['td_scale'],np.array(ts['final']['src_scale'])*np.sqrt(12)]
            close(a['deployment_coef'],bfull/scale,1e-11);close(a['deployment_intercept'],c-mean@(bfull/scale),1e-11)
            rawpred=x[ti]@a['deployment_coef']+a['deployment_intercept'];close(rawpred,pp,1e-9);maxdeploy=max(maxdeploy,float(abs(rawpred-pp).max()))
            assert (out/'selection.json').stat().st_mtime_ns<=(out/'final_certificate.json').stat().st_mtime_ns
            part=[r for r in predictions if r['setting']==s and int(r['fold'])==f and r['source']==source];assert [r['sample_id'] for r in part]==item['test_ids']
            close([float(r['prediction']) for r in part],pp,1e-9)
            for r,i in zip(part,ti):assert r['condition']==source+'_partial_group' and float(r['label'])==y[i] and int(r['binary'])==binary[i]
            canonical.setdefault((s,source),[]).extend(zip(ti,[float(r['prediction']) for r in part]))
        print('VERIFIED',s,f,flush=True)
    assert nsource==1248 and ntd==416 and ngroup==96 and counts==complete['counts']==fixed['counts']
    assert sum(v for k,v in counts.items() if k.startswith('inner_'))==1600 and sum(v for k,v in counts.items() if k.startswith('final_'))==64 and counts['inner_R49']==60
    assert complete['total_inner_heads']==1600 and complete['total_final_heads']==64 and complete['rows']==15192 and complete['outer_folds']==16
    assert complete['total_training_partitions']==96 and complete['new_training_partitions']==81 and complete['reused_R49_partitions']==15
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
        assert r['primary']==(r['condition']=='core_real_partial_group') and r['reaches_development_5pct']==(gain>=5) and not r['goal_complete']
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
