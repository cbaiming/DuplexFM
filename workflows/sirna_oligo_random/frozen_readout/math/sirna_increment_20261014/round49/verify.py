"""Independent sparse target incidence components and indicator projection audit."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;P=R.parent/'round4';D=R.parent/'round21';N=R.parent/'round25'


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


old=load('_r49_independent4',P/'verify.py');certify=load('_r49_independent36',R.parent/'round36/verify.py').certify
read,sha,table,check=old.read,old.sha,old.table,old.check


def close(a,b,tol=1e-9):np.testing.assert_allclose(a,b,rtol=tol,atol=tol)
def metric(y,p):
    assert np.isfinite(p).all() and np.std(p)>1e-14 and np.std(y)>1e-14
    return dict(PCC=float(np.corrcoef(y,p)[0,1]),MSE=float(np.mean((y-p)**2)))


def compare_mapping(actual,expected):
    assert set(actual)==set(expected)
    for k,value in expected.items():
        if isinstance(value,dict):compare_mapping(actual[k],value)
        elif isinstance(value,(str,bool)) or value is None:assert actual[k]==value,k
        else:close(actual[k],value)


def independent_groups(targets):
    sets=[];complement=dict(A='T',C='G',G='C',T='A')
    for value in targets:
        s=value.upper().replace('U','T');assert len(s)==57 and all(c in complement for c in s)
        fragments=set()
        for j in range(len(s)-24):
            part=s[j:j+25];reverse=''.join(complement[c] for c in reversed(part));fragments.add(min(part,reverse))
        sets.append(fragments)
    words=sorted(set().union(*sets));dictionary={s:i for i,s in enumerate(words)};rr=[];cc=[]
    for i,subset in enumerate(sets):
        for s in subset:rr.append(i);cc.append(dictionary[s])
    n=len(targets);inc=csr_matrix((np.ones(len(rr),dtype=np.int32),(rr,cc)),shape=(n,len(words)))
    k,labels=connected_components(inc@inc.T,directed=False)
    ordered=sorted(range(k),key=lambda label:int(np.flatnonzero(labels==label)[0]));mapping={label:i for i,label in enumerate(ordered)}
    labels=np.array([mapping[int(label)] for label in labels],int);sizes=np.bincount(labels);c=(n-1)/(n-1-.5*(k-1))
    return labels,dict(n=n,K=k,sizes=sizes.tolist(),unique_canonical_25mers=len(words),gamma=.5,normalizer=c,
        identity_on_centered_space=bool(k==1 or k==n),largest_fraction=float(max(sizes)/n),
        nonsingleton_groups=int(sum(sizes>1)),nonsingleton_rows=int(sum(sizes[sizes>1])),implicit_group_contrasts=k-1,
        confirmed_gene_or_assay=False,training_inputs_only=True)


def weighted_design(x,labels):
    x=np.asarray(x,float);g=np.asarray(labels,int);n=len(x);k=int(max(g))+1
    assert np.max(abs(x.mean(axis=0)))<1e-8
    if k==1 or k==n:return x.copy()
    q=np.zeros((n,k));q[np.arange(n),g]=1/np.sqrt(np.bincount(g)[g])
    c=(n-1)/(n-1-.5*(k-1));projection=q@(q.T@x)-x.mean(axis=0)
    return np.sqrt(c)*(x+(np.sqrt(.5)-1)*projection)


def audit_TD(zz,yc,b,cert):
    assert cert['mode']=='TD_partial_target_group_Ridge' and cert['alpha']==100
    expected=np.linalg.lstsq(np.r_[zz,10*np.eye(24)],np.r_[yc,np.zeros(24)],rcond=None)[0]
    close(b,expected,1e-10);rhs=zz.T@yc;normal=float(max(abs((zz.T@zz+100*np.eye(24))@b-rhs)))
    threshold=1e-8*(1+float(max(abs(rhs))));valid=normal<=threshold
    close(cert['normal_max'],normal,1e-10);close(cert['normal_threshold'],threshold,1e-10)
    close(cert['train_MSE'],np.mean((zz@b-yc)**2),1e-11);assert cert['certified']==valid
    return valid,normal,float(np.max(abs(b-expected)))


def residual_audit(z,y,b,c,labels,source):
    x=z if source else z[:,:24];r=y-x@b-c;centered=r-r.mean();k=int(max(labels))+1
    between=sum(len(a)*float(centered[a].mean())**2 for a in [np.flatnonzero(labels==j) for j in range(k)])
    total=float(centered@centered);n=len(labels);factor=(n-1)/(n-1-.5*(k-1))
    loss=factor*(total-.5*between)
    close(float(np.sum(weighted_design(centered,labels)**2)),loss,1e-10)
    return dict(raw_training_MSE=float(np.mean(r*r)),centered_residual_SSE=total,centered_between_SSE=between,
        centered_within_SSE=total-between,transformed_residual_SSE=loss)


def main():
    start=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    assert all(t['num_threads']==1 for t in threadpool_info())
    lock=read(R/'lock.json');check(R,lock['files'])
    for path,h in lock['dependencies'].items():assert sha(Path(path))==h,path
    assert lock['alpha']==100 and lock['lambda_ratio']==.1 and lock['gamma']==.5 and lock['training_target_canonical_kmer']==25
    assert lock['target_only_transitive_components'] and lock['trace_normalize_centered_space'] and lock['identity_reuse']
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json') and not d['outer_test_metrics_used']
    assert d['new_source_forward']==d['new_PCA']==d['new_original_scalers']==d['GPU_hours']==0
    assert all(t['num_threads']==1 for t in d['threadpool'])
    check(R/'pilot',d['pilot_files_sha256']);assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    runs=[read(p) for p in R.glob('execution_*.json')];assert len(runs)==1 and runs[0]['returncode']==0
    rowsout=table(R/'diagnostic_rows.csv');plan=read(P/'plan.json');records=[]
    counts=dict(new_source=0,new_TD=0,identity_reused=0,old_source=0,old_TD=0,rows=0,partitions=0)
    numeric=True;maxnormal=maxgap=maxpred=maxdeploy=maxlstsq=0.;partitions=[]
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids']
        assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids'])
        rows={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')};y=np.array([float(rows[s]['label']) for s in ids])
        td=np.array([[float(v) for v in rows[s]['td'].split(',')] for s in ids]);tdpath=P/'results/folds'/setting/'fold0/core_real/arrays.npz'
        with np.load(tdpath,allow_pickle=False) as a:
            tdpred=a['inner_oof'][:,3].copy();tdcoef=a['inner_coef'][:,3,:24].copy();tdintercept=a['inner_intercept'].copy()
        assert tdintercept.shape==(5,);grouping=[]
        for j,sp in enumerate(item['inner']):
            tr,va=np.array(sp['train']),np.array(sp['valid']);target=[rows[ids[i]]['target57'] for i in tr]
            gpath=R/'pilot/groups'/setting/f'inner{j}';gr=read(gpath/'complete.json');check(gpath,gr['files_sha256'])
            assert gr['lock_sha256']==sha(R/'lock.json');state=read(gpath/'groups.json')
            assert state['setting']==setting and state['inner']==j and state['train_indices']==tr.tolist() and state['train_ids']==[ids[i] for i in tr]
            assert state['target_strings_sha256']==hashlib.sha256('\n'.join(target).encode()).hexdigest()
            labels,meta=independent_groups(target);close(labels,state['labels'],0);compare_mapping(state['metadata'],meta)
            grouping.append((labels,meta,gpath));partitions.append(dict(setting=setting,inner=j,**meta));counts['partitions']+=1
            tm,ts=old.scale(td[tr],False);z=(td[tr]-tm)/ts;zv=(td[va]-tm)/ts
            close(tdintercept[j],y[tr].mean(),1e-12)
            close((z.T@z+100*np.eye(24))@tdcoef[j],z.T@(y[tr]-y[tr].mean()),1e-8)
            close(zv@tdcoef[j]+tdintercept[j],tdpred[va],1e-10);counts['old_TD']+=1
        for source in ('TD_only',)+tuple(old.SOURCES):
            is_source=source!='TD_only';origin=source if is_source else 'core_real'
            folder=old.EXPORT/origin;identity=read(folder/'complete.json')
            assert d['source_identities'][origin]==identity['source_identity'];check(folder,identity['files_sha256'])
            meta=table(folder/'row_identity.csv');idx={r['sample_id']:i for i,r in enumerate(meta)}
            for s in ids:assert all(rows[s][k]==meta[idx[s]][k] for k in ('sample_id','cohort','guide','target57'))
            raw=np.c_[td,np.load(folder/'source.npy',allow_pickle=False)[[idx[s] for s in ids]].astype(float)]
            pred=np.full(len(ids),np.nan);reference=pred.copy();coverage=[];blocks=[]
            for j,sp in enumerate(item['inner']):
                if time.monotonic()-start>900:raise TimeoutError('CPU audit cap')
                tr,va=np.array(sp['train']),np.array(sp['valid']);assert set(tr).isdisjoint(va)
                labels,groupmeta,gpath=grouping[j];out=R/'pilot'/setting/source/f'inner{j}'
                receipt=read(out/'complete.json');check(out,receipt['files_sha256']);assert receipt['lock_sha256']==sha(R/'lock.json')
                state=read(out/'state.json');cert=state['certificate'];reused=groupmeta['identity_on_centered_space']
                control=D/'pilot'/setting/origin/'dense'/f'inner{j}';native=N/'pilot'/setting/origin/f'inner{j}'
                assert (state['setting'],state['source'],state['inner'])==(setting,source,j)
                assert state['train_indices']==tr.tolist() and state['valid_indices']==va.tolist() and state['identity_reused']==reused
                assert state['transform_sha256']==sha(control/'transform.json') and state['TD_reference_sha256']==sha(tdpath)
                assert state['groups_sha256']==sha(gpath/'groups.json')
                t=read(control/'transform.json');tm,ts=old.scale(raw[tr,:24],False);sm,ss=old.scale(raw[tr,24:],True)
                for key,value in (('td_mean',tm),('td_scale',ts),('src_mean',sm),('src_scale',ss)):close(t[key],value,1e-11)
                mean=np.r_[tm,sm];scale=np.r_[ts,ss*np.sqrt(12.)];z=(raw[tr]-mean)/scale;zv=(raw[va]-mean)/scale
                if is_source:
                    assert state['native_reference_sha256']==sha(native/'model.npz')
                    nr=read(native/'complete.json');check(native,nr['files_sha256']);assert nr['lock_sha256']==sha(N/'lock.json')
                    ns=read(native/'state.json');assert ns['train_indices']==tr.tolist() and ns['valid_indices']==va.tolist()
                    assert ns['transform_sha256']==sha(control/'transform.json')
                    with np.load(native/'model.npz',allow_pickle=False) as a:nb=a['beta'].copy();nc=float(a['intercept']);npred=a['prediction'].copy()
                    nv,normal,gap,_=certify(z,y[tr],nb,nc,ns['certificate'],False);assert nv
                    close(zv@nb+nc,npred,1e-10);reference[va]=npred;counts['old_source']+=1
                    maxnormal=max(maxnormal,normal);maxgap=max(maxgap,gap)
                else:
                    assert state['native_reference_sha256'] is None;reference[va]=tdpred[va];nb=tdcoef[j];nc=tdintercept[j]
                with np.load(out/'model.npz',allow_pickle=False) as a:m={k:a[k].copy() for k in a.files}
                b=m['beta'];c=float(m['intercept']);close(c,y[tr].mean(),1e-12)
                x=z if is_source else z[:,:24];zz=weighted_design(x,labels);yc=weighted_design(y[tr]-y[tr].mean(),labels)
                if reused:
                    close(b,nb,0);close(c,nc,0);close(zz,x,0);counts['identity_reused']+=1
                elif is_source:counts['new_source']+=1
                else:counts['new_TD']+=1
                if is_source:
                    valid,normal,gap,_=certify(zz,yc+y[tr].mean(),b,c,cert,False);maxgap=max(maxgap,gap)
                else:
                    valid,normal,lstsq_error=audit_TD(zz,yc,b,cert);maxlstsq=max(maxlstsq,lstsq_error)
                numeric &= valid;assert valid==state['numerically_qualified']
                pp=(zv if is_source else zv[:,:24])@b+c;close(m['prediction'],pp,1e-10)
                if reused:close(pp,reference[va],1e-10)
                fullbeta=b if is_source else np.r_[b,np.zeros(192)]
                close(m['deployment_coef'],fullbeta/scale,1e-11);close(m['deployment_intercept'],c-mean@(fullbeta/scale),1e-10)
                rawp=raw[va]@m['deployment_coef']+m['deployment_intercept'];close(rawp,pp,1e-9)
                residual=residual_audit(z,y[tr],b,c,labels,is_source);compare_mapping(state['training_residual'],residual)
                # Weighted loss in the saved certificate must be the projected residual loss, not raw training MSE.
                close(cert['train_MSE']*len(tr),residual['transformed_residual_SSE'],1e-9)
                maxnormal=max(maxnormal,normal);maxpred=max(maxpred,float(abs(m['prediction']-pp).max()));maxdeploy=max(maxdeploy,float(abs(rawp-pp).max()))
                pred[va]=m['prediction'];coverage+=va.tolist()
                blocks.append(dict(inner=j,n=len(va),metrics=metric(y[va],pp),reference=metric(y[va],reference[va]),TD_metrics=metric(y[va],tdpred[va]),
                    K=groupmeta['K'],identity_reused=reused,training_residual=residual,numerically_qualified=valid))
            assert sorted(coverage)==list(range(len(ids))) and np.isfinite(pred).all()
            part=[r for r in rowsout if r['setting']==setting and r['source']==source];assert [r['sample_id'] for r in part]==ids
            for key,value in (('label',y),('prediction',pred),('reference_prediction',reference),('TD_prediction',tdpred)):
                close([float(r[key]) for r in part],value,0)
            counts['rows']+=len(part);m,rr,tm=metric(y,pred),metric(y,reference),metric(y,tdpred)
            rec=next(r for r in d['records'] if r['setting']==setting and r['source']==source);assert rec['n']==len(ids)
            for field,mm in (('metrics',m),('reference',rr),('TD_metrics',tm)):compare_mapping(rec[field],mm)
            delta=m['PCC']-rr['PCC'];reduction=100*(1-m['MSE']/rr['MSE'])
            close(rec['delta_PCC_vs_reference'],delta,1e-11);close(rec['MSE_reduction_vs_reference_pct'],reduction,1e-10)
            assert len(rec['inner_blocks'])==5
            for actual,expected in zip(rec['inner_blocks'],blocks):compare_mapping(actual,expected)
            records.append(dict(setting=setting,source=source,metrics=m,delta=delta,reduction=reduction))
    assert counts['new_source']+counts['new_TD']+counts['identity_reused']==60
    assert counts['old_source']==45 and counts['old_TD']==15 and counts['partitions']==15 and counts['rows']==len(rowsout)==9196
    assert d['condition_heads']==60 and d['new_source_heads']==counts['new_source'] and d['new_TD_heads']==counts['new_TD'] and d['identity_reused_heads']==counts['identity_reused']
    assert d['new_training_partitions']==15 and d['total_internal_OOF_rows']==9196
    assert d['reused_reference_native_heads']==45 and d['reused_reference_TD_heads']==15 and d['reused_unique_TD_rows']==2299
    assert len(d['records'])==12 and len(d['pilot_files_sha256'])==210 and d['partitions']==partitions
    core=[r for r in records if r['source']=='core_real']
    performance=sum(r['delta']>=.005 and r['reduction']>=0 for r in core)>=2 and all(r['delta']>=-.005 for r in core)
    pairs=[]
    for r in core:
        td=next(a for a in records if a['setting']==r['setting'] and a['source']=='TD_only')
        p=dict(setting=r['setting'],delta_PCC=r['metrics']['PCC']-td['metrics']['PCC'],MSE_reduction_pct=100*(1-r['metrics']['MSE']/td['metrics']['MSE']),delta_gain_above_TD=r['delta']-td['delta'])
        expected=next(x for x in d['paired_TD'] if x['setting']==r['setting']);compare_mapping(expected,p);pairs.append(p)
    increment=sum(p['delta_PCC']>=0 for p in pairs)>=2 and np.mean([p['MSE_reduction_pct'] for p in pairs])>=0
    qualified=bool(performance and increment and numeric)
    assert d['qualified']==qualified and d['performance_gate']==performance and d['source_increment_gate']==increment and d['all_numerically_qualified']==numeric
    result=dict(passed=True,qualified=qualified,all_numerically_qualified=bool(numeric),counts=counts,
        independent_input_graphs=15,independent_condition_full_joint_certificates=45,independent_old_full_joint_certificates=45,
        independent_condition_TD_augmented_lstsq=15,old_TD_normal_equations=15,max_normal_error=maxnormal,max_relative_dual_gap=maxgap,
        max_independent_TD_lstsq_difference=maxlstsq,max_prediction_difference=maxpred,max_deployment_difference=maxdeploy,
        GPU_hours=0,new_source_forward=0,new_PCA=0,new_original_scalers=0,outer_test_metrics_used=False,
        lock_sha256=sha(R/'lock.json'),diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-start)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:
            json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_source_forward=0),f,indent=2)
