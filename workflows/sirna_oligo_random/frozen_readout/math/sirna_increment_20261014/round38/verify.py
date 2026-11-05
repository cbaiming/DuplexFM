"""Independent augmented-SVD audit of paired input cross-prediction."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4';D=R.parent/'round21'
s=importlib.util.spec_from_file_location('_r38_audit4',P/'verify.py');old=importlib.util.module_from_spec(s);s.loader.exec_module(old)
read,sha,table,check=old.read,old.sha,old.table,old.check
CONDITIONS=('paired_initial','shuffled_initial','TD24')


def close(a,b,tol=1e-9):np.testing.assert_allclose(a,b,rtol=tol,atol=tol)
def stats(f,p):
    blocks=[]
    for j in range(6):
        part=f[:,32*j:32*j+32];res=part-p[:,32*j:32*j+32];n=float(np.einsum('ij,ij->',part,part));e=float(np.einsum('ij,ij->',res,res))
        blocks.append(dict(SSE=e,null_SSE=n,R2=1-e/n if n>0 else None))
    n=sum(b['null_SSE'] for b in blocks);e=sum(b['SSE'] for b in blocks)
    return dict(SSE=e,null_SSE=n,R2=1-e/n,blocks=blocks)


def compare_stats(actual,expected):
    for k in ('SSE','null_SSE','R2'):close(actual[k],expected[k],1e-10)
    for aa,bb in zip(actual['blocks'],expected['blocks']):
        for k in ('SSE','null_SSE','R2'):
            if aa[k] is None or bb[k] is None:assert aa[k]==bb[k]
            else:close(aa[k],bb[k],1e-10)


def main():
    start=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for p,h in lock['dependencies'].items():assert sha(Path(p))==h,p
    assert lock['alpha']==100 and lock['new_maps']==45 and lock['new_efficiency_heads']==0 and not lock['numeric_labels_read']
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json') and not d['numeric_labels_read'] and not d['outer_test_metrics_used']
    assert d['new_source_forward']==d['new_PCA']==d['GPU_hours']==d['new_efficiency_heads']==0 and all(t['num_threads']==1 for t in d['threadpool'])
    check(R/'maps',d['map_files_sha256']);assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    runs=[read(p) for p in R.glob('execution_*.json')];assert len(runs)==1 and runs[0]['returncode']==0
    identities=d['source_identities'];assert identities['core_real']['initial_sha256']==identities['initial_core']['checkpoint_sha256']
    assert identities['core_real']['variant']==identities['initial_core']['variant']=='core_real'
    assert identities['core_real']['source_receipt_sha256']==identities['initial_core']['source_receipt_sha256']
    rowsout=table(R/'diagnostic_rows.csv');plan=read(P/'plan.json');qualified_pools=[];counts=dict(maps=0,rows=0,unique=0)
    maxcoef=maxpred=maxgram=maxnormal=0.
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids']
        assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids']);counts['unique']+=len(ids)
        rawrows={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')};td=np.array([[float(v) for v in rawrows[s]['td'].split(',')] for s in ids])
        sources={}
        for source in ('core_real','initial_core'):
            folder=old.EXPORT/source;complete=read(folder/'complete.json');assert complete['source_identity']==identities[source];check(folder,complete['files_sha256'])
            meta=table(folder/'row_identity.csv');idx={r['sample_id']:i for i,r in enumerate(meta)}
            for sid in ids:assert all(rawrows[sid][k]==meta[idx[sid]][k] for k in ('sample_id','cohort','guide','target57'))
            sources[source]=np.load(folder/'source.npy',allow_pickle=False)[[idx[sid] for sid in ids]].astype(float)
        record=next(x for x in d['records'] if x['setting']==setting);assert record['n']==len(ids)
        pooled={k:dict(SSE=0.,null_SSE=0.,blocks=[dict(SSE=0.,null_SSE=0.) for _ in range(6)]) for k in CONDITIONS};coverage={k:[] for k in CONDITIONS}
        for j,sp in enumerate(item['inner']):
            if time.monotonic()-start>900:raise TimeoutError('CPU audit cap')
            tr,va=np.array(sp['train']),np.array(sp['valid']);z={};paths={}
            for source in ('core_real','initial_core'):
                path=D/'pilot'/setting/source/'dense'/f'inner{j}'/'transform.json';st=read(path);paths[source]=path
                mean,scale=old.scale(sources[source][tr],True);close(st['src_mean'],mean,1e-11);close(st['src_scale'],scale,1e-11)
                z[source]=(sources[source]-mean)/scale/np.sqrt(12.)
                if source=='core_real':
                    tm,ts=old.scale(td[tr],False);close(st['td_mean'],tm,1e-11);close(st['td_scale'],ts,1e-11);tdz=(td-tm)/ts
            f,b=z['core_real'],z['initial_core'];foldrecord=next(x for x in record['inner_blocks'] if x['inner']==j);assert foldrecord['n']==len(va)
            codes=[hashlib.sha256(('r38:42:'+setting+':'+str(j)+':'+ids[i]).encode('utf-8')).hexdigest() for i in tr];pi=np.argsort(np.array(codes),kind='stable')
            for condition in CONDITIONS:
                out=R/'maps'/setting/condition/f'inner{j}';receipt=read(out/'complete.json');check(out,receipt['files_sha256']);assert receipt['lock_sha256']==sha(R/'lock.json')
                state=read(out/'state.json');assert (state['setting'],state['condition'],state['inner'])==(setting,condition,j)
                assert state['train_indices']==tr.tolist() and state['valid_indices']==va.tolist() and state['transform_sha256']=={k:sha(p) for k,p in paths.items()}
                x=tdz if condition=='TD24' else b;xt=x[tr][pi] if condition=='shuffled_initial' else x[tr];p=xt.shape[1]
                with np.load(out/'model.npz',allow_pickle=False) as a:m={k:a[k].copy() for k in a.files}
                expected_perm=pi if condition=='shuffled_initial' else np.arange(len(tr));assert np.array_equal(m['training_permutation'],expected_perm)
                assert state['shuffled_fixed_points']==(int(sum(pi==np.arange(len(tr)))) if condition=='shuffled_initial' else None)
                coef=np.linalg.lstsq(np.r_[xt,10*np.eye(p)],np.r_[f[tr],np.zeros((p,192))],rcond=None)[0]
                close(m['coefficient'],coef,1e-11);maxcoef=max(maxcoef,float(abs(m['coefficient']-coef).max()))
                saved=m['coefficient'];normal=float(abs((xt.T@xt+100*np.eye(p))@saved-xt.T@f[tr]).max());assert normal<1e-8
                cert=state['certificate'];assert cert['alpha']==100;close(cert['normal_error'],normal,1e-10);close(cert['train_SSE'],np.sum((xt@saved-f[tr])**2),1e-10)
                gram=saved.T@(xt.T@xt)@saved;close(m['training_prediction_gram'],gram,1e-9);assert np.linalg.eigvalsh(gram)[0]>=-1e-8
                pp=x[va]@saved;close(m['prediction'],pp,1e-12);close(m['prediction'],x[va]@coef,1e-10)
                maxpred=max(maxpred,float(abs(pp-m['prediction']).max()));maxnormal=max(maxnormal,normal);maxgram=max(maxgram,float(abs(gram-m['training_prediction_gram']).max()))
                actual=stats(f[va],m['prediction']);compare_stats(actual,state['statistics']);compare_stats(actual,foldrecord['conditions'][condition])
                for key in ('SSE','null_SSE'):pooled[condition][key]+=actual[key]
                for k in range(6):
                    for key in ('SSE','null_SSE'):pooled[condition]['blocks'][k][key]+=actual['blocks'][k][key]
                rr=[r for r in rowsout if r['setting']==setting and r['condition']==condition and int(r['inner'])==j]
                assert [r['sample_id'] for r in rr]==[ids[i] for i in va]
                close([float(r['SSE']) for r in rr],np.sum((f[va]-m['prediction'])**2,axis=1),1e-10)
                close([float(r['null_SSE']) for r in rr],np.sum(f[va]**2,axis=1),1e-10)
                counts['maps']+=1;counts['rows']+=len(rr);coverage[condition]+=va.tolist()
        for condition in CONDITIONS:
            assert sorted(coverage[condition])==list(range(len(ids)));x=pooled[condition];x['R2']=1-x['SSE']/x['null_SSE']
            for bb in x['blocks']:bb['R2']=1-bb['SSE']/bb['null_SSE'] if bb['null_SSE']>0 else None
            compare_stats(x,record['conditions'][condition])
        a,b,c=[pooled[k]['R2'] for k in CONDITIONS];qualified=bool(a>=.1 and a-c>=.05 and a-b>=.05)
        q=dict(setting=setting,R2=a,delta_vs_TD24=a-c,delta_vs_shuffled=a-b,qualified=qualified)
        expected=next(r for r in d['pool_qualification'] if r['setting']==setting)
        for k in ('R2','delta_vs_TD24','delta_vs_shuffled'):close(q[k],expected[k],1e-10)
        assert q['qualified']==expected['qualified'];qualified_pools.append(q)
    assert counts==dict(maps=45,rows=6897,unique=2299) and len(rowsout)==6897 and d['new_maps']==45 and d['vector_width']==192
    assert d['vector_OOF_rows']==6897 and d['unique_fit_rows']==2299
    qualified=bool(sum(q['qualified'] for q in qualified_pools)>=2 and all(q['R2']>=0 for q in qualified_pools));assert qualified==d['qualified']
    result=dict(passed=True,qualified=qualified,counts=counts,independent_augmented_SVD_solves=45,max_coefficient_difference=maxcoef,max_prediction_difference=maxpred,
        max_Gram_difference=maxgram,max_normal_error=maxnormal,numeric_labels_read=False,new_efficiency_heads=0,new_source_forward=0,new_PCA=0,GPU_hours=0,
        lock_sha256=sha(R/'lock.json'),diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-start)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_efficiency_heads=0,new_source_forward=0),f,indent=2)
