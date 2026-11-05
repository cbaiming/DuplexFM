"""Independent scalar sequence rules, Euclidean ordering and response arithmetic."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;B=R.parent;P=B/'round4'
spec=importlib.util.spec_from_file_location('_r46_independent4',P/'verify.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read,sha,table,check=old.read,old.sha,old.table,old.check
CONDITIONS=(*old.SOURCES,'TD_only')


def close(a,b,tol=1e-10):np.testing.assert_allclose(a,b,rtol=tol,atol=tol,equal_nan=True)


def direct_eligibility(rows):
    rc=lambda s:s.translate(str.maketrans('ACGU','UGCA'))[::-1]
    tokens=[{min(r['target57'][p:p+25],rc(r['target57'][p:p+25])) for p in range(33)} for r in rows]
    n=len(rows);e=np.zeros((n,n),bool)
    for i in range(n):
        assert len(rows[i]['guide'])==19 and len(rows[i]['target57'])==57
        for j in range(i):
            allowed=sum(a!=b for a,b in zip(rows[i]['guide'],rows[j]['guide']))>2 and not (tokens[i]&tokens[j])
            e[i,j]=e[j,i]=allowed
    return e


def main(start):
    assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for p,h in lock['dependencies'].items():assert sha(Path(p))==h,p
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json')
    check(R/'graph_audit',d['graph_files_sha256']);assert d['rows_sha256']==sha(R/'diagnostic_rows.csv')
    details=table(R/'diagnostic_rows.csv');dr={(r['setting'],r['condition'],r['sample_id']):r for r in details}
    assert len(dr)==len(details)==9196
    plan=read(P/'plan.json');allrows=sum([table(old.BENCH/'data'/f'{s}.csv') for s in old.S[:3]],[])
    index={r['sample_id']:i for i,r in enumerate(allrows)}
    td=np.array([[float(v) for v in r['td'].split(',')] for r in allrows]);raw={}
    for source in old.SOURCES:
        folder=old.EXPORT/source;receipt=read(folder/'complete.json');check(folder,receipt['files_sha256'])
        assert d['source_identities'][source]==receipt['source_identity']
        metas=table(folder/'row_identity.csv');mi={r['sample_id']:i for i,r in enumerate(metas)}
        for r in allrows:assert all(r[k]==metas[mi[r['sample_id']]][k] for k in ('sample_id','guide','target57','cohort'))
        raw[source]=np.load(folder/'source.npy',allow_pickle=False)[[mi[r['sample_id']] for r in allrows]].astype(float)
    counts=dict(graphs=0,masks=0,rows=0,TD_models_checked=0,source_scales_checked=0);records=[]
    max_normal=0.;max_prediction=0.
    for setting,n_expected in zip(old.S[:3],(1564,319,416)):
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        ids=item['fit_ids'];assert d['fit_ids'][setting]==ids and len(ids)==n_expected and set(ids).isdisjoint(item['test_ids'])
        fi=np.array([index[s] for s in ids]);rows=[allrows[i] for i in fi]
        y=np.array([float(r['label']) for r in rows]);path=P/'results/folds'/setting/'fold0/core_real/arrays.npz'
        with np.load(path,allow_pickle=False) as a:
            tdpred=a['inner_oof'][:,3].copy();tdcoef=a['inner_coef'][:,3,:].copy();tdintercept=a['inner_intercept'][:,3].copy()
        blocks={c:[] for c in CONDITIONS};coverage=[]
        for j,sp in enumerate(item['inner']):
            if time.monotonic()-start>900:raise TimeoutError('Independent CPU audit cap')
            tr,va=np.array(sp['train']),np.array(sp['valid']);coverage+=sp['valid']
            assert set(tr).isdisjoint(va)
            vi=[ids[i] for i in va];yy=y[va];e=direct_eligibility([rows[i] for i in va])
            ep=R/'graph_audit'/setting/'eligibility'/f'inner{j}.npz'
            with np.load(ep,allow_pickle=False) as a:np.testing.assert_array_equal(e,a['eligible'])
            counts['masks']+=1
            tm,ts=old.scale(td[fi[tr]],False);zt=(td[fi[tr]]-tm)/ts;zv=(td[fi[va]]-tm)/ts
            beta=tdcoef[j,:24];assert np.count_nonzero(tdcoef[j,24:])==0
            normal=np.max(abs((zt.T@zt+100*np.eye(24))@beta-zt.T@(y[tr]-y[tr].mean())))/len(tr)
            max_normal=max(max_normal,float(normal));assert normal<1e-8
            close(tdintercept[j],y[tr].mean(),1e-12);replayed=zv@beta+tdintercept[j]
            max_prediction=max(max_prediction,float(max(abs(replayed-tdpred[va]))));close(replayed,tdpred[va])
            counts['TD_models_checked']+=1;r=yy-tdpred[va]
            for condition in CONDITIONS:
                source='core_real' if condition=='TD_only' else condition
                out=R/'graph_audit'/setting/condition/f'inner{j}';receipt=read(out/'complete.json');check(out,receipt['files_sha256'])
                assert receipt['lock_sha256']==sha(R/'lock.json');state=read(out/'state.json')
                tp=B/'round21/pilot'/setting/source/'dense'/f'inner{j}'/'transform.json';t=read(tp)
                assert state['transform_path']==str(tp.resolve()) and state['transform_sha256']==sha(tp)
                assert state['eligibility_path']==str(ep.resolve()) and state['eligibility_sha256']==sha(ep)
                assert state['TD_OOF_path']==str(path.resolve()) and state['TD_OOF_sha256']==sha(path)
                assert state['train_indices']==sp['train'] and state['valid_indices']==sp['valid'] and state['sample_ids']==vi
                assert state['setting']==setting and state['condition']==condition and state['inner']==j and state['n']==len(va)
                assert state['input_only_graph'] and state['uses_known_fit_responses_for_diagnostic'] and not state['deployable'] and not state['new_predictions']
                close(t['td_mean'],tm);close(t['td_scale'],ts)
                if condition=='TD_only':z=zv
                else:
                    sm,ss=old.scale(raw[source][fi[tr]],True);close(t['src_mean'],sm);close(t['src_scale'],ss)
                    z=(raw[source][fi[va]]-sm)/ss/np.sqrt(12);counts['source_scales_checked']+=1
                neighbors=np.full((len(va),10),-1,dtype=int)
                for i in range(len(va)):
                    distances=np.sum((z-z[i])**2,axis=1)
                    ordered=sorted((k for k in range(len(va)) if e[i,k]),key=lambda k:(float(distances[k]),vi[k]))[:10]
                    neighbors[i,:len(ordered)]=ordered
                with np.load(out/'graph.npz',allow_pickle=False) as a:arrays={key:a[key].copy() for key in a.files}
                np.testing.assert_array_equal(neighbors,arrays['neighbors']);np.testing.assert_array_equal(e.sum(1),arrays['eligible_count'])
                value_states={}
                for name,v in (('residual',r),('response',yy)):
                    kn=np.full(len(va),np.nan);nu=kn.copy()
                    for i in range(len(va)):
                        js=[k for k in neighbors[i] if k>=0];allowed=[k for k in range(len(va)) if e[i,k]]
                        assert len(js)==min(10,len(allowed))
                        if js:
                            kn[i]=sum((float(v[i])-float(v[k]))**2 for k in js)/len(js)
                            nu[i]=sum((float(v[i])-float(v[k]))**2 for k in allowed)/len(allowed)
                    close(arrays[name+'_knn'],kn);close(arrays[name+'_null'],nu)
                    active=e.sum(1)>0;g5=e.sum(1)>=5;n0=float(np.nansum(nu));nk=float(np.nansum(kn))
                    expected=dict(n=len(va),active_rows=int(active.sum()),rows_ge5=int(g5.sum()),coverage=float(active.mean()),
                        coverage_ge5=float(g5.mean()),knn_sum=nk,null_sum=n0,ratio=nk/n0 if n0>1e-14 else None)
                    for key,value in expected.items():
                        if value is None:assert state[name][key] is None
                        else:close(state[name][key],value)
                    value_states[name]=expected
                for i,s in enumerate(vi):
                    row=dr[setting,condition,s];assert int(row['inner'])==j and int(row['eligible_count'])==int(e[i].sum())
                    assert row['neighbors']==';'.join(vi[k] for k in neighbors[i] if k>=0)
                    assert int(row['neighbor_count'])==int((neighbors[i]>=0).sum())
                    for key,value in (('label',yy[i]),('TD_OOF',tdpred[va][i]),('residual',r[i])):close(float(row[key]),value)
                    for key in ('residual_knn','residual_null','response_knn','response_null'):
                        value=arrays[key][i]
                        if np.isnan(value):assert row[key]==''
                        else:close(float(row[key]),value)
                block=dict(inner=j,n=len(va),**value_states);blocks[condition].append(block)
                counts['graphs']+=1;counts['rows']+=len(va)
        assert sorted(coverage)==list(range(n_expected))
        for condition in CONDITIONS:
            rec=next(v for v in d['records'] if v['setting']==setting and v['condition']==condition)
            agg=dict(n=n_expected,blocks_below_null=sum(v['residual']['ratio'] is not None and v['residual']['ratio']<1 for v in blocks[condition]),
                coverage_qualified=all(v['residual']['coverage']>=.9 and v['residual']['coverage_ge5']>=.9 for v in blocks[condition]))
            for name in ('residual','response'):
                active=sum(v[name]['active_rows'] for v in blocks[condition]);k=sum(v[name]['knn_sum'] for v in blocks[condition]);n=sum(v[name]['null_sum'] for v in blocks[condition])
                agg[name]=dict(active_rows=active,knn_sum=k,null_sum=n,knn_mean=k/active if active else None,
                    null_mean=n/active if active else None,ratio=k/n if n>1e-14 else None)
                for key,value in agg[name].items():
                    if value is None:assert rec['aggregate'][name][key] is None
                    else:close(rec['aggregate'][name][key],value)
            for key in ('n','blocks_below_null','coverage_qualified'):assert rec['aggregate'][key]==agg[key]
            assert len(rec['inner_blocks'])==5
            for actual,expected in zip(rec['inner_blocks'],blocks[condition]):
                assert actual['inner']==expected['inner'] and actual['n']==expected['n']
                for name in ('residual','response'):
                    for key,value in expected[name].items():
                        if value is None:assert actual[name][key] is None
                        else:close(actual[name][key],value)
            records.append(dict(setting=setting,condition=condition,aggregate=agg))
    assert counts==dict(graphs=60,masks=15,rows=9196,TD_models_checked=15,source_scales_checked=45)
    decisions=[]
    for c in (x for x in records if x['condition']=='core_real'):
        tdrec=next(x for x in records if x['setting']==c['setting'] and x['condition']=='TD_only')
        a,b=c['aggregate']['residual']['ratio'],tdrec['aggregate']['residual']['ratio'];defined=a is not None and b is not None
        passed=defined and a<=.90 and a<=b-.05 and c['aggregate']['blocks_below_null']>=4 and c['aggregate']['coverage_qualified']
        exp=next(x for x in d['per_pool'] if x['setting']==c['setting'])
        assert exp['passed']==bool(passed)
        for key,value in (('Core_ratio',a),('TD_ratio',b),('Core_minus_TD',None if not defined else a-b)):
            if value is None:assert exp[key] is None
            else:close(exp[key],value)
        decisions.append(bool(passed))
    coverage=all(x['aggregate']['coverage_qualified'] for x in records);qualified=coverage and sum(decisions)>=2
    assert d['coverage_qualified']==coverage and d['qualified']==qualified
    assert d['qualification_only'] and not d['goal_achieved'] and not d['deployable'] and not d['outer_test_metrics_used']
    assert d['new_graphs']==60 and d['eligibility_masks']==15 and d['condition_rows']==9196 and d['reused_unique_TD_heads']==15
    assert d['new_efficiency_heads']==d['new_OOF_rows']==d['new_source_forward']==d['new_PCA_fits']==d['GPU_hours']==0
    result=dict(passed=True,qualified=bool(qualified),counts=counts,max_TD_normal_equation_error=max_normal,max_TD_prediction_error=max_prediction,
        input_only_graphs_checked=60,all_edges_within_same_inner_validation=True,all_sequence_exclusions_verified=True,
        new_source_forward=0,GPU_hours=0,new_OOF_rows=0,new_efficiency_heads=0,outer_test_metrics_used=False,
        lock_sha256=sha(R/'lock.json'),diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-start)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main(start);rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:
            json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_source_forward=0),f,indent=2)
