"""Independent distance/contingency/gain audit, with 15 native gate replays."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback,itertools
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
R=Path(__file__).resolve().parent;P=R.parent/'round4'
sp=importlib.util.spec_from_file_location('_r23_verify4',P/'verify.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
read,sha,table,check,close=old.read,old.sha,old.table,old.check,old.close

def ari(a,b):
    n=len(a);mat=np.zeros((2,2),int)
    for x,y in zip(a,b):mat[x,y]+=1
    comb=lambda x:x*(x-1)/2
    pairs=comb(n);cell=float(comb(mat).sum());row=float(comb(mat.sum(1)).sum());col=float(comb(mat.sum(0)).sum())
    expected=row*col/pairs;den=.5*(row+col)-expected
    return (cell-expected)/den if den else 1.

def stats(y,pt,ps,assignment):
    total_td=float(np.mean(np.square(pt-y)));groups=[]
    def corr(x,p):
        if np.sqrt(np.sum((x-x.mean())**2)*np.sum((p-p.mean())**2))<=1e-14:return None
        return float(np.corrcoef(x,p)[0,1])
    for g in (0,1):
        ii=np.flatnonzero(assignment==g);n=len(ii)
        if n:
            tm=float(np.mean(np.square(y[ii]-pt[ii])));sm=float(np.mean(np.square(y[ii]-ps[ii])))
            mg=float(np.mean(np.square(y[ii]-pt[ii])-np.square(y[ii]-ps[ii])))
        else:tm=sm=mg=None
        groups.append(dict(route=g,n=n,fraction=n/len(y),mean_gain=mg,TD_MSE=tm,source_MSE=sm,
            TD_PCC=corr(y[ii],pt[ii]) if n>1 else None,source_PCC=corr(y[ii],ps[ii]) if n>1 else None))
    gap=groups[1]['mean_gain']-groups[0]['mean_gain'] if all(g['n'] for g in groups) else None
    return dict(n=len(y),TD_MSE=total_td,source_MSE=float(np.mean(np.square(y-ps))),groups=groups,gain_gap=gap,
        normalized_gain_gap=gap/total_td if gap is not None and total_td>0 else None)

def compare(a,b):
    if isinstance(b,dict):
        assert set(a)==set(b)
        for k in b:compare(a[k],b[k])
    elif isinstance(b,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    elif b is None:assert a is None
    elif isinstance(b,str):assert a==b
    else:close(a,b,1e-10)

def main():
    start=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for n,h in lock['dependencies'].items():assert sha(Path(n))==h,n
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json') and d['new_unsupervised_gates']==15
    assert d['new_efficiency_fits']==d['new_source_forward']==d['GPU_hours']==0 and not d['outer_test_metrics_used']
    check(R/'gates',d['gates_files_sha256']);assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256'] and sha(R/'route_rows.csv')==d['route_rows_sha256']
    assert all(t['num_threads']==1 for t in d['threadpool'])
    executions=[read(p) for p in R.glob('execution_*.json')];assert len(executions)==1 and executions[0]['returncode']==0
    plan=read(P/'plan.json');rowsout=table(R/'diagnostic_rows.csv');rr=table(R/'route_rows.csv');gatecount=0;coveragecount=0;sourcecount=0
    max_replay=0.;max_ari=0.;coreflags=[];diagnostics=[]
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids'];assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids'])
        data={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')}
        td=np.array([[float(v) for v in data[s]['td'].split(',')] for s in ids]);y=np.array([float(data[s]['label']) for s in ids])
        refpath=P/'results/folds'/setting/'fold0/core_real/transforms.json';ref=read(refpath);assignments=[];oof=np.full(len(ids),-1,int);ib=np.full(len(ids),-1,int)
        covered=[];balanced=True;converged=True;gblocks=[]
        for j,split in enumerate(item['inner']):
            tr,va=np.array(split['train']),np.array(split['valid']);covered+=va.tolist();out=R/'gates'/setting/f'inner{j}'
            c=read(out/'complete.json');assert c['lock_sha256']==sha(R/'lock.json');check(out,c['files_sha256']);g=read(out/'gate.json')
            assert g['train_indices']==tr.tolist() and g['valid_indices']==va.tolist() and g['inner']==j and g['reference_transform_sha256']==sha(refpath)
            mu=td[tr].mean(0);sd=td[tr].std(0);sd[sd<1e-12]=1.;close(g['TD_mean'],mu,1e-11);close(g['TD_scale'],sd,1e-11)
            close(ref['inner'][j]['td_mean'],mu,1e-11);close(ref['inner'][j]['td_scale'],sd,1e-11);z=(td-mu)/sd;centers=np.asarray(g['centers'])
            assert tuple(centers[0])<=tuple(centers[1])
            dd=np.stack([np.einsum('ij,ij->i',z-center,z-center) for center in centers],axis=1);a=np.argmin(dd,axis=1)
            with np.load(out/'assignments.npz',allow_pickle=False) as saved:
                np.testing.assert_array_equal(a,saved['all_fit_assignment']);np.testing.assert_array_equal(a[va],saved['OOF_assignment'])
            tc=np.bincount(a[tr],minlength=2);vc=np.bincount(a[va],minlength=2);np.testing.assert_array_equal(tc,g['training_counts']);np.testing.assert_array_equal(vc,g['valid_counts'])
            close(g['inertia'],dd[tr,a[tr]].sum(),1e-9)
            replay=KMeans(n_clusters=2,init='k-means++',n_init=10,max_iter=300,tol=1e-4,random_state=42,algorithm='lloyd').fit(z[tr])
            rc=replay.cluster_centers_;rc=rc[sorted(range(2),key=lambda k:tuple(rc[k]))];close(rc,centers,1e-11);max_replay=max(max_replay,float(abs(rc-centers).max()))
            assert replay.n_iter_==g['iterations'];distinct=np.linalg.norm(centers[0]-centers[1])>1e-10;assert g['distinct']==distinct
            bc=bool(min(tc)/len(tr)>=.15 and min(vc)/len(va)>=.15);conv=bool(g['iterations']<300 and distinct)
            assert bc==g['balanced'] and conv==g['converged'];balanced &= bc;converged &= conv
            gblocks.append(dict(inner=j,training_counts=tc.tolist(),valid_counts=vc.tolist(),balanced=bc,converged=conv))
            assignments.append(a);oof[va]=a[va];ib[va]=j;gatecount+=1
        assert sorted(covered)==list(range(len(ids))) and (oof>=0).all()
        pairs=[dict(first=a,second=b,ARI=float(ari(assignments[a],assignments[b]))) for a,b in itertools.combinations(range(5),2)]
        gr=next(g for g in d['gate_records'] if g['setting']==setting)
        expected=dict(setting=setting,balanced=bool(balanced),all_converged=bool(converged),median_ARI=float(np.median([p['ARI'] for p in pairs])),ARI_pairs=pairs,inner_blocks=gblocks)
        compare(gr,expected)
        for a,b in zip(gr['ARI_pairs'],pairs):max_ari=max(max_ari,abs(a['ARI']-b['ARI']))
        part=[r for r in rr if r['setting']==setting];assert [r['sample_id'] for r in part]==ids
        np.testing.assert_array_equal([int(r['OOF_route']) for r in part],oof);np.testing.assert_array_equal([int(r['inner_block']) for r in part],ib);coveragecount+=len(part)
        for source in old.SOURCES:
            with np.load(P/'results/folds'/setting/'fold0'/source/'arrays.npz',allow_pickle=False) as a:pt=a['inner_oof'][:,3].copy();ps=a['inner_oof'][:,18].copy()
            rec=next(r for r in d['records'] if r['setting']==setting and r['source']==source);s=stats(y,pt,ps,oof);compare(rec['summary'],s)
            bs=[]
            for j in range(5):
                use=ib==j;ss=stats(y[use],pt[use],ps[use],oof[use]);compare(rec['inner_blocks'][j],dict(inner=j,summary=ss));bs.append(ss)
            gap=s['normalized_gain_gap'];consistent=sum(b['normalized_gain_gap'] is not None and b['normalized_gain_gap']*gap>0 for b in bs) if gap is not None else 0
            valid=bool(gap is not None and balanced and converged and gr['median_ARI']>=.5 and abs(gap)>=.05 and consistent>=4)
            assert valid==rec['qualified']
            if source=='core_real':coreflags.append(valid)
            diagnostics.append(dict(setting=setting,source=source,consistent_inner_blocks=consistent,qualified=valid))
            rows=[r for r in rowsout if r['setting']==setting and r['source']==source];assert [r['sample_id'] for r in rows]==ids
            close([float(r['label']) for r in rows],y,0);close([float(r['TD_prediction']) for r in rows],pt,0);close([float(r['source_prediction']) for r in rows],ps,0)
            close([float(r['gain']) for r in rows],(y-pt)**2-(y-ps)**2,1e-12)
            np.testing.assert_array_equal([int(r['OOF_route']) for r in rows],oof);np.testing.assert_array_equal([int(r['inner_block']) for r in rows],ib);sourcecount+=len(rows)
    assert gatecount==15 and coveragecount==len(rr)==2299 and sourcecount==len(rowsout)==6897
    qualified=bool(sum(coreflags)>=2);assert qualified==d['qualified']
    result=dict(passed=True,qualified=qualified,verified_gates=gatecount,native_unsupervised_replays=15,verified_route_OOF_rows=coveragecount,
        verified_source_OOF_rows=sourcecount,new_efficiency_fits=0,new_source_forward=0,GPU_hours=0,outer_test_metrics_used=False,
        max_centroid_replay_difference=max_replay,max_ARI_difference=max_ari,diagnostics=diagnostics,
        lock_sha256=sha(R/'lock.json'),diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-start)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_efficiency_fits=0),f,indent=2)
