"""Independent scalar arithmetic readback; no predictor/label numeric conversion."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,hashlib,json,math,statistics,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;B=R.parents[1]
BENCH=B/'core_six_settings_20261011/benchmark';EXPORT=B/'core_six_settings_20261011/results/exports'
SOURCES=('core_real','full_real','initial_core');SETTINGS=('Huesken','Mixset','Takayuki')


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def compare(a,b):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10)


def independently_inspect(x):
    n,p=x.shape;k=math.ceil(n/100);means=[];sd=[];nonconstant=[];shares=[];peaks=[];fourth=[]
    for j in range(p):
        v=[float(x[i,j]) for i in range(n)];mu=math.fsum(v)/n;d=[t-mu for t in v];sq=[t*t for t in d]
        total=math.fsum(sq);variance=total/n;scale=math.sqrt(variance);means.append(mu);sd.append(scale)
        nonconstant.append(max(v)>min(v));shares.append(math.fsum(sorted(sq,reverse=True)[:k])/total if total else 0.)
        peaks.append(max(abs(t) for t in d)/scale if scale else 0.)
        fourth.append((math.fsum(t*t for t in sq)/n)/(variance*variance) if variance else 0.)
    positive=[s for s,c in zip(sd,nonconstant) if c and s>0]
    floor=.01*(statistics.median(positive) if positive else 1.)
    eligible=[c and s>=floor for c,s in zip(nonconstant,sd)]
    flagged=[e and t>=.2 and m>=6 for e,t,m in zip(eligible,shares,peaks)]
    needed=max(16,math.ceil(.1*sum(eligible)));ev=[t for e,t in zip(eligible,shares) if e]
    return dict(n=n,columns=p,top_rows=k,mean=means,sd=sd,scale_floor=floor,eligible=eligible,
        top_fraction_squared_deviation=shares,max_abs_z=peaks,raw_kurtosis=fourth,flagged=flagged,
        eligible_columns=sum(eligible),flagged_columns=sum(flagged),required_columns=needed,pool_qualified=sum(flagged)>=needed,
        median_tail_share=statistics.median(ev) if ev else 0.,maximum_tail_share=max(ev) if ev else 0.,
        maximum_abs_z=max([m for e,m in zip(eligible,peaks) if e],default=0.))


def main():
    start=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');d=read(R/'diagnostic.json')
    for name,h in lock['files'].items():assert sha(R/name)==h,name
    for name,h in lock['dependencies'].items():assert sha(Path(name))==h,name
    assert lock['CPU_only'] and lock['source_seed']==42 and lock['CPU_limit_seconds']==300
    assert (lock['top_fraction'],lock['share_threshold'],lock['max_z_threshold'],lock['minimum_columns'],lock['minimum_fraction'],lock['required_core_pools'])==(.01,.2,6.,16,.1,2)
    assert d['lock_sha256']==sha(R/'lock.json')
    assert d['new_efficiency_heads']==d['new_source_forward']==d['new_PCA']==d['GPU_hours']==0
    assert not d['numeric_labels_read'] and not d['outer_test_metrics_used'] and all(t['num_threads']==1 for t in d['threadpool'])
    executions=[read(p) for p in R.glob('execution_*.json')];assert len(executions)==1 and executions[0]['returncode']==0
    plan=read(R.parent/'round4/plan.json');core_pools=[];blocks=columns=source_rows=unique_rows=0;maxerr=0.
    for setting in SETTINGS:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids']
        assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids']);unique_rows+=len(ids)
        rawrows={r['sample_id']:r for r in table(BENCH/'data'/f'{setting}.csv')}
        td=np.array([[float(t) for t in rawrows[s]['td'].split(',')] for s in ids])
        for source in (*SOURCES,'TD24'):
            if time.monotonic()-start>300:raise TimeoutError('Independent audit cap')
            if source=='TD24':
                x=td;record=next(r for r in d['TD_controls'] if r['setting']==setting)
            else:
                folder=EXPORT/source;complete=read(folder/'complete.json')
                assert complete['source_identity']==d['source_identities'][source]
                for name,h in complete['files_sha256'].items():assert sha(folder/name)==h,name
                rows=table(folder/'row_identity.csv');lookup={r['sample_id']:i for i,r in enumerate(rows)}
                for sample in ids:assert all(rawrows[sample][k]==rows[lookup[sample]][k] for k in ('sample_id','cohort','guide','target57'))
                x=np.load(folder/'source.npy',allow_pickle=False)[[lookup[s] for s in ids]].astype(float)
                record=next(r for r in d['records'] if r['setting']==setting and r['source']==source);source_rows+=len(ids)
            actual=independently_inspect(x);expected=record['statistics'];assert set(actual)==set(expected)
            for key,value in actual.items():
                if key in ('eligible','flagged','pool_qualified'):assert np.array_equal(value,expected[key]),key
                else:
                    compare(value,expected[key]);maxerr=max(maxerr,float(np.max(abs(np.asarray(value)-np.asarray(expected[key])))))
            if source=='core_real':core_pools.append(actual['pool_qualified'])
            blocks+=1;columns+=x.shape[1]
    assert blocks==12 and columns==1800 and source_rows==6897 and unique_rows==2299
    assert len(d['records'])==9 and len(d['TD_controls'])==3 and d['source_blocks']==9 and d['TD_blocks']==3
    assert d['source_row_occurrences']==6897 and d['unique_input_rows']==2299
    qualified=sum(core_pools)>=2;assert d['qualified']==qualified
    result=dict(passed=True,qualified=qualified,core_pool_qualification=core_pools,source_blocks=9,unique_TD_blocks=3,
        scalar_coordinate_checks=columns,source_row_occurrences=source_rows,unique_input_rows=unique_rows,
        max_absolute_statistic_difference=maxerr,numeric_labels_read=False,new_efficiency_heads=0,new_source_forward=0,new_PCA=0,
        GPU_hours=0,lock_sha256=sha(R/'lock.json'),diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-start)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:
            json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,new_efficiency_heads=0,GPU_hours=0),f,indent=2)
