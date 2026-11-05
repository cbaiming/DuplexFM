"""Independent scalar hypot / centered sums / old PCA equation verification."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,math,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent; P=R.parent/'round4'
spec=importlib.util.spec_from_file_location('_r15_audit4',P/'verify.py'); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
read,sha,table,check,close=old.read,old.sha,old.table,old.check,old.close


def moments(norm,pc):
    n=len(norm); mean=math.fsum(norm)/n; pm=math.fsum(pc)/n
    a=[v-mean for v in norm]; b=[v-pm for v in pc]
    sa=math.fsum(v*v for v in a); sb=math.fsum(v*v for v in b)
    denom=math.sqrt(sa*sb); corr=math.fsum(v*w for v,w in zip(a,b))/denom if denom>1e-14 else 0.
    sd=math.sqrt(sa/n); cv=sd/mean if mean>1e-12 else 0.
    ordered=sorted(norm); quantiles=[]
    for q in (.05,.5,.95):
        p=(n-1)*q; lo=math.floor(p); hi=math.ceil(p)
        quantiles.append(ordered[lo]+(ordered[hi]-ordered[lo])*(p-lo))
    return dict(norm_min=min(norm),norm_max=max(norm),norm_mean=mean,norm_sd=sd,norm_quantiles=quantiles,
        norm_CV=cv,norm_PC1_PCC=corr,norm_PC1_r2=corr*corr,
        pool_qualified=(cv>=.1 and corr*corr>=.2),all_nonzero=all(v>1e-12 for v in norm))


def main():
    started=time.monotonic(); assert not (R/'geometry.mutex').exists() and not (R/'geometry_verification.json').exists()
    lock=read(R/'geometry_lock.json'); check(R,lock['files'])
    for path,h in lock['dependencies'].items(): assert sha(Path(path))==h,path
    d=read(R/'geometry_result.json'); assert d['lock_sha256']==sha(R/'geometry_lock.json') and len(d['records'])==9
    assert not d['labels_numerically_used'] and not d['outer_test_metrics_used']
    assert d['new_efficiency_fits']==d['new_PCA_fits']==d['new_source_forward']==d['GPU_hours']==0
    ex=[read(p) for p in R.glob('geometry_execution_*.json')]; assert len(ex)==1 and ex[0]['returncode']==0
    plan=read(P/'plan.json'); records=[]; count=0
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0); ids=item['fit_ids']
        assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids'])
        lookup={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')}; rows=[lookup[i] for i in ids]
        td=np.array([[float(v) for v in r['td'].split(',')] for r in rows])
        for source in old.SOURCES:
            folder=old.EXPORT/source; meta=table(folder/'row_identity.csv'); index={r['sample_id']:i for i,r in enumerate(meta)}
            for r in rows: assert all(r[k]==meta[index[r['sample_id']]][k] for k in ('sample_id','cohort','guide','target57'))
            src=np.load(folder/'source.npy',allow_pickle=False)[[index[i] for i in ids]].astype(float); x=np.c_[td,src]
            t=read(P/'results/folds'/setting/'fold0'/source/'transforms.json')['final']
            pc=old.project(x,x,t)[:,24]; norm=[math.hypot(*row) for row in src]
            values=moments(norm,pc); record=next(r for r in d['records'] if r['setting']==setting and r['source']==source)
            assert record['n']==len(ids)
            for key,v in values.items():
                if isinstance(v,bool): assert record[key]==v
                else: close(v,record[key],1e-11)
            assert d['identities'][source]==read(folder/'complete.json')['source_identity']
            records.append(dict(source=source,**values)); count+=len(src)
    core=[r for r in records if r['source']=='core_real']
    qualified=sum(r['pool_qualified'] for r in core)>=2 and all(r['all_nonzero'] for r in core)
    assert qualified==d['qualified']
    result=dict(passed=True,qualified=qualified,independent_input_maps=9,verified_source_rows=count,
        PCA_equation_checks=9,scalar_hypot_and_fsum=True,labels_numerically_used=False,
        new_efficiency_fits=0,new_source_forward=0,new_PCA_fits=0,GPU_hours=0,
        result_sha256=sha(R/'geometry_result.json'),lock_sha256=sha(R/'geometry_lock.json'),
        CPU_wall_seconds=time.monotonic()-started)
    with (R/'geometry_verification.json').open('x',encoding='utf-8') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    start=time.monotonic(); rc=1; error=None
    try: main(); rc=0
    except Exception: error=traceback.format_exc(); raise
    finally:
        with (R/f'geometry_verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:
            json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0),f,indent=2)
