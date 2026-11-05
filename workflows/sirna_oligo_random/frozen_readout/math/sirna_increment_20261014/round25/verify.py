"""Independent full-system trace-norm certificates, frozen controls and metrics."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4';D=R.parent/'round21'
sp=importlib.util.spec_from_file_location('_r25_audit4',P/'verify.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
read,sha,table,check,close=old.read,old.sha,old.table,old.check,old.close


def metric(y,p):return dict(PCC=float(np.corrcoef(y,p)[0,1]),MSE=float(np.mean((y-p)**2)))


def main():
    started=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for name,h in lock['dependencies'].items():assert sha(Path(name))==h,name
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json') and not d['outer_test_metrics_used']
    assert d['new_source_forward']==d['new_PCA_fits']==d['GPU_hours']==0 and all(t['num_threads']==1 for t in d['threadpool'])
    check(R/'pilot',d['pilot_files_sha256']);assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    executions=[read(p) for p in R.glob('execution_*.json')];assert len(executions)==1 and executions[0]['returncode']==0
    plan=read(P/'plan.json');details=table(R/'diagnostic_rows.csv');records=[];count=covered=0;allvalid=True
    maxpred=maxgap=maxnormal=0.;ranks=[]
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids']
        assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids'])
        rows={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')};y=np.array([float(rows[s]['label']) for s in ids])
        td=np.array([[float(v) for v in rows[s]['td'].split(',')] for s in ids])
        for source in old.SOURCES:
            export=old.EXPORT/source;identity=read(export/'complete.json');assert d['source_identities'][source]==identity['source_identity']
            meta=table(export/'row_identity.csv');lookup={r['sample_id']:i for i,r in enumerate(meta)}
            for s in ids:assert all(rows[s][k]==meta[lookup[s]][k] for k in ('sample_id','cohort','guide','target57'))
            raw=np.c_[td,np.load(export/'source.npy',allow_pickle=False)[[lookup[s] for s in ids]].astype(float)]
            refs=read(P/'results/folds'/setting/'fold0'/source/'transforms.json')
            with np.load(P/'results/folds'/setting/'fold0'/source/'arrays.npz',allow_pickle=False) as a:ref=a['inner_oof'][:,18].copy()
            pred=np.full(len(ids),np.nan);dense=np.full(len(ids),np.nan);cover=[];branks=[];bvalid=[];bsteps=[]
            for j,split in enumerate(item['inner']):
                tr,va=np.array(split['train']),np.array(split['valid']);out=R/'pilot'/setting/source/f'inner{j}'
                c=read(out/'complete.json');check(out,c['files_sha256']);assert c['lock_sha256']==sha(R/'lock.json')
                s=read(out/'state.json');cert=s['certificate'];assert s['train_indices']==tr.tolist() and s['valid_indices']==va.tolist()
                assert (s['setting'],s['source'],s['inner'])==(setting,source,j)
                control=D/'pilot'/setting/source/'dense'/f'inner{j}';cc=read(control/'complete.json');check(control,cc['files_sha256'])
                assert cc['lock_sha256']==sha(D/'lock.json') and read(control/'state.json')['numerically_qualified']
                assert s['transform_sha256']==sha(control/'transform.json') and s['dense_model_sha256']==sha(control/'model.npz') and s['dense_complete_sha256']==sha(control/'complete.json')
                st=read(control/'transform.json');old.project(raw[tr],raw[tr],refs['inner'][j])
                for k in st:close(st[k],refs['inner'][j][k],0)
                mean=np.r_[st['td_mean'],st['src_mean']];scale=np.r_[st['td_scale'],np.array(st['src_scale'])*np.sqrt(12.)]
                z=(raw[tr]-mean)/scale;zv=(raw[va]-mean)/scale;close(z.mean(0),np.zeros(216),1e-10)
                with np.load(out/'model.npz',allow_pickle=False) as a:
                    b=a['beta'].copy();center=float(a['intercept']);pp=a['prediction'].copy();dc=a['deployment_coef'].copy();di=float(a['deployment_intercept'])
                with np.load(control/'model.npz',allow_pickle=False) as a:
                    db=a['beta'].copy();dcenter=float(a['intercept']);dp=a['prediction'].copy()
                close(center,y[tr].mean(),1e-12);close(dcenter,center,0)
                g=z.T@z+100*np.eye(216);rhs=z.T@(y[tr]-center)
                close(g@db,rhs,1e-8);close(zv@db+dcenter,dp,1e-10)
                td0=np.linalg.solve(g[:24,:24],rhs[:24]);coupling=np.linalg.solve(g[:24,:24],g[:24,24:])
                h=g[24:,24:]-g[24:,:24]@coupling;h=(h+h.T)/2;q=rhs[24:]-g[24:,:24]@td0
                critical=float(np.linalg.norm(q.reshape(6,32),2));lam=.1*critical
                close(cert['lambda_critical'],critical,1e-10);close(cert['lambda_value'],lam,1e-10)
                assert cert['alpha']==100 and cert['lambda_ratio']==.1 and cert['matrix_shape']==[6,32] and cert['mode']=='native_trace'
                normal=float(abs((g@b-rhs)[:24]).max());assert normal<1e-8;maxnormal=max(maxnormal,normal)
                singular=np.linalg.svd(b[24:].reshape(6,32),compute_uv=False);rank=int(sum(singular>1e-8));close(cert['singular_values'],singular,1e-11);assert cert['rank']==rank
                dual=q-h@b[24:];norm=float(np.linalg.norm(dual.reshape(6,32),2))
                if norm>lam:dual*=lam/norm
                close(cert['dual'],dual,1e-10);assert np.linalg.norm(dual.reshape(6,32),2)<=lam+1e-10
                primal=.5*b@g@b-rhs@b+lam*singular.sum();dv=np.r_[np.zeros(24),dual]
                dual_value=-.5*(rhs-dv)@np.linalg.solve(g,rhs-dv);gap=float(primal-dual_value)
                constant=.5*rhs[:24]@td0;close(cert['primal_profile'],primal+constant,1e-10);close(cert['dual_profile'],dual_value+constant,1e-10)
                scale_gap=max(1.,.5*q@np.linalg.solve(h,q));close(cert['gap_scale'],scale_gap,1e-10)
                assert gap>=-1e-8;close(cert['gap'],max(0.,gap),1e-10)
                assert cert['certified']==(cert['gap']<=1e-9*cert['gap_scale'])
                valid=bool(cert['certified'] and gap<=1e-9*scale_gap+1e-11);assert valid==s['numerically_qualified'];allvalid &= valid
                assert 0<cert['iterations']<=5000
                if not cert['certified']:assert cert['iterations']==5000
                assert int(cert['history'][-1][0])==cert['iterations'];close(cert['history'][-1][1],cert['gap'],1e-12)
                close(cert['train_MSE'],np.mean((z@b+center-y[tr])**2),1e-12)
                reconstruction=zv@b+center;close(pp,reconstruction,1e-10);maxpred=max(maxpred,float(abs(pp-reconstruction).max()))
                close(dc,b/scale,1e-12);close(di,center-mean@(b/scale),1e-12);close(raw[va]@dc+di,pp,1e-9)
                pred[va]=pp;dense[va]=dp;cover+=va.tolist();count+=1;maxgap=max(maxgap,abs(gap)/scale_gap)
                branks.append(rank);bvalid.append(valid);bsteps.append(cert['iterations']);ranks.append(dict(setting=setting,source=source,inner=j,rank=rank,iterations=cert['iterations'],relative_gap=gap/scale_gap))
            assert sorted(cover)==list(range(len(ids))) and np.isfinite(pred).all() and np.isfinite(dense).all()
            part=[r for r in details if r['setting']==setting and r['source']==source];assert [r['sample_id'] for r in part]==ids
            for k,values in [('label',y),('prediction',pred),('dense_prediction',dense),('reference_prediction',ref)]:close([float(r[k]) for r in part],values,0)
            covered+=len(part);m,dm,rr=metric(y,pred),metric(y,dense),metric(y,ref)
            rec=next(r for r in d['records'] if r['setting']==setting and r['source']==source);assert rec['n']==len(ids)
            for name,expected in [('metrics',m),('dense_metrics',dm),('reference',rr)]:
                for k in expected:close(rec[name][k],expected[k],1e-11)
            delta=m['PCC']-rr['PCC'];red=100*(1-m['MSE']/rr['MSE']);close(rec['delta_PCC_vs_R4'],delta,1e-11);close(rec['MSE_reduction_vs_R4_pct'],red,1e-10)
            for j,split in enumerate(item['inner']):
                va=np.array(split['valid']);bl=rec['inner_blocks'][j]
                assert bl['inner']==j and bl['n']==len(va) and bl['rank']==branks[j] and bl['iterations']==bsteps[j] and bl['numerically_qualified']==bvalid[j]
                for name,expected in [('metrics',metric(y[va],pred[va])),('dense_metrics',metric(y[va],dense[va])),('reference',metric(y[va],ref[va]))]:
                    for k in expected:close(bl[name][k],expected[k],1e-11)
            records.append(dict(setting=setting,source=source,delta=delta,reduction=red,metrics=m,dense_metrics=dm))
    assert count==45==d['new_efficiency_heads']==d['reused_dense_heads'] and covered==6897==len(details)==d['new_internal_OOF_rows']==d['reused_dense_internal_OOF_rows']
    core=[r for r in records if r['source']=='core_real'];perf=sum(r['delta']>=.005 and r['reduction']>=0 for r in core)>=2 and all(r['delta']>=-.005 for r in core)
    paired=[]
    for r in core:
        dp=r['metrics']['PCC']-r['dense_metrics']['PCC'];red=100*(1-r['metrics']['MSE']/r['dense_metrics']['MSE']);paired.append(red)
        a=next(v for v in d['paired_core_records'] if v['setting']==r['setting']);close(a['delta_PCC'],dp,1e-11);close(a['MSE_reduction_pct'],red,1e-10)
    pg=sum(v>=0 for v in paired)>=2 and np.mean(paired)>=0;qualified=bool(perf and pg and allvalid)
    assert d['performance_gate']==perf and d['paired_gate']==pg and d['qualified']==qualified and d['all_numerically_qualified']==allvalid
    result=dict(passed=True,qualified=qualified,verified_new_heads=count,reused_dense_heads=45,new_internal_OOF_rows=covered,reused_dense_internal_OOF_rows=6897,
        independent_full_joint_certificates=45,no_new_dense_scientific_fit=True,max_prediction_difference=maxpred,max_relative_dual_gap=maxgap,
        max_TD_normal_equation_error=maxnormal,all_numerically_qualified=bool(allvalid),rank_records=ranks,
        new_source_forward=0,new_PCA_fits=0,GPU_hours=0,outer_test_metrics_used=False,
        lock_sha256=sha(R/'lock.json'),diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-started)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rank_records'}),flush=True)


if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_source_forward=0),f,indent=2)
