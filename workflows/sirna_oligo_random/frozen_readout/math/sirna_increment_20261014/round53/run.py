"""One fixed inner-training pilot; no outer evaluation or adaptive grid."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, importlib.util, sys, time, traceback, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent; B=R.parent; P=B/'round4'; D=B/'round21'; A=B/'round49'
sys.path.insert(0,str(R))
from metric_scale import fit_scaling, apply_scaling, raw_deployment, fit, transform, qualify, prior
s=importlib.util.spec_from_file_location('_r53_old4',P/'run.py')
old=importlib.util.module_from_spec(s); s.loader.exec_module(old); io=old.io
CODE=('run.py','metric_scale.py','verify.py','close.py','test_metric_scale.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml')


def register():
    assert not (R/'lock.json').exists(); old.validate()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(x.get('tests',0)) for x in suites)==10
    assert not any(int(x.get(k,0)) for x in suites for k in ('errors','failures','skipped'))
    previous=io.read(B/'round52/closure_receipt.json'); assert previous['status']=='closed' and previous['readback_passed']
    io.check_files(B,previous['files_sha256'])
    lock=io.read(A/'lock.json'); io.check_files(A,lock['files']); deps=dict(lock['dependencies'])
    for name in lock['files']: deps[str((A/name).resolve())]=io.sha(A/name)
    for p in (A/'lock.json',A/'closure_receipt.json',A/'verification.json',A/'diagnostic.json',A/'diagnostic_rows.csv',
              A/'verify.py',A/'final_readback.json',B/'round52/closure_receipt.json',B/'round52/final_readback.json',Path(io.__file__)):
        deps[str(p.resolve())]=io.sha(p)
    assert io.read(A/'verification.json')['passed']
    for name,h in io.read(A/'diagnostic.json')['pilot_files_sha256'].items():
        path=A/'pilot'/name; assert io.sha(path)==h; deps[str(path.resolve())]=h
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in CODE},dependencies=deps,versions=io.versions(),
        CPU_only=True,CPU_limit_seconds=900,source_seed=42,primary='core_real',settings=old.SETTINGS[:3],sources=old.SOURCES,
        alpha=100.,lambda_ratio=.1,gamma=.5,blocks=[[0,24],[24,216]],block_energy_preserved=True,
        metric_scaling_no_strength_parameter=True,reference_round=49,identity_reuse=True,
        condition_heads=60,new_source_heads=39,new_TD_heads=13,identity_reused_heads=8,
        new_scaling_records=45,new_nonidentity_scalings=39,reused_training_partitions=15,total_internal_OOF_rows=9196,
        new_source_forward=0,new_PCA=0,new_original_scalers=0))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)


def validate():
    lock=io.read(R/'lock.json'); io.check_files(R,lock['files'])
    for path,h in lock['dependencies'].items(): assert io.sha(Path(path))==h,path
    assert lock['versions']==io.versions(); return lock


def stats(y,p):
    assert np.isfinite(p).all() and np.std(p)>1e-14 and np.std(y)>1e-14
    pc,mse=old.score(y,p[:,None]); return dict(PCC=float(pc[0]),MSE=float(mse[0]))


def run(start):
    validate(); root=R/'pilot'; assert not root.exists() and not (R/'diagnostic.json').exists()
    threads=threadpool_info(); assert all(t['num_threads']==1 for t in threads)
    rows,xx,identities=old.inputs(); lookup={r['sample_id']:i for i,r in enumerate(rows)}; plan=io.read(P/'plan.json')
    records=[]; details=[]; pools={}; scales=[]; numeric=True
    counts=dict(new_source=0,new_TD=0,identity_reused=0,scalings=0,nonidentity_scalings=0,rows=0)
    for setting in old.SETTINGS[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        ids=item['fit_ids']; pools[setting]=ids
        assert set(ids).isdisjoint(item['test_ids']) and set(ids).isdisjoint(item['validation_ids'])
        fi=np.array([lookup[s] for s in ids]); y=np.array([float(rows[i]['label']) for i in fi]); td_current=None
        for source in ('TD_only',)+tuple(old.SOURCES):
            is_source=source!='TD_only'; origin=source if is_source else 'core_real'; raw=xx[origin][fi]
            pred=np.full(len(ids),np.nan); ref=pred.copy(); coverage=[]; blocks=[]
            for j,sp in enumerate(item['inner']):
                if time.monotonic()-start>900: raise TimeoutError('CPU pilot cap')
                tr,va=np.array(sp['train']),np.array(sp['valid']); assert set(tr).isdisjoint(va)
                gp=A/'pilot/groups'/setting/f'inner{j}/groups.json'; group=io.read(gp)
                assert group['train_indices']==tr.tolist() and group['train_ids']==[ids[i] for i in tr]
                labels=np.array(group['labels']); reused=group['metadata']['identity_on_centered_space']
                stpath=D/'pilot'/setting/origin/'dense'/f'inner{j}/transform.json'; st=io.read(stpath); z=transform(raw,st)
                sf=root/'scales'/setting/origin/f'inner{j}'; scale_path=sf/'scaling.json'
                if not scale_path.exists():
                    factor,meta=fit_scaling(z[tr],labels)
                    io.write(scale_path,dict(setting=setting,source=origin,inner=j,train_indices=tr,train_ids=[ids[i] for i in tr],
                        factor=factor,metadata=meta,groups_sha256=io.sha(gp),transform_sha256=io.sha(stpath)))
                    io.write(sf/'complete.json',dict(lock_sha256=io.sha(R/'lock.json'),files_sha256={'scaling.json':io.sha(scale_path)}))
                    counts['scalings']+=1; counts['nonidentity_scalings']+=int(not reused)
                    scales.append(dict(setting=setting,source=origin,inner=j,**meta))
                else:
                    sr=io.read(sf/'complete.json'); io.check_files(sf,sr['files_sha256']); assert sr['lock_sha256']==io.sha(R/'lock.json')
                    saved=io.read(scale_path); factor=np.asarray(saved['factor']); meta=saved['metadata']
                assert meta['identity']==reused
                previous=A/'pilot'/setting/source/f'inner{j}'
                rc=io.read(previous/'complete.json'); io.check_files(previous,rc['files_sha256']); assert rc['lock_sha256']==io.sha(A/'lock.json')
                ns=io.read(previous/'state.json'); assert ns['numerically_qualified']
                assert ns['train_indices']==tr.tolist() and ns['valid_indices']==va.tolist()
                assert ns['transform_sha256']==io.sha(stpath) and ns['groups_sha256']==io.sha(gp)
                with np.load(previous/'model.npz',allow_pickle=False) as a:
                    oldbeta=a['beta'].copy(); oldcenter=float(a['intercept']); ref[va]=a['prediction']
                x=apply_scaling(z,factor); before=time.monotonic()
                if reused:
                    assert np.array_equal(factor,np.ones(216))
                    beta,center,cert=oldbeta,oldcenter,ns['certificate']; counts['identity_reused']+=1
                else:
                    beta,center,cert=fit(x[tr],y[tr],labels,is_source,deadline=start+900)
                    counts['new_source' if is_source else 'new_TD']+=1
                pp=(x[va] if is_source else x[va,:24])@beta+center
                if reused: np.testing.assert_allclose(pp,ref[va],rtol=1e-10,atol=1e-10)
                pred[va]=pp; coverage+=va.tolist(); valid=bool(cert['certified']); numeric &= valid
                coef,offset=raw_deployment(st,factor,beta,center,is_source)
                residual=prior.residual_stats(x[tr],y[tr],beta,center,labels,is_source)
                out=root/setting/source/f'inner{j}'
                old.npz_write(out/'model.npz',beta=beta,intercept=np.array(center),prediction=pp,deployment_coef=coef,deployment_intercept=np.array(offset))
                io.write(out/'state.json',dict(setting=setting,source=source,inner=j,train_indices=tr,valid_indices=va,
                    certificate=cert,numerically_qualified=valid,identity_reused=reused,training_residual=residual,
                    groups_sha256=io.sha(gp),transform_sha256=io.sha(stpath),scaling_sha256=io.sha(scale_path),
                    reference_model_sha256=io.sha(previous/'model.npz'),reference_state_sha256=io.sha(previous/'state.json'),
                    CPU_fit_or_reuse_seconds=time.monotonic()-before))
                io.write(out/'complete.json',dict(lock_sha256=io.sha(R/'lock.json'),files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
                blocks.append(dict(inner=j,n=len(va),metrics=stats(y[va],pp),reference=stats(y[va],ref[va]),
                    K=group['metadata']['K'],identity_reused=reused,training_residual=residual,numerically_qualified=valid))
            assert sorted(coverage)==list(range(len(ids))) and np.isfinite(pred).all() and np.isfinite(ref).all()
            if not is_source: td_current=pred.copy()
            assert td_current is not None
            m,rr=stats(y,pred),stats(y,ref)
            rec=dict(setting=setting,source=source,n=len(ids),metrics=m,reference=rr,TD_metrics=stats(y,td_current),
                delta_PCC_vs_reference=m['PCC']-rr['PCC'],MSE_reduction_vs_reference_pct=100*(1-m['MSE']/rr['MSE']),inner_blocks=blocks)
            records.append(rec)
            details.extend(dict(setting=setting,source=source,sample_id=s,label=float(y[k]),prediction=float(pred[k]),
                reference_prediction=float(ref[k]),TD_prediction=float(td_current[k])) for k,s in enumerate(ids))
            counts['rows']+=len(ids)
            print(setting,source,'deltaPCC',rec['delta_PCC_vs_reference'],'MSEred',rec['MSE_reduction_vs_reference_pct'],flush=True)
    assert counts==dict(new_source=39,new_TD=13,identity_reused=8,scalings=45,nonidentity_scalings=39,rows=9196)
    gates=qualify(records,numeric); io.write_csv(R/'diagnostic_rows.csv',details)
    io.write(R/'diagnostic.json',dict(**gates,records=records,fit_ids=pools,source_identities=identities,scalings=scales,counts=counts,
        condition_heads=60,new_training_partitions=0,reused_training_partitions=15,reused_reference_heads=60,
        new_source_forward=0,new_PCA=0,new_original_scalers=0,GPU_hours=0,outer_test_metrics_used=False,threadpool=threads,
        lock_sha256=io.sha(R/'lock.json'),diagnostic_rows_sha256=io.sha(R/'diagnostic_rows.csv'),
        pilot_files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    validate(); print('QUALIFIED',gates['qualified'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=('register','run'))
    if ap.parse_args().mode=='register': register()
    else:
        mutex=R/'execution.mutex'
        with mutex.open('x') as f: f.write(str(os.getpid()))
        start=time.monotonic(); rc=1; error=None
        try: run(start); rc=0
        except Exception: error=traceback.format_exc(); raise
        finally:
            io.write(R/f'execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_source_forward=0))
            mutex.unlink()
