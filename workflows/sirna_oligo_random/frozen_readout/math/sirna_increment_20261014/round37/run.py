"""One registered native-energy information comparison; frozen CPU inputs only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;P=R.parent/'round4';D=R.parent/'round21';N=R.parent/'round25'
sys.path.insert(0,str(R))
from energy import fit_energy,transform,canonical,fit,deployment,qualify
s=importlib.util.spec_from_file_location('_r37_prior4',P/'run.py');old=importlib.util.module_from_spec(s);s.loader.exec_module(old)
io=old.io;FILES=('run.py','energy.py','verify.py','test_energy.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml')


def register():
    assert not (R/'lock.json').exists();old.validate()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==9
    assert not any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped'))
    previous=io.read(R.parent/'round36/closure_receipt.json');assert previous['status']=='closed' and previous['readback_passed']
    for name,h in previous['files_sha256'].items():assert io.sha(R.parent/name)==h,name
    prior=io.read(N/'lock.json');io.check_files(N,prior['files']);deps=dict(prior['dependencies'])
    for name in prior['files']:deps[str((N/name).resolve())]=io.sha(N/name)
    for p in (N/'lock.json',N/'closure_receipt.json',N/'diagnostic.json',N/'verification.json',Path(io.__file__),R.parent/'round36/closure_receipt.json',R.parent/'round36/verify.py'):
        deps[str(p.resolve())]=io.sha(p)
    assert io.read(N/'verification.json')['passed']
    for name,h in io.read(N/'diagnostic.json')['pilot_files_sha256'].items():
        p=N/'pilot'/name;assert io.sha(p)==h;deps[str(p.resolve())]=h
    for setting in old.SETTINGS[:3]:
        p=P/'results/folds'/setting/'fold0/core_real/arrays.npz';deps[str(p.resolve())]=io.sha(p)
    for source in old.SOURCES:
        for f in ('static28.npy','complete.json','row_identity.csv'):
            p=old.SOURCE/'exports'/source/f;deps[str(p.resolve())]=io.sha(p)
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,versions=io.versions(),CPU_only=True,
        CPU_limit_seconds=900,source_seed=42,settings=old.SETTINGS[:3],sources=old.SOURCES,primary='core_real',alpha=100.,lambda_ratio=.1,
        native_dimensions=220,static_dimensions=28,matrix_shape=[6,32],new_native_heads=45,new_TD28_heads=15,new_heads=60,new_internal_OOF_rows=9196,
        reused_native_heads=45,reused_unique_TD24_heads=15,new_source_forward=0,new_PCA=0))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)


def validate():
    lock=io.read(R/'lock.json');io.check_files(R,lock['files'])
    for p,h in lock['dependencies'].items():assert io.sha(Path(p))==h,p
    assert io.versions()==lock['versions'];return lock


def stats(y,p):
    pc,ms=old.score(y,p[:,None]);return dict(PCC=float(pc[0]),MSE=float(ms[0]))


def save(out,z,y,tr,va,beta,center,cert,coef,offset,extra):
    prediction=z[va]@beta+center
    old.npz_write(out/'model.npz',beta=beta,intercept=np.array(center),prediction=prediction,deployment_coef=coef,deployment_intercept=np.array(offset))
    io.write(out/'state.json',dict(train_indices=tr,valid_indices=va,certificate=cert,numerically_qualified=bool(cert['certified']),**extra))
    io.write(out/'complete.json',dict(lock_sha256=io.sha(R/'lock.json'),files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
    return prediction


def run(start):
    validate();root=R/'pilot';assert not root.exists() and not (R/'diagnostic.json').exists()
    threads=threadpool_info();assert all(t['num_threads']==1 for t in threads)
    rows,xx,identities=old.inputs();lookup={r['sample_id']:i for i,r in enumerate(rows)};plan=io.read(P/'plan.json')
    energy=None;energy_sha={}
    for source in old.SOURCES:
        folder=old.SOURCE/'exports'/source;c=io.read(folder/'complete.json');io.check_files(folder,c['files_sha256'])
        meta=io.table(folder/'row_identity.csv');index={r['sample_id']:i for i,r in enumerate(meta)}
        aligned=np.load(folder/'static28.npy',allow_pickle=False)[[index[r['sample_id']] for r in rows]]
        assert aligned.shape==(3329,28) and np.isfinite(aligned).all()
        assert np.array_equal(aligned[:,:24],xx[source][:,:24]) and np.isin(aligned[:,27],[0.,1.]).all()
        assert not aligned[aligned[:,27]==0,24:27].any()
        if energy is None:energy=aligned[:,24:]
        else:assert np.array_equal(energy,aligned[:,24:])
        energy_sha[source]=io.sha(folder/'static28.npy')
    assert len(set(energy_sha.values()))==1
    records=[];tdrecords=[];details=[];pools={};coverage_info={};count=0;numeric=True
    for setting in old.SETTINGS[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids'];pools[setting]=ids
        assert set(ids).isdisjoint(item['test_ids']);fi=np.array([lookup[s] for s in ids]);y=np.array([float(rows[i]['label']) for i in fi]);e=energy[fi]
        coverage_info[setting]=dict(rows=len(fi),valid=int(e[:,3].sum()),invalid=int((1-e[:,3]).sum()))
        tdpath=P/'results/folds'/setting/'fold0/core_real/arrays.npz'
        with np.load(tdpath,allow_pickle=False) as a:td24=a['inner_oof'][:,3].copy()
        td28=np.full(len(ids),np.nan)
        for source in old.SOURCES:
            raw=xx[source][fi];pred=np.full(len(ids),np.nan);ref=pred.copy();blocks=[]
            for j,split in enumerate(item['inner']):
                if time.monotonic()-start>900:raise TimeoutError('CPU stage cap')
                tr,va=np.array(split['train']),np.array(split['valid']);native=N/'pilot'/setting/source/f'inner{j}';control=D/'pilot'/setting/source/'dense'/f'inner{j}'
                nc=io.read(native/'complete.json');io.check_files(native,nc['files_sha256']);ns=io.read(native/'state.json');st=io.read(control/'transform.json')
                assert ns['train_indices']==tr.tolist() and ns['valid_indices']==va.tolist() and ns['transform_sha256']==io.sha(control/'transform.json')
                with np.load(native/'model.npz',allow_pickle=False) as a:ref[va]=a['prediction']
                tout=root/setting/'TD_energy28'/f'inner{j}'
                if source=='core_real':
                    t=fit_energy(e[tr]);io.write(tout/'energy_transform.json',t);z=transform(raw,e,st,t)
                    before=time.monotonic();b,c,cert=fit(z[tr,:28],y[tr]);coef,offset=deployment(st,t,b,c)
                    td28[va]=save(tout,z[:,:28],y,tr,va,b,c,cert,coef,offset,dict(setting=setting,source='TD_energy28',inner=j,
                        transform_sha256=io.sha(control/'transform.json'),energy_transform_sha256=io.sha(tout/'energy_transform.json'),CPU_fit_wall_seconds=time.monotonic()-before))
                    count+=1;numeric &= bool(cert['certified'])
                else:
                    t=io.read(tout/'energy_transform.json');z=transform(raw,e,st,t)
                out=root/setting/source/f'inner{j}';before=time.monotonic();b,c,cert=fit(z[tr],y[tr],deadline=start+900)
                coef,offset=deployment(st,t,b,c);pp=save(out,z,y,tr,va,b,c,cert,coef,offset,dict(setting=setting,source=source,inner=j,
                    transform_sha256=io.sha(control/'transform.json'),energy_transform_sha256=io.sha(tout/'energy_transform.json'),
                    native_reference_sha256=io.sha(native/'model.npz'),TD28_reference_sha256=io.sha(tout/'model.npz'),CPU_fit_wall_seconds=time.monotonic()-before))
                pred[va]=pp;count+=1;numeric &= bool(cert['certified'])
                blocks.append(dict(inner=j,n=len(va),metrics=stats(y[va],pp),reference=stats(y[va],ref[va]),TD28_metrics=stats(y[va],td28[va]),
                    iterations=cert['iterations'],rank=cert['rank'],numerically_qualified=bool(cert['certified'])))
            assert np.isfinite(pred).all() and np.isfinite(ref).all() and np.isfinite(td28).all()
            m,rr,tdm=stats(y,pred),stats(y,ref),stats(y,td28)
            rec=dict(setting=setting,source=source,n=len(ids),metrics=m,reference=rr,TD28_metrics=tdm,TD24_metrics=stats(y,td24),
                delta_PCC_vs_R25=m['PCC']-rr['PCC'],MSE_reduction_vs_R25_pct=100*(1-m['MSE']/rr['MSE']),inner_blocks=blocks)
            records.append(rec)
            details.extend(dict(setting=setting,source=source,sample_id=s,label=float(y[k]),prediction=float(pred[k]),reference_prediction=float(ref[k]),reference_kind='R25',TD28_prediction=float(td28[k]),TD24_prediction=float(td24[k])) for k,s in enumerate(ids))
            print(setting,source,'deltaPCC',rec['delta_PCC_vs_R25'],'MSEred',rec['MSE_reduction_vs_R25_pct'],flush=True)
        tdm,tdold=stats(y,td28),stats(y,td24)
        tdrecords.append(dict(setting=setting,source='TD_energy28',n=len(ids),metrics=tdm,TD24_metrics=tdold,delta_PCC=tdm['PCC']-tdold['PCC'],MSE_reduction_pct=100*(1-tdm['MSE']/tdold['MSE'])))
        details.extend(dict(setting=setting,source='TD_energy28',sample_id=s,label=float(y[k]),prediction=float(td28[k]),reference_prediction=float(td24[k]),reference_kind='TD24',TD28_prediction=float(td28[k]),TD24_prediction=float(td24[k])) for k,s in enumerate(ids))
    assert count==60 and len(details)==9196;gates=qualify(records,numeric);io.write_csv(R/'diagnostic_rows.csv',details)
    io.write(R/'diagnostic.json',dict(**gates,records=records,TD28_records=tdrecords,fit_ids=pools,energy_coverage=coverage_info,source_identities=identities,static28_sha256=energy_sha,
        new_native_heads=45,new_TD28_heads=15,new_heads=60,new_internal_OOF_rows=9196,reused_native_heads=45,reused_unique_TD24_heads=15,
        new_source_forward=0,new_PCA=0,GPU_hours=0,outer_test_metrics_used=False,threadpool=threads,lock_sha256=io.sha(R/'lock.json'),
        diagnostic_rows_sha256=io.sha(R/'diagnostic_rows.csv'),pilot_files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    validate();print('QUALIFIED',gates['qualified'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('register','run'))
    if ap.parse_args().mode=='register':register()
    else:
        mutex=R/'execution.mutex'
        with mutex.open('x') as f:f.write(str(os.getpid()))
        start=time.monotonic();rc=1;error=None
        try:run(start);rc=0
        except Exception:error=traceback.format_exc();raise
        finally:
            io.write(R/f'execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_source_forward=0));mutex.unlink()
