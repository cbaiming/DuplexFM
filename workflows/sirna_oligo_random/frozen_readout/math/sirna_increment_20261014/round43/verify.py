"""Independent PCA/SVD, augmented solve, reference and deployment readback."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import csv, hashlib, importlib.util, json, time, traceback
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
R = Path(__file__).resolve().parent; B = R.parent; RESEARCH = B.parent
sp = importlib.util.spec_from_file_location('_r43_independent_pca', B / 'round4/verify.py')
v4 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v4)
read, sha, table, close = v4.read, v4.sha, v4.table, v4.close


def write(p, d):
    with p.open('x', encoding='utf-8') as f: json.dump(d, f, ensure_ascii=False, indent=2); f.write('\n')


def stats(y, p):
    return dict(PCC=float(pearsonr(y, p).statistic), MSE=float(np.mean((y-p)**2)))


def verify(start):
    lock = read(R / 'lock.json')
    for n, h in lock['files'].items(): assert sha(R / n) == h
    for n, h in lock['dependencies'].items(): assert sha(Path(n)) == h, n
    assert not (R / 'execution.mutex').exists()
    ex = list(R.glob('execution_*.json')); assert len(ex) == 1 and read(ex[0])['returncode'] == 0
    d = read(R / 'diagnostic.json'); assert d['lock_sha256'] == sha(R / 'lock.json')
    assert d['diagnostic_rows_sha256'] == sha(R / 'diagnostic_rows.csv')
    for n, h in d['pilot_files_sha256'].items(): assert sha(R / 'pilot' / n) == h
    assert len(d['records']) == 3
    bench = RESEARCH / 'core_six_settings_20261011/benchmark/data'
    rows = sum([table(bench / f'{s}.csv') for s in lock['settings']], [])
    assert len(rows) == 3329 and len({r['sample_id'] for r in rows}) == 3329
    lookup = {r['sample_id']: i for i, r in enumerate(rows)}
    td = np.array([[float(v) for v in r['td'].split(',')] for r in rows])
    oldsrc = RESEARCH / 'expert_selection_20261014/results_download/verified/features'
    identity = table(oldsrc / 'row_identity.csv'); idx = {r['sample_id']: i for i, r in enumerate(identity)}
    source_receipt = read(oldsrc / 'complete.json')
    assert source_receipt == d['source_identity'] and source_receipt['source_audit']['source_frozen']
    assert source_receipt['source_seed'] == 42 and source_receipt['source_checkpoint_sha256'] == lock['source_checkpoint_sha256']
    for n, h in source_receipt['files_sha256'].items(): assert sha(oldsrc / n) == h
    for r in rows: assert all(r[k] == identity[idx[r['sample_id']]][k] for k in ('sample_id','cohort','guide','target57'))
    slot = np.load(oldsrc / 'slots.npy', allow_pickle=False)
    assert slot.shape == (3688, 160)
    x = np.c_[td, slot[[idx[r['sample_id']] for r in rows]].astype(float)]
    export = RESEARCH / 'core_six_settings_20261011/results/exports/core_real'
    identity = table(export / 'row_identity.csv'); idx = {r['sample_id']: i for i, r in enumerate(identity)}
    for r in rows: assert all(r[k] == identity[idx[r['sample_id']]][k] for k in ('sample_id','cohort','guide','target57'))
    core = np.load(export / 'source.npy', allow_pickle=False)
    assert core.shape == (3688,192)
    core_x = np.c_[td, core[[idx[r['sample_id']] for r in rows]].astype(float)]
    plan = read(B / 'round4/plan.json'); assert plan['folds'] == read(B / 'round1/plan.json')['folds']
    detail = table(R / 'diagnostic_rows.csv'); assert len(detail) == 2299
    count = 0; max_normal = 0.; max_deploy = 0.; max_solve = 0.; independent_records = []
    for record in d['records']:
        setting = record['setting']
        item = next(f for f in plan['folds'] if f['setting'] == setting and f['fold'] == 0)
        ids = item['fit_ids']; ii = np.array([lookup[s] for s in ids]); raw = x[ii]; raw_core = core_x[ii]
        assert set(ids).isdisjoint(item['test_ids']) and d['fit_ids'][setting] == ids
        y = np.array([float(rows[i]['label']) for i in ii])
        saved4 = B / 'round4/results/folds' / setting / 'fold0/core_real'
        saved1 = B / 'round1/results/folds' / setting / 'fold0'
        t4, t1 = read(saved4 / 'transforms.json')['inner'], read(saved1 / 'transforms.json')['inner']
        with np.load(saved4 / 'arrays.npz', allow_pickle=False) as a:
            co4=a['inner_coef']; ce4=a['inner_intercept']; oo4=a['inner_oof']
        with np.load(saved1 / 'arrays.npz', allow_pickle=False) as a:
            co1=a['inner_coef']; ce1=a['inner_intercept']; oo1=a['inner_oof']
        refs = {k: np.full(len(ids), np.nan) for k in ('R4_core','TD','R25_core','R1_group')}
        pred = np.full(len(ids), np.nan); coverage=[]
        for j, split in enumerate(item['inner']):
            if time.monotonic() - start > 900: raise TimeoutError('CPU audit cap')
            tr, va = np.array(split['train']), np.array(split['valid']); coverage += va.tolist()
            out = R / 'pilot' / setting / f'inner{j}'
            receipt, st, t = read(out / 'complete.json'), read(out / 'state.json'), read(out / 'transform.json')
            assert receipt['lock_sha256'] == sha(R / 'lock.json')
            for n, h in receipt['files_sha256'].items(): assert sha(out / n) == h
            assert st['train_indices'] == tr.tolist() and st['valid_indices'] == va.tolist()
            z = v4.project(raw[tr], raw[tr], t); zv = v4.project(raw[tr], raw[va], t)
            src = (raw[tr,24:] - t['src_mean']) / t['src_scale'] - t['pca_mean']
            _, singular, vt = np.linalg.svd(src, full_matrices=False)
            close(np.array(t['eigenvalues']), singular[:16]**2/len(tr),1e-9)
            comp = np.array(t['components'])
            close(comp.T@comp,vt[:16].T@vt[:16],1e-8)
            with np.load(out / 'model.npz', allow_pickle=False) as a:
                b=a['beta']; c=float(a['intercept']); p=a['prediction']; dc=a['deployment_coef']; di=float(a['deployment_intercept'])
            close(c,y[tr].mean(),1e-12)
            normal = float(np.max(abs((z.T@z+100*np.eye(40))@b-z.T@(y[tr]-c)))/len(tr))
            assert normal < 1e-8 and st['numerically_qualified']; close(st['normal_residual'],normal,1e-12)
            independent = np.linalg.lstsq(np.r_[z,10*np.eye(40)],np.r_[y[tr]-c,np.zeros(40)],rcond=None)[0]
            close(independent,b,1e-10); max_solve=max(max_solve,float(np.max(abs(independent-b))))
            expected=zv@independent+c; close(expected,p,1e-10)
            deployed=raw[va]@dc+di; close(deployed,p,1e-9)
            max_normal=max(max_normal,normal); max_deploy=max(max_deploy,float(np.max(abs(deployed-p))))
            pred[va]=expected
            # Old R4 Core and fixed TD: same current labels, exact existing PCA.
            ctrain=v4.project(raw_core[tr],raw_core[tr],t4[j]); cval=v4.project(raw_core[tr],raw_core[va],t4[j])
            close(ce4[j],y[tr].mean(),1e-12)
            for name,k,n in [('R4_core',18,40),('TD',3,24)]:
                old_b=co4[j,k]; assert np.count_nonzero(old_b[n:])==0
                close((ctrain[:,:n].T@ctrain[:,:n]+100*np.eye(n))@old_b[:n],ctrain[:,:n].T@(y[tr]-ce4[j]),1e-7)
                refs[name][va]=cval@old_b+ce4[j]; close(refs[name][va],oo4[va,k],1e-10)
            # Original group PCA4 reference, literal independent standardization.
            mean=raw[tr].mean(0); scale=raw[tr].std(0); scale[scale<1e-12]=1
            close(mean,t1[j]['mean'],1e-12); close(scale,t1[j]['scale'],1e-12)
            a_train=(raw[tr]-mean)/scale; a_val=(raw[va]-mean)/scale
            group_train=[a_train[:,:24]]; group_val=[a_val[:,:24]]
            for k in range(5):
                sl=slice(24+32*k,24+32*(k+1)); cc=np.array(t1[j]['components'][k]); ev=np.array(t1[j]['eigenvalues'][k]); sd=np.array(t1[j]['pc_scale'][k])
                gram=a_train[:,sl].T@a_train[:,sl]/len(tr)
                close(cc@cc.T,np.eye(4),1e-10); close(gram@cc.T,cc.T*ev,1e-8)
                close(ev,np.maximum(np.linalg.eigvalsh(gram)[-4:][::-1],0),1e-9)
                expected_sd=np.sqrt(ev);expected_sd[expected_sd<1e-12]=1;close(sd,expected_sd,1e-10)
                group_train.append(a_train[:,sl]@cc.T/sd);group_val.append(a_val[:,sl]@cc.T/sd)
            gt,gv=np.c_[tuple(group_train)],np.c_[tuple(group_val)]
            close(ce1[j],y[tr].mean(),1e-12)
            close((gt.T@gt+100*np.eye(44))@co1[j,3],gt.T@(y[tr]-ce1[j]),1e-7)
            refs['R1_group'][va]=gv@co1[j,3]+ce1[j];close(refs['R1_group'][va],oo1[va,3],1e-10)
            # R25 old numerical certificate is retained; only deployment readback here.
            nr=B/'round25/pilot'/setting/'core_real'/f'inner{j}'
            ns,nc=read(nr/'state.json'),read(nr/'complete.json')
            assert ns['train_indices']==tr.tolist() and ns['valid_indices']==va.tolist() and ns['numerically_qualified']
            for n,h in nc['files_sha256'].items():assert sha(nr/n)==h
            with np.load(nr/'model.npz',allow_pickle=False) as a:
                refs['R25_core'][va]=raw_core[va]@a['deployment_coef']+float(a['deployment_intercept'])
                close(refs['R25_core'][va],a['prediction'],1e-10)
            block=record['inner_blocks'][j];assert block['inner']==j and block['n']==len(va)
            for k,value in stats(y[va],expected).items():close(value,block['metrics'][k],1e-10)
            for name,pref in refs.items():
                for k,value in stats(y[va],pref[va]).items():close(value,block['references'][name][k],1e-10)
            count+=1
        assert sorted(coverage)==list(range(len(ids)))
        new=stats(y,pred); derived=dict(setting=setting)
        for k,value in new.items():close(value,record['metrics'][k],1e-10)
        for name,pref in refs.items():
            rr=stats(y,pref)
            for k,value in rr.items():close(value,record['references'][name][k],1e-10)
            derived['delta_PCC_vs_'+name]=new['PCC']-rr['PCC']
            derived['MSE_reduction_vs_'+name+'_pct']=100*(1-new['MSE']/rr['MSE'])
            close(derived['delta_PCC_vs_'+name],record['delta_PCC_vs_'+name],1e-9)
            close(derived['MSE_reduction_vs_'+name+'_pct'],record['MSE_reduction_vs_'+name+'_pct'],1e-9)
        independent_records.append(derived)
        part=[r for r in detail if r['setting']==setting];assert [r['sample_id'] for r in part]==ids
        for i,r in enumerate(part):
            close(float(r['label']),y[i],1e-12);close(float(r['prediction']),pred[i],1e-10)
            for name,pref in refs.items():close(float(r[name+'_prediction']),pref[i],1e-10)
    assert count==15
    matched=sum(r['delta_PCC_vs_R4_core']>=.005 and r['MSE_reduction_vs_R4_core_pct']>=0 for r in independent_records)>=2 and all(r['delta_PCC_vs_R4_core']>=-.005 for r in independent_records)
    native=sum(r['MSE_reduction_vs_R25_core_pct']>=0 for r in independent_records)>=2 and sum(r['MSE_reduction_vs_R25_core_pct'] for r in independent_records)>=0 and all(r['delta_PCC_vs_R25_core']>=-.005 for r in independent_records)
    tdgate=sum(r['delta_PCC_vs_TD']>=0 for r in independent_records)>=2 and sum(r['MSE_reduction_vs_TD_pct'] for r in independent_records)>=0
    assert d['matched_readout_gate']==matched and d['strong_native_gate']==native and d['TD_increment_gate']==tdgate
    assert d['qualified']==bool(matched and native and tdgate) and d['all_numerically_qualified']
    for n,h in lock['files'].items():assert sha(R/n)==h
    for n,h in lock['dependencies'].items():assert sha(Path(n))==h
    result=dict(passed=True,qualified=d['qualified'],verified_new_heads=15,independent_augmented_solves=15,
        independent_SVD_projections=15,verified_old_R4_core_heads=15,verified_old_TD_heads=15,
        verified_old_R1_group_heads=15,verified_old_R25_deployments=15,new_internal_OOF_rows=2299,
        max_normal_error=max_normal,max_independent_solve_difference=max_solve,max_deployment_difference=max_deploy,
        new_source_forward=0,GPU_hours=0,outer_test_metrics_used=False,lock_sha256=sha(R/'lock.json'),
        diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-start)
    write(R/'verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:verify(start);rc=0
    except Exception:error=traceback.format_exc();raise
    finally:write(R/f'verification_execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_source_forward=0))
