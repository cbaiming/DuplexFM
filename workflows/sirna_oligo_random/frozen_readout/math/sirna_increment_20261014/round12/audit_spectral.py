"""Independent original-PCA/common-scale and diagnostic SVD reconstructions."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4';B=R.parents[1]
spec=importlib.util.spec_from_file_location('_r12_original_audit',P/'verify.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def expand(train,other,t):
    p=t['projection'];train_white=old.project(train,train,p);other_white=old.project(train,other,p)
    actual=np.var(train_white[:,24:]*np.asarray(p['pc_scale']),axis=0)
    scale=float(np.sqrt(actual.mean()));scale=scale if scale>1e-12 else 1.
    old.close(actual,t['raw_pc_variance'],1e-10);old.close(scale,t['common_pc_scale'],1e-11)
    old.close(t['source_variance_trace'],actual.sum()/scale**2,1e-10);assert t['new_PCA_fit'] is False
    if actual.mean()>1e-24:old.close(t['source_variance_trace'],16.,1e-10)
    return np.c_[other_white[:,:24],other_white[:,24:]*np.asarray(p['pc_scale'])/scale]

def verify_diagnostic():
    start=time.monotonic();assert not (R/'diagnostic_verification.json').exists()
    lock=old.read(R/'diagnostic_lock.json');old.check(R,lock['files'])
    for p,h in lock['dependencies'].items():assert old.sha(Path(p))==h,p
    d=old.read(R/'diagnostic.json');assert d['lock_sha256']==old.sha(R/'diagnostic_lock.json')
    assert not d['real_labels_used'] and not d['new_prediction_metrics'] and d['unsupervised_half_decompositions']==18
    plan=old.read(P/'plan.json');count=0;eligible=[]
    for record in d['rows']:
        setting,source=record['setting'],record['source'];item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        ids=item['fit_ids'];assert ids==d['fit_ids'][setting]
        folder=B/'core_six_settings_20261011/results/exports'/source
        meta=old.table(folder/'row_identity.csv');ix={r['sample_id']:i for i,r in enumerate(meta)}
        x=np.load(folder/'source.npy',allow_pickle=False)[[ix[s] for s in ids]].astype(float)
        t=old.read(P/'results/folds'/setting/'fold0'/source/'transforms.json')['final']
        z=(x-t['src_mean'])/t['src_scale'];eig=np.asarray(t['eigenvalues']);old.close(eig,record['eigenvalues'])
        order=sorted(range(len(ids)),key=lambda i:hashlib.sha256(('R12:42:'+ids[i]).encode()).hexdigest())
        halves=[order[:len(order)//2],order[len(order)//2:]];assert halves==record['half_indices']
        vv=[]
        for half in halves:
            h=z[half]-z[half].mean(0);_,_,v=np.linalg.svd(h,full_matrices=False);vv.append(v[:16]);count+=1
        lead=float(np.linalg.norm(vv[0][:8]@vv[1][:8].T,'fro')**2/8)
        tail=float(np.linalg.norm(vv[0][8:]@vv[1][8:].T,'fro')**2/8)
        old.close(lead,record['half_top8_overlap'],1e-9);old.close(tail,record['half_next8_overlap'],1e-9)
        old.close(lead-tail,record['top_minus_next_overlap'],1e-9)
        ratio=eig[0]/max(eig[-1],1e-12);old.close(ratio,record['variance_ratio_first_to_16'],1e-10)
        qualifies=bool(eig[-1]>1e-12 and ratio>=3);assert qualifies==record['qualified']
        if source=='core_real':eligible.append(qualifies)
    assert count==18 and len(eligible)==3 and d['qualified']==(sum(eligible)>=2)
    result=dict(passed=True,qualified=d['qualified'],independent_half_SVDs=count,real_labels_used=False,new_efficiency_fits=0,
        GPU_hours=0,new_source_forward=0,CPU_wall_seconds=time.monotonic()-start,
        diagnostic_sha256=old.sha(R/'diagnostic.json'),lock_sha256=old.sha(R/'diagnostic_lock.json'),code_sha256=old.sha(Path(__file__)))
    with (R/'diagnostic_verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':verify_diagnostic()
