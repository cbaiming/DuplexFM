"""Independent Kronecker-transform audit; no new training/encoder inference."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4'
spec=importlib.util.spec_from_file_location('_r10_verify_prior',P/'run.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
io=old.io

def main():
    start=time.monotonic();assert not (R/'verification.json').exists()
    lock=io.read(R/'diagnostic_lock.json');io.check_files(R,lock['files'])
    for p,h in lock['dependencies'].items():assert io.sha(Path(p))==h,p
    d=io.read(R/'diagnostic.json');assert d['lock_sha256']==io.sha(R/'diagnostic_lock.json')
    assert d['real_labels_used'] is False and d['outer_test_metrics_used'] is False
    assert d['new_fits']==d['new_source_forward']==d['GPU_hours']==0
    rows,xx,_=old.inputs();ids={r['sample_id']:i for i,r in enumerate(rows)}
    plan=io.read(P/'plan.json')
    helmert=np.array([[1,1,1],[1,-1,0],[1,1,-2]],float)/np.sqrt([[3],[2],[6]])
    rotation=np.kron(helmert,np.eye(64))
    np.testing.assert_allclose(rotation@rotation.T,np.eye(192),atol=1e-14,rtol=0)
    check=[]
    for record in d['statistics']:
        setting,view=record['setting'],record['view']
        item=next(p for p in plan['folds'] if p['setting']==setting and p['fold']==0)
        assert item['fit_ids']==d['fit_ids'][setting]
        x=xx[view][[ids[s] for s in item['fit_ids']],24:]
        t=io.read(P/'results/folds'/setting/'fold0'/view/'transforms.json')['final']
        z=(x-np.asarray(t['src_mean']))/np.asarray(t['src_scale'])-np.asarray(t['pca_mean'])
        comp=np.asarray(t['components']);zz=z@rotation.T
        predicted=(z@comp.T)@(comp@rotation.T)
        v=np.einsum('ij,ij->j',zz,zz);err=np.einsum('ij,ij->j',zz-predicted,zz-predicted)
        total=float(v.sum());a=float(v[:64].sum());b=float(v[64:].sum())
        common=1-float(err[:64].sum())/a;contrast=1-float(err[64:].sum())/b
        expected=dict(common_variance_fraction=a/total,contrast_variance_fraction=b/total,
            common_reconstruction_explained=common,contrast_reconstruction_explained=contrast,
            common_minus_contrast=common-contrast,global_PCA16_reconstruction_explained=1-float(err.sum())/total)
        for key,value in expected.items():np.testing.assert_allclose(record[key],value,atol=1e-12,rtol=1e-12)
        qualifies=b/total>=.1 and contrast<.5 and common-contrast>=.2
        assert record['qualifies']==qualifies and record['n']==len(x)
        if view=='core_real':check.append(qualifies)
    assert len(d['statistics'])==9 and len(check)==3 and d['qualified']==(sum(check)>=2)
    receipts=list(R.glob('diagnostic_execution_*.json'));assert len(receipts)==1 and io.read(receipts[0])['returncode']==0
    result=dict(passed=True,qualified=d['qualified'],maps=9,real_labels_used=False,new_fits=0,new_source_forward=0,GPU_hours=0,
        lock_sha256=io.sha(R/'diagnostic_lock.json'),diagnostic_sha256=io.sha(R/'diagnostic.json'),
        verifier_sha256=io.sha(Path(__file__)),CPU_wall_seconds=time.monotonic()-start)
    io.write(R/'verification.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
