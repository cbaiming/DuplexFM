"""Validated reuse of a published-in-task seed42 Core+Access source cache."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4';B=R.parents[1]
spec=importlib.util.spec_from_file_location('_r14_existing_io',P/'run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
io=old.io
EXPORT=B/'core_access_transfer_20261012/results/extracted'
SOURCE=B/'core_access_seed42_20261012/results/h100_source'
TERMINAL='2bb1c1f9e8d2b0187c98ffcaa527c43bc31720a529ed8337ab730b7d47a3ca20'
INITIAL='708f97a8f6396ad97d212dd84594425da3c8ab3047d66804712c04812d479c7d'

def align(rows,meta,source):
    ids=[r['sample_id'] for r in meta]
    if len(ids)!=len(set(ids)) or len(source)!=len(meta):raise ValueError('Invalid cache row identity')
    order={s:i for i,s in enumerate(ids)};selected=[]
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Duplicated requested row')
    for r in rows:
        i=order[r['sample_id']]
        if not all(r[k]==meta[i][k] for k in ('sample_id','cohort','guide','target57')):raise ValueError('Input mismatch')
        selected.append(i)
    x=np.asarray(source[selected],float)
    if not np.isfinite(x).all():raise ValueError('Nonfinite source')
    return x

def inputs():
    old.validate();rows,_,_=old.inputs();deps={}
    def bind(p):deps[str(p.resolve())]=io.sha(p);return deps[str(p.resolve())]
    verification=B/'core_access_transfer_20261012/results/local_verification.json';v=io.read(verification)
    assert v['status']=='passed' and bind(EXPORT/'bundle_manifest.json')==v['manifest_sha256'];bind(verification)
    manifest=io.read(EXPORT/'bundle_manifest.json')['files_sha256'];io.check_files(EXPORT,manifest)
    for name,h in manifest.items():deps[str((EXPORT/name).resolve())]=h
    contract=io.read(EXPORT/'contract.json');execution=io.read(EXPORT/'execution_lock.json')
    assert contract['seed']==execution['seed']==42 and execution['contract_sha256']==io.sha(EXPORT/'contract.json')
    for setting,n in zip(old.SETTINGS[:3],(2196,469,664)):
        p=old.BENCH/'data'/f'{setting}.csv';assert bind(p)==execution['data_sha256'][setting]
        assert sum(r['cohort']==setting for r in rows)==n
    folder=SOURCE/'extracted/runs/core_access_real';complete=io.read(folder/'complete.json')
    assert complete['status']=='complete' and complete['updates']==32728 and complete['reference_epochs']==8 and complete['selection']=='terminal_update'
    assert bind(folder/'terminal.pt')==complete['terminal_sha256']==TERMINAL
    assert bind(folder/'initial.pt')==complete['initial_sha256']==INITIAL
    assert bind(folder/'signature.json')==complete['signature_sha256'];bind(folder/'complete.json')
    local=io.read(SOURCE/'numerical_local_verification.json');assert local['status']=='passed'
    assert local['new_terminal_sha256']==TERMINAL and local['new_initial_sha256']==INITIAL
    assert bind(SOURCE/'extracted/result_verification.json')==local['source_result_verification_sha256'];bind(SOURCE/'numerical_local_verification.json')
    remote_terminal=[h for p,h in contract['remote_sha256'].items() if p.endswith('/runs/core_access_real/terminal.pt')]
    assert remote_terminal==[TERMINAL]
    audits=io.read(EXPORT/'forward_audit.json');assert len(audits)==2 and sum(a['cache']['rows'] for a in audits)==3688
    removed=[8,9,10,11,12,19,20,21,22,23,32,33,34,35,36,37,38,39]
    for a in audits:
        assert a['frozen'] and a['physical_perturbation_invariant'] and a['removed_channels']==removed
        assert a['batch_parity_max_abs']==0 and a['access_zero_diagnostic_max_abs']>0 and a['design_endpoint_biological']==37
    meta=io.table(EXPORT/'row_identity.csv');source=np.load(EXPORT/'source.npy',allow_pickle=False)
    assert source.shape==(3688,192) and len(meta)==3688
    selected=align(rows,meta,source);assert selected.shape==(3329,192)
    td=np.array([[float(x) for x in r['td'].split(',')] for r in rows]);assert td.shape==(3329,24)
    identity=dict(name='core_access_real',source_seed=42,checkpoint_sha256=TERMINAL,initial_sha256=INITIAL,
        updates=32728,reference_epochs=8,feature_dimension=192,design_endpoint=37,
        source_sha256=io.sha(EXPORT/'source.npy'),row_identity_sha256=io.sha(EXPORT/'row_identity.csv'),
        matched_initial_feature_cache_available=False)
    return rows,{'core_access_real':np.c_[td,selected]},identity,deps

if __name__=='__main__':
    start=time.monotonic();assert not (R/'asset_verification.json').exists()
    rows,xx,identity,deps=inputs()
    io.write(R/'asset_verification.json',dict(passed=True,source_identity=identity,rows=3329,
        retained_cohorts={s:sum(r['cohort']==s for r in rows) for s in old.SETTINGS[:3]},
        no_numeric_test_metrics_parsed=True,new_fits=0,new_source_forward=0,GPU_hours=0,
        files_sha256=deps,code_sha256=io.sha(Path(__file__)),protocol_sha256=io.sha(R/'ASSET_PROTOCOL.md'),CPU_wall_seconds=time.monotonic()-start))
    print('ASSET_PASSED',io.sha(R/'asset_verification.json'),identity,flush=True)
