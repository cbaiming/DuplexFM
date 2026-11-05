"""Frozen PairAC source192, three bounded CPU workers, SHA-committed blocks."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import argparse,csv,gc,io,json,subprocess,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE=Path(os.environ.get('DUPLEXFM_ABLATION_ROOT',str(ROOT.parents[1]/'ablations/seeds43_44')))
FULL_SOURCE=Path(os.environ.get('DUPLEXFM_FULL_ROOT',str(ROOT.parents[1]/'source_full')))
OLD_EXPORT=ROOT.parent/'full'
ASSETS=Path(os.environ.get('DUPLEXFM_ASSETS',str(ROOT.parents[1]/'external_assets')))
BASE=Path(os.environ.get('DUPLEXFM_DATA_ROOT',str(ROOT.parents[1]/'external_data')))
BENCH=BASE/'duplexfm_core_six_settings_20261011/benchmark/data'
STORES=(BASE/'duplexfm_site1d_moe_transfer_20261010/inputs',BASE/'duplexfm_core_six_settings_20261011/new_inputs')
ABLATIONS=('no_pair_seed43', 'no_cross_seed43', 'no_access_seed43', 'no_pair_seed44', 'no_cross_seed44', 'no_access_seed44')
SOURCES=ABLATIONS

def model_identity(name):
    if name not in SOURCES:raise ValueError('Unregistered source')
    return name,name.rsplit('_seed',1)[0]

def mask_audit(model,maps,name):
    import torch
    variant,condition=model_identity(name)
    assert model.variant==variant and model.condition==condition and model.fusion.ablation==condition
    ranges={'no_pair':(0,4),'no_cross':(4,12),'no_access':(12,14)}
    if condition in ranges:
        lo,hi=ranges[condition];assert not torch.count_nonzero(maps[:,lo:hi])
    return {'model_variant':variant,'condition':condition,'disabled_channels':list(range(*ranges[condition])) if condition in ranges else []}
sys.path.insert(0,str(SOURCE/'code'))
from duplexfm_mechanism.common import FileLock,read,sha,verify_files,check_disk
from duplexfm_mechanism.durable_io import durable_json as write
from duplexfm_mechanism.runtime import verify_registration
from duplexfm_mechanism.model import PairACModel
from duplexfm_mechanism.protocol import source_config
from duplexfm_joint.data import PairStore
from duplexfm_joint.io_utils import seed_all,load_checkpoint,load_trainable_state

def table(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def status(name,**value):
    write(ROOT/(f'status_{name}.json'),dict(unix=time.time(),pid=os.getpid(),**value))
    print(json.dumps(dict(source=name,**value)),flush=True)

def array(p,a):
    p.parent.mkdir(parents=True,exist_ok=True);s=io.BytesIO();np.save(s,a,allow_pickle=False);data=s.getvalue()
    if p.exists():assert p.read_bytes()==data;return
    tmp=p.with_suffix('.tmp')
    with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def align(store,path,rows):
    stored=table(path);assert len(stored)==len(store)
    lookup={r['sample_id']:i for i,r in enumerate(stored)};assert len(lookup)==len(stored)
    ids=np.array([lookup[r['sample_id']] for r in rows],dtype=np.int64);alphabet=np.array(list('ACGU'))
    for r,i in zip(rows,ids):
        gi,ti=int(store.guide_ids[i]),int(store.target_ids[i])
        guide=''.join(alphabet[store.guide_tokens[gi,:int(store.guide_lengths[gi])]])
        target=''.join(alphabet[store.target_tokens[ti,:int(store.target_lengths[ti])]])
        assert guide==r['guide'] and target==r['target57'],r['sample_id']
    return ids

def resource_guard(cgroup=Path('/sys/fs/cgroup')):
    quota,period=(cgroup/'cpu.max').read_text().split()
    cores=len(os.sched_getaffinity(0))
    if quota!='max':cores=min(cores,int(quota)/int(period))
    limit=(cgroup/'memory.max').read_text().strip()
    used=int((cgroup/'memory.current').read_text())
    available=None if limit=='max' else int(limit)-used
    assert cores>=6,'Three CPU workers require at least six available quota cores'
    assert available is None or available>=20*(1<<30),'Insufficient memory headroom'
    return dict(quota_cores=cores,memory_headroom_bytes=available,workers=3,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'])

def prepare():
    req=read(ROOT/'request.json')
    for n,h in req['execution_files_sha256'].items():assert sha(ROOT/n)==h,n
    assert sha(SOURCE/'registration.json')==req['source_registration_sha256']
    assert sha(SOURCE/'source_training_complete.json')==req['source_complete_sha256']
    verify_registration(SOURCE,ASSETS)
    assert read(SOURCE/'parallel_ledger.json')['active'] is None
    assert read(SOURCE/'source_training_complete.json')['pending']==[]
    rows=[]
    for co,n in (('Huesken',2196),('Mixset',469),('Takayuki',664)):
        path=BENCH/(co+'.csv');assert sha(path)==req['benchmark_sha256'][co]
        part=table(path);assert len(part)==n
        rows.extend({k:r[k] for k in ('sample_id','cohort','guide','target57')} for r in part)
    assert len(rows)==len({r['sample_id'] for r in rows})==3329
    complement=str.maketrans('ACGU','UGCA')
    for r in rows:
        assert len(r['guide'])==19 and len(r['target57'])==57
        assert not set(r['guide']+r['target57'])-set('ACGU')
        assert r['guide'].translate(complement)[::-1]==r['target57'][19:38]
    inputs=[]
    for root,part in zip(STORES,(rows[:2665],rows[2665:])):
        with PairStore(root) as store:
            result=store.verify_integrity();assert result['manifest_sha256']==req['input_manifest_sha256'][str(root)]
            ids=align(store,root/'rows.csv',part)
            inputs.append({'root':str(root),'indices':ids.tolist(),'consumed_files_sha256':result['files_sha256']})
    identities={}
    for name in SOURCES:
        root=SOURCE
        variant,condition=model_identity(name)
        p=root/'runs'/variant/'terminal.pt';assert sha(p)==req['checkpoint_sha256'][name]
        assert sha(root/'registration.json')==req['registrations'][name]
        identities[name]={'variant':name,'model_variant':variant,'condition':condition,'source_seed':int(name[-2:]),
            'source_root':str(root),'checkpoint':'terminal.pt','checkpoint_sha256':sha(p),
            'source_registration_sha256':sha(root/'registration.json')}
    lock=dict(request_sha256=sha(ROOT/'request.json'),rows=rows,inputs=inputs,identities=identities,
              source_registration_sha256=sha(SOURCE/'registration.json'),CPU_threads_per_worker=1,
              workers=3,batch_size=32,precision='CPU FP32',GPU_hours=0)
    write(ROOT/'export_lock.json',lock,immutable=True)
    return lock

def manual_design_pool(model,batch):
    """Independent coordinate-based pool; does not call native site_pool."""
    import torch
    encoded=model.interaction_inputs(batch);h,valid=model.fusion.grid_features(encoded);result=[]
    for i in range(h.shape[0]):
        values=[];positions=[]
        for gp in range(19):
            gi=torch.where(batch['guide_pos'][i]==gp)[0].item()
            tj=torch.where(batch['target_pos'][i]==37-gp)[0].item()
            assert valid[i,gi,tj];values.append(h[i,:,gi,tj]);positions.append(gp)
        v=torch.stack(values,1);pieces=[]
        for region in ([i for i in positions if 1<=i<=7],[i for i in positions if 8<=i<=11],
                       [i for i in positions if i==0 or i>=12]):
            z=v[:,region];pieces.extend((z.mean(1),z.amax(1)))
        result.append(torch.cat(pieces))
    return torch.stack(result)

def load_model(name,identity):
    import torch
    seed=identity['source_seed'];seed_all(seed);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=Path(identity['source_root'])/'runs'/identity['model_variant']/identity['checkpoint'];assert sha(p)==identity['checkpoint_sha256']
    saved=load_checkpoint(p);assert saved['state']['step']==32728
    model=PairACModel(source_config(ASSETS),identity['model_variant']).cpu();load_trainable_state(model,saved)
    current=dict(model.named_parameters())
    assert all(torch.equal(current[n].detach(),v) for n,v in saved['trainable_parameters'].items())
    model.requires_grad_(False).eval();assert not any(p.requires_grad for p in model.parameters())
    assert model.condition==identity['condition'] and model.source_seed==seed
    del saved
    return model

def worker(name):
    import torch
    start=time.monotonic();rc=1;error=None
    lock=read(ROOT/'export_lock.json');identity=lock['identities'][name]
    req=read(ROOT/'request.json')
    for n,h in req['execution_files_sha256'].items():assert sha(ROOT/n)==h
    out=ROOT/'features'/name;out.mkdir(parents=True,exist_ok=True)
    with FileLock(out/'worker.lock'):
        if (out/'complete.json').exists():
            c=read(out/'complete.json');verify_files(out,c['files_sha256']);assert c['identity']==identity
            assert c['export_lock_sha256']==sha(ROOT/'export_lock.json');return
        try:
            status(name,phase='load');m=load_model(name,identity)
            with PairStore(STORES[0]) as store,torch.inference_mode():
                ids=np.asarray(lock['inputs'][0]['indices'],dtype=np.int64)
                b=store.collate(ids[:4]);f=m.source192(b,19);one=m.source192(store.collate(ids[:1]),19)
                torch.testing.assert_close(f[:1],one,atol=1e-4,rtol=1e-5)
                assert torch.equal(f,m.source192(b,19))
                independent=manual_design_pool(m,b);torch.testing.assert_close(f,independent,atol=1e-5,rtol=1e-5)
                encoded=m.interaction_inputs(b);maps,valid=m.fusion.make_maps(encoded)
                assert maps.shape[1]==16 and not torch.count_nonzero(maps.masked_select(~valid[:,None].expand_as(maps)))
                masked=mask_audit(m,maps,name)
                changed=dict(b,accessibility=b['accessibility'].masked_fill(~b['target_mask'],123.),
                             target=b['target'].masked_fill(~b['target_mask'],99))
                torch.testing.assert_close(f,m.source192(changed,19),atol=0,rtol=0)
                evidence={'passed':True,'frozen':True,'parameters_restored':132,'batch_max_abs':float((f[:1]-one).abs().max()),
                    'manual_design_pool_max_abs':float((f-independent).abs().max()),'PAD_invariant':True,'input_channels':16,
                    'fixed_mask':masked,'BIO_Inta_manual_products':False,'export_lock_sha256':sha(ROOT/'export_lock.json')}
                write(out/'forward_audit.json',evidence,immutable=True)
            blocks=ROOT/'blocks'/name;blocks.mkdir(parents=True,exist_ok=True);features=[];offset=0
            for inp in lock['inputs']:
                ids=np.asarray(inp['indices'],dtype=np.int64)
                with PairStore(inp['root']) as store,torch.inference_mode():
                    for lo in range(0,len(ids),32):
                        check_disk(ROOT);hi=min(lo+32,len(ids));pos=offset+lo
                        p=blocks/f'{pos:05d}.npy';r=p.with_suffix('.json')
                        bound={'export_lock_sha256':sha(ROOT/'export_lock.json'),'source':name,'offset':pos,'rows':hi-lo,
                               'sample_ids':[x['sample_id'] for x in lock['rows'][pos:offset+hi]]}
                        if r.exists():
                            c=read(r);assert c['bound']==bound and sha(p)==c['sha256'];a=np.load(p,allow_pickle=False)
                        else:
                            a=m.source192(store.collate(ids[lo:hi]),19).float().numpy()
                            assert a.shape==(hi-lo,192) and np.isfinite(a).all()
                            array(p,a);write(r,dict(bound=bound,sha256=sha(p)),immutable=True)
                        assert a.shape==(hi-lo,192) and np.isfinite(a).all();features.append(a)
                        if lo%256==0:status(name,phase='export',done=offset+hi,total=3329)
                offset+=len(ids)
            a=np.concatenate(features);expected=3329
            assert a.shape==(expected,192);array(out/'source.npy',a)
            write(out/'complete.json',{'status':'complete','identity':identity,'source_shape':[expected,192],'row_offset':0,
                'CPU_wall_seconds':time.monotonic()-start,'export_lock_sha256':sha(ROOT/'export_lock.json'),
                'files_sha256':{p.name:sha(p) for p in out.iterdir() if p.suffix in ('.json','.npy')}},immutable=True)
            rc=0;status(name,phase='complete');del m;gc.collect()
        except BaseException:
            error=traceback.format_exc();status(name,phase='failed',error=error);raise
        finally:
            write(ROOT/f'worker_{name}_cost_{time.time_ns()}.json',dict(returncode=rc,error=error,
                   CPU_wall_seconds=time.monotonic()-start,GPU_hours=0),immutable=True)

def main():
    start=time.monotonic();rc=1;error=None
    with FileLock(ROOT/'controller.lock'),FileLock(SOURCE/'cpu_export.lock'):
        assert not (ROOT/'complete.json').exists(),'Completed export is not repeated'
        try:
            resources=resource_guard();lock=prepare()
            write(ROOT/f'resources_{time.time_ns()}.json',resources,immutable=True)
            status('controller',phase='export',workers=3)
            def launch(name):
                if (ROOT/'features'/name/'complete.json').exists():
                    c=read(ROOT/'features'/name/'complete.json');verify_files(ROOT/'features'/name,c['files_sha256'])
                    assert c['identity']==lock['identities'][name] and c['export_lock_sha256']==sha(ROOT/'export_lock.json')
                    return 0
                with (ROOT/f'{name}_{time.time_ns()}.log').open('xb') as log:
                    p=subprocess.run([sys.executable,'-u',str(Path(__file__)),'--source',name],
                        stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,timeout=3600)
                if p.returncode:raise RuntimeError(f'{name} exited {p.returncode}')
                return p.returncode
            with ThreadPoolExecutor(max_workers=3) as pool:
                for future in as_completed([pool.submit(launch,n) for n in SOURCES]):assert future.result()==0
            files={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'features').rglob('*') if p.suffix in ('.json','.npy')}
            for n in ('export_lock.json','request.json','export_cpu.py','PROTOCOL.md'):files[n]=sha(ROOT/n)
            write(ROOT/'complete.json',dict(status='complete',sources=list(SOURCES),rows_per_ablation=3329,full_extra_rows=0,
                  new_source_rows=19974,files_sha256=files,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0),immutable=True)
            rc=0;status('controller',phase='complete')
        except BaseException:
            error=traceback.format_exc();status('controller',phase='failed',error=error);raise
        finally:write(ROOT/f'controller_cost_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0),immutable=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',choices=SOURCES);a=p.parse_args()
    worker(a.source) if a.source else main()
