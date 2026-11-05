"""All source endpoints fixed before test evaluation; committed per-set results."""
import os,time
from pathlib import Path
from .common import read,sha,verify_files,backup
from .durable_io import durable_json as write,valid_checkpoint

def check_complete(root):
    root=Path(root);path=root/'evaluation/complete.json'
    if not path.exists():return False
    v=read(path);verify_files(root,v['files_sha256'])
    for name,digest in v['source_endpoints'].items():
        if sha(root/'runs'/name/'terminal.pt')!=digest:raise ValueError('Evaluated endpoint changed')
    from .protocol import TRAIN_VARIANTS
    if set(v['source_endpoints'])!=set(TRAIN_VARIANTS):raise ValueError('Incomplete ablation endpoints')
    return v.get('status')=='complete' and v.get('seed_count')==3 and v.get('condition_count')==3 and v.get('test_count_per_seed')==4

def evaluate(root,assets,guard):
    import numpy as np
    import torch
    from duplexfm_joint.data import PairStore
    from duplexfm_joint.io_utils import seed_all,load_trainable_state,to_device,binary_metrics
    from .model import PairACModel
    from .protocol import source_config,TRAIN_VARIANTS,OFFICIAL_TESTS,variant_parts
    from .parallel import check_complete as source_complete
    root,assets=Path(root),Path(assets)
    if not all(source_complete(root,v) for v in TRAIN_VARIANTS):raise ValueError('All source endpoints must be fixed first')
    if check_complete(root):return {'status':'complete','reused':True}
    out=root/'evaluation';out.mkdir(exist_ok=True)
    endpoints={v:sha(root/'runs'/v/'terminal.pt') for v in TRAIN_VARIANTS}
    signature={'source_endpoints':endpoints,'registration_sha256':sha(root/'registration.json'),
               'selection':'fixed_terminal','tests':list(OFFICIAL_TESTS)}
    write(out/'signature.json',signature,immutable=True)
    records=[];last_backup=time.time()
    with PairStore(assets/'mirbench') as store:
        store.verify_integrity()
        for variant in TRAIN_VARIANTS:
            guard.check();seed_all(variant_parts(variant)[1])
            model=None
            for name in OFFICIAL_TESTS:
                dest=out/variant/name;dest.mkdir(parents=True,exist_ok=True)
                marker=dest/'complete.json'
                rows=store.indices(name)
                if marker.exists():
                    record=read(marker);verify_files(root,record['files_sha256'])
                    if record['signature']!=signature or record['rows']!=len(rows):raise ValueError('Existing evaluation changed')
                    records.append(record);continue
                if model is None:
                    model=PairACModel(source_config(assets),variant).to('cuda:0')
                    cp=valid_checkpoint(root/'runs'/variant/'terminal.pt',read(root/'runs'/variant/'signature.json'))
                    load_trainable_state(model,cp);model.eval();del cp
                # Each shard has its own checked receipt. Interrupted shards are
                # recalculated; committed shards are never recomputed.
                shards=[]
                for lo in range(0,len(rows),8192):
                    guard.check();hi=min(lo+8192,len(rows));file=dest/f'{lo:08d}.npz';commit=file.with_suffix('.json')
                    if commit.exists():
                        c=read(commit)
                        if c['sha256']!=sha(file) or (c['lo'],c['hi'])!=(lo,hi):raise ValueError('Evaluation shard changed')
                    else:
                        truth=[];scores=[]
                        with torch.inference_mode():
                            for begin in range(lo,hi,512):
                                guard.check();ids=rows[begin:min(begin+512,hi)]
                                batch=store.collate(ids)
                                assert np.array_equal(batch['row_indices'].numpy(),ids)
                                batch=to_device(batch,torch.device('cuda:0'))
                                with torch.autocast('cuda',dtype=torch.bfloat16):p=model(batch).float().sigmoid()
                                truth.append(batch['labels'].cpu().numpy());scores.append(p.cpu().numpy())
                        tmp=file.with_suffix('.tmp')
                        with tmp.open('wb') as f:
                            np.savez(f,rows=rows[lo:hi],labels=np.concatenate(truth),scores=np.concatenate(scores));f.flush();os.fsync(f.fileno())
                        os.replace(tmp,file)
                        write(commit,{'sha256':sha(file),'lo':lo,'hi':hi},immutable=True)
                    with np.load(file,allow_pickle=False) as a:
                        if not np.array_equal(a['rows'],rows[lo:hi]):raise ValueError('Evaluation order differs')
                        shards.append((a['labels'].copy(),a['scores'].copy()))
                    if time.time()-last_backup>=3600:
                        backup(root,[p for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.npz')],'evaluation_hourly')
                        last_backup=time.time()
                y,p=(np.concatenate([s[i] for s in shards]) for i in (0,1))
                metric=binary_metrics(y,p)
                record={'variant':variant,'test':name,'rows':len(rows),'metrics':metric,'signature':signature,
                        'files_sha256':{p.relative_to(root).as_posix():sha(p) for p in sorted(dest.glob('*')) if p.suffix in ('.npz','.json')}}
                write(marker,record,immutable=True);records.append(record)
            del model;torch.cuda.empty_cache()
    from .reference_initial import reference
    from .ablation_summary import summarize
    full,_=reference(root)
    summary=summarize(records,read(full/'evaluation/summary.json'))
    summary['full_reference_manifest_sha256']=sha(root/'code/reference_full.json')
    write(out/'summary.json',summary,immutable=True)
    files={p.relative_to(root).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ('.json','.npz')}
    write(out/'complete.json',{'status':'complete','condition_count':3,'seed_count':3,'test_count_per_seed':4,'source_endpoints':endpoints,'files_sha256':files},immutable=True)
    backup(root,[root/p for p in files]+[out/'complete.json'],'evaluation_final')
    return {'status':'complete','summary':summary}
