"""Actual public encoders: CPU acceptance and leased H100 BF16 preflight."""
from pathlib import Path
import gc,os,time
from .common import read,sha,verify_files
from .durable_io import durable_json as write

def smoke(root,assets,output,*,h100=False,guard=None):
    import numpy as np
    import torch
    from duplexfm_joint.data import PairStore
    from duplexfm_joint.io_utils import seed_all,load_checkpoint
    from .model import PairACModel
    from .protocol import source_config,ENGINEERING_VARIANTS
    from .training_hourly import run_engine
    root,assets,out=Path(root),Path(assets),Path(output)
    out.mkdir(parents=True,exist_ok=True)
    if h100:
        if torch.cuda.device_count()!=1 or 'H100' not in torch.cuda.get_device_name(0):raise ValueError('One leased H100 required')
        device,amp=torch.device('cuda:0'),'bf16'
    else:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU acceptance must hide CUDA')
        device,amp=torch.device('cpu'),'fp32'
    started=time.time();torch.set_num_threads(2)
    report={'engineering_only':True,'formal_source_updates':0,'device':str(device),'amp':amp,
            'H100_GPU_acceptance_completed':h100,'variants':{}}
    initial_pairs=[]
    with PairStore(assets/'mirbench') as store:
        integrity=store.verify_integrity();train,val=store.indices('train')[:4],store.indices('val')[:4]
        for variant in ENGINEERING_VARIANTS:
            if guard:guard.check()
            seed=int(variant[4:]);seed_all(seed)
            model=PairACModel(source_config(assets),variant).to(device)
            opts={'seed':seed,'reference_rows':2,'batch_size':2,'eval_batch_size':2,'max_updates':1,'amp':amp,
                  'micro_batch_size':1,'checkpoint_steps':1,'backup_seconds':3600}
            pair=model.fusion.pair_embedding.weight.detach().cpu().clone();initial_pairs.append(pair)
            lora={n:p for n,p in model.named_parameters() if 'lora_B' in n}
            assert len(lora)==48 and all(torch.count_nonzero(p)==0 for p in lora.values())
            names=[n for n,_ in model.named_parameters()]
            assert not any('energy' in n or 'bio' in n or 'inta' in n for n in names)
            assert model.fusion.input_channels==16 and sum(p.numel() for p in model.fusion.pair_embedding.parameters())==64
            result=run_engine(model,store,train[:2],val[:2],out/variant,opts,
                              {'engineering_only':True,'variant':variant,'data_manifest':integrity['manifest_sha256']},guard=guard)
            assert result['status']=='complete'
            pair_change=float((pair-model.fusion.pair_embedding.weight.detach().cpu()).abs().max())
            assert pair_change>0
            report['variants'][variant]={'seed':seed,'input_channels':16,'pair_parameters':64,
                'pair_max_update':pair_change,'fresh_zero_B_count':len(lora),
                'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                'gradient_audit':read(out/variant/'gradient_audit.json')}
            del model,lora;gc.collect()
            if h100:torch.cuda.empty_cache()
        assert all(not torch.equal(initial_pairs[i],initial_pairs[j]) for i in range(3) for j in range(i))
        report['three_distinct_pair_initializations']=True
        if not h100:
            opts={'seed':42,'reference_rows':4,'batch_size':2,'eval_batch_size':2,'max_updates':4,'amp':'fp32',
                  'micro_batch_size':1,'checkpoint_steps':1,'backup_seconds':3600}
            sig={'engineering_only':True,'data_manifest':integrity['manifest_sha256']}
            for label in ('continuous','resumed'):
                seed_all(42);model=PairACModel(source_config(assets),'seed42')
                run_engine(model,store,train,val,out/label,opts,sig,stop_after_step=1 if label=='resumed' else None)
                del model;gc.collect()
            seed_all(999);model=PairACModel(source_config(assets),'seed42')
            run_engine(model,store,train,val,out/'resumed',opts,sig,resume=True)
            a,b=(load_checkpoint(out/n/'last.pt') for n in ('continuous','resumed'))
            errors={'parameters':max(float((p-b['trainable_parameters'][n]).abs().max()) for n,p in a['trainable_parameters'].items()),
                    'optimizer':0.,'torch_rng_equal':torch.equal(a['rng']['torch'],b['rng']['torch']),
                    'history_equal':a['state']['history']==b['state']['history']}
            for k,vals in a['optimizer']['state'].items():
                for n,p in vals.items():
                    q=b['optimizer']['state'][k][n]
                    if isinstance(p,torch.Tensor):errors['optimizer']=max(errors['optimizer'],float((p-q).abs().max()))
                    elif p!=q:raise ValueError('Optimizer scalar mismatch')
            if errors['parameters'] or errors['optimizer'] or not errors['torch_rng_equal'] or not errors['history_equal']:
                raise ValueError(str(errors))
            report['exact_resume']=errors
    report.update(status='passed',wall_seconds=time.time()-started)
    write(out/'report.json',report,immutable=True)
    return report

def h100_preflight(root,assets,guard):
    root=Path(root);marker=root/'calibration/h100_preflight_complete.json'
    if marker.exists():
        receipt=read(marker);verify_files(root,receipt['files_sha256'])
        if receipt['execution_sha256']!=sha(root/'parallel_execution.json'):raise ValueError('Execution changed')
        return receipt
    out=root/'calibration'/('h100_preflight_'+str(time.time_ns()))
    smoke(root,assets,out,h100=True,guard=guard)
    receipt={'status':'passed','execution_sha256':sha(root/'parallel_execution.json'),
             'files_sha256':{(out/'report.json').relative_to(root).as_posix():sha(out/'report.json')}}
    write(marker,receipt,immutable=True)
    return receipt
