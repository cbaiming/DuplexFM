"""Real frozen foundations, all masks/seeds, optimizer invariants and exact resume."""
import gc, os, time
from pathlib import Path
from .common import read, sha, verify_files
from .durable_io import durable_json as write


def smoke(root,assets,output,*,h100=False,guard=None):
    import torch
    from duplexfm_joint.data import PairStore
    from duplexfm_joint.io_utils import seed_all, load_checkpoint
    from .model import PairACModel
    from .protocol import source_config, ENGINEERING_VARIANTS, TRAIN_VARIANTS, variant_parts
    from .reference_initial import enforce_reference_initial
    from .training_hourly import run_engine
    root,assets,out=Path(root),Path(assets),Path(output)
    out.mkdir(parents=True,exist_ok=True)
    if h100:
        if torch.cuda.device_count()!=1 or 'H100' not in torch.cuda.get_device_name(0):
            raise ValueError('One leased H100 is required')
        device,amp=torch.device('cuda:0'),'bf16'
    else:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU acceptance must hide CUDA')
        device,amp=torch.device('cpu'),'fp32'
    start=time.time();torch.set_num_threads(2)
    report={'engineering_only':True,'formal_source_updates':0,'device':str(device),'amp':amp,
            'H100_GPU_acceptance_completed':h100,'variants':{}}
    with PairStore(assets/'mirbench') as store:
        integrity=store.verify_integrity();train,val=store.indices('train')[:4],store.indices('val')[:4]
        for variant in ENGINEERING_VARIANTS:
            if guard:guard.check()
            condition,seed=variant_parts(variant);seed_all(seed)
            model=PairACModel(source_config(assets),variant).to(device)
            proof=enforce_reference_initial(model,root,seed)
            params={n:p for n,p in model.named_parameters() if p.requires_grad}
            masked=model.masked_parameter_names()
            before={n:p.detach().cpu().clone() for n,p in params.items()}
            assert len(params)==132 and sum(p.numel() for p in params.values())==1198218
            lora=[p for n,p in params.items() if 'lora_B' in n]
            assert len(lora)==48 and all(torch.count_nonzero(p)==0 for p in lora)
            opts={'seed':seed,'reference_rows':2,'batch_size':2,'eval_batch_size':2,'max_updates':1,'amp':amp,
                  'micro_batch_size':1,'checkpoint_steps':1,'backup_seconds':3600}
            result=run_engine(model,store,train[:2],val[:2],out/variant,opts,
                {'engineering_only':True,'variant':variant,'reference_initial':proof,
                 'data_manifest':integrity['manifest_sha256']},guard=guard)
            assert result['status']=='complete'
            for n in masked:
                if not torch.equal(before[n],params[n].detach().cpu()):raise ValueError('Masked parameter updated: '+n)
            pair_change=float((before['fusion.pair_embedding.weight']-model.fusion.pair_embedding.weight.detach().cpu()).abs().max())
            assert (pair_change==0) if condition=='no_pair' else (pair_change>0)
            report['variants'][variant]={'seed':seed,'condition':condition,'reference_initial':proof,
                'trainable_parameters':1198218,'parameter_tensors':132,'masked_parameter_tensors':len(masked),
                'pair_max_update':pair_change,'gradient_audit':read(out/variant/'gradient_audit.json')}
            del model,params,before,lora;gc.collect()
            if h100:torch.cuda.empty_cache()
        if not h100:
            # Test the genuinely inactive LoRA/cross optimizer case as well as
            # active-cross masks; original shared engine is used unmodified.
            report['exact_resume']={}
            for variant in TRAIN_VARIANTS:
                condition,seed=variant_parts(variant)
                opts={'seed':seed,'reference_rows':4,'batch_size':2,'eval_batch_size':2,'max_updates':2,
                      'amp':'fp32','micro_batch_size':1,'checkpoint_steps':1,'backup_seconds':3600}
                sig={'engineering_only':True,'variant':variant,'data_manifest':integrity['manifest_sha256']}
                for label in ('continuous','resumed'):
                    seed_all(seed);model=PairACModel(source_config(assets),variant)
                    enforce_reference_initial(model,root,seed)
                    run_engine(model,store,train,val,out/'resume'/variant/label,opts,sig,
                               stop_after_step=1 if label=='resumed' else None)
                    del model;gc.collect()
                seed_all(999);model=PairACModel(source_config(assets),variant)
                run_engine(model,store,train,val,out/'resume'/variant/'resumed',opts,sig,resume=True)
                a,b=(load_checkpoint(out/'resume'/variant/n/'last.pt') for n in ('continuous','resumed'))
                errors={'parameters':max(float((p-b['trainable_parameters'][n]).abs().max()) for n,p in a['trainable_parameters'].items()),
                        'optimizer':0.,'torch_rng_equal':torch.equal(a['rng']['torch'],b['rng']['torch']),
                        'history_equal':a['state']['history']==b['state']['history']}
                assert set(a['optimizer']['state'])==set(b['optimizer']['state'])
                for k,values in a['optimizer']['state'].items():
                    for n,p in values.items():
                        q=b['optimizer']['state'][k][n]
                        if isinstance(p,torch.Tensor):errors['optimizer']=max(errors['optimizer'],float((p-q).abs().max()))
                        elif p!=q:raise ValueError('Optimizer scalar mismatch')
                if errors['parameters'] or errors['optimizer'] or not errors['torch_rng_equal'] or not errors['history_equal']:
                    raise ValueError(str(errors))
                report['exact_resume'][variant]=errors
                del model,a,b;gc.collect()
    report.update(status='passed',wall_seconds=time.time()-start)
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
