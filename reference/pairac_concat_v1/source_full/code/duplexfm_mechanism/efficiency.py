"""Bounded PairAC seed42 throughput calibration; no learning-rate or label-based HPO."""
from __future__ import annotations

import math
import os
from pathlib import Path
import time

from .common import canonical, read, sha, verify_files
from .durable_io import durable_json as write

MICRO_BATCHES = (64, 128, 256, 512)
CPU_THREADS = (1, 2, 4)
CARDS = 2
WARMUP = 2
MEASURED = 6


def cpu_capacity():
    """Respect container quota as well as Linux affinity, not host CPU count."""
    limits=[os.cpu_count() or 1]
    if hasattr(os,'sched_getaffinity'):
        limits.append(len(os.sched_getaffinity(0)))
    v2=Path('/sys/fs/cgroup/cpu.max')
    if v2.exists():
        quota,period=v2.read_text().split()
        if quota!='max':
            limits.append(max(1,math.floor(int(quota)/int(period))))
    for base in (Path('/sys/fs/cgroup/cpu'),Path('/sys/fs/cgroup')):
        q,p=base/'cpu.cfs_quota_us',base/'cpu.cfs_period_us'
        if q.exists() and p.exists() and int(q.read_text())>0:
            limits.append(max(1,math.floor(int(q.read_text())/int(p.read_text()))))
    return min(limits)


def slice_batch(batch, begin, end, size):
    import torch
    if isinstance(batch, torch.Tensor):
        if batch.ndim == 0 or len(batch) != size:
            raise ValueError('Ambiguous non-example tensor in batch slicing')
        return batch[begin:end]
    if isinstance(batch, dict):
        return {k:slice_batch(v, begin, end, size) for k,v in batch.items()}
    if isinstance(batch, tuple):
        return tuple(slice_batch(v, begin, end, size) for v in batch)
    if isinstance(batch, list) and len(batch) == size:
        return batch[begin:end]
    raise ValueError(f'Unsupported batch component: {type(batch)}')


def backward(model, batch, labels, micro_batch, device, amp):
    import torch
    from torch.nn import functional as F
    from duplexfm_joint.io_utils import autocast_context
    size = len(labels)
    if not 1 <= micro_batch <= 512 or size < 1:
        raise ValueError('Micro-batch must preserve the fixed global batch')
    total = torch.zeros((), dtype=torch.float32, device=device)
    for begin in range(0, size, micro_batch):
        end = min(begin + micro_batch, size)
        with autocast_context(device, amp):
            logits = model(slice_batch(batch, begin, end, size)).float().reshape(-1)
            if logits.shape != labels[begin:end].shape:
                raise ValueError('Micro-batch logits/label shape differs')
            loss = F.binary_cross_entropy_with_logits(logits, labels[begin:end]) * ((end - begin) / size)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite micro-batch loss')
        loss.backward()
        total += loss.detach()
    return total


def choose(records, memory_limit=.85, throughput_fraction=.97):
    eligible = [r for r in records if r['status']=='complete' and r.get('parallel_cpu_eligible',True)
                and r['peak_reserved_bytes']/r['total_memory_bytes']<=memory_limit]
    if not eligible:
        raise ValueError('No efficiency candidate passed memory guard')
    fastest = max(r['examples_per_second'] for r in eligible)
    near = [r for r in eligible if r['examples_per_second'] >= fastest * throughput_fraction]
    return min(near, key=lambda r:(r['micro_batch'],r['cpu_threads']))


def load_plan(root):
    root = Path(root)
    complete = read(root / 'calibration/complete.json')
    verify_files(root, complete['files_sha256'])
    plan = read(root / 'efficiency_plan.json')
    if (plan['global_batch'] != 512 or plan['registration_sha256'] != sha(root/'registration.json')
            or plan['execution_sha256'] != sha(root/'parallel_execution.json')
            or plan['micro_batch'] not in MICRO_BATCHES or plan['cpu_threads'] not in CPU_THREADS
            or cpu_capacity()<CARDS*plan['cpu_threads']+2):
        raise ValueError('Efficiency plan identities changed')
    return plan


def calibrate(root, assets, lease, guard):
    import numpy as np
    import torch
    from duplexfm_joint.data import PairStore
    from duplexfm_joint.io_utils import seed_all, rng_state, restore_rng, trainable_state, load_trainable_state, to_device
    from .data_plan import BatchSchedule
    from .model import PairACModel
    from .protocol import source_config
    from .training_hourly import optimizer_for
    root, assets = Path(root), Path(assets)
    if (root/'calibration/complete.json').exists():
        return {'status':'complete', 'plan':load_plan(root)}
    folder = root/'calibration'; folder.mkdir(exist_ok=True)
    from .engineering_three import h100_preflight
    h100_preflight(root, assets, guard)
    code_lock = {'source_registration':sha(root/'registration.json'), 'execution':sha(root/'parallel_execution.json')}
    write(folder/'signature.json', code_lock, immutable=True)
    seed_all(42)
    with PairStore(assets/'mirbench') as store:
        store.verify_integrity()
        fit=np.load(root/'preparation/train.npy', allow_pickle=False)
        model=PairACModel(source_config(assets),'seed42').to('cuda:0')
        baseline=trainable_state(model); random_state=rng_state(torch.device('cuda:0'))
        schedule=BatchSchedule(fit,2094107,512,42)
        total_memory=torch.cuda.get_device_properties(0).total_memory
        vcpus=cpu_capacity()
        records=[]
        for micro in MICRO_BATCHES:
            for threads in CPU_THREADS:
                guard.check()
                path=folder/f'mb{micro}_cpu{threads}.json'
                if path.exists():
                    row=read(path)
                    if row['signature']!=code_lock or row['hardware_name']!=torch.cuda.get_device_name(0):
                        raise ValueError('Partial calibration belongs to different execution/hardware')
                    records.append(row);continue
                torch.set_num_threads(threads)
                load_trainable_state(model,baseline);restore_rng(random_state,torch.device('cuda:0'))
                optimizer=optimizer_for(model,{'learning_rate':3e-4,'lora_learning_rate':2e-5,'weight_decay':1e-4})
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                row={'micro_batch':micro,'cpu_threads':threads,'global_batch':512,'signature':code_lock,
                     'hardware_name':torch.cuda.get_device_name(0),'total_memory_bytes':total_memory,
                     'warmup_updates':WARMUP,'measured_updates':MEASURED,
                     'available_vcpus':vcpus,'parallel_cpu_eligible':CARDS*threads+2<=vcpus}
                try:
                    start=None
                    for step in range(WARMUP+MEASURED):
                        guard.check()
                        if step==WARMUP:
                            torch.cuda.synchronize();start=time.perf_counter()
                        batch=to_device(store.collate(schedule.at(step)),torch.device('cuda:0'))
                        optimizer.zero_grad(set_to_none=True);model.train()
                        backward(model,batch,batch['labels'].float().reshape(-1),micro,torch.device('cuda:0'),'bf16')
                        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                        optimizer.step()
                        torch.cuda.synchronize()
                    elapsed=time.perf_counter()-start
                    # Keep the scientific validation batch at 512. Its actual
                    # inference memory must pass too; no metric selects a plan.
                    model.eval()
                    validation=np.load(root/'preparation/validation.npy',allow_pickle=False)[:512]
                    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
                        scores=model(to_device(store.collate(validation),torch.device('cuda:0')))
                    if not torch.isfinite(scores).all():
                        raise ValueError('Nonfinite fixed-batch validation smoke')
                    del scores
                    torch.cuda.synchronize()
                    row.update(status='complete',wall_seconds=elapsed,examples_per_second=MEASURED*512/elapsed,
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                               validation_batch_512_passed=True)
                except torch.cuda.OutOfMemoryError as exc:
                    row.update(status='OOM',error=str(exc),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                write(path,row,immutable=True);records.append(row)
                del optimizer
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
    selected=choose(records)
    plan={'global_batch':512,'micro_batch':selected['micro_batch'],'cpu_threads':selected['cpu_threads'],
          'hardware_name':selected['hardware_name'],'memory_mib':lease['gpu']['memory_mib'],
          'registration_sha256':sha(root/'registration.json'),'execution_sha256':sha(root/'parallel_execution.json'),
          'selection':'smallest micro-batch then CPU threads within 97% of fastest passing 85% reserved-memory guard',
          'learning_rate_search':False,'all_controls_share_plan':True,'fresh_calibration_weights_discarded':True}
    plan.update(available_vcpus_at_search=vcpus, reserved_controller_vcpus=2,
                parallel_cpu_requirement=CARDS*selected['cpu_threads']+2)
    write(root/'efficiency_plan.json',plan,immutable=True)
    files={p.relative_to(root).as_posix():sha(p) for p in sorted(folder.glob('*.json'))}
    files['efficiency_plan.json']=sha(root/'efficiency_plan.json')
    write(folder/'complete.json',{'status':'complete','files_sha256':files,'candidate_count':len(records)},immutable=True)
    return {'status':'complete','plan':plan}
