#!/usr/bin/env python3
"""Single-GPU full-population observed-label LoRA fine-tuning.

Versioned entry point: published training programs remain reproducible.
Default initialization is the public mRNABERT backbone, fresh LoRA and fresh
FiLM/residual head, not the already supervised accessibility checkpoint.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('TORCH_FORCE_WEIGHTS_ONLY_LOAD', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MRNABERT_ENABLE_LEGACY_TRITON', '0')
RUNTIME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME))

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
import rna_backbone as rb
from tools import accessibility_v2 as v2
from tools import train_gse145805_mrnabert_lora_film_residual_transformer_two_view_per_base as original
from tools.train_gse145805_mrnabert_lora_residual_transformer_per_base import tokenize_batch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir', 'model-dir', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--micro-batch-size', type=int, default=32)
    p.add_argument('--gradient-accumulation', type=int, default=4)
    p.add_argument('--eval-batch-size', type=int, default=128)
    p.add_argument('--samples-per-epoch', type=int, default=40960)
    p.add_argument('--min-train-coverage', type=float, default=0.5)
    p.add_argument('--region-weight', type=float, default=0.0)
    p.add_argument('--min-region-coverage', type=float, default=0.5)
    p.add_argument('--selection-tolerance', type=float, default=1e-4)
    p.add_argument('--lora-learning-rate', type=float, default=5e-5)
    p.add_argument('--head-learning-rate', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--gradient-clip', type=float, default=1.0)
    p.add_argument('--warmup-fraction', type=float, default=0.03)
    p.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after-epoch', type=int, default=0,
                   help='Controlled interruption for verification; does not alter the learning-rate plan')
    cfg = p.parse_args()
    for k in ('epochs', 'patience', 'micro_batch_size', 'gradient_accumulation', 'eval_batch_size', 'samples_per_epoch'):
        if getattr(cfg, k) <= 0:
            p.error(f'{k} must be positive')
    for k in ('min_train_coverage', 'min_region_coverage'):
        if not 0 < getattr(cfg, k) <= 1:
            p.error(f'{k} must lie in (0,1]')
    if cfg.region_weight < 0 or cfg.selection_tolerance < 0 or not 0 <= cfg.warmup_fraction < 1:
        p.error('Invalid loss, selection or warmup setting')
    for k in ('lora_learning_rate', 'head_learning_rate', 'gradient_clip'):
        if getattr(cfg, k) <= 0 or not math.isfinite(getattr(cfg, k)):
            p.error(f'{k} must be finite and positive')
    if cfg.weight_decay < 0 or cfg.stop_after_epoch < 0:
        p.error('Invalid weight decay or stop epoch')
    return cfg


def capture(model):
    return original.capture_trainable_state(model)


def load_state(model, state):
    expected = get_peft_model_state_dict(model.encoder)
    v2.require_match({k: list(v.shape) for k, v in state['lora_state'].items()},
                     {k: list(v.shape) for k, v in expected.items()}, 'LoRA tensor keys/shapes')
    for block in state.values():
        if any(not torch.isfinite(x).all() for x in block.values()):
            raise ValueError('Nonfinite saved model weights')
    result = set_peft_model_state_dict(model.encoder, state['lora_state'])
    if result.unexpected_keys:
        raise ValueError(f'Unexpected LoRA parameters: {result.unexpected_keys}')
    model.head.load_state_dict(state['head_state'], strict=True)


def save_checkpoint(path, payload):
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def build_model(cfg):
    tokenizer = rb.load_gene_tokenizer(str(cfg.model_dir), None)
    backbone = rb.load_gene_model(str(cfg.model_dir), None)
    backbone.requires_grad_(False)
    targets = [name for name, _ in backbone.named_modules() if name.endswith('Wqkv')]
    if not targets or backbone.config.hidden_size != 768:
        raise ValueError('Incompatible mRNABERT backbone')
    encoder = get_peft_model(backbone, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False, r=16, lora_alpha=32, target_modules=['Wqkv'], lora_dropout=0.05, bias='none'))
    model = original.LoRASpecialTokenFiLMResidualTransformerPerBase(encoder).to(cfg.device)
    bad = [name for name, p in encoder.named_parameters() if p.requires_grad and 'lora_' not in name]
    if bad:
        raise ValueError(f'Unexpected trainable backbone parameters: {bad}')
    return tokenizer, model


def batch(arrays, ids, tokenizer, device):
    seqs = [bytes(s).decode('ascii').rstrip('\0') for s in arrays['sequences'][ids]]
    # Compact on-disk uint8 lengths are valid metadata, but PyTorch interprets
    # uint8 index tensors as masks. SEP token positions require int64 indices.
    encoded, lengths = tokenize_batch(tokenizer, seqs, np.asarray(arrays['lengths'][ids], dtype=np.int64), device)
    y = torch.tensor(np.asarray(arrays['median_accessibility_profiles'][ids]), device=device)
    return encoded, lengths, y


def metrics_with_strata(pred, y, lengths, coverage, cfg):
    result = v2.profile_metrics(pred, y, lengths, cfg.min_region_coverage)
    result['coverage_strata'] = {}
    for name, mask in [('complete', coverage >= 1), ('partial', coverage < 1),
                       ('50_to_80_percent', coverage < 0.8), ('80_to_100_percent', (coverage >= .8) & (coverage < 1))]:
        if mask.any():
            result['coverage_strata'][name] = v2.profile_metrics(pred[mask], y[mask], lengths[mask], cfg.min_region_coverage)
    return result


@torch.inference_mode()
def evaluate(model, arrays, ids, tokenizer, device, cfg):
    model.eval()
    preds = []
    for start in range(0, len(ids), cfg.eval_batch_size):
        encoded, lengths, _ = batch(arrays, ids[start:start+cfg.eval_batch_size], tokenizer, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=cfg.precision == 'bf16' and device.type == 'cuda'):
            pred = model(encoded, lengths)
        preds.append(pred.float().cpu().numpy())
    pred = np.concatenate(preds)
    y = np.asarray(arrays['median_accessibility_profiles'][ids])
    metrics = metrics_with_strata(pred, y, arrays['lengths'][ids], arrays['coverage'][ids], cfg)
    return metrics, pred


def build_contract(cfg, data_hashes, ids, device):
    plan = {k: v for k, v in vars(cfg).items()
            if k not in ('data_dir', 'model_dir', 'output_dir', 'resume', 'stop_after_epoch')}
    plan['device'] = device.type
    # Hash the actual effective runtime and custom model/tokenizer implementation.
    runtime_files = {str(p.relative_to(RUNTIME)): p for p in RUNTIME.rglob('*.py')}
    model_files = {str(p.relative_to(cfg.model_dir)): p for p in cfg.model_dir.rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts}
    env = {k: importlib.metadata.version(k) for k in ('numpy', 'torch', 'peft', 'transformers')}
    env.update({'python': sys.version, 'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                'torch_threads': torch.get_num_threads(), 'legacy_triton': os.environ['MRNABERT_ENABLE_LEGACY_TRITON'],
                'visible_gpu_count': torch.cuda.device_count(), 'cublas_workspace': os.environ['CUBLAS_WORKSPACE_CONFIG']})
    return {'schema': v2.SCHEMA, 'plan': plan, 'data': data_hashes,
            'selected_ids': {k: v2.identity(v.tolist()) for k, v in ids.items()},
            'runtime': v2.file_contract(runtime_files), 'backbone': v2.file_contract(model_files),
            'environment': env, 'initialization': 'public backbone; fresh LoRA/FiLM/residual head',
            'selection': '0.5 micro PCC + 0.5 mean within-window PCC; near ties use region MAE; undefined PCC falls back to sequence MAE',
            'sampler': 'row permutations seeded by [seed, epoch, 7301], repeat to equal samples per epoch',
            'regions': list(v2.REGIONS), 'target_coordinates': 'central 19nt, start=floor((length-19)/2); candidate only'}


def train(cfg):
    device = torch.device(cfg.device)
    if device.type != 'cuda' or torch.cuda.device_count() != 1:
        raise RuntimeError('Full training is authorized on exactly one visible GPU')
    if cfg.region_weight != 0 or cfg.min_train_coverage != .5:
        raise ValueError('This full run uses observed-only L1 and >=50% label coverage')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if cfg.precision == 'bf16' and device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 requested but unsupported')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(cfg.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = cfg.deterministic
    torch.set_float32_matmul_precision('highest')
    original.film.seed_all(cfg.seed)
    arrays, data_hashes, data_manifest = v2.load_dataset(cfg.data_dir)
    ids = {k: np.flatnonzero(arrays['splits'] == i).astype(np.int64)
           for i, k in enumerate(('train', 'val', 'test'))}
    ids['train'] = ids['train'][arrays['coverage'][ids['train']] >= cfg.min_train_coverage]
    if any(not len(x) for x in ids.values()):
        raise ValueError('Empty split after coverage filtering')
    if cfg.samples_per_epoch != len(ids['train']):
        raise ValueError('Full training must visit every eligible training row once per epoch')
    contract = build_contract(cfg, data_hashes, ids, device)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    done_path = cfg.output_dir / 'complete.json'
    config_path = cfg.output_dir / 'run_config.json'
    last_path = cfg.output_dir / 'last_checkpoint.pt'
    # Validate before touching the previous run's configuration or artifacts.
    if done_path.exists():
        v2.validate_completion(done_path, contract)
        print(json.dumps({'status': 'verified_completed_cache', 'output': str(cfg.output_dir)}), flush=True)
        return
    checkpoint = None
    if cfg.resume:
        previous = json.loads(config_path.read_text())
        v2.require_match(previous['contract'], contract, 'resume configuration')
        checkpoint = torch.load(last_path, map_location='cpu', weights_only=True)
        v2.require_match(checkpoint['contract'], contract, 'resume checkpoint')
    elif config_path.exists() or last_path.exists():
        raise FileExistsError('Existing run: use --resume or choose a new output directory')
    tokenizer, model = build_model(cfg)
    initial = capture(model)
    lora = [p for p in model.encoder.parameters() if p.requires_grad]
    head = list(model.head.parameters())
    optimizer = torch.optim.AdamW([{'params': lora, 'lr': cfg.lora_learning_rate},
                                  {'params': head, 'lr': cfg.head_learning_rate}], weight_decay=cfg.weight_decay)
    effective = cfg.micro_batch_size * cfg.gradient_accumulation
    steps = math.ceil(cfg.samples_per_epoch / effective)
    scheduler, warmup = original.baseline.make_scheduler(optimizer, steps * cfg.epochs, cfg.warmup_fraction)
    history, best, best_state, best_epoch, stale, start_epoch = [], None, None, 0, 0, 1
    if checkpoint is not None:
        load_state(model, checkpoint['trainable_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        history, best, best_state = checkpoint['history'], checkpoint['best'], checkpoint['best_state']
        best_epoch, stale = checkpoint['best_epoch'], checkpoint['stale_epochs']
        start_epoch = checkpoint['epoch'] + 1
    else:
        v2.write_json(config_path, {'contract': contract, 'split_rows': {k: len(x) for k, x in ids.items()},
            'trainable_lora': sum(p.numel() for p in lora), 'trainable_head': sum(p.numel() for p in head),
            'optimizer_steps_per_epoch': steps, 'warmup_steps': warmup,
            'dataset_interpretation': data_manifest.get('interpretation')})
        np.savez(cfg.output_dir / 'selected_ids.npz', **ids)
    if checkpoint is not None:
        v2.restore_rng(checkpoint['rng_state'])
    print(json.dumps({'status': 'training', 'start_epoch': start_epoch,
          'steps_per_epoch': steps, 'split_rows': {k: len(x) for k, x in ids.items()}}), flush=True)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, cfg.epochs + 1):
        if stale >= cfg.patience:
            break
        started = time.perf_counter()
        order = v2.epoch_indices(ids['train'], cfg.samples_per_epoch, cfg.seed, epoch)
        model.train()
        running = {'loss': 0.0, 'base_l1': 0.0, 'region_l1': 0.0}
        last_progress = time.monotonic()
        for offset in range(0, len(order), effective):
            group = order[offset:offset + effective]
            optimizer.zero_grad(set_to_none=True)
            for mb in range(0, len(group), cfg.micro_batch_size):
                subset = group[mb:mb + cfg.micro_batch_size]
                encoded, lengths, y = batch(arrays, subset, tokenizer, device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=cfg.precision == 'bf16' and device.type == 'cuda'):
                    pred = model(encoded, lengths)
                    loss, pieces = v2.observed_loss(pred, y, lengths, cfg.region_weight, cfg.min_region_coverage)
                (loss * (len(subset) / len(group))).backward()
                for key, value in [('loss', loss.detach()), ('base_l1', pieces['base_l1']), ('region_l1', pieces['region_l1'])]:
                    running[key] += float(value) * len(subset)
            torch.nn.utils.clip_grad_norm_(lora + head, cfg.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if time.monotonic() - last_progress >= 600:
                seen = offset + len(group)
                progress = {'status': 'training', 'epoch_in_progress': epoch,
                    'completed_epochs': epoch - 1, 'rows_seen_in_epoch': seen,
                    'rows_per_epoch': len(order), 'elapsed_epoch_seconds': time.perf_counter() - started,
                    'mean_training_loss_so_far': running['loss'] / seen, 'updated_unix': time.time()}
                v2.write_json(cfg.output_dir / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
                last_progress = time.monotonic()
        validation, _ = evaluate(model, arrays, ids['val'], tokenizer, device, cfg)
        candidate = v2.selection_value(validation)
        row = {'epoch': epoch, 'train': {k: v / len(order) for k, v in running.items()},
               'validation': validation, 'selection': candidate,
               'learning_rates': [float(g['lr']) for g in optimizer.param_groups],
               'seconds': time.perf_counter() - started, 'optimizer_steps_total': epoch * steps,
               'unique_training_rows': int(len(np.unique(order)))}
        history.append(row)
        if v2.is_better(candidate, best, cfg.selection_tolerance):
            best, best_epoch, best_state, stale = candidate, epoch, capture(model), 0
        else:
            stale += 1
        save_checkpoint(last_path, {'contract': contract, 'epoch': epoch,
            'trainable_state': capture(model), 'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(), 'rng_state': v2.capture_rng(),
            'history': history, 'best': best, 'best_epoch': best_epoch,
            'best_state': best_state, 'stale_epochs': stale})
        v2.write_json(cfg.output_dir / 'history.json', history)
        v2.write_json(cfg.output_dir / 'status.json', {'status': 'training', 'epoch': epoch,
            'planned_epochs': cfg.epochs, 'epoch_seconds': row['seconds'],
            'estimated_remaining_seconds': (cfg.epochs - epoch) * row['seconds'], 'updated_unix': time.time()})
        print(json.dumps({'epoch': epoch, 'seconds': round(row['seconds'], 2),
              'validation_pcc': validation['micro_pearson'], 'validation_region_mae': validation['region_mae'],
              'best_epoch': best_epoch}), flush=True)
        if cfg.stop_after_epoch and epoch >= cfg.stop_after_epoch and epoch < cfg.epochs and stale < cfg.patience:
            v2.write_json(cfg.output_dir / 'status.json', {'status': 'paused_at_epoch_boundary', 'epoch': epoch})
            return
    if best_state is None:
        raise RuntimeError('No checkpoint selected')
    load_state(model, best_state)
    validation, _ = evaluate(model, arrays, ids['val'], tokenizer, device, cfg)
    test, pred = evaluate(model, arrays, ids['test'], tokenizer, device, cfg)
    changes = {k: math.sqrt(sum(float(((v - initial[k][name]).double() ** 2).sum())
                               for name, v in block.items())) for k, block in best_state.items()}
    if not all(v > 0 and math.isfinite(v) for v in changes.values()):
        raise RuntimeError('Fine-tuning did not update both LoRA and head')
    save_checkpoint(cfg.output_dir / 'best_state.pt', {'trainable_state': best_state, 'contract': contract, 'epoch': best_epoch})
    np.savez_compressed(cfg.output_dir / 'test_predictions.npz', row_ids=ids['test'], predictions=pred,
        targets=arrays['median_accessibility_profiles'][ids['test']], lengths=arrays['lengths'][ids['test']],
        components=arrays['transcript_components'][ids['test']])
    metrics = {'best_epoch': best_epoch, 'epochs_run': len(history), 'validation': validation, 'test': test,
               'parameter_change_l2': changes, 'epoch_training_seconds': sum(x['seconds'] for x in history),
               'peak_gpu_gib': torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None}
    v2.write_json(cfg.output_dir / 'metrics.json', metrics)
    artifact_names = ['best_state.pt', 'test_predictions.npz', 'selected_ids.npz', 'metrics.json', 'history.json', 'run_config.json', 'last_checkpoint.pt']
    v2.write_json(done_path, {'status': 'complete', 'contract': contract,
        'artifacts': {name: v2.sha256(cfg.output_dir / name) for name in artifact_names}, 'finished_unix': time.time()})
    v2.write_json(cfg.output_dir / 'status.json', {'status': 'complete', 'best_epoch': best_epoch, 'updated_unix': time.time()})
    print(json.dumps({'status': 'complete', 'best_epoch': best_epoch, 'test_pcc': test['micro_pearson']}), flush=True)


def main():
    cfg = parse_args()
    # Linux advisory lock releases automatically on interruption and prevents two
    # workers from writing a run concurrently, including during strict resume.
    import fcntl
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_dir / '.run.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train(cfg)


if __name__ == '__main__':
    main()
