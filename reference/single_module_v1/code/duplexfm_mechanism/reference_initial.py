"""Exact original PairAC initial states; no trained task weights are imported."""
from pathlib import Path
from .common import read, sha, verify_files


def reference(root):
    manifest = read(Path(root) / 'code/reference_full.json')
    folder = Path(manifest['root'])
    verify_files(folder, manifest['files_sha256'])
    return folder, manifest


def enforce_reference_initial(model, root, seed, *, restore_random_state=True):
    import torch
    from duplexfm_joint.io_utils import load_checkpoint, trainable_state, restore_rng
    from .protocol import contract, source_config
    folder, manifest = reference(root)
    path = folder / 'runs' / f'seed{seed}' / 'initial.pt'
    saved = load_checkpoint(path)
    inputs, opts = saved['signature']['inputs'], saved['signature']['options']
    spec = contract()
    expected = {'seed':seed, 'reference_rows':spec['reference_train_rows'],
                'batch_size':512, 'micro_batch_size':512, 'eval_batch_size':512,
                'max_updates':32728, 'amp':'bf16', 'learning_rate':3e-4,
                'lora_learning_rate':2e-5, 'weight_decay':1e-4,
                'clip_grad_norm':1., 'checkpoint_steps':500, 'backup_seconds':3600}
    expected_config={k:v for k,v in source_config('/unused').items() if not k.endswith('_model')}
    if (saved['state']['step'] != 0 or saved['optimizer']['state'] or
            inputs['variant'] != f'seed{seed}' or inputs['source_seed'] != seed or
            opts != expected or inputs['model_config'] != expected_config):
        raise ValueError('Original reference is not a compatible pre-update PairAC checkpoint')
    for key, filename in [('fit_sha256','train.npy'),('validation_sha256','validation.npy')]:
        if sha(Path(root)/'preparation'/filename) != inputs[key]:
            raise ValueError('Ablation and original Full row partitions differ')
    current = trainable_state(model)
    counts = {}
    for kind in ('trainable_parameters','fusion_buffers'):
        if set(current[kind]) != set(saved[kind]):
            raise ValueError('Full and ablation tensor names differ')
        for name, tensor in current[kind].items():
            expected_tensor = saved[kind][name]
            if tensor.dtype != expected_tensor.dtype or not torch.equal(tensor, expected_tensor):
                raise ValueError('Exact Full initialization differs: '+name)
            if 'lora_B' in name and torch.count_nonzero(tensor):
                raise ValueError('Initial task LoRA is not fresh')
        counts[kind] = len(current[kind])
    if restore_random_state:
        device=next(model.parameters()).device
        if device.type=='cuda' and saved['rng']['cuda'] is None:
            raise ValueError('Original CUDA RNG is missing')
        restore_rng(saved['rng'],device)
    return {'reference_initial_sha256':sha(path),'reference_manifest_sha256':sha(Path(root)/'code/reference_full.json'),
            'source_seed':seed,'tensor_counts':counts,'bitwise_initial_parameters_equal':True,
            'original_rng_restored':restore_random_state,
            'scope':'task tensors/buffers plus separately SHA-locked public foundations; CUDA RNG only on H100'}
