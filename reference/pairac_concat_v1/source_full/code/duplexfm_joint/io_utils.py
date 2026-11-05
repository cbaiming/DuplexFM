"""Small shared utilities; checkpoint files contain no foundation-model weights."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def data_manifest_hashes(root):
    root = Path(root)
    return {str(path.relative_to(root)): sha256(path) for path in
            (root / "manifest.json", root / "complete.json", root / "meta.json") if path.is_file()}


def package_source_hashes():
    return {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state(device):
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    item = state["numpy"]
    np.random.set_state((item[0], np.asarray(item[1], dtype=np.uint32), item[2], item[3], item[4]))
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=False)
    if isinstance(batch, dict):
        return {key: to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [to_device(value, device) for value in batch]
    return batch


def autocast_context(device, amp):
    if amp == "fp32":
        return contextlib.nullcontext()
    if amp != "bf16":
        raise ValueError("amp must be 'fp32' or 'bf16'; fp16 is intentionally unsupported")
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 requires a supported CUDA device; use fp32 for CPU tests")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def binary_metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(labels) == 0 or labels.shape != probabilities.shape:
        raise ValueError("Empty or mismatched prediction arrays")
    if not np.isfinite(probabilities).all() or not np.isfinite(labels).all():
        raise ValueError("Nonfinite labels or predictions")
    if not np.isin(labels, (0, 1)).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("miRBench evaluation requires binary labels and probabilities in [0, 1]")
    decisions = probabilities >= 0.5
    both_classes = np.unique(labels).size == 2
    return {
        "rows": len(labels), "positives": int(labels.sum()),
        "aps": float(average_precision_score(labels, probabilities)) if labels.sum() > 0 else None,
        "auroc": float(roc_auc_score(labels, probabilities)) if both_classes else None,
        "f1": float(f1_score(labels, decisions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, decisions)),
        "precision": float(precision_score(labels, decisions, zero_division=0)),
        "recall": float(recall_score(labels, decisions, zero_division=0)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "threshold": 0.5,
    }


def trainable_state(model):
    parameters = {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}
    buffers = {name: value.detach().cpu().clone() for name, value in model.fusion.named_buffers()}
    return {"trainable_parameters": parameters, "fusion_buffers": buffers}


def load_trainable_state(model, checkpoint):
    current = {name: value for name, value in model.named_parameters() if value.requires_grad}
    saved = checkpoint["trainable_parameters"]
    if set(saved) != set(current):
        raise ValueError(f"Trainable parameter names differ: missing={sorted(set(current)-set(saved))}, extra={sorted(set(saved)-set(current))}")
    buffers = dict(model.fusion.named_buffers())
    saved_buffers = checkpoint["fusion_buffers"]
    if set(saved_buffers) != set(buffers):
        raise ValueError("Fusion buffer names differ; use the exact checkpoint architecture")
    with torch.no_grad():
        for source, destination in ((saved, current), (saved_buffers, buffers)):
            for name, tensor in source.items():
                target = destination[name]
                if tuple(tensor.shape) != tuple(target.shape):
                    raise ValueError(f"Checkpoint shape mismatch at {name}")
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    raise ValueError(f"Nonfinite checkpoint tensor at {name}")
                target.copy_(tensor.to(device=target.device, dtype=target.dtype))


def save_checkpoint(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def load_checkpoint(path):
    # These locally generated checkpoints use tensors and basic Python types only.
    return torch.load(path, map_location="cpu", weights_only=True)


def base_provenance(model_config):
    """Hash actual local base files once per invocation, including custom model code."""
    report = {}
    extensions = {".json", ".bin", ".safetensors", ".txt", ".model", ".py", ".vocab", ".tiktoken"}
    for key in ("target_model", "guide_model"):
        root = Path(model_config[key]).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"{key} must identify a local base-model directory: {root}")
        files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix in extensions
                       and ".git" not in path.relative_to(root).parts and "__pycache__" not in path.parts)
        if not files:
            raise ValueError(f"No foundation-model files found: {root}")
        report[key] = {"path": str(root), "files": {path.relative_to(root).as_posix(): sha256(path) for path in files}}
    return report


def provenance_content(provenance):
    if isinstance(provenance, dict):
        return {key: provenance_content(value) for key, value in provenance.items() if key != "path"}
    if isinstance(provenance, list):
        return [provenance_content(value) for value in provenance]
    return provenance


def subset(indices, maximum, seed):
    indices = np.asarray(indices, dtype=np.int64)
    if maximum is None or int(maximum) <= 0 or len(indices) <= int(maximum):
        return indices.copy()
    chosen = np.random.default_rng(seed).choice(len(indices), size=int(maximum), replace=False)
    # Stable source order for validation/row-level auditing; selection uses no labels.
    return indices[np.sort(chosen)]


@torch.inference_mode()
def predict(model, store, indices, device, batch_size, amp="fp32"):
    model.eval()
    rows, labels, scores = [], [], []
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0 or len(np.unique(indices)) != len(indices):
        raise ValueError("Evaluation partition must be nonempty with unique row indices")
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        batch = store.collate(selected)
        actual = batch["row_indices"]
        actual = actual.cpu().numpy() if isinstance(actual, torch.Tensor) else np.asarray(actual)
        if not np.array_equal(actual, selected):
            raise ValueError("Collate changed row order or row identity")
        batch = to_device(batch, device)
        with autocast_context(device, amp):
            logits = model(batch).float().reshape(-1)
        y = batch["labels"].float().reshape(-1)
        if logits.shape != y.shape:
            raise ValueError("Model logits and labels must have matching batch shape")
        rows.append(actual.copy())
        labels.append(y.cpu().numpy())
        scores.append(logits.sigmoid().cpu().numpy())
    rows, labels, scores = np.concatenate(rows), np.concatenate(labels), np.concatenate(scores)
    return rows, labels, scores, binary_metrics(labels, scores)
