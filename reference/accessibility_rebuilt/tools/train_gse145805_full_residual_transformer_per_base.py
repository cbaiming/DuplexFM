#!/usr/bin/env python3
"""Train a full-data residual Transformer for per-nucleotide accessibility.

The public mRNABERT backbone remains frozen and is read from the audited FP16
token-embedding cache.  A direct token-linear path preserves the localized
backbone signal, while a three-layer Transformer predicts a gated residual
correction using the complete 50--60 nt fragment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from tools import diagnose_gse145805_mrnabert_per_base_heads as diagnostic


SPLITS = ("train", "val", "test")
MAX_LENGTH = diagnostic.MAX_LENGTH
MIDDLE_LENGTH = diagnostic.MIDDLE_LENGTH
INPUT_DIM = diagnostic.INPUT_DIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universal-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.03)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--residual-gate-init", type=float, default=0.20)
    parser.add_argument("--huber-delta", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--micro-weight", type=float, default=0.50)
    parser.add_argument("--within-sequence-weight", type=float, default=0.50)
    parser.add_argument("--min-improvement", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    name = requested.strip().lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FullProfileSplit:
    def __init__(
        self,
        cache_dir: Path,
        profiles: np.ndarray,
        universal_lengths: np.ndarray,
        universal_splits: np.ndarray,
        split: str,
        split_code: int,
        limit: int,
        seed: int,
    ) -> None:
        self.split = split
        self.embeddings = np.load(
            cache_dir / f"embeddings_{split}_fp16.npy", mmap_mode="r"
        )
        self.source_sequence_ids = np.load(
            cache_dir / f"sequence_ids_{split}.npy", mmap_mode="r"
        )
        self.source_lengths = np.load(
            cache_dir / f"lengths_{split}.npy", mmap_mode="r"
        )
        self.profiles = profiles
        total = len(self.source_sequence_ids)
        if self.embeddings.shape != (total, MAX_LENGTH, INPUT_DIM):
            raise RuntimeError(f"Unexpected {split} embedding shape: {self.embeddings.shape}")
        if self.source_lengths.shape != (total,):
            raise RuntimeError(f"Unexpected {split} length shape")
        if limit:
            if limit > total:
                raise ValueError(f"Requested {limit} rows from {split}, but only {total} exist")
            rng = np.random.default_rng(seed + 1009 * (split_code + 1))
            self.positions: np.ndarray | None = np.sort(
                rng.choice(total, size=limit, replace=False)
            ).astype(np.int64)
        else:
            self.positions = None
        selected_ids = self.sequence_ids_numpy()
        selected_lengths = self.lengths_numpy()
        if not np.array_equal(selected_lengths, np.asarray(universal_lengths[selected_ids])):
            raise RuntimeError(f"Universal lengths do not align with {split} cache")
        if not np.all(universal_splits[selected_ids] == split_code):
            raise RuntimeError(f"Universal split assignments do not align for {split}")

    def __len__(self) -> int:
        return len(self.positions) if self.positions is not None else len(self.source_sequence_ids)

    def _source_index(self, start: int, end: int) -> slice | np.ndarray:
        return self.positions[start:end] if self.positions is not None else slice(start, end)

    def sequence_ids_numpy(self) -> np.ndarray:
        if self.positions is None:
            return np.asarray(self.source_sequence_ids)
        return np.asarray(self.source_sequence_ids[self.positions])

    def lengths_numpy(self) -> np.ndarray:
        if self.positions is None:
            return np.asarray(self.source_lengths)
        return np.asarray(self.source_lengths[self.positions])

    def batch(
        self,
        start: int,
        end: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        source_index = self._source_index(start, end)
        sequence_ids = np.asarray(self.source_sequence_ids[source_index], dtype=np.int64)
        embeddings = torch.from_numpy(
            np.array(self.embeddings[source_index], dtype=np.float16, copy=True)
        ).to(device, non_blocking=True)
        lengths = torch.from_numpy(
            np.array(self.source_lengths[source_index], dtype=np.int64, copy=True)
        ).to(device, non_blocking=True)
        labels_numpy = np.asarray(self.profiles[sequence_ids], dtype=np.float32)
        labels = torch.from_numpy(np.array(labels_numpy, copy=True)).to(
            device, non_blocking=True
        )
        return embeddings, lengths, labels, sequence_ids


def valid_mask(lengths: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(MAX_LENGTH, device=lengths.device)[None, :]
    return positions < lengths[:, None]


class ResidualFullTransformerPerBase(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 256,
        heads: int = 8,
        layers: int = 3,
        ffn_dim: int = 1024,
        dropout: float = 0.05,
        residual_gate_init: float = 0.20,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if not 0.0 < residual_gate_init < 1.0:
            raise ValueError("residual_gate_init must lie strictly between zero and one")
        self.input_norm = nn.LayerNorm(input_dim)
        self.direct_output = nn.Linear(input_dim, 1)
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Embedding(MAX_LENGTH, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.residual_norm = nn.LayerNorm(hidden_dim)
        self.residual_output = nn.Linear(hidden_dim, 1)
        gate_logit = math.log(residual_gate_init / (1.0 - residual_gate_init))
        self.residual_gate_logit = nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))

    def residual_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.residual_gate_logit)

    def forward(self, embeddings: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = valid_mask(lengths)
        normalized = self.input_norm(embeddings)
        direct_logit = self.direct_output(normalized).squeeze(-1)
        positions = torch.arange(MAX_LENGTH, device=embeddings.device)
        hidden = self.projection(normalized) + self.position(positions)[None, :, :]
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        residual_logit = self.residual_output(self.residual_norm(hidden)).squeeze(-1)
        logit = direct_logit + self.residual_gate() * residual_logit
        return torch.sigmoid(logit)


def masked_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    mask = valid_mask(lengths)
    target = torch.nan_to_num(target, nan=0.0)
    point_loss = F.huber_loss(
        prediction.float(), target.float(), reduction="none", delta=float(delta)
    )
    return ((point_loss * mask).sum(dim=1) / mask.sum(dim=1)).mean()


def block_ranges(
    total: int,
    batch_size: int,
    *,
    seed: int | None,
) -> list[tuple[int, int]]:
    blocks = [
        (start, min(start + batch_size, total))
        for start in range(0, total, batch_size)
    ]
    if seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(blocks)
    return blocks


class StreamingValidation:
    def __init__(self) -> None:
        self.count = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0
        self.absolute_error = 0.0
        self.sequence_correlation_sum = 0.0
        self.sequence_correlation_count = 0

    def update(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        lengths: np.ndarray,
    ) -> None:
        valid, _ = diagnostic.region_masks(lengths)
        x = np.asarray(observed[valid], dtype=np.float64)
        y = np.asarray(predicted[valid], dtype=np.float64)
        self.count += len(x)
        self.sum_x += float(x.sum())
        self.sum_y += float(y.sum())
        self.sum_x2 += float(np.dot(x, x))
        self.sum_y2 += float(np.dot(y, y))
        self.sum_xy += float(np.dot(x, y))
        self.absolute_error += float(np.abs(y - x).sum())
        correlations = diagnostic.per_sequence_pearson(observed, predicted, valid)
        finite = correlations[np.isfinite(correlations)]
        self.sequence_correlation_sum += float(finite.sum())
        self.sequence_correlation_count += int(len(finite))

    def result(self, micro_weight: float, within_weight: float) -> dict[str, float]:
        numerator = self.sum_xy - self.sum_x * self.sum_y / self.count
        denominator = math.sqrt(
            max(self.sum_x2 - self.sum_x * self.sum_x / self.count, 0.0)
            * max(self.sum_y2 - self.sum_y * self.sum_y / self.count, 0.0)
        )
        micro = numerator / denominator
        within = self.sequence_correlation_sum / self.sequence_correlation_count
        return {
            "micro_pearson": float(micro),
            "mean_within_sequence_pearson": float(within),
            "mae": float(self.absolute_error / self.count),
            "selection_score": float(micro_weight * micro + within_weight * within),
        }


@torch.inference_mode()
def evaluate_basic(
    model: ResidualFullTransformerPerBase,
    store: FullProfileSplit,
    device: torch.device,
    cfg: argparse.Namespace,
) -> tuple[dict[str, float], float]:
    model.eval()
    accumulator = StreamingValidation()
    total_loss = 0.0
    total_rows = 0
    for start, end in block_ranges(len(store), cfg.batch_size, seed=None):
        embeddings, lengths, target, _ = store.batch(start, end, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = model(embeddings, lengths)
            loss = masked_huber_loss(prediction, target, lengths, cfg.huber_delta)
        observed_numpy = target.float().cpu().numpy()
        predicted_numpy = prediction.float().cpu().numpy()
        accumulator.update(
            observed_numpy,
            predicted_numpy,
            lengths.cpu().numpy(),
        )
        total_loss += float(loss) * (end - start)
        total_rows += end - start
    return (
        accumulator.result(cfg.micro_weight, cfg.within_sequence_weight),
        total_loss / total_rows,
    )


@torch.inference_mode()
def evaluate_detailed(
    model: ResidualFullTransformerPerBase,
    store: FullProfileSplit,
    device: torch.device,
    cfg: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    observed = np.full((len(store), MAX_LENGTH), np.nan, dtype=np.float32)
    predicted = np.full((len(store), MAX_LENGTH), np.nan, dtype=np.float32)
    for start, end in block_ranges(len(store), cfg.batch_size, seed=None):
        embeddings, lengths, target, _ = store.batch(start, end, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = model(embeddings, lengths)
        observed[start:end] = target.float().cpu().numpy()
        predicted[start:end] = prediction.float().cpu().numpy()
    result = diagnostic.detailed_profile_metrics(
        observed,
        predicted,
        store.lengths_numpy(),
        include_spearman=True,
    )
    return result, observed, predicted


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_fraction: float,
) -> tuple[torch.optim.lr_scheduler.LambdaLR, int]:
    warmup_steps = max(1, round(total_steps * warmup_fraction))

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier), warmup_steps


def atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    cfg = parse_args()
    if cfg.batch_size <= 0 or cfg.epochs <= 0 or cfg.patience <= 0:
        raise ValueError("batch-size, epochs, and patience must be positive")
    if not 0.0 <= cfg.warmup_fraction < 1.0:
        raise ValueError("warmup-fraction must lie in [0, 1)")
    if not math.isclose(
        cfg.micro_weight + cfg.within_sequence_weight, 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("micro-weight and within-sequence-weight must sum to one")
    required = [
        cfg.universal_dir / "median_accessibility_profiles.npy",
        cfg.universal_dir / "lengths.npy",
        cfg.universal_dir / "splits.npy",
        cfg.cache_dir / "cache_index.json",
    ]
    for split in SPLITS:
        required.extend(
            [
                cfg.cache_dir / f"embeddings_{split}_fp16.npy",
                cfg.cache_dir / f"sequence_ids_{split}.npy",
                cfg.cache_dir / f"lengths_{split}.npy",
            ]
        )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.output_dir / "metrics.json"
    if metrics_path.is_file() and not cfg.force:
        raise FileExistsError(f"Completed output already exists: {metrics_path}")

    seed_all(cfg.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = resolve_device(cfg.device)
    started = time.perf_counter()
    profiles_path = cfg.universal_dir / "median_accessibility_profiles.npy"
    profiles = np.load(profiles_path, mmap_mode="r")
    universal_lengths = np.load(cfg.universal_dir / "lengths.npy", mmap_mode="r")
    universal_splits = np.load(cfg.universal_dir / "splits.npy", mmap_mode="r")
    if profiles.shape != (len(universal_lengths), MAX_LENGTH):
        raise RuntimeError(f"Unexpected profile matrix shape: {profiles.shape}")
    limits = (cfg.max_train_rows, cfg.max_val_rows, cfg.max_test_rows)
    stores = {
        split: FullProfileSplit(
            cfg.cache_dir,
            profiles,
            universal_lengths,
            universal_splits,
            split,
            split_code,
            limits[split_code],
            cfg.seed,
        )
        for split_code, split in enumerate(SPLITS)
    }

    architecture = {
        "input_dim": INPUT_DIM,
        "hidden_dim": cfg.hidden_dim,
        "attention_heads": cfg.attention_heads,
        "layers": cfg.layers,
        "ffn_dim": cfg.ffn_dim,
        "dropout": cfg.dropout,
        "residual_gate_init": cfg.residual_gate_init,
        "direct_token_path": True,
        "prediction": "sigmoid per valid nucleotide",
    }
    run_config = {
        "architecture": architecture,
        "data": {
            "split_sizes": {split: len(store) for split, store in stores.items()},
            "full_split_sizes": {
                split: int(np.sum(universal_splits == code))
                for code, split in enumerate(SPLITS)
            },
            "split_policy": "fixed transcript-component-disjoint seed-42 split",
            "profile_sha256": sha256_file(profiles_path),
            "cache_index_sha256": sha256_file(cfg.cache_dir / "cache_index.json"),
        },
        "optimization": {
            "optimizer": "AdamW",
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "patience": cfg.patience,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "warmup_fraction": cfg.warmup_fraction,
            "schedule": "linear warm-up then cosine decay to zero",
            "gradient_clip": cfg.gradient_clip,
            "loss": f"per-sequence mean masked Huber(delta={cfg.huber_delta:g})",
            "selection_metric": (
                f"{cfg.micro_weight:g}*micro Pearson + "
                f"{cfg.within_sequence_weight:g}*mean within-sequence Pearson"
            ),
            "seed": cfg.seed,
            "precision": "BF16",
        },
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    write_json(cfg.output_dir / "run_config.json", run_config)

    model = ResidualFullTransformerPerBase(
        input_dim=INPUT_DIM,
        hidden_dim=cfg.hidden_dim,
        heads=cfg.attention_heads,
        layers=cfg.layers,
        ffn_dim=cfg.ffn_dim,
        dropout=cfg.dropout,
        residual_gate_init=cfg.residual_gate_init,
    ).to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    steps_per_epoch = math.ceil(len(stores["train"]) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler, warmup_steps = make_scheduler(optimizer, total_steps, cfg.warmup_fraction)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -math.inf
    best_epoch = 0
    stale_epochs = 0
    start_epoch = 1
    last_path = cfg.output_dir / "last_checkpoint.pt"
    if cfg.resume:
        if not last_path.is_file():
            raise FileNotFoundError(last_path)
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint["architecture"] != architecture:
            raise RuntimeError("Resume checkpoint architecture differs")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        history = checkpoint["history"]
        best_state = checkpoint["best_state"]
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        start_epoch = int(checkpoint["epoch"]) + 1

    print(
        json.dumps(
            {
                "stage": "start",
                "parameter_count": parameter_count,
                "steps_per_epoch": steps_per_epoch,
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
                "split_sizes": {split: len(store) for split, store in stores.items()},
                "initial_residual_gate": float(model.residual_gate().detach().cpu()),
            }
        ),
        flush=True,
    )

    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_rows = 0
        for start, end in block_ranges(
            len(stores["train"]), cfg.batch_size, seed=cfg.seed + epoch
        ):
            embeddings, lengths, target, _ = stores["train"].batch(start, end, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(embeddings, lengths)
                loss = masked_huber_loss(prediction, target, lengths, cfg.huber_delta)
            loss.backward()
            if cfg.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
            scheduler.step()
            train_loss_sum += float(loss.detach()) * (end - start)
            train_rows += end - start

        validation, val_loss = evaluate_basic(model, stores["val"], device, cfg)
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_rows,
            "val_loss": val_loss,
            "val_micro_pearson": validation["micro_pearson"],
            "val_mean_within_sequence_pearson": validation[
                "mean_within_sequence_pearson"
            ],
            "val_selection_score": validation["selection_score"],
            "val_mae": validation["mae"],
            "residual_gate": float(model.residual_gate().detach().cpu()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["selection_score"] > best_score + cfg.min_improvement:
            best_score = validation["selection_score"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        atomic_torch_save(
            {
                "epoch": epoch,
                "architecture": architecture,
                "model_state": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_state": best_state,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "history": history,
            },
            last_path,
        )
        if stale_epochs >= cfg.patience:
            break

    if best_state is None:
        raise RuntimeError("No best checkpoint was selected")
    model.load_state_dict(best_state, strict=True)
    validation_metrics, _, _ = evaluate_detailed(
        model, stores["val"], device, cfg
    )
    test_metrics, test_observed, test_predicted = evaluate_detailed(
        model, stores["test"], device, cfg
    )
    with (cfg.output_dir / "history.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(history)
    np.savez_compressed(
        cfg.output_dir / "test_predictions.npz",
        sequence_ids=stores["test"].sequence_ids_numpy(),
        lengths=stores["test"].lengths_numpy(),
        observed=test_observed.astype(np.float16),
        predicted=test_predicted.astype(np.float16),
    )
    final_gate = float(model.residual_gate().detach().cpu())
    atomic_torch_save(
        {
            "state_dict": best_state,
            "architecture": architecture,
            "seed": cfg.seed,
            "best_epoch": best_epoch,
            "best_validation_selection_score": best_score,
            "residual_gate": final_gate,
            "label": "per-position universal-median GSE145805 accessibility",
            "backbone": "frozen public mRNABERT token embeddings",
        },
        cfg.output_dir / "best_model.pt",
    )
    result = {
        "status": "pass",
        "architecture": architecture,
        "trainable_parameters": parameter_count,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_validation_selection_score": best_score,
        "best_residual_gate": final_gate,
        "split_sizes": {split: len(store) for split, store in stores.items()},
        "validation": validation_metrics,
        "test": test_metrics,
        "optimization": run_config["optimization"],
        "runtime": {
            **run_config["runtime"],
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30
                if device.type == "cuda"
                else None
            ),
            "peak_reserved_gib": (
                torch.cuda.max_memory_reserved(device) / 2**30
                if device.type == "cuda"
                else None
            ),
        },
    }
    write_json(metrics_path, result)
    print(
        json.dumps(
            {
                "status": "complete",
                "best_epoch": best_epoch,
                "best_validation_selection_score": best_score,
                "residual_gate": final_gate,
                "test_micro_pearson": test_metrics["micro_nucleotide"]["pearson"],
                "test_within_sequence_pearson": test_metrics[
                    "within_sequence_pearson"
                ]["mean"],
                "test_derived_s_access_pearson": test_metrics["derived_s_access"][
                    "pearson"
                ],
                "metrics": str(metrics_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
