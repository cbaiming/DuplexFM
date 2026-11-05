#!/usr/bin/env python3
"""Compare frozen-mRNABERT heads for nucleotide-level RNA accessibility.

The experiment keeps the complete 50--60 nt context and the frozen mRNABERT
token embeddings fixed.  Only the readout changes:

1. token_linear: each contextual token embedding is read independently;
2. local_cnn: a small dilated CNN adds an explicit local receptive field;
3. full_transformer: two self-attention layers can redistribute information
   across the complete fragment before predicting every nucleotide.

All heads use the same transcript-component-disjoint subset and per-position
universal-median GSE145805 labels.  The unit of splitting is the sequence, and
padding positions never contribute to the loss or metrics.
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
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch import nn


SPLITS = ("train", "val", "test")
HEADS = ("token_linear", "local_cnn", "full_transformer")
MAX_LENGTH = 60
MIDDLE_LENGTH = 25
INPUT_DIM = 768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universal-dir", required=True, type=Path)
    parser.add_argument("--full-cache-dir", required=True, type=Path)
    parser.add_argument("--subset-cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--heads", nargs="+", choices=HEADS, default=list(HEADS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=100_000)
    parser.add_argument("--val-limit", type=int, default=20_000)
    parser.add_argument("--test-limit", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--huber-delta", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
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
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
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


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous)).hexdigest()


def safe_pearson(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if len(observed) < 2 or observed.std() == 0.0 or predicted.std() == 0.0:
        return float("nan")
    observed = observed - observed.mean()
    predicted = predicted - predicted.mean()
    denominator = math.sqrt(float(np.dot(observed, observed) * np.dot(predicted, predicted)))
    return float(np.dot(observed, predicted) / denominator) if denominator else float("nan")


def safe_spearman(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 2 or np.std(observed) == 0.0 or np.std(predicted) == 0.0:
        return float("nan")
    return float(stats.spearmanr(observed, predicted).statistic)


def continuous_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    include_spearman: bool,
) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[valid], predicted[valid]
    error = predicted - observed
    denominator = float(np.square(observed - observed.mean()).sum())
    result = {
        "n": int(len(observed)),
        "pearson": safe_pearson(observed, predicted),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": (
            float(1.0 - np.square(error).sum() / denominator)
            if denominator > 0.0
            else float("nan")
        ),
        "observed_mean": float(observed.mean()),
        "observed_sd": float(observed.std()),
        "predicted_mean": float(predicted.mean()),
        "predicted_sd": float(predicted.std()),
    }
    if include_spearman:
        result["spearman"] = safe_spearman(observed, predicted)
    return result


def region_masks(lengths: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    lengths = np.asarray(lengths, dtype=np.int64)
    positions = np.arange(MAX_LENGTH)[None, :]
    valid = positions < lengths[:, None]
    left = (lengths - MIDDLE_LENGTH) // 2
    right = left + MIDDLE_LENGTH
    masks = {
        "upstream": valid & (positions < left[:, None]),
        "middle25": valid & (positions >= left[:, None]) & (positions < right[:, None]),
        "downstream": valid & (positions >= right[:, None]),
    }
    return valid, masks


def masked_row_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    numerator = np.where(mask, values, 0.0).sum(axis=1, dtype=np.float64)
    denominator = mask.sum(axis=1)
    if np.any(denominator == 0):
        raise RuntimeError("A requested region is empty")
    return np.asarray(numerator / denominator, dtype=np.float32)


def per_sequence_pearson(
    observed: np.ndarray,
    predicted: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    counts = valid.sum(axis=1, keepdims=True).astype(np.float64)
    observed_safe = np.where(valid, observed, 0.0).astype(np.float64)
    predicted_safe = np.where(valid, predicted, 0.0).astype(np.float64)
    observed_mean = observed_safe.sum(axis=1, keepdims=True) / counts
    predicted_mean = predicted_safe.sum(axis=1, keepdims=True) / counts
    observed_centered = np.where(valid, observed_safe - observed_mean, 0.0)
    predicted_centered = np.where(valid, predicted_safe - predicted_mean, 0.0)
    numerator = np.sum(observed_centered * predicted_centered, axis=1)
    denominator = np.sqrt(
        np.sum(np.square(observed_centered), axis=1)
        * np.sum(np.square(predicted_centered), axis=1)
    )
    output = np.full(len(observed), np.nan, dtype=np.float64)
    usable = denominator > 0.0
    output[usable] = numerator[usable] / denominator[usable]
    return output


def detailed_profile_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    lengths: np.ndarray,
    *,
    include_spearman: bool,
) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=np.float32)
    predicted = np.asarray(predicted, dtype=np.float32)
    valid, masks = region_masks(lengths)
    if not np.isfinite(observed[valid]).all() or not np.isfinite(predicted[valid]).all():
        raise RuntimeError("Non-finite values occurred at valid nucleotide positions")
    flat_observed = observed[valid]
    flat_predicted = predicted[valid]
    sequence_correlations = per_sequence_pearson(observed, predicted, valid)
    finite_sequence_correlations = sequence_correlations[
        np.isfinite(sequence_correlations)
    ]

    region_results: dict[str, Any] = {}
    observed_region_means: dict[str, np.ndarray] = {}
    predicted_region_means: dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        observed_means = masked_row_mean(observed, mask)
        predicted_means = masked_row_mean(predicted, mask)
        observed_region_means[name] = observed_means
        predicted_region_means[name] = predicted_means
        region_results[name] = continuous_metrics(
            observed_means,
            predicted_means,
            include_spearman=include_spearman,
        )

    observed_global = masked_row_mean(observed, valid)
    predicted_global = masked_row_mean(predicted, valid)
    observed_s_access = observed_region_means["middle25"] * (
        0.65 + 0.35 * observed_global
    )
    predicted_s_access = predicted_region_means["middle25"] * (
        0.65 + 0.35 * predicted_global
    )
    return {
        "micro_nucleotide": continuous_metrics(
            flat_observed,
            flat_predicted,
            include_spearman=include_spearman,
        ),
        "within_sequence_pearson": {
            "sequence_count": int(len(sequence_correlations)),
            "finite_sequence_count": int(len(finite_sequence_correlations)),
            "mean": float(np.mean(finite_sequence_correlations)),
            "median": float(np.median(finite_sequence_correlations)),
            "q05_q25_q75_q95": [
                float(value)
                for value in np.quantile(
                    finite_sequence_correlations,
                    [0.05, 0.25, 0.75, 0.95],
                )
            ],
        },
        "region_mean": region_results,
        "derived_s_access": continuous_metrics(
            observed_s_access,
            predicted_s_access,
            include_spearman=include_spearman,
        ),
    }


class SubsetSplit:
    def __init__(self, root: Path, split: str):
        self.split = split
        self.embeddings = np.load(
            root / f"embeddings_{split}_fp16.npy", mmap_mode="r"
        )
        self.labels = np.load(root / f"profiles_{split}_fp32.npy", mmap_mode="r")
        self.lengths = np.load(root / f"lengths_{split}.npy", mmap_mode="r")
        self.sequence_ids = np.load(
            root / f"sequence_ids_{split}.npy", mmap_mode="r"
        )
        expected = (len(self.sequence_ids), MAX_LENGTH)
        if self.embeddings.shape != (len(self.sequence_ids), MAX_LENGTH, INPUT_DIM):
            raise RuntimeError(f"Invalid embedding shape for {split}: {self.embeddings.shape}")
        if self.labels.shape != expected or self.lengths.shape != (len(self.sequence_ids),):
            raise RuntimeError(f"Invalid label/length shape for {split}")

    def __len__(self) -> int:
        return len(self.sequence_ids)

    def batch(
        self,
        start: int,
        end: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = torch.from_numpy(
            np.array(self.embeddings[start:end], dtype=np.float16, copy=True)
        ).to(device, non_blocking=True)
        labels = torch.from_numpy(
            np.array(self.labels[start:end], dtype=np.float32, copy=True)
        ).to(device, non_blocking=True)
        lengths = torch.from_numpy(
            np.array(self.lengths[start:end], dtype=np.int64, copy=True)
        ).to(device, non_blocking=True)
        return embeddings, lengths, labels


def build_subset_cache(cfg: argparse.Namespace) -> dict[str, Any]:
    cfg.subset_cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cfg.subset_cache_dir / "subset_cache_index.json"
    full_index_path = cfg.full_cache_dir / "cache_index.json"
    full_index = json.loads(full_index_path.read_text(encoding="utf-8"))
    profiles_path = cfg.universal_dir / "median_accessibility_profiles.npy"
    profiles = np.load(profiles_path, mmap_mode="r")
    universal_lengths = np.load(cfg.universal_dir / "lengths.npy", mmap_mode="r")
    universal_splits = np.load(cfg.universal_dir / "splits.npy", mmap_mode="r")
    if profiles.shape != (len(universal_lengths), MAX_LENGTH):
        raise RuntimeError(f"Unexpected profile matrix shape: {profiles.shape}")

    limits = {
        "train": int(cfg.train_limit),
        "val": int(cfg.val_limit),
        "test": int(cfg.test_limit),
    }
    plan: dict[str, Any] = {
        "seed": int(cfg.seed),
        "selection": "uniform without replacement over each complete cache split",
        "full_cache_index_sha256": sha256_file(full_index_path),
        "profile_sha256": sha256_file(profiles_path),
        "profile_shape": list(profiles.shape),
        "splits": {},
    }
    selections: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_code, split in enumerate(SPLITS):
        source_sequence_ids = np.load(
            cfg.full_cache_dir / f"sequence_ids_{split}.npy", mmap_mode="r"
        )
        source_lengths = np.load(
            cfg.full_cache_dir / f"lengths_{split}.npy", mmap_mode="r"
        )
        if len(source_sequence_ids) != int(full_index["splits"][split]["total_count"]):
            raise RuntimeError(f"Full cache index count mismatch for {split}")
        limit = min(limits[split], len(source_sequence_ids))
        if limit <= 0:
            raise ValueError(f"{split} limit must be positive")
        rng = np.random.default_rng(int(cfg.seed) + 1009 * (split_code + 1))
        source_positions = np.sort(
            rng.choice(len(source_sequence_ids), size=limit, replace=False)
        ).astype(np.int64)
        sequence_ids = np.asarray(source_sequence_ids[source_positions], dtype=np.int64)
        selected_lengths = np.asarray(source_lengths[source_positions], dtype=np.uint8)
        if not np.array_equal(selected_lengths, np.asarray(universal_lengths[sequence_ids])):
            raise RuntimeError(f"Universal lengths do not align with {split} cache")
        if not np.all(universal_splits[sequence_ids] == split_code):
            raise RuntimeError(f"Universal split labels do not align with {split} cache")
        selections[split] = (source_positions, sequence_ids)
        plan["splits"][split] = {
            "source_count": int(len(source_sequence_ids)),
            "selected_count": int(limit),
            "source_positions_sha256": sha256_array(source_positions),
            "sequence_ids_sha256": sha256_array(sequence_ids),
        }

    if index_path.is_file():
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        if existing != plan:
            raise RuntimeError("Existing subset cache plan differs from this run")
        required = []
        for split in SPLITS:
            required.extend(
                [
                    cfg.subset_cache_dir / f"embeddings_{split}_fp16.npy",
                    cfg.subset_cache_dir / f"profiles_{split}_fp32.npy",
                    cfg.subset_cache_dir / f"lengths_{split}.npy",
                    cfg.subset_cache_dir / f"sequence_ids_{split}.npy",
                    cfg.subset_cache_dir / f"source_positions_{split}.npy",
                ]
            )
        if all(path.is_file() for path in required):
            print("Reusing completed per-base subset cache", flush=True)
            return plan
        raise RuntimeError("Subset cache index exists but one or more arrays are missing")

    print("Building deterministic subset cache", flush=True)
    for split in SPLITS:
        source_positions, sequence_ids = selections[split]
        source_embeddings = np.load(
            cfg.full_cache_dir / f"embeddings_{split}_fp16.npy", mmap_mode="r"
        )
        selected_lengths = np.asarray(universal_lengths[sequence_ids], dtype=np.uint8)
        selected_profiles = np.asarray(profiles[sequence_ids], dtype=np.float32)
        valid = np.arange(MAX_LENGTH)[None, :] < selected_lengths[:, None]
        if not np.isfinite(selected_profiles[valid]).all():
            raise RuntimeError(f"Non-finite valid labels in {split}")
        if not np.isnan(selected_profiles[~valid]).all():
            raise RuntimeError(f"Non-NaN padding labels in {split}")
        np.save(cfg.subset_cache_dir / f"source_positions_{split}.npy", source_positions)
        np.save(cfg.subset_cache_dir / f"sequence_ids_{split}.npy", sequence_ids)
        np.save(cfg.subset_cache_dir / f"lengths_{split}.npy", selected_lengths)
        np.save(cfg.subset_cache_dir / f"profiles_{split}_fp32.npy", selected_profiles)

        final_path = cfg.subset_cache_dir / f"embeddings_{split}_fp16.npy"
        partial_path = cfg.subset_cache_dir / f"embeddings_{split}_fp16.partial.npy"
        if partial_path.exists() or final_path.exists():
            raise RuntimeError(f"Unexpected pre-existing subset embedding file for {split}")
        output = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.float16,
            shape=(len(source_positions), MAX_LENGTH, INPUT_DIM),
        )
        started = time.perf_counter()
        copy_chunk = 256
        for start in range(0, len(source_positions), copy_chunk):
            end = min(start + copy_chunk, len(source_positions))
            output[start:end] = source_embeddings[source_positions[start:end]]
            if end == len(source_positions) or end % 10_000 == 0:
                elapsed = time.perf_counter() - started
                print(
                    json.dumps(
                        {
                            "subset_copy": split,
                            "completed": end,
                            "total": len(source_positions),
                            "rows_per_second": end / max(elapsed, 1e-9),
                        }
                    ),
                    flush=True,
                )
        output.flush()
        del output
        partial_path.replace(final_path)
    write_json(index_path, plan)
    return plan


def valid_mask(lengths: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(MAX_LENGTH, device=lengths.device)[None, :]
    return positions < lengths[:, None]


class TokenLinearHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(INPUT_DIM)
        self.output = nn.Linear(INPUT_DIM, 1)

    def forward(self, embeddings: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        del lengths
        return torch.sigmoid(self.output(self.norm(embeddings)).squeeze(-1))


class ResidualConvBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(
            width,
            width,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        update = self.norm(hidden).transpose(1, 2)
        update = self.conv(update).transpose(1, 2)
        update = self.dropout(F.gelu(update))
        return (hidden + update) * mask.unsqueeze(-1)


class LocalCNNHead(nn.Module):
    """Two dilated 5-mers yield an explicit 13-nt local receptive field."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(INPUT_DIM)
        self.projection = nn.Linear(INPUT_DIM, hidden_dim)
        self.blocks = nn.ModuleList(
            (
                ResidualConvBlock(hidden_dim, dilation=1, dropout=dropout),
                ResidualConvBlock(hidden_dim, dilation=2, dropout=dropout),
            )
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, embeddings: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = valid_mask(lengths)
        hidden = self.projection(self.input_norm(embeddings)) * mask.unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden, mask)
        return torch.sigmoid(self.output(self.output_norm(hidden)).squeeze(-1))


class FullTransformerHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden-dim must be divisible by attention-heads")
        self.input_norm = nn.LayerNorm(INPUT_DIM)
        self.projection = nn.Linear(INPUT_DIM, hidden_dim)
        self.position = nn.Embedding(MAX_LENGTH, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=4 * hidden_dim,
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
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, embeddings: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = valid_mask(lengths)
        positions = torch.arange(MAX_LENGTH, device=embeddings.device)
        hidden = self.projection(self.input_norm(embeddings))
        hidden = hidden + self.position(positions)[None, :, :]
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        return torch.sigmoid(self.output(hidden).squeeze(-1))


def make_model(name: str, cfg: argparse.Namespace) -> nn.Module:
    if name == "token_linear":
        return TokenLinearHead()
    if name == "local_cnn":
        return LocalCNNHead(cfg.hidden_dim, cfg.dropout)
    if name == "full_transformer":
        return FullTransformerHead(
            cfg.hidden_dim,
            cfg.attention_heads,
            cfg.transformer_layers,
            cfg.dropout,
        )
    raise ValueError(name)


def masked_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    mask = valid_mask(lengths)
    finite_target = torch.nan_to_num(target, nan=0.0)
    point_loss = F.huber_loss(
        prediction.float(),
        finite_target.float(),
        reduction="none",
        delta=float(delta),
    )
    per_sequence = (point_loss * mask).sum(dim=1) / mask.sum(dim=1)
    return per_sequence.mean()


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


@torch.inference_mode()
def infer(
    model: nn.Module,
    store: SubsetSplit,
    device: torch.device,
    batch_size: int,
    huber_delta: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    predictions = np.full((len(store), MAX_LENGTH), np.nan, dtype=np.float32)
    targets = np.full((len(store), MAX_LENGTH), np.nan, dtype=np.float32)
    total_loss = 0.0
    for start, end in block_ranges(len(store), batch_size, seed=None):
        embeddings, lengths, target = store.batch(start, end, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = model(embeddings, lengths)
            loss = masked_huber_loss(prediction, target, lengths, huber_delta)
        predictions[start:end] = prediction.float().cpu().numpy()
        targets[start:end] = target.float().cpu().numpy()
        total_loss += float(loss) * (end - start)
    return predictions, targets, total_loss / len(store)


def train_head(
    name: str,
    cfg: argparse.Namespace,
    stores: dict[str, SubsetSplit],
    device: torch.device,
) -> dict[str, Any]:
    output = cfg.output_dir / name
    metrics_path = output / "metrics.json"
    if metrics_path.is_file() and not cfg.force:
        print(f"Reusing completed head: {name}", flush=True)
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    seed_all(int(cfg.seed))
    model = make_model(name, cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    history: list[dict[str, Any]] = []
    best_validation_pearson = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    train_started = time.perf_counter()

    for epoch in range(1, int(cfg.epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_sequences = 0
        for start, end in block_ranges(
            len(stores["train"]),
            int(cfg.batch_size),
            seed=int(cfg.seed) + epoch,
        ):
            embeddings, lengths, target = stores["train"].batch(start, end, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = model(embeddings, lengths)
                loss = masked_huber_loss(
                    prediction,
                    target,
                    lengths,
                    float(cfg.huber_delta),
                )
            loss.backward()
            if cfg.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
            total_loss += float(loss.detach()) * (end - start)
            total_sequences += end - start

        val_prediction, val_target, val_loss = infer(
            model,
            stores["val"],
            device,
            int(cfg.batch_size),
            float(cfg.huber_delta),
        )
        val_valid, _ = region_masks(np.asarray(stores["val"].lengths))
        validation_pearson = safe_pearson(
            val_target[val_valid], val_prediction[val_valid]
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_sequences,
            "val_loss": val_loss,
            "val_micro_pearson": validation_pearson,
            "val_micro_mae": float(
                np.mean(np.abs(val_prediction[val_valid] - val_target[val_valid]))
            ),
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(json.dumps({"head": name, **row}), flush=True)
        if validation_pearson > best_validation_pearson + 1e-5:
            best_validation_pearson = validation_pearson
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(cfg.patience):
                break

    if best_state is None:
        raise RuntimeError(f"No checkpoint was selected for {name}")
    model.load_state_dict(best_state, strict=True)
    val_prediction, val_target, val_loss = infer(
        model,
        stores["val"],
        device,
        int(cfg.batch_size),
        float(cfg.huber_delta),
    )
    test_prediction, test_target, test_loss = infer(
        model,
        stores["test"],
        device,
        int(cfg.batch_size),
        float(cfg.huber_delta),
    )
    validation_metrics = detailed_profile_metrics(
        val_target,
        val_prediction,
        np.asarray(stores["val"].lengths),
        include_spearman=True,
    )
    test_metrics = detailed_profile_metrics(
        test_target,
        test_prediction,
        np.asarray(stores["test"].lengths),
        include_spearman=True,
    )

    with (output / "history.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(history)
    np.savez_compressed(
        output / "test_predictions.npz",
        sequence_ids=np.asarray(stores["test"].sequence_ids),
        lengths=np.asarray(stores["test"].lengths),
        observed=test_target.astype(np.float16),
        predicted=test_prediction.astype(np.float16),
    )
    architecture = {
        "name": name,
        "input_dim": INPUT_DIM,
        "hidden_dim": int(cfg.hidden_dim) if name != "token_linear" else None,
        "attention_heads": int(cfg.attention_heads) if name == "full_transformer" else None,
        "transformer_layers": (
            int(cfg.transformer_layers) if name == "full_transformer" else None
        ),
        "local_cnn_receptive_field_nt": 13 if name == "local_cnn" else None,
        "dropout": float(cfg.dropout) if name != "token_linear" else 0.0,
    }
    torch.save(
        {
            "state_dict": best_state,
            "architecture": architecture,
            "seed": int(cfg.seed),
            "best_epoch": int(best_epoch),
            "best_validation_micro_pearson": float(best_validation_pearson),
            "label": "per-position universal-median GSE145805 accessibility",
            "backbone": "frozen public mRNABERT token embeddings",
        },
        output / "best_model.pt",
    )
    result = {
        "head": name,
        "architecture": architecture,
        "parameter_count": parameter_count,
        "best_epoch": int(best_epoch),
        "epochs_completed": int(len(history)),
        "best_validation_micro_pearson": float(best_validation_pearson),
        "validation_huber_loss": float(val_loss),
        "test_huber_loss": float(test_loss),
        "validation": validation_metrics,
        "test": test_metrics,
        "training_elapsed_seconds": float(time.perf_counter() - train_started),
    }
    write_json(metrics_path, result)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def write_comparison(
    results: dict[str, dict[str, Any]],
    path: Path,
) -> list[dict[str, Any]]:
    rows = []
    for name, result in results.items():
        row = {
            "head": name,
            "parameters": result["parameter_count"],
            "best_epoch": result["best_epoch"],
            "val_micro_pearson": result["validation"]["micro_nucleotide"]["pearson"],
            "test_micro_pearson": result["test"]["micro_nucleotide"]["pearson"],
            "test_micro_spearman": result["test"]["micro_nucleotide"]["spearman"],
            "test_micro_mae": result["test"]["micro_nucleotide"]["mae"],
            "test_within_sequence_pearson_mean": result["test"]["within_sequence_pearson"]["mean"],
            "test_within_sequence_pearson_median": result["test"]["within_sequence_pearson"]["median"],
            "test_upstream_mean_pearson": result["test"]["region_mean"]["upstream"]["pearson"],
            "test_middle25_mean_pearson": result["test"]["region_mean"]["middle25"]["pearson"],
            "test_downstream_mean_pearson": result["test"]["region_mean"]["downstream"]["pearson"],
            "test_derived_s_access_pearson": result["test"]["derived_s_access"]["pearson"],
            "elapsed_seconds": result["training_elapsed_seconds"],
        }
        rows.append(row)
    rows.sort(key=lambda row: row["val_micro_pearson"], reverse=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    cfg = parse_args()
    required = [
        cfg.universal_dir / "median_accessibility_profiles.npy",
        cfg.universal_dir / "lengths.npy",
        cfg.universal_dir / "splits.npy",
        cfg.full_cache_dir / "cache_index.json",
    ]
    for split in SPLITS:
        required.extend(
            [
                cfg.full_cache_dir / f"embeddings_{split}_fp16.npy",
                cfg.full_cache_dir / f"sequence_ids_{split}.npy",
                cfg.full_cache_dir / f"lengths_{split}.npy",
            ]
        )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if cfg.batch_size <= 0 or cfg.epochs <= 0 or cfg.patience <= 0:
        raise ValueError("batch-size, epochs, and patience must be positive")
    if len(set(cfg.heads)) != len(cfg.heads):
        raise ValueError("Head names must be unique")
    final_path = cfg.output_dir / "comparison.json"
    if final_path.is_file() and not cfg.force:
        raise FileExistsError(f"Completed comparison already exists: {final_path}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(int(cfg.seed))
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = resolve_device(cfg.device)
    total_started = time.perf_counter()

    subset_manifest = build_subset_cache(cfg)
    stores = {split: SubsetSplit(cfg.subset_cache_dir, split) for split in SPLITS}
    results: dict[str, dict[str, Any]] = {}
    for name in cfg.heads:
        results[name] = train_head(name, cfg, stores, device)
    comparison_rows = write_comparison(results, cfg.output_dir / "comparison.tsv")
    best_name = comparison_rows[0]["head"]
    comparison = {
        "status": "pass",
        "question": (
            "Is per-nucleotide accessibility already localized in frozen mRNABERT "
            "tokens, or does it require local/full-context redistribution?"
        ),
        "selection_rule": (
            "Each head's checkpoint is selected only by validation micro-nucleotide "
            "Pearson; the test split is evaluated once after selection."
        ),
        "best_head_by_validation": best_name,
        "best_head_test_metrics": results[best_name]["test"],
        "heads": results,
        "comparison_rows": comparison_rows,
        "data": {
            "label": "per-position median accessibility across identical-sequence human observations",
            "split": "transcript-component-disjoint fixed seed-42 split",
            "subset_counts": {split: len(stores[split]) for split in SPLITS},
            "full_counts": {
                split: subset_manifest["splits"][split]["source_count"]
                for split in SPLITS
            },
            "subset_manifest": subset_manifest,
        },
        "training": {
            "seed": int(cfg.seed),
            "batch_size": int(cfg.batch_size),
            "maximum_epochs": int(cfg.epochs),
            "patience": int(cfg.patience),
            "optimizer": "AdamW",
            "learning_rate": float(cfg.learning_rate),
            "weight_decay": float(cfg.weight_decay),
            "loss": f"per-sequence mean masked Huber(delta={cfg.huber_delta:g})",
            "mixed_precision": "BF16" if device.type == "cuda" else "disabled",
            "backbone": "frozen; cached FP16 mRNABERT token embeddings",
        },
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "elapsed_seconds": float(time.perf_counter() - total_started),
        },
    }
    write_json(final_path, comparison)
    print(
        json.dumps(
            {
                "best_head_by_validation": best_name,
                "comparison_rows": comparison_rows,
                "result": str(final_path),
            },
            ensure_ascii=False,
            indent=2,
            default=json_default,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
