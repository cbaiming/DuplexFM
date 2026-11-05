#!/usr/bin/env python3
"""Fine-tune mRNABERT with LoRA for per-nucleotide RNA accessibility.

The public mRNABERT weights remain frozen.  Rank-limited LoRA updates are
injected into every fused attention Wqkv projection.  Nucleotide-aligned
mRNABERT states are then processed by a small residual Transformer head to
predict one accessibility value for every valid base in a 50--60 nt fragment.

This program deliberately reuses the audited Universal-median labels, fixed
transcript-component split, loss, validation metrics, and checkpoint-selection
rule of the frozen-embedding baseline so the two paths can be compared fairly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

import rna_backbone as rb
import route2_dual_lora as route2
from tools import train_gse145805_full_residual_transformer_per_base as baseline


SPLITS = baseline.SPLITS
MAX_LENGTH = baseline.MAX_LENGTH
INPUT_DIM = baseline.INPUT_DIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universal-dir", required=True, type=Path)
    parser.add_argument("--split-cache-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--micro-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.03)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
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


class RawProfileSplit:
    """Raw sequences in the exact order used by the frozen embedding cache."""

    def __init__(
        self,
        split_cache_dir: Path,
        sequences: np.ndarray,
        profiles: np.ndarray,
        universal_lengths: np.ndarray,
        universal_splits: np.ndarray,
        split: str,
        split_code: int,
        limit: int,
        seed: int,
    ) -> None:
        self.split = split
        self.sequences = sequences
        self.profiles = profiles
        self.source_sequence_ids = np.load(
            split_cache_dir / f"sequence_ids_{split}.npy", mmap_mode="r"
        )
        self.source_lengths = np.load(
            split_cache_dir / f"lengths_{split}.npy", mmap_mode="r"
        )
        total = len(self.source_sequence_ids)
        if self.source_lengths.shape != (total,):
            raise RuntimeError(f"Unexpected {split} length vector")
        if limit:
            if limit > total:
                raise ValueError(
                    f"Requested {limit} rows from {split}, but only {total} exist"
                )
            rng = np.random.default_rng(seed + 1009 * (split_code + 1))
            self.positions: np.ndarray | None = np.sort(
                rng.choice(total, size=limit, replace=False)
            ).astype(np.int64)
        else:
            self.positions = None
        selected_ids = self.sequence_ids_numpy()
        selected_lengths = self.lengths_numpy()
        if not np.array_equal(
            selected_lengths, np.asarray(universal_lengths[selected_ids])
        ):
            raise RuntimeError(f"Universal lengths do not align for {split}")
        if not np.all(universal_splits[selected_ids] == split_code):
            raise RuntimeError(f"Universal split assignments do not align for {split}")

    def __len__(self) -> int:
        return (
            len(self.positions)
            if self.positions is not None
            else len(self.source_sequence_ids)
        )

    def _source_index(self, start: int, end: int) -> slice | np.ndarray:
        if self.positions is None:
            return slice(start, end)
        return self.positions[start:end]

    def sequence_ids_numpy(self) -> np.ndarray:
        if self.positions is None:
            return np.asarray(self.source_sequence_ids)
        return np.asarray(self.source_sequence_ids[self.positions])

    def lengths_numpy(self) -> np.ndarray:
        if self.positions is None:
            return np.asarray(self.source_lengths)
        return np.asarray(self.source_lengths[self.positions])

    def batch_numpy(
        self, start: int, end: int
    ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
        source_index = self._source_index(start, end)
        sequence_ids = np.asarray(
            self.source_sequence_ids[source_index], dtype=np.int64
        )
        lengths = np.asarray(self.source_lengths[source_index], dtype=np.int64)
        encoded_sequences = np.asarray(self.sequences[sequence_ids])
        sequences = [
            bytes(value).decode("ascii").rstrip("\x00")
            for value in encoded_sequences
        ]
        if any(len(sequence) != int(length) for sequence, length in zip(sequences, lengths)):
            raise RuntimeError(f"Stored sequence length mismatch in {self.split}")
        targets = np.asarray(self.profiles[sequence_ids], dtype=np.float32)
        return sequences, lengths, targets, sequence_ids


class LoRAResidualPerBase(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int,
        heads: int,
        layers: int,
        ffn_dim: int,
        dropout: float,
        residual_gate_init: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = baseline.ResidualFullTransformerPerBase(
            input_dim=INPUT_DIM,
            hidden_dim=hidden_dim,
            heads=heads,
            layers=layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            residual_gate_init=residual_gate_init,
        )

    def forward(
        self,
        encoded: dict[str, torch.Tensor],
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        output = self.encoder(**encoded)
        hidden = rb.extract_last_hidden_state(output)
        if hidden.ndim != 3 or hidden.shape[2] != INPUT_DIM:
            raise RuntimeError(f"Unexpected mRNABERT output shape: {tuple(hidden.shape)}")
        if hidden.shape[1] < MAX_LENGTH + 1:
            raise RuntimeError(f"mRNABERT output is too short: {tuple(hidden.shape)}")
        tokens = hidden[:, 1 : MAX_LENGTH + 1]
        mask = baseline.valid_mask(lengths)
        tokens = tokens * mask.unsqueeze(-1)
        return self.head(tokens, lengths)

    def residual_gate(self) -> torch.Tensor:
        return self.head.residual_gate()


def tokenize_batch(
    tokenizer: Any,
    sequences: list[str],
    lengths_numpy: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    encoded = tokenizer(
        sequences,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH + 2,
        return_tensors="pt",
    )
    observed_token_lengths = encoded["attention_mask"].sum(dim=1).cpu().numpy() - 2
    if not np.array_equal(observed_token_lengths, lengths_numpy):
        raise RuntimeError(
            "mRNABERT token-to-nucleotide alignment failed: "
            f"observed={observed_token_lengths[:8].tolist()}, "
            f"expected={lengths_numpy[:8].tolist()}"
        )
    encoded = {
        key: value.to(device, non_blocking=True)
        for key, value in encoded.items()
    }
    lengths = torch.from_numpy(np.array(lengths_numpy, copy=True)).to(
        device, non_blocking=True
    )
    return encoded, lengths


def capture_trainable_state(model: LoRAResidualPerBase) -> dict[str, Any]:
    return {
        "lora_state": {
            key: value.detach().cpu().clone()
            for key, value in get_peft_model_state_dict(model.encoder).items()
        },
        "head_state": {
            key: value.detach().cpu().clone()
            for key, value in model.head.state_dict().items()
        },
    }


def load_trainable_state(model: LoRAResidualPerBase, state: dict[str, Any]) -> None:
    incompatible = set_peft_model_state_dict(model.encoder, state["lora_state"])
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if unexpected:
        raise RuntimeError(f"Unexpected LoRA state keys: {unexpected[:8]}")
    model.head.load_state_dict(state["head_state"], strict=True)


def prepare_batch(
    store: RawProfileSplit,
    start: int,
    end: int,
    tokenizer: Any,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, np.ndarray]:
    sequences, lengths_numpy, targets_numpy, sequence_ids = store.batch_numpy(
        start, end
    )
    encoded, lengths = tokenize_batch(
        tokenizer, sequences, lengths_numpy, device
    )
    targets = torch.from_numpy(np.array(targets_numpy, copy=True)).to(
        device, non_blocking=True
    )
    return encoded, lengths, targets, sequence_ids


@torch.inference_mode()
def evaluate_basic(
    model: LoRAResidualPerBase,
    store: RawProfileSplit,
    tokenizer: Any,
    device: torch.device,
    cfg: argparse.Namespace,
) -> tuple[dict[str, float], float]:
    model.eval()
    accumulator = baseline.StreamingValidation()
    total_loss = 0.0
    total_rows = 0
    for start, end in baseline.block_ranges(
        len(store), cfg.eval_batch_size, seed=None
    ):
        encoded, lengths, target, _ = prepare_batch(
            store, start, end, tokenizer, device
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = model(encoded, lengths)
            loss = baseline.masked_huber_loss(
                prediction, target, lengths, cfg.huber_delta
            )
        accumulator.update(
            target.float().cpu().numpy(),
            prediction.float().cpu().numpy(),
            lengths.cpu().numpy(),
        )
        rows = end - start
        total_loss += float(loss) * rows
        total_rows += rows
    return (
        accumulator.result(cfg.micro_weight, cfg.within_sequence_weight),
        total_loss / total_rows,
    )


@torch.inference_mode()
def evaluate_detailed(
    model: LoRAResidualPerBase,
    store: RawProfileSplit,
    tokenizer: Any,
    device: torch.device,
    cfg: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    observed = np.full((len(store), MAX_LENGTH), np.nan, dtype=np.float32)
    predicted = np.full((len(store), MAX_LENGTH), np.nan, dtype=np.float32)
    for start, end in baseline.block_ranges(
        len(store), cfg.eval_batch_size, seed=None
    ):
        encoded, lengths, target, _ = prepare_batch(
            store, start, end, tokenizer, device
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = model(encoded, lengths)
        observed[start:end] = target.float().cpu().numpy()
        predicted[start:end] = prediction.float().cpu().numpy()
    metrics = baseline.diagnostic.detailed_profile_metrics(
        observed,
        predicted,
        store.lengths_numpy(),
        include_spearman=True,
    )
    return metrics, observed, predicted


def validate_args(cfg: argparse.Namespace) -> None:
    positive_integer_names = (
        "micro_batch_size",
        "eval_batch_size",
        "gradient_accumulation",
        "epochs",
        "patience",
        "lora_rank",
        "lora_alpha",
    )
    for name in positive_integer_names:
        if int(getattr(cfg, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if not 0.0 <= cfg.warmup_fraction < 1.0:
        raise ValueError("warmup-fraction must lie in [0, 1)")
    if not math.isclose(
        cfg.micro_weight + cfg.within_sequence_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("micro-weight and within-sequence-weight must sum to one")


def main() -> None:
    cfg = parse_args()
    validate_args(cfg)
    required = [
        cfg.universal_dir / "sequences.npy",
        cfg.universal_dir / "median_accessibility_profiles.npy",
        cfg.universal_dir / "lengths.npy",
        cfg.universal_dir / "splits.npy",
        cfg.model_dir / "config.json",
        cfg.model_dir / "pytorch_model.bin",
    ]
    for split in SPLITS:
        required.extend(
            [
                cfg.split_cache_dir / f"sequence_ids_{split}.npy",
                cfg.split_cache_dir / f"lengths_{split}.npy",
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
    device = baseline.resolve_device(cfg.device)
    started = time.perf_counter()

    sequences = np.load(cfg.universal_dir / "sequences.npy", mmap_mode="r")
    profiles_path = cfg.universal_dir / "median_accessibility_profiles.npy"
    profiles = np.load(profiles_path, mmap_mode="r")
    universal_lengths = np.load(
        cfg.universal_dir / "lengths.npy", mmap_mode="r"
    )
    universal_splits = np.load(
        cfg.universal_dir / "splits.npy", mmap_mode="r"
    )
    if profiles.shape != (len(sequences), MAX_LENGTH):
        raise RuntimeError(f"Unexpected profile matrix shape: {profiles.shape}")
    limits = (cfg.max_train_rows, cfg.max_val_rows, cfg.max_test_rows)
    stores = {
        split: RawProfileSplit(
            cfg.split_cache_dir,
            sequences,
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

    tokenizer = rb.load_gene_tokenizer(str(cfg.model_dir), None)
    backbone = rb.load_gene_model(str(cfg.model_dir), None)
    if int(backbone.config.hidden_size) != INPUT_DIM:
        raise RuntimeError(
            f"Unexpected mRNABERT hidden size: {backbone.config.hidden_size}"
        )
    backbone.requires_grad_(False)
    target_summary = route2.discover_lora_targets(
        backbone=backbone,
        requested="auto",
        label="mRNABERT accessibility",
    )
    if target_summary["target_modules"] != ["Wqkv"]:
        raise RuntimeError(f"Unexpected mRNABERT LoRA targets: {target_summary}")
    encoder = get_peft_model(
        backbone,
        LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=target_summary["target_modules"],
            lora_dropout=cfg.lora_dropout,
            bias="none",
        ),
    )
    model = LoRAResidualPerBase(
        encoder=encoder,
        hidden_dim=cfg.hidden_dim,
        heads=cfg.attention_heads,
        layers=cfg.layers,
        ffn_dim=cfg.ffn_dim,
        dropout=cfg.dropout,
        residual_gate_init=cfg.residual_gate_init,
    ).to(device)

    lora_parameters = [
        parameter
        for name, parameter in model.encoder.named_parameters()
        if parameter.requires_grad and "lora_" in name
    ]
    unexpected_encoder_parameters = [
        name
        for name, parameter in model.encoder.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if not lora_parameters or unexpected_encoder_parameters:
        raise RuntimeError(
            "Invalid trainable encoder parameters: "
            f"lora={len(lora_parameters)}, unexpected={unexpected_encoder_parameters[:8]}"
        )
    head_parameters = list(model.head.parameters())
    trainable_parameters = [*lora_parameters, *head_parameters]
    trainable_parameter_count = int(
        sum(parameter.numel() for parameter in trainable_parameters)
    )
    lora_parameter_count = int(
        sum(parameter.numel() for parameter in lora_parameters)
    )
    head_parameter_count = int(
        sum(parameter.numel() for parameter in head_parameters)
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_parameters,
                "lr": cfg.lora_learning_rate,
                "weight_decay": cfg.weight_decay,
                "group_name": "lora",
            },
            {
                "params": head_parameters,
                "lr": cfg.head_learning_rate,
                "weight_decay": cfg.weight_decay,
                "group_name": "head",
            },
        ]
    )
    micro_batches_per_epoch = math.ceil(
        len(stores["train"]) / cfg.micro_batch_size
    )
    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch / cfg.gradient_accumulation
    )
    total_optimizer_steps = optimizer_steps_per_epoch * cfg.epochs
    scheduler, warmup_steps = baseline.make_scheduler(
        optimizer, total_optimizer_steps, cfg.warmup_fraction
    )

    architecture = {
        "backbone": "public mRNABERT with trainable LoRA and frozen base weights",
        "tokenization_mode": rb.MRNABERT_TOKENIZATION_MODE,
        "lora_target_modules": target_summary["target_modules"],
        "lora_matched_modules": target_summary["matched_full_names"],
        "lora_rank": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "input_dim": INPUT_DIM,
        "hidden_dim": cfg.hidden_dim,
        "attention_heads": cfg.attention_heads,
        "layers": cfg.layers,
        "ffn_dim": cfg.ffn_dim,
        "head_dropout": cfg.dropout,
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
            "profile_sha256": baseline.sha256_file(profiles_path),
            "split_cache_index_sha256": baseline.sha256_file(
                cfg.split_cache_dir / "cache_index.json"
            ),
        },
        "optimization": {
            "optimizer": "AdamW with separate LoRA and head learning rates",
            "micro_batch_size": cfg.micro_batch_size,
            "eval_batch_size": cfg.eval_batch_size,
            "gradient_accumulation": cfg.gradient_accumulation,
            "nominal_effective_batch_size": (
                cfg.micro_batch_size * cfg.gradient_accumulation
            ),
            "epochs": cfg.epochs,
            "patience": cfg.patience,
            "lora_learning_rate": cfg.lora_learning_rate,
            "head_learning_rate": cfg.head_learning_rate,
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
        "parameters": {
            "lora": lora_parameter_count,
            "head": head_parameter_count,
            "total_trainable": trainable_parameter_count,
        },
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        },
    }
    baseline.write_json(cfg.output_dir / "run_config.json", run_config)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
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
        load_trainable_state(model, checkpoint["trainable_state"])
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
                "trainable_parameters": trainable_parameter_count,
                "lora_parameters": lora_parameter_count,
                "head_parameters": head_parameter_count,
                "micro_batches_per_epoch": micro_batches_per_epoch,
                "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "total_optimizer_steps": total_optimizer_steps,
                "warmup_steps": warmup_steps,
                "split_sizes": {
                    split: len(store) for split, store in stores.items()
                },
                "initial_residual_gate": float(
                    model.residual_gate().detach().cpu()
                ),
            }
        ),
        flush=True,
    )

    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        blocks = baseline.block_ranges(
            len(stores["train"]),
            cfg.micro_batch_size,
            seed=cfg.seed + epoch,
        )
        train_loss_sum = 0.0
        train_rows = 0
        for group_start in range(0, len(blocks), cfg.gradient_accumulation):
            group = blocks[
                group_start : group_start + cfg.gradient_accumulation
            ]
            group_rows = sum(end - start for start, end in group)
            optimizer.zero_grad(set_to_none=True)
            for start, end in group:
                encoded, lengths, target, _ = prepare_batch(
                    stores["train"], start, end, tokenizer, device
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    prediction = model(encoded, lengths)
                    loss = baseline.masked_huber_loss(
                        prediction, target, lengths, cfg.huber_delta
                    )
                rows = end - start
                (loss * (rows / group_rows)).backward()
                train_loss_sum += float(loss.detach()) * rows
                train_rows += rows
            if cfg.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, cfg.gradient_clip
                )
            optimizer.step()
            scheduler.step()

        validation, val_loss = evaluate_basic(
            model, stores["val"], tokenizer, device, cfg
        )
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
            "lora_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "head_learning_rate": float(optimizer.param_groups[1]["lr"]),
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["selection_score"] > best_score + cfg.min_improvement:
            best_score = validation["selection_score"]
            best_epoch = epoch
            best_state = capture_trainable_state(model)
            stale_epochs = 0
        else:
            stale_epochs += 1

        baseline.atomic_torch_save(
            {
                "epoch": epoch,
                "architecture": architecture,
                "trainable_state": capture_trainable_state(model),
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
    load_trainable_state(model, best_state)
    validation_metrics, _, _ = evaluate_detailed(
        model, stores["val"], tokenizer, device, cfg
    )
    test_metrics, test_observed, test_predicted = evaluate_detailed(
        model, stores["test"], tokenizer, device, cfg
    )
    with (cfg.output_dir / "history.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(history[0]), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(history)
    np.savez_compressed(
        cfg.output_dir / "test_predictions.npz",
        sequence_ids=stores["test"].sequence_ids_numpy(),
        lengths=stores["test"].lengths_numpy(),
        observed=test_observed.astype(np.float16),
        predicted=test_predicted.astype(np.float16),
    )
    model.encoder.save_pretrained(
        cfg.output_dir / "best_mrnabert_lora",
        safe_serialization=True,
    )
    final_gate = float(model.residual_gate().detach().cpu())
    baseline.atomic_torch_save(
        {
            "head_state_dict": best_state["head_state"],
            "architecture": architecture,
            "seed": cfg.seed,
            "best_epoch": best_epoch,
            "best_validation_selection_score": best_score,
            "residual_gate": final_gate,
            "label": "per-position universal-median GSE145805 accessibility",
            "backbone": "public mRNABERT plus best_mrnabert_lora adapter",
        },
        cfg.output_dir / "best_head.pt",
    )
    result = {
        "status": "pass",
        "architecture": architecture,
        "trainable_parameters": run_config["parameters"],
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
    baseline.write_json(metrics_path, result)
    print(
        json.dumps(
            {
                "status": "complete",
                "best_epoch": best_epoch,
                "best_validation_selection_score": best_score,
                "residual_gate": final_gate,
                "test_micro_pearson": test_metrics["micro_nucleotide"][
                    "pearson"
                ],
                "test_within_sequence_pearson": test_metrics[
                    "within_sequence_pearson"
                ]["mean"],
                "test_derived_s_access_pearson": test_metrics[
                    "derived_s_access"
                ]["pearson"],
                "metrics": str(metrics_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
