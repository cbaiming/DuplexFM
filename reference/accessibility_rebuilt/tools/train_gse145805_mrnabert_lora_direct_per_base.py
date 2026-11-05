#!/usr/bin/env python3
"""Train a direct mRNABERT-LoRA nucleotide-accessibility regressor.

Architecture
------------
    mRNABERT + LoRA(Wqkv) -> token LayerNorm -> Linear(768, 1) -> sigmoid

The model predicts one continuous accessibility value for every valid base in
a 50--60 nt fragment.  The objective is the mean absolute error (L1): loss is
first averaged across valid nucleotides within each sequence and then across
the batch.  This gives every sequence equal weight regardless of length.

The default LoRA configuration and optimizer parameter groups match the
mRNABERT side of the DuplexFM Cross/Score experts: r=16, alpha=32, LoRA
dropout=0.05, fused Wqkv targets, mRNABERT-LoRA LR=2e-5, and task-head
LR=1e-4.  Public backbone weights stay frozen.
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
from tools import train_gse145805_mrnabert_lora_residual_transformer_per_base as shared


SPLITS = baseline.SPLITS
MAX_LENGTH = baseline.MAX_LENGTH
INPUT_DIM = baseline.INPUT_DIM
RawProfileSplit = shared.RawProfileSplit
prepare_batch = shared.prepare_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universal-dir", required=True, type=Path)
    parser.add_argument("--split-cache-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--micro-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lora-learning-rate", type=float, default=2e-5)
    parser.add_argument("--linear-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
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


class DirectNucleotideHead(nn.Module):
    """A position-shared linear readout from contextual mRNABERT tokens."""

    def __init__(self, input_dim: int = INPUT_DIM) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(input_dim)
        self.output = nn.Linear(input_dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.output(self.token_norm(tokens)).squeeze(-1)
        return torch.sigmoid(logits)


class LoRADirectPerBase(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = DirectNucleotideHead(INPUT_DIM)

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
        return self.head(tokens)


def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Mean valid-base absolute error per sequence, followed by batch mean."""

    mask = baseline.valid_mask(lengths)
    target = torch.nan_to_num(target, nan=0.0)
    point_loss = torch.abs(prediction.float() - target.float())
    per_sequence = (point_loss * mask).sum(dim=1) / mask.sum(dim=1)
    return per_sequence.mean()


def capture_trainable_state(model: LoRADirectPerBase) -> dict[str, Any]:
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


def load_trainable_state(model: LoRADirectPerBase, state: dict[str, Any]) -> None:
    incompatible = set_peft_model_state_dict(model.encoder, state["lora_state"])
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if unexpected:
        raise RuntimeError(f"Unexpected LoRA state keys: {unexpected[:8]}")
    model.head.load_state_dict(state["head_state"], strict=True)


@torch.inference_mode()
def evaluate_basic(
    model: LoRADirectPerBase,
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
            loss = masked_l1_loss(prediction, target, lengths)
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
    model: LoRADirectPerBase,
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


def make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    optimizer_steps_per_epoch: int,
    epochs: int,
    warmup_epochs: float,
) -> tuple[torch.optim.lr_scheduler.LambdaLR, int, int]:
    total_steps = optimizer_steps_per_epoch * epochs
    warmup_steps = round(optimizer_steps_per_epoch * warmup_epochs)
    warmup_steps = min(max(warmup_steps, 0), max(total_steps - 1, 0))

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    return scheduler, warmup_steps, total_steps


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
    if cfg.warmup_epochs < 0 or cfg.warmup_epochs >= cfg.epochs:
        raise ValueError("warmup-epochs must lie in [0, epochs)")
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
        cfg.split_cache_dir / "cache_index.json",
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
        label="mRNABERT direct accessibility",
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
    model = LoRADirectPerBase(encoder).to(device)

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
    linear_parameters = list(model.head.parameters())
    trainable_parameters = [*lora_parameters, *linear_parameters]
    lora_parameter_count = int(
        sum(parameter.numel() for parameter in lora_parameters)
    )
    linear_parameter_count = int(
        sum(parameter.numel() for parameter in linear_parameters)
    )
    trainable_parameter_count = lora_parameter_count + linear_parameter_count

    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_parameters,
                "lr": cfg.lora_learning_rate,
                "weight_decay": cfg.weight_decay,
                "group_name": "mrnabert_lora",
            },
            {
                "params": linear_parameters,
                "lr": cfg.linear_learning_rate,
                "weight_decay": cfg.weight_decay,
                "group_name": "linear_head",
            },
        ]
    )
    micro_batches_per_epoch = math.ceil(
        len(stores["train"]) / cfg.micro_batch_size
    )
    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch / cfg.gradient_accumulation
    )
    scheduler, warmup_steps, total_optimizer_steps = make_warmup_cosine_scheduler(
        optimizer,
        optimizer_steps_per_epoch,
        cfg.epochs,
        cfg.warmup_epochs,
    )

    architecture = {
        "name": "mRNABERT-LoRA Direct",
        "backbone": "public mRNABERT with frozen base weights",
        "tokenization_mode": rb.MRNABERT_TOKENIZATION_MODE,
        "lora_target_modules": target_summary["target_modules"],
        "lora_matched_modules": target_summary["matched_full_names"],
        "lora_rank": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "head": "position-shared LayerNorm(768) + Linear(768,1) + sigmoid",
        "context_layer_after_mrnabert": None,
        "prediction": "one continuous value per valid nucleotide",
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
            "profile_target": "cross-cell-line median accessibility per nucleotide",
            "profile_sha256": baseline.sha256_file(profiles_path),
            "split_cache_index_sha256": baseline.sha256_file(
                cfg.split_cache_dir / "cache_index.json"
            ),
        },
        "optimization": {
            "optimizer": "AdamW with separate mRNABERT-LoRA and linear-head groups",
            "micro_batch_size": cfg.micro_batch_size,
            "eval_batch_size": cfg.eval_batch_size,
            "gradient_accumulation": cfg.gradient_accumulation,
            "nominal_effective_batch_size": (
                cfg.micro_batch_size * cfg.gradient_accumulation
            ),
            "epochs": cfg.epochs,
            "patience": cfg.patience,
            "lora_learning_rate": cfg.lora_learning_rate,
            "linear_learning_rate": cfg.linear_learning_rate,
            "weight_decay": cfg.weight_decay,
            "warmup_epochs": cfg.warmup_epochs,
            "warmup_steps": warmup_steps,
            "schedule": "linear warm-up then cosine decay to zero",
            "gradient_clip": cfg.gradient_clip,
            "loss": "per-sequence mean masked L1 (absolute error)",
            "selection_metric": (
                f"{cfg.micro_weight:g}*micro Pearson + "
                f"{cfg.within_sequence_weight:g}*mean within-sequence Pearson"
            ),
            "seed": cfg.seed,
            "precision": "BF16",
        },
        "parameters": {
            "lora": lora_parameter_count,
            "linear_head": linear_parameter_count,
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
                "architecture": architecture["name"],
                "trainable_parameters": trainable_parameter_count,
                "lora_parameters": lora_parameter_count,
                "linear_head_parameters": linear_parameter_count,
                "micro_batches_per_epoch": micro_batches_per_epoch,
                "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "total_optimizer_steps": total_optimizer_steps,
                "warmup_steps": warmup_steps,
                "split_sizes": {
                    split: len(store) for split, store in stores.items()
                },
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
                    loss = masked_l1_loss(prediction, target, lengths)
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
            "train_l1": train_loss_sum / train_rows,
            "val_l1": val_loss,
            "val_micro_pearson": validation["micro_pearson"],
            "val_mean_within_sequence_pearson": validation[
                "mean_within_sequence_pearson"
            ],
            "val_selection_score": validation["selection_score"],
            "val_mae": validation["mae"],
            "lora_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "linear_learning_rate": float(optimizer.param_groups[1]["lr"]),
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
    baseline.atomic_torch_save(
        {
            "head_state_dict": best_state["head_state"],
            "architecture": architecture,
            "seed": cfg.seed,
            "best_epoch": best_epoch,
            "best_validation_selection_score": best_score,
            "label": "per-position universal-median GSE145805 accessibility",
            "backbone": "public mRNABERT plus best_mrnabert_lora adapter",
        },
        cfg.output_dir / "best_linear_head.pt",
    )
    result = {
        "status": "pass",
        "architecture": architecture,
        "trainable_parameters": run_config["parameters"],
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_validation_selection_score": best_score,
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
