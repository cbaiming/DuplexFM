#!/usr/bin/env python3
"""Train the two-view, special-token-conditioned accessibility predictor.

Architecture
------------
    mRNABERT + LoRA(Wqkv)
      -> CLS/SEP-conditioned FiLM of nucleotide states
      -> position-shared LayerNorm(768) + Linear(768, 1) + sigmoid

The model predicts one accessibility value for each real nucleotide.  CLS and
SEP remain in the mRNABERT input and condition FiLM, but never receive labels
and never appear in the per-base output.  Training uses canonical and shifted
views; validation and test are restricted to the canonical view by contract.
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
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

import rna_backbone as rb
import route2_dual_lora as route2
from tools import diagnose_gse145805_mrnabert_per_base_heads as diagnostic
from tools import train_gse145805_full_residual_transformer_per_base as baseline
from tools import train_gse145805_mrnabert_lora_direct_per_base as direct
from tools import train_gse145805_mrnabert_lora_residual_transformer_per_base as shared


SPLITS = ("train", "val", "test")
MAX_LENGTH = 60
INPUT_DIM = 768
VIEW_CANONICAL = 0
VIEW_SHIFTED = 1
ARCHITECTURE_NAME = "mRNABERT-LoRA Direct CLS-SEP FiLM Two-View"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universal-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--micro-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lora-learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--film-hidden-dim", type=int, default=128)
    parser.add_argument("--film-dropout", type=float, default=0.05)
    parser.add_argument("--film-gamma-bound", type=float, default=0.50)
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


class TwoViewProfileSplit:
    """A view-aware raw sequence/profile split backed by memory maps."""

    def __init__(
        self,
        sequences: np.ndarray,
        profiles: np.ndarray,
        lengths: np.ndarray,
        splits: np.ndarray,
        view_ids: np.ndarray,
        split_name: str,
        split_code: int,
        limit: int,
        seed: int,
    ) -> None:
        self.sequences = sequences
        self.profiles = profiles
        self.all_lengths = lengths
        self.all_view_ids = view_ids
        self.split_name = split_name
        selected = splits == split_code
        if split_name != "train":
            selected &= view_ids == VIEW_CANONICAL
        self.source_sequence_ids = np.flatnonzero(selected).astype(np.int64)
        if limit:
            if limit > len(self.source_sequence_ids):
                raise ValueError(
                    f"Requested {limit} {split_name} rows, but only "
                    f"{len(self.source_sequence_ids)} are available"
                )
            rng = np.random.default_rng(seed + 1009 * (split_code + 1))
            positions = np.sort(
                rng.choice(len(self.source_sequence_ids), size=limit, replace=False)
            )
            self.source_sequence_ids = self.source_sequence_ids[positions]
        if split_name != "train" and np.any(
            view_ids[self.source_sequence_ids] != VIEW_CANONICAL
        ):
            raise RuntimeError(f"Non-canonical row found in {split_name}")

    def __len__(self) -> int:
        return len(self.source_sequence_ids)

    def sequence_ids_numpy(self) -> np.ndarray:
        return np.asarray(self.source_sequence_ids)

    def lengths_numpy(self) -> np.ndarray:
        return np.asarray(self.all_lengths[self.source_sequence_ids], dtype=np.int64)

    def view_ids_numpy(self) -> np.ndarray:
        return np.asarray(self.all_view_ids[self.source_sequence_ids], dtype=np.uint8)

    def batch_numpy(
        self, start: int, end: int
    ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
        sequence_ids = np.asarray(self.source_sequence_ids[start:end], dtype=np.int64)
        lengths = np.asarray(self.all_lengths[sequence_ids], dtype=np.int64)
        sequences = [
            bytes(value).decode("ascii").rstrip("\x00")
            for value in np.asarray(self.sequences[sequence_ids])
        ]
        if any(
            len(sequence) != int(length)
            for sequence, length in zip(sequences, lengths)
        ):
            raise RuntimeError(f"Stored sequence length mismatch in {self.split_name}")
        targets = np.asarray(self.profiles[sequence_ids], dtype=np.float32)
        return sequences, lengths, targets, sequence_ids


class SpecialTokenFiLMHead(nn.Module):
    """Use CLS and SEP states to modulate all real-nucleotide states."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        film_hidden_dim: int = 128,
        film_dropout: float = 0.05,
        gamma_bound: float = 0.50,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or film_hidden_dim <= 0:
            raise ValueError("FiLM dimensions must be positive")
        if gamma_bound <= 0:
            raise ValueError("gamma_bound must be positive")
        self.input_dim = int(input_dim)
        self.gamma_bound = float(gamma_bound)
        self.context_norm = nn.LayerNorm(2 * input_dim)
        self.context_hidden = nn.Sequential(
            nn.Linear(2 * input_dim, film_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(film_dropout)),
        )
        self.context_to_film = nn.Linear(film_hidden_dim, 2 * input_dim)
        self.token_norm = nn.LayerNorm(input_dim)
        self.output = nn.Linear(input_dim, 1)
        # The model starts as the old direct readout.  FiLM is learned only if
        # CLS/SEP context improves the objective.
        nn.init.zeros_(self.context_to_film.weight)
        nn.init.zeros_(self.context_to_film.bias)

    def forward(
        self,
        nucleotide_tokens: torch.Tensor,
        cls_state: torch.Tensor,
        sep_state: torch.Tensor,
    ) -> torch.Tensor:
        context = torch.cat((cls_state, sep_state), dim=-1)
        raw_gamma, beta = self.context_to_film(
            self.context_hidden(self.context_norm(context))
        ).chunk(2, dim=-1)
        gamma = self.gamma_bound * torch.tanh(raw_gamma)
        modulated = (1.0 + gamma[:, None, :]) * nucleotide_tokens
        modulated = modulated + beta[:, None, :]
        logits = self.output(self.token_norm(modulated)).squeeze(-1)
        return torch.sigmoid(logits)


class LoRASpecialTokenFiLMPerBase(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        film_hidden_dim: int = 128,
        film_dropout: float = 0.05,
        gamma_bound: float = 0.50,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = SpecialTokenFiLMHead(
            input_dim=INPUT_DIM,
            film_hidden_dim=film_hidden_dim,
            film_dropout=film_dropout,
            gamma_bound=gamma_bound,
        )

    @staticmethod
    def special_token_states(
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden.shape[0]
        if attention_mask.shape[:2] != hidden.shape[:2]:
            raise RuntimeError("attention_mask and mRNABERT states do not align")
        observed = attention_mask.sum(dim=1).to(lengths.dtype)
        if not torch.equal(observed, lengths + 2):
            raise RuntimeError("Expected exactly CLS + nucleotides + SEP tokens")
        sep_indices = lengths + 1
        rows = torch.arange(batch_size, device=hidden.device)
        if torch.any(attention_mask[rows, sep_indices] == 0):
            raise RuntimeError("SEP index points to padding")
        return hidden[:, 0], hidden[rows, sep_indices]

    def forward(
        self,
        encoded: dict[str, torch.Tensor],
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        output = self.encoder(**encoded)
        hidden = rb.extract_last_hidden_state(output)
        if hidden.ndim != 3 or hidden.shape[2] != INPUT_DIM:
            raise RuntimeError(f"Unexpected mRNABERT output shape: {tuple(hidden.shape)}")
        if hidden.shape[1] < MAX_LENGTH + 2:
            raise RuntimeError(f"mRNABERT output is too short: {tuple(hidden.shape)}")
        cls_state, sep_state = self.special_token_states(
            hidden, encoded["attention_mask"], lengths
        )
        nucleotide_tokens = hidden[:, 1 : MAX_LENGTH + 1]
        mask = baseline.valid_mask(lengths)
        prediction = self.head(nucleotide_tokens, cls_state, sep_state)
        return prediction * mask


def prepare_batch(
    store: TwoViewProfileSplit,
    start: int,
    end: int,
    tokenizer: Any,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, np.ndarray]:
    sequences, lengths_numpy, targets_numpy, sequence_ids = store.batch_numpy(
        start, end
    )
    encoded, lengths = shared.tokenize_batch(
        tokenizer, sequences, lengths_numpy, device
    )
    targets = torch.from_numpy(np.array(targets_numpy, copy=True)).to(
        device, non_blocking=True
    )
    return encoded, lengths, targets, sequence_ids


def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    mask = baseline.valid_mask(lengths)
    finite_target = torch.nan_to_num(target, nan=0.0)
    point_loss = torch.abs(prediction.float() - finite_target.float())
    per_sequence = (point_loss * mask).sum(dim=1) / mask.sum(dim=1)
    return per_sequence.mean()


def capture_trainable_state(
    model: LoRASpecialTokenFiLMPerBase,
) -> dict[str, Any]:
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


def load_trainable_state(
    model: LoRASpecialTokenFiLMPerBase, state: dict[str, Any]
) -> None:
    incompatible = set_peft_model_state_dict(model.encoder, state["lora_state"])
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if unexpected:
        raise RuntimeError(f"Unexpected LoRA state keys: {unexpected[:8]}")
    model.head.load_state_dict(state["head_state"], strict=True)


@torch.inference_mode()
def evaluate_basic(
    model: LoRASpecialTokenFiLMPerBase,
    store: TwoViewProfileSplit,
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


def boundary_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    lengths: np.ndarray,
) -> dict[str, Any]:
    positions = np.arange(MAX_LENGTH)[None, :]
    valid = positions < lengths[:, None]
    masks = {
        "first_10nt": valid & (positions < 10),
        "interior": valid & (positions >= 10) & (positions < (lengths - 10)[:, None]),
        "last_10nt": valid & (positions >= (lengths - 10)[:, None]),
    }
    return {
        name: diagnostic.continuous_metrics(
            observed[mask], predicted[mask], include_spearman=True
        )
        for name, mask in masks.items()
    }


@torch.inference_mode()
def evaluate_detailed(
    model: LoRASpecialTokenFiLMPerBase,
    store: TwoViewProfileSplit,
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
    lengths = store.lengths_numpy()
    metrics = diagnostic.detailed_profile_metrics(
        observed, predicted, lengths, include_spearman=True
    )
    metrics["boundary_nucleotide"] = boundary_metrics(
        observed, predicted, lengths
    )
    return metrics, observed, predicted


def validate_args(cfg: argparse.Namespace) -> None:
    for name in (
        "micro_batch_size",
        "eval_batch_size",
        "gradient_accumulation",
        "epochs",
        "patience",
        "lora_rank",
        "lora_alpha",
        "film_hidden_dim",
    ):
        if int(getattr(cfg, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if not 0.0 <= cfg.film_dropout < 1.0:
        raise ValueError("film-dropout must lie in [0,1)")
    if cfg.film_gamma_bound <= 0:
        raise ValueError("film-gamma-bound must be positive")
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
    cfg.universal_dir = cfg.universal_dir.resolve()
    cfg.model_dir = cfg.model_dir.resolve()
    cfg.output_dir = cfg.output_dir.resolve()
    required = [
        cfg.universal_dir / "sequences.npy",
        cfg.universal_dir / "median_accessibility_profiles.npy",
        cfg.universal_dir / "lengths.npy",
        cfg.universal_dir / "splits.npy",
        cfg.universal_dir / "view_ids.npy",
        cfg.universal_dir / "manifest.json",
        cfg.model_dir / "config.json",
        cfg.model_dir / "pytorch_model.bin",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    data_manifest = json.loads(
        (cfg.universal_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if data_manifest.get("status") != "pass" or data_manifest.get("format_version") != 1:
        raise RuntimeError("Unsupported or incomplete two-view dataset manifest")
    if (
        data_manifest.get("split_policy", {}).get("validation_and_test")
        != "canonical view only"
    ):
        raise RuntimeError("Dataset does not guarantee canonical-only evaluation")
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
    lengths = np.load(cfg.universal_dir / "lengths.npy", mmap_mode="r")
    splits = np.load(cfg.universal_dir / "splits.npy", mmap_mode="r")
    view_ids = np.load(cfg.universal_dir / "view_ids.npy", mmap_mode="r")
    row_count = len(sequences)
    if not (
        profiles.shape == (row_count, MAX_LENGTH)
        and lengths.shape == (row_count,)
        and splits.shape == (row_count,)
        and view_ids.shape == (row_count,)
    ):
        raise RuntimeError("Two-view dataset arrays have inconsistent shapes")
    if np.any(~np.isin(view_ids, (VIEW_CANONICAL, VIEW_SHIFTED))):
        raise RuntimeError("Unknown view ID")
    if np.any(view_ids[splits != 0] != VIEW_CANONICAL):
        raise RuntimeError("Shifted row found outside the training split")

    limits = (cfg.max_train_rows, cfg.max_val_rows, cfg.max_test_rows)
    stores = {
        name: TwoViewProfileSplit(
            sequences,
            profiles,
            lengths,
            splits,
            view_ids,
            name,
            code,
            limits[code],
            cfg.seed,
        )
        for code, name in enumerate(SPLITS)
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
        label="mRNABERT two-view FiLM accessibility",
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
    model = LoRASpecialTokenFiLMPerBase(
        encoder,
        film_hidden_dim=cfg.film_hidden_dim,
        film_dropout=cfg.film_dropout,
        gamma_bound=cfg.film_gamma_bound,
    ).to(device)

    lora_parameters = [
        parameter
        for name, parameter in model.encoder.named_parameters()
        if parameter.requires_grad and "lora_" in name
    ]
    unexpected_encoder = [
        name
        for name, parameter in model.encoder.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if not lora_parameters or unexpected_encoder:
        raise RuntimeError(
            "Invalid trainable encoder parameters: "
            f"lora={len(lora_parameters)}, unexpected={unexpected_encoder[:8]}"
        )
    head_parameters = list(model.head.parameters())
    trainable_parameters = [*lora_parameters, *head_parameters]
    lora_count = int(sum(parameter.numel() for parameter in lora_parameters))
    head_count = int(sum(parameter.numel() for parameter in head_parameters))

    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_parameters,
                "lr": cfg.lora_learning_rate,
                "weight_decay": cfg.weight_decay,
                "group_name": "mrnabert_lora",
            },
            {
                "params": head_parameters,
                "lr": cfg.head_learning_rate,
                "weight_decay": cfg.weight_decay,
                "group_name": "film_direct_head",
            },
        ]
    )
    micro_batches = math.ceil(len(stores["train"]) / cfg.micro_batch_size)
    optimizer_steps_per_epoch = math.ceil(
        micro_batches / cfg.gradient_accumulation
    )
    scheduler, warmup_steps, total_optimizer_steps = (
        direct.make_warmup_cosine_scheduler(
            optimizer,
            optimizer_steps_per_epoch,
            cfg.epochs,
            cfg.warmup_epochs,
        )
    )

    architecture = {
        "schema_version": 1,
        "name": ARCHITECTURE_NAME,
        "backbone": "public mRNABERT with frozen base weights",
        "tokenization_mode": rb.MRNABERT_TOKENIZATION_MODE,
        "special_token_policy": {
            "encoder_input": "CLS and SEP retained",
            "conditioning": "concatenate final CLS and SEP states for FiLM",
            "labels": "real nucleotides only",
            "prediction_output": "real nucleotides only",
        },
        "lora_target_modules": target_summary["target_modules"],
        "lora_matched_modules": target_summary["matched_full_names"],
        "lora_rank": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "film": {
            "context": "LayerNorm([CLS;SEP]) -> Linear(1536,H) -> GELU -> Dropout -> Linear(H,1536)",
            "hidden_dim": cfg.film_hidden_dim,
            "dropout": cfg.film_dropout,
            "gamma_bound": cfg.film_gamma_bound,
            "gamma": f"{cfg.film_gamma_bound:g}*tanh(raw_gamma)",
            "beta": "unbounded raw beta",
            "final_projection_initialization": "zeros",
        },
        "readout": "position-shared LayerNorm(768) + Linear(768,1) + sigmoid",
        "context_layer_after_mrnabert": None,
        "prediction": "one continuous value per valid nucleotide",
    }
    data_contract = {
        "dataset_manifest_sha256": baseline.sha256_file(
            cfg.universal_dir / "manifest.json"
        ),
        "profile_sha256": baseline.sha256_file(profiles_path),
        "splits_sha256": baseline.sha256_file(cfg.universal_dir / "splits.npy"),
        "view_ids_sha256": baseline.sha256_file(
            cfg.universal_dir / "view_ids.npy"
        ),
    }
    train_views = stores["train"].view_ids_numpy()
    run_config = {
        "architecture": architecture,
        "data": {
            "split_sizes": {name: len(store) for name, store in stores.items()},
            "training_view_sizes": {
                "canonical": int(np.sum(train_views == VIEW_CANONICAL)),
                "shifted": int(np.sum(train_views == VIEW_SHIFTED)),
            },
            "evaluation_view": "canonical only",
            "split_policy": "fixed reference transcript-component split",
            "profile_target": "cross-cell-line median accessibility per nucleotide",
            "contract": data_contract,
        },
        "optimization": {
            "optimizer": "AdamW with separate mRNABERT-LoRA and FiLM-head groups",
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
            "lora": lora_count,
            "film_direct_head": head_count,
            "total_trainable": lora_count + head_count,
        },
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
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
        if checkpoint["data_contract"] != data_contract:
            raise RuntimeError("Resume checkpoint dataset contract differs")
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
                "trainable_parameters": lora_count + head_count,
                "lora_parameters": lora_count,
                "head_parameters": head_count,
                "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "total_optimizer_steps": total_optimizer_steps,
                "split_sizes": {name: len(store) for name, store in stores.items()},
                "training_view_sizes": run_config["data"]["training_view_sizes"],
            }
        ),
        flush=True,
    )

    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        blocks = baseline.block_ranges(
            len(stores["train"]), cfg.micro_batch_size, seed=cfg.seed + epoch
        )
        train_loss_sum = 0.0
        train_rows = 0
        for group_start in range(0, len(blocks), cfg.gradient_accumulation):
            group = blocks[group_start : group_start + cfg.gradient_accumulation]
            group_rows = sum(end - start for start, end in group)
            optimizer.zero_grad(set_to_none=True)
            for start, end in group:
                encoded, batch_lengths, target, _ = prepare_batch(
                    stores["train"], start, end, tokenizer, device
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    prediction = model(encoded, batch_lengths)
                    loss = masked_l1_loss(prediction, target, batch_lengths)
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
                "data_contract": data_contract,
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
        writer = csv.DictWriter(handle, fieldnames=list(history[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(history)
    np.savez_compressed(
        cfg.output_dir / "test_predictions.npz",
        sequence_ids=stores["test"].sequence_ids_numpy(),
        view_ids=stores["test"].view_ids_numpy(),
        lengths=stores["test"].lengths_numpy(),
        observed=test_observed.astype(np.float16),
        predicted=test_predicted.astype(np.float16),
    )
    model.encoder.save_pretrained(
        cfg.output_dir / "best_mrnabert_lora", safe_serialization=True
    )
    baseline.atomic_torch_save(
        {
            "head_state_dict": best_state["head_state"],
            "architecture": architecture,
            "data_contract": data_contract,
            "seed": cfg.seed,
            "best_epoch": best_epoch,
            "best_validation_selection_score": best_score,
            "label": "per-position universal-median GSE145805 accessibility",
            "backbone": "public mRNABERT plus best_mrnabert_lora adapter",
        },
        cfg.output_dir / "best_film_head.pt",
    )
    result = {
        "status": "pass",
        "architecture": architecture,
        "data_contract": data_contract,
        "trainable_parameters": run_config["parameters"],
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_validation_selection_score": best_score,
        "split_sizes": {name: len(store) for name, store in stores.items()},
        "training_view_sizes": run_config["data"]["training_view_sizes"],
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
                "test_micro_pearson": test_metrics["micro_nucleotide"]["pearson"],
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
