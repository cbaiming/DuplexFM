#!/usr/bin/env python3
"""Train the two-view CLS/SEP-FiLM residual Transformer accessibility model.

Architecture
------------
    mRNABERT + LoRA(Wqkv)
      -> CLS/SEP-conditioned FiLM of real-nucleotide states
      -> LayerNorm(768)
      -> gated direct logit plus a Linear(768, 256), two-layer pre-norm
         residual Transformer correction
      -> sigmoid accessibility prediction for each real nucleotide

CLS and SEP remain in the mRNABERT input and condition FiLM, but they never
receive labels or enter the residual Transformer's valid-token attention set.
Training uses canonical and shifted views; validation and test use only the
canonical view by dataset contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
from tools import train_gse145805_full_residual_transformer_per_base as baseline
from tools import train_gse145805_mrnabert_lora_film_two_view_per_base as film


SPLITS = film.SPLITS
MAX_LENGTH = film.MAX_LENGTH
INPUT_DIM = film.INPUT_DIM
VIEW_CANONICAL = film.VIEW_CANONICAL
VIEW_SHIFTED = film.VIEW_SHIFTED
ARCHITECTURE_NAME = (
    "mRNABERT-LoRA CLS-SEP FiLM Residual Transformer Two-View"
)


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
    parser.add_argument("--lora-learning-rate", type=float, default=5e-5)
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.03)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--film-hidden-dim", type=int, default=128)
    parser.add_argument("--film-dropout", type=float, default=0.05)
    parser.add_argument("--film-gamma-bound", type=float, default=0.50)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--residual-gate-init", type=float, default=0.20)
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


class SpecialTokenFiLMResidualTransformerHead(nn.Module):
    """FiLM real-nucleotide states, then apply the audited residual head."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        film_hidden_dim: int = 128,
        film_dropout: float = 0.05,
        gamma_bound: float = 0.50,
        hidden_dim: int = 256,
        heads: int = 8,
        layers: int = 2,
        ffn_dim: int = 1024,
        dropout: float = 0.05,
        residual_gate_init: float = 0.20,
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
        self.residual_head = baseline.ResidualFullTransformerPerBase(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            layers=layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            residual_gate_init=residual_gate_init,
        )
        # FiLM starts as an identity transform. The residual head therefore
        # starts from exactly its audited unconditioned behavior.
        nn.init.zeros_(self.context_to_film.weight)
        nn.init.zeros_(self.context_to_film.bias)

    def modulate(
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
        return (1.0 + gamma[:, None, :]) * nucleotide_tokens + beta[:, None, :]

    def forward(
        self,
        nucleotide_tokens: torch.Tensor,
        cls_state: torch.Tensor,
        sep_state: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        if nucleotide_tokens.ndim != 3:
            raise RuntimeError("Expected a batch of nucleotide token states")
        if nucleotide_tokens.shape[1:] != (MAX_LENGTH, self.input_dim):
            raise RuntimeError(
                "Unexpected nucleotide state shape: "
                f"{tuple(nucleotide_tokens.shape)}"
            )
        if cls_state.shape != (nucleotide_tokens.shape[0], self.input_dim):
            raise RuntimeError(f"Unexpected CLS state shape: {tuple(cls_state.shape)}")
        if sep_state.shape != cls_state.shape:
            raise RuntimeError(f"Unexpected SEP state shape: {tuple(sep_state.shape)}")
        if lengths.shape != (nucleotide_tokens.shape[0],):
            raise RuntimeError(f"Unexpected length shape: {tuple(lengths.shape)}")
        mask = baseline.valid_mask(lengths)
        modulated = self.modulate(nucleotide_tokens, cls_state, sep_state)
        # For short fragments, the fixed-width slice contains SEP and padding
        # after the last real nucleotide. Zero it before projection, and also
        # provide lengths to the Transformer's key-padding mask.
        modulated = modulated.masked_fill(~mask.unsqueeze(-1), 0.0)
        prediction = self.residual_head(modulated, lengths)
        return prediction.masked_fill(~mask, 0.0)

    def residual_gate(self) -> torch.Tensor:
        return self.residual_head.residual_gate()


class LoRASpecialTokenFiLMResidualTransformerPerBase(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        film_hidden_dim: int = 128,
        film_dropout: float = 0.05,
        gamma_bound: float = 0.50,
        hidden_dim: int = 256,
        heads: int = 8,
        layers: int = 2,
        ffn_dim: int = 1024,
        dropout: float = 0.05,
        residual_gate_init: float = 0.20,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = SpecialTokenFiLMResidualTransformerHead(
            input_dim=INPUT_DIM,
            film_hidden_dim=film_hidden_dim,
            film_dropout=film_dropout,
            gamma_bound=gamma_bound,
            hidden_dim=hidden_dim,
            heads=heads,
            layers=layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            residual_gate_init=residual_gate_init,
        )

    special_token_states = staticmethod(
        film.LoRASpecialTokenFiLMPerBase.special_token_states
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
        if hidden.shape[1] < MAX_LENGTH + 2:
            raise RuntimeError(f"mRNABERT output is too short: {tuple(hidden.shape)}")
        cls_state, sep_state = self.special_token_states(
            hidden, encoded["attention_mask"], lengths
        )
        nucleotide_tokens = hidden[:, 1 : MAX_LENGTH + 1]
        return self.head(nucleotide_tokens, cls_state, sep_state, lengths)

    def residual_gate(self) -> torch.Tensor:
        return self.head.residual_gate()


def capture_trainable_state(
    model: LoRASpecialTokenFiLMResidualTransformerPerBase,
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
    model: LoRASpecialTokenFiLMResidualTransformerPerBase,
    state: dict[str, Any],
) -> None:
    incompatible = set_peft_model_state_dict(model.encoder, state["lora_state"])
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if unexpected:
        raise RuntimeError(f"Unexpected LoRA state keys: {unexpected[:8]}")
    model.head.load_state_dict(state["head_state"], strict=True)


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
        "hidden_dim",
        "attention_heads",
        "layers",
        "ffn_dim",
    ):
        if int(getattr(cfg, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    for name in ("lora_learning_rate", "head_learning_rate"):
        if float(getattr(cfg, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    for name in ("lora_dropout", "film_dropout", "dropout"):
        if not 0.0 <= float(getattr(cfg, name)) < 1.0:
            raise ValueError(f"{name.replace('_', '-')} must lie in [0,1)")
    if cfg.film_gamma_bound <= 0:
        raise ValueError("film-gamma-bound must be positive")
    if cfg.hidden_dim % cfg.attention_heads:
        raise ValueError("hidden-dim must be divisible by attention-heads")
    if not 0.0 < cfg.residual_gate_init < 1.0:
        raise ValueError("residual-gate-init must lie strictly between zero and one")
    if not 0.0 <= cfg.warmup_fraction < 1.0:
        raise ValueError("warmup-fraction must lie in [0,1)")
    if cfg.weight_decay < 0 or cfg.gradient_clip < 0 or cfg.min_improvement < 0:
        raise ValueError("weight decay, gradient clip, and min improvement must be nonnegative")
    for name in ("max_train_rows", "max_val_rows", "max_test_rows"):
        if int(getattr(cfg, name)) < 0:
            raise ValueError(f"{name.replace('_', '-')} must be nonnegative")
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

    film.seed_all(cfg.seed)
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
        name: film.TwoViewProfileSplit(
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
        raise RuntimeError(f"Unexpected mRNABERT hidden size: {backbone.config.hidden_size}")
    backbone.requires_grad_(False)
    target_summary = route2.discover_lora_targets(
        backbone=backbone,
        requested="auto",
        label="mRNABERT two-view FiLM residual accessibility",
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
    model = LoRASpecialTokenFiLMResidualTransformerPerBase(
        encoder,
        film_hidden_dim=cfg.film_hidden_dim,
        film_dropout=cfg.film_dropout,
        gamma_bound=cfg.film_gamma_bound,
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
                "group_name": "film_residual_transformer_head",
            },
        ]
    )
    micro_batches = math.ceil(len(stores["train"]) / cfg.micro_batch_size)
    optimizer_steps_per_epoch = math.ceil(
        micro_batches / cfg.gradient_accumulation
    )
    total_optimizer_steps = optimizer_steps_per_epoch * cfg.epochs
    scheduler, warmup_steps = baseline.make_scheduler(
        optimizer, total_optimizer_steps, cfg.warmup_fraction
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
            "transformer_attention": "real nucleotides only",
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
            "initial_transform": "identity",
        },
        "residual_transformer": {
            "input_norm": "LayerNorm(768)",
            "direct_logit": "position-shared Linear(768,1)",
            "input_projection": f"Linear(768,{cfg.hidden_dim})",
            "position_embedding": f"learned absolute positions 0..{MAX_LENGTH - 1}",
            "hidden_dim": cfg.hidden_dim,
            "attention_heads": cfg.attention_heads,
            "layers": cfg.layers,
            "ffn_dim": cfg.ffn_dim,
            "dropout": cfg.dropout,
            "activation": "GELU",
            "norm_first": True,
            "final_encoder_norm": True,
            "residual_readout": f"LayerNorm({cfg.hidden_dim}) -> Linear({cfg.hidden_dim},1)",
            "residual_gate_init": cfg.residual_gate_init,
            "logit_fusion": "direct_logit + sigmoid(gate_logit) * residual_logit",
            "padding_mask": "valid nucleotide positions from true sequence lengths",
        },
        "prediction": "sigmoid one continuous value per valid nucleotide",
    }
    data_contract = {
        "dataset_manifest_sha256": baseline.sha256_file(
            cfg.universal_dir / "manifest.json"
        ),
        "profile_sha256": baseline.sha256_file(profiles_path),
        "splits_sha256": baseline.sha256_file(cfg.universal_dir / "splits.npy"),
        "view_ids_sha256": baseline.sha256_file(cfg.universal_dir / "view_ids.npy"),
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
            "optimizer": "AdamW with separate mRNABERT-LoRA and FiLM-residual-head groups",
            "micro_batch_size": cfg.micro_batch_size,
            "eval_batch_size": cfg.eval_batch_size,
            "gradient_accumulation": cfg.gradient_accumulation,
            "nominal_effective_batch_size": cfg.micro_batch_size * cfg.gradient_accumulation,
            "epochs": cfg.epochs,
            "patience": cfg.patience,
            "lora_learning_rate": cfg.lora_learning_rate,
            "head_learning_rate": cfg.head_learning_rate,
            "weight_decay": cfg.weight_decay,
            "warmup_fraction": cfg.warmup_fraction,
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
            "film_residual_transformer_head": head_count,
            "total_trainable": lora_count + head_count,
        },
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
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
                "micro_batches_per_epoch": micro_batches,
                "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "total_optimizer_steps": total_optimizer_steps,
                "warmup_steps": warmup_steps,
                "split_sizes": {name: len(store) for name, store in stores.items()},
                "training_view_sizes": run_config["data"]["training_view_sizes"],
                "initial_residual_gate": float(model.residual_gate().detach().cpu()),
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
                encoded, batch_lengths, target, _ = film.prepare_batch(
                    stores["train"], start, end, tokenizer, device
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    prediction = model(encoded, batch_lengths)
                    loss = film.masked_l1_loss(prediction, target, batch_lengths)
                rows = end - start
                (loss * (rows / group_rows)).backward()
                train_loss_sum += float(loss.detach()) * rows
                train_rows += rows
            if cfg.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, cfg.gradient_clip)
            optimizer.step()
            scheduler.step()

        validation, val_loss = film.evaluate_basic(
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
    validation_metrics, _, _ = film.evaluate_detailed(
        model, stores["val"], tokenizer, device, cfg
    )
    test_metrics, test_observed, test_predicted = film.evaluate_detailed(
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
    final_gate = float(model.residual_gate().detach().cpu())
    baseline.atomic_torch_save(
        {
            "head_state_dict": best_state["head_state"],
            "architecture": architecture,
            "data_contract": data_contract,
            "seed": cfg.seed,
            "best_epoch": best_epoch,
            "best_validation_selection_score": best_score,
            "residual_gate": final_gate,
            "label": "per-position universal-median GSE145805 accessibility",
            "backbone": "public mRNABERT plus best_mrnabert_lora adapter",
        },
        cfg.output_dir / "best_film_residual_head.pt",
    )
    result = {
        "status": "pass",
        "architecture": architecture,
        "data_contract": data_contract,
        "trainable_parameters": run_config["parameters"],
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_validation_selection_score": best_score,
        "best_residual_gate": final_gate,
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
                "residual_gate": final_gate,
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
