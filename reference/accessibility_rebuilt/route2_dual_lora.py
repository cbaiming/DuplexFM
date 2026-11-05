from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import mirna_handcrafted_features as mhf
import repeated_group_split_eval as rge
import rna_backbone as rb
import route1_finetune as r1
import supervised_poc as sp
import utrbert_pll_feature as upf


@dataclass
class Route2Config:
    train_dir: str = "train_data"
    test_dir: str = "test_data"
    output_dir: str = "outputs"
    run_name: str = ""
    train_files: str = ""
    test_files: str = ""
    config_preset: str = "custom"
    feature_value: str = "three_prime_utr"
    group_key: str = "noncodingRNA_fam"
    val_ratio: float = 0.15
    val_split_candidates: int = 20
    reuse_split_manifest_path: str = ""
    batch_size: int = 128
    eval_batch_size: int = 256
    max_epochs: int = 8
    early_stopping_patience: int = 2
    seed: int = 42
    device: str = "auto"
    gene_model_name: str = "multimolecule/utrbert-3mer"
    mirna_model_name: str = "multimolecule/rnafm"
    max_gene_len: int = 64
    max_mirna_len: int = 32
    weight_decay: float = 1e-2
    dropout: float = 0.2
    classifier_hidden_dim: int = 256
    interaction_hidden_dim: int = 128
    attention_heads: int = 4
    interaction_top_k: int = 16
    handcrafted_feature_set: str = "none"
    handcrafted_include_groups: str = ""
    handcrafted_exclude_groups: str = ""
    gene_sequence_score_feature: str = "none"
    gene_sequence_score_cache_dir: str = upf.DEFAULT_CACHE_ROOT
    chunksize: int = 200000
    limit_train_rows: int = 0
    limit_test_rows: int = 0
    num_workers: int = 0
    prefetch_factor: int = 4
    cache_tokenized_sequences: bool = True
    token_cache_batch_size: int = 4096
    amp_dtype: str = "bf16"
    progress_bar: bool = True
    cuda_prefetch: bool = True
    fused_optimizer: bool = True
    exclude_test_overlap: bool = True
    model_selection_metric: str = "auprc"
    threshold_metric: str = "balanced_accuracy"
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_step: float = 0.01
    lora_backend: str = "peft"
    gene_lora_r: int = 8
    mirna_lora_r: int = 8
    gene_lora_alpha: int = 16
    mirna_lora_alpha: int = 16
    lora_dropout: float = 0.05
    gene_lora_target_modules: str = "auto"
    mirna_lora_target_modules: str = "auto"
    learning_rate_head: float = 1e-4
    learning_rate_gene_lora: float = 2e-5
    learning_rate_mirna_lora: float = 1e-5
    scheduler: str = "cosine"
    warmup_epochs: int = 1
    family_balanced_sampling: bool = False


def parse_args() -> Route2Config:
    parser = argparse.ArgumentParser(
        description="路线2：双 backbone LoRA + interaction head 快速微调脚本"
    )
    parser.add_argument("--train-dir", default="train_data")
    parser.add_argument("--test-dir", default="test_data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--train-files", default="")
    parser.add_argument("--test-files", default="")
    parser.add_argument(
        "--config-preset",
        default="custom",
        choices=["custom", "route2-lora-base", "route2-lora-best"],
    )
    parser.add_argument("--feature-value", default="three_prime_utr")
    parser.add_argument("--group-key", default="noncodingRNA_fam")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--val-split-candidates", type=int, default=20)
    parser.add_argument("--reuse-split-manifest-path", default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gene-model-name", default="multimolecule/utrbert-3mer")
    parser.add_argument("--mirna-model-name", default="multimolecule/rnafm")
    parser.add_argument("--max-gene-len", type=int, default=64)
    parser.add_argument("--max-mirna-len", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--classifier-hidden-dim", type=int, default=256)
    parser.add_argument("--interaction-hidden-dim", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--interaction-top-k", type=int, default=16)
    parser.add_argument(
        "--handcrafted-feature-set",
        default="none",
        choices=["none", "classic_seed_context"],
    )
    parser.add_argument("--handcrafted-include-groups", default="")
    parser.add_argument("--handcrafted-exclude-groups", default="")
    parser.add_argument(
        "--gene-sequence-score-feature",
        default="none",
        choices=["none", "utrbert_pll"],
    )
    parser.add_argument(
        "--gene-sequence-score-cache-dir",
        default=upf.DEFAULT_CACHE_ROOT,
    )
    parser.add_argument("--chunksize", type=int, default=200000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--disable-token-cache", action="store_true")
    parser.add_argument("--token-cache-batch-size", type=int, default=4096)
    parser.add_argument(
        "--amp-dtype",
        default="bf16",
        choices=["auto", "float16", "bfloat16", "fp16", "bf16"],
    )
    parser.add_argument("--disable-progress-bar", action="store_true")
    parser.add_argument("--disable-cuda-prefetch", action="store_true")
    parser.add_argument("--disable-fused-optimizer", action="store_true")
    parser.add_argument("--exclude-test-overlap", dest="exclude_test_overlap", action="store_true")
    parser.add_argument("--keep-test-overlap", dest="exclude_test_overlap", action="store_false")
    parser.add_argument(
        "--model-selection-metric",
        default="auprc",
        choices=["auprc", "auroc"],
    )
    parser.add_argument("--lora-backend", default="peft", choices=["peft"])
    parser.add_argument("--gene-lora-r", type=int, default=8)
    parser.add_argument("--mirna-lora-r", type=int, default=8)
    parser.add_argument("--gene-lora-alpha", type=int, default=16)
    parser.add_argument("--mirna-lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gene-lora-target-modules", default="auto")
    parser.add_argument("--mirna-lora-target-modules", default="auto")
    parser.add_argument("--learning-rate-head", type=float, default=1e-4)
    parser.add_argument("--learning-rate-gene-lora", type=float, default=2e-5)
    parser.add_argument("--learning-rate-mirna-lora", type=float, default=1e-5)
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "none"])
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--family-balanced-sampling", action="store_true")
    parser.set_defaults(exclude_test_overlap=True)

    config_kwargs = vars(parser.parse_args())
    config_kwargs["cache_tokenized_sequences"] = not config_kwargs.pop("disable_token_cache")
    config_kwargs["progress_bar"] = not config_kwargs.pop("disable_progress_bar")
    config_kwargs["cuda_prefetch"] = not config_kwargs.pop("disable_cuda_prefetch")
    config_kwargs["fused_optimizer"] = not config_kwargs.pop("disable_fused_optimizer")
    return Route2Config(**config_kwargs)


def apply_route2_preset(config: Route2Config) -> Route2Config:
    preset = config.config_preset.strip().lower()
    if preset in {"", "custom"}:
        return config
    if preset == "route2-lora-base":
        config.max_gene_len = 64
        config.max_mirna_len = 32
        config.interaction_hidden_dim = 128
        config.classifier_hidden_dim = 256
        config.attention_heads = 4
        config.interaction_top_k = 16
        config.dropout = 0.2
        config.model_selection_metric = "auprc"
        config.gene_lora_r = 8
        config.mirna_lora_r = 8
        config.gene_lora_alpha = 16
        config.mirna_lora_alpha = 16
        config.lora_dropout = 0.05
        config.batch_size = 128
        config.eval_batch_size = 256
        config.learning_rate_head = 1e-4
        config.learning_rate_gene_lora = 2e-5
        config.learning_rate_mirna_lora = 1e-5
        config.scheduler = "cosine"
        config.warmup_epochs = 1
        config.max_epochs = 8
        config.early_stopping_patience = 2
        config.family_balanced_sampling = False
        return config
    if preset == "route2-lora-best":
        config.max_gene_len = 128
        config.max_mirna_len = 32
        config.interaction_hidden_dim = 256
        config.classifier_hidden_dim = 512
        config.attention_heads = 8
        config.interaction_top_k = 32
        config.dropout = 0.1
        config.model_selection_metric = "auprc"
        config.gene_lora_r = 16
        config.mirna_lora_r = 16
        config.gene_lora_alpha = 32
        config.mirna_lora_alpha = 32
        config.lora_dropout = 0.05
        config.batch_size = 96
        config.eval_batch_size = 256
        config.learning_rate_head = 1e-4
        config.learning_rate_gene_lora = 2e-5
        config.learning_rate_mirna_lora = 1e-5
        config.scheduler = "cosine"
        config.warmup_epochs = 1
        config.max_epochs = 10
        config.early_stopping_patience = 3
        config.family_balanced_sampling = True
        return config
    raise ValueError(f"未知 config_preset: {config.config_preset!r}")


def _normalize_group_csv(value: Optional[Any]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        candidates = value
    else:
        candidates = str(value).split(",")
    return [str(item).strip() for item in candidates if str(item).strip()]


def get_selected_handcrafted_feature_names(config: Route2Config) -> List[str]:
    return mhf.resolve_selected_feature_names(
        feature_set=config.handcrafted_feature_set,
        include_groups=config.handcrafted_include_groups,
        exclude_groups=config.handcrafted_exclude_groups,
    )


def get_selected_handcrafted_feature_columns(config: Route2Config) -> List[str]:
    return mhf.get_selected_prefixed_feature_columns(
        feature_set=config.handcrafted_feature_set,
        include_groups=config.handcrafted_include_groups,
        exclude_groups=config.handcrafted_exclude_groups,
    )


def get_selected_model_feature_names(config: Route2Config) -> List[str]:
    return get_selected_handcrafted_feature_names(config) + upf.get_feature_names(
        config.gene_sequence_score_feature
    )


def get_selected_model_feature_columns(config: Route2Config) -> List[str]:
    return get_selected_handcrafted_feature_columns(config) + upf.get_feature_columns(
        config.gene_sequence_score_feature
    )


def get_selected_handcrafted_groups(config: Route2Config) -> Dict[str, List[str]]:
    selected_feature_names = set(get_selected_handcrafted_feature_names(config))
    available_feature_groups = mhf.get_feature_groups(config.handcrafted_feature_set)
    return {
        group_name: list(feature_names)
        for group_name, feature_names in available_feature_groups.items()
        if any(feature_name in selected_feature_names for feature_name in feature_names)
    }


def build_sequence_score_summary(
    config: Route2Config,
    split_df: pd.DataFrame,
    test_frames: Dict[str, pd.DataFrame],
    train_files: List[Path],
    test_files: List[Path],
    gene_model_source: str,
    device,
    show_progress: bool,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Any], Dict[str, Any]]:
    normalized_feature = upf.normalize_sequence_score_feature(config.gene_sequence_score_feature)
    summary: Dict[str, Any] = {
        "feature_name": normalized_feature,
        "enabled": normalized_feature != upf.SCORE_FEATURE_NONE,
        "feature_columns": upf.get_feature_columns(normalized_feature),
        "cache_dir": config.gene_sequence_score_cache_dir,
        "split_df": {"row_count": int(len(split_df)), "columns": []},
        "test_frames": {},
    }
    cache_manifest_ref: Dict[str, Any] = {
        "feature_name": normalized_feature,
        "enabled": normalized_feature != upf.SCORE_FEATURE_NONE,
        "cache_dir": config.gene_sequence_score_cache_dir,
    }
    if normalized_feature == upf.SCORE_FEATURE_NONE:
        return split_df, test_frames, summary, cache_manifest_ref

    data_signature = upf.build_data_signature(
        train_files=[path.name for path in train_files],
        test_files=[path.name for path in test_files],
        feature_value=config.feature_value,
        model_source=gene_model_source,
    )
    unique_genes = upf.collect_unique_genes_from_route2_frames(split_df, test_frames)
    cache_result = upf.build_or_load_pll_cache(
        unique_genes=unique_genes,
        cache_dir=Path(config.gene_sequence_score_cache_dir),
        model_source=gene_model_source,
        device=device,
        data_signature=data_signature,
        batch_size=max(1, min(config.eval_batch_size, 256)),
        show_progress=show_progress,
    )
    score_table = upf.load_score_table(cache_result.cache_dir)
    split_df = upf.augment_dataframe_with_pll_feature(
        split_df,
        score_table=score_table,
        feature_name=normalized_feature,
    )
    summary["split_df"] = upf.summarize_feature_columns(split_df, normalized_feature)

    augmented_test_frames: Dict[str, pd.DataFrame] = {}
    for test_name, test_df in test_frames.items():
        augmented_test_df = upf.augment_dataframe_with_pll_feature(
            test_df,
            score_table=score_table,
            feature_name=normalized_feature,
        )
        augmented_test_frames[test_name] = augmented_test_df
        summary["test_frames"][test_name] = upf.summarize_feature_columns(
            augmented_test_df,
            normalized_feature,
        )
    summary["cache_reused"] = bool(cache_result.reused)
    summary["cache_manifest_path"] = str(cache_result.manifest_path)
    summary["cache_summary_path"] = str(cache_result.summary_path)
    summary["cache_summary"] = cache_result.summary
    cache_manifest_ref.update(
        {
            "cache_manifest_path": str(cache_result.manifest_path),
            "cache_summary_path": str(cache_result.summary_path),
            "cache_reused": bool(cache_result.reused),
            "manifest": cache_result.manifest,
        }
    )
    return split_df, augmented_test_frames, summary, cache_manifest_ref


def check_and_import_peft():
    try:
        import peft
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model

        return {
            "module": peft,
            "LoraConfig": LoraConfig,
            "PeftModel": PeftModel,
            "TaskType": TaskType,
            "get_peft_model": get_peft_model,
        }
    except Exception as exc:
        raise RuntimeError(
            "未能导入 peft。请先安装 route2 LoRA 依赖：\n"
            "pip install peft\n"
            "如果当前环境尚未安装 torch / transformers / multimolecule，也需要一并安装。"
        ) from exc


def parse_target_module_names(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def discover_lora_targets(backbone, requested: str, label: str) -> Dict[str, Any]:
    if sp.torch is None or sp.nn is None:
        raise RuntimeError("torch 不可用，无法发现 LoRA target modules。")

    linear_names: List[str] = []
    for module_name, module in backbone.named_modules():
        if isinstance(module, sp.nn.Linear):
            linear_names.append(module_name)

    if not linear_names:
        raise RuntimeError(f"{label} backbone 中未找到任何 Linear 模块。")

    if requested.strip().lower() != "auto":
        target_modules = parse_target_module_names(requested)
        matched_full_names = []
        for name in linear_names:
            if any(name == target or name.endswith(f".{target}") for target in target_modules):
                matched_full_names.append(name)
        if not matched_full_names:
            raise RuntimeError(
                f"{label} 手动指定的 LoRA target modules {target_modules} 未命中任何 Linear 模块。\n"
                f"可用候选示例: {linear_names[:50]}"
            )
        return {
            "mode": "manual",
            "target_modules": target_modules,
            "matched_full_names": matched_full_names,
            "candidate_linear_modules_preview": linear_names[:200],
        }

    preferred_leaf_names = {
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "o_proj",
        "query",
        "key",
        "value",
        # mRNABERT/MosaicBERT uses one fused QKV projection.
        "Wqkv",
    }
    matched_full_names = []
    target_leaf_names = []
    for name in linear_names:
        leaf_name = name.split(".")[-1]
        if leaf_name in preferred_leaf_names:
            target_leaf_names.append(leaf_name)
            matched_full_names.append(name)

    used_attention_dense_fallback = False
    if not matched_full_names:
        for name in linear_names:
            leaf_name = name.split(".")[-1]
            if "attention" in name.lower() and leaf_name == "dense":
                target_leaf_names.append(leaf_name)
                matched_full_names.append(name)
        if matched_full_names:
            used_attention_dense_fallback = True

    if used_attention_dense_fallback:
        target_modules = _unique_preserve_order(matched_full_names)
    else:
        target_modules = _unique_preserve_order(target_leaf_names)
    if not target_modules:
        raise RuntimeError(
            f"{label} 自动发现 LoRA target modules 失败。\n"
            f"Linear 候选模块示例: {linear_names[:200]}"
        )

    return {
        "mode": "auto",
        "target_modules": target_modules,
        "matched_full_names": matched_full_names,
        "candidate_linear_modules_preview": linear_names[:200],
    }


def extract_trainable_submodule_state_dict(module) -> Dict[str, "sp.torch.Tensor"]:
    trainable_names = {
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
        if name in trainable_names
    }


def load_trainable_submodule_state_dict(module, state_dict: Dict[str, "sp.torch.Tensor"]) -> None:
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"载入子模块 trainable state_dict 时遇到未知参数: {unexpected}")
    if missing:
        trainable_names = {
            name
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
        remaining_missing = sorted(set(missing) & trainable_names)
        if remaining_missing:
            raise RuntimeError(
                f"载入子模块 trainable state_dict 后仍缺少可训练参数: {remaining_missing}"
            )


def extract_head_state_dict(model) -> Dict[str, "sp.torch.Tensor"]:
    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("gene_encoder.")
        and not name.startswith("mirna_encoder.")
    }
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }


def load_head_state_dict(model, state_dict: Dict[str, "sp.torch.Tensor"]) -> None:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"载入 head state_dict 时遇到未知参数: {unexpected}")
    if missing:
        trainable_names = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and not name.startswith("gene_encoder.")
            and not name.startswith("mirna_encoder.")
        }
        remaining_missing = sorted(set(missing) & trainable_names)
        if remaining_missing:
            raise RuntimeError(f"载入 head state_dict 后仍缺少可训练参数: {remaining_missing}")


def summarize_trainable_parameters(model) -> Dict[str, Any]:
    total_params = 0
    trainable_params = 0
    gene_lora_params = 0
    mirna_lora_params = 0
    head_params = 0
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total_params += count
        if not parameter.requires_grad:
            continue
        trainable_params += count
        if name.startswith("gene_encoder."):
            gene_lora_params += count
        elif name.startswith("mirna_encoder."):
            mirna_lora_params += count
        else:
            head_params += count
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_ratio": float(trainable_params / max(total_params, 1)),
        "gene_lora_trainable_parameters": gene_lora_params,
        "mirna_lora_trainable_parameters": mirna_lora_params,
        "head_trainable_parameters": head_params,
    }


if sp.torch is not None:

    class DualLoRAInteractionModel(sp.nn.Module):
        def __init__(
            self,
            gene_encoder,
            mirna_encoder,
            interaction_hidden_dim: int,
            classifier_hidden_dim: int,
            attention_heads: int,
            interaction_top_k: int,
            dropout: float,
            handcrafted_feature_dim: int = 0,
        ):
            super().__init__()
            self.gene_encoder = gene_encoder
            self.mirna_encoder = mirna_encoder
            self.interaction_hidden_dim = interaction_hidden_dim
            self.interaction_top_k = interaction_top_k
            self.handcrafted_feature_dim = max(0, int(handcrafted_feature_dim))

            gene_hidden = self.gene_encoder.config.hidden_size
            mirna_hidden = self.mirna_encoder.config.hidden_size

            heads = max(1, min(attention_heads, interaction_hidden_dim))
            while interaction_hidden_dim % heads != 0 and heads > 1:
                heads -= 1

            self.gene_projection = sp.nn.Linear(gene_hidden, interaction_hidden_dim)
            self.mirna_projection = sp.nn.Linear(mirna_hidden, interaction_hidden_dim)
            self.mirna_to_gene_attention = sp.nn.MultiheadAttention(
                embed_dim=interaction_hidden_dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True,
            )
            self.gene_to_mirna_attention = sp.nn.MultiheadAttention(
                embed_dim=interaction_hidden_dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True,
            )

            self.base_feature_dim = interaction_hidden_dim * 6 + 5
            interaction_feature_dim = self.base_feature_dim + self.handcrafted_feature_dim
            self.classifier = sp.nn.Sequential(
                sp.nn.Linear(interaction_feature_dim, classifier_hidden_dim),
                sp.nn.GELU(),
                sp.nn.Dropout(dropout),
                sp.nn.Linear(classifier_hidden_dim, 1),
            )

        def _encode_backbone(
            self,
            encoder,
            input_ids: "sp.torch.Tensor",
            attention_mask: "sp.torch.Tensor",
            token_type_ids: Optional["sp.torch.Tensor"] = None,
        ) -> "sp.torch.Tensor":
            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            outputs = encoder(**kwargs)
            return rb.extract_last_hidden_state(outputs)

        def _interaction_stats(
            self,
            scores: "sp.torch.Tensor",
            pair_mask: "sp.torch.Tensor",
            mirna_attention_mask: "sp.torch.Tensor",
            gene_attention_mask: "sp.torch.Tensor",
        ) -> "sp.torch.Tensor":
            valid_scores = scores.masked_fill(~pair_mask, 0.0)
            valid_count = pair_mask.sum(dim=(1, 2)).clamp(min=1).float()
            global_mean = valid_scores.sum(dim=(1, 2)) / valid_count

            masked_scores = scores.masked_fill(~pair_mask, float("-inf"))
            global_max = masked_scores.amax(dim=(1, 2))
            global_max = sp.torch.where(
                sp.torch.isfinite(global_max),
                global_max,
                sp.torch.zeros_like(global_max),
            )

            row_max = masked_scores.max(dim=2).values
            row_max = sp.torch.where(sp.torch.isfinite(row_max), row_max, sp.torch.zeros_like(row_max))
            row_max = row_max * mirna_attention_mask.float()
            row_count = mirna_attention_mask.sum(dim=1).clamp(min=1).float()
            row_max_mean = row_max.sum(dim=1) / row_count

            col_max = masked_scores.max(dim=1).values
            col_max = sp.torch.where(sp.torch.isfinite(col_max), col_max, sp.torch.zeros_like(col_max))
            col_max = col_max * gene_attention_mask.float()
            col_count = gene_attention_mask.sum(dim=1).clamp(min=1).float()
            col_max_mean = col_max.sum(dim=1) / col_count

            flat_scores = masked_scores.view(scores.size(0), -1)
            top_k = max(1, min(self.interaction_top_k, flat_scores.size(1)))
            topk_values = flat_scores.topk(top_k, dim=1).values
            topk_values = sp.torch.where(
                sp.torch.isfinite(topk_values),
                topk_values,
                sp.torch.zeros_like(topk_values),
            )
            valid_topk = pair_mask.view(scores.size(0), -1).sum(dim=1)
            valid_topk = valid_topk.clamp(min=1, max=top_k).float()
            topk_mean = topk_values.sum(dim=1) / valid_topk

            return sp.torch.stack(
                [global_mean, global_max, row_max_mean, col_max_mean, topk_mean],
                dim=-1,
            )

        def forward_features(
            self,
            gene_input_ids: "sp.torch.Tensor",
            gene_attention_mask: "sp.torch.Tensor",
            mirna_input_ids: "sp.torch.Tensor",
            mirna_attention_mask: "sp.torch.Tensor",
            gene_token_type_ids: Optional["sp.torch.Tensor"] = None,
            mirna_token_type_ids: Optional["sp.torch.Tensor"] = None,
            handcrafted_features: Optional["sp.torch.Tensor"] = None,
        ) -> "sp.torch.Tensor":
            gene_tokens_raw = self._encode_backbone(
                encoder=self.gene_encoder,
                input_ids=gene_input_ids,
                attention_mask=gene_attention_mask,
                token_type_ids=gene_token_type_ids,
            )
            mirna_tokens_raw = self._encode_backbone(
                encoder=self.mirna_encoder,
                input_ids=mirna_input_ids,
                attention_mask=mirna_attention_mask,
                token_type_ids=mirna_token_type_ids,
            )

            gene_tokens = self.gene_projection(gene_tokens_raw)
            mirna_tokens = self.mirna_projection(mirna_tokens_raw)

            mirna_cross, _ = self.mirna_to_gene_attention(
                query=mirna_tokens,
                key=gene_tokens,
                value=gene_tokens,
                key_padding_mask=~gene_attention_mask.bool(),
                need_weights=False,
            )
            gene_cross, _ = self.gene_to_mirna_attention(
                query=gene_tokens,
                key=mirna_tokens,
                value=mirna_tokens,
                key_padding_mask=~mirna_attention_mask.bool(),
                need_weights=False,
            )

            interaction_scores = sp.torch.matmul(
                mirna_tokens,
                gene_tokens.transpose(1, 2),
            ) / math.sqrt(self.interaction_hidden_dim)
            pair_mask = (
                mirna_attention_mask.bool().unsqueeze(-1)
                & gene_attention_mask.bool().unsqueeze(1)
            )

            interaction_features = sp.torch.cat(
                [
                    sp.masked_mean_pooling(gene_tokens, gene_attention_mask),
                    sp.masked_mean_pooling(mirna_tokens, mirna_attention_mask),
                    sp.masked_mean_pooling(gene_cross, gene_attention_mask),
                    sp.masked_max_pooling(gene_cross, gene_attention_mask),
                    sp.masked_mean_pooling(mirna_cross, mirna_attention_mask),
                    sp.masked_max_pooling(mirna_cross, mirna_attention_mask),
                    self._interaction_stats(
                        scores=interaction_scores,
                        pair_mask=pair_mask,
                        mirna_attention_mask=mirna_attention_mask,
                        gene_attention_mask=gene_attention_mask,
                    ),
                ],
                dim=-1,
            )
            if self.handcrafted_feature_dim:
                if handcrafted_features is None:
                    raise RuntimeError("模型要求 handcrafted_features，但当前 batch 未提供。")
                if handcrafted_features.size(-1) != self.handcrafted_feature_dim:
                    raise RuntimeError(
                        "handcrafted_features 维度不匹配："
                        f"expected {self.handcrafted_feature_dim}, got {handcrafted_features.size(-1)}"
                    )
                interaction_features = sp.torch.cat([interaction_features, handcrafted_features], dim=-1)
            return interaction_features

        def forward(
            self,
            gene_input_ids: "sp.torch.Tensor",
            gene_attention_mask: "sp.torch.Tensor",
            mirna_input_ids: "sp.torch.Tensor",
            mirna_attention_mask: "sp.torch.Tensor",
            gene_token_type_ids: Optional["sp.torch.Tensor"] = None,
            mirna_token_type_ids: Optional["sp.torch.Tensor"] = None,
            handcrafted_features: Optional["sp.torch.Tensor"] = None,
        ) -> "sp.torch.Tensor":
            interaction_features = self.forward_features(
                gene_input_ids=gene_input_ids,
                gene_attention_mask=gene_attention_mask,
                mirna_input_ids=mirna_input_ids,
                mirna_attention_mask=mirna_attention_mask,
                gene_token_type_ids=gene_token_type_ids,
                mirna_token_type_ids=mirna_token_type_ids,
                handcrafted_features=handcrafted_features,
            )
            return self.classifier(interaction_features).squeeze(-1)


    class FeatureGroupMoEInteractionModel(sp.nn.Module):
        expert_names = (
            "learned_representation",
            "interaction_statistics",
            "biological_priors",
        )

        def __init__(
            self,
            gene_encoder,
            mirna_encoder,
            interaction_hidden_dim: int,
            expert_hidden_dim: int,
            router_hidden_dim: int,
            attention_heads: int,
            interaction_top_k: int,
            dropout: float,
            biological_feature_dim: int = 0,
            gate_temperature: float = 1.0,
        ):
            super().__init__()
            self.gene_encoder = gene_encoder
            self.mirna_encoder = mirna_encoder
            self.interaction_hidden_dim = interaction_hidden_dim
            self.interaction_top_k = interaction_top_k
            self.biological_feature_dim = max(0, int(biological_feature_dim))
            self.gate_temperature = max(1e-6, float(gate_temperature))

            gene_hidden = self.gene_encoder.config.hidden_size
            mirna_hidden = self.mirna_encoder.config.hidden_size

            heads = max(1, min(attention_heads, interaction_hidden_dim))
            while interaction_hidden_dim % heads != 0 and heads > 1:
                heads -= 1

            self.gene_projection = sp.nn.Linear(gene_hidden, interaction_hidden_dim)
            self.mirna_projection = sp.nn.Linear(mirna_hidden, interaction_hidden_dim)
            self.mirna_to_gene_attention = sp.nn.MultiheadAttention(
                embed_dim=interaction_hidden_dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True,
            )
            self.gene_to_mirna_attention = sp.nn.MultiheadAttention(
                embed_dim=interaction_hidden_dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True,
            )

            learned_dim = interaction_hidden_dim * 6
            stats_dim = 5
            biological_dim = self.biological_feature_dim
            self.feature_group_dims = {
                "learned_representation": learned_dim,
                "interaction_statistics": stats_dim,
                "biological_priors": biological_dim,
            }
            self.experts = sp.nn.ModuleDict(
                {
                    "learned_representation": self._build_expert(
                        input_dim=learned_dim,
                        hidden_dim=expert_hidden_dim,
                        dropout=dropout,
                    ),
                    "interaction_statistics": self._build_expert(
                        input_dim=stats_dim,
                        hidden_dim=expert_hidden_dim,
                        dropout=dropout,
                    ),
                    "biological_priors": self._build_expert(
                        input_dim=biological_dim,
                        hidden_dim=expert_hidden_dim,
                        dropout=dropout,
                    ),
                }
            )
            router_input_dim = expert_hidden_dim * len(self.expert_names)
            self.router = sp.nn.Sequential(
                sp.nn.Linear(router_input_dim, router_hidden_dim),
                sp.nn.GELU(),
                sp.nn.Dropout(dropout),
                sp.nn.Linear(router_hidden_dim, len(self.expert_names)),
            )

        @staticmethod
        def _build_expert(input_dim: int, hidden_dim: int, dropout: float):
            if input_dim <= 0:
                return sp.nn.ModuleDict(
                    {
                        "projection": sp.nn.Identity(),
                        "body": sp.nn.Sequential(
                            sp.nn.Linear(1, hidden_dim),
                            sp.nn.GELU(),
                            sp.nn.Dropout(dropout),
                        ),
                        "logit": sp.nn.Linear(hidden_dim, 1),
                    }
                )
            return sp.nn.ModuleDict(
                {
                    "projection": sp.nn.Identity(),
                    "body": sp.nn.Sequential(
                        sp.nn.Linear(input_dim, hidden_dim),
                        sp.nn.GELU(),
                        sp.nn.Dropout(dropout),
                    ),
                    "logit": sp.nn.Linear(hidden_dim, 1),
                }
            )

        def _encode_backbone(
            self,
            encoder,
            input_ids: "sp.torch.Tensor",
            attention_mask: "sp.torch.Tensor",
            token_type_ids: Optional["sp.torch.Tensor"] = None,
        ) -> "sp.torch.Tensor":
            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids
            outputs = encoder(**kwargs)
            return rb.extract_last_hidden_state(outputs)

        def _interaction_stats(
            self,
            scores: "sp.torch.Tensor",
            pair_mask: "sp.torch.Tensor",
            mirna_attention_mask: "sp.torch.Tensor",
            gene_attention_mask: "sp.torch.Tensor",
        ) -> "sp.torch.Tensor":
            valid_scores = scores.masked_fill(~pair_mask, 0.0)
            valid_count = pair_mask.sum(dim=(1, 2)).clamp(min=1).float()
            global_mean = valid_scores.sum(dim=(1, 2)) / valid_count

            masked_scores = scores.masked_fill(~pair_mask, float("-inf"))
            global_max = masked_scores.amax(dim=(1, 2))
            global_max = sp.torch.where(
                sp.torch.isfinite(global_max),
                global_max,
                sp.torch.zeros_like(global_max),
            )

            row_max = masked_scores.max(dim=2).values
            row_max = sp.torch.where(sp.torch.isfinite(row_max), row_max, sp.torch.zeros_like(row_max))
            row_max = row_max * mirna_attention_mask.float()
            row_count = mirna_attention_mask.sum(dim=1).clamp(min=1).float()
            row_max_mean = row_max.sum(dim=1) / row_count

            col_max = masked_scores.max(dim=1).values
            col_max = sp.torch.where(sp.torch.isfinite(col_max), col_max, sp.torch.zeros_like(col_max))
            col_max = col_max * gene_attention_mask.float()
            col_count = gene_attention_mask.sum(dim=1).clamp(min=1).float()
            col_max_mean = col_max.sum(dim=1) / col_count

            flat_scores = masked_scores.view(scores.size(0), -1)
            top_k = max(1, min(self.interaction_top_k, flat_scores.size(1)))
            topk_values = flat_scores.topk(top_k, dim=1).values
            topk_values = sp.torch.where(
                sp.torch.isfinite(topk_values),
                topk_values,
                sp.torch.zeros_like(topk_values),
            )
            valid_topk = pair_mask.view(scores.size(0), -1).sum(dim=1)
            valid_topk = valid_topk.clamp(min=1, max=top_k).float()
            topk_mean = topk_values.sum(dim=1) / valid_topk

            return sp.torch.stack(
                [global_mean, global_max, row_max_mean, col_max_mean, topk_mean],
                dim=-1,
            )

        def forward_feature_groups(
            self,
            gene_input_ids: "sp.torch.Tensor",
            gene_attention_mask: "sp.torch.Tensor",
            mirna_input_ids: "sp.torch.Tensor",
            mirna_attention_mask: "sp.torch.Tensor",
            gene_token_type_ids: Optional["sp.torch.Tensor"] = None,
            mirna_token_type_ids: Optional["sp.torch.Tensor"] = None,
            handcrafted_features: Optional["sp.torch.Tensor"] = None,
        ) -> Dict[str, "sp.torch.Tensor"]:
            gene_tokens_raw = self._encode_backbone(
                encoder=self.gene_encoder,
                input_ids=gene_input_ids,
                attention_mask=gene_attention_mask,
                token_type_ids=gene_token_type_ids,
            )
            mirna_tokens_raw = self._encode_backbone(
                encoder=self.mirna_encoder,
                input_ids=mirna_input_ids,
                attention_mask=mirna_attention_mask,
                token_type_ids=mirna_token_type_ids,
            )

            gene_tokens = self.gene_projection(gene_tokens_raw)
            mirna_tokens = self.mirna_projection(mirna_tokens_raw)
            mirna_cross, _ = self.mirna_to_gene_attention(
                query=mirna_tokens,
                key=gene_tokens,
                value=gene_tokens,
                key_padding_mask=~gene_attention_mask.bool(),
                need_weights=False,
            )
            gene_cross, _ = self.gene_to_mirna_attention(
                query=gene_tokens,
                key=mirna_tokens,
                value=mirna_tokens,
                key_padding_mask=~mirna_attention_mask.bool(),
                need_weights=False,
            )

            interaction_scores = sp.torch.matmul(
                mirna_tokens,
                gene_tokens.transpose(1, 2),
            ) / math.sqrt(self.interaction_hidden_dim)
            pair_mask = (
                mirna_attention_mask.bool().unsqueeze(-1)
                & gene_attention_mask.bool().unsqueeze(1)
            )
            learned_features = sp.torch.cat(
                [
                    sp.masked_mean_pooling(gene_tokens, gene_attention_mask),
                    sp.masked_mean_pooling(mirna_tokens, mirna_attention_mask),
                    sp.masked_mean_pooling(gene_cross, gene_attention_mask),
                    sp.masked_max_pooling(gene_cross, gene_attention_mask),
                    sp.masked_mean_pooling(mirna_cross, mirna_attention_mask),
                    sp.masked_max_pooling(mirna_cross, mirna_attention_mask),
                ],
                dim=-1,
            )
            stat_features = self._interaction_stats(
                scores=interaction_scores,
                pair_mask=pair_mask,
                mirna_attention_mask=mirna_attention_mask,
                gene_attention_mask=gene_attention_mask,
            )
            if self.biological_feature_dim:
                if handcrafted_features is None:
                    raise RuntimeError("MoE biological_priors expert requires handcrafted_features.")
                if handcrafted_features.size(-1) != self.biological_feature_dim:
                    raise RuntimeError(
                        "MoE biological_priors feature dim mismatch: "
                        f"expected {self.biological_feature_dim}, got {handcrafted_features.size(-1)}"
                    )
                biological_features = handcrafted_features
            else:
                biological_features = learned_features.new_zeros((learned_features.size(0), 0))
            return {
                "learned_representation": learned_features,
                "interaction_statistics": stat_features,
                "biological_priors": biological_features,
            }

        def forward_expert_outputs_from_groups(
            self,
            feature_groups: Dict[str, "sp.torch.Tensor"],
        ) -> Dict[str, "sp.torch.Tensor"]:
            hidden_states = []
            logits = []
            outputs: Dict[str, "sp.torch.Tensor"] = {}
            batch_size = next(iter(feature_groups.values())).size(0)
            for expert_name in self.expert_names:
                expert_input = feature_groups[expert_name]
                if expert_input.size(-1) == 0:
                    expert_input = expert_input.new_zeros((batch_size, 1))
                expert_hidden = self.experts[expert_name]["body"](expert_input)
                expert_logit = self.experts[expert_name]["logit"](expert_hidden).squeeze(-1)
                outputs[f"logit_{expert_name}"] = expert_logit
                hidden_states.append(expert_hidden)
                logits.append(expert_logit)
            router_input = sp.torch.cat(hidden_states, dim=-1)
            gate_logits = self.router(router_input) / self.gate_temperature
            gate_weights = sp.torch.softmax(gate_logits, dim=-1)
            expert_logits = sp.torch.stack(logits, dim=1)
            outputs["gate_weights"] = gate_weights
            outputs["expert_logits"] = expert_logits
            outputs["moe_fused"] = (gate_weights * expert_logits).sum(dim=1)
            return outputs

        def forward_outputs(
            self,
            gene_input_ids: "sp.torch.Tensor",
            gene_attention_mask: "sp.torch.Tensor",
            mirna_input_ids: "sp.torch.Tensor",
            mirna_attention_mask: "sp.torch.Tensor",
            gene_token_type_ids: Optional["sp.torch.Tensor"] = None,
            mirna_token_type_ids: Optional["sp.torch.Tensor"] = None,
            handcrafted_features: Optional["sp.torch.Tensor"] = None,
        ) -> Dict[str, "sp.torch.Tensor"]:
            feature_groups = self.forward_feature_groups(
                gene_input_ids=gene_input_ids,
                gene_attention_mask=gene_attention_mask,
                mirna_input_ids=mirna_input_ids,
                mirna_attention_mask=mirna_attention_mask,
                gene_token_type_ids=gene_token_type_ids,
                mirna_token_type_ids=mirna_token_type_ids,
                handcrafted_features=handcrafted_features,
            )
            return self.forward_expert_outputs_from_groups(feature_groups)

        def forward(
            self,
            gene_input_ids: "sp.torch.Tensor",
            gene_attention_mask: "sp.torch.Tensor",
            mirna_input_ids: "sp.torch.Tensor",
            mirna_attention_mask: "sp.torch.Tensor",
            gene_token_type_ids: Optional["sp.torch.Tensor"] = None,
            mirna_token_type_ids: Optional["sp.torch.Tensor"] = None,
            handcrafted_features: Optional["sp.torch.Tensor"] = None,
        ) -> "sp.torch.Tensor":
            return self.forward_outputs(
                gene_input_ids=gene_input_ids,
                gene_attention_mask=gene_attention_mask,
                mirna_input_ids=mirna_input_ids,
                mirna_attention_mask=mirna_attention_mask,
                gene_token_type_ids=gene_token_type_ids,
                mirna_token_type_ids=mirna_token_type_ids,
                handcrafted_features=handcrafted_features,
            )["moe_fused"]

else:

    class DualLoRAInteractionModel:  # pragma: no cover - runtime requires torch
        pass

    class FeatureGroupMoEInteractionModel:  # pragma: no cover - runtime requires torch
        pass


def create_optimizer(model, config: Route2Config, device):
    if sp.torch is None:
        raise RuntimeError("torch 不可用，无法创建优化器。")

    head_parameters = []
    gene_lora_parameters = []
    mirna_lora_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("gene_encoder."):
            gene_lora_parameters.append(parameter)
        elif name.startswith("mirna_encoder."):
            mirna_lora_parameters.append(parameter)
        else:
            head_parameters.append(parameter)

    param_groups = [
        {"params": head_parameters, "lr": config.learning_rate_head, "weight_decay": config.weight_decay, "name": "head"},
        {"params": gene_lora_parameters, "lr": config.learning_rate_gene_lora, "weight_decay": config.weight_decay, "name": "gene_lora"},
        {"params": mirna_lora_parameters, "lr": config.learning_rate_mirna_lora, "weight_decay": config.weight_decay, "name": "mirna_lora"},
    ]
    param_groups = [group for group in param_groups if group["params"]]
    optimizer_kwargs = {}
    if device.type == "cuda":
        if config.fused_optimizer:
            try:
                return sp.torch.optim.AdamW(param_groups, fused=True, **optimizer_kwargs)
            except (TypeError, RuntimeError):
                pass
        try:
            return sp.torch.optim.AdamW(param_groups, foreach=True, **optimizer_kwargs)
        except TypeError:
            pass
    return sp.torch.optim.AdamW(param_groups, **optimizer_kwargs)


def create_scheduler(optimizer, config: Route2Config):
    if sp.torch is None or config.scheduler == "none":
        return None

    total_epochs = max(1, config.max_epochs)
    warmup_epochs = max(0, min(config.warmup_epochs, total_epochs))

    def lr_lambda(epoch_index: int) -> float:
        current_epoch = epoch_index + 1
        if warmup_epochs > 0 and current_epoch <= warmup_epochs:
            return float(current_epoch / warmup_epochs)
        if current_epoch >= total_epochs:
            return 0.0
        if total_epochs == warmup_epochs:
            return 1.0
        progress = (current_epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        progress = min(max(progress, 0.0), 1.0)
        return float(0.5 * (1.0 + math.cos(math.pi * progress)))

    return sp.torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def current_learning_rates(optimizer) -> Dict[str, float]:
    values = {}
    for index, group in enumerate(optimizer.param_groups):
        group_name = group.get("name", f"group_{index}")
        values[group_name] = float(group["lr"])
    return values


def run_epoch_route2(
    model,
    loader,
    criterion,
    device,
    train: bool,
    optimizer=None,
    scaler=None,
    amp_enabled: bool = False,
    amp_dtype=None,
    capture_predictions: bool = False,
    group_key: str = "noncodingRNA_fam",
    progress_desc: Optional[str] = None,
    show_progress: bool = True,
) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
    if sp.torch is None:
        raise RuntimeError("torch 不可用，无法运行训练/评估。")
    if train and optimizer is None:
        raise ValueError("训练模式下 optimizer 不能为空。")

    if train:
        model.train()
        optimizer.zero_grad(set_to_none=True)
    else:
        model.eval()

    total_loss = 0.0
    all_probs: List[float] = []
    all_labels: List[int] = []
    records: List[Dict[str, Any]] = []

    progress = sp.make_progress(
        loader,
        total=len(loader),
        desc=progress_desc,
        disable=not show_progress,
        leave=False,
        unit="batch",
    )

    for step, batch in enumerate(progress, start=1):
        if not getattr(loader, "device_prefetched", False):
            batch = sp.move_batch_to_device(batch, device)
        labels = batch["labels"]

        grad_context = nullcontext() if train else sp.torch.inference_mode()
        with grad_context:
            if amp_enabled:
                with sp.autocast(dtype=amp_dtype):
                    logits = model(
                        gene_input_ids=batch["gene_input_ids"],
                        gene_attention_mask=batch["gene_attention_mask"],
                        mirna_input_ids=batch["mirna_input_ids"],
                        mirna_attention_mask=batch["mirna_attention_mask"],
                        gene_token_type_ids=batch.get("gene_token_type_ids"),
                        mirna_token_type_ids=batch.get("mirna_token_type_ids"),
                        handcrafted_features=batch.get("handcrafted_features"),
                    )
                    loss = criterion(logits, labels)
            else:
                logits = model(
                    gene_input_ids=batch["gene_input_ids"],
                    gene_attention_mask=batch["gene_attention_mask"],
                    mirna_input_ids=batch["mirna_input_ids"],
                    mirna_attention_mask=batch["mirna_attention_mask"],
                    gene_token_type_ids=batch.get("gene_token_type_ids"),
                    mirna_token_type_ids=batch.get("mirna_token_type_ids"),
                    handcrafted_features=batch.get("handcrafted_features"),
                )
                loss = criterion(logits, labels)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        probabilities = sp.torch.sigmoid(logits.detach()).cpu().tolist()
        label_values = [int(value) for value in labels.detach().cpu().tolist()]
        total_loss += float(loss.item()) * len(label_values)
        all_probs.extend(float(value) for value in probabilities)
        all_labels.extend(label_values)

        if show_progress and (step == 1 or step % 20 == 0 or step == len(loader)):
            progress.set_postfix(
                loss=f"{total_loss / max(len(all_labels), 1):.4f}",
                rows=len(all_labels),
            )

        if capture_predictions:
            predicted_labels = [1 if prob >= 0.5 else 0 for prob in probabilities]
            for index, row_id in enumerate(batch["row_ids"]):
                record = {
                    "row_id": int(row_id),
                    group_key: batch["group_values"][index],
                    "label": label_values[index],
                    "probability": float(probabilities[index]),
                    "pred_label": predicted_labels[index],
                    "correct": int(predicted_labels[index] == label_values[index]),
                }
                if "genes" in batch:
                    record["gene"] = batch["genes"][index]
                if "mirnas" in batch:
                    record["noncodingRNA"] = batch["mirnas"][index]
                records.append(record)
    progress.close()

    metrics = sp.compute_binary_metrics(all_labels, all_probs)
    metrics["loss"] = total_loss / max(len(all_labels), 1)
    prediction_df = pd.DataFrame(records) if capture_predictions else None
    return metrics, prediction_df


def save_best_route2_artifacts(
    model,
    output_dir: Path,
    config: Route2Config,
    best_epoch: int,
    selected_threshold: float,
    best_val_raw_metrics: Dict[str, float],
    best_val_selected_metrics: Dict[str, float],
) -> None:
    gene_adapter_dir = output_dir / "best_gene_adapter"
    mirna_adapter_dir = output_dir / "best_mirna_adapter"
    model.gene_encoder.save_pretrained(gene_adapter_dir)
    model.mirna_encoder.save_pretrained(mirna_adapter_dir)

    head_payload = {
        "model_name": "dual_lora_interaction",
        "config": asdict(config),
        "best_epoch": best_epoch,
        "model_selection_metric": config.model_selection_metric,
        "best_val_raw_metrics": best_val_raw_metrics,
        "best_val_selected_metrics": best_val_selected_metrics,
        "selected_threshold": selected_threshold,
        "state_dict_kind": "head_only_trainable",
        "state_dict": extract_head_state_dict(model),
    }
    sp.safe_torch_save(head_payload, output_dir / "best_head_checkpoint.pt")


def load_route2_model_from_artifacts(
    config: Route2Config,
    UtrBertModel,
    RnaFmModel,
    peft_objects,
    gene_model_source: str,
    mirna_model_source: str,
    output_dir: Path,
):
    if sp.torch is None:
        raise RuntimeError("torch 不可用，无法从 LoRA artifacts 复载模型。")

    gene_base = rb.load_gene_model(gene_model_source, UtrBertModel)
    mirna_base = RnaFmModel.from_pretrained(mirna_model_source)
    for parameter in gene_base.parameters():
        parameter.requires_grad = False
    for parameter in mirna_base.parameters():
        parameter.requires_grad = False

    PeftModel = peft_objects["PeftModel"]
    gene_encoder = PeftModel.from_pretrained(gene_base, str(output_dir / "best_gene_adapter"))
    mirna_encoder = PeftModel.from_pretrained(mirna_base, str(output_dir / "best_mirna_adapter"))

    model = DualLoRAInteractionModel(
        gene_encoder=gene_encoder,
        mirna_encoder=mirna_encoder,
        interaction_hidden_dim=config.interaction_hidden_dim,
        classifier_hidden_dim=config.classifier_hidden_dim,
        attention_heads=config.attention_heads,
        interaction_top_k=config.interaction_top_k,
        dropout=config.dropout,
        handcrafted_feature_dim=len(get_selected_model_feature_columns(config)),
    )

    head_payload = sp.torch.load(output_dir / "best_head_checkpoint.pt", map_location="cpu")
    load_head_state_dict(model, head_payload["state_dict"])
    return model


def build_lora_model(
    config: Route2Config,
    UtrBertModel,
    RnaFmModel,
    peft_objects,
    gene_model_source: str,
    mirna_model_source: str,
) -> Tuple[Any, Dict[str, Any]]:
    if sp.torch is None:
        raise RuntimeError("torch 不可用，无法构建 LoRA 模型。")

    gene_base = rb.load_gene_model(gene_model_source, UtrBertModel)
    mirna_base = RnaFmModel.from_pretrained(mirna_model_source)
    for parameter in gene_base.parameters():
        parameter.requires_grad = False
    for parameter in mirna_base.parameters():
        parameter.requires_grad = False

    gene_target_summary = discover_lora_targets(
        backbone=gene_base,
        requested=config.gene_lora_target_modules,
        label="gene",
    )
    mirna_target_summary = discover_lora_targets(
        backbone=mirna_base,
        requested=config.mirna_lora_target_modules,
        label="miRNA",
    )

    LoraConfig = peft_objects["LoraConfig"]
    TaskType = peft_objects["TaskType"]
    get_peft_model = peft_objects["get_peft_model"]

    gene_encoder = get_peft_model(
        gene_base,
        LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=config.gene_lora_r,
            lora_alpha=config.gene_lora_alpha,
            target_modules=gene_target_summary["target_modules"],
            lora_dropout=config.lora_dropout,
            bias="none",
        ),
    )
    mirna_encoder = get_peft_model(
        mirna_base,
        LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=config.mirna_lora_r,
            lora_alpha=config.mirna_lora_alpha,
            target_modules=mirna_target_summary["target_modules"],
            lora_dropout=config.lora_dropout,
            bias="none",
        ),
    )

    model = DualLoRAInteractionModel(
        gene_encoder=gene_encoder,
        mirna_encoder=mirna_encoder,
        interaction_hidden_dim=config.interaction_hidden_dim,
        classifier_hidden_dim=config.classifier_hidden_dim,
        attention_heads=config.attention_heads,
        interaction_top_k=config.interaction_top_k,
        dropout=config.dropout,
        handcrafted_feature_dim=len(get_selected_model_feature_columns(config)),
    )

    lora_summary = {
        "backend": config.lora_backend,
        "gene": {
            "r": config.gene_lora_r,
            "alpha": config.gene_lora_alpha,
            "dropout": config.lora_dropout,
            **gene_target_summary,
        },
        "mirna": {
            "r": config.mirna_lora_r,
            "alpha": config.mirna_lora_alpha,
            "dropout": config.lora_dropout,
            **mirna_target_summary,
        },
    }
    return model, lora_summary


def build_feature_group_moe_lora_model(
    config: Route2Config,
    UtrBertModel,
    RnaFmModel,
    peft_objects,
    gene_model_source: str,
    mirna_model_source: str,
    expert_hidden_dim: int,
    router_hidden_dim: int,
    gate_temperature: float,
) -> Tuple[Any, Dict[str, Any]]:
    if sp.torch is None:
        raise RuntimeError("torch 不可用，无法构建 Route2 MoE LoRA 模型。")

    gene_base = rb.load_gene_model(gene_model_source, UtrBertModel)
    mirna_base = RnaFmModel.from_pretrained(mirna_model_source)
    for parameter in gene_base.parameters():
        parameter.requires_grad = False
    for parameter in mirna_base.parameters():
        parameter.requires_grad = False

    gene_target_summary = discover_lora_targets(
        backbone=gene_base,
        requested=config.gene_lora_target_modules,
        label="gene",
    )
    mirna_target_summary = discover_lora_targets(
        backbone=mirna_base,
        requested=config.mirna_lora_target_modules,
        label="miRNA",
    )

    LoraConfig = peft_objects["LoraConfig"]
    TaskType = peft_objects["TaskType"]
    get_peft_model = peft_objects["get_peft_model"]
    gene_encoder = get_peft_model(
        gene_base,
        LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=config.gene_lora_r,
            lora_alpha=config.gene_lora_alpha,
            target_modules=gene_target_summary["target_modules"],
            lora_dropout=config.lora_dropout,
            bias="none",
        ),
    )
    mirna_encoder = get_peft_model(
        mirna_base,
        LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=config.mirna_lora_r,
            lora_alpha=config.mirna_lora_alpha,
            target_modules=mirna_target_summary["target_modules"],
            lora_dropout=config.lora_dropout,
            bias="none",
        ),
    )
    model = FeatureGroupMoEInteractionModel(
        gene_encoder=gene_encoder,
        mirna_encoder=mirna_encoder,
        interaction_hidden_dim=config.interaction_hidden_dim,
        expert_hidden_dim=expert_hidden_dim,
        router_hidden_dim=router_hidden_dim,
        attention_heads=config.attention_heads,
        interaction_top_k=config.interaction_top_k,
        dropout=config.dropout,
        biological_feature_dim=len(get_selected_model_feature_columns(config)),
        gate_temperature=gate_temperature,
    )
    lora_summary = {
        "backend": config.lora_backend,
        "model_type": "feature_group_moe",
        "moe": {
            "expert_hidden_dim": expert_hidden_dim,
            "router_hidden_dim": router_hidden_dim,
            "gate_temperature": gate_temperature,
            "feature_group_dims": dict(model.feature_group_dims),
            "expert_names": list(model.expert_names),
        },
        "gene": {
            "r": config.gene_lora_r,
            "alpha": config.gene_lora_alpha,
            "dropout": config.lora_dropout,
            **gene_target_summary,
        },
        "mirna": {
            "r": config.mirna_lora_r,
            "alpha": config.mirna_lora_alpha,
            "dropout": config.lora_dropout,
            **mirna_target_summary,
        },
    }
    return model, lora_summary


def build_final_report(
    output_dir: Path,
    config: Route2Config,
    prep_summary: Dict[str, Any],
    merged_train_summary: Dict[str, Any],
    leakage_report: Dict[str, Any],
    validation_manifest: Dict[str, Any],
    train_summary: Dict[str, Any],
    aggregate_df: pd.DataFrame,
    lora_summary: Dict[str, Any],
    parameter_summary: Dict[str, Any],
    sequence_score_feature_summary: Dict[str, Any],
) -> None:
    train_rows = []
    for file_name, summary in prep_summary["train_files"].items():
        train_rows.append(
            {
                "file": file_name,
                "raw_rows": summary["raw_rows"],
                "three_prime_utr_rows": summary["filtered_rows"],
                "retained_pairs": summary["retained_rows"],
                "conflict_pairs": summary["conflict_pair_count"],
                "duplicate_rows": summary["duplicate_row_count"],
            }
        )
    test_rows = []
    for file_name, summary in prep_summary["test_files"].items():
        test_rows.append(
            {
                "file": file_name,
                "raw_rows": summary["raw_rows"],
                "three_prime_utr_rows": summary["filtered_rows"],
                "retained_pairs": summary["retained_rows"],
                "conflict_pairs": summary["conflict_pair_count"],
                "duplicate_rows": summary["duplicate_row_count"],
            }
        )

    aggregate_display = aggregate_df[
        [
            "report_name",
            "row_count",
            "positive_rate",
            "accuracy",
            "balanced_accuracy",
            "f1",
            "mcc",
            "auroc",
            "auprc",
            "loss",
        ]
    ].copy()

    report_lines = [
        "# Route 2 Dual LoRA Report",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Summary",
        "",
        "- Comparison mode: `single-seed time-constrained comparison`",
        f"- Config preset: `{config.config_preset}`",
        f"- Feature filter: `{config.feature_value}`",
        f"- Train files: {len(prep_summary['train_files'])}",
        f"- Test files: {len(prep_summary['test_files'])}",
        f"- Best epoch: {train_summary['best_epoch']}",
        f"- Model selection metric: `{train_summary['model_selection_metric']}`",
        f"- Selected threshold ({config.threshold_metric}): {train_summary['selected_threshold']:.4f}",
        f"- Handcrafted groups: `{config.handcrafted_include_groups or 'all/none by feature_set'}`",
        f"- Gene sequence score feature: `{config.gene_sequence_score_feature}`",
        "",
        "## Data Preparation",
        "",
        "### Train Files",
        "",
        r1.dataframe_to_markdown(pd.DataFrame(train_rows)),
        "",
        "### Test Files",
        "",
        r1.dataframe_to_markdown(pd.DataFrame(test_rows)),
        "",
        "### Training Pool",
        "",
        f"- Before overlap exclusion: {merged_train_summary['before_overlap_exclusion']['retained_rows']} pairs",
        f"- Removed exact train/test overlap pairs: {merged_train_summary['removed_exact_pair_overlap_count']}",
        f"- Final retained train pairs: {merged_train_summary['after_overlap_exclusion']['retained_rows']}",
        "",
        "## LoRA Setup",
        "",
        f"- Backend: `{lora_summary['backend']}`",
        f"- Gene target modules: `{', '.join(lora_summary['gene']['target_modules'])}`",
        f"- miRNA target modules: `{', '.join(lora_summary['mirna']['target_modules'])}`",
        f"- Gene LoRA r/alpha/dropout: {lora_summary['gene']['r']} / {lora_summary['gene']['alpha']} / {lora_summary['gene']['dropout']}",
        f"- miRNA LoRA r/alpha/dropout: {lora_summary['mirna']['r']} / {lora_summary['mirna']['alpha']} / {lora_summary['mirna']['dropout']}",
        f"- Family-balanced sampling: `{config.family_balanced_sampling}`",
        f"- Trainable parameters: {parameter_summary['trainable_parameters']} / {parameter_summary['total_parameters']} "
        f"({parameter_summary['trainable_ratio']:.4%})",
        "",
        "## Extra Sequence Feature",
        "",
        f"- Enabled: `{sequence_score_feature_summary['enabled']}`",
        f"- Feature name: `{sequence_score_feature_summary['feature_name']}`",
        f"- Feature columns: `{', '.join(sequence_score_feature_summary.get('feature_columns', [])) or 'none'}`",
        f"- Cache dir: `{sequence_score_feature_summary.get('cache_dir', '')}`",
        "",
        "## Validation Split",
        "",
        f"- Selected split seed: {validation_manifest['selected_split_seed']}",
        f"- Split score: {validation_manifest['split_score']:.6f}",
        f"- Train rows: {validation_manifest['split_manifest']['split_summary']['train']['row_count']}",
        f"- Val rows: {validation_manifest['split_manifest']['split_summary']['val']['row_count']}",
        "",
        "## Training Outcome",
        "",
        f"- Best val AUROC: {train_summary['best_val_raw_metrics']['auroc']:.4f}",
        f"- Best val AUPRC: {train_summary['best_val_raw_metrics']['auprc']:.4f}",
        f"- Best val selection metric ({train_summary['model_selection_metric']}): "
        f"{train_summary['best_val_raw_metrics'][train_summary['model_selection_metric']]:.4f}",
        f"- Best val balanced accuracy at selected threshold: {train_summary['best_val_selected_metrics']['balanced_accuracy']:.4f}",
        "",
        "## Test Results",
        "",
        r1.dataframe_to_markdown(aggregate_display),
        "",
    ]
    (output_dir / "final_report.md").write_text("\n".join(report_lines), encoding="utf-8")


PAPER_REPORT_NAME_MAP = {
    "AGO2_eCLIP_Klimentova2022_test": "Klimentova2022 test",
    "AGO2_CLASH_Hejret2023_test": "Hejret2023 test",
    "AGO2_eCLIP_Manakov2022_test": "Manakov2022 test",
    "AGO2_eCLIP_Manakov2022_leftout": "Manakov2022 leftout",
}


def infer_route2_display_name(train_files: List[Path], config: Route2Config) -> str:
    preset_label = {
        "route2-lora-base": "base",
        "route2-lora-best": "best",
    }.get(config.config_preset.strip().lower(), "custom")

    if len(train_files) == 1:
        train_name = train_files[0].name.lower()
        if "hejret2023" in train_name:
            dataset_label = "Hejret-only"
        elif "manakov2022" in train_name:
            dataset_label = "Manakov-only"
        else:
            dataset_label = train_files[0].stem
    else:
        dataset_label = "merged"
    return f"Route2 LoRA {dataset_label} {preset_label}"


def write_paper_row(
    output_dir: Path,
    train_files: List[Path],
    config: Route2Config,
    aggregate_df: pd.DataFrame,
) -> None:
    dataset_rows = aggregate_df[aggregate_df["report_name"].isin(PAPER_REPORT_NAME_MAP)].copy()
    if dataset_rows.empty:
        return

    metrics_by_report = {
        row["report_name"]: float(row["auprc"])
        for _, row in dataset_rows.iterrows()
    }
    display_name = infer_route2_display_name(train_files, config)
    ordered_values = {
        label: metrics_by_report.get(report_name, float("nan"))
        for report_name, label in PAPER_REPORT_NAME_MAP.items()
    }
    mean_aps = float(np.nanmean(list(ordered_values.values())))

    csv_row = {
        "display_name": display_name,
        "source": "this work (single seed)",
        "Mean APS": f"{mean_aps:.3f}",
        **{label: f"{value:.3f}" for label, value in ordered_values.items()},
    }
    pd.DataFrame([csv_row]).to_csv(output_dir / "paper_row.csv", index=False)

    json_payload = {
        "display_name": display_name,
        "source": "this work (single seed)",
        "mean_aps": mean_aps,
        "aps_by_test_dataset": ordered_values,
        "config_preset": config.config_preset,
        "train_files": [path.name for path in train_files],
        "seed": config.seed,
    }
    sp.save_json(output_dir / "paper_row.json", json_payload)


def train_route2_model(
    model,
    train_loader,
    val_loader,
    config: Route2Config,
    device,
    group_key: str,
    threshold_grid: List[float],
    output_dir: Path,
    journal,
) -> Tuple[Any, Dict[str, Any], float]:
    if sp.torch is None:
        raise RuntimeError("torch 不可用，无法训练 route2 模型。")

    model = model.to(device)
    optimizer = create_optimizer(model=model, config=config, device=device)
    scheduler = create_scheduler(optimizer=optimizer, config=config)
    criterion = sp.nn.BCEWithLogitsLoss()
    amp_enabled = device.type == "cuda"
    amp_dtype = sp.resolve_amp_dtype(config.amp_dtype, device)
    scaler = sp.GradScaler(enabled=amp_enabled and amp_dtype == sp.torch.float16)

    best_epoch = 0
    best_score = float("-inf")
    best_val_raw_metrics: Optional[Dict[str, float]] = None
    best_val_selected_metrics: Optional[Dict[str, float]] = None
    best_selected_threshold: float = 0.5
    history_rows: List[Dict[str, Any]] = []
    patience_used = 0

    journal.log(
        "train_start",
        max_epochs=config.max_epochs,
        early_stopping_patience=config.early_stopping_patience,
        model_selection_metric=config.model_selection_metric,
        threshold_metric=config.threshold_metric,
        family_balanced_sampling=config.family_balanced_sampling,
    )

    for epoch in range(1, config.max_epochs + 1):
        train_metrics, _ = run_epoch_route2(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            train=True,
            optimizer=optimizer,
            scaler=scaler if amp_enabled else None,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            group_key=group_key,
            progress_desc=f"route2 train e{epoch:02d}",
            show_progress=config.progress_bar,
        )
        if scheduler is not None:
            scheduler.step()

        val_raw_metrics, val_prediction_df = run_epoch_route2(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            train=False,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            capture_predictions=True,
            group_key=group_key,
            progress_desc=f"route2 val e{epoch:02d}",
            show_progress=config.progress_bar,
        )
        if val_prediction_df is None:
            raise RuntimeError("验证阶段未返回预测结果。")

        val_threshold, val_selected_metrics, _ = rge.select_threshold(
            labels=val_prediction_df["label"].astype(int).tolist(),
            probabilities=val_prediction_df["probability"].astype(float).tolist(),
            threshold_grid=threshold_grid,
            metric_name=config.threshold_metric,
        )

        lr_values = current_learning_rates(optimizer)
        history_row = {"epoch": epoch, "val_selected_threshold": val_threshold, **{f"lr_{k}": v for k, v in lr_values.items()}}
        history_row.update({f"train_{key}": value for key, value in train_metrics.items()})
        history_row.update({f"val_raw_{key}": value for key, value in val_raw_metrics.items()})
        history_row.update({f"val_selected_{key}": value for key, value in val_selected_metrics.items()})
        history_rows.append(history_row)

        journal.log(
            "epoch_completed",
            epoch=epoch,
            train_loss=train_metrics["loss"],
            train_auroc=train_metrics["auroc"],
            train_auprc=train_metrics["auprc"],
            val_loss=val_raw_metrics["loss"],
            val_auroc=val_raw_metrics["auroc"],
            val_auprc=val_raw_metrics["auprc"],
            val_selected_threshold=val_threshold,
            val_selected_metric=val_selected_metrics[config.threshold_metric],
            learning_rates=lr_values,
        )

        monitor_score = val_raw_metrics[config.model_selection_metric]
        if math.isnan(monitor_score):
            monitor_score = float("-inf")

        print(
            f"[route2] Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} train_auprc={train_metrics['auprc']:.4f} | "
            f"val_loss={val_raw_metrics['loss']:.4f} val_auprc={val_raw_metrics['auprc']:.4f} "
            f"| select_by_{config.model_selection_metric}={monitor_score:.4f}"
        )

        if monitor_score > best_score:
            best_score = monitor_score
            best_epoch = epoch
            best_val_raw_metrics = dict(val_raw_metrics)
            best_val_selected_metrics = dict(val_selected_metrics)
            best_selected_threshold = float(val_threshold)
            save_best_route2_artifacts(
                model=model,
                output_dir=output_dir,
                config=config,
                best_epoch=best_epoch,
                selected_threshold=best_selected_threshold,
                best_val_raw_metrics=best_val_raw_metrics,
                best_val_selected_metrics=best_val_selected_metrics,
            )
            patience_used = 0
            journal.log(
                "best_artifacts_updated",
                best_epoch=best_epoch,
                best_val_metric=best_score,
                selected_threshold=best_selected_threshold,
            )
        else:
            patience_used += 1
            if patience_used >= config.early_stopping_patience:
                journal.log(
                    "early_stopping_triggered",
                    epoch=epoch,
                    best_epoch=best_epoch,
                )
                break

    if best_val_raw_metrics is None or best_val_selected_metrics is None:
        raise RuntimeError("训练过程中未能选出最佳模型。")

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(output_dir / "history.tsv", sep="\t", index=False)

    train_summary = {
        "best_epoch": best_epoch,
        "model_selection_metric": config.model_selection_metric,
        "best_val_raw_metrics": best_val_raw_metrics,
        "best_val_selected_metrics": best_val_selected_metrics,
        "selected_threshold": best_selected_threshold,
    }
    return model, train_summary, best_selected_threshold


def run_route2_dual_lora(config: Route2Config) -> Path:
    if sp.torch is None:
        raise RuntimeError("route2 运行需要 torch。")

    config = apply_route2_preset(config)
    sp.set_seed(config.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = config.run_name.strip() or timestamp
    run_dir = Path(config.output_dir) / f"route2_dual_lora_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    journal = r1.ExperimentJournal(run_dir / "experiment_journal.jsonl")
    journal.log("route2_start", config=asdict(config))

    sp.ensure_compatible_environment()
    peft_objects = check_and_import_peft()
    RnaTokenizer, UtrBertModel, RnaFmModel = sp.check_and_import_multimolecule()
    device = sp.resolve_device(config.device)
    sp.configure_runtime_for_device(device)
    effective_num_workers = sp.resolve_num_workers(config.num_workers, device)
    effective_eval_batch_size = sp.resolve_eval_batch_size(config.batch_size, config.eval_batch_size)
    gene_model_source = rb.resolve_gene_model_source(config.gene_model_name)
    mirna_model_source = sp.resolve_model_source(config.mirna_model_name, "rnafm")
    resolved_amp_dtype = sp.resolve_amp_dtype(config.amp_dtype, device)
    selected_handcrafted_features = get_selected_handcrafted_feature_names(config)
    selected_model_features = get_selected_model_feature_names(config)
    selected_handcrafted_groups = get_selected_handcrafted_groups(config)
    include_group_names = _normalize_group_csv(config.handcrafted_include_groups)
    exclude_group_names = _normalize_group_csv(config.handcrafted_exclude_groups)

    run_config_payload = {
        **asdict(config),
        "resolved_device": str(device),
        "resolved_num_workers": effective_num_workers,
        "resolved_eval_batch_size": effective_eval_batch_size,
        "resolved_amp_dtype": str(resolved_amp_dtype) if resolved_amp_dtype is not None else None,
        "resolved_gene_model_name": gene_model_source,
        "resolved_mirna_model_name": mirna_model_source,
        "handcrafted_feature_columns": get_selected_handcrafted_feature_columns(config),
        "model_feature_columns": get_selected_model_feature_columns(config),
        "selected_handcrafted_groups": selected_handcrafted_groups,
        "selected_handcrafted_features": selected_handcrafted_features,
        "selected_model_features": selected_model_features,
        "parsed_handcrafted_include_groups": include_group_names,
        "parsed_handcrafted_exclude_groups": exclude_group_names,
        "gene_sequence_score_feature_columns": upf.get_feature_columns(config.gene_sequence_score_feature),
    }
    sp.save_json(run_dir / "run_config.json", run_config_payload)
    feature_reference_manifest = mhf.build_feature_reference_manifest(
        config.handcrafted_feature_set,
        include_groups=config.handcrafted_include_groups,
        exclude_groups=config.handcrafted_exclude_groups,
    )
    sp.save_json(run_dir / "feature_reference_manifest.json", feature_reference_manifest)
    journal.log("feature_reference_manifest_written", manifest=feature_reference_manifest)
    sp.save_json(
        run_dir / "sequence_score_feature_reference.json",
        upf.build_feature_reference_manifest(config.gene_sequence_score_feature),
    )

    train_files = r1.select_named_tsv_files(config.train_dir, config.train_files)
    test_files = r1.select_named_tsv_files(config.test_dir, config.test_files)

    train_frames: List[pd.DataFrame] = []
    train_summaries: Dict[str, Any] = {}
    for path in train_files:
        clean_df, summary = r1.load_clean_pair_dataframe(
            path=path,
            feature_value=config.feature_value,
            group_key=config.group_key,
            chunksize=config.chunksize,
            show_progress=config.progress_bar,
            source_label=f"train:{path.stem}",
        )
        train_frames.append(clean_df)
        train_summaries[path.name] = summary
        journal.log("train_file_prepared", file=path.name, summary=summary)

    test_frames: Dict[str, pd.DataFrame] = {}
    test_summaries: Dict[str, Any] = {}
    for path in test_files:
        clean_df, summary = r1.load_clean_pair_dataframe(
            path=path,
            feature_value=config.feature_value,
            group_key=config.group_key,
            chunksize=config.chunksize,
            show_progress=config.progress_bar,
            source_label=f"test:{path.stem}",
        )
        clean_df["source_dataset"] = path.name
        test_frames[path.name] = clean_df
        test_summaries[path.name] = summary
        journal.log("test_file_prepared", file=path.name, summary=summary)

    prep_summary = {"train_files": train_summaries, "test_files": test_summaries}
    sp.save_json(run_dir / "prep_summary.json", prep_summary)

    merged_train_df, merged_train_base_summary = r1.merge_training_dataframes(
        train_frames=train_frames,
        group_key=config.group_key,
    )
    journal.log("merged_train_created", summary=merged_train_base_summary)

    leakage_before = r1.build_leakage_report(
        train_df=merged_train_df,
        test_frames=test_frames,
        group_key=config.group_key,
    )
    if config.exclude_test_overlap:
        merged_train_df, removed_overlap_count = r1.exclude_test_overlap_rows(
            train_df=merged_train_df,
            test_frames=test_frames,
        )
    else:
        removed_overlap_count = 0

    merged_train_after_summary = {
        "retained_rows": int(len(merged_train_df)),
        "positive_rate": float(merged_train_df["label"].mean()),
        "group_count": int(merged_train_df[config.group_key].nunique()),
        "unique_gene_count": int(merged_train_df["gene"].nunique()),
        "unique_mirna_count": int(merged_train_df["noncodingRNA"].nunique()),
        "label_counts": {
            str(key): int(value)
            for key, value in merged_train_df["label"].value_counts().sort_index().to_dict().items()
        },
    }
    leakage_after = r1.build_leakage_report(
        train_df=merged_train_df,
        test_frames=test_frames,
        group_key=config.group_key,
    )
    leakage_report = {
        "before_exclusion": leakage_before,
        "after_exclusion": leakage_after,
        "exclude_test_overlap": config.exclude_test_overlap,
        "removed_exact_pair_overlap_count": removed_overlap_count,
    }
    if config.exclude_test_overlap and leakage_after["union_exact_pair_overlap_count"] != 0:
        raise RuntimeError("已启用 exclude_test_overlap，但训练集与测试集仍存在 exact pair overlap。")
    sp.save_json(run_dir / "leakage_report.json", leakage_report)

    merged_train_summary = {
        "before_overlap_exclusion": merged_train_base_summary,
        "after_overlap_exclusion": merged_train_after_summary,
        "removed_exact_pair_overlap_count": removed_overlap_count,
    }
    sp.save_json(run_dir / "merged_train_summary.json", merged_train_summary)
    journal.log("leakage_removal", leakage_report=leakage_report)

    split_df, split_manifest, selected_split_seed, split_score = r1.select_best_train_val_split(
        df=merged_train_df,
        group_key=config.group_key,
        val_ratio=config.val_ratio,
        base_seed=config.seed,
        candidates=config.val_split_candidates,
    )
    split_df = split_df.reset_index(drop=True)
    split_df["row_id"] = split_df.index.astype(int)
    train_families = set(split_df.loc[split_df["split"] == "train", config.group_key].astype(str))
    val_families = set(split_df.loc[split_df["split"] == "val", config.group_key].astype(str))
    if train_families & val_families:
        raise RuntimeError("train/val split 存在 noncodingRNA family 泄漏。")
    validation_manifest = {
        "selected_split_seed": selected_split_seed,
        "split_score": split_score,
        "split_manifest": split_manifest,
    }
    sp.save_json(run_dir / "validation_manifest.json", validation_manifest)
    split_df.drop(columns=["gene_seq_id", "mirna_seq_id"], errors="ignore").to_csv(
        run_dir / "validation_assignments.tsv",
        sep="\t",
        index=False,
    )
    journal.log(
        "validation_split_completed",
        selected_split_seed=selected_split_seed,
        split_score=split_score,
        split_summary=split_manifest["split_summary"],
    )

    split_df, test_frames, sequence_summary = r1.assign_global_sequence_ids(split_df, test_frames)
    journal.log("sequence_ids_assigned", summary=sequence_summary)

    handcrafted_summary: Dict[str, Any] = {
        "feature_set": config.handcrafted_feature_set,
        "enabled": bool(selected_handcrafted_features),
        "available_feature_groups": mhf.get_feature_groups(config.handcrafted_feature_set),
        "selected_handcrafted_groups": selected_handcrafted_groups,
        "selected_handcrafted_features": selected_handcrafted_features,
        "handcrafted_include_groups": include_group_names,
        "handcrafted_exclude_groups": exclude_group_names,
        "split_df": {"row_count": int(len(split_df)), "columns": []},
        "test_frames": {},
    }
    if selected_handcrafted_features:
        split_df = mhf.augment_dataframe_with_features(
            split_df,
            feature_set=config.handcrafted_feature_set,
            include_groups=config.handcrafted_include_groups,
            exclude_groups=config.handcrafted_exclude_groups,
        )
        handcrafted_summary["split_df"] = mhf.summarize_feature_columns(
            split_df,
            feature_set=config.handcrafted_feature_set,
            include_groups=config.handcrafted_include_groups,
            exclude_groups=config.handcrafted_exclude_groups,
        )
        augmented_test_frames: Dict[str, pd.DataFrame] = {}
        for test_name, test_df in test_frames.items():
            augmented_test_df = mhf.augment_dataframe_with_features(
                test_df,
                feature_set=config.handcrafted_feature_set,
                include_groups=config.handcrafted_include_groups,
                exclude_groups=config.handcrafted_exclude_groups,
            )
            augmented_test_frames[test_name] = augmented_test_df
            handcrafted_summary["test_frames"][test_name] = mhf.summarize_feature_columns(
                augmented_test_df,
                feature_set=config.handcrafted_feature_set,
                include_groups=config.handcrafted_include_groups,
                exclude_groups=config.handcrafted_exclude_groups,
            )
        test_frames = augmented_test_frames
    sp.save_json(run_dir / "handcrafted_feature_summary.json", handcrafted_summary)
    journal.log("handcrafted_features_prepared", summary=handcrafted_summary)

    split_df, test_frames, sequence_score_feature_summary, pll_cache_manifest_ref = build_sequence_score_summary(
        config=config,
        split_df=split_df,
        test_frames=test_frames,
        train_files=train_files,
        test_files=test_files,
        gene_model_source=gene_model_source,
        device=device,
        show_progress=config.progress_bar,
    )
    sp.save_json(run_dir / "sequence_score_feature_summary.json", sequence_score_feature_summary)
    sp.save_json(run_dir / "pll_cache_manifest_ref.json", pll_cache_manifest_ref)
    journal.log("sequence_score_feature_prepared", summary=sequence_score_feature_summary)

    cache_source_df = pd.concat(
        [split_df[["gene", "noncodingRNA"]]]
        + [frame[["gene", "noncodingRNA"]] for frame in test_frames.values()],
        ignore_index=True,
    )
    gene_tokenizer = rb.load_gene_tokenizer(gene_model_source, RnaTokenizer)
    mirna_tokenizer = RnaTokenizer.from_pretrained(mirna_model_source)
    gene_cache, mirna_cache = sp.build_dual_token_caches(
        df=cache_source_df,
        gene_tokenizer=gene_tokenizer,
        mirna_tokenizer=mirna_tokenizer,
        max_gene_len=config.max_gene_len,
        max_mirna_len=config.max_mirna_len,
        batch_size=config.token_cache_batch_size,
        enabled=config.cache_tokenized_sequences,
        show_progress=config.progress_bar,
    )
    journal.log(
        "token_cache_built",
        gene_cache_enabled=gene_cache is not None,
        mirna_cache_enabled=mirna_cache is not None,
    )

    collator = sp.DualTokenizerCollator(
        gene_tokenizer=gene_tokenizer,
        mirna_tokenizer=mirna_tokenizer,
        max_gene_len=config.max_gene_len,
        max_mirna_len=config.max_mirna_len,
        gene_cache=gene_cache,
        mirna_cache=mirna_cache,
    )

    train_df = split_df[split_df["split"] == "train"].reset_index(drop=True)
    val_df = split_df[split_df["split"] == "val"].reset_index(drop=True)
    train_sampler = None
    train_sampler_summary = {"enabled": False}
    if config.family_balanced_sampling:
        train_sampler, train_sampler_summary = r1.build_family_weighted_sampler(
            train_df=train_df,
            group_key=config.group_key,
            seed=config.seed,
        )
    journal.log("train_sampler_ready", summary=train_sampler_summary)

    train_loader = r1.build_single_loader(
        df=train_df,
        group_key=config.group_key,
        batch_size=config.batch_size,
        collator=collator,
        device=device,
        num_workers=effective_num_workers,
        prefetch_factor=config.prefetch_factor,
        include_sequences=False,
        shuffle=train_sampler is None,
        enable_cuda_prefetch=config.cuda_prefetch,
        sampler=train_sampler,
    )
    val_loader = r1.build_single_loader(
        df=val_df,
        group_key=config.group_key,
        batch_size=effective_eval_batch_size,
        collator=collator,
        device=device,
        num_workers=effective_num_workers,
        prefetch_factor=config.prefetch_factor,
        include_sequences=False,
        shuffle=False,
        enable_cuda_prefetch=config.cuda_prefetch,
    )

    threshold_grid = rge.build_threshold_grid(
        rge.EvalConfig(
            input_file="",
            output_dir="",
            threshold_metric=config.threshold_metric,
            threshold_min=config.threshold_min,
            threshold_max=config.threshold_max,
            threshold_step=config.threshold_step,
        )
    )

    model, lora_summary = build_lora_model(
        config=config,
        UtrBertModel=UtrBertModel,
        RnaFmModel=RnaFmModel,
        peft_objects=peft_objects,
        gene_model_source=gene_model_source,
        mirna_model_source=mirna_model_source,
    )
    sp.save_json(run_dir / "lora_summary.json", lora_summary)
    journal.log("lora_model_built", lora_summary=lora_summary)

    parameter_summary = summarize_trainable_parameters(model)
    sp.save_json(run_dir / "trainable_parameter_summary.json", parameter_summary)
    journal.log("trainable_parameter_summary", summary=parameter_summary)

    _, train_summary, _ = train_route2_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        group_key=config.group_key,
        threshold_grid=threshold_grid,
        output_dir=run_dir,
        journal=journal,
    )
    sp.save_json(run_dir / "train_summary.json", train_summary)

    reloaded_model = load_route2_model_from_artifacts(
        config=config,
        UtrBertModel=UtrBertModel,
        RnaFmModel=RnaFmModel,
        peft_objects=peft_objects,
        gene_model_source=gene_model_source,
        mirna_model_source=mirna_model_source,
        output_dir=run_dir,
    )
    reloaded_model = reloaded_model.to(device)
    journal.log("best_model_reloaded_from_artifacts")

    criterion = sp.nn.BCEWithLogitsLoss()
    amp_enabled = device.type == "cuda"
    amp_dtype = sp.resolve_amp_dtype(config.amp_dtype, device)
    val_raw_metrics, val_prediction_df = run_epoch_route2(
        model=reloaded_model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        train=False,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        capture_predictions=True,
        group_key=config.group_key,
        progress_desc="route2 val best",
        show_progress=config.progress_bar,
    )
    if val_prediction_df is None:
        raise RuntimeError("最佳模型验证阶段未返回预测结果。")
    selected_threshold, val_selected_metrics, val_threshold_curve = rge.select_threshold(
        labels=val_prediction_df["label"].astype(int).tolist(),
        probabilities=val_prediction_df["probability"].astype(float).tolist(),
        threshold_grid=threshold_grid,
        metric_name=config.threshold_metric,
    )
    val_threshold_curve.to_csv(run_dir / "val_threshold_search.tsv", sep="\t", index=False)
    train_summary["selected_threshold"] = float(selected_threshold)
    train_summary["best_val_raw_metrics"] = dict(val_raw_metrics)
    train_summary["best_val_selected_metrics"] = dict(val_selected_metrics)
    sp.save_json(run_dir / "train_summary.json", train_summary)

    all_prediction_frames: List[pd.DataFrame] = []
    test_rows: List[Dict[str, Any]] = []
    leftout_row: Optional[Dict[str, Any]] = None
    for test_file_name, test_df in test_frames.items():
        metrics_row, prediction_df = r1.evaluate_test_file(
            model=reloaded_model,
            test_file_name=test_file_name,
            test_df=test_df.reset_index(drop=True),
            selected_threshold=selected_threshold,
            collator=collator,
            config=config,
            device=device,
            output_root=run_dir,
            journal=journal,
        )
        row = {
            "report_name": Path(test_file_name).stem,
            "test_file": test_file_name,
            "selected_threshold": selected_threshold,
            "is_ood": int("leftout" in test_file_name.lower()),
            **metrics_row,
        }
        test_rows.append(row)
        all_prediction_frames.append(prediction_df)
        if test_file_name == "AGO2_eCLIP_Manakov2022_leftout.tsv":
            leftout_row = row.copy()

    overall_predictions = pd.concat(all_prediction_frames, ignore_index=True)
    overall_metrics = r1.summarize_thresholded_predictions(
        prediction_df=overall_predictions,
        threshold=selected_threshold,
        loss=float(
            np.average(
                np.asarray([row["loss"] for row in test_rows], dtype=np.float64),
                weights=np.asarray([row["row_count"] for row in test_rows], dtype=np.float64),
            )
        ),
    )
    test_rows.append(
        {
            "report_name": "overall_weighted",
            "test_file": "ALL_TEST_FILES",
            "selected_threshold": selected_threshold,
            "is_ood": 0,
            **overall_metrics,
        }
    )
    if leftout_row is not None:
        test_rows.append(
            {
                **leftout_row,
                "report_name": "ood_leftout",
                "is_ood": 1,
            }
        )

    aggregate_df = pd.DataFrame(test_rows)
    aggregate_df.to_csv(run_dir / "aggregate_test_metrics.csv", index=False)
    sp.save_json(
        run_dir / "aggregate_test_metrics.json",
        {"rows": aggregate_df.to_dict(orient="records")},
    )
    journal.log("aggregate_metrics_written", rows=aggregate_df.to_dict(orient="records"))

    write_paper_row(
        output_dir=run_dir,
        train_files=train_files,
        config=config,
        aggregate_df=aggregate_df,
    )
    journal.log("paper_row_written", path=str(run_dir / "paper_row.csv"))

    build_final_report(
        output_dir=run_dir,
        config=config,
        prep_summary=prep_summary,
        merged_train_summary=merged_train_summary,
        leakage_report=leakage_report,
        validation_manifest=validation_manifest,
        train_summary=train_summary,
        aggregate_df=aggregate_df,
        lora_summary=lora_summary,
        parameter_summary=parameter_summary,
        sequence_score_feature_summary=sequence_score_feature_summary,
    )
    journal.log("final_report_written", path=str(run_dir / "final_report.md"))
    print(f"[Done] 结果已写入 {run_dir}")
    return run_dir


def main() -> None:
    config = parse_args()
    run_route2_dual_lora(config)


if __name__ == "__main__":
    main()
