from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import mirna_handcrafted_features as mhf
import repeated_group_split_eval as rge
import supervised_poc as sp


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class ExperimentJournal:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **payload: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default))
            handle.write("\n")


@dataclass
class Route1Config:
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
    batch_size: int = 128
    eval_batch_size: int = 256
    max_epochs: int = 5
    early_stopping_patience: int = 5
    seed: int = 42
    device: str = "auto"
    gene_model_name: str = "multimolecule/utrbert-3mer"
    mirna_model_name: str = "multimolecule/rnafm"
    max_gene_len: int = 64
    max_mirna_len: int = 32
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    dropout: float = 0.2
    classifier_hidden_dim: int = 256
    interaction_hidden_dim: int = 128
    attention_heads: int = 4
    interaction_top_k: int = 16
    architecture: str = "cross_attention"
    local_attention_radius: int = 5
    interaction_cnn_channels: int = 16
    interaction_scorer: str = "dot_product"
    feature_processor: str = "interaction_stats"
    handcrafted_feature_set: str = "none"
    chunksize: int = 200000
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
    family_balanced_sampling: bool = False
    subsample_train_positive_count: int = 0
    subsample_train_negative_count: int = 0
    subsample_seed: int = 0


def parse_args() -> Route1Config:
    parser = argparse.ArgumentParser(
        description="路线1微调脚本：合并 train_data，冻结 backbone，只训练交互头，并独立评估 test_data"
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
        choices=["custom", "route1-v2-base", "route1-v2-best"],
    )
    parser.add_argument("--feature-value", default="three_prime_utr")
    parser.add_argument("--group-key", default="noncodingRNA_fam")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--val-split-candidates", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gene-model-name", default="multimolecule/utrbert-3mer")
    parser.add_argument("--mirna-model-name", default="multimolecule/rnafm")
    parser.add_argument("--max-gene-len", type=int, default=64)
    parser.add_argument("--max-mirna-len", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--classifier-hidden-dim", type=int, default=256)
    parser.add_argument("--interaction-hidden-dim", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--interaction-top-k", type=int, default=16)
    parser.add_argument(
        "--architecture",
        default="cross_attention",
        choices=["cross_attention", "hierarchical_cnn"],
    )
    parser.add_argument("--local-attention-radius", type=int, default=5)
    parser.add_argument("--interaction-cnn-channels", type=int, default=16)
    parser.add_argument(
        "--interaction-scorer",
        default="dot_product",
        choices=["dot_product", "biaffine"],
    )
    parser.add_argument(
        "--feature-processor",
        default="interaction_stats",
        choices=["interaction_stats", "dot_product_stats", "biaffine_stats"],
    )
    parser.add_argument(
        "--handcrafted-feature-set",
        default="none",
        choices=["none", "classic_seed_context", "classic_seed_context_extended"],
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
    parser.add_argument("--family-balanced-sampling", action="store_true")
    parser.add_argument("--subsample-train-positive-count", type=int, default=0)
    parser.add_argument("--subsample-train-negative-count", type=int, default=0)
    parser.add_argument("--subsample-seed", type=int, default=0)
    parser.set_defaults(exclude_test_overlap=True)
    args = parser.parse_args()
    config_kwargs = vars(args)
    config_kwargs["cache_tokenized_sequences"] = not config_kwargs.pop("disable_token_cache")
    config_kwargs["progress_bar"] = not config_kwargs.pop("disable_progress_bar")
    config_kwargs["cuda_prefetch"] = not config_kwargs.pop("disable_cuda_prefetch")
    config_kwargs["fused_optimizer"] = not config_kwargs.pop("disable_fused_optimizer")
    return Route1Config(**config_kwargs)


def collect_tsv_files(directory: str) -> List[Path]:
    files = sorted(Path(directory).glob("*.tsv"))
    if not files:
        raise FileNotFoundError(f"未在 {directory!r} 中找到任何 .tsv 文件。")
    return files


def parse_csv_names(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def select_named_tsv_files(directory: str, requested_names: str) -> List[Path]:
    all_files = collect_tsv_files(directory)
    requested = parse_csv_names(requested_names)
    if not requested:
        return all_files

    file_map = {path.name: path for path in all_files}
    missing = [name for name in requested if name not in file_map]
    if missing:
        raise FileNotFoundError(
            f"在 {directory!r} 中找不到以下文件: {missing}. 可用文件: {sorted(file_map)}"
        )
    return [file_map[name] for name in requested]


def apply_config_preset(config: Route1Config) -> Route1Config:
    preset = config.config_preset.strip().lower()
    if preset in {"", "custom"}:
        return config
    if preset == "route1-v2-base":
        config.max_gene_len = 64
        config.max_mirna_len = 32
        config.interaction_hidden_dim = 128
        config.classifier_hidden_dim = 256
        config.attention_heads = 4
        config.interaction_top_k = 16
        config.dropout = 0.2
        config.learning_rate = 2e-4
        config.weight_decay = 1e-2
        config.max_epochs = 12
        config.early_stopping_patience = 3
        config.model_selection_metric = "auprc"
        return config
    if preset == "route1-v2-best":
        config.max_gene_len = 128
        config.max_mirna_len = 32
        config.interaction_hidden_dim = 256
        config.classifier_hidden_dim = 512
        config.attention_heads = 8
        config.interaction_top_k = 32
        config.dropout = 0.1
        config.learning_rate = 1e-4
        config.weight_decay = 1e-2
        config.max_epochs = 12
        config.early_stopping_patience = 3
        config.model_selection_metric = "auprc"
        config.family_balanced_sampling = True
        return config
    raise ValueError(f"未知 config_preset: {config.config_preset!r}")


def choose_group_value(values: Iterable[Any]) -> Tuple[str, bool]:
    normalized = []
    for value in values:
        text = "unknown" if pd.isna(value) else str(value).strip()
        if not text:
            text = "unknown"
        normalized.append(text)

    non_unknown = sorted({value for value in normalized if value != "unknown"})
    if not non_unknown:
        return "unknown", False
    return non_unknown[0], len(non_unknown) > 1


def combine_source_values(values: Iterable[Any]) -> str:
    normalized = sorted({str(value).strip() for value in values if str(value).strip()})
    return ";".join(normalized)


def dedupe_pair_dataframe(
    df: pd.DataFrame,
    group_key: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if df.empty:
        empty_summary = {
            "input_rows": 0,
            "post_conflict_rows": 0,
            "retained_rows": 0,
            "conflict_pair_count": 0,
            "conflict_row_count": 0,
            "duplicate_row_count": 0,
            "group_conflict_pair_count": 0,
        }
        return df.copy(), empty_summary

    pair_columns = ["gene", "noncodingRNA"]
    unique_pair_labels = df[pair_columns + ["label"]].drop_duplicates()
    conflicting_label_rows = unique_pair_labels.duplicated(
        subset=pair_columns, keep=False
    )
    conflict_pairs = unique_pair_labels.loc[
        conflicting_label_rows, pair_columns
    ].drop_duplicates()
    conflict_row_count = 0
    working = df.copy()
    if not conflict_pairs.empty:
        conflict_lookup = conflict_pairs.assign(_label_conflict=True)
        working = working.merge(
            conflict_lookup,
            on=pair_columns,
            how="left",
            sort=False,
            validate="many_to_one",
        )
        conflict_mask = working.pop("_label_conflict").eq(True).to_numpy(dtype=bool)
        conflict_row_count = int(conflict_mask.sum())
        working = working.loc[~conflict_mask].copy()

    first_pair_rows = working.drop_duplicates(subset=pair_columns, keep="first")
    duplicate_row_count = int(len(working) - len(first_pair_rows))
    # In the standard miRNA datasets, family is a deterministic property of the
    # miRNA and every input frame has one source label. Preserve first-occurrence
    # group order with a vectorized path instead of iterating millions of pairs.
    mirna_group_counts = working.groupby("noncodingRNA", sort=False)[group_key].nunique(
        dropna=False
    )
    group_is_mirna_constant = bool(mirna_group_counts.empty or mirna_group_counts.max() <= 1)
    source_is_constant = bool(working["source_dataset"].nunique(dropna=False) <= 1)
    if group_is_mirna_constant and source_is_constant:
        dedup_df = first_pair_rows.reset_index(drop=True)
        dedup_df = dedup_df[["gene", "noncodingRNA", group_key, "label", "source_dataset"]]
        summary = {
            "input_rows": int(len(df)),
            "post_conflict_rows": int(len(working)),
            "retained_rows": int(len(dedup_df)),
            "conflict_pair_count": int(len(conflict_pairs)),
            "conflict_row_count": conflict_row_count,
            "duplicate_row_count": duplicate_row_count,
            "group_conflict_pair_count": 0,
            "dedupe_backend": "vectorized_constant_group_source",
        }
        return dedup_df, summary

    grouped = working.groupby(pair_columns, sort=False, as_index=False)
    group_conflict_pair_count = 0
    aggregated_rows: List[Dict[str, Any]] = []
    for (gene, mirna), group in grouped:
        chosen_group_value, has_group_conflict = choose_group_value(group[group_key].tolist())
        if has_group_conflict:
            group_conflict_pair_count += 1
        aggregated_rows.append(
            {
                "gene": gene,
                "noncodingRNA": mirna,
                group_key: chosen_group_value,
                "label": int(group["label"].iloc[0]),
                "source_dataset": combine_source_values(group["source_dataset"].tolist()),
            }
        )

    dedup_df = pd.DataFrame(aggregated_rows)
    if not dedup_df.empty:
        dedup_df = dedup_df[["gene", "noncodingRNA", group_key, "label", "source_dataset"]]

    summary = {
        "input_rows": int(len(df)),
        "post_conflict_rows": int(len(working)),
        "retained_rows": int(len(dedup_df)),
        "conflict_pair_count": int(len(conflict_pairs)),
        "conflict_row_count": conflict_row_count,
        "duplicate_row_count": duplicate_row_count,
        "group_conflict_pair_count": group_conflict_pair_count,
    }
    return dedup_df, summary


def load_clean_pair_dataframe(
    path: Path,
    feature_value: str,
    group_key: str,
    chunksize: int,
    show_progress: bool,
    source_label: str,
    raw_row_limit: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    usecols = ["gene", "noncodingRNA", group_key, "feature", "label"]
    raw_rows = 0
    filtered_rows = 0
    chunks: List[pd.DataFrame] = []
    normalized_feature_value = str(feature_value).strip().lower()
    use_all_features = normalized_feature_value in {"all", "*", "any", ""}
    retained_feature_counts: Dict[str, int] = {}

    progress = sp.make_progress(
        desc=f"load {source_label}",
        disable=not show_progress,
        unit="rows",
        unit_scale=True,
    )
    effective_chunksize = int(chunksize)
    if raw_row_limit > 0:
        effective_chunksize = min(effective_chunksize, int(raw_row_limit))
    for chunk in pd.read_csv(path, sep="\t", usecols=usecols, chunksize=effective_chunksize):
        if raw_row_limit > 0 and raw_rows + len(chunk) > raw_row_limit:
            chunk = chunk.iloc[: raw_row_limit - raw_rows].copy()
        raw_rows += len(chunk)
        progress.update(len(chunk))
        if use_all_features:
            chunk = chunk.copy()
        else:
            chunk = chunk[chunk["feature"].fillna("") == feature_value].copy()
        filtered_rows += len(chunk)
        if chunk.empty:
            continue
        feature_counts = chunk["feature"].fillna("unknown").astype(str).value_counts().to_dict()
        for feature_name, count in feature_counts.items():
            retained_feature_counts[feature_name] = retained_feature_counts.get(feature_name, 0) + int(count)

        chunk["gene"] = chunk["gene"].map(sp.normalize_rna)
        chunk["noncodingRNA"] = chunk["noncodingRNA"].map(sp.normalize_rna)
        chunk[group_key] = (
            chunk[group_key]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .replace("", "unknown")
        )
        chunk["label"] = chunk["label"].astype(int)
        chunk["source_dataset"] = path.name
        chunks.append(chunk[["gene", "noncodingRNA", group_key, "label", "source_dataset"]])
        if raw_row_limit > 0 and raw_rows >= raw_row_limit:
            break
    progress.close()

    if not chunks:
        if use_all_features:
            raise ValueError(f"未在 {path} 中找到可用样本。")
        raise ValueError(f"未在 {path} 中找到 feature == {feature_value!r} 的样本。")

    combined = pd.concat(chunks, ignore_index=True)
    dedup_df, dedup_summary = dedupe_pair_dataframe(combined, group_key=group_key)
    if dedup_df.empty:
        raise ValueError(f"{path} 在清洗后为空。")

    label_counts = dedup_df["label"].value_counts().sort_index().to_dict()
    summary = {
        "path": str(path),
        "source_dataset": path.name,
        "raw_row_limit": int(raw_row_limit),
        "raw_input_truncated_for_poc": bool(raw_row_limit > 0),
        "feature_filter": "all" if use_all_features else str(feature_value),
        "used_all_features": bool(use_all_features),
        "raw_rows": int(raw_rows),
        "filtered_rows": int(filtered_rows),
        "feature_counts_after_filter": {
            str(key): int(retained_feature_counts[key])
            for key in sorted(retained_feature_counts)
        },
        **dedup_summary,
        "label_counts": {str(key): int(value) for key, value in label_counts.items()},
        "positive_rate": float(dedup_df["label"].mean()),
        "group_count": int(dedup_df[group_key].nunique()),
        "unique_gene_count": int(dedup_df["gene"].nunique()),
        "unique_mirna_count": int(dedup_df["noncodingRNA"].nunique()),
    }
    return dedup_df, summary


def merge_training_dataframes(
    train_frames: List[pd.DataFrame],
    group_key: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    merged = pd.concat(train_frames, ignore_index=True)
    dedup_df, dedup_summary = dedupe_pair_dataframe(merged, group_key=group_key)
    if dedup_df.empty:
        raise ValueError("合并训练集在清洗后为空。")

    label_counts = dedup_df["label"].value_counts().sort_index().to_dict()
    summary = {
        "input_rows": int(len(merged)),
        **dedup_summary,
        "label_counts": {str(key): int(value) for key, value in label_counts.items()},
        "positive_rate": float(dedup_df["label"].mean()),
        "group_count": int(dedup_df[group_key].nunique()),
        "unique_gene_count": int(dedup_df["gene"].nunique()),
        "unique_mirna_count": int(dedup_df["noncodingRNA"].nunique()),
    }
    return dedup_df, summary


def pair_index_from_df(df: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(df[["gene", "noncodingRNA"]])


def build_leakage_report(
    train_df: pd.DataFrame,
    test_frames: Dict[str, pd.DataFrame],
    group_key: str,
) -> Dict[str, Any]:
    pair_columns = ["gene", "noncodingRNA"]
    train_pairs = train_df[pair_columns].drop_duplicates()
    train_families = set(train_df[group_key].astype(str))
    per_test: Dict[str, Any] = {}
    union_test_pair_frames = []
    for file_name, test_df in test_frames.items():
        test_pairs = test_df[pair_columns].drop_duplicates()
        overlap_count = int(
            len(train_pairs.merge(test_pairs, on=pair_columns, how="inner", sort=False))
        )
        test_families = set(test_df[group_key].astype(str))
        test_only_families = sorted(test_families - train_families)
        per_test[file_name] = {
            "test_row_count": int(len(test_df)),
            "exact_pair_overlap_count": overlap_count,
            "family_overlap_count": int(len(train_families & test_families)),
            "test_only_family_count": int(len(test_only_families)),
            "example_test_only_families": test_only_families[:20],
        }
        union_test_pair_frames.append(test_df[pair_columns].copy())

    union_test_pairs = pd.concat(union_test_pair_frames, ignore_index=True).drop_duplicates()
    union_overlap_count = int(
        len(train_pairs.merge(union_test_pairs, on=pair_columns, how="inner", sort=False))
    )
    return {
        "total_train_rows": int(len(train_df)),
        "total_train_unique_pairs": int(len(train_df)),
        "per_test_file": per_test,
        "union_exact_pair_overlap_count": union_overlap_count,
    }


def exclude_test_overlap_rows(
    train_df: pd.DataFrame,
    test_frames: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, int]:
    pair_columns = ["gene", "noncodingRNA"]
    union_test_pairs = pd.concat(
        [frame[pair_columns].copy() for frame in test_frames.values()],
        ignore_index=True,
    ).drop_duplicates()
    overlap_lookup = union_test_pairs.assign(_test_overlap=True)
    merged = train_df.merge(
        overlap_lookup,
        on=pair_columns,
        how="left",
        sort=False,
        validate="many_to_one",
    )
    overlap_mask = merged.pop("_test_overlap").eq(True).to_numpy(dtype=bool)
    removed_count = int(overlap_mask.sum())
    if removed_count == 0:
        return train_df.copy().reset_index(drop=True), 0
    return merged.loc[~overlap_mask].reset_index(drop=True), removed_count


def randomized_group_train_val_split(
    df: pd.DataFrame,
    group_key: str,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not 0.0 < val_ratio < 0.5:
        raise ValueError("val_ratio 必须满足 0 < val_ratio < 0.5。")

    train_ratio = 1.0 - val_ratio
    group_stats = (
        df.groupby(group_key)
        .agg(sample_count=("label", "size"), pos_count=("label", "sum"))
        .reset_index()
    )
    group_stats["neg_count"] = group_stats["sample_count"] - group_stats["pos_count"]
    group_stats["pos_rate"] = group_stats["pos_count"] / group_stats["sample_count"]
    mixed_groups = group_stats[
        (group_stats["pos_count"] > 0) & (group_stats["neg_count"] > 0)
    ].copy()
    if len(mixed_groups) < 2:
        raise ValueError("混合标签 family 数量不足，无法构建 train/val split。")

    split_order = ["train", "val"]
    ratio_map = {"train": train_ratio, "val": val_ratio}
    total_count = float(group_stats["sample_count"].sum())
    total_pos = float(group_stats["pos_count"].sum())
    global_pos_rate = total_pos / max(total_count, 1.0)
    rng = random.Random(seed)

    top_train_pool_size = min(len(mixed_groups), max(5, len(mixed_groups) // 6))
    top_train_pool = mixed_groups.nlargest(top_train_pool_size, "sample_count")
    train_seed = top_train_pool.sample(n=1, random_state=seed).iloc[0].to_dict()

    remaining_mixed = mixed_groups.loc[mixed_groups[group_key] != train_seed[group_key]].copy()
    small_pool_size = min(len(remaining_mixed), max(6, len(remaining_mixed) // 3))
    val_seed = (
        remaining_mixed.nsmallest(small_pool_size, "sample_count")
        .sample(n=1, random_state=seed + 1)
        .iloc[0]
        .to_dict()
    )

    split_state = {
        split: {"rows": [], "sample_count": 0, "pos_count": 0, "neg_count": 0}
        for split in split_order
    }
    seeded_groups = set()
    for split_name, group_row in zip(split_order, [train_seed, val_seed]):
        sp._append_group_to_split(split_name, group_row, split_state)
        seeded_groups.add(group_row[group_key])

    remaining = group_stats.loc[~group_stats[group_key].isin(seeded_groups)].copy()
    remaining["base_priority"] = remaining["sample_count"] * (
        1.0 + (remaining["pos_rate"] - global_pos_rate).abs()
    )
    remaining["jitter"] = [rng.uniform(-0.2, 0.2) for _ in range(len(remaining))]
    remaining["priority"] = remaining["base_priority"] * (1.0 + remaining["jitter"])
    remaining = remaining.sort_values(["priority", "sample_count"], ascending=False)

    target_count = {split: ratio_map[split] * total_count for split in split_order}
    target_pos = {split: ratio_map[split] * total_pos for split in split_order}

    def assignment_score(split_name: str, group_row: Dict[str, Any]) -> float:
        new_count = split_state[split_name]["sample_count"] + int(group_row["sample_count"])
        new_pos = split_state[split_name]["pos_count"] + int(group_row["pos_count"])
        count_ratio = new_count / max(target_count[split_name], 1.0)
        pos_ratio = new_pos / max(target_pos[split_name], 1.0)
        count_gap = abs(count_ratio - 1.0)
        pos_gap = abs(pos_ratio - 1.0)
        overflow = max(count_ratio - 1.0, 0.0)
        jitter = rng.uniform(0.0, 0.02)
        return (1.5 * count_gap) + (0.75 * pos_gap) + (3.0 * overflow) + jitter

    for _, row in remaining.iterrows():
        group_row = row.to_dict()
        best_split = min(split_order, key=lambda name: assignment_score(name, group_row))
        sp._append_group_to_split(best_split, group_row, split_state)

    manifest_groups: List[Dict[str, Any]] = []
    group_to_split: Dict[str, str] = {}
    split_summary: Dict[str, Any] = {}
    for split_name in split_order:
        rows = split_state[split_name]["rows"]
        if split_state[split_name]["pos_count"] == 0 or split_state[split_name]["neg_count"] == 0:
            raise ValueError(f"{split_name} split 未覆盖正负两类。")
        split_summary[split_name] = {
            "row_count": int(split_state[split_name]["sample_count"]),
            "target_row_ratio": float(ratio_map[split_name]),
            "actual_row_ratio": float(split_state[split_name]["sample_count"] / total_count),
            "positive_count": int(split_state[split_name]["pos_count"]),
            "negative_count": int(split_state[split_name]["neg_count"]),
            "positive_rate": float(
                split_state[split_name]["pos_count"] / split_state[split_name]["sample_count"]
            ),
            "group_count": len(rows),
        }
        for row in rows:
            manifest_groups.append(
                {
                    group_key: row[group_key],
                    "split": split_name,
                    "sample_count": int(row["sample_count"]),
                    "pos_count": int(row["pos_count"]),
                    "neg_count": int(row["neg_count"]),
                    "pos_rate": float(row["pos_rate"]),
                }
            )
            group_to_split[str(row[group_key])] = split_name

    out_df = df.copy()
    out_df["split"] = out_df[group_key].map(lambda value: group_to_split[str(value)])
    for split_name in split_order:
        subset = out_df[out_df["split"] == split_name]
        if subset.empty or subset["label"].nunique() < 2:
            raise ValueError(f"{split_name} split 无法用于训练或验证。")

    manifest = {
        "group_key": group_key,
        "ratios": ratio_map,
        "split_seed": seed,
        "global_summary": {
            "row_count": int(len(out_df)),
            "positive_count": int(out_df["label"].sum()),
            "negative_count": int(len(out_df) - out_df["label"].sum()),
            "positive_rate": float(out_df["label"].mean()),
            "group_count": int(out_df[group_key].nunique()),
        },
        "split_summary": split_summary,
        "group_assignments": manifest_groups,
    }
    return out_df, manifest


def score_train_val_manifest(manifest: Dict[str, Any]) -> float:
    global_positive_rate = manifest["global_summary"]["positive_rate"]
    score = 0.0
    for split_name, summary in manifest["split_summary"].items():
        score += abs(summary["actual_row_ratio"] - summary["target_row_ratio"])
        score += 0.5 * abs(summary["positive_rate"] - global_positive_rate)
    return float(score)


def select_best_train_val_split(
    df: pd.DataFrame,
    group_key: str,
    val_ratio: float,
    base_seed: int,
    candidates: int,
) -> Tuple[pd.DataFrame, Dict[str, Any], int, float]:
    if candidates <= 0:
        raise ValueError("val_split_candidates 必须大于 0。")

    best_df: Optional[pd.DataFrame] = None
    best_manifest: Optional[Dict[str, Any]] = None
    best_seed = base_seed
    best_score = float("inf")
    for candidate_index in range(candidates):
        candidate_seed = base_seed * 1000 + candidate_index
        candidate_df, candidate_manifest = randomized_group_train_val_split(
            df=df,
            group_key=group_key,
            val_ratio=val_ratio,
            seed=candidate_seed,
        )
        candidate_score = score_train_val_manifest(candidate_manifest)
        if candidate_score < best_score:
            best_df = candidate_df
            best_manifest = candidate_manifest
            best_seed = candidate_seed
            best_score = candidate_score

    if best_df is None or best_manifest is None:
        raise RuntimeError("未能选出可用的 train/val split。")
    return best_df, best_manifest, best_seed, best_score


def assign_global_sequence_ids(
    split_df: pd.DataFrame,
    test_frames: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, int]]:
    combined_parts = [split_df[["gene", "noncodingRNA"]].copy()]
    split_len = len(split_df)
    test_names = list(test_frames.keys())
    test_lengths = [len(test_frames[name]) for name in test_names]
    for name in test_names:
        combined_parts.append(test_frames[name][["gene", "noncodingRNA"]].copy())

    combined = pd.concat(combined_parts, ignore_index=True)
    gene_seq_ids, unique_genes = pd.factorize(combined["gene"], sort=False)
    mirna_seq_ids, unique_mirnas = pd.factorize(combined["noncodingRNA"], sort=False)
    combined["gene_seq_id"] = gene_seq_ids.astype(np.int32, copy=False)
    combined["mirna_seq_id"] = mirna_seq_ids.astype(np.int32, copy=False)

    split_df = split_df.copy()
    split_df["gene_seq_id"] = combined.iloc[:split_len]["gene_seq_id"].to_numpy(copy=False)
    split_df["mirna_seq_id"] = combined.iloc[:split_len]["mirna_seq_id"].to_numpy(copy=False)
    if "row_id" not in split_df.columns:
        split_df["row_id"] = np.arange(len(split_df), dtype=np.int64)

    updated_tests: Dict[str, pd.DataFrame] = {}
    offset = split_len
    for name, length in zip(test_names, test_lengths):
        test_df = test_frames[name].copy()
        slice_df = combined.iloc[offset : offset + length]
        test_df["gene_seq_id"] = slice_df["gene_seq_id"].to_numpy(copy=False)
        test_df["mirna_seq_id"] = slice_df["mirna_seq_id"].to_numpy(copy=False)
        if "row_id" not in test_df.columns:
            test_df["row_id"] = np.arange(length, dtype=np.int64)
        updated_tests[name] = test_df
        offset += length

    sequence_summary = {
        "unique_gene_count": int(len(unique_genes)),
        "unique_mirna_count": int(len(unique_mirnas)),
    }
    return split_df, updated_tests, sequence_summary


def subsample_train_split(
    train_df: pd.DataFrame,
    positive_count: int,
    negative_count: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if positive_count < 0 or negative_count < 0:
        raise ValueError("subsample_train_positive_count / subsample_train_negative_count 不能为负数。")

    if positive_count == 0 and negative_count == 0:
        return train_df.reset_index(drop=True), {
            "enabled": False,
            "original_row_count": int(len(train_df)),
            "sampled_row_count": int(len(train_df)),
            "positive_count": int(train_df["label"].sum()),
            "negative_count": int(len(train_df) - train_df["label"].sum()),
        }

    positive_df = train_df[train_df["label"] == 1].copy()
    negative_df = train_df[train_df["label"] == 0].copy()

    requested_pos = positive_count or len(positive_df)
    requested_neg = negative_count or len(negative_df)
    if requested_pos > len(positive_df):
        raise ValueError(
            f"请求采样 {requested_pos} 个正样本，但训练集仅有 {len(positive_df)} 个正样本。"
        )
    if requested_neg > len(negative_df):
        raise ValueError(
            f"请求采样 {requested_neg} 个负样本，但训练集仅有 {len(negative_df)} 个负样本。"
        )

    sampled_pos = positive_df.sample(n=requested_pos, random_state=seed)
    sampled_neg = negative_df.sample(n=requested_neg, random_state=seed + 1)
    sampled_df = pd.concat([sampled_pos, sampled_neg], ignore_index=True)
    sampled_df = sampled_df.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    sampled_df["row_id"] = np.arange(len(sampled_df), dtype=np.int64)
    return sampled_df, {
        "enabled": True,
        "seed": seed,
        "original_row_count": int(len(train_df)),
        "sampled_row_count": int(len(sampled_df)),
        "requested_positive_count": int(requested_pos),
        "requested_negative_count": int(requested_neg),
        "positive_count": int(sampled_df["label"].sum()),
        "negative_count": int(len(sampled_df) - sampled_df["label"].sum()),
    }


def build_family_weighted_sampler(
    train_df: pd.DataFrame,
    group_key: str,
    seed: int,
):
    family_counts = train_df[group_key].astype(str).value_counts()
    weights = train_df[group_key].astype(str).map(
        lambda value: 1.0 / math.sqrt(float(family_counts[str(value)]))
    ).to_numpy(dtype=np.float64, copy=False)
    generator = sp.torch.Generator()
    generator.manual_seed(seed)
    sampler = sp.torch.utils.data.WeightedRandomSampler(
        weights=sp.torch.as_tensor(weights, dtype=sp.torch.double),
        num_samples=len(train_df),
        replacement=True,
        generator=generator,
    )
    summary = {
        "enabled": True,
        "group_key": group_key,
        "weight_formula": "1/sqrt(family_count)",
        "row_count": int(len(train_df)),
        "family_count": int(train_df[group_key].nunique()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
    }
    return sampler, summary


def build_single_loader(
    df: pd.DataFrame,
    group_key: str,
    batch_size: int,
    collator,
    device,
    num_workers: int,
    prefetch_factor: int,
    include_sequences: bool,
    shuffle: bool,
    enable_cuda_prefetch: bool,
    sampler=None,
    generator=None,
):
    if "row_id" not in df.columns:
        df = df.copy()
        df["row_id"] = np.arange(len(df), dtype=np.int64)

    pin_memory = device.type == "cuda"
    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "collate_fn": collator,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }
    if sampler is None:
        loader_kwargs["shuffle"] = shuffle
        if generator is not None:
            loader_kwargs["generator"] = generator
    else:
        loader_kwargs["sampler"] = sampler
    if pin_memory and "pin_memory_device" in sp.inspect.signature(sp.DataLoader).parameters:
        loader_kwargs["pin_memory_device"] = "cuda"
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor

    indices = np.arange(len(df), dtype=np.int64)
    loader = sp.DataLoader(
        sp.PairDataset(
            df,
            indices,
            group_key=group_key,
            include_sequences=include_sequences,
        ),
        **loader_kwargs,
    )
    if enable_cuda_prefetch and device.type == "cuda":
        loader = sp.CUDAPrefetchLoader(loader, device=device)
    return loader


def compute_probabilistic_metrics(labels: List[int], probabilities: List[float]) -> Dict[str, float]:
    raw_metrics = sp.compute_binary_metrics(labels, probabilities)
    return {
        "row_count": raw_metrics["row_count"],
        "positive_rate": raw_metrics["positive_rate"],
        "auroc": raw_metrics["auroc"],
        "auprc": raw_metrics["auprc"],
    }


def summarize_thresholded_predictions(
    prediction_df: pd.DataFrame,
    threshold: float,
    loss: float,
) -> Dict[str, float]:
    labels = prediction_df["label"].astype(int).tolist()
    probabilities = prediction_df["probability"].astype(float).tolist()
    threshold_metrics = rge.compute_threshold_metrics(labels, probabilities, threshold)
    raw_metrics = compute_probabilistic_metrics(labels, probabilities)
    return {
        **threshold_metrics,
        "auroc": raw_metrics["auroc"],
        "auprc": raw_metrics["auprc"],
        "loss": float(loss),
    }


def build_model(
    config: Route1Config,
    UtrBertModel,
    RnaFmModel,
    gene_model_source: str,
    mirna_model_source: str,
):
    handcrafted_feature_dim = len(mhf.get_feature_names(config.handcrafted_feature_set))
    if config.architecture == "hierarchical_cnn":
        return sp.HierarchicalCrossAttentionCnnInteractionModel(
            UtrBertModel=UtrBertModel,
            RnaFmModel=RnaFmModel,
            gene_model_name=gene_model_source,
            mirna_model_name=mirna_model_source,
            interaction_hidden_dim=config.interaction_hidden_dim,
            classifier_hidden_dim=config.classifier_hidden_dim,
            attention_heads=config.attention_heads,
            interaction_top_k=config.interaction_top_k,
            dropout=config.dropout,
            local_attention_radius=config.local_attention_radius,
            interaction_cnn_channels=config.interaction_cnn_channels,
            handcrafted_feature_dim=handcrafted_feature_dim,
        )
    if config.architecture != "cross_attention":
        raise ValueError(f"未知 Route1 architecture: {config.architecture!r}")
    return sp.CrossAttentionInteractionModel(
        UtrBertModel=UtrBertModel,
        RnaFmModel=RnaFmModel,
        gene_model_name=gene_model_source,
        mirna_model_name=mirna_model_source,
        interaction_hidden_dim=config.interaction_hidden_dim,
        classifier_hidden_dim=config.classifier_hidden_dim,
        attention_heads=config.attention_heads,
        interaction_top_k=config.interaction_top_k,
        dropout=config.dropout,
        interaction_scorer=config.interaction_scorer,
        feature_processor=config.feature_processor,
        handcrafted_feature_dim=handcrafted_feature_dim,
    )


def write_feature_reference_manifest(output_dir: Path, feature_set: str) -> Dict[str, Any]:
    manifest = mhf.build_feature_reference_manifest(feature_set)
    sp.save_json(output_dir / "feature_reference_manifest.json", manifest)
    return manifest


def train_route1_model(
    model,
    train_loader,
    val_loader,
    config: Route1Config,
    device,
    group_key: str,
    threshold_grid: List[float],
    output_dir: Path,
    journal: ExperimentJournal,
    checkpoint_model_name: str = "cross_attention_interaction",
) -> Tuple[Any, Dict[str, Any], float]:
    model = model.to(device)
    optimizer = sp.build_adamw_optimizer(
        model=model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=device,
        use_fused=config.fused_optimizer,
    )
    criterion = sp.nn.BCEWithLogitsLoss()
    amp_enabled = device.type == "cuda"
    amp_dtype = sp.resolve_amp_dtype(config.amp_dtype, device)
    scaler = sp.GradScaler(enabled=amp_enabled and amp_dtype == sp.torch.float16)

    best_state = sp.extract_trainable_state_dict(model)
    best_epoch = 0
    best_score = float("-inf")
    best_val_raw_metrics: Optional[Dict[str, float]] = None
    best_val_selected_metrics: Optional[Dict[str, float]] = None
    history_rows: List[Dict[str, Any]] = []
    patience_used = 0

    journal.log(
        "train_start",
        max_epochs=config.max_epochs,
        early_stopping_patience=config.early_stopping_patience,
        model_selection_metric=config.model_selection_metric,
        threshold_metric=config.threshold_metric,
    )

    for epoch in range(1, config.max_epochs + 1):
        train_metrics, _ = sp.run_epoch(
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
            progress_desc=f"route1 train e{epoch:02d}",
            show_progress=config.progress_bar,
        )
        val_raw_metrics, val_prediction_df = sp.run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            train=False,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            capture_predictions=True,
            group_key=group_key,
            progress_desc=f"route1 val e{epoch:02d}",
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

        history_row = {"epoch": epoch, "val_selected_threshold": val_threshold}
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
        )

        monitor_score = val_raw_metrics[config.model_selection_metric]
        if math.isnan(monitor_score):
            monitor_score = float("-inf")

        print(
            f"[route1] Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_auroc={train_metrics['auroc']:.4f} train_auprc={train_metrics['auprc']:.4f} | "
            f"val_loss={val_raw_metrics['loss']:.4f} "
            f"val_auroc={val_raw_metrics['auroc']:.4f} val_auprc={val_raw_metrics['auprc']:.4f} "
            f"| select_by_{config.model_selection_metric}={monitor_score:.4f}"
        )

        if monitor_score > best_score:
            best_score = monitor_score
            best_epoch = epoch
            best_state = sp.extract_trainable_state_dict(model)
            best_val_raw_metrics = val_raw_metrics
            best_val_selected_metrics = val_selected_metrics
            patience_used = 0
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

    sp.load_trainable_state_dict(model, best_state)
    val_raw_metrics, val_prediction_df = sp.run_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        train=False,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        capture_predictions=True,
        group_key=group_key,
        progress_desc="route1 val best",
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
    val_threshold_curve.to_csv(output_dir / "val_threshold_search.tsv", sep="\t", index=False)

    checkpoint_payload = {
        "model_name": checkpoint_model_name,
        "config": asdict(config),
        "best_epoch": best_epoch,
        "model_selection_metric": config.model_selection_metric,
        "best_val_raw_metrics": val_raw_metrics,
        "best_val_selected_metrics": val_selected_metrics,
        "selected_threshold": selected_threshold,
        "state_dict_kind": "trainable_only",
        "state_dict": sp.extract_trainable_state_dict(model),
    }
    sp.safe_torch_save(checkpoint_payload, output_dir / "best_checkpoint.pt")

    journal.log(
        "best_model_selected",
        best_epoch=best_epoch,
        best_val_auroc=val_raw_metrics["auroc"],
        best_val_auprc=val_raw_metrics["auprc"],
        selected_threshold=selected_threshold,
        val_selected_metric=val_selected_metrics[config.threshold_metric],
    )

    train_summary = {
        "best_epoch": best_epoch,
        "model_selection_metric": config.model_selection_metric,
        "best_val_raw_metrics": val_raw_metrics,
        "best_val_selected_metrics": val_selected_metrics,
        "selected_threshold": selected_threshold,
    }
    return model, train_summary, selected_threshold


def evaluate_test_file(
    model,
    test_file_name: str,
    test_df: pd.DataFrame,
    selected_threshold: float,
    collator,
    config: Route1Config,
    device,
    output_root: Path,
    journal: ExperimentJournal,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    criterion = sp.nn.BCEWithLogitsLoss()
    amp_enabled = device.type == "cuda"
    amp_dtype = sp.resolve_amp_dtype(config.amp_dtype, device)

    test_loader = build_single_loader(
        df=test_df,
        group_key=config.group_key,
        batch_size=sp.resolve_eval_batch_size(config.batch_size, config.eval_batch_size),
        collator=collator,
        device=device,
        num_workers=sp.resolve_num_workers(config.num_workers, device),
        prefetch_factor=config.prefetch_factor,
        include_sequences=True,
        shuffle=False,
        enable_cuda_prefetch=config.cuda_prefetch,
    )

    journal.log("test_eval_start", test_file=test_file_name, row_count=int(len(test_df)))
    raw_metrics, prediction_df = sp.run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        train=False,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        capture_predictions=True,
        group_key=config.group_key,
        progress_desc=f"test {Path(test_file_name).stem}",
        show_progress=config.progress_bar,
    )
    if prediction_df is None:
        raise RuntimeError(f"{test_file_name} 测试阶段未返回预测结果。")

    prediction_df = rge.apply_threshold_to_predictions(prediction_df, selected_threshold)
    prediction_df["source_dataset"] = test_file_name
    prediction_df["test_dataset"] = Path(test_file_name).stem

    threshold_metrics = summarize_thresholded_predictions(
        prediction_df=prediction_df,
        threshold=selected_threshold,
        loss=raw_metrics["loss"],
    )
    test_dir = output_root / "test" / Path(test_file_name).stem
    test_dir.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(test_dir / "predictions.tsv", sep="\t", index=False)
    group_analysis_df = sp.compute_group_error_analysis(prediction_df, group_key=config.group_key)
    group_analysis_df.to_csv(
        test_dir / f"{config.group_key}_analysis.tsv",
        sep="\t",
        index=False,
    )

    metrics_payload = {
        "test_file": test_file_name,
        "selected_threshold": selected_threshold,
        "test_metrics": threshold_metrics,
    }
    sp.save_json(test_dir / "metrics.json", metrics_payload)
    pd.DataFrame(
        [
            {
                "test_file": test_file_name,
                "selected_threshold": selected_threshold,
                **threshold_metrics,
            }
        ]
    ).to_csv(test_dir / "metrics.csv", index=False)

    journal.log(
        "test_eval_end",
        test_file=test_file_name,
        row_count=threshold_metrics["row_count"],
        auroc=threshold_metrics["auroc"],
        auprc=threshold_metrics["auprc"],
        balanced_accuracy=threshold_metrics["balanced_accuracy"],
    )
    return threshold_metrics, prediction_df


def format_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(format_value(row[column]) for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def build_final_report(
    output_dir: Path,
    config: Route1Config,
    prep_summary: Dict[str, Any],
    merged_train_summary: Dict[str, Any],
    leakage_report: Dict[str, Any],
    validation_manifest: Dict[str, Any],
    train_summary: Dict[str, Any],
    aggregate_df: pd.DataFrame,
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
        "# Route 1 Finetune Report",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Summary",
        "",
        f"- Config preset: `{config.config_preset}`",
        f"- Feature filter: `{config.feature_value}`",
        f"- Train files: {len(prep_summary['train_files'])}",
        f"- Test files: {len(prep_summary['test_files'])}",
        f"- Best epoch: {train_summary['best_epoch']}",
        f"- Model selection metric: `{train_summary['model_selection_metric']}`",
        f"- Selected threshold ({config.threshold_metric}): {train_summary['selected_threshold']:.4f}",
        "",
        "## Data Preparation",
        "",
        "### Train Files",
        "",
        dataframe_to_markdown(pd.DataFrame(train_rows)),
        "",
        "### Test Files",
        "",
        dataframe_to_markdown(pd.DataFrame(test_rows)),
        "",
        "### Training Pool",
        "",
        f"- Before overlap exclusion: {merged_train_summary['before_overlap_exclusion']['retained_rows']} pairs",
        f"- Removed exact train/test overlap pairs: {merged_train_summary['removed_exact_pair_overlap_count']}",
        f"- Final retained train pairs: {merged_train_summary['after_overlap_exclusion']['retained_rows']}",
        "",
        "## Leakage Check",
        "",
        f"- Union exact pair overlap before exclusion: {leakage_report['before_exclusion']['union_exact_pair_overlap_count']}",
        f"- Union exact pair overlap after exclusion: {leakage_report['after_exclusion']['union_exact_pair_overlap_count']}",
        f"- Family overlap is reported only and not removed.",
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
        dataframe_to_markdown(aggregate_display),
        "",
    ]
    (output_dir / "final_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def run_route1_finetune(config: Route1Config) -> Path:
    config = apply_config_preset(config)
    sp.set_seed(config.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = config.run_name.strip() or timestamp
    run_dir = Path(config.output_dir) / f"route1_finetune_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    journal = ExperimentJournal(run_dir / "experiment_journal.jsonl")
    journal.log("route1_start", config=asdict(config))

    sp.ensure_compatible_environment()
    RnaTokenizer, UtrBertModel, RnaFmModel = sp.check_and_import_multimolecule()
    device = sp.resolve_device(config.device)
    sp.configure_runtime_for_device(device)
    effective_num_workers = sp.resolve_num_workers(config.num_workers, device)
    effective_eval_batch_size = sp.resolve_eval_batch_size(config.batch_size, config.eval_batch_size)
    gene_model_source = sp.resolve_model_source(config.gene_model_name, "utrbert-3mer")
    mirna_model_source = sp.resolve_model_source(config.mirna_model_name, "rnafm")
    resolved_amp_dtype = sp.resolve_amp_dtype(config.amp_dtype, device)

    run_config_payload = {
        **asdict(config),
        "resolved_device": str(device),
        "resolved_num_workers": effective_num_workers,
        "resolved_eval_batch_size": effective_eval_batch_size,
        "resolved_amp_dtype": str(resolved_amp_dtype) if resolved_amp_dtype is not None else None,
        "resolved_gene_model_name": gene_model_source,
        "resolved_mirna_model_name": mirna_model_source,
        "handcrafted_feature_columns": mhf.get_prefixed_feature_columns(config.handcrafted_feature_set),
    }
    sp.save_json(run_dir / "run_config.json", run_config_payload)
    feature_reference_manifest = write_feature_reference_manifest(
        output_dir=run_dir,
        feature_set=config.handcrafted_feature_set,
    )
    journal.log("feature_reference_manifest_written", manifest=feature_reference_manifest)

    train_files = select_named_tsv_files(config.train_dir, config.train_files)
    test_files = select_named_tsv_files(config.test_dir, config.test_files)

    train_frames: List[pd.DataFrame] = []
    train_summaries: Dict[str, Any] = {}
    for path in train_files:
        clean_df, summary = load_clean_pair_dataframe(
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
        clean_df, summary = load_clean_pair_dataframe(
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

    prep_summary = {
        "train_files": train_summaries,
        "test_files": test_summaries,
    }
    sp.save_json(run_dir / "prep_summary.json", prep_summary)

    merged_train_df, merged_train_base_summary = merge_training_dataframes(
        train_frames=train_frames,
        group_key=config.group_key,
    )
    journal.log("merged_train_created", summary=merged_train_base_summary)

    leakage_before = build_leakage_report(
        train_df=merged_train_df,
        test_frames=test_frames,
        group_key=config.group_key,
    )
    if config.exclude_test_overlap:
        merged_train_df, removed_overlap_count = exclude_test_overlap_rows(
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
    leakage_after = build_leakage_report(
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

    split_df, split_manifest, selected_split_seed, split_score = select_best_train_val_split(
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

    split_df, test_frames, sequence_summary = assign_global_sequence_ids(split_df, test_frames)
    journal.log("sequence_ids_assigned", summary=sequence_summary)

    handcrafted_summary: Dict[str, Any] = {
        "feature_set": config.handcrafted_feature_set,
        "enabled": config.handcrafted_feature_set != "none",
        "split_df": {"row_count": int(len(split_df)), "columns": []},
        "test_frames": {},
    }
    if config.handcrafted_feature_set != "none":
        split_df = mhf.augment_dataframe_with_features(
            split_df,
            feature_set=config.handcrafted_feature_set,
        )
        split_feature_summary = mhf.summarize_feature_columns(
            split_df,
            feature_set=config.handcrafted_feature_set,
        )
        augmented_test_frames: Dict[str, pd.DataFrame] = {}
        for test_name, test_df in test_frames.items():
            augmented_test_df = mhf.augment_dataframe_with_features(
                test_df,
                feature_set=config.handcrafted_feature_set,
            )
            augmented_test_frames[test_name] = augmented_test_df
            handcrafted_summary["test_frames"][test_name] = mhf.summarize_feature_columns(
                augmented_test_df,
                feature_set=config.handcrafted_feature_set,
            )
        test_frames = augmented_test_frames
        handcrafted_summary["split_df"] = split_feature_summary
    sp.save_json(run_dir / "handcrafted_feature_summary.json", handcrafted_summary)
    journal.log("handcrafted_features_prepared", summary=handcrafted_summary)

    cache_source_df = pd.concat(
        [split_df[["gene", "noncodingRNA"]]]
        + [frame[["gene", "noncodingRNA"]] for frame in test_frames.values()],
        ignore_index=True,
    )
    gene_tokenizer = RnaTokenizer.from_pretrained(gene_model_source)
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
    subsample_seed = config.subsample_seed or config.seed
    train_df, train_subsample_summary = subsample_train_split(
        train_df=train_df,
        positive_count=config.subsample_train_positive_count,
        negative_count=config.subsample_train_negative_count,
        seed=subsample_seed,
    )
    sp.save_json(run_dir / "train_subsample_summary.json", train_subsample_summary)
    journal.log("train_subsample_prepared", summary=train_subsample_summary)

    train_sampler = None
    sampler_summary = {"enabled": False}
    if config.family_balanced_sampling:
        train_sampler, sampler_summary = build_family_weighted_sampler(
            train_df=train_df,
            group_key=config.group_key,
            seed=config.seed,
        )
    sp.save_json(run_dir / "train_sampler_summary.json", sampler_summary)
    journal.log("train_sampler_prepared", summary=sampler_summary)

    train_loader = build_single_loader(
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
    val_loader = build_single_loader(
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
    model = build_model(
        config=config,
        UtrBertModel=UtrBertModel,
        RnaFmModel=RnaFmModel,
        gene_model_source=gene_model_source,
        mirna_model_source=mirna_model_source,
    )
    model, train_summary, selected_threshold = train_route1_model(
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

    all_prediction_frames: List[pd.DataFrame] = []
    test_rows: List[Dict[str, Any]] = []
    leftout_row: Optional[Dict[str, Any]] = None
    for test_file_name, test_df in test_frames.items():
        metrics_row, prediction_df = evaluate_test_file(
            model=model,
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
    overall_metrics = summarize_thresholded_predictions(
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

    build_final_report(
        output_dir=run_dir,
        config=config,
        prep_summary=prep_summary,
        merged_train_summary=merged_train_summary,
        leakage_report=leakage_report,
        validation_manifest=validation_manifest,
        train_summary=train_summary,
        aggregate_df=aggregate_df,
    )
    journal.log("final_report_written", path=str(run_dir / "final_report.md"))
    print(f"[Done] 结果已写入 {run_dir}")
    return run_dir


def main() -> None:
    config = parse_args()
    run_route1_finetune(config)


if __name__ == "__main__":
    main()
