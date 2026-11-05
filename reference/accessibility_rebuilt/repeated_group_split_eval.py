from __future__ import annotations

import argparse
import copy
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)

import supervised_poc as sp


@dataclass
class EvalConfig:
    input_file: str
    output_dir: str
    num_runs: int = 5
    split_candidates_per_run: int = 5
    split_seed_start: int = 42
    train_seed_start: int = 42
    models: str = "baseline_pooled,cross_attention_interaction"
    threshold_metric: str = "balanced_accuracy"
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_step: float = 0.01
    save_checkpoints: bool = False
    group_key: str = "noncodingRNA_fam"
    feature_value: str = "three_prime_utr"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 8
    eval_batch_size: int = 0
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
    chunksize: int = 200000
    num_workers: int = 0
    prefetch_factor: int = 2
    cache_tokenized_sequences: bool = True
    token_cache_batch_size: int = 4096
    amp_dtype: str = "auto"
    progress_bar: bool = True
    cuda_prefetch: bool = True
    fused_optimizer: bool = True
    smoke_test: bool = False
    smoke_train_rows: int = 128
    smoke_val_rows: int = 64
    smoke_test_rows: int = 64


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(
        description="多次 group split + 验证集选阈值 的监督式评估脚本"
    )
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--split-candidates-per-run", type=int, default=5)
    parser.add_argument("--split-seed-start", type=int, default=42)
    parser.add_argument("--train-seed-start", type=int, default=42)
    parser.add_argument("--models", default="baseline_pooled,cross_attention_interaction")
    parser.add_argument(
        "--threshold-metric",
        default="balanced_accuracy",
        choices=["balanced_accuracy", "accuracy", "f1", "mcc"],
    )
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--group-key", default="noncodingRNA_fam")
    parser.add_argument("--feature-value", default="three_prime_utr")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=0)
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
    parser.add_argument("--chunksize", type=int, default=200000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--disable-token-cache", action="store_true")
    parser.add_argument("--token-cache-batch-size", type=int, default=4096)
    parser.add_argument(
        "--amp-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "fp16", "bf16"],
    )
    parser.add_argument("--disable-progress-bar", action="store_true")
    parser.add_argument("--disable-cuda-prefetch", action="store_true")
    parser.add_argument("--disable-fused-optimizer", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-train-rows", type=int, default=128)
    parser.add_argument("--smoke-val-rows", type=int, default=64)
    parser.add_argument("--smoke-test-rows", type=int, default=64)
    config_kwargs = vars(parser.parse_args())
    config_kwargs["cache_tokenized_sequences"] = not config_kwargs.pop("disable_token_cache")
    config_kwargs["progress_bar"] = not config_kwargs.pop("disable_progress_bar")
    config_kwargs["cuda_prefetch"] = not config_kwargs.pop("disable_cuda_prefetch")
    config_kwargs["fused_optimizer"] = not config_kwargs.pop("disable_fused_optimizer")
    return EvalConfig(**config_kwargs)


def config_to_run_config(config: EvalConfig, seed: int) -> sp.RunConfig:
    return sp.RunConfig(
        input_file=config.input_file,
        output_dir=config.output_dir,
        group_key=config.group_key,
        feature_value=config.feature_value,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        max_epochs=config.max_epochs,
        early_stopping_patience=config.early_stopping_patience,
        seed=seed,
        device=config.device,
        gene_model_name=config.gene_model_name,
        mirna_model_name=config.mirna_model_name,
        max_gene_len=config.max_gene_len,
        max_mirna_len=config.max_mirna_len,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        dropout=config.dropout,
        classifier_hidden_dim=config.classifier_hidden_dim,
        interaction_hidden_dim=config.interaction_hidden_dim,
        attention_heads=config.attention_heads,
        interaction_top_k=config.interaction_top_k,
        chunksize=config.chunksize,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        cache_tokenized_sequences=config.cache_tokenized_sequences,
        token_cache_batch_size=config.token_cache_batch_size,
        amp_dtype=config.amp_dtype,
        progress_bar=config.progress_bar,
        cuda_prefetch=config.cuda_prefetch,
        fused_optimizer=config.fused_optimizer,
        smoke_test=config.smoke_test,
        smoke_train_rows=config.smoke_train_rows,
        smoke_val_rows=config.smoke_val_rows,
        smoke_test_rows=config.smoke_test_rows,
    )


def parse_model_names(models_arg: str) -> List[str]:
    models = [item.strip() for item in models_arg.split(",") if item.strip()]
    allowed = {"baseline_pooled", "cross_attention_interaction"}
    unknown = sorted(set(models) - allowed)
    if unknown:
        raise ValueError(f"未知模型名: {unknown}")
    if not models:
        raise ValueError("至少需要一个模型名。")
    return models


def build_threshold_grid(config: EvalConfig) -> List[float]:
    if config.threshold_step <= 0:
        raise ValueError("threshold_step 必须大于 0。")
    if not 0 < config.threshold_min < config.threshold_max < 1:
        raise ValueError("threshold_min / threshold_max 必须满足 0 < min < max < 1。")

    values = []
    current = config.threshold_min
    while current <= config.threshold_max + 1e-9:
        values.append(round(current, 6))
        current += config.threshold_step
    return values


def compute_threshold_metrics(
    labels: List[int],
    probabilities: List[float],
    threshold: float,
) -> Dict[str, float]:
    predictions = [1 if value >= threshold else 0 for value in probabilities]
    return {
        "row_count": int(len(labels)),
        "positive_rate": float(sum(labels) / max(len(labels), 1)),
        "predicted_positive_rate": float(sum(predictions) / max(len(predictions), 1)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }


def select_threshold(
    labels: List[int],
    probabilities: List[float],
    threshold_grid: List[float],
    metric_name: str,
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    rows: List[Dict[str, float]] = []
    best_tuple = None
    best_threshold = 0.5
    best_metrics: Dict[str, float] | None = None

    for threshold in threshold_grid:
        metrics = compute_threshold_metrics(labels, probabilities, threshold)
        row = {"threshold": threshold, **metrics}
        rows.append(row)
        rank_tuple = (
            metrics[metric_name],
            metrics["mcc"],
            metrics["f1"],
            metrics["accuracy"],
            -abs(threshold - 0.5),
        )
        if best_tuple is None or rank_tuple > best_tuple:
            best_tuple = rank_tuple
            best_threshold = threshold
            best_metrics = metrics

    if best_metrics is None:
        raise RuntimeError("阈值搜索失败。")

    return best_threshold, best_metrics, pd.DataFrame(rows)


def apply_threshold_to_predictions(
    prediction_df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    out = prediction_df.copy()
    out["pred_label"] = (out["probability"] >= threshold).astype(int)
    out["correct"] = (out["pred_label"] == out["label"]).astype(int)
    return out


def build_model(
    model_name: str,
    config: EvalConfig,
    UtrBertModel,
    RnaFmModel,
    gene_model_source: str,
    mirna_model_source: str,
):
    if model_name == "baseline_pooled":
        return sp.PooledBaselineModel(
            UtrBertModel=UtrBertModel,
            RnaFmModel=RnaFmModel,
            gene_model_name=gene_model_source,
            mirna_model_name=mirna_model_source,
            classifier_hidden_dim=config.classifier_hidden_dim,
            dropout=config.dropout,
        )
    if model_name == "cross_attention_interaction":
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
        )
    raise ValueError(f"不支持的模型: {model_name}")


def randomized_greedy_group_split(
    df: pd.DataFrame,
    group_key: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    sp.validate_split_ratios(train_ratio, val_ratio, test_ratio)

    group_stats = (
        df.groupby(group_key)
        .agg(sample_count=("label", "size"), pos_count=("label", "sum"))
        .reset_index()
    )
    group_stats["neg_count"] = group_stats["sample_count"] - group_stats["pos_count"]
    group_stats["pos_rate"] = group_stats["pos_count"] / group_stats["sample_count"]
    mixed_groups = group_stats[(group_stats["pos_count"] > 0) & (group_stats["neg_count"] > 0)].copy()
    if len(mixed_groups) < 3:
        raise ValueError("可用于 split 初始化的混合标签 group 不足。")

    split_order = ["train", "val", "test"]
    ratio_map = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    total_count = float(group_stats["sample_count"].sum())
    total_pos = float(group_stats["pos_count"].sum())
    global_pos_rate = total_pos / max(total_count, 1.0)
    rng = random.Random(seed)

    top_train_pool_size = min(len(mixed_groups), max(5, len(mixed_groups) // 6))
    top_train_pool = mixed_groups.nlargest(top_train_pool_size, "sample_count")
    train_seed = top_train_pool.sample(n=1, random_state=seed).iloc[0].to_dict()

    remaining_mixed = mixed_groups.loc[mixed_groups[group_key] != train_seed[group_key]].copy()
    small_pool_size = min(len(remaining_mixed), max(6, len(remaining_mixed) // 3))
    small_pool = remaining_mixed.nsmallest(small_pool_size, "sample_count")
    sampled_small = small_pool.sample(n=2, random_state=seed + 1, replace=False)
    val_seed = sampled_small.iloc[0].to_dict()
    test_seed = sampled_small.iloc[1].to_dict()

    split_state = {
        split: {"rows": [], "sample_count": 0, "pos_count": 0, "neg_count": 0}
        for split in split_order
    }
    seeded_groups = set()
    for split_name, group_row in zip(split_order, [train_seed, val_seed, test_seed]):
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

    manifest_groups = []
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
            row_copy = {
                group_key: row[group_key],
                "split": split_name,
                "sample_count": int(row["sample_count"]),
                "pos_count": int(row["pos_count"]),
                "neg_count": int(row["neg_count"]),
                "pos_rate": float(row["pos_rate"]),
            }
            manifest_groups.append(row_copy)
            group_to_split[str(row[group_key])] = split_name

    out_df = df.copy()
    out_df["split"] = out_df[group_key].map(lambda value: group_to_split[str(value)])
    for split_name in split_order:
        subset = out_df[out_df["split"] == split_name]
        if subset.empty or subset["label"].nunique() < 2:
            raise ValueError(f"{split_name} split 无法用于训练或评估。")

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


def score_split_manifest(manifest: Dict[str, Any]) -> float:
    global_positive_rate = manifest["global_summary"]["positive_rate"]
    score = 0.0
    for split_name, summary in manifest["split_summary"].items():
        score += abs(summary["actual_row_ratio"] - summary["target_row_ratio"])
        score += 0.5 * abs(summary["positive_rate"] - global_positive_rate)
    return float(score)


def select_best_split_candidate(
    df: pd.DataFrame,
    group_key: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    base_seed: int,
    candidates: int,
) -> Tuple[pd.DataFrame, Dict[str, Any], int, float]:
    if candidates <= 0:
        raise ValueError("split_candidates_per_run 必须大于 0。")

    best_df: pd.DataFrame | None = None
    best_manifest: Dict[str, Any] | None = None
    best_seed = base_seed
    best_score = float("inf")

    for candidate_index in range(candidates):
        candidate_seed = base_seed * 1000 + candidate_index
        candidate_df, candidate_manifest = randomized_greedy_group_split(
            df=df,
            group_key=group_key,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=candidate_seed,
        )
        candidate_score = score_split_manifest(candidate_manifest)
        if candidate_score < best_score:
            best_df = candidate_df
            best_manifest = candidate_manifest
            best_seed = candidate_seed
            best_score = candidate_score

    if best_df is None or best_manifest is None:
        raise RuntimeError("未能选出可用的 split 候选。")

    return best_df, best_manifest, best_seed, best_score


def train_and_evaluate_model(
    model_name: str,
    model,
    loaders: Dict[str, sp.DataLoader],
    config: EvalConfig,
    run_seed: int,
    device,
    model_dir: Path,
    group_key: str,
    threshold_grid: List[float],
) -> Dict[str, Any]:
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
    best_monitor_score = float("-inf")
    best_val_threshold_metrics: Dict[str, float] | None = None
    history_rows: List[Dict[str, Any]] = []
    patience_used = 0
    max_epochs = 1 if config.smoke_test else config.max_epochs
    early_stopping_patience = 1 if config.smoke_test else max(5, config.early_stopping_patience)

    sp.set_seed(run_seed)
    for epoch in range(1, max_epochs + 1):
        train_metrics, _ = sp.run_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            device=device,
            train=True,
            optimizer=optimizer,
            scaler=scaler if amp_enabled else None,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            group_key=group_key,
            progress_desc=f"{model_name} train e{epoch:02d}",
            show_progress=config.progress_bar,
        )
        val_metrics_raw, val_prediction_df = sp.run_epoch(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
            train=False,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            capture_predictions=True,
            group_key=group_key,
            progress_desc=f"{model_name} val e{epoch:02d}",
            show_progress=config.progress_bar,
        )
        if val_prediction_df is None:
            raise RuntimeError("验证集预测为空。")

        val_labels = val_prediction_df["label"].astype(int).tolist()
        val_probs = val_prediction_df["probability"].astype(float).tolist()
        val_threshold, val_threshold_metrics, val_curve = select_threshold(
            labels=val_labels,
            probabilities=val_probs,
            threshold_grid=threshold_grid,
            metric_name=config.threshold_metric,
        )

        history_row = {"epoch": epoch, "val_selected_threshold": val_threshold}
        history_row.update({f"train_{key}": value for key, value in train_metrics.items()})
        history_row.update({f"val_raw_{key}": value for key, value in val_metrics_raw.items()})
        history_row.update({f"val_selected_{key}": value for key, value in val_threshold_metrics.items()})
        history_rows.append(history_row)

        monitor_score = val_threshold_metrics[config.threshold_metric]
        print(
            f"[{model_name}] Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_{config.threshold_metric}={monitor_score:.4f} "
            f"val_thr={val_threshold:.2f} val_auroc={val_metrics_raw['auroc']:.4f}"
        )

        if monitor_score > best_monitor_score:
            best_monitor_score = monitor_score
            best_epoch = epoch
            best_state = sp.extract_trainable_state_dict(model)
            best_val_threshold_metrics = val_threshold_metrics
            patience_used = 0
        else:
            patience_used += 1
            if patience_used >= early_stopping_patience:
                print(
                    f"[{model_name}] Early stopping at epoch {epoch:02d} "
                    f"(best epoch = {best_epoch:02d})."
                )
                break

    if best_val_threshold_metrics is None:
        raise RuntimeError("未能选出最佳验证阈值。")

    sp.load_trainable_state_dict(model, best_state)
    val_metrics_raw, val_prediction_df = sp.run_epoch(
        model=model,
        loader=loaders["val"],
        criterion=criterion,
        device=device,
        train=False,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        capture_predictions=True,
        group_key=group_key,
        progress_desc=f"{model_name} val best",
        show_progress=config.progress_bar,
    )
    test_metrics_raw, test_prediction_df = sp.run_epoch(
        model=model,
        loader=loaders["test"],
        criterion=criterion,
        device=device,
        train=False,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        capture_predictions=True,
        group_key=group_key,
        progress_desc=f"{model_name} test",
        show_progress=config.progress_bar,
    )
    if val_prediction_df is None or test_prediction_df is None:
        raise RuntimeError("验证集或测试集预测为空。")

    best_val_threshold, best_val_threshold_metrics, final_val_curve = select_threshold(
        labels=val_prediction_df["label"].astype(int).tolist(),
        probabilities=val_prediction_df["probability"].astype(float).tolist(),
        threshold_grid=threshold_grid,
        metric_name=config.threshold_metric,
    )
    final_test_prediction_df = apply_threshold_to_predictions(test_prediction_df, best_val_threshold)
    test_threshold_metrics = compute_threshold_metrics(
        labels=final_test_prediction_df["label"].astype(int).tolist(),
        probabilities=final_test_prediction_df["probability"].astype(float).tolist(),
        threshold=best_val_threshold,
    )
    test_threshold_metrics["loss"] = test_metrics_raw["loss"]
    test_threshold_metrics["auroc"] = test_metrics_raw["auroc"]
    test_threshold_metrics["auprc"] = test_metrics_raw["auprc"]

    model_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history_rows).to_csv(model_dir / "history.tsv", sep="\t", index=False)
    final_val_curve.to_csv(model_dir / "val_threshold_search.tsv", sep="\t", index=False)
    final_test_prediction_df.to_csv(model_dir / "test_predictions.tsv", sep="\t", index=False)
    sp.compute_group_error_analysis(
        final_test_prediction_df,
        group_key=group_key,
    ).to_csv(model_dir / f"{group_key}_analysis.tsv", sep="\t", index=False)

    metrics_payload = {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "threshold_metric": config.threshold_metric,
        "selected_threshold": best_val_threshold,
        "val_raw_metrics": val_metrics_raw,
        "val_selected_threshold_metrics": best_val_threshold_metrics,
        "test_metrics": test_threshold_metrics,
    }
    sp.save_json(model_dir / "metrics.json", metrics_payload)
    pd.DataFrame(
        [
            {
                "model_name": model_name,
                "best_epoch": best_epoch,
                "threshold_metric": config.threshold_metric,
                "selected_threshold": best_val_threshold,
                **{f"test_{key}": value for key, value in test_threshold_metrics.items()},
            }
        ]
    ).to_csv(model_dir / "metrics.csv", index=False)

    if config.save_checkpoints:
        sp.safe_torch_save(
            {
                "model_name": model_name,
                "best_epoch": best_epoch,
                "config": asdict(config),
                "state_dict_kind": "trainable_only",
                "state_dict": sp.extract_trainable_state_dict(model),
                "selected_threshold": best_val_threshold,
            },
            model_dir / "best_checkpoint.pt",
        )

    return {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "selected_threshold": best_val_threshold,
        "threshold_metric": config.threshold_metric,
        **{f"val_selected_{key}": value for key, value in best_val_threshold_metrics.items()},
        **{f"test_{key}": value for key, value in test_threshold_metrics.items()},
    }


def aggregate_results(results_df: pd.DataFrame, metric_columns: List[str]) -> pd.DataFrame:
    rows = []
    for model_name, group in results_df.groupby("model_name"):
        row: Dict[str, Any] = {
            "model_name": model_name,
            "num_runs": int(len(group)),
        }
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
            row[f"{column}_min"] = float(group[column].min())
            row[f"{column}_max"] = float(group[column].max())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    config = parse_args()
    model_names = parse_model_names(config.models)
    threshold_grid = build_threshold_grid(config)

    sp.ensure_compatible_environment()
    RnaTokenizer, UtrBertModel, RnaFmModel = sp.check_and_import_multimolecule()
    device = sp.resolve_device(config.device)
    sp.configure_runtime_for_device(device)
    effective_num_workers = sp.resolve_num_workers(config.num_workers, device)
    effective_eval_batch_size = sp.resolve_eval_batch_size(
        config.batch_size,
        config.eval_batch_size,
    )
    effective_early_stopping_patience = (
        1 if config.smoke_test else max(5, config.early_stopping_patience)
    )
    gene_model_source = sp.resolve_model_source(config.gene_model_name, "utrbert-3mer")
    mirna_model_source = sp.resolve_model_source(config.mirna_model_name, "rnafm")
    resolved_amp_dtype = sp.resolve_amp_dtype(config.amp_dtype, device)
    print(f"[Info] device = {device}")
    print(f"[Info] num_workers = {effective_num_workers}")
    print(f"[Info] eval_batch_size = {effective_eval_batch_size}")
    print(f"[Info] early_stopping_patience = {effective_early_stopping_patience}")
    print(f"[Info] cuda_prefetch = {config.cuda_prefetch}")
    if resolved_amp_dtype is not None:
        print(f"[Info] amp dtype = {resolved_amp_dtype}")

    base_run_config = config_to_run_config(config, seed=config.seed)
    prepared = sp.load_filtered_dataframe(base_run_config)
    gene_tokenizer = RnaTokenizer.from_pretrained(gene_model_source)
    mirna_tokenizer = RnaTokenizer.from_pretrained(mirna_model_source)
    shared_gene_cache = None
    shared_mirna_cache = None
    if not config.smoke_test:
        shared_gene_cache, shared_mirna_cache = sp.build_dual_token_caches(
            df=prepared.dataframe,
            gene_tokenizer=gene_tokenizer,
            mirna_tokenizer=mirna_tokenizer,
            max_gene_len=config.max_gene_len,
            max_mirna_len=config.max_mirna_len,
            batch_size=config.token_cache_batch_size,
            enabled=config.cache_tokenized_sequences,
            show_progress=config.progress_bar,
        )

    dataset_name = Path(config.input_file).stem
    root_dir = Path(config.output_dir) / f"{dataset_name}_repeated_eval"
    root_dir.mkdir(parents=True, exist_ok=True)
    sp.save_json(root_dir / "eval_config.json", {**asdict(config), "resolved_device": str(device)})
    sp.save_json(root_dir / "data_summary.json", prepared.summary)

    all_run_rows: List[Dict[str, Any]] = []
    run_manifest_rows: List[Dict[str, Any]] = []

    for run_index in range(config.num_runs):
        requested_split_seed = config.split_seed_start + run_index
        train_seed = config.train_seed_start + run_index
        run_name = f"run_{run_index + 1:03d}_split_{requested_split_seed}_train_{train_seed}"
        run_dir = root_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        run_df, split_manifest, selected_split_seed, split_score = select_best_split_candidate(
            df=prepared.dataframe,
            group_key=config.group_key,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            base_seed=requested_split_seed,
            candidates=config.split_candidates_per_run,
        )
        if config.smoke_test:
            run_df, smoke_summary = sp.build_smoke_test_subset(
                df=run_df,
                seed=train_seed,
                split_row_targets={
                    "train": config.smoke_train_rows,
                    "val": config.smoke_val_rows,
                    "test": config.smoke_test_rows,
                },
            )
            sp.save_json(run_dir / "smoke_test_summary.json", smoke_summary)

        run_df.drop(columns=["gene_seq_id", "mirna_seq_id"], errors="ignore").to_csv(
            run_dir / "split_assignments.tsv",
            sep="\t",
            index=False,
        )
        sp.save_json(run_dir / "split_manifest.json", split_manifest)

        run_manifest_rows.append(
            {
                "run_name": run_name,
                "run_index": run_index + 1,
                "requested_split_seed": requested_split_seed,
                "selected_split_seed": selected_split_seed,
                "split_score": split_score,
                "train_seed": train_seed,
                "train_rows": int((run_df["split"] == "train").sum()),
                "val_rows": int((run_df["split"] == "val").sum()),
                "test_rows": int((run_df["split"] == "test").sum()),
                "train_groups": int(run_df.loc[run_df["split"] == "train", config.group_key].nunique()),
                "val_groups": int(run_df.loc[run_df["split"] == "val", config.group_key].nunique()),
                "test_groups": int(run_df.loc[run_df["split"] == "test", config.group_key].nunique()),
            }
        )

        sp.set_seed(train_seed)
        collator = sp.DualTokenizerCollator(
            gene_tokenizer=gene_tokenizer,
            mirna_tokenizer=mirna_tokenizer,
            max_gene_len=config.max_gene_len,
            max_mirna_len=config.max_mirna_len,
            gene_cache=shared_gene_cache,
            mirna_cache=shared_mirna_cache,
        )
        if config.smoke_test:
            run_gene_cache, run_mirna_cache = sp.build_dual_token_caches(
                df=run_df,
                gene_tokenizer=gene_tokenizer,
                mirna_tokenizer=mirna_tokenizer,
                max_gene_len=config.max_gene_len,
                max_mirna_len=config.max_mirna_len,
                batch_size=config.token_cache_batch_size,
                enabled=config.cache_tokenized_sequences,
                show_progress=config.progress_bar,
            )
            collator = sp.DualTokenizerCollator(
                gene_tokenizer=gene_tokenizer,
                mirna_tokenizer=mirna_tokenizer,
                max_gene_len=config.max_gene_len,
                max_mirna_len=config.max_mirna_len,
                gene_cache=run_gene_cache,
                mirna_cache=run_mirna_cache,
            )
        loaders = sp.build_dataloaders(
            df=run_df,
            group_key=config.group_key,
            batch_size=config.batch_size,
            eval_batch_size=effective_eval_batch_size,
            collator=collator,
            device=device,
            num_workers=effective_num_workers,
            prefetch_factor=config.prefetch_factor,
            enable_cuda_prefetch=config.cuda_prefetch,
        )

        for model_name in model_names:
            print(
                f"[Run {run_index + 1:03d}/{config.num_runs:03d}] "
                f"{model_name} | split_seed={selected_split_seed} train_seed={train_seed}"
            )
            model = build_model(
                model_name=model_name,
                config=config,
                UtrBertModel=UtrBertModel,
                RnaFmModel=RnaFmModel,
                gene_model_source=gene_model_source,
                mirna_model_source=mirna_model_source,
            )
            result = train_and_evaluate_model(
                model_name=model_name,
                model=model,
                loaders=loaders,
                config=config,
                run_seed=train_seed,
                device=device,
                model_dir=run_dir / model_name,
                group_key=config.group_key,
                threshold_grid=threshold_grid,
            )
            all_run_rows.append(
                {
                    "run_name": run_name,
                    "run_index": run_index + 1,
                    "requested_split_seed": requested_split_seed,
                    "selected_split_seed": selected_split_seed,
                    "split_score": split_score,
                    "train_seed": train_seed,
                    **result,
                }
            )
            del model
            if device.type == "cuda":
                sp.torch.cuda.empty_cache()

    run_manifest_df = pd.DataFrame(run_manifest_rows)
    run_manifest_df.to_csv(root_dir / "run_manifest.tsv", sep="\t", index=False)

    results_df = pd.DataFrame(all_run_rows)
    results_df.to_csv(root_dir / "per_run_metrics.csv", index=False)
    sp.save_json(
        root_dir / "per_run_metrics.json",
        {"runs": all_run_rows},
    )

    metric_columns = [
        "best_epoch",
        "selected_threshold",
        f"val_selected_{config.threshold_metric}",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1",
        "test_mcc",
        "test_auroc",
        "test_auprc",
        "test_loss",
    ]
    aggregate_df = aggregate_results(results_df, metric_columns=metric_columns)
    aggregate_df.to_csv(root_dir / "aggregate_metrics.csv", index=False)
    sp.save_json(
        root_dir / "aggregate_metrics.json",
        {"models": aggregate_df.to_dict(orient="records")},
    )

    print(f"[Done] 结果已写入 {root_dir}")


if __name__ == "__main__":
    main()
