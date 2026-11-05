"""Verbatim historical cleaning definitions; extracted to avoid unrelated training imports."""
from __future__ import annotations
from typing import Any, Dict, Iterable, Tuple, List
from pathlib import Path
import pandas as pd
import cleaning_support as sp

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
