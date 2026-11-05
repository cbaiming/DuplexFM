from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import supervised_poc as sp


SCORE_FEATURE_NONE = "none"
SCORE_FEATURE_UTRBERT_PLL = "utrbert_pll"
SCORE_COLUMN = "hf_gene_utrbert_pll_nll_norm"
SCORING_MODE = "pll_masked_single_position_length_norm_v1"
NORMALIZATION_VERSION = "normalize_rna_v1"
DEFAULT_CACHE_ROOT = "outputs/utrbert_pll_feature_cache/route2_mirna_three_prime_utr_v1"

REFERENCE_INFO = {
    "feature_name": SCORE_FEATURE_UTRBERT_PLL,
    "score_column": SCORE_COLUMN,
    "summary": (
        "Length-normalized masked pseudo-negative-log-likelihood computed on the gene/mRNA "
        "sequence with UTRBERT-3mer."
    ),
    "note": (
        "This is an MLM-compatible pseudo-log-likelihood feature, not a strict autoregressive "
        "negative log-likelihood."
    ),
    "reference_paper": {
        "title": "Self-Prompting Genome Language Models for RNA Fitness Prediction",
        "url": "https://openreview.net/forum?id=NNMUNQ0mOc",
    },
}


@dataclass(frozen=True)
class CacheBuildResult:
    cache_dir: Path
    score_table_path: Path
    manifest_path: Path
    summary_path: Path
    reused: bool
    manifest: Dict[str, Any]
    summary: Dict[str, Any]


def normalize_sequence_score_feature(value: Optional[str]) -> str:
    if value is None:
        return SCORE_FEATURE_NONE
    normalized = str(value).strip().lower()
    if normalized in {"", SCORE_FEATURE_NONE}:
        return SCORE_FEATURE_NONE
    if normalized != SCORE_FEATURE_UTRBERT_PLL:
        raise ValueError(
            f"未知 gene sequence score feature: {value!r}. "
            f"支持: {SCORE_FEATURE_NONE!r}, {SCORE_FEATURE_UTRBERT_PLL!r}"
        )
    return normalized


def get_feature_names(feature_name: str) -> List[str]:
    normalized = normalize_sequence_score_feature(feature_name)
    if normalized == SCORE_FEATURE_NONE:
        return []
    return [SCORE_COLUMN.removeprefix("hf_")]


def get_feature_columns(feature_name: str) -> List[str]:
    normalized = normalize_sequence_score_feature(feature_name)
    if normalized == SCORE_FEATURE_NONE:
        return []
    return [SCORE_COLUMN]


def build_feature_reference_manifest(feature_name: str) -> Dict[str, Any]:
    normalized = normalize_sequence_score_feature(feature_name)
    if normalized == SCORE_FEATURE_NONE:
        return {
            "feature_name": SCORE_FEATURE_NONE,
            "enabled": False,
            "feature_columns": [],
            "reference": None,
        }
    return {
        "feature_name": normalized,
        "enabled": True,
        "feature_columns": [SCORE_COLUMN],
        "reference": REFERENCE_INFO,
    }


def summarize_feature_columns(df: pd.DataFrame, feature_name: str) -> Dict[str, Any]:
    normalized = normalize_sequence_score_feature(feature_name)
    if normalized == SCORE_FEATURE_NONE:
        return {
            "feature_name": SCORE_FEATURE_NONE,
            "enabled": False,
            "row_count": int(len(df)),
            "columns": [],
        }

    if SCORE_COLUMN not in df.columns:
        raise ValueError(f"DataFrame 缺少 PLL 特征列: {SCORE_COLUMN}")

    values = pd.to_numeric(df[SCORE_COLUMN], errors="coerce")
    return {
        "feature_name": normalized,
        "enabled": True,
        "row_count": int(len(df)),
        "columns": [SCORE_COLUMN],
        "nan_count": int(values.isna().sum()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def default_cache_dir() -> str:
    return DEFAULT_CACHE_ROOT


def build_data_signature(
    train_files: Sequence[str],
    test_files: Sequence[str],
    feature_value: str,
    model_source: str,
) -> Dict[str, Any]:
    return {
        "feature_value": str(feature_value),
        "model_source": str(model_source),
        "scoring_mode": SCORING_MODE,
        "normalization_version": NORMALIZATION_VERSION,
        "train_files": [str(item) for item in train_files],
        "test_files": [str(item) for item in test_files],
    }


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    sp.save_json(path, payload)


def _collect_unique_genes(dataframes: Iterable[pd.DataFrame]) -> List[str]:
    unique_genes = set()
    for df in dataframes:
        genes = df["gene"].map(sp.normalize_rna)
        unique_genes.update(gene for gene in genes.tolist() if gene)
    return sorted(unique_genes)


def collect_unique_genes_from_route2_frames(
    split_df: pd.DataFrame,
    test_frames: Dict[str, pd.DataFrame],
) -> List[str]:
    frames: List[pd.DataFrame] = [split_df]
    frames.extend(test_frames.values())
    return _collect_unique_genes(frames)


def _load_tokenizer_and_model(model_source: str, device):
    if sp.torch is None:
        raise RuntimeError("torch 不可用，无法计算 UTRBERT PLL 特征。")
    sp.ensure_compatible_environment()
    try:
        import multimolecule
        from multimolecule import RnaTokenizer
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "当前环境无法导入 multimolecule，无法实现基于 UTRBERT 的 pseudo-NLL 特征。"
        ) from exc

    model = None
    model_errors: List[str] = []
    for class_name in ("UtrBertForPreTraining", "UtrBertForMaskedLM"):
        model_cls = getattr(multimolecule, class_name, None)
        if model_cls is None:
            continue
        try:
            model = model_cls.from_pretrained(model_source)
            break
        except Exception as exc:  # pragma: no cover - runtime dependency
            model_errors.append(f"{class_name}: {type(exc).__name__}: {exc}")

    if model is None:
        try:
            from transformers import AutoModelForMaskedLM

            model = AutoModelForMaskedLM.from_pretrained(model_source, trust_remote_code=True)
        except Exception as exc:  # pragma: no cover - runtime dependency
            model_errors.append(f"AutoModelForMaskedLM: {type(exc).__name__}: {exc}")
            raise RuntimeError(
                "无法加载支持 MLM logits 的 UTRBERT 模型。尝试过的路径: "
                + " | ".join(model_errors)
            ) from exc

    tokenizer = RnaTokenizer.from_pretrained(model_source)
    model = model.to(device)
    model.eval()
    return tokenizer, model


def _extract_logits(outputs) -> "sp.torch.Tensor":
    if hasattr(outputs, "logits") and outputs.logits is not None:
        return outputs.logits
    if hasattr(outputs, "prediction_logits") and outputs.prediction_logits is not None:
        return outputs.prediction_logits
    if isinstance(outputs, dict):
        for key in ("logits", "prediction_logits"):
            value = outputs.get(key)
            if value is not None:
                return value
    if isinstance(outputs, tuple) and outputs:
        return outputs[0]
    raise RuntimeError("无法从 UTRBERT pretraining 输出中提取 logits。")


def _special_id_set(tokenizer, model) -> set[int]:
    candidate_values = [
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "cls_token_id", None),
        getattr(tokenizer, "sep_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "mask_token_id", None),
        getattr(tokenizer, "unk_token_id", None),
        getattr(model.config, "null_token_id", None),
    ]
    return {int(value) for value in candidate_values if value is not None}


def compute_pll_scores(
    sequences: Sequence[str],
    model_source: str,
    device,
    batch_size: int = 256,
    show_progress: bool = True,
) -> pd.DataFrame:
    normalized_sequences = [sp.normalize_rna(seq) for seq in sequences if sp.normalize_rna(seq)]
    if not normalized_sequences:
        return pd.DataFrame(columns=["gene", SCORE_COLUMN, "pll_token_count"])

    tokenizer, model = _load_tokenizer_and_model(model_source=model_source, device=device)
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is None:
        raise RuntimeError("RnaTokenizer 未提供 mask_token_id，无法计算 masked PLL。")

    special_ids = _special_id_set(tokenizer, model)
    rows: List[Dict[str, Any]] = []
    progress = sp.make_progress(
        total=len(normalized_sequences),
        desc="utrbert pll",
        disable=not show_progress,
        unit="seq",
    )

    with sp.torch.inference_mode():
        for start in range(0, len(normalized_sequences), batch_size):
            batch_sequences = normalized_sequences[start : start + batch_size]
            tokenized = tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            token_type_ids = tokenized.get("token_type_ids")

            valid_positions = attention_mask.bool()
            for special_id in special_ids:
                valid_positions &= input_ids != int(special_id)

            batch_size_actual, seq_len = input_ids.shape
            nll_sum = sp.torch.zeros(batch_size_actual, dtype=sp.torch.float64)
            token_count = sp.torch.zeros(batch_size_actual, dtype=sp.torch.int32)

            for position in range(seq_len):
                active_mask = valid_positions[:, position]
                if not bool(active_mask.any()):
                    continue
                active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)
                masked_input_ids = input_ids.index_select(0, active_indices).clone()
                original_token_ids = masked_input_ids[:, position].clone()
                masked_input_ids[:, position] = int(mask_token_id)
                masked_attention_mask = attention_mask.index_select(0, active_indices)
                model_kwargs = {
                    "input_ids": masked_input_ids.to(device),
                    "attention_mask": masked_attention_mask.to(device),
                }
                if token_type_ids is not None:
                    model_kwargs["token_type_ids"] = token_type_ids.index_select(0, active_indices).to(device)
                logits = _extract_logits(model(**model_kwargs))
                position_logits = logits[:, position, :]
                log_probs = sp.torch.log_softmax(position_logits.float(), dim=-1)
                gathered = log_probs.gather(
                    dim=1,
                    index=original_token_ids.to(device).unsqueeze(1),
                ).squeeze(1)
                nll_values = (-gathered).detach().cpu()
                nll_sum.index_add_(
                    0,
                    active_indices.cpu(),
                    nll_values.to(dtype=sp.torch.float64),
                )
                token_count.index_add_(
                    0,
                    active_indices.cpu(),
                    sp.torch.ones_like(active_indices.cpu(), dtype=sp.torch.int32),
                )

            token_count_np = token_count.numpy()
            pll_values = (nll_sum.numpy() / np.clip(token_count_np.astype(np.float64), a_min=1.0, a_max=None))
            for sequence, pll_value, count_value in zip(batch_sequences, pll_values.tolist(), token_count_np.tolist()):
                rows.append(
                    {
                        "gene": sequence,
                        SCORE_COLUMN: float(pll_value),
                        "pll_token_count": int(count_value),
                    }
                )
            progress.update(len(batch_sequences))
    progress.close()
    return pd.DataFrame(rows)


def _validate_existing_manifest(
    manifest_path: Path,
    expected_signature: Dict[str, Any],
) -> Dict[str, Any]:
    manifest = _load_json(manifest_path)
    existing_signature = manifest.get("data_signature", {})
    if existing_signature != expected_signature:
        raise RuntimeError(
            "现有 UTRBERT PLL cache 与当前实验配置不匹配。"
        )
    return manifest


def build_or_load_pll_cache(
    unique_genes: Sequence[str],
    cache_dir: Path,
    model_source: str,
    device,
    data_signature: Dict[str, Any],
    batch_size: int = 256,
    show_progress: bool = True,
    force_rebuild: bool = False,
    limit_genes: int = 0,
) -> CacheBuildResult:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    score_table_path = cache_dir / "gene_score_table.tsv"
    manifest_path = cache_dir / "cache_manifest.json"
    summary_path = cache_dir / "cache_summary.json"

    normalized_unique_genes = [sp.normalize_rna(seq) for seq in unique_genes if sp.normalize_rna(seq)]
    normalized_unique_genes = sorted(set(normalized_unique_genes))
    if limit_genes > 0:
        normalized_unique_genes = normalized_unique_genes[: int(limit_genes)]

    if (
        not force_rebuild
        and score_table_path.exists()
        and manifest_path.exists()
        and summary_path.exists()
    ):
        manifest = _validate_existing_manifest(
            manifest_path=manifest_path,
            expected_signature=data_signature,
        )
        summary = _load_json(summary_path)
        return CacheBuildResult(
            cache_dir=cache_dir,
            score_table_path=score_table_path,
            manifest_path=manifest_path,
            summary_path=summary_path,
            reused=True,
            manifest=manifest,
            summary=summary,
        )

    score_df = compute_pll_scores(
        sequences=normalized_unique_genes,
        model_source=model_source,
        device=device,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    score_df.to_csv(score_table_path, sep="\t", index=False)

    manifest = {
        "feature_name": SCORE_FEATURE_UTRBERT_PLL,
        "score_column": SCORE_COLUMN,
        "scoring_mode": SCORING_MODE,
        "normalization_version": NORMALIZATION_VERSION,
        "data_signature": data_signature,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cache_dir": str(cache_dir),
        "score_table_path": str(score_table_path),
        "row_count": int(len(score_df)),
    }
    summary = {
        "feature_name": SCORE_FEATURE_UTRBERT_PLL,
        "score_column": SCORE_COLUMN,
        "cache_dir": str(cache_dir),
        "unique_gene_count": int(len(score_df)),
        "pll_token_count_mean": float(score_df["pll_token_count"].mean()) if not score_df.empty else 0.0,
        "score_mean": float(score_df[SCORE_COLUMN].mean()) if not score_df.empty else float("nan"),
        "score_std": float(score_df[SCORE_COLUMN].std(ddof=0)) if not score_df.empty else float("nan"),
        "score_min": float(score_df[SCORE_COLUMN].min()) if not score_df.empty else float("nan"),
        "score_max": float(score_df[SCORE_COLUMN].max()) if not score_df.empty else float("nan"),
    }
    _save_json(manifest_path, manifest)
    _save_json(summary_path, summary)
    return CacheBuildResult(
        cache_dir=cache_dir,
        score_table_path=score_table_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
        reused=False,
        manifest=manifest,
        summary=summary,
    )


def load_score_table(cache_dir: Path) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    score_table_path = cache_dir / "gene_score_table.tsv"
    if not score_table_path.exists():
        raise FileNotFoundError(f"未找到 gene PLL score table: {score_table_path}")
    return pd.read_csv(score_table_path, sep="\t")


def augment_dataframe_with_pll_feature(
    df: pd.DataFrame,
    score_table: pd.DataFrame,
    feature_name: str,
) -> pd.DataFrame:
    normalized = normalize_sequence_score_feature(feature_name)
    if normalized == SCORE_FEATURE_NONE:
        return df.copy()

    if "gene" not in df.columns:
        raise ValueError("DataFrame 缺少 gene 列，无法追加 UTRBERT PLL 特征。")
    if SCORE_COLUMN not in score_table.columns:
        raise ValueError(f"score_table 缺少列 {SCORE_COLUMN}")

    score_map = pd.Series(
        score_table[SCORE_COLUMN].to_numpy(copy=False),
        index=score_table["gene"].astype(str),
    ).to_dict()
    augmented = df.copy()
    augmented[SCORE_COLUMN] = augmented["gene"].map(score_map)
    missing_count = int(augmented[SCORE_COLUMN].isna().sum())
    if missing_count:
        raise RuntimeError(
            f"UTRBERT PLL 特征 merge 后仍有 {missing_count} 行缺失，说明 cache 与当前数据不匹配。"
        )
    augmented[SCORE_COLUMN] = augmented[SCORE_COLUMN].astype(np.float32)
    return augmented
