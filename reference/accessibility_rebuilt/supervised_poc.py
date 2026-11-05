from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import random
import sys
import warnings
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency at runtime
    tqdm = None

TORCH_IMPORT_ERROR = None
try:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover - exercised at runtime
    torch = None
    nn = None
    GradScaler = None
    autocast = None
    Dataset = object
    DataLoader = None
    TORCH_IMPORT_ERROR = exc


def ensure_runtime_dependencies() -> None:
    if TORCH_IMPORT_ERROR is not None:
        raise RuntimeError(
            "未能导入 torch，请先安装训练依赖：\n"
            "pip install torch"
        ) from TORCH_IMPORT_ERROR


def ensure_compatible_environment() -> None:
    ensure_runtime_dependencies()

    if sys.version_info >= (3, 13):
        detected = ".".join(str(part) for part in sys.version_info[:3])
        warnings.warn(
            "[Warn] "
            f"检测到 Python {detected}。"
            "multimolecule 在 Python 3.13/3.14 下可能因为 dataclass 字段顺序问题导入失败。"
            "如果后续 import multimolecule 报 hidden_size/num_labels 错误，"
            "请运行 `python patch_multimolecule_compat.py`，"
            "或改用 Python 3.10/3.11 + transformers 4.45.2。",
            stacklevel=2,
        )

    try:
        import transformers
        tf_version = transformers.__version__
    except Exception as exc:
        raise RuntimeError(
            "未能导入 transformers，请先安装它：\n"
            "pip install transformers"
        ) from exc

    print(f"[Info] transformers version = {tf_version}")

    major = int(tf_version.split(".")[0])
    if major >= 5:
        print(
            "[Warn] 检测到 transformers>=5。"
            "multimolecule 在该版本下较容易出现导入兼容问题。"
            "若后续 import multimolecule 失败，建议降级 transformers。"
        )


def check_and_import_multimolecule():
    try:
        import multimolecule  # noqa: F401
        from multimolecule import RnaFmModel, RnaTokenizer, UtrBertModel

        # multimolecule 0.2 requires a 28-token RNA-FM vocabulary, while the
        # locally trained legacy checkpoint and tokenizer use the original 26.
        # Preserve that checkpoint shape instead of expanding and randomly
        # initializing two unused embedding rows.
        from multimolecule.models.rnafm.configuration_rnafm import RnaFmConfig

        if not getattr(RnaFmConfig, "_legacy_vocab26_compat", False):
            original_init = RnaFmConfig.__init__

            def legacy_compatible_init(self, *args, **kwargs):
                legacy_vocab26 = kwargs.get("vocab_size") == 26
                if legacy_vocab26:
                    kwargs["vocab_size"] = 28
                original_init(self, *args, **kwargs)
                if legacy_vocab26:
                    self.vocab_size = 26

            RnaFmConfig.__init__ = legacy_compatible_init
            RnaFmConfig._legacy_vocab26_compat = True

        return RnaTokenizer, UtrBertModel, RnaFmModel
    except TypeError as exc:
        msg = str(exc)
        if "non-default argument 'hidden_size' follows default argument 'num_labels'" in msg:
            detected = ".".join(str(part) for part in sys.version_info[:3])
            raise RuntimeError(
                "\n[环境兼容错误]\n"
                f"当前 Python 版本为 {detected}，multimolecule 在该版本下导入失败。\n"
                "这是 multimolecule 配置类字段顺序与 transformers dataclass 化流程冲突导致的，"
                "不是你的数据或脚本逻辑错误。\n\n"
                "优先尝试以下两种方法之一：\n"
                "1. 在当前环境执行：python patch_multimolecule_compat.py\n"
                "2. 改用 Python 3.10 或 3.11 重新创建环境，例如：\n"
                "conda create -n bioinfo311 python=3.11 -y\n"
                "conda activate bioinfo311\n"
                "pip install torch 'transformers==4.45.2' multimolecule pandas scikit-learn huggingface_hub\n"
            ) from exc
        raise
    except ImportError as exc:
        msg = str(exc)
        if "TransformersKwargs" in msg:
            raise RuntimeError(
                "\n[环境兼容错误]\n"
                "当前的 multimolecule 与 transformers 版本不兼容，"
                "导致无法从 transformers.utils 导入 TransformersKwargs。\n\n"
                "建议执行以下命令重新安装兼容版本：\n"
                "pip uninstall -y transformers multimolecule\n"
                "pip install 'transformers==4.45.2'\n"
                "pip install multimolecule\n\n"
                "如果 4.45.2 仍不兼容，再尝试：\n"
                "pip install 'transformers==4.41.2'\n"
            ) from exc

        raise RuntimeError(
            f"导入 multimolecule 失败：{exc}\n"
            "请检查 multimolecule / transformers / torch 的安装版本。"
        ) from exc


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=default)


def normalize_rna(seq: str) -> str:
    if pd.isna(seq):
        return ""
    return str(seq).strip().upper().replace(" ", "").replace("T", "U")


def resolve_model_source(model_name: str, local_dir_name: str) -> str:
    local_path = Path("model") / local_dir_name
    if local_path.is_dir():
        print(f"[Info] 使用本地模型目录: {local_path}")
        return str(local_path)
    return model_name


def extract_trainable_state_dict(model) -> Dict[str, "torch.Tensor"]:
    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }


def load_trainable_state_dict(model, state_dict: Dict[str, "torch.Tensor"]) -> None:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"载入 trainable state_dict 时遇到未知参数: {unexpected}")
    if missing:
        trainable_names = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        remaining_missing = sorted(set(missing) & trainable_names)
        if remaining_missing:
            raise RuntimeError(
                f"载入 trainable state_dict 后仍缺少可训练参数: {remaining_missing}"
            )


def safe_torch_save(payload: Dict[str, Any], path: Path) -> None:
    try:
        torch.save(payload, path)
    except RuntimeError as exc:
        message = str(exc)
        if "iostream error" not in message and "unexpected pos" not in message:
            raise
        print(
            "[Warn] torch.save 使用默认 zip 序列化失败，"
            "正在退回旧序列化格式重试。"
        )
        torch.save(payload, path, _use_new_zipfile_serialization=False)


class _NoOpProgress:
    def __init__(self, iterable=None):
        self.iterable = iterable

    def __iter__(self):
        if self.iterable is None:
            return iter(())
        return iter(self.iterable)

    def update(self, _: int = 1) -> None:
        return None

    def set_postfix(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def make_progress(
    iterable=None,
    *,
    total: Optional[int] = None,
    desc: Optional[str] = None,
    disable: bool = False,
    leave: bool = False,
    unit: str = "it",
    unit_scale: bool = False,
):
    if disable or tqdm is None:
        return _NoOpProgress(iterable)
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        leave=leave,
        dynamic_ncols=True,
        unit=unit,
        unit_scale=unit_scale,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train/val/test 比例之和必须为 1.0，当前为 {total:.6f}"
        )


@dataclass
class RunConfig:
    input_file: str
    output_dir: str
    models: str = "baseline_pooled,cross_attention_interaction"
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


@dataclass
class PreparedData:
    dataframe: pd.DataFrame
    summary: Dict[str, Any]


@dataclass
class TokenizedSequenceCache:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    token_type_ids: Optional[np.ndarray] = None


def load_filtered_dataframe(config: RunConfig) -> PreparedData:
    usecols = ["gene", "noncodingRNA", config.group_key, "feature", "label"]
    chunks: List[pd.DataFrame] = []
    raw_rows = 0
    filtered_rows = 0

    reader = pd.read_csv(
        config.input_file,
        sep="\t",
        usecols=usecols,
        chunksize=config.chunksize,
    )
    progress = make_progress(
        desc="load/filter data",
        disable=not config.progress_bar,
        unit="rows",
        unit_scale=True,
    )
    for chunk in reader:
        raw_rows += len(chunk)
        progress.update(len(chunk))
        chunk = chunk[chunk["feature"].fillna("") == config.feature_value].copy()
        filtered_rows += len(chunk)
        if chunk.empty:
            continue

        chunk["gene"] = chunk["gene"].map(normalize_rna)
        chunk["noncodingRNA"] = chunk["noncodingRNA"].map(normalize_rna)
        chunk[config.group_key] = (
            chunk[config.group_key]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .replace("", "unknown")
        )
        chunk["label"] = chunk["label"].astype(int)
        chunks.append(chunk[["gene", "noncodingRNA", config.group_key, "label"]])
    progress.close()

    if not chunks:
        raise ValueError(
            f"未在 {config.input_file} 中找到 feature == {config.feature_value!r} 的样本。"
        )

    df = pd.concat(chunks, ignore_index=True)

    pair_labels = df.groupby(["gene", "noncodingRNA"])["label"].nunique()
    conflict_pairs = pair_labels[pair_labels > 1]
    conflict_rows = 0
    if not conflict_pairs.empty:
        pair_index = pd.MultiIndex.from_frame(df[["gene", "noncodingRNA"]])
        conflict_index = pd.MultiIndex.from_tuples(conflict_pairs.index)
        conflict_mask = pair_index.isin(conflict_index)
        conflict_rows = int(conflict_mask.sum())
        df = df.loc[~conflict_mask].copy()

    df = df.reset_index(drop=True)
    df["row_id"] = df.index.astype(int)
    gene_seq_ids, unique_genes = pd.factorize(df["gene"], sort=False)
    mirna_seq_ids, unique_mirnas = pd.factorize(df["noncodingRNA"], sort=False)
    df["gene_seq_id"] = gene_seq_ids.astype(np.int32, copy=False)
    df["mirna_seq_id"] = mirna_seq_ids.astype(np.int32, copy=False)

    label_counts = df["label"].value_counts().sort_index().to_dict()
    summary = {
        "input_file": config.input_file,
        "feature_value": config.feature_value,
        "raw_rows": raw_rows,
        "filtered_rows": filtered_rows,
        "retained_rows": len(df),
        "conflict_pair_count": int(len(conflict_pairs)),
        "conflict_row_count": conflict_rows,
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "positive_rate": float(df["label"].mean()),
        "group_key": config.group_key,
        "group_count": int(df[config.group_key].nunique()),
        "unique_gene_count": int(len(unique_genes)),
        "unique_mirna_count": int(len(unique_mirnas)),
    }
    return PreparedData(dataframe=df, summary=summary)


def configure_runtime_for_device(device: "torch.device") -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def resolve_amp_dtype(
    amp_dtype_arg: str,
    device: "torch.device",
) -> Optional["torch.dtype"]:
    if device.type != "cuda":
        return None

    amp_dtype_arg = amp_dtype_arg.lower()
    if amp_dtype_arg == "auto":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if amp_dtype_arg not in mapping:
        raise ValueError(f"不支持的 amp_dtype: {amp_dtype_arg}")
    return mapping[amp_dtype_arg]


def _build_sequence_token_cache(
    sequences: np.ndarray,
    tokenizer,
    max_length: int,
    batch_size: int,
    label: str,
    show_progress: bool,
    persistent_cache_dir: Optional[Path] = None,
) -> TokenizedSequenceCache:
    unique_sequences = pd.unique(sequences)
    if len(unique_sequences) == 0:
        raise ValueError(f"{label} 序列为空，无法构建 token cache。")

    tokenizer_name = str(getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__))
    digest = hashlib.sha256()
    digest.update(tokenizer_name.encode("utf-8"))
    digest.update(str(int(max_length)).encode("ascii"))
    for sequence in unique_sequences:
        digest.update(str(sequence).encode("utf-8"))
        digest.update(b"\0")
    cache_key = digest.hexdigest()[:24]
    cache_path = None
    if persistent_cache_dir is not None:
        cache_path = Path(persistent_cache_dir) / label / cache_key
        manifest_path = cache_path / "manifest.json"
        input_ids_path = cache_path / "input_ids.npy"
        attention_mask_path = cache_path / "attention_mask.npy"
        token_type_path = cache_path / "token_type_ids.npy"
        if manifest_path.is_file() and input_ids_path.is_file() and attention_mask_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                input_ids = np.load(input_ids_path, mmap_mode="r")
                attention_mask = np.load(attention_mask_path, mmap_mode="r")
                token_type_ids = np.load(token_type_path, mmap_mode="r") if token_type_path.is_file() else None
                expected_shape = (len(unique_sequences), int(max_length))
                if (
                    manifest.get("cache_key") == cache_key
                    and tuple(input_ids.shape) == expected_shape
                    and tuple(attention_mask.shape) == expected_shape
                    and (token_type_ids is None or tuple(token_type_ids.shape) == expected_shape)
                ):
                    print(
                        f"[Info] Reusing persistent {label} token cache: "
                        f"{cache_path} ({len(unique_sequences)} sequences)"
                    )
                    return TokenizedSequenceCache(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )
            except (OSError, ValueError, json.JSONDecodeError):
                pass

    if cache_path is not None and os.environ.get(
        "MIRBENCH_STATIC_CACHE_READ_ONLY", ""
    ).strip().lower() in {"1", "true", "yes"}:
        raise RuntimeError(f"Read-only persistent token cache miss: {cache_path}")

    print(
        f"[Info] 构建 {label} token cache: "
        f"{len(unique_sequences)} unique sequences, max_length={max_length}"
    )

    try:
        vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    except (TypeError, ValueError):
        vocab_size = 0
    input_dtype = np.uint16 if 0 < vocab_size <= np.iinfo(np.uint16).max else np.int32
    cache_shape = (len(unique_sequences), int(max_length))
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)
        input_ids = np.lib.format.open_memmap(
            cache_path / "input_ids.npy", mode="w+", dtype=input_dtype, shape=cache_shape
        )
        attention_mask = np.lib.format.open_memmap(
            cache_path / "attention_mask.npy", mode="w+", dtype=np.uint8, shape=cache_shape
        )
    else:
        input_ids = np.empty(cache_shape, dtype=input_dtype)
        attention_mask = np.empty(cache_shape, dtype=np.uint8)
    token_type_ids = None
    has_token_type_ids = False

    progress = make_progress(
        total=len(unique_sequences),
        desc=f"tokenize {label}",
        disable=not show_progress,
        unit="seq",
    )
    for start in range(0, len(unique_sequences), batch_size):
        batch = unique_sequences[start : start + batch_size].tolist()
        encoded = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        stop = start + len(batch)
        input_ids[start:stop] = np.asarray(encoded["input_ids"], dtype=input_dtype)
        attention_mask[start:stop] = np.asarray(encoded["attention_mask"], dtype=np.uint8)
        progress.update(len(batch))

        encoded_token_types = encoded.get("token_type_ids")
        if encoded_token_types is not None:
            if token_type_ids is None:
                if cache_path is not None:
                    token_type_ids = np.lib.format.open_memmap(
                        cache_path / "token_type_ids.npy",
                        mode="w+",
                        dtype=np.uint8,
                        shape=cache_shape,
                    )
                else:
                    token_type_ids = np.empty(cache_shape, dtype=np.uint8)
            has_token_type_ids = True
            token_type_ids[start:stop] = np.asarray(encoded_token_types, dtype=np.uint8)
    progress.close()

    for array in (input_ids, attention_mask, token_type_ids):
        if isinstance(array, np.memmap):
            array.flush()
    if cache_path is not None:
        manifest = {
            "cache_key": cache_key,
            "label": label,
            "tokenizer": tokenizer_name,
            "max_length": int(max_length),
            "sequence_count": int(len(unique_sequences)),
            "input_ids_dtype": str(np.dtype(input_dtype)),
            "has_token_type_ids": bool(has_token_type_ids),
        }
        (cache_path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return TokenizedSequenceCache(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids if has_token_type_ids else None,
    )


def build_dual_token_caches(
    df: pd.DataFrame,
    gene_tokenizer,
    mirna_tokenizer,
    max_gene_len: int,
    max_mirna_len: int,
    batch_size: int,
    enabled: bool,
    show_progress: bool,
    persistent_cache_dir: Optional[Path] = None,
) -> Tuple[Optional[TokenizedSequenceCache], Optional[TokenizedSequenceCache]]:
    if not enabled:
        return None, None

    gene_cache = _build_sequence_token_cache(
        sequences=df["gene"].to_numpy(copy=False),
        tokenizer=gene_tokenizer,
        max_length=max_gene_len,
        batch_size=batch_size,
        label="gene",
        show_progress=show_progress,
        persistent_cache_dir=persistent_cache_dir,
    )
    mirna_cache = _build_sequence_token_cache(
        sequences=df["noncodingRNA"].to_numpy(copy=False),
        tokenizer=mirna_tokenizer,
        max_length=max_mirna_len,
        batch_size=batch_size,
        label="miRNA",
        show_progress=show_progress,
        persistent_cache_dir=persistent_cache_dir,
    )
    return gene_cache, mirna_cache


def parse_model_names(models_arg: str) -> List[str]:
    models = [item.strip() for item in models_arg.split(",") if item.strip()]
    allowed = {"baseline_pooled", "cross_attention_interaction"}
    unknown = sorted(set(models) - allowed)
    if unknown:
        raise ValueError(f"未知模型名: {unknown}")
    if not models:
        raise ValueError("至少需要一个模型名。")
    return models


def resolve_num_workers(num_workers: int, device: "torch.device") -> int:
    if num_workers > 0:
        return num_workers
    cpu_count = os.cpu_count() or 1
    if device.type == "cuda":
        return max(2, min(8, cpu_count // 2))
    return max(0, min(4, cpu_count // 2))


def resolve_eval_batch_size(train_batch_size: int, eval_batch_size: int) -> int:
    if eval_batch_size > 0:
        return eval_batch_size
    return max(train_batch_size, train_batch_size * 4)


def build_adamw_optimizer(
    model,
    learning_rate: float,
    weight_decay: float,
    device: "torch.device",
    use_fused: bool,
):
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer_kwargs = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
    }
    if device.type == "cuda":
        if use_fused:
            try:
                return torch.optim.AdamW(parameters, fused=True, **optimizer_kwargs)
            except (TypeError, RuntimeError):
                pass
        try:
            return torch.optim.AdamW(parameters, foreach=True, **optimizer_kwargs)
        except TypeError:
            pass
    return torch.optim.AdamW(parameters, **optimizer_kwargs)


def _record_cuda_stream(batch: Dict[str, Any], stream) -> None:
    for value in batch.values():
        if torch.is_tensor(value) and value.device.type == "cuda":
            value.record_stream(stream)


class CUDAPrefetchLoader:
    device_prefetched = True

    def __init__(self, loader, device: "torch.device"):
        self.loader = loader
        self.device = device
        self.stream = None
        self.loader_iter = None
        self.next_batch = None

    def __len__(self) -> int:
        return len(self.loader)

    def _preload(self) -> None:
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            self.next_batch = move_batch_to_device(batch, self.device)

    def __iter__(self):
        if self.device.type != "cuda":
            yield from self.loader
            return

        self.stream = torch.cuda.Stream(device=self.device)
        self.loader_iter = iter(self.loader)
        self._preload()
        while self.next_batch is not None:
            current_stream = torch.cuda.current_stream(device=self.device)
            current_stream.wait_stream(self.stream)
            batch = self.next_batch
            _record_cuda_stream(batch, current_stream)
            self._preload()
            yield batch


def _append_group_to_split(
    split_name: str,
    group_row: Dict[str, Any],
    split_state: Dict[str, Dict[str, Any]],
) -> None:
    split_state[split_name]["rows"].append(group_row)
    split_state[split_name]["sample_count"] += int(group_row["sample_count"])
    split_state[split_name]["pos_count"] += int(group_row["pos_count"])
    split_state[split_name]["neg_count"] += int(group_row["neg_count"])


def greedy_group_split(
    df: pd.DataFrame,
    group_key: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    validate_split_ratios(train_ratio, val_ratio, test_ratio)

    group_stats = (
        df.groupby(group_key)
        .agg(
            sample_count=("label", "size"),
            pos_count=("label", "sum"),
        )
        .reset_index()
    )
    group_stats["neg_count"] = group_stats["sample_count"] - group_stats["pos_count"]
    group_stats["pos_rate"] = group_stats["pos_count"] / group_stats["sample_count"]

    split_order = ["train", "val", "test"]
    ratio_map = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }
    total_count = float(group_stats["sample_count"].sum())
    total_pos = float(group_stats["pos_count"].sum())
    global_pos_rate = total_pos / max(total_count, 1.0)

    mixed_groups = group_stats[
        (group_stats["pos_count"] > 0) & (group_stats["neg_count"] > 0)
    ].sort_values(["sample_count", "pos_count"], ascending=False)
    if len(mixed_groups) < len(split_order):
        raise ValueError(
            "可用于初始化 split 的混合标签 family 不足，无法保证 train/val/test 都包含正负样本。"
        )

    split_state = {
        split: {
            "rows": [],
            "sample_count": 0,
            "pos_count": 0,
            "neg_count": 0,
        }
        for split in split_order
    }

    train_seed = mixed_groups.iloc[0].to_dict()
    remaining_mixed = mixed_groups.iloc[1:].sort_values(["sample_count", "pos_count"])
    val_seed = remaining_mixed.iloc[0].to_dict()
    test_seed = remaining_mixed.iloc[1].to_dict()

    seeded_groups = set()
    for split_name, group_row in zip(
        split_order,
        [train_seed, val_seed, test_seed],
    ):
        _append_group_to_split(split_name, group_row, split_state)
        seeded_groups.add(group_row[group_key])

    remaining = group_stats.loc[~group_stats[group_key].isin(seeded_groups)].copy()
    remaining["priority"] = remaining["sample_count"] * (
        1.0 + (remaining["pos_rate"] - global_pos_rate).abs()
    )
    remaining = remaining.sort_values(["priority", "sample_count"], ascending=False)

    target_count = {
        split: ratio_map[split] * total_count
        for split in split_order
    }
    target_pos = {
        split: ratio_map[split] * total_pos
        for split in split_order
    }
    rng = random.Random(seed)

    def assignment_score(split_name: str, group_row: Dict[str, Any]) -> float:
        new_count = split_state[split_name]["sample_count"] + int(group_row["sample_count"])
        new_pos = split_state[split_name]["pos_count"] + int(group_row["pos_count"])

        count_ratio = new_count / max(target_count[split_name], 1.0)
        pos_ratio = new_pos / max(target_pos[split_name], 1.0)
        count_gap = abs(count_ratio - 1.0)
        pos_gap = abs(pos_ratio - 1.0)
        overflow = max(count_ratio - 1.0, 0.0)
        return (1.5 * count_gap) + (0.75 * pos_gap) + (3.0 * overflow) + rng.random() * 1e-8

    for _, row in remaining.iterrows():
        group_row = row.to_dict()
        best_split = min(split_order, key=lambda name: assignment_score(name, group_row))
        _append_group_to_split(best_split, group_row, split_state)

    manifest_groups = []
    group_to_split: Dict[str, str] = {}
    split_summary: Dict[str, Any] = {}

    for split_name in split_order:
        rows = split_state[split_name]["rows"]
        if split_state[split_name]["pos_count"] == 0 or split_state[split_name]["neg_count"] == 0:
            raise ValueError(f"{split_name} split 未覆盖正负两类，请调整切分逻辑。")

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

    if out_df["split"].isna().any():
        raise ValueError("存在未成功映射到 split 的 group。")

    for split_name in split_order:
        subset = out_df[out_df["split"] == split_name]
        if subset.empty:
            raise ValueError(f"{split_name} split 为空。")
        if subset["label"].nunique() < 2:
            raise ValueError(f"{split_name} split 缺少正类或负类。")

    manifest = {
        "group_key": group_key,
        "ratios": ratio_map,
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


def compute_binary_metrics(labels: List[int], probabilities: List[float]) -> Dict[str, float]:
    if len(labels) != len(probabilities):
        raise ValueError("labels 和 probabilities 长度不一致。")

    predictions = [1 if value >= 0.5 else 0 for value in probabilities]

    def safe_metric(metric_fn, *args):
        try:
            return float(metric_fn(*args))
        except ValueError:
            return float("nan")

    return {
        "row_count": int(len(labels)),
        "positive_rate": float(sum(labels) / max(len(labels), 1)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": safe_metric(balanced_accuracy_score, labels, predictions),
        "f1": safe_metric(f1_score, labels, predictions),
        "mcc": safe_metric(matthews_corrcoef, labels, predictions),
        "auroc": safe_metric(roc_auc_score, labels, probabilities),
        "auprc": safe_metric(average_precision_score, labels, probabilities),
    }


def build_smoke_test_subset(
    df: pd.DataFrame,
    seed: int,
    split_row_targets: Dict[str, int],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    sampled_splits: List[pd.DataFrame] = []
    summary: Dict[str, Any] = {}

    for split_index, split_name in enumerate(["train", "val", "test"]):
        subset = df[df["split"] == split_name].copy()
        target_rows = min(split_row_targets[split_name], len(subset))
        if subset["label"].nunique() < 2:
            raise ValueError(f"{split_name} split 缺少正负样本，无法构建 smoke-test 子集。")
        if target_rows < 2:
            raise ValueError(f"{split_name} split 样本过少，无法构建 smoke-test 子集。")

        positives = subset[subset["label"] == 1]
        negatives = subset[subset["label"] == 0]

        positive_target = round(target_rows * (len(positives) / len(subset)))
        positive_target = max(1, min(len(positives), positive_target))
        negative_target = target_rows - positive_target
        negative_target = max(1, min(len(negatives), negative_target))

        if positive_target + negative_target > target_rows:
            if positive_target > negative_target:
                positive_target -= 1
            else:
                negative_target -= 1

        while positive_target + negative_target < target_rows:
            if positive_target < len(positives):
                positive_target += 1
                continue
            if negative_target < len(negatives):
                negative_target += 1
                continue
            break

        sampled_positive = positives.sample(
            n=positive_target,
            random_state=seed + split_index * 13 + 1,
        )
        sampled_negative = negatives.sample(
            n=negative_target,
            random_state=seed + split_index * 13 + 2,
        )
        sampled_subset = pd.concat([sampled_positive, sampled_negative], ignore_index=True)
        sampled_subset = sampled_subset.sample(
            frac=1.0,
            random_state=seed + split_index * 13 + 3,
        ).reset_index(drop=True)

        sampled_splits.append(sampled_subset)
        summary[split_name] = {
            "original_rows": int(len(subset)),
            "sampled_rows": int(len(sampled_subset)),
            "positive_count": int(sampled_subset["label"].sum()),
            "negative_count": int(len(sampled_subset) - sampled_subset["label"].sum()),
            "positive_rate": float(sampled_subset["label"].mean()),
        }

    sampled_df = pd.concat(sampled_splits, ignore_index=True)
    sampled_df = sampled_df.reset_index(drop=True)
    return sampled_df, summary


if torch is not None:

    class PairDataset(Dataset):
        def __init__(
            self,
            df: pd.DataFrame,
            indices: np.ndarray,
            group_key: str,
            include_sequences: bool = True,
        ):
            self.indices = np.asarray(indices, dtype=np.int64)
            self.row_ids = df["row_id"].to_numpy(copy=False)
            self.labels = df["label"].to_numpy(dtype=np.float32, copy=False)
            self.groups = df[group_key].to_numpy(copy=False)
            self.genes = df["gene"].to_numpy(copy=False) if include_sequences else None
            self.mirnas = (
                df["noncodingRNA"].to_numpy(copy=False) if include_sequences else None
            )
            self.gene_seq_ids = (
                df["gene_seq_id"].to_numpy(dtype=np.int32, copy=False)
                if "gene_seq_id" in df.columns
                else None
            )
            self.mirna_seq_ids = (
                df["mirna_seq_id"].to_numpy(dtype=np.int32, copy=False)
                if "mirna_seq_id" in df.columns
                else None
            )
            self.handcrafted_feature_columns = sorted(
                column for column in df.columns if str(column).startswith("hf_")
            )
            self.handcrafted_features = (
                df[self.handcrafted_feature_columns].to_numpy(dtype=np.float32, copy=False)
                if self.handcrafted_feature_columns
                else None
            )

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            row_index = int(self.indices[idx])
            item = {
                "row_id": int(self.row_ids[row_index]),
                "label": float(self.labels[row_index]),
                "group_value": self.groups[row_index],
            }
            if self.genes is not None:
                item["gene"] = self.genes[row_index]
            if self.mirnas is not None:
                item["mirna"] = self.mirnas[row_index]
            if self.gene_seq_ids is not None:
                item["gene_seq_id"] = int(self.gene_seq_ids[row_index])
            if self.mirna_seq_ids is not None:
                item["mirna_seq_id"] = int(self.mirna_seq_ids[row_index])
            if self.handcrafted_features is not None:
                item["handcrafted_features"] = self.handcrafted_features[row_index]
            return item


    class DualTokenizerCollator:
        def __init__(
            self,
            gene_tokenizer,
            mirna_tokenizer,
            max_gene_len: int,
            max_mirna_len: int,
            gene_cache: Optional[TokenizedSequenceCache] = None,
            mirna_cache: Optional[TokenizedSequenceCache] = None,
        ):
            self.gene_tokenizer = gene_tokenizer
            self.mirna_tokenizer = mirna_tokenizer
            self.max_gene_len = max_gene_len
            self.max_mirna_len = max_mirna_len
            self.gene_cache = gene_cache
            self.mirna_cache = mirna_cache

        def _tokenize_from_cache(
            self,
            seq_ids: List[int],
            cache: TokenizedSequenceCache,
        ) -> Dict[str, torch.Tensor]:
            index_array = np.asarray(seq_ids, dtype=np.int64)
            payload = {
                "input_ids": torch.from_numpy(cache.input_ids[index_array]).long(),
                "attention_mask": torch.from_numpy(cache.attention_mask[index_array]).long(),
            }
            if cache.token_type_ids is not None:
                payload["token_type_ids"] = torch.from_numpy(
                    cache.token_type_ids[index_array]
                ).long()
            return payload

        def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            genes = [item["gene"] for item in batch] if "gene" in batch[0] else None
            mirnas = [item["mirna"] for item in batch] if "mirna" in batch[0] else None
            gene_seq_ids = [item.get("gene_seq_id") for item in batch]
            mirna_seq_ids = [item.get("mirna_seq_id") for item in batch]

            if self.gene_cache is not None and all(value is not None for value in gene_seq_ids):
                gene_inputs = self._tokenize_from_cache(gene_seq_ids, self.gene_cache)
            else:
                if genes is None:
                    raise RuntimeError("缺少 gene 序列，且无法从 cache 回填 token。")
                gene_inputs = self.gene_tokenizer(
                    genes,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_gene_len,
                )

            if self.mirna_cache is not None and all(value is not None for value in mirna_seq_ids):
                mirna_inputs = self._tokenize_from_cache(mirna_seq_ids, self.mirna_cache)
            else:
                if mirnas is None:
                    raise RuntimeError("缺少 miRNA 序列，且无法从 cache 回填 token。")
                mirna_inputs = self.mirna_tokenizer(
                    mirnas,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_mirna_len,
                )

            payload: Dict[str, Any] = {
                "gene_input_ids": gene_inputs["input_ids"],
                "gene_attention_mask": gene_inputs["attention_mask"],
                "mirna_input_ids": mirna_inputs["input_ids"],
                "mirna_attention_mask": mirna_inputs["attention_mask"],
                "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
                "row_ids": [item["row_id"] for item in batch],
                "group_values": [item["group_value"] for item in batch],
            }
            if genes is not None:
                payload["genes"] = genes
            if mirnas is not None:
                payload["mirnas"] = mirnas
            if "handcrafted_features" in batch[0]:
                handcrafted = np.stack(
                    [np.asarray(item["handcrafted_features"], dtype=np.float32) for item in batch],
                    axis=0,
                )
                payload["handcrafted_features"] = torch.from_numpy(handcrafted)

            if "token_type_ids" in gene_inputs:
                payload["gene_token_type_ids"] = gene_inputs["token_type_ids"]
            if "token_type_ids" in mirna_inputs:
                payload["mirna_token_type_ids"] = mirna_inputs["token_type_ids"]
            return payload


    def masked_mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        masked = hidden_states * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts


    def masked_max_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).bool()
        masked = hidden_states.masked_fill(~mask, float("-inf"))
        values = masked.max(dim=1).values
        return torch.where(torch.isfinite(values), values, torch.zeros_like(values))


    class FrozenDualEncoderBase(nn.Module):
        def __init__(
            self,
            UtrBertModel,
            RnaFmModel,
            gene_model_name: str,
            mirna_model_name: str,
        ):
            super().__init__()
            self.gene_encoder = UtrBertModel.from_pretrained(gene_model_name)
            self.mirna_encoder = RnaFmModel.from_pretrained(mirna_model_name)

            for parameter in self.gene_encoder.parameters():
                parameter.requires_grad = False
            for parameter in self.mirna_encoder.parameters():
                parameter.requires_grad = False

            self.gene_encoder.eval()
            self.mirna_encoder.eval()

        def train(self, mode: bool = True):
            super().train(mode)
            self.gene_encoder.eval()
            self.mirna_encoder.eval()
            return self

        def _encode_backbone(
            self,
            encoder,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids

            with torch.no_grad():
                outputs = encoder(**kwargs)
            return outputs.last_hidden_state


    class PooledBaselineModel(FrozenDualEncoderBase):
        def __init__(
            self,
            UtrBertModel,
            RnaFmModel,
            gene_model_name: str,
            mirna_model_name: str,
            classifier_hidden_dim: int,
            dropout: float,
        ):
            super().__init__(
                UtrBertModel=UtrBertModel,
                RnaFmModel=RnaFmModel,
                gene_model_name=gene_model_name,
                mirna_model_name=mirna_model_name,
            )
            gene_hidden = self.gene_encoder.config.hidden_size
            mirna_hidden = self.mirna_encoder.config.hidden_size
            self.classifier = nn.Sequential(
                nn.Linear(gene_hidden + mirna_hidden, classifier_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, 1),
            )

        def forward(
            self,
            gene_input_ids: torch.Tensor,
            gene_attention_mask: torch.Tensor,
            mirna_input_ids: torch.Tensor,
            mirna_attention_mask: torch.Tensor,
            gene_token_type_ids: Optional[torch.Tensor] = None,
            mirna_token_type_ids: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            gene_tokens = self._encode_backbone(
                encoder=self.gene_encoder,
                input_ids=gene_input_ids,
                attention_mask=gene_attention_mask,
                token_type_ids=gene_token_type_ids,
            )
            mirna_tokens = self._encode_backbone(
                encoder=self.mirna_encoder,
                input_ids=mirna_input_ids,
                attention_mask=mirna_attention_mask,
                token_type_ids=mirna_token_type_ids,
            )

            gene_vec = masked_mean_pooling(gene_tokens, gene_attention_mask)
            mirna_vec = masked_mean_pooling(mirna_tokens, mirna_attention_mask)
            features = torch.cat([gene_vec, mirna_vec], dim=-1)
            return self.classifier(features).squeeze(-1)


    class Route1TokenInteractionHead(nn.Module):
        def __init__(
            self,
            gene_input_dim: int,
            mirna_input_dim: int,
            interaction_hidden_dim: int,
            classifier_hidden_dim: int,
            attention_heads: int,
            interaction_top_k: int,
            dropout: float,
            interaction_scorer: str = "dot_product",
            feature_processor: str = "interaction_stats",
            handcrafted_feature_dim: int = 0,
        ):
            super().__init__()

            self.interaction_hidden_dim = interaction_hidden_dim
            self.interaction_top_k = interaction_top_k
            self.interaction_scorer = interaction_scorer
            self.feature_processor = feature_processor
            self.handcrafted_feature_dim = max(0, int(handcrafted_feature_dim))
            gene_hidden = gene_input_dim
            mirna_hidden = mirna_input_dim

            heads = max(1, min(attention_heads, interaction_hidden_dim))
            while interaction_hidden_dim % heads != 0 and heads > 1:
                heads -= 1

            self.gene_projection = nn.Linear(gene_hidden, interaction_hidden_dim)
            self.mirna_projection = nn.Linear(mirna_hidden, interaction_hidden_dim)
            self.mirna_to_gene_attention = nn.MultiheadAttention(
                embed_dim=interaction_hidden_dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True,
            )
            self.gene_to_mirna_attention = nn.MultiheadAttention(
                embed_dim=interaction_hidden_dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True,
            )
            valid_scorers = {"dot_product", "biaffine"}
            if interaction_scorer not in valid_scorers:
                raise ValueError(
                    "interaction_scorer must be one of {'dot_product', 'biaffine'}."
                )
            if feature_processor not in {"interaction_stats", "dot_product_stats", "biaffine_stats"}:
                raise ValueError(
                    "feature_processor must be one of "
                    "{'interaction_stats', 'dot_product_stats', 'biaffine_stats'}."
                )
            scorer_choices = {interaction_scorer}
            if feature_processor == "biaffine_stats":
                scorer_choices.add("biaffine")
            if "biaffine" in scorer_choices:
                self.biaffine_weight = nn.Parameter(
                    torch.empty(interaction_hidden_dim, interaction_hidden_dim)
                )
                self.biaffine_mirna = nn.Parameter(torch.empty(interaction_hidden_dim))
                self.biaffine_gene = nn.Parameter(torch.empty(interaction_hidden_dim))
                self.biaffine_bias = nn.Parameter(torch.zeros(1))
                nn.init.xavier_uniform_(self.biaffine_weight)
                nn.init.normal_(self.biaffine_mirna, mean=0.0, std=0.02)
                nn.init.normal_(self.biaffine_gene, mean=0.0, std=0.02)
            self.base_feature_dim = interaction_hidden_dim * 6 + 5
            self.feature_dim = self.base_feature_dim + self.handcrafted_feature_dim
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, classifier_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, 1),
            )

        def _resolved_scorer(self) -> str:
            if self.feature_processor == "dot_product_stats":
                return "dot_product"
            if self.feature_processor == "biaffine_stats":
                return "biaffine"
            return self.interaction_scorer

        def _compute_interaction_scores(
            self,
            mirna_tokens: torch.Tensor,
            gene_tokens: torch.Tensor,
        ) -> torch.Tensor:
            scorer = self._resolved_scorer()
            if scorer == "dot_product":
                return torch.matmul(
                    mirna_tokens,
                    gene_tokens.transpose(1, 2),
                ) / math.sqrt(self.interaction_hidden_dim)

            bilinear_scores = torch.einsum(
                "bmd,dk,bgk->bmg",
                mirna_tokens,
                self.biaffine_weight,
                gene_tokens,
            )
            mirna_bias = torch.einsum("bmd,d->bm", mirna_tokens, self.biaffine_mirna).unsqueeze(-1)
            gene_bias = torch.einsum("bgd,d->bg", gene_tokens, self.biaffine_gene).unsqueeze(1)
            return bilinear_scores + mirna_bias + gene_bias + self.biaffine_bias

        def _interaction_stats(
            self,
            scores: torch.Tensor,
            pair_mask: torch.Tensor,
            mirna_attention_mask: torch.Tensor,
            gene_attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            valid_scores = scores.masked_fill(~pair_mask, 0.0)
            valid_count = pair_mask.sum(dim=(1, 2)).clamp(min=1).float()
            global_mean = valid_scores.sum(dim=(1, 2)) / valid_count

            masked_scores = scores.masked_fill(~pair_mask, float("-inf"))
            global_max = masked_scores.amax(dim=(1, 2))
            global_max = torch.where(torch.isfinite(global_max), global_max, torch.zeros_like(global_max))

            row_max = masked_scores.max(dim=2).values
            row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
            row_max = row_max * mirna_attention_mask.float()
            row_count = mirna_attention_mask.sum(dim=1).clamp(min=1).float()
            row_max_mean = row_max.sum(dim=1) / row_count

            col_max = masked_scores.max(dim=1).values
            col_max = torch.where(torch.isfinite(col_max), col_max, torch.zeros_like(col_max))
            col_max = col_max * gene_attention_mask.float()
            col_count = gene_attention_mask.sum(dim=1).clamp(min=1).float()
            col_max_mean = col_max.sum(dim=1) / col_count

            flat_scores = masked_scores.view(scores.size(0), -1)
            top_k = max(1, min(self.interaction_top_k, flat_scores.size(1)))
            topk_values = flat_scores.topk(top_k, dim=1).values
            topk_values = torch.where(
                torch.isfinite(topk_values),
                topk_values,
                torch.zeros_like(topk_values),
            )
            valid_topk = pair_mask.view(scores.size(0), -1).sum(dim=1)
            valid_topk = valid_topk.clamp(min=1, max=top_k).float()
            topk_mean = topk_values.sum(dim=1) / valid_topk

            return torch.stack(
                [global_mean, global_max, row_max_mean, col_max_mean, topk_mean],
                dim=-1,
            )

        def forward_features_from_embeddings(
            self,
            gene_token_embeddings: torch.Tensor,
            gene_attention_mask: torch.Tensor,
            mirna_token_embeddings: torch.Tensor,
            mirna_attention_mask: torch.Tensor,
            handcrafted_features: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            gene_tokens = self.gene_projection(gene_token_embeddings)
            mirna_tokens = self.mirna_projection(mirna_token_embeddings)

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

            interaction_scores = self._compute_interaction_scores(
                mirna_tokens=mirna_tokens,
                gene_tokens=gene_tokens,
            )
            pair_mask = (
                mirna_attention_mask.bool().unsqueeze(-1)
                & gene_attention_mask.bool().unsqueeze(1)
            )

            features = torch.cat(
                [
                    masked_mean_pooling(gene_tokens, gene_attention_mask),
                    masked_mean_pooling(mirna_tokens, mirna_attention_mask),
                    masked_mean_pooling(gene_cross, gene_attention_mask),
                    masked_max_pooling(gene_cross, gene_attention_mask),
                    masked_mean_pooling(mirna_cross, mirna_attention_mask),
                    masked_max_pooling(mirna_cross, mirna_attention_mask),
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
                features = torch.cat([features, handcrafted_features], dim=-1)
            return features

        def forward_from_embeddings(
            self,
            gene_token_embeddings: torch.Tensor,
            gene_attention_mask: torch.Tensor,
            mirna_token_embeddings: torch.Tensor,
            mirna_attention_mask: torch.Tensor,
            handcrafted_features: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            features = self.forward_features_from_embeddings(
                gene_token_embeddings=gene_token_embeddings,
                gene_attention_mask=gene_attention_mask,
                mirna_token_embeddings=mirna_token_embeddings,
                mirna_attention_mask=mirna_attention_mask,
                handcrafted_features=handcrafted_features,
            )
            return self.classifier(features).squeeze(-1)


    class CrossAttentionInteractionModel(FrozenDualEncoderBase):
        def __init__(
            self,
            UtrBertModel,
            RnaFmModel,
            gene_model_name: str,
            mirna_model_name: str,
            interaction_hidden_dim: int,
            classifier_hidden_dim: int,
            attention_heads: int,
            interaction_top_k: int,
            dropout: float,
            interaction_scorer: str = "dot_product",
            feature_processor: str = "interaction_stats",
            handcrafted_feature_dim: int = 0,
        ):
            super().__init__(
                UtrBertModel=UtrBertModel,
                RnaFmModel=RnaFmModel,
                gene_model_name=gene_model_name,
                mirna_model_name=mirna_model_name,
            )
            self.interaction_head = Route1TokenInteractionHead(
                gene_input_dim=int(self.gene_encoder.config.hidden_size),
                mirna_input_dim=int(self.mirna_encoder.config.hidden_size),
                interaction_hidden_dim=interaction_hidden_dim,
                classifier_hidden_dim=classifier_hidden_dim,
                attention_heads=attention_heads,
                interaction_top_k=interaction_top_k,
                dropout=dropout,
                interaction_scorer=interaction_scorer,
                feature_processor=feature_processor,
                handcrafted_feature_dim=handcrafted_feature_dim,
            )

        def forward_features(
            self,
            gene_input_ids: torch.Tensor,
            gene_attention_mask: torch.Tensor,
            mirna_input_ids: torch.Tensor,
            mirna_attention_mask: torch.Tensor,
            gene_token_type_ids: Optional[torch.Tensor] = None,
            mirna_token_type_ids: Optional[torch.Tensor] = None,
            handcrafted_features: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            gene_tokens = self._encode_backbone(
                encoder=self.gene_encoder,
                input_ids=gene_input_ids,
                attention_mask=gene_attention_mask,
                token_type_ids=gene_token_type_ids,
            )
            mirna_tokens = self._encode_backbone(
                encoder=self.mirna_encoder,
                input_ids=mirna_input_ids,
                attention_mask=mirna_attention_mask,
                token_type_ids=mirna_token_type_ids,
            )
            return self.interaction_head.forward_features_from_embeddings(
                gene_token_embeddings=gene_tokens,
                gene_attention_mask=gene_attention_mask,
                mirna_token_embeddings=mirna_tokens,
                mirna_attention_mask=mirna_attention_mask,
                handcrafted_features=handcrafted_features,
            )

        def forward_from_embeddings(
            self,
            gene_token_embeddings: torch.Tensor,
            gene_attention_mask: torch.Tensor,
            mirna_token_embeddings: torch.Tensor,
            mirna_attention_mask: torch.Tensor,
            handcrafted_features: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            return self.interaction_head.forward_from_embeddings(
                gene_token_embeddings=gene_token_embeddings,
                gene_attention_mask=gene_attention_mask,
                mirna_token_embeddings=mirna_token_embeddings,
                mirna_attention_mask=mirna_attention_mask,
                handcrafted_features=handcrafted_features,
            )

        def forward(
            self,
            gene_input_ids: torch.Tensor,
            gene_attention_mask: torch.Tensor,
            mirna_input_ids: torch.Tensor,
            mirna_attention_mask: torch.Tensor,
            gene_token_type_ids: Optional[torch.Tensor] = None,
            mirna_token_type_ids: Optional[torch.Tensor] = None,
            handcrafted_features: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            features = self.forward_features(
                gene_input_ids=gene_input_ids,
                gene_attention_mask=gene_attention_mask,
                mirna_input_ids=mirna_input_ids,
                mirna_attention_mask=mirna_attention_mask,
                gene_token_type_ids=gene_token_type_ids,
                mirna_token_type_ids=mirna_token_type_ids,
                handcrafted_features=handcrafted_features,
            )
            return self.interaction_head.classifier(features).squeeze(-1)


    class HierarchicalCrossAttentionCnnInteractionModel(FrozenDualEncoderBase):
        def __init__(
            self,
            UtrBertModel,
            RnaFmModel,
            gene_model_name: str,
            mirna_model_name: str,
            interaction_hidden_dim: int,
            classifier_hidden_dim: int,
            attention_heads: int,
            interaction_top_k: int,
            dropout: float,
            local_attention_radius: int = 5,
            interaction_cnn_channels: int = 16,
            handcrafted_feature_dim: int = 0,
        ):
            super().__init__(
                UtrBertModel=UtrBertModel,
                RnaFmModel=RnaFmModel,
                gene_model_name=gene_model_name,
                mirna_model_name=mirna_model_name,
            )
            self.interaction_hidden_dim = interaction_hidden_dim
            self.interaction_top_k = interaction_top_k
            self.local_attention_radius = max(0, int(local_attention_radius))
            self.interaction_cnn_channels = max(1, int(interaction_cnn_channels))
            self.handcrafted_feature_dim = max(0, int(handcrafted_feature_dim))

            gene_hidden = int(self.gene_encoder.config.hidden_size)
            mirna_hidden = int(self.mirna_encoder.config.hidden_size)
            heads = max(1, min(attention_heads, interaction_hidden_dim))
            while interaction_hidden_dim % heads != 0 and heads > 1:
                heads -= 1

            self.gene_projection = nn.Linear(gene_hidden, interaction_hidden_dim)
            self.mirna_projection = nn.Linear(mirna_hidden, interaction_hidden_dim)
            self.local_mirna_to_gene_attention = nn.MultiheadAttention(
                embed_dim=interaction_hidden_dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True,
            )
            self.local_gene_to_mirna_attention = nn.MultiheadAttention(
                embed_dim=interaction_hidden_dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True,
            )
            self.global_mirna_to_gene_attention = nn.MultiheadAttention(
                embed_dim=interaction_hidden_dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True,
            )
            self.global_gene_to_mirna_attention = nn.MultiheadAttention(
                embed_dim=interaction_hidden_dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True,
            )
            self.local_mirna_norm = nn.LayerNorm(interaction_hidden_dim)
            self.local_gene_norm = nn.LayerNorm(interaction_hidden_dim)
            self.global_mirna_norm = nn.LayerNorm(interaction_hidden_dim)
            self.global_gene_norm = nn.LayerNorm(interaction_hidden_dim)
            self.interaction_cnn = nn.Sequential(
                nn.Conv2d(1, self.interaction_cnn_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout2d(dropout),
                nn.Conv2d(
                    self.interaction_cnn_channels,
                    self.interaction_cnn_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(),
            )

            self.base_feature_dim = interaction_hidden_dim * 6 + 5 + self.interaction_cnn_channels * 2
            self.feature_dim = self.base_feature_dim + self.handcrafted_feature_dim
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, classifier_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, 1),
            )

        def _local_band_mask(
            self,
            query_len: int,
            key_len: int,
            device: torch.device,
        ) -> torch.Tensor:
            if self.local_attention_radius <= 0:
                return torch.zeros(query_len, key_len, dtype=torch.bool, device=device)
            query_positions = torch.arange(query_len, device=device, dtype=torch.float32)
            key_positions = torch.arange(key_len, device=device, dtype=torch.float32)
            if query_len > 1:
                query_positions = query_positions / float(query_len - 1)
            else:
                query_positions = torch.zeros_like(query_positions)
            if key_len > 1:
                key_positions = key_positions / float(key_len - 1)
            else:
                key_positions = torch.zeros_like(key_positions)
            distance = (query_positions[:, None] - key_positions[None, :]).abs()
            radius = float(self.local_attention_radius) / float(max(query_len, key_len) - 1 or 1)
            return distance > radius

        def _interaction_stats(
            self,
            scores: torch.Tensor,
            pair_mask: torch.Tensor,
            mirna_attention_mask: torch.Tensor,
            gene_attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            valid_scores = scores.masked_fill(~pair_mask, 0.0)
            valid_count = pair_mask.sum(dim=(1, 2)).clamp(min=1).float()
            global_mean = valid_scores.sum(dim=(1, 2)) / valid_count

            masked_scores = scores.masked_fill(~pair_mask, float("-inf"))
            global_max = masked_scores.amax(dim=(1, 2))
            global_max = torch.where(torch.isfinite(global_max), global_max, torch.zeros_like(global_max))

            row_max = masked_scores.max(dim=2).values
            row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
            row_max = row_max * mirna_attention_mask.float()
            row_count = mirna_attention_mask.sum(dim=1).clamp(min=1).float()
            row_max_mean = row_max.sum(dim=1) / row_count

            col_max = masked_scores.max(dim=1).values
            col_max = torch.where(torch.isfinite(col_max), col_max, torch.zeros_like(col_max))
            col_max = col_max * gene_attention_mask.float()
            col_count = gene_attention_mask.sum(dim=1).clamp(min=1).float()
            col_max_mean = col_max.sum(dim=1) / col_count

            flat_scores = masked_scores.view(scores.size(0), -1)
            top_k = max(1, min(self.interaction_top_k, flat_scores.size(1)))
            topk_values = flat_scores.topk(top_k, dim=1).values
            topk_values = torch.where(
                torch.isfinite(topk_values),
                topk_values,
                torch.zeros_like(topk_values),
            )
            valid_topk = pair_mask.view(scores.size(0), -1).sum(dim=1)
            valid_topk = valid_topk.clamp(min=1, max=top_k).float()
            topk_mean = topk_values.sum(dim=1) / valid_topk

            return torch.stack(
                [global_mean, global_max, row_max_mean, col_max_mean, topk_mean],
                dim=-1,
            )

        def _cnn_interaction_features(
            self,
            scores: torch.Tensor,
            pair_mask: torch.Tensor,
        ) -> torch.Tensor:
            masked_scores = scores.masked_fill(~pair_mask, 0.0).unsqueeze(1)
            cnn_map = self.interaction_cnn(masked_scores)
            cnn_mask = pair_mask.unsqueeze(1)
            valid_count = cnn_mask.sum(dim=(2, 3)).clamp(min=1).float()
            cnn_mean = (cnn_map * cnn_mask.float()).sum(dim=(2, 3)) / valid_count
            cnn_max = cnn_map.masked_fill(~cnn_mask, float("-inf")).amax(dim=(2, 3))
            cnn_max = torch.where(torch.isfinite(cnn_max), cnn_max, torch.zeros_like(cnn_max))
            return torch.cat([cnn_mean, cnn_max], dim=-1)

        def forward_features(
            self,
            gene_input_ids: torch.Tensor,
            gene_attention_mask: torch.Tensor,
            mirna_input_ids: torch.Tensor,
            mirna_attention_mask: torch.Tensor,
            gene_token_type_ids: Optional[torch.Tensor] = None,
            mirna_token_type_ids: Optional[torch.Tensor] = None,
            handcrafted_features: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
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
            gene_valid = gene_attention_mask.bool()
            mirna_valid = mirna_attention_mask.bool()
            gene_tokens = gene_tokens * gene_valid.unsqueeze(-1).float()
            mirna_tokens = mirna_tokens * mirna_valid.unsqueeze(-1).float()

            mirna_local, _ = self.local_mirna_to_gene_attention(
                query=mirna_tokens,
                key=gene_tokens,
                value=gene_tokens,
                attn_mask=self._local_band_mask(
                    query_len=mirna_tokens.size(1),
                    key_len=gene_tokens.size(1),
                    device=mirna_tokens.device,
                ),
                need_weights=False,
            )
            gene_local, _ = self.local_gene_to_mirna_attention(
                query=gene_tokens,
                key=mirna_tokens,
                value=mirna_tokens,
                attn_mask=self._local_band_mask(
                    query_len=gene_tokens.size(1),
                    key_len=mirna_tokens.size(1),
                    device=gene_tokens.device,
                ),
                need_weights=False,
            )
            mirna_local = self.local_mirna_norm(mirna_tokens + mirna_local)
            gene_local = self.local_gene_norm(gene_tokens + gene_local)
            mirna_local = mirna_local * mirna_valid.unsqueeze(-1).float()
            gene_local = gene_local * gene_valid.unsqueeze(-1).float()

            mirna_global, _ = self.global_mirna_to_gene_attention(
                query=mirna_local,
                key=gene_local,
                value=gene_local,
                key_padding_mask=~gene_valid,
                need_weights=False,
            )
            gene_global, _ = self.global_gene_to_mirna_attention(
                query=gene_local,
                key=mirna_local,
                value=mirna_local,
                key_padding_mask=~mirna_valid,
                need_weights=False,
            )
            mirna_global = self.global_mirna_norm(mirna_local + mirna_global)
            gene_global = self.global_gene_norm(gene_local + gene_global)
            mirna_global = mirna_global * mirna_valid.unsqueeze(-1).float()
            gene_global = gene_global * gene_valid.unsqueeze(-1).float()

            interaction_scores = torch.matmul(
                mirna_global,
                gene_global.transpose(1, 2),
            ) / math.sqrt(self.interaction_hidden_dim)
            pair_mask = mirna_valid.unsqueeze(-1) & gene_valid.unsqueeze(1)

            features = torch.cat(
                [
                    masked_mean_pooling(gene_tokens, gene_attention_mask),
                    masked_mean_pooling(mirna_tokens, mirna_attention_mask),
                    masked_mean_pooling(gene_global, gene_attention_mask),
                    masked_max_pooling(gene_global, gene_attention_mask),
                    masked_mean_pooling(mirna_global, mirna_attention_mask),
                    masked_max_pooling(mirna_global, mirna_attention_mask),
                    self._interaction_stats(
                        scores=interaction_scores,
                        pair_mask=pair_mask,
                        mirna_attention_mask=mirna_attention_mask,
                        gene_attention_mask=gene_attention_mask,
                    ),
                    self._cnn_interaction_features(
                        scores=interaction_scores,
                        pair_mask=pair_mask,
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
                features = torch.cat([features, handcrafted_features], dim=-1)
            return features

        def forward(
            self,
            gene_input_ids: torch.Tensor,
            gene_attention_mask: torch.Tensor,
            mirna_input_ids: torch.Tensor,
            mirna_attention_mask: torch.Tensor,
            gene_token_type_ids: Optional[torch.Tensor] = None,
            mirna_token_type_ids: Optional[torch.Tensor] = None,
            handcrafted_features: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            features = self.forward_features(
                gene_input_ids=gene_input_ids,
                gene_attention_mask=gene_attention_mask,
                mirna_input_ids=mirna_input_ids,
                mirna_attention_mask=mirna_attention_mask,
                gene_token_type_ids=gene_token_type_ids,
                mirna_token_type_ids=mirna_token_type_ids,
                handcrafted_features=handcrafted_features,
            )
            return self.classifier(features).squeeze(-1)


    class JointSelfAttentionInteractionModel(FrozenDualEncoderBase):
        def __init__(
            self,
            UtrBertModel,
            RnaFmModel,
            gene_model_name: str,
            mirna_model_name: str,
            interaction_hidden_dim: int,
            classifier_hidden_dim: int,
            attention_heads: int,
            interaction_top_k: int,
            dropout: float,
            self_attention_heads: Optional[int] = None,
            self_attention_layers: int = 1,
            modality_embedding: bool = True,
        ):
            super().__init__(
                UtrBertModel=UtrBertModel,
                RnaFmModel=RnaFmModel,
                gene_model_name=gene_model_name,
                mirna_model_name=mirna_model_name,
            )

            self.interaction_hidden_dim = interaction_hidden_dim
            self.interaction_top_k = interaction_top_k
            self.self_attention_layers = max(1, int(self_attention_layers))
            self.use_modality_embedding = bool(modality_embedding)

            gene_hidden = self.gene_encoder.config.hidden_size
            mirna_hidden = self.mirna_encoder.config.hidden_size

            heads = self_attention_heads or attention_heads
            heads = max(1, min(int(heads), interaction_hidden_dim))
            while interaction_hidden_dim % heads != 0 and heads > 1:
                heads -= 1

            self.gene_projection = nn.Linear(gene_hidden, interaction_hidden_dim)
            self.mirna_projection = nn.Linear(mirna_hidden, interaction_hidden_dim)
            self.self_attention_layers_list = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        embed_dim=interaction_hidden_dim,
                        num_heads=heads,
                        dropout=dropout,
                        batch_first=True,
                    )
                    for _ in range(self.self_attention_layers)
                ]
            )
            self.self_attention_norms = nn.ModuleList(
                [nn.LayerNorm(interaction_hidden_dim) for _ in range(self.self_attention_layers)]
            )

            if self.use_modality_embedding:
                self.gene_modality_embedding = nn.Parameter(
                    torch.zeros(1, 1, interaction_hidden_dim)
                )
                self.mirna_modality_embedding = nn.Parameter(
                    torch.zeros(1, 1, interaction_hidden_dim)
                )
            else:
                self.register_parameter("gene_modality_embedding", None)
                self.register_parameter("mirna_modality_embedding", None)

            feature_dim = interaction_hidden_dim * 6 + 5
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, classifier_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, 1),
            )

        def _interaction_stats(
            self,
            scores: torch.Tensor,
            pair_mask: torch.Tensor,
            mirna_attention_mask: torch.Tensor,
            gene_attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            valid_scores = scores.masked_fill(~pair_mask, 0.0)
            valid_count = pair_mask.sum(dim=(1, 2)).clamp(min=1).float()
            global_mean = valid_scores.sum(dim=(1, 2)) / valid_count

            masked_scores = scores.masked_fill(~pair_mask, float("-inf"))
            global_max = masked_scores.amax(dim=(1, 2))
            global_max = torch.where(torch.isfinite(global_max), global_max, torch.zeros_like(global_max))

            row_max = masked_scores.max(dim=2).values
            row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
            row_max = row_max * mirna_attention_mask.float()
            row_count = mirna_attention_mask.sum(dim=1).clamp(min=1).float()
            row_max_mean = row_max.sum(dim=1) / row_count

            col_max = masked_scores.max(dim=1).values
            col_max = torch.where(torch.isfinite(col_max), col_max, torch.zeros_like(col_max))
            col_max = col_max * gene_attention_mask.float()
            col_count = gene_attention_mask.sum(dim=1).clamp(min=1).float()
            col_max_mean = col_max.sum(dim=1) / col_count

            flat_scores = masked_scores.view(scores.size(0), -1)
            top_k = max(1, min(self.interaction_top_k, flat_scores.size(1)))
            topk_values = flat_scores.topk(top_k, dim=1).values
            topk_values = torch.where(
                torch.isfinite(topk_values),
                topk_values,
                torch.zeros_like(topk_values),
            )
            valid_topk = pair_mask.view(scores.size(0), -1).sum(dim=1)
            valid_topk = valid_topk.clamp(min=1, max=top_k).float()
            topk_mean = topk_values.sum(dim=1) / valid_topk

            return torch.stack(
                [global_mean, global_max, row_max_mean, col_max_mean, topk_mean],
                dim=-1,
            )

        def _run_joint_self_attention(
            self,
            gene_tokens: torch.Tensor,
            mirna_tokens: torch.Tensor,
            gene_attention_mask: torch.Tensor,
            mirna_attention_mask: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if self.use_modality_embedding:
                gene_tokens = gene_tokens + self.gene_modality_embedding
                mirna_tokens = mirna_tokens + self.mirna_modality_embedding

            combined_tokens = torch.cat([gene_tokens, mirna_tokens], dim=1)
            combined_attention_mask = torch.cat(
                [gene_attention_mask, mirna_attention_mask],
                dim=1,
            ).bool()
            valid_mask = combined_attention_mask.unsqueeze(-1).float()

            for attention_layer, norm_layer in zip(
                self.self_attention_layers_list,
                self.self_attention_norms,
            ):
                attn_output, _ = attention_layer(
                    query=combined_tokens,
                    key=combined_tokens,
                    value=combined_tokens,
                    key_padding_mask=~combined_attention_mask,
                    need_weights=False,
                )
                combined_tokens = norm_layer(combined_tokens + attn_output)
                combined_tokens = combined_tokens * valid_mask

            gene_length = gene_tokens.size(1)
            gene_self = combined_tokens[:, :gene_length, :]
            mirna_self = combined_tokens[:, gene_length:, :]
            return gene_self, mirna_self

        def forward(
            self,
            gene_input_ids: torch.Tensor,
            gene_attention_mask: torch.Tensor,
            mirna_input_ids: torch.Tensor,
            mirna_attention_mask: torch.Tensor,
            gene_token_type_ids: Optional[torch.Tensor] = None,
            mirna_token_type_ids: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
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
            gene_self, mirna_self = self._run_joint_self_attention(
                gene_tokens=gene_tokens,
                mirna_tokens=mirna_tokens,
                gene_attention_mask=gene_attention_mask,
                mirna_attention_mask=mirna_attention_mask,
            )

            interaction_scores = torch.matmul(
                mirna_self,
                gene_self.transpose(1, 2),
            ) / math.sqrt(self.interaction_hidden_dim)
            pair_mask = (
                mirna_attention_mask.bool().unsqueeze(-1)
                & gene_attention_mask.bool().unsqueeze(1)
            )

            feature_blocks = [
                masked_mean_pooling(gene_tokens, gene_attention_mask),
                masked_mean_pooling(mirna_tokens, mirna_attention_mask),
                masked_mean_pooling(gene_self, gene_attention_mask),
                masked_max_pooling(gene_self, gene_attention_mask),
                masked_mean_pooling(mirna_self, mirna_attention_mask),
                masked_max_pooling(mirna_self, mirna_attention_mask),
                self._interaction_stats(
                    scores=interaction_scores,
                    pair_mask=pair_mask,
                    mirna_attention_mask=mirna_attention_mask,
                    gene_attention_mask=gene_attention_mask,
                ),
            ]
            features = torch.cat(feature_blocks, dim=-1)
            return self.classifier(features).squeeze(-1)


def build_dataloaders(
    df: pd.DataFrame,
    group_key: str,
    batch_size: int,
    eval_batch_size: int,
    collator,
    device: torch.device,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    enable_cuda_prefetch: bool = True,
) -> Dict[str, DataLoader]:
    pin_memory = device.type == "cuda"
    loaders = {}
    split_values = df["split"].to_numpy(copy=False)
    for split_name in ["train", "val", "test"]:
        split_indices = np.flatnonzero(split_values == split_name)
        effective_batch_size = batch_size if split_name == "train" else eval_batch_size
        loader_kwargs: Dict[str, Any] = {
            "batch_size": effective_batch_size,
            "shuffle": split_name == "train",
            "collate_fn": collator,
            "pin_memory": pin_memory,
            "num_workers": num_workers,
        }
        if pin_memory and "pin_memory_device" in inspect.signature(DataLoader).parameters:
            loader_kwargs["pin_memory_device"] = "cuda"
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = prefetch_factor

        loader = DataLoader(
            PairDataset(
                df,
                split_indices,
                group_key=group_key,
                include_sequences=split_name == "test",
            ),
            **loader_kwargs,
        )
        if enable_cuda_prefetch and device.type == "cuda":
            loader = CUDAPrefetchLoader(loader, device=device)
        loaders[split_name] = loader
    return loaders


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=device.type == "cuda")
        else:
            moved[key] = value
    return moved


def run_epoch(
    model,
    loader,
    criterion,
    device: torch.device,
    train: bool,
    optimizer=None,
    scaler=None,
    amp_enabled: bool = False,
    amp_dtype: Optional["torch.dtype"] = None,
    capture_predictions: bool = False,
    group_key: str = "noncodingRNA_fam",
    progress_desc: Optional[str] = None,
    show_progress: bool = True,
) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
    if train and optimizer is None:
        raise ValueError("训练模式下 optimizer 不能为空。")

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_probs: List[float] = []
    all_labels: List[int] = []
    records: List[Dict[str, Any]] = []
    progress = make_progress(
        loader,
        total=len(loader),
        desc=progress_desc,
        disable=not show_progress,
        leave=False,
        unit="batch",
    )
    for step, batch in enumerate(progress, start=1):
        if not getattr(loader, "device_prefetched", False):
            batch = move_batch_to_device(batch, device)
        labels = batch["labels"]

        grad_context = nullcontext() if train else torch.inference_mode()
        with grad_context:
            if amp_enabled:
                with autocast(dtype=amp_dtype):
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
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        probabilities = torch.sigmoid(logits.detach()).cpu().tolist()
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

    metrics = compute_binary_metrics(all_labels, all_probs)
    metrics["loss"] = total_loss / max(len(all_labels), 1)
    prediction_df = pd.DataFrame(records) if capture_predictions else None
    return metrics, prediction_df


def compute_group_error_analysis(prediction_df: pd.DataFrame, group_key: str) -> pd.DataFrame:
    def summarize(group: pd.DataFrame) -> pd.Series:
        labels = group["label"].astype(int).tolist()
        preds = group["pred_label"].astype(int).tolist()
        return pd.Series(
            {
                "row_count": len(group),
                "positive_count": int(group["label"].sum()),
                "negative_count": int(len(group) - group["label"].sum()),
                "positive_rate": float(group["label"].mean()),
                "predicted_positive_rate": float(group["pred_label"].mean()),
                "mean_probability": float(group["probability"].mean()),
                "accuracy": float(accuracy_score(labels, preds)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels, preds)
                )
                if len(set(labels)) > 1
                else float("nan"),
                "mcc": float(matthews_corrcoef(labels, preds))
                if len(set(labels)) > 1
                else float("nan"),
            }
        )

    analysis = prediction_df.groupby(group_key).apply(summarize).reset_index()
    return analysis.sort_values(["row_count", "accuracy"], ascending=[False, True]).reset_index(drop=True)


def train_one_model(
    model_name: str,
    model,
    loaders: Dict[str, DataLoader],
    config: RunConfig,
    device: torch.device,
    dataset_dir: Path,
    group_key: str,
) -> Dict[str, Any]:
    model_dir = dataset_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    optimizer = build_adamw_optimizer(
        model=model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=device,
        use_fused=config.fused_optimizer,
    )
    criterion = nn.BCEWithLogitsLoss()
    amp_enabled = device.type == "cuda"
    amp_dtype = resolve_amp_dtype(config.amp_dtype, device)
    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    best_state = extract_trainable_state_dict(model)
    best_epoch = 0
    best_score = float("-inf")
    best_val_metrics: Optional[Dict[str, float]] = None
    history: List[Dict[str, Any]] = []
    patience_used = 0
    max_epochs = 1 if config.smoke_test else config.max_epochs
    early_stopping_patience = 1 if config.smoke_test else max(5, config.early_stopping_patience)

    for epoch in range(1, max_epochs + 1):
        train_metrics, _ = run_epoch(
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
        val_metrics, _ = run_epoch(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
            train=False,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            group_key=group_key,
            progress_desc=f"{model_name} val e{epoch:02d}",
            show_progress=config.progress_bar,
        )

        history_row = {"epoch": epoch}
        history_row.update({f"train_{key}": value for key, value in train_metrics.items()})
        history_row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(history_row)

        monitor_score = val_metrics["auroc"]
        if math.isnan(monitor_score):
            monitor_score = float("-inf")

        print(
            f"[{model_name}] Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} train_auroc={train_metrics['auroc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_auroc={val_metrics['auroc']:.4f}"
        )

        if monitor_score > best_score:
            best_score = monitor_score
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_state = extract_trainable_state_dict(model)
            patience_used = 0
        else:
            patience_used += 1
            if patience_used >= early_stopping_patience:
                print(
                    f"[{model_name}] Early stopping at epoch {epoch:02d} "
                    f"(best epoch = {best_epoch:02d})."
                )
                break

    load_trainable_state_dict(model, best_state)
    test_metrics, prediction_df = run_epoch(
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

    if prediction_df is None:
        raise RuntimeError("测试阶段未返回预测结果。")

    history_df = pd.DataFrame(history)
    history_df.to_csv(model_dir / "history.tsv", sep="\t", index=False)

    prediction_df.to_csv(model_dir / "test_predictions.tsv", sep="\t", index=False)
    group_analysis_df = compute_group_error_analysis(prediction_df, group_key=group_key)
    group_analysis_df.to_csv(model_dir / f"{group_key}_analysis.tsv", sep="\t", index=False)

    metrics_payload = {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(model_dir / "metrics.json", metrics_payload)
    pd.DataFrame(
        [
            {
                "model_name": model_name,
                "best_epoch": best_epoch,
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        ]
    ).to_csv(model_dir / "metrics.csv", index=False)

    checkpoint_payload = {
        "model_name": model_name,
        "input_file": config.input_file,
        "group_key": group_key,
        "config": asdict(config),
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "state_dict_kind": "trainable_only",
        "state_dict": extract_trainable_state_dict(model),
    }
    safe_torch_save(checkpoint_payload, model_dir / "best_checkpoint.pt")

    return {
        "model_name": model_name,
        "best_epoch": best_epoch,
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }


def run_supervised_poc(config: RunConfig) -> None:
    ensure_compatible_environment()
    RnaTokenizer, UtrBertModel, RnaFmModel = check_and_import_multimolecule()

    device = resolve_device(config.device)
    configure_runtime_for_device(device)
    model_names = parse_model_names(config.models)
    effective_num_workers = resolve_num_workers(config.num_workers, device)
    effective_eval_batch_size = resolve_eval_batch_size(config.batch_size, config.eval_batch_size)
    effective_early_stopping_patience = (
        1 if config.smoke_test else max(5, config.early_stopping_patience)
    )
    resolved_amp_dtype = resolve_amp_dtype(config.amp_dtype, device)
    print(f"[Info] device = {device}")
    print(f"[Info] num_workers = {effective_num_workers}")
    print(f"[Info] eval_batch_size = {effective_eval_batch_size}")
    print(f"[Info] early_stopping_patience = {effective_early_stopping_patience}")
    print(f"[Info] cuda_prefetch = {config.cuda_prefetch}")
    if resolved_amp_dtype is not None:
        print(f"[Info] amp dtype = {resolved_amp_dtype}")
    gene_model_source = resolve_model_source(config.gene_model_name, "utrbert-3mer")
    mirna_model_source = resolve_model_source(config.mirna_model_name, "rnafm")

    prepared = load_filtered_dataframe(config)
    dataset_name = Path(config.input_file).stem
    dataset_dir = Path(config.output_dir) / dataset_name
    if config.smoke_test:
        dataset_dir = dataset_dir / "smoke_test"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    save_json(dataset_dir / "data_summary.json", prepared.summary)
    print(
        f"[Info] retained_rows = {prepared.summary['retained_rows']}, "
        f"group_count = {prepared.summary['group_count']}, "
        f"positive_rate = {prepared.summary['positive_rate']:.4f}"
    )

    split_df, split_manifest = greedy_group_split(
        df=prepared.dataframe,
        group_key=config.group_key,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )
    if config.smoke_test:
        split_df, smoke_summary = build_smoke_test_subset(
            df=split_df,
            seed=config.seed,
            split_row_targets={
                "train": config.smoke_train_rows,
                "val": config.smoke_val_rows,
                "test": config.smoke_test_rows,
            },
        )
        save_json(dataset_dir / "smoke_test_summary.json", smoke_summary)
        print(
            "[Info] smoke-test rows = "
            f"train:{smoke_summary['train']['sampled_rows']} "
            f"val:{smoke_summary['val']['sampled_rows']} "
            f"test:{smoke_summary['test']['sampled_rows']}"
        )
    split_df.drop(columns=["gene_seq_id", "mirna_seq_id"], errors="ignore").to_csv(
        dataset_dir / "split_assignments.tsv",
        sep="\t",
        index=False,
    )
    save_json(dataset_dir / "split_manifest.json", split_manifest)
    save_json(
        dataset_dir / "run_config.json",
        {
            **asdict(config),
            "resolved_device": str(device),
            "resolved_gene_model_name": gene_model_source,
            "resolved_mirna_model_name": mirna_model_source,
        },
    )

    gene_tokenizer = RnaTokenizer.from_pretrained(gene_model_source)
    mirna_tokenizer = RnaTokenizer.from_pretrained(mirna_model_source)
    gene_cache, mirna_cache = build_dual_token_caches(
        df=split_df,
        gene_tokenizer=gene_tokenizer,
        mirna_tokenizer=mirna_tokenizer,
        max_gene_len=config.max_gene_len,
        max_mirna_len=config.max_mirna_len,
        batch_size=config.token_cache_batch_size,
        enabled=config.cache_tokenized_sequences,
        show_progress=config.progress_bar,
    )
    collator = DualTokenizerCollator(
        gene_tokenizer=gene_tokenizer,
        mirna_tokenizer=mirna_tokenizer,
        max_gene_len=config.max_gene_len,
        max_mirna_len=config.max_mirna_len,
        gene_cache=gene_cache,
        mirna_cache=mirna_cache,
    )
    loaders = build_dataloaders(
        df=split_df,
        group_key=config.group_key,
        batch_size=config.batch_size,
        eval_batch_size=effective_eval_batch_size,
        collator=collator,
        device=device,
        num_workers=effective_num_workers,
        prefetch_factor=config.prefetch_factor,
        enable_cuda_prefetch=config.cuda_prefetch,
    )

    model_summaries = []
    for model_name in model_names:
        if model_name == "baseline_pooled":
            model = PooledBaselineModel(
                UtrBertModel=UtrBertModel,
                RnaFmModel=RnaFmModel,
                gene_model_name=gene_model_source,
                mirna_model_name=mirna_model_source,
                classifier_hidden_dim=config.classifier_hidden_dim,
                dropout=config.dropout,
            )
        else:
            model = CrossAttentionInteractionModel(
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
        model_summaries.append(
            train_one_model(
                model_name=model_name,
                model=model,
                loaders=loaders,
                config=config,
                device=device,
                dataset_dir=dataset_dir,
                group_key=config.group_key,
            )
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    comparison_df = pd.DataFrame(model_summaries)
    comparison_df.to_csv(dataset_dir / "model_comparison.csv", index=False)
    save_json(
        dataset_dir / "model_comparison.json",
        {"models": model_summaries},
    )


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="监督式 PoC：Cross-Attention + Interaction Matrix 预测 microRNA target interaction"
    )
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", default="baseline_pooled,cross_attention_interaction")
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
    args = parser.parse_args()
    config_kwargs = vars(args)
    config_kwargs["cache_tokenized_sequences"] = not config_kwargs.pop("disable_token_cache")
    config_kwargs["progress_bar"] = not config_kwargs.pop("disable_progress_bar")
    config_kwargs["cuda_prefetch"] = not config_kwargs.pop("disable_cuda_prefetch")
    config_kwargs["fused_optimizer"] = not config_kwargs.pop("disable_fused_optimizer")
    return RunConfig(**config_kwargs)


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    run_supervised_poc(config)


if __name__ == "__main__":
    main()
