from __future__ import annotations

"""Compatibility helpers for interchangeable gene-sequence backbones.

The original Route2 code assumes a MultiMolecule UTRBERT model and tokenizer.
mRNABERT is a Hugging Face remote-code model with two important differences:

* its base model returns a tuple instead of ``last_hidden_state``;
* its BERT tokenizer expects whitespace-separated nucleotide/codon tokens.

miRBench supplies a short local target-site window without CDS boundaries or a
reading frame.  Therefore the scientifically well-defined representation here
is nucleotide-level tokenization for every region (``A T G ...``), not an
inferred codon tokenization.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


MRNABERT_MODEL_ID = "YYLY66/mRNABERT"
MRNABERT_TOKENIZATION_MODE = "all_region_nucleotide_v1"
MRNABERT_LOCAL_DIR_NAMES = ("mrnabert", "mRNABERT")


def _read_config(source: str | Path) -> dict[str, Any]:
    path = Path(str(source)) / "config.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def is_mrnabert_source(source: str | Path) -> bool:
    normalized = str(source).replace("\\", "/").lower()
    if "mrnabert" in normalized:
        return True
    auto_map = _read_config(source).get("auto_map", {})
    if not isinstance(auto_map, Mapping):
        return False
    auto_model = str(auto_map.get("AutoModel", "")).lower()
    return auto_model.endswith("bert_layers.bertmodel")


def resolve_gene_model_source(model_name: str) -> str:
    """Resolve only the local directory belonging to the requested model.

    The legacy helper always preferred ``model/utrbert-3mer`` even when a
    different model name was requested.  That would silently turn an intended
    mRNABERT experiment back into UTRBERT.
    """

    explicit = Path(str(model_name))
    if explicit.is_dir():
        return str(explicit)
    if is_mrnabert_source(model_name):
        for directory_name in MRNABERT_LOCAL_DIR_NAMES:
            local = Path("model") / directory_name
            if local.is_dir():
                print(f"[Info] Using local mRNABERT directory: {local}")
                return str(local)
        return str(model_name)
    local_utrbert = Path("model") / "utrbert-3mer"
    if local_utrbert.is_dir():
        print(f"[Info] Using local UTRBERT directory: {local_utrbert}")
        return str(local_utrbert)
    return str(model_name)


def prepare_mrnabert_nucleotide_sequence(sequence: Any) -> str:
    """Convert one miRBench target-site window to mRNABERT nucleotide tokens."""

    raw = re.sub(r"\s+", "", str(sequence).strip().upper()).replace("U", "T")
    normalized = "".join(base if base in "ATCGN" else "N" for base in raw)
    return " ".join(normalized)


class MRNABertNucleotideTokenizer:
    """Thin adapter that preserves the Hugging Face tokenizer interface."""

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer
        base_name = str(
            getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
        )
        self.name_or_path = f"{base_name}::{MRNABERT_TOKENIZATION_MODE}"
        self.tokenization_mode = MRNABERT_TOKENIZATION_MODE

    @staticmethod
    def _prepare(text: Any) -> Any:
        if isinstance(text, str):
            return prepare_mrnabert_nucleotide_sequence(text)
        if isinstance(text, Sequence):
            return [prepare_mrnabert_nucleotide_sequence(item) for item in text]
        return prepare_mrnabert_nucleotide_sequence(text)

    def __call__(self, text: Any, *args: Any, **kwargs: Any) -> Any:
        return self._tokenizer(self._prepare(text), *args, **kwargs)

    def batch_encode_plus(self, batch_text_or_text_pairs: Any, *args: Any, **kwargs: Any) -> Any:
        prepared = self._prepare(batch_text_or_text_pairs)
        return self._tokenizer.batch_encode_plus(prepared, *args, **kwargs)

    def __len__(self) -> int:
        return len(self._tokenizer)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)


def load_gene_tokenizer(source: str, legacy_tokenizer_class: Any) -> Any:
    if not is_mrnabert_source(source):
        return legacy_tokenizer_class.from_pretrained(source)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("mRNABERT requires transformers.AutoTokenizer") from exc
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    return MRNABertNucleotideTokenizer(tokenizer)


def load_rna_tokenizer(source: str, legacy_tokenizer_class: Any) -> Any:
    """Load a tokenizer while keeping the existing RNA-FM path unchanged."""

    if is_mrnabert_source(source):
        return load_gene_tokenizer(source, legacy_tokenizer_class)
    return legacy_tokenizer_class.from_pretrained(source)


def _disable_legacy_triton_attention(model: Any) -> int:
    """Use the stable PyTorch attention path for the bundled 2022 Triton kernel.

    The custom kernel predates Blackwell.  It can be explicitly re-enabled with
    ``MRNABERT_ENABLE_LEGACY_TRITON=1`` after a separate validation.
    """

    enabled = os.environ.get("MRNABERT_ENABLE_LEGACY_TRITON", "0").strip().lower()
    if enabled in {"1", "true", "yes", "on"}:
        return 0
    patched = 0
    for module in model.modules():
        forward = getattr(module, "forward", None)
        globals_dict = getattr(forward, "__globals__", None)
        if not isinstance(globals_dict, dict):
            continue
        if "flash_attn_qkvpacked_func" in globals_dict:
            globals_dict["flash_attn_qkvpacked_func"] = None
            patched += 1
    setattr(model, "_mrnabert_legacy_triton_disabled", bool(patched))
    return patched


def _load_mrnabert_cpu_first(source: str, config: Any, auto_model_class: Any) -> Any:
    """Instantiate custom mRNABERT on CPU before loading its state dict.

    Transformers 5 initializes every ``from_pretrained`` model on the meta
    device.  mRNABERT's remote ``BertEncoder.__init__`` immediately constructs
    a persistent ALiBi tensor and is not meta-safe.  ``from_config`` retains
    ordinary CPU initialization, after which the official checkpoint can be
    loaded without changing any parameter names or values.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("mRNABERT requires PyTorch") from exc
    model = auto_model_class.from_config(config, trust_remote_code=True)
    local_weight = Path(source) / "pytorch_model.bin"
    if not local_weight.is_file():
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError("Remote mRNABERT loading requires huggingface_hub") from exc
        local_weight = Path(hf_hub_download(repo_id=source, filename="pytorch_model.bin"))
    state_dict = torch.load(local_weight, map_location="cpu", weights_only=True)
    if isinstance(state_dict, Mapping) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise RuntimeError(f"Unexpected mRNABERT checkpoint payload: {type(state_dict).__name__}")
    unknown_checkpoint_keys = [
        key for key in state_dict if not (str(key).startswith("bert.") or str(key).startswith("cls."))
    ]
    if unknown_checkpoint_keys:
        raise RuntimeError(f"Unexpected mRNABERT checkpoint key prefix: {unknown_checkpoint_keys[:8]}")
    encoder_state = {
        str(key)[len("bert.") :]: value
        for key, value in state_dict.items()
        if str(key).startswith("bert.")
    }
    incompatible = model.load_state_dict(encoder_state, strict=False)
    allowed_missing = {"pooler.dense.weight", "pooler.dense.bias"}
    disallowed_missing = set(incompatible.missing_keys) - allowed_missing
    if disallowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "mRNABERT checkpoint mismatch: "
            f"missing={sorted(disallowed_missing)}, unexpected={incompatible.unexpected_keys}"
        )
    return model


def load_gene_model(source: str, legacy_model_class: Any) -> Any:
    if not is_mrnabert_source(source):
        return legacy_model_class.from_pretrained(source)
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("mRNABERT requires transformers AutoConfig/AutoModel") from exc
    config = AutoConfig.from_pretrained(source, trust_remote_code=True)
    model = _load_mrnabert_cpu_first(source, config, AutoModel)
    patched = _disable_legacy_triton_attention(model)
    if patched:
        print(
            "[Info] Disabled mRNABERT's legacy Triton attention kernel; "
            "using the stable PyTorch path."
        )
    return model


def extract_last_hidden_state(outputs: Any) -> Any:
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    if isinstance(outputs, Mapping) and "last_hidden_state" in outputs:
        return outputs["last_hidden_state"]
    if isinstance(outputs, (tuple, list)) and outputs:
        return outputs[0]
    try:
        return outputs[0]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Unsupported backbone output type: {type(outputs).__name__}"
        ) from exc


def describe_gene_backbone(
    source: str,
    *,
    tokenizer: Optional[Any] = None,
    model: Optional[Any] = None,
) -> dict[str, Any]:
    mrnabert = is_mrnabert_source(source)
    payload: dict[str, Any] = {
        "source": str(source),
        "family": "mrnabert" if mrnabert else "utrbert",
        "tokenization_mode": (
            MRNABERT_TOKENIZATION_MODE if mrnabert else "legacy_rna_tokenizer"
        ),
        "legacy_triton_attention_enabled": bool(
            mrnabert
            and os.environ.get("MRNABERT_ENABLE_LEGACY_TRITON", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        ),
    }
    if tokenizer is not None:
        payload["tokenizer_class"] = tokenizer.__class__.__name__
        payload["tokenizer_name_or_path"] = str(
            getattr(tokenizer, "name_or_path", "")
        )
    if model is not None:
        payload["model_class"] = model.__class__.__name__
        payload["hidden_size"] = int(getattr(model.config, "hidden_size", 0))
    return payload
