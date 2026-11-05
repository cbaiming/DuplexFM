"""Public foundation encoders + fresh LoRA + live, trainable Q/K projections.

No DuplexFM task checkpoint is read here. Frozen accessibility is a separately
provenanced input profile, never a shared instance of the trainable encoder.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F




def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


class LoRALinear(nn.Module):
    """Standard frozen W + (alpha/r) BA; B=0 gives exact base initialization."""
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank <= 0:
            raise ValueError("LoRA requires a linear layer and positive rank")
        self.base = base.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = float(alpha) / rank
        self.dropout = nn.Dropout(dropout)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scale


def inject_lora(model, suffixes, rank, alpha, dropout):
    model.requires_grad_(False)
    names = [name for name, layer in model.named_modules()
             if isinstance(layer, nn.Linear) and name.rsplit(".", 1)[-1] in suffixes]
    if not names:
        raise RuntimeError(f"No LoRA targets found for {suffixes}")
    for name in names:
        parent, _, leaf = name.rpartition(".")
        owner = model.get_submodule(parent) if parent else model
        setattr(owner, leaf, LoRALinear(getattr(owner, leaf), rank, alpha, dropout))
    return names


def _source_manifest(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Copy the foundation model directory first: {root}")
    if (root / "adapter_config.json").exists():
        raise ValueError("Expected a foundation-model directory, not an old LoRA adapter")
    files = {p.name: sha256(p) for p in sorted(root.iterdir()) if p.is_file()
             and p.suffix in {".json", ".txt", ".py", ".bin", ".safetensors"}}
    if not any(x.endswith((".bin", ".safetensors")) for x in files):
        raise ValueError(f"Missing foundation weights: {root}")
    return {"path": str(root), "sha256": files}


def load_target(root):
    """CPU-first load is needed by mRNABERT's non-meta-safe ALiBi constructor."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    root = str(Path(root).resolve())
    config = AutoConfig.from_pretrained(root, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_config(config, trust_remote_code=True)
    state = torch.load(Path(root) / "pytorch_model.bin", map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    if any(not k.startswith(("bert.", "cls.")) for k in state):
        raise ValueError("Unexpected mRNABERT base checkpoint keys")
    state = {k[5:]: v for k, v in state.items() if k.startswith("bert.")}
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) - {"pooler.dense.weight", "pooler.dense.bias"} or result.unexpected_keys:
        raise RuntimeError(f"mRNABERT checkpoint mismatch: {result}")
    # The bundled 2022 Triton kernel is not required. Use its PyTorch fallback on
    # both A6000 and H100, rather than a hardware-dependent unvalidated kernel.
    for module in model.modules():
        scope = getattr(getattr(module, "forward", None), "__globals__", {})
        if "flash_attn_qkvpacked_func" in scope:
            scope["flash_attn_qkvpacked_func"] = None
    tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=True, local_files_only=True)
    return model, tokenizer


def load_guide(root):
    from multimolecule import RnaFmModel, RnaTokenizer
    from multimolecule.models.rnafm.configuration_rnafm import RnaFmConfig
    root = str(Path(root).resolve())
    # multimolecule 0.2 validates a newer 28-token vocabulary. Our published
    # foundation weights/tokenizer contain 26 tokens; never resize/randomize it.
    payload = json.loads((Path(root) / "config.json").read_text())
    expected_vocab = payload["vocab_size"]
    if expected_vocab == 26:
        payload["vocab_size"] = 28
    config = RnaFmConfig(**payload)
    config.vocab_size = expected_vocab
    model, info = RnaFmModel.from_pretrained(root, config=config, local_files_only=True,
                                           output_loading_info=True)
    # Only an unused pooler may be absent; reject silently random encoder weights.
    missing = [k for k in info.get("missing_keys", []) if not k.startswith("pooler.")]
    unexpected = [k for k in info.get("unexpected_keys", [])
                  if not k.startswith(("lm_head.", "ss_head."))]
    if missing or unexpected or info.get("mismatched_keys") or info.get("error_msgs"):
        raise RuntimeError(f"RNA-FM checkpoint mismatch: {info}")
    tokenizer = RnaTokenizer.from_pretrained(root, local_files_only=True)
    if len(tokenizer) != expected_vocab:
        raise ValueError(f"RNA-FM vocabulary mismatch: {len(tokenizer)} != {expected_vocab}")
    return model, tokenizer


class JointModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.base_provenance = {
            "target": _source_manifest(config["target_model"]),
            "guide": _source_manifest(config["guide_model"]),
            "initialization": "public_foundation_weights_plus_fresh_zero_B_LoRA",
            "old_DuplexFM_task_checkpoint_loaded": False,
            "accessibility": "separate_frozen_input_cache; no shared trainable weights",
            "cross": "new_QK_score_projections; no_unused_V_branch",
        }
        reference_path = Path(__file__).resolve().parents[1] / "foundation_reference.json"
        if reference_path.is_file():
            reference = json.loads(reference_path.read_text())
            for kind in ("target", "guide"):
                for filename, expected in reference[kind]["sha256"].items():
                    if self.base_provenance[kind]["sha256"].get(filename) != expected:
                        raise ValueError(f"Foundation reference hash mismatch: {kind}/{filename}")
            self.base_provenance["foundation_reference_verified"] = True
        else:
            raise FileNotFoundError("foundation_reference.json is required to rule out modified/merged task weights")
        # Isolate ALL RNG streams: MultiMolecule 0.2's first import executes
        # torch.manual_seed(1016), including CUDA. CPU-only fork_rng would make
        # a fresh process and repeated in-process loading use different dropout
        # streams despite identical task seeds and identical parameter values.
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                target, self.target_tokenizer = load_target(config["target_model"])
                guide, self.guide_tokenizer = load_guide(config["guide_model"])
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        rank, alpha = int(config.get("lora_rank", 8)), float(config.get("lora_alpha", 16))
        dropout = float(config.get("lora_dropout", .05))
        self.base_provenance["target_lora_modules"] = inject_lora(target, {"Wqkv"}, rank, alpha, dropout)
        self.base_provenance["guide_lora_modules"] = inject_lora(guide, {"query", "key", "value"}, rank, alpha, dropout)
        self.target_encoder, self.guide_encoder = target, guide
        dim = int(config.get("projection_dim", 256))
        self.target_projection = nn.Sequential(nn.Linear(target.config.hidden_size, dim), nn.LayerNorm(dim), nn.GELU())
        self.guide_projection = nn.Sequential(nn.Linear(guide.config.hidden_size, dim), nn.LayerNorm(dim), nn.GELU())
        self.q_projection, self.k_projection = nn.Linear(dim, dim), nn.Linear(dim, dim)
        from duplexfm_mechanism.model import PairACFusion
        self.fusion = PairACFusion(config)
        if dim % self.fusion.heads:
            raise ValueError("projection_dim must be divisible by heads")

    def _encode(self, sequences, target):
        if any(not s or set(s) - set("ACGU") for s in sequences):
            raise ValueError("Encoder input must be a real, nonempty ACGU sequence, never artificial PAD/N/X")
        tokenizer = self.target_tokenizer if target else self.guide_tokenizer
        texts = [" ".join(s.replace("U", "T")) for s in sequences] if target else sequences
        tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=False,
                           return_special_tokens_mask=True)
        observed = tokens["attention_mask"].bool() & ~tokens.pop("special_tokens_mask").bool()
        if observed.sum(1).tolist() != list(map(len, sequences)):
            raise ValueError("Foundation tokenizer must produce exactly one token per nucleotide")
        if ((tokens["input_ids"] == tokenizer.unk_token_id) & observed).any():
            raise ValueError("Unexpected unknown nucleotide token")
        device = next(self.parameters()).device
        kwargs = {k: v.to(device) for k, v in tokens.items()
                  if k in {"input_ids", "attention_mask", "token_type_ids"}}
        encoder = self.target_encoder if target else self.guide_encoder
        if not target:
            kwargs.pop("token_type_ids", None)
        output = encoder(**kwargs)
        hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        projection = self.target_projection if target else self.guide_projection
        qk = self.q_projection if target else self.k_projection
        hidden = qk(projection(hidden))
        return [hidden[i, mask.to(device)] for i, mask in enumerate(observed)]

    def interaction_inputs(self, batch):
        """Encode real sequences; differentiably scatter into padded map coordinates."""
        ts = self._encode(batch["target_sequences"], True)
        gs = self._encode(batch["guide_sequences"], False)
        q, k = [], []
        target_width, guide_width = batch["target"].shape[1], batch["guide"].shape[1]
        for i, (t, g) in enumerate(zip(ts, gs)):
            ti = batch["target_mask"][i].nonzero(as_tuple=True)[0].to(t.device)
            gi = batch["guide_mask"][i].nonzero(as_tuple=True)[0].to(g.device)
            if len(ti) != len(t) or len(gi) != len(g):
                raise ValueError("Observed sequence and fusion mask lengths disagree")
            q.append(t.new_zeros((target_width, t.shape[-1])).index_copy(0, ti, t))
            k.append(g.new_zeros((guide_width, g.shape[-1])).index_copy(0, gi, g))
        return {**batch, "q": torch.stack(q), "k": torch.stack(k)}

    def forward_features(self, batch):
        return self.fusion.forward_features(self.interaction_inputs(batch))

    def forward(self, batch):
        return self.fusion(self.interaction_inputs(batch))
