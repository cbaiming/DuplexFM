"""Inspect the frozen architecture's 192D interface with synthetic CPU inputs.

This uses a randomly initialized PairACFusion and synthetic projected Q/K and
reactivity values. It does not load pretrained encoders or learned task weights,
and its output is not a biologically validated embedding or a prediction.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "1")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reference" / "pairac_concat_v1" / "source_full" / "code"
sys.path.insert(0, str(SOURCE))

import torch

from duplexfm_mechanism.model import PairACFusion
from duplexfm_mechanism.protocol import source_config


def synthetic_inputs(seed: int = 42, batch_size: int = 2) -> dict:
    """Build guide19/target57 tensors, with target centered in a 60-base array."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    guide = torch.randint(0, 4, (batch_size, 19), generator=rng)
    target = torch.full((batch_size, 60), 4, dtype=torch.long)
    target[:, 1:58] = torch.randint(0, 4, (batch_size, 57), generator=rng)
    # Token IDs A,C,G,U = 0,1,2,3. The toy design core is complementary.
    target[:, 20:39] = 3 - guide.flip(1)
    guide_pos = torch.arange(19).expand(batch_size, -1).clone()
    target_pos = torch.full((batch_size, 60), -1, dtype=torch.long)
    target_pos[:, 1:58] = torch.arange(57)
    target_mask = target_pos >= 0
    return {
        "guide": guide,
        "target": target,
        "guide_pos": guide_pos,
        "target_pos": target_pos,
        "guide_mask": torch.ones_like(guide, dtype=torch.bool),
        "target_mask": target_mask,
        "guide_lengths": torch.full((batch_size,), 19, dtype=torch.long),
        "target_lengths": torch.full((batch_size,), 57, dtype=torch.long),
        "accessibility": torch.rand((batch_size, 60), generator=rng).masked_fill(~target_mask, 0),
        "q": torch.randn((batch_size, 60, 256), generator=rng),
        "k": torch.randn((batch_size, 19, 256), generator=rng),
    }


def synthetic_embedding(seed: int = 42, batch_size: int = 2):
    """Return toy inputs, fusion module and the design site's pooled tensor.

    The registered inference interface is PairACModel.source192(batch, 19).
    Here we call the same fusion/pooling implementation directly because no
    pretrained encoder or source checkpoint is included in the example.
    """
    batch = synthetic_inputs(seed, batch_size)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        # source_config supplies architecture values; Fusion does not read assets.
        fusion = PairACFusion(source_config(Path("synthetic_assets_unused"))).cpu().eval()
    with torch.inference_mode():
        grid, valid = fusion.grid_features(batch)
        sites, allowed = fusion.site_pool(grid, valid, batch)
        # guide position 0 aligns to target position 37 (core 19:38).
        # Site indices use padded tensor coordinates, not original coordinates.
        guide_index = (batch["guide_pos"] == 0).long().argmax(1)
        target_index = (batch["target_pos"] == 37).long().argmax(1)
        site_index = guide_index + target_index
        if not allowed.gather(1, site_index[:, None]).all():
            raise ValueError("Synthetic design site is not valid")
        embedding = sites.gather(1, site_index[:, None, None].expand(-1, 1, 192))[:, 0]
    return batch, fusion, embedding


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(1)
    _, fusion, embedding = synthetic_embedding(args.seed, args.batch_size)
    print(json.dumps({
        "example": "synthetic_design_site_embedding",
        "model_state": "randomly_initialized_fusion; no_pretrained_weights",
        "inputs": "synthetic_q_k_and_reactivity; guide19_target57",
        "input_channels": fusion.input_channels,
        "embedding_shape": list(embedding.shape),
        "finite": bool(torch.isfinite(embedding).all()),
        "device": str(embedding.device),
        "seed": args.seed,
        "biological_performance_validated": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
