"""CPU checks for the public synthetic example; no trained model is evaluated."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "synthetic_embedding.py"
spec = importlib.util.spec_from_file_location("public_synthetic_embedding", EXAMPLE)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)
torch = demo.torch
torch.set_num_threads(1)


def test_design_site_matches_manual_antiparallel_pooling():
    batch, fusion, embedding = demo.synthetic_embedding(seed=42, batch_size=2)
    with torch.inference_mode():
        grid, valid = fusion.grid_features(batch)
        expected = []
        for row in range(2):
            # Independent coordinate lookup avoids the candidate-index helper.
            values = []
            for guide_pos in range(19):
                gi = torch.where(batch["guide_pos"][row] == guide_pos)[0].item()
                ti = torch.where(batch["target_pos"][row] == 37 - guide_pos)[0].item()
                assert valid[row, gi, ti]
                values.append(grid[row, :, gi, ti])
            values = torch.stack(values, 1)
            parts = []
            for positions in (range(1, 8), range(8, 12), (0, *range(12, 19))):
                region = values[:, list(positions)]
                parts.extend((region.mean(1), region.amax(1)))
            expected.append(torch.cat(parts))
    assert embedding.shape == (2, 192)
    assert torch.isfinite(embedding).all()
    torch.testing.assert_close(embedding, torch.stack(expected), atol=1e-6, rtol=1e-5)


def test_cli_runs_outside_checkout_and_labels_synthetic_output(tmp_path):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), "--batch-size", "1"],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True, timeout=60,
    )
    payload = json.loads(result.stdout)
    assert payload["embedding_shape"] == [1, 192]
    assert payload["input_channels"] == 16
    assert payload["device"] == "cpu"
    assert payload["finite"] is True
    assert payload["biological_performance_validated"] is False
    assert "no_pretrained_weights" in payload["model_state"]
