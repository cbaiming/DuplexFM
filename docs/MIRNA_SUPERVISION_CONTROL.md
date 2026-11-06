# Matched control without miRNA task supervision

[Completed results](../results/siRNA_oligo_random/mirna_supervision_control/REPORT.md)
 · [New-file guide](MIRNA_CONTROL_FILES.md)

This experiment compares each exact saved pre-update state (`initial`, step 0)
with its corresponding fixed terminal state (`trained`, step 32,728), for source
seeds 42, 43 and 44. It asks whether miRNA task training improves the frozen
interaction representation used to predict siRNA efficacy.

Both states retain the same public RNA-FM and mRNABERT weights, independently
supervised structural-reactivity predictor, architecture, input sequences,
designed-site pooling and source192 interface. The initial state has fresh task
LoRA, projections and interaction parameters. It is **without miRNA task
supervision**, not without all pretraining or all supervision. Both frozen
encoders receive the same supervised siRNA readout.

## Fixed comparison

The input/split SHA values are those in `MODEL_REGISTRY.json`: 3,688 unique
inputs, 26 outer groups and 4,157 held-out predictions per state/seed/mode.
Huesken, Mixset, Takayuki and Huesken-to-Mixset are the four main settings;
Wang and ALAS1 are reported separately as extensions.

| Mode | Readout inputs | Purpose |
| --- | --- | --- |
| `source_only` | Frozen source192; zero TD columns | Information in the interaction representation |
| `source_plus_td` | Frozen source192 + TD24 | Contribution in the principal prediction pipeline |
| `TD24` | TD24; zero source columns | Deterministic baseline, fitted once |

All modes use R55, the fixed five inner folds and alpha candidates 0.1, 1, 10,
100 and 1000. Scaling, target25 groups, alpha selection and final head fitting
use only the relevant training partition. Source scaling, nuclear penalty and
group-intercept treatment remain unchanged. Trained predictions must reproduce
the saved main results within absolute tolerance 1e-7.

No architectures, modes, seeds, alpha grids or source checkpoints are selected
using the new outer-test comparison.

## Feature export

`scripts/export_mirna_supervision_control.py` loads the original source code,
separately supplied checkpoints and public backbones. A private JSON config
provides `assets`, `checkpoint_registry` and `stores` with verified input hashes.
Keep local asset/SSH paths out of the public repository.

Original checkpoint SHA values must match `MODEL_REGISTRY.json`. Compact files
require a verified compact-file SHA, original-checkpoint binding, correct update
count and a registry record of bitwise tensor restoration. The exporter checks
restored parameters before freezing them.

The A6000 export uses FP32, evaluation/inference mode and disabled TF32. Sample
IDs and guide/target sequences are aligned to the canonical input order.
Existing frozen accessibility arrays are reused. No efficacy labels enter the
forward pass. Trained features are checked against the archived CPU FP32 arrays;
the readout uses the archived trained arrays to preserve the exact reference.

The output feature manifest binds every finite float32 N-by-192 array to its
SHA256, ordered sample IDs and exact source checkpoint identity. Consult the
exporter's `--help` and the config example for the required input-store fields.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/export_mirna_supervision_control.py \
  --config /path/to/private_config.json --device cuda:0 --batch-size 32 \
  --output /path/to/new_export
```

## Run the readout

The delivered `data/sirna_oligo_random/mirna_supervision_features.json` binds the
three initial feature banks and the existing three trained banks. The saved
predictions and four summary tables can be verified without model weights:

```bash
python scripts/reproduce_mirna_supervision_control.py
```

First run the default seed42/Huesken/fold0 qualification:

```bash
python scripts/run_mirna_supervision_control.py \
  --features-manifest data/sirna_oligo_random/mirna_supervision_features.json \
  --output /path/to/new_smoke_output
```

Then run all fixed outer groups and all matched seeds:

```bash
python scripts/run_mirna_supervision_control.py \
  --features-manifest data/sirna_oligo_random/mirna_supervision_features.json \
  --seed 42 43 44 --all --output /path/to/new_full_output
```

Both modes run by default. Output directories must be new. The small convex
readouts run on CPU on the SSH server after GPU feature extraction.

Outputs include held-out predictions, train-only transformations, selected
alphas, numerical certificates, pooled per-seed metrics, mean/sample SDs and
paired trained-minus-initial differences. Each source seed is scored separately;
predictions are not ensembled. A partial smoke run is not a completed benchmark.

## Interpretation

The completed [result report](../results/siRNA_oligo_random/mirna_supervision_control/REPORT.md)
contains all settings, paired differences and validation evidence.

Report all settings, both modes and all seeds, including negative differences.
Compute the four-setting macro within each seed before its mean/sample SD;
keep the extension macro separate. Higher PCC/Spearman and lower MSE/MAE are
better. Three source seeds quantify source initialization/training variation.
Folds and individual predictions are not independent training replicates.
Huesken-to-Mixset reuses the Mixset test samples. Do not run a t-test treating
the 26 outer groups as independent experiments. The scripts report descriptive
paired differences without significance claims or confidence intervals.

A positive difference supports useful transfer from miRNA task training under
this fixed protocol. The completed [equal-budget shuffled-label control](SHUFFLED_LABEL_CONTROL.md)
compares real and permuted source labels at 32,728 updates for matched seed42.
Its main-macro real-minus-shuffled PCC differences are +0.0083 (source only) and
+0.0289 (source plus TD24). This single-seed comparison is reported separately
from the three-seed initial-state experiment. Neither comparison establishes
biological causality, necessity on every dataset or a new blind-cohort result.
