# miRNA supervision control: file index

Start with the highlighted result in the [repository README](../README.md),
then read the [full results](../results/siRNA_oligo_random/mirna_supervision_control/REPORT.md),
the [matched-control protocol](MIRNA_SUPERVISION_CONTROL.md), and the scripts below.

The original exact-initialization addition contains **20 files**, including this index. The subsequent equal-budget control, data identity audit and checkpoint clarification are indexed in [REVIEWER_REVISION_FILES.md](REVIEWER_REVISION_FILES.md). Files follow the
repository's existing `scripts/`, `data/`, `results/`, `tests/`, `audit/` and
`docs/` layout; their paths are retained so imports and manifest references
continue to resolve.

## Scripts (3 files)

| File | Purpose |
| --- | --- |
| [export_mirna_supervision_control.py](../scripts/export_mirna_supervision_control.py) | Export matched initial/trained frozen features and check checkpoint identity and trained-feature parity. Requires source assets and model weights. |
| [run_mirna_supervision_control.py](../scripts/run_mirna_supervision_control.py) | Fit the fixed nested siRNA readout and save predictions, metrics and completion checks. |
| [reproduce_mirna_supervision_control.py](../scripts/reproduce_mirna_supervision_control.py) | Verify saved predictions and recompute the four result tables without model weights or a GPU. |

## Features and predictions (5 files)

| File | Purpose |
| --- | --- |
| [initial_seed42.npy](../data/sirna_oligo_random/features/initial_seed42.npy) | Exact pre-miRNA-update frozen feature bank for source seed 42. |
| [initial_seed43.npy](../data/sirna_oligo_random/features/initial_seed43.npy) | Exact pre-miRNA-update frozen feature bank for source seed 43. |
| [initial_seed44.npy](../data/sirna_oligo_random/features/initial_seed44.npy) | Exact pre-miRNA-update frozen feature bank for source seed 44. |
| [mirna_supervision_features.json](../data/sirna_oligo_random/mirna_supervision_features.json) | Bind the three new initial banks and three existing trained banks to their hashes, ordered samples and source checkpoints. |
| [mirna_supervision_control.csv](../data/sirna_oligo_random/predictions/mirna_supervision_control.csv) | All 54,041 saved held-out predictions, including the deterministic TD24 baseline. |

## Results (6 files)

| File | Purpose |
| --- | --- |
| [REPORT.md](../results/siRNA_oligo_random/mirna_supervision_control/REPORT.md) | Main findings, all setting-level results, limitations and reproduction commands. |
| [per_seed.csv](../results/siRNA_oligo_random/mirna_supervision_control/per_seed.csv) | PCC, Spearman, MSE and MAE by condition, source seed, readout mode and setting. |
| [summary.csv](../results/siRNA_oligo_random/mirna_supervision_control/summary.csv) | Condition-level means and sample SD across source seeds. |
| [paired_deltas_per_seed.csv](../results/siRNA_oligo_random/mirna_supervision_control/paired_deltas_per_seed.csv) | Trained-minus-initial differences computed within each matched source seed. |
| [paired_deltas_summary.csv](../results/siRNA_oligo_random/mirna_supervision_control/paired_deltas_summary.csv) | Means and sample SD of the matched differences. |
| [complete.json](../results/siRNA_oligo_random/mirna_supervision_control/complete.json) | Completion checks, fold/fit/prediction counts and archived-prediction agreement. |

## Tests (3 files)

| File | Purpose |
| --- | --- |
| [test_export_mirna_supervision_control.py](../tests/test_export_mirna_supervision_control.py) | Feature-export and identity/parity checks. |
| [test_mirna_supervision_control.py](../tests/test_mirna_supervision_control.py) | Matched-control readout, split and metric checks. |
| [test_reproduce_mirna_supervision_control.py](../tests/test_reproduce_mirna_supervision_control.py) | Saved-prediction and result-reproduction checks. |

## Provenance and documentation (3 files)

| File | Purpose |
| --- | --- |
| [mirna_supervision_control.json](../audit/mirna_supervision_control.json) | A6000 export and readout provenance, asset hashes, parity checks and validation evidence. |
| [MIRNA_SUPERVISION_CONTROL.md](MIRNA_SUPERVISION_CONTROL.md) | Scientific comparison, fixed protocol, export/readout commands and interpretation boundaries. |
| [MIRNA_CONTROL_FILES.md](MIRNA_CONTROL_FILES.md) | This inventory and reading order. |

## Verify the delivered results

Run from the repository root with the repository dependencies available:

```bash
python scripts/reproduce_mirna_supervision_control.py
```

This checks fixed input/split identities and recomputes metrics, source-seed
sample SD and paired differences from the saved predictions. For feature export
or refitting, follow the [protocol](MIRNA_SUPERVISION_CONTROL.md).
