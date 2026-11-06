# Reviewer revision: files and reading order

Start with the highlighted equal-budget table in the [README](../README.md), then the [complete report](../results/siRNA_oligo_random/shuffled_supervision_control/REPORT.md).

## Added code and tests

| File | Purpose |
|---|---|
| [export_shuffled_supervision_control.py](../scripts/export_shuffled_supervision_control.py) | Audit equal-budget source training, export frozen shuffled features, and refit both R55 modes. |
| [reproduce_shuffled_supervision_control.py](../scripts/reproduce_shuffled_supervision_control.py) | Independently verify all saved predictions, reference identities and result tables. |
| [test_shuffled_supervision_control.py](../tests/test_shuffled_supervision_control.py) | Reject nonmatching training budgets and validation-label contamination. |
| [test_reproduce_shuffled_supervision_control.py](../tests/test_reproduce_shuffled_supervision_control.py) | Reject wrong source seeds and changed reference predictions; enforce single-seed reporting. |

## Added data and results

| File | Purpose |
|---|---|
| [shuffled_seed42.npy](../data/sirna_oligo_random/features/shuffled_seed42.npy) | 3,688 × 192 frozen feature bank from the shuffled-label source. |
| [shuffled_supervision_features.json](../data/sirna_oligo_random/shuffled_supervision_features.json) | Feature/checkpoint identities, source-budget evidence and frozen-input hashes. |
| [shuffled_supervision_control.csv](../data/sirna_oligo_random/predictions/shuffled_supervision_control.csv) | 29,099 predictions spanning seed42 initial, real-label and shuffled-label modes plus TD24. |
| [REPORT.md](../results/siRNA_oligo_random/shuffled_supervision_control/REPORT.md) | Highlighted results, every setting, interpretation and reproduction commands. |
| [per_seed.csv](../results/siRNA_oligo_random/shuffled_supervision_control/per_seed.csv) | PCC, Spearman, MSE and MAE for each setting and macro. |
| [summary.csv](../results/siRNA_oligo_random/shuffled_supervision_control/summary.csv) | Single-source summaries with empty SD fields. |
| [paired_deltas.csv](../results/siRNA_oligo_random/shuffled_supervision_control/paired_deltas.csv) | Real-label minus shuffled and initial differences. |
| [complete.json](../results/siRNA_oligo_random/shuffled_supervision_control/complete.json) | 52 outer condition/mode fits and 1,352 nested/final fits with numerical certificates. |

## Added provenance and explanatory documents

| File | Purpose |
|---|---|
| [shuffled_supervision_control.json](../audit/shuffled_supervision_control.json) | Original initialization, training-budget and train-only permutation audit. |
| [oligoformer_data_audit.json](../audit/oligoformer_data_audit.json) | Takayuki data identity and 664-row comparison; baseline rerun provenance. |
| [icshape_checkpoint_identity.json](../audit/icshape_checkpoint_identity.json) | Rehashed original epoch22 and reconstructed epoch9 checkpoint identities. |
| [reviewer_revision_verification.json](../audit/reviewer_revision_verification.json) | Final saved-result, test and data/checkpoint consistency verification. |
| [SHUFFLED_LABEL_CONTROL.md](SHUFFLED_LABEL_CONTROL.md) | Matched source budget, readout protocol and reproduction commands. |
| [OLIGOFORMER_REMEASUREMENT_AUDIT.md](OLIGOFORMER_REMEASUREMENT_AUDIT.md) | Computational rerun, pinned Takayuki source and baseline implementation changes. |
| [ICSHAPE_CHECKPOINTS.md](ICSHAPE_CHECKPOINTS.md) | Separate roles, training populations, batch sizes and results of the two reactivity predictors. |
| [REVIEWER_REVISION_FILES.md](REVIEWER_REVISION_FILES.md) | This index. |

The earlier three-seed initialization control is indexed separately in [MIRNA_CONTROL_FILES.md](MIRNA_CONTROL_FILES.md). This revision updates README, Methods, Data, Results, reproduction instructions and existing control documents to link the completed comparison. `MODEL_REGISTRY.json`, `DATA_MANIFEST.json` and `RELEASE_MANIFEST.json` record the resulting identities and inventory.

The project-weight packages and verified public-foundation download procedure are documented in [WEIGHTS.md](WEIGHTS.md).

## Verify

```bash
python scripts/verify_release.py
python scripts/reproduce_mirna_supervision_control.py
python scripts/reproduce_shuffled_supervision_control.py
python scripts/reproduce_accessibility_mae.py
```

These commands check the delivered artifacts and saved-result calculations. Regenerating source features requires the separately supplied model and backbone assets described in the protocols.
