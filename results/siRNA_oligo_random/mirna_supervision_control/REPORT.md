# Contribution of miRNA task supervision

## Key result

| Main four-setting macro PCC | Without miRNA task supervision | With miRNA task supervision | Paired increase |
| --- | ---: | ---: | ---: |
| Representation only | 0.4772 ± 0.0217 | **0.5179 ± 0.0090** | **+0.0407 ± 0.0240** |
| Representation + TD24 | 0.6157 ± 0.0015 | **0.6330 ± 0.0054** | **+0.0172 ± 0.0062** |

Mean ± sample SD across paired source seeds 42, 43 and 44.
**The main macro improves in both modes, while Huesken-to-Mixset decreases in both.**
These descriptive results support an average contribution of miRNA task training;
public pretraining and independent structural supervision are retained.

[New-file guide](../../../docs/MIRNA_CONTROL_FILES.md) ·
[Protocol and commands](../../../docs/MIRNA_SUPERVISION_CONTROL.md)

Completed on 2026-11-06. Each initial checkpoint is the exact saved task
state before any miRNA updates, paired with its own terminal source model.
Public foundation weights and the independently supervised structural input
are retained. All encoders are frozen; both conditions receive the same
supervised siRNA R55 readout. No new source training was performed.

The following detailed tables include all settings and both macros; no settings
were selected after the comparison.

## Representation only

PCC, mean ± sample SD across source seeds 42, 43 and 44. The difference
is computed within each matched seed before its mean and SD.

| Setting | Without miRNA task training | With miRNA task training | Paired difference |
| --- | ---: | ---: | ---: |
| Huesken | 0.4463 ± 0.0216 | 0.5126 ± 0.0285 | +0.0663 ± 0.0446 |
| Mixset | 0.4290 ± 0.0291 | 0.4576 ± 0.0210 | +0.0286 ± 0.0155 |
| Takayuki | 0.5492 ± 0.0134 | 0.6538 ± 0.0078 | +0.1045 ± 0.0112 |
| Huesken-to-Mixset | 0.4842 ± 0.0242 | 0.4475 ± 0.0048 | -0.0367 ± 0.0287 |
| Main four-setting macro | 0.4772 ± 0.0217 | 0.5179 ± 0.0090 | +0.0407 ± 0.0240 |
| Wang (extension) | 0.3172 ± 0.0160 | 0.5418 ± 0.0798 | +0.2246 ± 0.0939 |
| ALAS1 (extension) | 0.2174 ± 0.0237 | 0.4626 ± 0.0824 | +0.2451 ± 0.1031 |
| Extension two-setting macro | 0.2673 ± 0.0186 | 0.5022 ± 0.0797 | +0.2349 ± 0.0982 |

## Representation + TD24

PCC, mean ± sample SD across source seeds 42, 43 and 44. The difference
is computed within each matched seed before its mean and SD.

| Setting | Without miRNA task training | With miRNA task training | Paired difference |
| --- | ---: | ---: | ---: |
| Huesken | 0.6184 ± 0.0016 | 0.6235 ± 0.0020 | +0.0051 ± 0.0008 |
| Mixset | 0.5766 ± 0.0024 | 0.6075 ± 0.0060 | +0.0309 ± 0.0053 |
| Takayuki | 0.6874 ± 0.0041 | 0.7225 ± 0.0040 | +0.0351 ± 0.0080 |
| Huesken-to-Mixset | 0.5805 ± 0.0018 | 0.5783 ± 0.0127 | -0.0022 ± 0.0124 |
| Main four-setting macro | 0.6157 ± 0.0015 | 0.6330 ± 0.0054 | +0.0172 ± 0.0062 |
| Wang (extension) | 0.4760 ± 0.0173 | 0.6235 ± 0.0201 | +0.1475 ± 0.0043 |
| ALAS1 (extension) | 0.5315 ± 0.0148 | 0.6140 ± 0.0149 | +0.0825 ± 0.0258 |
| Extension two-setting macro | 0.5037 ± 0.0113 | 0.6187 ± 0.0173 | +0.1150 ± 0.0150 |

## Other main-macro metrics

| Mode | Metric | Without miRNA task training | With miRNA task training | Paired difference |
| --- | --- | ---: | ---: | ---: |
| Representation only | Spearman | 0.4674 ± 0.0200 | 0.5228 ± 0.0027 | +0.0554 ± 0.0181 |
| Representation only | MSE | 0.0530 ± 0.0012 | 0.0510 ± 0.0005 | -0.0020 ± 0.0013 |
| Representation only | MAE | 0.1875 ± 0.0029 | 0.1820 ± 0.0011 | -0.0055 ± 0.0029 |
| Representation + TD24 | Spearman | 0.6200 ± 0.0011 | 0.6367 ± 0.0057 | +0.0166 ± 0.0057 |
| Representation + TD24 | MSE | 0.0441 ± 0.0001 | 0.0429 ± 0.0002 | -0.0012 ± 0.0002 |
| Representation + TD24 | MAE | 0.1669 ± 0.0004 | 0.1642 ± 0.0004 | -0.0027 ± 0.0005 |

The deterministic TD24-only main macro PCC is 0.6038. It is fitted once;
no artificial source-seed SD is assigned. Lower MSE/MAE is better.

## Validation and interpretation

- Three matched source seeds, 26 outer groups, 8,788 nested/final head fits and 54,041 saved predictions.
- All trained and TD24 predictions reproduce their archives; maximum absolute prediction error is 5.11e-15.
- A single RTX A6000 exported all six source states in FP32 (64.5 seconds reserved wall time). The three trained GPU feature banks passed elementwise CPU-bank parity at atol 1e-4/rtol 1e-5; maximum absolute difference was 5.23e-5.
- The fixed readout ran on the server CPU in 361.2 seconds. The final focused suite passed 49 tests.
- Source-seed SD is descriptive; neither folds nor repeated Mixset samples are independent training replicates. No significance test or confidence interval is claimed.
- This initial-state comparison estimates the contribution of miRNA task updates. The completed [equal-budget shuffled-label comparison](../shuffled_supervision_control/REPORT.md) separately tests real versus permuted labels at the same 32,728-update budget for seed42.
- Interpretation is specific to the OligoFormer-derived random-fold protocol, with repeated downstream development taken into account.

## Files and reproduction

The CSV files in this directory contain per-seed, mean/SD and paired-difference metrics.
Predictions are in `data/sirna_oligo_random/predictions/mirna_supervision_control.csv`.
Initial feature banks and their checkpoint/row bindings are in
`data/sirna_oligo_random/mirna_supervision_features.json`. Execution provenance is
in `audit/mirna_supervision_control.json`. See the [protocol](../../../docs/MIRNA_SUPERVISION_CONTROL.md).

```bash
python scripts/reproduce_mirna_supervision_control.py
```
