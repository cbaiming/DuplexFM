# Equal-budget shuffled-miRNA-label source control

**Real miRNA labels yield a higher main-macro PCC than shuffled labels at the same source-training budget in the matched seed42 comparison.**

| Readout | Initial, no miRNA updates | Shuffled labels, 32,728 updates | Real labels, 32,728 updates | Real − shuffled |
|---|---:|---:|---:|---:|
| Representation only | 0.4866 | 0.4997 | **0.5080** | **+0.0083** |
| Representation + TD24 | 0.6140 | 0.6083 | **0.6371** | **+0.0289** |

Main four-setting macro PCC; **one source seed, 42**. These entries have no source-seed SD. They must not be compared as if they were the three-seed means from the [exact-initialization experiment](../mirna_supervision_control/REPORT.md).

[Protocol and commands](../../../docs/SHUFFLED_LABEL_CONTROL.md) · [Revision file guide](../../../docs/REVIEWER_REVISION_FILES.md)

The comparison holds the initialization, public foundation models, frozen structural inputs, source rows, optimizer options, batch512 and 32,728-update terminal checkpoint rule fixed. A single fixed permutation is applied only to the 2,094,107 training labels; the 1,042,521 positives are preserved. Both source fits consume 16,752,856 row presentations. The archived completed source fit is audited and reused; this revision exports its siRNA features and refits the readouts.

## Representation only

PCC for matched source seed42.

| Setting | Initial | Shuffled labels | Real labels | Real − shuffled |
|---|---:|---:|---:|---:|
| Huesken | 0.4490 | 0.4803 | 0.4837 | +0.0034 |
| Mixset | 0.4416 | 0.4351 | 0.4539 | +0.0188 |
| Takayuki | 0.5590 | 0.5983 | 0.6510 | +0.0527 |
| Huesken→Mixset | 0.4968 | 0.4851 | 0.4433 | -0.0418 |
| Main four-setting macro | 0.4866 | 0.4997 | 0.5080 | +0.0083 |
| Wang (extension) | 0.3354 | 0.3961 | 0.4556 | +0.0595 |
| ALAS1 (extension) | 0.2407 | 0.2663 | 0.3675 | +0.1012 |
| Extension two-setting macro | 0.2880 | 0.3312 | 0.4115 | +0.0803 |

## Representation + TD24

PCC for matched source seed42.

| Setting | Initial | Shuffled labels | Real labels | Real − shuffled |
|---|---:|---:|---:|---:|
| Huesken | 0.6166 | 0.6124 | 0.6217 | +0.0093 |
| Mixset | 0.5768 | 0.5754 | 0.6137 | +0.0383 |
| Takayuki | 0.6837 | 0.6668 | 0.7268 | +0.0600 |
| Huesken→Mixset | 0.5791 | 0.5785 | 0.5863 | +0.0078 |
| Main four-setting macro | 0.6140 | 0.6083 | 0.6371 | +0.0289 |
| Wang (extension) | 0.4580 | 0.5008 | 0.6011 | +0.1003 |
| ALAS1 (extension) | 0.5393 | 0.5410 | 0.5968 | +0.0558 |
| Extension two-setting macro | 0.4987 | 0.5209 | 0.5990 | +0.0780 |

## Other main-macro metrics

| Readout | Metric | Shuffled labels | Real labels | Real − shuffled |
|---|---|---:|---:|---:|
| Representation only | Spearman | 0.4945 | 0.5216 | +0.0271 |
| Representation only | MSE | 0.0521 | 0.0515 | -0.0006 |
| Representation only | MAE | 0.1853 | 0.1831 | -0.0022 |
| Representation + TD24 | Spearman | 0.6099 | 0.6409 | +0.0310 |
| Representation + TD24 | MSE | 0.0449 | 0.0427 | -0.0021 |
| Representation + TD24 | MAE | 0.1686 | 0.1638 | -0.0048 |

Lower MSE/MAE is better. TD24 alone has macro PCC 0.6038 and is represented once, without a source-seed SD. Every per-setting PCC, Spearman, MSE and MAE remains in [per_seed.csv](per_seed.csv).

## Interpretation

In representation-only transfer, real labels exceed shuffled labels in five of six settings, but Huesken→Mixset decreases by -0.0418 PCC. With TD24, real-label PCC is higher in all six settings for this seed. These are descriptive results under the stated random-fold protocol, not a universal-transfer or significance claim.

Shuffled-label source training also raises the representation-only macro above the exact initialization (0.4866 → 0.4997). Thus the entire trained-versus-initial gain should not be attributed to true-label information. The real-versus-shuffled difference addresses label information conditional on the matched training budget. It does not establish biological causality or that miRNA labels outperform every alternative supervised source task.

The three-seed initial-versus-trained result remains separate: macro PCC 0.4772 ± 0.0217 → 0.5179 ± 0.0090 for representation-only transfer, and 0.6157 ± 0.0015 → 0.6330 ± 0.0054 with TD24. Its source-seed uncertainty cannot be assigned to the single shuffled source. Folds and the reused Mixset samples are not independent source-training replicates.

## Execution and verification

- Completed on 2026-11-06 using the existing source archive and one RTX A6000 for FP32 feature export; source training was not repeated.
- 3,688 × 192 shuffled features, 52 new outer condition/mode fits, 1,352 nested/final readouts and 29,099 combined predictions.
- Frozen feature export took 12.6 seconds; the CPU readout took 67.8 seconds. Historical shuffled source training took 34,479.5 seconds.
- Original, shuffled and initial source identities and training-label permutation checks are in [the audit](../../../audit/shuffled_supervision_control.json). Export provenance and frozen input hashes are bound by [the feature manifest](../../../data/sirna_oligo_random/shuffled_supervision_features.json).
- The saved-result verifier checks all predictions and result tables, including unchanged real-label/initial/TD24 predictions.

```bash
python scripts/reproduce_shuffled_supervision_control.py
```
