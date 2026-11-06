# Purged-sequence siRNA evaluation supplement

This supplement restores an existing evaluation of DuplexFM PairAC, the matched TD24 R55 control, OligoFormer corrected reproduction, RNAiSpline and BERT-siRNA on common sequence-purged partitions. It contains saved predictions and audit records; no model was retrained for this release.

The four settings are Huesken (2,196 samples), Mixset (469), Takayuki (664), and Huesken to Mixset (469 test samples). The two Mixset uses refer to the same measurements. Split seed 20261007 is fixed independently of response labels. Three model seeds share the same partitions. Near neighbors are removed from the training side of each fold, and every eligible sample remains in its outer test evaluation.

## Pooled OOF PCC

| Method | Huesken | Mixset | Takayuki | Huesken to Mixset | Macro4 |
|---|---:|---:|---:|---:|---:|
| DuplexFM PairAC + TD24 (R55) | 0.6000 +/- 0.0037 | 0.5658 +/- 0.0097 | 0.5848 +/- 0.0090 | 0.5734 +/- 0.0130 | 0.5810 +/- 0.0071 |
| RNAiSpline | 0.6067 +/- 0.0101 | 0.5337 +/- 0.0201 | 0.6198 +/- 0.0338 | 0.5600 +/- 0.0041 | 0.5801 +/- 0.0114 |
| OligoFormer corrected reproduction | 0.5890 +/- 0.0068 | 0.4624 +/- 0.0337 | 0.5501 +/- 0.0135 | 0.5281 +/- 0.0162 | 0.5324 +/- 0.0040 |
| TD24 (matched R55) | 0.5856 | 0.5405 | 0.5229 | 0.5732 | 0.5555 |
| BERT-siRNA | 0.4511 +/- 0.0624 | 0.1925 +/- 0.0635 | 0.0917 +/- 0.0655 | 0.4379 +/- 0.0034 | 0.2933 +/- 0.0429 |

Values are the mean of three seed-specific PCCs, with sample SD. TD24 R55 is deterministic and is reported once. Within-domain PCC is calculated after pooling five out-of-fold prediction sets for each seed; cross-domain PCC uses the fixed Mixset test cohort. Macro4 is the unweighted mean of the four setting scores for each seed, followed by the seed mean and SD. It is a descriptive average, not four independent cohort estimates. RNAiSpline and PairAC have similar Macro4 point estimates; these data do not establish a statistically significant difference or a general ranking.

A second table family reports the unweighted mean of fold metrics for each seed. The `pooled_oof` and `fold_mean` files have separate names because these summaries answer different questions. They must not be combined with the main repository's random-split HPO table. The historical purged baseline runs used fixed configurations, while later random-split runs used different optimization budgets. Changes between those tables cannot be assigned exclusively to split randomness.

## Files and verification

- `data/`: three exact master input CSVs, including sequence identity, continuous response, original binary truth and TD24.
- `splits/`: sixteen original partition manifests, including fit, validation, test and purged IDs.
- `predictions/all_models.csv`: 49,374 predictions, preserving each saved score and its original serialized response in `source_label`; `label` uses the exact master response.
- `results/`: fold metrics, per-seed scores, setting summaries and actual split sample counts.
- `audit/`: source/checkpoint hashes, model settings, selection receipts, matched baseline fit records and independent verification.
- `reference/oligoformer_reproduction.py`: the exact SHA-bound model definition used by the historical OligoFormer reproduction.

Run from this directory with Python 3.10 or later:

```sh
python scripts/verify.py
```

Only Python's standard library is required. The verifier independently reconstructs sequence-neighbor relations, checks separation and coverage, checks master truth and scores, verifies model selection receipts and recomputes all reported metrics. It does not rerun training or regenerate neural features. `audit/verification.json` records the release check; use `--write-report` to replace this report before integration. After integration, use the default read-only command; writing a new report also requires refreshing the main release inventory.

## Integration

The extracted folder is usable as a separate supplemental release asset. For an existing DuplexFM checkout that enforces `RELEASE_MANIFEST.json`, use the accompanying integration helper to add it under `supplementary/purged_sequence_v1` and refresh the current checkout's inventory. Keep the existing checkout's release/data manifest metadata. Adding loose files without updating that inventory will cause the main release verifier to reject the checkout.

See [PROTOCOL.md](PROTOCOL.md) for split and metric definitions and [MODEL_NOTES.md](MODEL_NOTES.md) for training and interpretation boundaries. This package adds no new license grant over the source datasets, pretrained assets or third-party implementations. Their original terms and the DuplexFM repository's license and third-party notices continue to apply; no pretrained weights are redistributed here.
