# Equal-budget shuffled-miRNA-label control

This control compares real miRNA labels with a fixed permutation of the same source-training labels. It complements the three-seed exact-initialization comparison by holding the number of source-training updates fixed. Results and all setting-level differences are in the [completed report](../results/siRNA_oligo_random/shuffled_supervision_control/REPORT.md).

## Matched source training

| Component | Real-label source | Shuffled-label source |
|---|---|---|
| Source seed | 42 | 42 |
| Initialization | Exact saved seed42 task parameters, fusion buffers and RNG | Bitwise-identical parameters, buffers and RNG |
| Public backbones and structural input | RNA-FM, mRNABERT and the frozen original reactivity predictor | Same |
| Source training rows | 2,094,107 | Same rows and inputs |
| Training labels | Original binary miRNA labels | One fixed RNG42 permutation restricted to training rows |
| Positive training labels | 1,042,521 | 1,042,521 |
| Batch / microbatch | 512 / 512 | 512 / 512 |
| Updates / reference epochs | 32,728 / 8 | 32,728 / 8 |
| Total source row presentations | 16,752,856 | 16,752,856 |
| Main / LoRA learning rate | 3e-4 / 2e-5 | 3e-4 / 2e-5 |
| Weight decay / gradient clipping | 1e-4 / 1.0 | 1e-4 / 1.0 |
| Precision | BF16 | BF16 |
| Checkpoint | Fixed terminal update | Fixed terminal update |

The permutation changes 1,046,888 training labels. Validation and test labels do not enter the permutation. No validation score or siRNA result chooses the source terminal checkpoint. Exact training signatures, checkpoint hashes, permutation identity and completion records are checked by [the source audit/export script](../scripts/export_shuffled_supervision_control.py).

The completed archived source training is reused; the present revision rechecks its provenance and extracts its frozen siRNA features on an RTX A6000. It does not rerun the source-training budget. Equal updates and options do not assert bitwise-identical training arithmetic across historical hardware/runtime environments.

## siRNA evaluation

All conditions use the same 3,688 eligible inputs, OligoFormer-derived random-fold protocol, 26 outer groups, five inner folds and R55 readout. The pinned Takayuki data are [verified separately](OLIGOFORMER_REMEASUREMENT_AUDIT.md). Each condition produces 4,157 held-out predictions because Mixset is evaluated in both Mixset and Huesken→Mixset settings.

Both frozen source192-only and source192-plus-TD24 modes are evaluated. Scaling and regularization selection use the relevant training partition. Real-label, initial-state and TD24 predictions are reused unchanged from the verified initial-versus-trained comparison; 52 new shuffled-condition outer fits, including their nested fits, add 1,352 fitted readouts. The combined comparison has 29,099 predictions across seven condition/mode variants.

Only **one shuffled source seed** is available. Compare shuffled seed42 with real-label seed42 and initial seed42. The three-seed real-label mean belongs to the separate initial-versus-trained experiment. No source-seed SD, independent-fold significance test or confidence interval is assigned to this single-source comparison. Both main and extension settings, including negative differences, are retained.

## Verify saved results

```bash
python scripts/reproduce_shuffled_supervision_control.py
```

This verifies source-budget metadata, frozen-feature identity, exact-once outer-test membership, unchanged reference predictions and all result-table cells. It requires the repository's scientific Python dependencies, without model weights or a GPU.

To refit from the included shuffled frozen features:

```bash
python scripts/export_shuffled_supervision_control.py readout \
  --features-manifest data/sirna_oligo_random/shuffled_supervision_features.json \
  --previous-predictions data/sirna_oligo_random/predictions/mirna_supervision_control.csv \
  --output ../shuffled_readout_refit
```

To regenerate the frozen features, separately supplied source checkpoints, exact PairStore inputs/accessibility arrays and foundation weights are required. A private config extends the [initial-control export configuration](MIRNA_SUPERVISION_CONTROL.md#feature-export) with `shuffle_source_root`, pointing to the hash-bound source-training archive. The command refuses to overwrite an existing output:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/export_shuffled_supervision_control.py export \
  --config /path/to/private_shuffle_config.json --device cuda:0 --batch-size 32 \
  --output /path/to/new_shuffled_export
```

The public feature manifest and audit retain hashes and scientific provenance. Machine-local asset paths and private runtime configuration are excluded from the release.
