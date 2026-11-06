# OligoFormer remeasurement and Takayuki input identity

**Verified on 2026-11-06:** the OligoFormer comparison in this repository is a computational rerun under the common siRNA evaluation protocol. It uses the pinned official Takayuki `data/Taka.csv` sequences. It is not a transcription of the original paper's scores, and “remeasurement” here means model retraining and evaluation, not a new wet-lab assay.

The machine-readable evidence is [oligoformer_data_audit.json](../audit/oligoformer_data_audit.json). The supplied predictions are [hpo.csv](../data/sirna_oligo_random/predictions/hpo.csv).

## Takayuki input identity

Takayuki uses **`data/Taka.csv`** from the registered [OligoFormer revision](https://github.com/lulab/OligoFormer/tree/e2f53ad63387bbe166bf123949151e2bc9bf6ec3). The file identity, eligibility rules and retained inputs are recorded below.

| Evidence | Verified value |
|---|---|
| Registered upstream revision | `e2f53ad63387bbe166bf123949151e2bc9bf6ec3` |
| Source | [data/Taka.csv at the registered revision](https://github.com/lulab/OligoFormer/blob/e2f53ad63387bbe166bf123949151e2bc9bf6ec3/data/Taka.csv) |
| Source size and rows | 190,506 bytes; 702 rows |
| Source SHA-256 | `221672544ee0bd7e4c8029d9e8f5d7463c90aa9f5623976002b5e68c35300b2f` |
| Retained Takayuki rows | 664 |
| Excluded source rows | 38 target windows containing non-ACGU flank padding |

The file at the registered revision and the current upstream file had identical bytes when checked. Its SHA also equals the raw Takayuki source SHA retained in [input_manifest.json](../data/sirna_oligo_random/input_manifest.json).

All **664/664** prepared rows were compared with the pinned upstream source. The guide sequence, target57 sequence, continuous `label`, inherited binary `y`, all 24 thermodynamic values, and the original row index encoded by `sample_id` agreed. Labels and TD24 were compared numerically because CSV serialization can change the printed decimal precision. No second efficacy normalization was applied. The retained continuous response range is 0–1.

The common eligibility rule requires a 19-nt A/C/G/U guide, a 57-nt A/C/G/U target and valid TD24. Both DuplexFM and the rerun baselines use the same prepared rows. `label` is the continuous regression response; `y` is the inherited binary class. The two columns must not be interchanged or reconstructed with an assumed universal threshold.

## What was rerun

The random evaluation reuses OligoFormer's two seed42 shuffles and outer-fold convention, then applies the registered eligibility filter. The original cohorts have five outer folds; Huesken→Mixset has one cross-dataset evaluation. Wang and ALAS1 add five folds each. The resulting common plan contains 26 outer groups and 3,688 unique input rows.

OligoFormer-corrected, BERT-siRNA and RNAiSpline were each fitted under that common plan for seeds42/43/44. For every model and seed, the retained prediction bank contains 4,157 predictions covering all six settings and 26 outer groups; Mixset appears again in Huesken→Mixset. Model/seed summaries are computed from these saved predictions rather than copied from published benchmark tables.

The [registered search](../workflows/sirna_oligo_random/baselines/hpo_specification.json) fixes eight learning-rate/weight-decay/batch candidates. Seed42 inner validation performs the registered 8→3→1 search; its second stage uses sample-weighted MSE from the first two inner folds. Exact ties follow fixed candidate order. After the winner is frozen, each final seed uses all five inner folds for epoch selection, and a fresh fit on the complete outer training partition uses the median selected epoch. Outer-test responses do not select a candidate or epoch. The [worker](../workflows/sirna_oligo_random/baselines/hpo_worker.py) and [backend](../workflows/sirna_oligo_random/baselines/hpo_backend.py) preserve these rules.

## Baseline implementation and selection

**OligoFormer-corrected** names this repository's adapted model implementation. Its source identity and implementation details are recorded separately from the input-data identity above.

The registered author code is [scripts/model.py](https://github.com/lulab/OligoFormer/blob/e2f53ad63387bbe166bf123949151e2bc9bf6ec3/scripts/model.py) and [scripts/train_single.py](https://github.com/lulab/OligoFormer/blob/e2f53ad63387bbe166bf123949151e2bc9bf6ec3/scripts/train_single.py). The rerun is not an unchanged execution of those files:

| Component | Registered upstream code | Rerun implementation |
|---|---|---|
| Sequence recurrent layout | `nn.LSTM` default sequence-first layout receives the constructed batch-first tensor | Explicit `batch_first=True`; guide lengths and padding masks are handled explicitly |
| Attention block | Custom `EncoderLayer` | PyTorch `TransformerEncoderLayer`, with batch-first input and registered dimensions |
| Efficacy output | Two outputs with softmax; training uses column 1 | One sigmoid output fitted to the continuous response by MSE |
| Frozen features | RNA-FM and thermodynamic inputs | 128 guide RNA-FM + 128 target RNA-FM + TD24 = 280 features; normalization fitted on training rows |
| Optimization and selection | Adam, exponential scheduler, joint validation-loss/AUC checkpoint rule | Registered AdamW policy and inner-validation MSE; fixed-epoch complete outer refit |

The adapted model source has SHA-256 `13fac6ac99d2d5145b5e5fc87f8b7804f91fd5d6f2c8eef421e112c5189b6177`. The [adapter](../workflows/sirna_oligo_random/baselines/vendor/baseline_gpu_models.py) and HPO backend enforce this identity. Exact model-source assets and backbone weights remain necessary to retrain; the retained adapter alone is not a standalone installation of every upstream method.

BERT-siRNA loads the SHA-locked `siRNABertRegressor` class and registered DNABERT6 weights. RNAiSpline uses the locked model/preprocessing source, a 60-epoch label-free reconstruction stage restricted to each training partition, and the registered supervised stage. Its vectorized spline basis is checked against native forward values and gradients; its deterministic maximum-pooling compatibility operation uses the first argmax. These are retrained comparisons with disclosed adapters. DSIR is separately labeled as inference with published Huesken-supervised weights and is not assigned a three-seed retraining SD.

OligoFormer was computationally retrained and evaluated under the shared OligoFormer-derived random-fold protocol using the pinned Takayuki input. Its adapted implementation is reported as OligoFormer-corrected, with the implementation and model-selection details above.

Structural-reactivity checkpoint identities are a separate issue, documented in [ICSHAPE_CHECKPOINTS.md](ICSHAPE_CHECKPOINTS.md), [ACCESSIBILITY_COMPARISON.md](ACCESSIBILITY_COMPARISON.md) and [METHODS.md](METHODS.md#supervised-structural-reactivity-regression). The original epoch22 predictor used by PairAC and the reconstructed epoch9 predictor must retain their separate names and results.
