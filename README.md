# DuplexFM

Code and data for learning transferable RNA interaction representations from miRNA targeting.

DuplexFM combines contextual guide–target score maps, learned nucleotide-pair embeddings and predicted target structural reactivity. The PairAC encoder produces a 192-dimensional interaction representation. For siRNA efficacy prediction, the miRNA-trained encoder is frozen and a supervised linear readout is fitted to its representation, alone or with 24 siRNA-specific sequence and thermodynamic descriptors (TD24).

## New result: equal-budget miRNA-label control

**Real miRNA labels yield higher average siRNA PCC than shuffled labels at the same 32,728-update budget in the matched seed42 comparison.**

| Readout | Initial, no miRNA updates | Shuffled labels, 32,728 updates | Real labels, 32,728 updates | Real − shuffled |
|---|---:|---:|---:|---:|
| Representation only | 0.4866 | 0.4997 | **0.5080** | **+0.0083** |
| Representation + TD24 | 0.6140 | 0.6083 | **0.6371** | **+0.0289** |

Values are main four-setting macro PCC for **one matched source seed (42)**.
The representation-only Huesken→Mixset comparison decreases; with TD24, PCC improves in all six settings for this seed. No statistical significance or source-seed SD is claimed for the shuffled control.

**[Equal-budget results](results/siRNA_oligo_random/shuffled_supervision_control/REPORT.md)**
 · [Revision file guide](docs/REVIEWER_REVISION_FILES.md)
 · [Takayuki input and OligoFormer rerun](docs/OLIGOFORMER_REMEASUREMENT_AUDIT.md)
 · [icSHAPE checkpoint identities](docs/ICSHAPE_CHECKPOINTS.md)

## Exact-initialization control: three source seeds

**MiRNA task training improves average siRNA efficacy prediction in the matched control.**
Both conditions use the same frozen architecture, public pretrained backbones,
structural inputs, siRNA splits and R55 readout. The control uses the exact saved
initialization before any miRNA task update.

| Main four-setting macro PCC | Without miRNA task supervision | With miRNA task supervision | Paired increase |
|---|---:|---:|---:|
| Representation only | 0.4772 ± 0.0217 | **0.5179 ± 0.0090** | **+0.0407 ± 0.0240** |
| Representation + TD24 | 0.6157 ± 0.0015 | **0.6330 ± 0.0054** | **+0.0172 ± 0.0062** |

Values are mean ± sample SD across paired source seeds 42, 43 and 44.
**Huesken→Mixset did not improve in either mode**; this is an average benefit,
not a universal or significance claim. Public pretraining and independent
structural supervision remain in both conditions.

**[Full results](results/siRNA_oligo_random/mirna_supervision_control/REPORT.md)**
 · [New-file guide](docs/MIRNA_CONTROL_FILES.md)
 · [Protocol and commands](docs/MIRNA_SUPERVISION_CONTROL.md)

## Main experiments

| Evaluation | Result |
|---|---:|
| miRBench v7, four-test macro AP | 0.8786 ± 0.0014 |
| Oligo random splits, four-setting macro PCC, DuplexFM + TD24 | 0.6330 ± 0.0054 |
| Oligo random splits, four-setting macro PCC, representation only | 0.5179 ± 0.0090 |
| Additional Wang/ALAS1 settings, macro PCC, DuplexFM + TD24 | 0.6187 ± 0.0173 |

Results summarize source seeds 42, 43 and 44; predictions are not ensembled. The four siRNA settings are Huesken, Mixset, Takayuki and Huesken→Mixset. Wang and ALAS1 use extensions of the random-split rule and are averaged separately. Representation-only transfer still uses a supervised siRNA readout.

This repository focuses on the main miRNA and siRNA experiments and the supervised structural-reactivity comparison. Ablation implementations are retained as research code; their result tables are outside this repository's reporting scope.

## Supplementary sequence-purged evaluation

We additionally report an earlier evaluation on common eligible inputs under fixed `purged_sequence_v1` partitions. Direct sequence neighbors are excluded across fit, validation and test: guide Hamming distance at most two, or a shared canonical target 25-mer.

| Method | Huesken | Mixset | Takayuki | Huesken→Mixset | Macro4 |
|---|---:|---:|---:|---:|---:|
| DuplexFM + TD24 (R55) | 0.6000 ± 0.0037 | 0.5658 ± 0.0097 | 0.5848 ± 0.0090 | 0.5734 ± 0.0130 | 0.5810 ± 0.0071 |
| RNAiSpline | 0.6067 ± 0.0101 | 0.5337 ± 0.0201 | 0.6198 ± 0.0338 | 0.5600 ± 0.0041 | 0.5801 ± 0.0114 |
| TD24, matched R55 | 0.5856 | 0.5405 | 0.5229 | 0.5732 | 0.5555 |
| OligoFormer reproduction | 0.5890 ± 0.0068 | 0.4624 ± 0.0337 | 0.5501 ± 0.0135 | 0.5281 ± 0.0162 | 0.5324 ± 0.0040 |
| BERT-siRNA | 0.4511 ± 0.0624 | 0.1925 ± 0.0635 | 0.0917 ± 0.0655 | 0.4379 ± 0.0034 | 0.2933 ± 0.0429 |

Within-domain PCC pools the five outer test folds for each seed; values are mean ± sample SD across seeds 42, 43 and 44. TD24 is deterministic. DuplexFM and RNAiSpline have similar Macro4 point estimates; these results do not establish a statistically significant difference. Macro4 reuses the same 469-sample Mixset cohort in two settings.

This split controls direct sequence neighbors but does not hold out target genes, transcripts or entire transitive sequence components. These historical fixed-configuration runs and the main random-split HPO runs have different optimization budgets; score changes cannot be attributed solely to the partitions or training randomness.

[Full results, input data and predictions](supplementary/purged_sequence_v1/README.md) · [Partition protocol](supplementary/purged_sequence_v1/PROTOCOL.md) · [Model identities and training settings](supplementary/purged_sequence_v1/MODEL_NOTES.md)

## Organization

| Path | Contents |
|---|---|
| reference/pairac_concat_v1/ | Model, source training, feature extraction, readout and ablation implementations |
| workflows/accessibility/ | Supervised reactivity models and data preparation |
| data/ | Data sources, prepared inputs, fixed splits, features and saved predictions |
| results/miRNA/ | Main three-seed miRBench results |
| results/siRNA_oligo_random/ | Current siRNA benchmark and representation readouts |
| results/accessibility/ | Reactivity MAE analyses and supplementary metrics |
| supplementary/purged_sequence_v1/ | Sequence-purged siRNA inputs, partitions, predictions and verified results |
| scripts/ | Data preparation and analysis reproduction |

## Reproduction

Use Python 3.11 and the scientific dependencies in reference/pairac_concat_v1/requirements-review.txt. Install a PyTorch build appropriate for your hardware if running model code.

    python scripts/verify_release.py
    python scripts/reproduce_tables.py
    python scripts/recompute_mirna_ap.py
    python scripts/reproduce_oligo_results.py --verify-predictions
    python scripts/reproduce_mirna_supervision_control.py
    python scripts/reproduce_shuffled_supervision_control.py

The last two commands verify the initial-versus-trained and equal-budget shuffled-label controls without loading model weights. The [revision file guide](docs/REVIEWER_REVISION_FILES.md) links their code, data, tests and evidence.

The sequence-purged supplement can be verified separately using Python's standard library:

```sh
python supplementary/purged_sequence_v1/scripts/verify.py
```

The prediction checks require the corresponding data. Official miRBench data preparation, source-model evaluation and frozen siRNA refitting are described in [reproduction instructions](docs/REPRODUCIBILITY.md). See [methods](docs/METHODS.md), [results](docs/RESULTS.md) and [data](docs/DATA.md) for the experimental contracts.

Official miRBench code and dataset access instructions are maintained at [katarinagresova/miRBench](https://github.com/katarinagresova/miRBench). The pinned data record is Zenodo 20540907. Prepared source inputs must be checked against the historical identities before claiming exact training reproduction.

## Data and weights

Project-trained weights are available as two locally verified release archives (23.74 MB main and 23.66 MB optional); public hosting is pending. See [weight contents, public-backbone downloads and SHA checks](docs/WEIGHTS.md). Available features and predictions support analysis of their represented samples without loading those weights. New-sequence inference requires the exact source weights and permitted public backbones. Some historical source-training inputs have not been recovered; a newly prepared bank is a matched reconstruction unless its byte identities agree.

The siRNA benchmark reuses OligoFormer-derived preprocessing and random-fold conventions. OligoFormer is computationally retrained and evaluated using the pinned Takayuki `data/Taka.csv`; see the [remeasurement and data audit](docs/OLIGOFORMER_REMEASUREMENT_AUDIT.md). Huesken→Mixset reuses Mixset. Conclusions concern this retrospective protocol, with source exposure, public pretraining and repeated downstream development taken into account.

Citation metadata are in [CITATION.cff](CITATION.cff). Retain the [project notice](LICENSE-NOTICE.md), [data terms](DATA_LICENSE_NOTICES.md) and [third-party notices](THIRD_PARTY_NOTICES.md).
