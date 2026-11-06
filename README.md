# DuplexFM

Code and data for learning transferable RNA interaction representations from miRNA targeting.

DuplexFM combines contextual guide–target score maps, learned nucleotide-pair embeddings and predicted target structural reactivity. The PairAC encoder produces a 192-dimensional interaction representation. For siRNA efficacy prediction, the miRNA-trained encoder is frozen and a supervised linear readout is fitted to its representation, alone or with 24 siRNA-specific sequence and thermodynamic descriptors (TD24).

## New result: contribution of miRNA supervision

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

**[Full results and reviewer-response draft](results/siRNA_oligo_random/mirna_supervision_control/REPORT.md)**
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

## Organization

| Path | Contents |
|---|---|
| reference/pairac_concat_v1/ | Model, source training, feature extraction, readout and ablation implementations |
| workflows/accessibility/ | Supervised reactivity models and data preparation |
| data/ | Data sources, prepared inputs, fixed splits, features and saved predictions |
| results/miRNA/ | Main three-seed miRBench results |
| results/siRNA_oligo_random/ | Current siRNA benchmark and representation readouts |
| results/accessibility/ | Reactivity MAE analyses and supplementary metrics |
| scripts/ | Data preparation and analysis reproduction |

## Reproduction

Use Python 3.11 and the scientific dependencies in reference/pairac_concat_v1/requirements-review.txt. Install a PyTorch build appropriate for your hardware if running model code.

    python scripts/verify_release.py
    python scripts/reproduce_tables.py
    python scripts/recompute_mirna_ap.py
    python scripts/reproduce_oligo_results.py --verify-predictions
    python scripts/reproduce_mirna_supervision_control.py

The last command verifies all saved predictions and tables for the new matched
control without loading model weights. See the [new-file guide](docs/MIRNA_CONTROL_FILES.md)
for its scripts, data, results, tests and execution evidence.

The prediction checks require the corresponding data. Official miRBench data preparation, source-model evaluation and frozen siRNA refitting are described in [reproduction instructions](docs/REPRODUCIBILITY.md). See [methods](docs/METHODS.md), [results](docs/RESULTS.md) and [data](docs/DATA.md) for the experimental contracts.

Official miRBench code and dataset access instructions are maintained at [katarinagresova/miRBench](https://github.com/katarinagresova/miRBench). The pinned data record is Zenodo 20540907. Prepared source inputs must be checked against the historical identities before claiming exact training reproduction.

## Data and weights

Project-trained weights are distributed separately. Available features and predictions support analysis of their represented samples without loading those weights. New-sequence inference requires the exact source weights and permitted public backbones. Some historical source-training inputs have not been recovered; a newly prepared bank is a matched reconstruction unless its byte identities agree.

Random siRNA folds expose sequence neighbors; Huesken→Mixset reuses Mixset. Source overlap, public-pretraining exposure and repeated downstream development limit claims of independent generalization. Results do not establish biological causality.

Citation metadata are in [CITATION.cff](CITATION.cff). Retain the [project notice](LICENSE-NOTICE.md), [data terms](DATA_LICENSE_NOTICES.md) and [third-party notices](THIRD_PARTY_NOTICES.md).
