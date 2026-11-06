# Main results

The siRNA tables follow OligoFormer random outer folds and their Wang/ALAS1 extensions. Results are not mixed with earlier purged-sequence experiments. Source seeds 42, 43 and 44 are evaluated separately; reported uncertainties are sample SDs, not prediction ensembles.

## Contribution of miRNA task supervision

The exact initial-versus-terminal control retains the same public backbones,
structural inputs, frozen architecture and R55 siRNA evaluation. Main-macro PCC
changes from 0.4772 ± 0.0217 to 0.5179 ± 0.0090 for representation-only transfer
(paired difference 0.0407 ± 0.0240), and from 0.6157 ± 0.0015 to 0.6330 ± 0.0054
with TD24 (paired difference 0.0172 ± 0.0062). The three-seed mean improves in
five of the six settings; Huesken→Mixset decreases in both modes.

These are descriptive differences across matched source seeds, with no
independent-fold significance claim. This control estimates the contribution of
miRNA task updates; it retains other pretraining/supervision. The complete
[control report](../results/siRNA_oligo_random/mirna_supervision_control/REPORT.md)
includes all settings, both modes, MSE/MAE, saved predictions and a response draft.
The [new-file guide](MIRNA_CONTROL_FILES.md) lists all added scripts and evidence.

## miRBench v7

Full DuplexFM achieves four-test macro AP **0.878612 ± 0.001418**. Each source seed's AP is averaged equally over the four tests before summarizing across seeds. Per-test and per-seed values are in results/miRNA/.

## siRNA benchmark

| Method | Huesken | Mixset | Takayuki | Huesken→Mixset | Macro4 | Wang | ALAS1 | Macro2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TD24, R55 | 0.6092 | 0.5810 | 0.6507 | 0.5744 | 0.6038 | 0.5311 | 0.5380 | 0.5345 |
| OligoFormer-corrected | 0.6482 ± 0.0008 | 0.5522 ± 0.0189 | 0.7662 ± 0.0049 | 0.5329 ± 0.0166 | 0.6249 ± 0.0101 | 0.4395 ± 0.0246 | 0.5200 ± 0.0330 | 0.4798 ± 0.0247 |
| BERT-siRNA | 0.6082 ± 0.0193 | 0.4220 ± 0.0127 | 0.7511 ± 0.0129 | 0.4876 ± 0.0288 | 0.5672 ± 0.0094 | 0.3089 ± 0.1019 | 0.2662 ± 0.0020 | 0.2875 ± 0.0503 |
| RNAiSpline | 0.6252 ± 0.0054 | 0.5972 ± 0.0030 | 0.7258 ± 0.0365 | 0.5595 ± 0.0118 | 0.6269 ± 0.0109 | 0.6390 ± 0.0555 | 0.5641 ± 0.0333 | 0.6016 ± 0.0314 |
| DSIR, published fixed model | / | 0.5674 | 0.6039 | 0.5674 | / | 0.6620 | 0.6289 | 0.6454 |
| DuplexFM + TD24 | 0.6235 ± 0.0020 | 0.6075 ± 0.0060 | 0.7225 ± 0.0040 | 0.5783 ± 0.0127 | **0.6330 ± 0.0054** | 0.6235 ± 0.0201 | 0.6140 ± 0.0149 | 0.6187 ± 0.0173 |

PCC is calculated from pooled outer-fold predictions within each seed. Macro4 covers the first four settings; Macro2 covers Wang/ALAS1. A slash marks an unavailable or inappropriate comparison. DSIR's published weights used Huesken, so Huesken/Macro4 are excluded; its cross-dataset entry repeats Mixset.

DuplexFM + TD24 reaches Macro4 0.6329608 versus 0.6038249 for matched TD24, a **4.8252% ratio-of-macro-means increase**. This is not the mean of separate per-setting relative changes. OligoFormer-corrected remains stronger on Huesken/Takayuki; DSIR's additional-setting mean is higher. Seed42 inner search for the neural baselines differs from the fixed frozen-representation readout. Common IDs and folds do not equalize all source exposure or optimization budgets.

See results/siRNA_oligo_random/benchmark/ for per-seed, per-fold and model-selection evidence.

## Representation readouts

| Frozen representation | Macro4 PCC | Macro2 PCC |
|---|---:|---:|
| Full, representation only, R55 | 0.5179 ± 0.0090 | 0.5022 ± 0.0797 |
| Full + TD24, R55 | 0.6330 ± 0.0054 | 0.6187 ± 0.0173 |
| Full50 + TD24, PCA16/Ridge | 0.6172 ± 0.0015 | 0.6144 ± 0.0196 |
| Official CNN-M + TD24, PCA16/Ridge | 0.5955 | 0.5326 |
| Official CNN-M160, ordinary Ridge | 0.0502 | 0.1697 |

Representation-only transfer still fits a supervised siRNA readout. The common50nt/PCA16/Ridge comparison is separate from the main target57/R55 protocol. CNN-M has one official source checkpoint, so a three-source SD is not available. The two CNN rows change both dimensionality treatment and TD24; their difference cannot isolate TD24 alone. The main representation and common-reader tables remain under results/siRNA_oligo_random/modules/; the directory name reflects their experimental origin.

## Structural-reactivity MAE

| Supervised predictor | Internal-test MAE | Strict GSE74353 MAE |
|---|---:|---:|
| Sequence CNN | 0.1767 | 0.2041 |
| Sequence + LinearFold-V CNN | 0.1744 | 0.2017 |
| Reconstructed mRNABERT | 0.1621 | 0.1920 |
| Original PairAC reactivity predictor | / | 0.1923 |

The internal test contains 207,979 windows/10,759,454 measured positions. Strict external evaluation contains 7,347 windows/347 genes/404,194 positions. MAE pools measured nucleotides; observed zeros remain, missing/PAD positions do not. The two pools are not averaged.

Primary presentation by MAE was chosen after examining existing results. Checkpoints retain the original validation correlation criterion; none was reselected for MAE. Lower absolute error does not imply superior correlation. Complete metrics remain under results/accessibility/supplementary_metrics/. The original PairAC predictor has missing source-membership records and no certified matched internal score; its external value is descriptive rather than evidence of previously unseen inputs.

## Interpretation

Four primary settings represent three cohorts because Mixset is reused. Sequence neighbors occur across random folds, and source keep-overlap, public pretraining and repeated downstream development remain relevant. These experiments support representation utility under the stated conditions, not universal transfer, a new blind external confirmation or biological causality. Ablation and single-branch implementations are retained, but their results are not reported in this repository.
