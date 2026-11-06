# Experimental methods

## Interaction representation

The PairAC architecture directly concatenates eight contextual guide–target score maps, four learned ordered-base-pair embedding channels, two target structural-reactivity channels and two coordinate channels. The guide and target backbones are RNA-FM and mRNABERT. Public base weights remain frozen; LoRA, projections and the interaction encoder are trained on the miRNA task. The separately trained reactivity predictor remains frozen. The reactivity channels encode its per-position prediction and local contrast.

All 16 A/C/G/U ordered base combinations have trainable four-value lookup entries. Embedding values do not encode preassigned biological categories. Invalid and padded positions are masked. Local convolution produces 32 channels; antiparallel pooling uses mean and maximum statistics in three regions to obtain a 192-dimensional site representation. A site encoder and attention pooling aggregate candidate-site information for miRNA classification. BIO, IntaRNA, manually specified product channels and MoE routing are absent.

For siRNA, the 19-nt guide's designed site occupies target57 coordinates 19:38. Centered padding to width 60 retains original nucleotide coordinates. Transfer uses the frozen designed-site representation, not a miRNA classification probability. Model implementation and coordinate handling are under reference/pairac_concat_v1/source_full/code/.

## miRNA supervision

The historical miRBench v7 all-region contract contains 2,094,107 training and 401,428 validation rows with the original keep-overlap partitions. Source seeds are 42, 43 and 44. Fixed training lasts 32,728 updates, corresponding to eight reference epochs. Batch size is 512 for training and evaluation; precision is BF16. Main/LoRA learning rates are 3e-4/2e-5, weight decay 1e-4 and gradient clipping norm 1.0. LoRA rank, alpha and dropout are 8, 16 and 0.05. The fixed terminal checkpoint is used.

Four tests are evaluated: Hejret AGO2-CLASH, Klimentova AGO2-eCLIP, Manakov AGO2-eCLIP and its left-out partition. AP is computed per test and averaged equally within source seed; the reported mean and sample SD summarize three source seeds. Tests do not choose a training epoch.

Official data are accessed through the miRBench repository and its fixed Zenodo record. Newly prepared token/row/split/cache arrays must match historical manifests before they can replace the original source bank. Otherwise they define a matched reconstruction.

## Oligo random-split siRNA evaluation

Original OligoFormer raw rows are shuffled twice with seed42 before the accepted eligibility filtering. The retained cohorts are Huesken 2,196, Mixset 469, Takayuki 664, Wang 131 and ALAS1 228, totaling 3,688 distinct inputs. Huesken, Mixset and Takayuki use five outer folds; Huesken→Mixset uses one cross-dataset evaluation. Wang/ALAS1 are fivefold extensions, not original paper experiments.

The 26 outer groups yield 4,157 predictions per model/seed because Mixset is evaluated again under transfer. Five inner folds are constructed from each outer training set, with seed42 shuffling. Test labels do not select parameters. Random folds neither purge sequence neighbors nor isolate genes.

### Frozen readout

The primary supervised readout combines TD24 and source192. Scaling is fitted only on the relevant training partition. R55 uses a 1% median-standard-deviation floor, a source divisor sqrt(12), training target25-group intercepts penalized with n/K, and a nuclear penalty on the 6 × 32 source-coefficient matrix at ratio 0.1. Deployment uses a global intercept without estimating offsets for test groups.

Alpha is selected from 0.1, 1, 10, 100 and 1000 by inner out-of-fold PCC, with MSE and stronger regularization as tie-breaks. Representation-only transfer removes TD24 during fitting; its unchanged numerical interface uses 24 zero columns whose fitted coefficients remain zero. No output from another predictor is stacked.

The supplementary common-reader comparison uses one official CNN-M checkpoint and Full target50 representations. Both use train-only PCA16 and ordinary Ridge with inner-MSE selection. This differs from the principal target57/R55 protocol. The unprojected CNN-M160 readout is separately identified.

### Matched miRNA-supervision control

The exact pre-update task state of each Full source seed is frozen and compared
with its matched terminal state, retaining public foundation weights and the
same independently supervised structural-reactivity input. Both source192-only
and source192-plus-TD24 use the identical fixed Oligo/R55 protocol. The initial
state removes miRNA task updates while retaining other pretraining/supervision.
The [control protocol](MIRNA_SUPERVISION_CONTROL.md) documents checkpoint binding,
A6000 FP32 parity, train-only transformations and descriptive paired reporting.

### Other siRNA methods

OligoFormer-corrected, BERT-siRNA and RNAiSpline use the same eligible inputs and outer/inner membership. Registered seed42 inner search selects learning rate, weight decay and batch, then fixes them for final seeds42/43/44. Corrected implementations are not unchanged author code. Their siRNA-training SD and DuplexFM's source-training SD quantify different variability.

DSIR uses published Huesken-supervised weights. Huesken and the four-setting mean are therefore excluded from an independent test; Huesken→Mixset repeats the same Mixset prediction. A three-seed SD is not assigned to this fixed model.

PCC, Spearman, MSE and MAE use pooled out-of-fold predictions per model, seed and setting. The four principal settings and two extensions are averaged separately within each seed before reporting mean/sample SD. Neither fold PCCs nor different seeds' predictions are averaged to construct the main PCC.

## Supervised structural-reactivity regression

The sequence CNN, sequence + LinearFold-V CNN and reconstructed mRNABERT predictor share masked per-window L1 training and the same validation checkpoint criterion: 0.5 pooled PCC + 0.5 mean defined within-window PCC, with regional MAE breaking ties within 1e-4. This correlation-based criterion selects checkpoints; it is not the differentiable training loss.

Source supervision aggregates GSE145805 cells. Strict GSE74353 evaluation uses continuous measured 50–60-nt windows; high-similarity matches to supervised train plus validation and then seen genes are removed. MAE is pooled over observed positions, including measured zeros and excluding missing/PAD. Using MAE as the primary presentation measure was a post-hoc reporting choice; original selection code and full metrics remain available.

The reconstructed epoch9 predictor and PairAC's original epoch22 predictor are different. The original supervised membership table is missing. Filtering against reconstructed supervision cannot certify previously unseen inputs for the original model.
