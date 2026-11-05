# Structural-reactivity prediction: MAE comparison

This analysis evaluates continuous icSHAPE reactivity prediction. MAE pools observed nucleotide positions, retaining measured zeros and excluding missing/PAD positions. Lower is better; internal and external cohorts are not averaged.

| Predictor | Internal-test MAE | Strict GSE74353 MAE |
|---|---:|---:|
| Sequence CNN | 0.1767 | 0.2041 |
| Sequence + LinearFold-V CNN | 0.1744 | 0.2017 |
| Reconstructed mRNABERT | 0.1621 | 0.1920 |
| Original PairAC reactivity predictor | / | 0.1923 |

The internal test has 207,979 windows and 10,759,454 measured positions. The external subset has 7,347 windows from 347 genes and 404,194 measured positions. Windows span 50–60 nt.

All three traceable supervised models use the same source split, masked per-window L1 loss and validation selection criterion: 0.5 pooled PCC + 0.5 mean defined within-window PCC; near ties within 1e-4 use regional MAE. Selected epochs are CNN18, structure-CNN16 and reconstructed mRNABERT9. No checkpoint was reselected for MAE. Architecture, pretraining, batch and optimization budgets differ.

MAE was chosen as the primary presentation measure after inspecting existing results. It supports a comparison of absolute reactivity error, not a claim of superior correlation or secondary-structure accuracy. Original validation logic and all recorded metrics remain under results/accessibility/supplementary_metrics/.

The original PairAC epoch22 predictor is not the reconstructed epoch9 model. Its historical membership table is missing, so the filtered external inputs cannot be certified unseen for that predictor; matched internal-test MAE is unavailable. Do not substitute reconstructed internal performance for it.

The final external filter rejects local matches with ≥90% identity and ≥80% coverage of both true sequences, in either orientation, against reconstructed supervised train plus validation, followed by seen-gene exclusion. Public-backbone pretraining exposure remains unknown. GSE74353 had already been examined; the result is not a new blinded test.

To verify observed-position error summaries:

    python scripts/reproduce_accessibility_mae.py

CNN/model code is under workflows/accessibility/ and reference/accessibility_rebuilt/. Data and restoration instructions are in [DATA.md](DATA.md); original loss/selection details are retained in [Methods](METHODS.md#supervised-structural-reactivity-regression).

The default MAE verifier recomputes two internal CNN results and all four supervised/original-model external results from SHA-pinned per-window error sums. Validation and reconstructed-mRNABERT internal MAE remain retained aggregate cells until their raw predictions are supplied. To score regenerated profiles independently, use python scripts/reproduce_accessibility_mae.py --predictions predictions.npz, with labels, predictions and true lengths arrays. Missing/PAD are excluded and measured zeros are retained.
