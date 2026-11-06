# Evaluation protocol

Protocol ID: `purged_sequence_v1`. Partitions were registered before the evaluation scores were calculated. The release preserves the accepted master CSV and partition bytes and their original SHA256 identifiers.

Inputs are 5-prime to 3-prime RNA, with T converted to U. Eligible inputs have a canonical 19 nt guide and 57 nt target. The target design site is `target57[19:38]`, equal to the guide reverse complement. Non-ACGU input rows were excluded under the existing input contract. The continuous response and binary truth are distinct original columns; binary truth is never reconstructed by thresholding the response.

## Partition construction

Two samples are directly related when guide Hamming distance is at most two, or the target57 strings share a canonical 25-mer. A 25-mer and its reverse complement use the same lexicographically minimal identity. Identical guide19 plus target19 pairs form indivisible hard groups. The registered construction orders each sequence-connected component by reverse Cuthill-McKee, orders components by size with a stable hash tie-break, then divides cumulative positions into five contiguous outer blocks. Split seed 20261007 affects identity ordering only. Response labels do not construct the partitions.

Each outer test block is fixed first. Its directly related candidates are removed from the training side. Approximately 10% of the remaining candidates, in complete hard groups and fixed sequence order, form validation; direct neighbors of validation are then removed from fit. Purged IDs and reasons are recorded explicitly. For Huesken to Mixset, all eligible Mixset samples form test and Huesken candidates are purged before fit/validation are constructed. The verifier rebuilds the direct sequence relation independently from the input strings.

This protocol guarantees no direct neighbor edges between fit, validation and test. It does not isolate entire transitive sequence components and does not hold out genes or transcripts. Indirect sequence paths may cross sets. It differs from the earlier full-component protocol, whose September P13 results are not included here. No test sample is silently dropped: each eligible sample is tested exactly once per within-domain model seed, and the fixed cross-domain test has all 469 Mixset samples. See `results/split_audit.csv` for actual training budgets in every fold.

## Prediction and metrics

All methods share the exact test identities, outer fit identities, validation identities and binary truth. PairAC and TD24 use fixed R55 model selection within outer fit and leave outer validation unused. OligoFormer corrected, RNAiSpline and BERT-siRNA use outer-validation MSE to choose their supervised checkpoint. Their label-use and optimization budgets therefore differ. Every test set is reserved for the final selected model prediction.

Predictions are retained as saved, without clipping or calibration. The standardized `label` column uses the exact master response; `source_label` preserves the response serialized in each originating prediction file. The verifier permits at most 1e-7 absolute serialization difference but does not alter the prediction scores. This avoids treating CSV roundoff as a change of measurement.

PCC is centered Pearson correlation; Spearman uses average ranks for tied values. MSE, RMSE and MAE use continuous responses. AP is average precision using distinct score thresholds, not trapezoidal PR area. AUROC uses average ranks for ties. F1 uses score >= 0.5 with the original binary truth. Undefined correlations and single-class classification metrics are blank. The verifier implements these definitions using the Python standard library.

Primary `pooled_oof` summaries pool all five within-domain test folds before computing each seed's metrics. Secondary `fold_mean` summaries first compute each fold and then use equal fold weights. Cross-domain summaries use the single fixed test fold. Per-seed Macro4 averages four setting-specific metrics equally. The three-seed summaries report arithmetic mean and sample SD (ddof=1), not a metric computed after averaging predictions. The deterministic TD24 control is not replicated into artificial seed runs.

## Scope

This is a reproducibility supplement for existing development data. It is not a new blind benchmark. Source-training overlap/context exposure and repeated development remain limitations. Source pretraining supervision, native input features and optimization budgets vary between methods. Seeds measure different sources of variation: PairAC source initialization versus siRNA training initialization for the neural baselines. Their SDs are descriptive and do not establish statistical significance. The two Mixset uses are dependent. No causal claim about a single architecture change or split effect follows from these tables.
