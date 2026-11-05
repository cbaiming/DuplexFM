# siRNA analysis under OligoFormer-derived random splits

`frozen_readout/` preserves the numerical R55 implementation used for the current PairAC comparisons, including its scaling, group adjustment, inner-fold selection and independent numerical certificates. `baselines/` preserves the registered HPO backend, model adapter and analysis implementation for OligoFormer-corrected, BERT-siRNA and RNAiSpline.

`baselines/hpo_specification.json` records the original configuration with machine-specific staging paths replaced. Its source SHA identifies the unchanged original scientific specification. These modules are research implementation references; they expect the original qualified local asset layout and are not stand-alone training launchers. Model and backbone weights are distributed separately.

For analysis of the supplied predictions, use `scripts/reproduce_oligo_results.py --verify-predictions` from the repository root. It checks the input and split identities, exact-once test membership, labels and per-seed PCC before reproducing the reported summaries. This does not retrain the source or siRNA models.

Methods, settings and data availability are described in [Methods](../../docs/METHODS.md) and [Data](../../docs/DATA.md).
