# Model identities and training settings

The package restores previously completed runs on one shared set of sequence-purged partitions. Exact identifiers and numeric settings are in `audit/models.json`. The current repository's random-split HPO runs are a separate evaluation and are not reproduced by these historical fixed-configuration results.

## DuplexFM and the TD24 control

PairAC uses the three registered miRNA source training endpoints, seeds 42, 43 and 44, each at eight epochs and 32,728 updates. The source model is frozen for siRNA evaluation. The readout receives TD24 plus the source192 representation. The matched TD24 control uses the same R55 group-intercept and inner-selection treatment. Both select alpha from 0.1, 1, 10, 100 and 1000 using inner OOF PCC within outer fit, then MSE and stronger regularization to break ties; outer validation is unused. Group offsets are omitted at deployment. The archived selection receipts bind final fit/test IDs and chosen alpha. The shared R55 treatment should be distinguished from a plain TD24 ridge baseline.

## OligoFormer reproduction

The reported implementation is the previously registered `OligoFormer_corrected` reproduction. Its exact model definition is included under `reference/`, bound by the historical execution-lock SHA. It uses batch-first packed recurrent sequences, the complete Transformer residual/feed-forward output path and a single sigmoid efficacy output, with fit-only feature normalization. The upstream implementation uses different recurrent tensor semantics, output/attention paths and optimization defaults. Both workflows use continuous efficacy supervision; this is not a claim that the upstream method was exclusively a classification model.

The accepted historical configuration is batch 128, AdamW at 3e-4 with weight decay 1e-4, dropout 0.02, at most 120 epochs, patience 15, MSE min-delta 1e-5 and gradient clipping at one. BF16 forward passes use the registered deterministic ATen recurrent backend. Outer-validation MSE selects the checkpoint. The source identity and these settings allow the result to be interpreted as this specified reproduction, rather than an unqualified score for the authors' original code or published model ceiling. The comparison does not identify which implementation, data partition or optimizer difference explains any score change.

## RNAiSpline and BERT-siRNA

RNAiSpline uses native SSL restricted to the outer fit IDs, followed by supervised training selected on outer-validation MSE. Its accepted records show 60 SSL epochs, supervised training for at most 50 epochs, patience 20 and batch 16. No old task-supervised checkpoint is reused. Each included receipt verifies SSL IDs equal fit IDs and test prediction occurs only after selection.

BERT-siRNA fine-tunes the public DNABERT-6 encoder on guide19 represented by fourteen overlapping six-mers plus CLS/SEP tokens. A sigmoid MLP maps 768 to 128 to 128 to 64 to one output. It uses Adam at 5e-5, weight decay 0.01, dropout 0.1, batch 100, FP32 and at most 30 epochs. Outer-validation MSE selects the checkpoint. These runs used a fixed configuration without HPO.

Both baselines use three siRNA training seeds, share the accepted fit/validation/test ID lists exactly, and retain their own native training schedules. Their recorded checkpoint and prediction hashes are included without distributing weights. RNAiSpline is a close comparator in this four-setting aggregate. No significance claim is made from the small difference in Macro4 mean PCC.

## Reproduction boundary

`python scripts/verify.py` verifies saved-prediction evidence. It does not reconstruct all original runtimes, generate pretrained features, reload every checkpoint or rerun training. The bundled OligoFormer definition requires PyTorch if imported, but the verification command never imports it. Original data and third-party terms remain applicable; the package does not relicense their contents.
