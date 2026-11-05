# Portable miRBench → PairAC source workflow

This workflow supplies the missing source-data preparation and explicit train/evaluate/predict entry points. It uses the preserved current PairAC model, not the earlier five-expert DuplexFM/P13 model. No project checkpoint is downloaded automatically.

## Data preparation

Install the tested CPU preparation dependencies first:

```bash
python -m pip install -r workflows/mirbench/requirements-preparation.txt
```

Obtain all five pinned official miRBench v7 gzip TSVs using the repository's acquisition script, then supply their directory:

```bash
python scripts/prepare_mirbench.py --raw-dir external_data/mirbench_v7 --output external_data/pairac_store --tokens-only
```

`--tokens-only` is an explicit incomplete preparation mode. It writes token/map/index arrays and `tokens_complete.json`, with `PairStore_ready:false`; it does not create a usable manifest. Complete the store only with the exact historical separately frozen reactivity cache:

```bash
python scripts/prepare_mirbench.py --output external_data/pairac_store --complete-existing --accessibility-cache /user/supplied/accessibility.npy
```

Alternatively provide `--accessibility-cache` during initial preparation. Every rebuilt token/map/split array must match the archived SHA-256 contract in `data/mirbench_v7/historical_inputs.json`; the reactivity cache must match its exact archived SHA before a complete `manifest.json` is emitted. There is no synthetic, zero-filled, newly trained or reconstructed-supervision substitute for that cache.

The fixed family assignment file is copied unchanged: SHA256 `16945472723531c1f6de1f840fbff45d15e3b3c481db419880bcbfbde384e377`, 187 families. The extracted cleaner definitions are verbatim from the original module with SHA256 `c932664a7d2f9a4eb9000f738e91de7c496c4be05c5208e76765f866a78419b1`; `provenance.json` records function-level hashes. Only unrelated imports were removed. Original project license notices apply through the root LICENSE-NOTICE.md.

Order is Manakov train, Klimentova test, Hejret test, Manakov test, Manakov left-out. The original cleaner normalizes RNA, removes conflicting-label pairs and duplicate pairs within each frame, retains all feature categories, and preserves first occurrence. Training reuses the saved family assignments; no new split is sampled and source/test exact overlaps are retained. Target/guide factorization preserves first occurrence across the entire ordered table. Target sequences are50nt, stored with A/C/G/U0/1/2/3; guide tails use PAD4. PairStore centers targets within width60 only at collation.

Expected total rows2,841,732; unique targets1,266,774; unique guides1,562; fit2,094,107; validation401,428. All12 token/map/split array identities are checked, not merely these counts. No physical interaction arrays are built or opened.

## Required local model assets

`--assets` must contain exact `mrnabert/` and `rnafm/` model directories. The current reference `foundation_reference.json` validates per-file hashes including model implementation, tokenizer, configuration and weights. Model names alone are insufficient. Project task weights are user-supplied with `--checkpoint`; default expected hashes identify historical seeds42/43/44. A new-execution terminal can be supplied with its explicit `--checkpoint-sha256` only when this portable wrapper's saved signature also matches the requested source seed, original protocol, architecture, source input arrays, source-code files and foundation identities. A matching caller-provided hash alone does not qualify arbitrary checkpoints.

The historical target snapshot is YYLY66/mRNABERT revision a1eb7df25804d23f08646e1cb996b234d7208a40. The guide snapshot used multimolecule/rnafm revision bcf7cc6e8f4385449c0288b3e1f41f21d68f1be3; legacy acquisition records identify a checksum-equivalent Baiming123/DuplexFM_RNA-FM mirror revision49a31010359f645fcfb4a5266b2f0db34a37da8a. This implementation task did not test current network access. A newer upstream RNA-FM snapshot is not an equivalent LoRA base.

The deployed reactivity predictor is a separate frozen five-cell Two-View model. Known local component identities are recorded in `reactivity_registry.json`. Its exact historical cache is still required here; this workflow does not silently regenerate it with a different precision, hardware stack or supervision pool. The reconstructed HepG2 comparison model is a different checkpoint. Original reactivity supervision membership is unresolved.

## Source training

```bash
python scripts/reproduce_mirna.py train --store external_data/pairac_store --assets /user/local/models --seed 42 --device cuda:0 --amp bf16 --output /user/new/run_seed42
```

Repeat for seeds43 and44 as separate runs.

The preserved `training_hourly.run_engine` supplies the original BCE, AdamW settings, two learning-rate groups, clipping, exact batch schedule, initial state, optimizer/RNG/cursor checkpoints, fixed32,728-update/eight-epoch terminal, and validation. It does not select the best validation checkpoint. Global batch512 is fixed; `--micro-batch` defaults512. Changing it preserves sample-weighted gradient accumulation but can change dropout streams. A CPU `--device cpu --amp fp32` execution is a declared precision change.

This wrapper creates a new execution and does not reconstruct the historical two-H100 scheduler, GPU leases, timing budget or source initialization bytes. Cross-hardware/software bitwise equality is not established. Use `--resume` only for this wrapper's existing exact signature/checkpoint. Creating `STOP_REQUESTED` inside the run directory requests a checkpointed stop; deleting that file is necessary before resuming. The source engine saves every500updates; the wrapper does not promise the old controller's independent hourly backup service.

## Regenerate real miRNA predictions and AP

```bash
python scripts/reproduce_mirna.py evaluate --store external_data/pairac_store --assets /user/local/models --seed 42 --checkpoint /user/supplied/terminal.pt --device cuda:0 --amp bf16 --output /user/new/evaluation_seed42
```

This loads the actual RNA encoders, task LoRA/projections/pair embedding/fusion state, and checks terminal step32,728. It predicts the four official tests in their exact saved row order, writes8192-row NPZ shards with rows/labels/scores and per-test metrics. It is real miRNA interaction prediction, not zero-shot siRNA efficacy.

For an explicit small CPU qualification only:

```bash
python scripts/reproduce_mirna.py predict --store external_data/pairac_store --assets /user/local/models --seed 42 --checkpoint /user/supplied/terminal.pt --partition AGO2_CLASH_Hejret2023_test --limit 4 --device cpu --amp fp32 --batch-size 1 --output /user/new/cpu_qualification
```

`--limit` marks partial qualification and cannot be compared as a complete benchmark. FP32 output is labeled as a precision change from historical BF16. Optional `--compare-ap reference.json --ap-atol 1e-6` requires complete four-test evaluation. The JSON format is `{ "per_test_AP": { "TEST_NAME": { "42":0.0,"43":0.0,"44":0.0 } } }` (replace with actual expected values); a flat test→value map is also accepted. Failed comparisons produce no complete marker. Predictions do not average source seeds.

## Verification boundary

The implementation was tested on synthetic ordered frames and the complete official Hejret test (1,018rows,495positives). Exact family file and extracted source function hashes were verified. Full five-file preparation and the1.27million-target reactivity cache were not rerun in this task, and no RNA encoder/source training was executed. Until missing raw files, exact cache and user-supplied weight assets are available, do not claim fresh-checkout performance reproduction. The scripts fail explicitly on those missing inputs.
