# Reproducing the main experiments

Use Python 3.11 and the pinned scientific dependencies in reference/pairac_concat_v1/requirements-review.txt. Install an appropriate PyTorch build for model execution. Run commands from the repository root.

## Integrity and saved results

    python scripts/verify_release.py
    python scripts/reproduce_tables.py
    python scripts/recompute_mirna_ap.py
    python scripts/reproduce_oligo_results.py --verify-predictions
    python scripts/reproduce_accessibility_mae.py

The first two check file identities and aggregation. AP verification checks Full seeds42/43/44 on four tests, using retained labels and scores. siRNA verification checks the current Oligo plan, exact-once IDs and pooled per-seed metrics. MAE verification uses the retained observed-position error sums. Required prediction/statistic data must be installed first. These commands neither retrain source models nor use test results to select parameters.

## Official miRBench data and prepared inputs

    python scripts/fetch_mirbench.py --download
    python scripts/prepare_mirbench.py --raw-dir external_data/mirbench_v7 --output external_data/pairac_store --tokens-only

Then attach the separately supplied exact cache:

    python scripts/prepare_mirbench.py --output external_data/pairac_store --complete-existing --accessibility-cache /path/to/accessibility.npy

Load a separately supplied terminal checkpoint for one source seed:

    python scripts/reproduce_mirna.py evaluate --store external_data/pairac_store --assets /path/to/models --seed 42 --checkpoint /path/to/terminal.pt --device cuda:0 --amp bf16 --output ../mirna_seed42

Repeat evaluation for seeds43 and44. Full source-training commands and asset identities are in [the source workflow](../workflows/mirbench/README.md).

The fetch interface follows the pinned official [miRBench repository](https://github.com/katarinagresova/miRBench) and Zenodo 20540907, preserving publisher filenames/checksums. Preparation converts official rows into the documented sequence, identity, label and split inputs. Inspect the CLI and provide the actual local data and permitted assets.

The exact original cleaner and187-family assignment file are included. Preparation verifies all12 original token/map/split array SHA values and only publishes a complete PairStore after the thirteenth, historical reactivity-cache SHA also agrees. Full five-file rebuilding and encoder inference were not executed during packaging. The saved-prediction AP check remains distinct from fresh encoder inference. Fresh inference requires separately distributed task weights, public backbones and the correct reactivity model/cache; it must not silently fall back to random parameters.

## Frozen siRNA readout

    python scripts/refit_oligo.py --seed 42 --setting Huesken --fold 0 --output ../oligo_fold_check

Explicitly refit all26 outer groups for the three source seeds:

    python scripts/refit_oligo.py --seed 42 43 44 --all --output ../oligo_full_refit

The default source mode includes TD24+source192. The one-fold CPU qualification reproduced431 saved predictions to a maximum absolute difference of2.22e-16.

Install data/sirna_oligo_random/ and its matching frozen Full features. Refitting uses original sample IDs, outer/inner membership, responses and target25 groups. The numerical objective is preserved under reference/pairac_concat_v1/sirna_transfer/runtime/sirna_increment_20261014/round55/. Alpha uses only inner training/validation predictions. Do not estimate transformations or test-group offsets on outer-test data.

Saved-prediction verification and readout refitting answer different reproducibility questions. Neither generates source representations for new sequences. Three source seeds are scored separately; averaging predictions would be a different experiment.

## Matched miRNA-supervision control

The [control protocol](MIRNA_SUPERVISION_CONTROL.md) provides a portable entry
point for exact initial-versus-trained frozen encoders. It retains R55 and
verifies trained predictions against both the TD24-plus-source and source-only
archives before reporting the new control.

    python scripts/reproduce_mirna_supervision_control.py
    python scripts/run_mirna_supervision_control.py --features-manifest data/sirna_oligo_random/mirna_supervision_features.json --seed 42 43 44 --all --output ../supervision_control_refit

## Accessibility comparison

    python scripts/restore_data.py --dataset icshape_source --output external_data/icshape_source
    python scripts/stage_accessibility.py --data external_data/icshape_source/data --workdir external_data/cnn_run

Restoration checks every piece and member; staging does not train. Original CNN training/loss/selection code is under workflows/accessibility/cnn/. Set ICSHAPE_DATA to the restored absolute data directory. LinearFold-V additionally requires its qualified source/build and DUPLEXFM_LINEARFOLD_ROOT.

The supervised loss is masked per-window L1. The original validation checkpoint rule uses pooled and within-window PCC, with regional MAE near ties. MAE is a post-hoc presentation focus; selection logic is not changed. Full results remain under results/accessibility/supplementary_metrics/. Keep original PairAC and reconstructed predictor identities separate.

## Implementation tests

    python scripts/run_cpu_checks.py
    python examples/synthetic_embedding.py

Synthetic checks exercise masking, coordinates, feature construction, pooling and numerical solver behavior in isolated processes. Ablation code tests may remain even though their experimental results are outside the reporting scope. The embedding example uses random parameters and does not demonstrate pretrained biological prediction.

## Scope

Available data support saved-result verification and frozen-feature analyses for represented samples. Separately distributed weights, the exact historical reactivity cache and an executed full preparation/inference run remain necessary for complete raw-sequence performance reproduction. The cleaner, fixed family assignments and token/map/split hashes are now included. Preparing an equivalent-format dataset is not proof of that identity. Public-pretraining membership, random-split neighbors and repeated downstream development remain scientific limitations even after numerical agreement.
