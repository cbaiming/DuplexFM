# Data and model assets

Source terms are recorded in [DATA_LICENSE_NOTICES.md](../DATA_LICENSE_NOTICES.md). Project-trained weights are distributed separately. Upstream backbones, datasets and software retain their own conditions.

## miRBench v7

The official [miRBench repository](https://github.com/katarinagresova/miRBench) provides the dataset access and preprocessing reference. The pinned dataset record is [Zenodo 20540907](https://zenodo.org/records/20540907), licensed CC BY 4.0. The official GitHub interface downloads the data from Zenodo; GitHub code access does not imply that every large data file is stored in Git.

data/mirbench_v7/SOURCE.json records filenames, sizes, publisher checksums and available local files. The four complete bundled gzip files total 6,629,782 bytes: Hejret train/test, Klimentova test and Manakov left-out. Complete Manakov train/test require acquisition through the fixed source. Partial files are not treated as complete. Fetching and preparing are separate steps.

The retained Full prediction banks at data/mirna_predictions/seed42, seed43 and seed44 contain 135 NPZ shards across four tests, supporting 12 AP recalculations. Row-level source predictions do not reconstruct source training inputs. The exact historical PairStore arrays and matching reactivity cache have not been recovered. A new preparation must be described as a matched reconstruction unless its row/split/token/cache identities match the historical manifest.

## siRNA random-split benchmark

The current prepared data are under data/sirna_oligo_random/.

| Path | Contents |
|---|---|
| input_rows.json | Sample IDs, guide19, target57, TD24, continuous response and binary label |
| native_split_plan.json | Ordered outer train/test membership and five inner folds |
| input_manifest.json | Cohort identities, original split universe, eligibility and checksums |
| predictions/ | Main benchmark and retained representation-readout predictions |
| features/ | Available frozen Full source representations for seeds42/43/44 |

There are 3,688 unique rows: Huesken 2,196; Mixset 469; Takayuki 664; Wang 131; ALAS1 228. Across 26 outer groups, each complete model/seed produces 4,157 predictions because Huesken→Mixset evaluates Mixset again.

Original cohorts follow two seed42 shuffles of the OligoFormer raw rows before eligibility filtering. Wang/ALAS1 are same-rule extensions. Inner folds only use outer training rows. Prepared response values and labels are copied unchanged; do not apply normalization again. Preserve representation array order and sample IDs during refitting.

These folds contain sequence neighbors and are not gene-isolated tests. Four primary settings are not four distinct cohorts. The random-split results cannot be combined with earlier differently split experiments.

## Accessibility data

Reconstructed GSE145805 supervision has 2,332,794 windows: train 1,928,795, validation 196,020 and test 207,979. The data-only compressed archive has 200,623,792 bytes and SHA-256 54d31587b33697fa9c9bcfa416ce284b658b646cc3773842bd97b80237e02813. Its three pieces, each at most 95,000,000 bytes, reside under data/large/icshape_source/. The index binds each piece and extracted member. Restored arrays/provenance total 857,860,306 bytes; no model state is included.

    python scripts/restore_data.py --dataset icshape_source --output external_data/icshape_source

The external GSE74353 data use continuously measured 50–60-nt windows. A final screen removes local matches with ≥90% identity and ≥80% true coverage of both sequences, in either orientation, against reconstructed train plus validation. It retains 20,208 sequence-filtered windows, or 7,347 windows/347 genes after also excluding seen genes. Masks and per-window MAE statistics are retained. Filtering is heuristic and does not exhaustively certify homology.

PairAC's original epoch22 reactivity predictor and the reconstructed epoch9 model are different. The original predictor's supervised membership is missing; purging against reconstructed supervision cannot establish that the original model never encountered the external inputs.

## Weights and inference assets

RNA-FM and mRNABERT identities are preserved in reference/pairac_concat_v1/source_full/code/foundation_reference.json. Recorded upstream sources are [RNA-FM](https://huggingface.co/multimolecule/rnafm) and [mRNABERT](https://huggingface.co/YYLY66/mRNABERT). Changing a historical revision changes the reproduction identity.

MODEL_REGISTRY.json identifies the Full source checkpoints and readout. Cached features allow analysis of their saved samples; new-sequence inference requires exact task weights, public encoders and the correct reactivity predictor/cache. A matching architecture or random initialization cannot replace those assets while reproducing a reported score.

Prepared siRNA and accessibility data are included at the paths above. Their original source-specific terms and outstanding redistribution review remain documented in DATA_LICENSE_NOTICES.md; combining files does not supply new rights. Project weights and the historical source reactivity cache remain separate required assets.
