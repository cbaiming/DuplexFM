# Third-party materials and license status

This document records provenance and unresolved release decisions as of the original 2026-10-25 audit. It is not a substitute for the exact licenses accompanying upstream artifacts and does not determine whether a particular downstream artifact is a derivative work.

## Project materials

The existing project [LICENSE-NOTICE.md](LICENSE-NOTICE.md) provisionally marks original materials as Creative Commons Attribution-NonCommercial 4.0 International and expressly excludes third-party datasets, pretrained models, software and other referenced materials. It also leaves the final software-license choice for author confirmation. This candidate does not change that choice or assert that the entire dependency and artifact chain has one common license.

## Backbone and software dependencies

| Material | Role | Recorded status and release action |
|---|---|---|
| [mRNABERT](https://huggingface.co/YYLY66/mRNABERT) | Target encoder, tokenizer and custom model implementation | The exact revision is pinned in [DATA_AND_WEIGHTS.md](docs/DATA.md). Its license was not verified for this candidate. Confirm the applicable terms and attribution before distributing copied implementation files, weights or related model exports. |
| [RNA-FM via MultiMolecule](https://huggingface.co/multimolecule/rnafm) | Guide encoder and tokenizer | The required historical snapshot includes AGPL v3 `license.md` and a separate `license-faq.md`. Preserve and review both exact documents when preparing any distribution. The present upstream revision must not silently substitute for the training snapshot. |
| MultiMolecule Python package | `RnaFmModel`, `RnaTokenizer` and configuration implementation | The source imports the installed package. Retain dependency version information and inspect the corresponding package terms; installing it as a dependency does not assign it the project license. |
| PyTorch, NumPy, SciPy, scikit-learn, Transformers and supporting packages | Training, numerical optimization, data handling and model loading | Version records are in the reference requirements files. These packages are dependencies rather than vendored distributions in this candidate; their own notices and terms remain applicable. |

The historical RNA-FM license artifacts were checked against the local files and agree with the archived model manifest:

| File | Bytes | SHA-256 |
|---|---:|---|
| `license.md` | 34,121 | `e91ab743b5267586856dd817d1fc5933ecea9432c864cd08366152ab647cd055` |
| `license-faq.md` | 22,800 | `2fb65fe94806b24728be77eb7d1cbfab1d6048f342e8df7610fb8c1ee8e2ce68` |

The FAQ accompanying that snapshot discusses distributed model weights, modified/fine-tuned weights and corresponding source, including training materials. Those upstream statements are a reason to review the exact proposed checkpoint/LoRA/data distribution together. This record does not decide their legal application to DuplexFM, and does not conclude that a particular release license or artifact format satisfies them.

The source package references backbone files by hashes and does not vendor their weight files or custom model implementations. Any future mirror must preserve applicable upstream notices, document its source and revision, and expose a verified acquisition route. The mirror's current availability was not tested in this preparation.

## Datasets and comparison methods

miRBench v7, GSE145805-derived structural-reactivity inputs, OligoFormer-associated siRNA tables and any additional comparison datasets retain their respective source terms. A project code notice does not establish permission to redistribute processed copies. The source dataset and transformation provenance must accompany a future permitted download/reconstruction workflow.

Older local release candidates also contained Wang and ENsiRNA patent-derived tables with unresolved redistribution review. They are not included by implication in this PairAC candidate. Baseline implementation repositories and their checkpoints should be referenced with exact versions and citations instead of being copied wholesale without checking their notices.

Before public release, the authors need to finalize the software license, verify the exact backbone and dataset terms, supply complete citations, decide which original and third-party artifacts may be distributed, and test the promised public acquisition routes. The present package makes those remaining decisions explicit while allowing source and aggregate-result review.


2026-11-04 update: see [DATA_LICENSE_NOTICES.md](DATA_LICENSE_NOTICES.md) for the saved official miRBench v7 CC BY 4.0 record and the distinct status of other datasets. No common license is applied to third-party data.
