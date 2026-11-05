# Data provenance and license notices

Data provenance for the DuplexFM PairAC main experiments. Paths below are relative to the repository root. This document distinguishes recorded source terms from items awaiting source-specific review; it does not assign one project license to every dataset or model artifact.

## Project materials

The existing [LICENSE-NOTICE.md](LICENSE-NOTICE.md) provisionally marks original project materials as **CC BY-NC 4.0**, with the final software-license choice pending author confirmation. It expressly excludes third-party datasets, pretrained models and software. This release preserves that status. See also [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

No pretrained backbone weights, PairAC task weights, accessibility checkpoints, LoRA exports or optimizer states are included. This is the authors' current release choice. Cached representations and saved predictions are data artifacts; they do not provide a pretrained model for new sequences.

## Materials included in the public repository

| Material | Recorded source terms | Provenance and action |
|---|---|---|
| Four official miRBench v7 gzip files in `data/mirbench_v7/raw/` | **CC BY 4.0**, recorded in the official Zenodo metadata | Cite [miRBench datasets, record 20540907](https://doi.org/10.5281/zenodo.20540907). `data/mirbench_v7/SOURCE.json` retains source creators, URLs, publisher checksums, availability and SHA-256 identities. The four source files are unchanged; publisher MD5, size and decompression checks passed. |
| miRNA row-level prediction banks in `data/mirna_predictions/` | Underlying miRBench labels retain their source attribution; project-generated predictions are separate original outputs | Keep the dataset citation and the condition, seed, test-set and row identities. These saved outputs were copied without rerunning inference. Their presence does not imply availability of all underlying source-training inputs. |
| Aggregate result tables and analysis code | Original project notice, with third-party terms remaining separate | Preserve result provenance, protocol names and limitations. Metrics and model-generated outputs should not be presented as original experimental measurements. |

The optional positive-only miRBench full-dataset file is not part of the required official supervised splits. Two required official split files are still missing from this repository; see [Data description](docs/DATA.md). Incomplete downloads are excluded.

## Included derived data and source-specific review

The following prepared assets are included in this combined bundle at the repository paths below. Source-specific redistribution review is still required before public distribution. Bundling files does not establish permission or change attribution and distribution terms. An unresolved record is not a finding that redistribution is necessarily prohibited.

| Repository destination | Source or material | Review needed before public merge |
|---|---|---|
| `data/sirna_oligo_random/input_rows.json` | Huesken, Mixset, Takayuki, Wang2009 and ALAS1 patent-derived benchmark tables | Record the exact upstream sources, dataset-specific terms and citations. OligoFormer-associated tables and patent-derived material are not covered by the project code notice. Preserve filtering and response-transformation provenance. |
| `data/sirna_oligo_random/` | Frozen split/group/row identities, cached features and per-row predictions derived from those benchmarks | Carry the relevant source notices forward. Record the identity-preserving transformations and the relationship between original and exported file hashes. |
| `data/accessibility/external/` and `data/accessibility/mae_evidence/` | GSE74353-derived windows, measurements, masks and error summaries | Record the measurement-source terms and the terms of the matching transcript reference/annotation, together with coordinate and filtering provenance. [GSE74353](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE74353) availability alone is not a dataset-specific license declaration. |
| `data/large/icshape_source/` | Reconstructed GSE145805 supervision arrays and provenance, packaged without weights | Record terms for [GSE145805](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE145805) measurements and the sequence reference/annotation. Identify this as the reconstructed dataset, rather than the missing historical PairAC accessibility dataset. |

The multipart index records archive, part and extracted-member hashes. Splitting or compressing a dataset changes its packaging, not its source terms. Keep the index and any applicable notices with every published part.

## Backbone and tool notices

Exact backbone identities are documented in [Data and model assets](docs/DATA.md). The mRNABERT distribution terms remain to be verified for the intended release scope. The historical RNA-FM snapshot includes AGPLv3 and a separate license FAQ containing statements about adapted weights, corresponding source and training data. Preserve and review those exact documents before any future weight distribution; a small adapter export does not itself settle that review.

The external sequence filter uses CD-HIT-EST-2D 4.8.1 with a documented boundary correction. Retain the upstream version, license, attribution, patch identity and regression fixtures when distributing its modified source or binary. Python packages and other external tools retain their own licenses.

## Release and reproducibility scope

The [RECOMB 2027 call for papers](https://recomb.org/recomb2027/call_for_papers.html) requires publicly accessible source code with a sufficiently documented README by the submission deadline, and a clear link from the paper. That requirement is separate from demonstrating complete performance reproduction from a fresh checkout. Do not describe code availability, saved-prediction checks or cached-feature readout reproduction as proof of fresh source training or new-sequence inference.

Before public distribution, finish the source-specific notice table, preserve the frozen scientific identities, and rerun the declared integrity and data checks. No license decision should silently change the reported model, dataset or evaluation protocol.


## Additional source-term check, 2026-11-04

The current official OligoFormer repository's [LICENSE](https://github.com/lulab/OligoFormer/blob/main/LICENSE) is a proprietary software license with a redistribution restriction. This package does not copy its executable/model implementation as project-owned software. This observation does not establish a separate license for each underlying published experimental dataset; those tables remain in the source-specific data review group. Do not describe the whole upstream repository as MIT/open-source or assign its terms to unrelated original measurements without checking.

The modified CD-HIT source in `third_party/cdhit/` carries its separate upstream GPLv2 license and modification notice. The project license notice expressly excludes third-party code/data/models.

## OligoFormer-derived random-split evaluation

The current siRNA supplement adds unchanged input identities, frozen random/inner-fold memberships, predictions and three concatenated Full source feature arrays under data/sirna_oligo_random/. It uses the same underlying measured cohorts as the earlier study, with source-specific redistribution review still required. Different splits or derived representations do not supply a new license for original measurements. Project-trained weights are excluded.
