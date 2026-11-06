# Project weights and public foundation models

Project-trained weights are packaged separately from RNA-FM and mRNABERT. The archives below have been created and verified locally; a public hosting URL has not yet been configured.

| Archive | Compressed size | Included |
|---|---:|---|
| `duplexfm_project_weights_20261106.tar.gz` | 23,738,758 bytes (23.74 MB) | Full source seeds42/43/44; original epoch22 structural-reactivity head and LoRA; portable metadata |
| `duplexfm_optional_control_weights_20261106.tar.gz` | 23,656,568 bytes (23.66 MB) | Exact initial tensors for seeds42/43/44; shuffled-label seed42; separate reconstructed epoch9 predictor |

The main package contains 132 trained task tensors per source seed and the registered fusion-buffer dictionary (empty for these checkpoints). It does not duplicate the frozen public foundation parameters. The compact files support inference and feature extraction; optimizer/RNG state for training continuation is excluded. Fitted siRNA R55 readout coefficients and normalization are separate from source encoder weights.

The inventories are [project_weights_main.json](../audit/project_weights_main.json) and [project_weights_optional.json](../audit/project_weights_optional.json). Tensor bytes are unchanged. The original reactivity LoRA configuration replaces one machine-local base-model path with `public/mrnabert`; the manifest records its original and portable configuration SHA separately.

| Archive | SHA-256 |
|---|---|
| Main | `bbcac083a3a794201ca1c06aa23bbe3d4b9fd1770bfdbc3c4202b8efb8950bdb` |
| Optional | `ec3b206d3c2ffc14261c20ad85f03984c305db0f1f46cf2cf36574deb471db77` |

## Public foundation downloads

The required file identities are fixed in [foundation_reference.json](../reference/pairac_concat_v1/source_full/code/foundation_reference.json). The downloader resolves an immutable upstream commit, then checks every required file's SHA256 before installing it. It refuses a different model version.

```bash
python -m pip install huggingface_hub
python scripts/fetch_public_backbones.py --model mrnabert \
  --mrnabert-revision a1eb7df25804d23f08646e1cb996b234d7208a40 \
  --output external_data/models
```

The current upstream mRNABERT weight-file SHA agrees with the historical reference. The current upstream RNA-FM `model.safetensors` SHA differs from the recorded file identity. This byte-identity check does not establish whether their numerical tensors are equivalent. Exact reproduction therefore requires the recorded RNA-FM historical snapshot or a mirror of those exact files. [The online check](../audit/public_backbone_online_check.json) records both observations. No current upstream file is silently substituted.

A separate historical RNA-FM snapshot has also been prepared for mirror hosting: `rnafm_historical_snapshot_20261106.tar.gz`, 234,649,052 compressed bytes, SHA-256 `e50c371f0aff6dcdc4527a2474f769edd096ac8bb6bc2f7bfec4493b92bebd4a`. It contains the four required files and the upstream license/FAQ, and is separate from both project-weight archives.

After a matching historical RNA-FM mirror is hosted, specify its repository and immutable revision:

```bash
python scripts/fetch_public_backbones.py --model rnafm \
  --rnafm-repo YOUR_HF_REPO --rnafm-revision YOUR_COMMIT \
  --output external_data/models
```

`YOUR_HF_REPO` and `YOUR_COMMIT` are placeholders for the future mirror, not existing download locations. If files are stored below a repository subdirectory, add `--rnafm-prefix SUBDIRECTORY`; the same options are available for mRNABERT. SHA expectations stay unchanged when the hosting location changes. Upstream license notices are retained when supplied by that snapshot.

To verify existing local snapshots without network access:

```bash
python scripts/fetch_public_backbones.py --verify-only --output external_data/models
```

## Using the compact project weights

Extract the project archive outside the repository's tracked source tree. `scripts/reproduce_mirna.py` recognizes the three registered compact checkpoint hashes and verifies their seed, terminal update and original-checkpoint binding. For example, after separately preparing the qualified PairStore and structural cache:

```bash
python scripts/reproduce_mirna.py evaluate \
  --assets external_data/models --store external_data/pairac_store \
  --seed 42 --checkpoint external_data/duplexfm_project_weights_20261106/source/seed42/terminal.pt \
  --device cuda:0 --amp bf16 --output ../mirna_compact_seed42
```

The packaged seed42 compact model has been loaded on CPU and checked against one archived source192 feature row, with maximum absolute difference 1.91e-6. This check reuses the exact frozen structural profile; it is distinct from regenerating raw-sequence structural profiles. See [project_weights_load_check.json](../audit/project_weights_load_check.json), [the reproduction instructions](REPRODUCIBILITY.md) and [the structural checkpoint identities](ICSHAPE_CHECKPOINTS.md).
