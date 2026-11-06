# icSHAPE checkpoint identities and reporting

**The reactivity predictor used by DuplexFM is the original epoch22 checkpoint. The reconstructed epoch9 predictor is a separate comparison model.** The epoch9 internal-test score must not be attributed to epoch22, and the new miRNA supervision controls retain the original frozen structural inputs in every condition.

| Item | Original PairAC predictor | Reconstructed comparison predictor |
|---|---|---|
| Selected checkpoint | Epoch 22; 25 epochs executed | Epoch 9; 14 epochs executed |
| Role | Supplies frozen structural-reactivity channels for the released interaction representations | Evaluated alongside sequence CNN and sequence + LinearFold-V CNN |
| Recorded train / validation / test windows | 1,509,015 / 161,167 / 160,112 | 1,928,795 / 196,020 / 207,979 |
| Initialization | Public mRNABERT with task-trained LoRA and FiLM/residual head | Public mRNABERT with freshly initialized LoRA and FiLM/residual head |
| Training microbatch / accumulation | 768 / 1 | 128 / 1 |
| Precision and optimizer | BF16, AdamW | BF16, AdamW |
| LoRA / head learning rate | 5e-5 / 3e-4 | 5e-5 / 3e-4 |
| Training loss | Per-window mean masked L1 | Per-window mean masked L1 |
| Matched 207,979-window internal-test MAE | Not assigned | 0.1621083 |
| Same 7,347-window GSE74353 subset MAE | 0.1923284 | 0.1919775 |

The two predictors share a model family but have different checkpoint bytes, data membership and batch budgets. The original training log reports MAE 0.1604278 on its own historical 160,112-window test set; this is not a matched comparison with the reconstructed 207,979-window test set. The original membership table has not been recovered, so its external score remains descriptive with respect to historical training exposure.

Checkpoint selection uses the validation score `0.5 pooled PCC + 0.5 mean within-window PCC`. The reconstructed run resolves near ties within 1e-4 using regional MAE and falls back to per-window MAE when correlation is undefined. MAE was chosen for the comparison's presentation after the existing results were inspected; checkpoints were not reselected for that table. A lower absolute reactivity error does not establish better correlation or secondary-structure prediction.

## Verified identities

Checkpoint bytes were rehashed on 2026-11-06. The complete machine-readable record, including training metadata and original source-data hashes, is [audit/icshape_checkpoint_identity.json](../audit/icshape_checkpoint_identity.json).

| Asset | SHA-256 |
|---|---|
| Original epoch22 `best_film_residual_head.pt` | `61904966a61fd209c6ef00f37474e92fde10f7bfe5667568b7a6ca4c91550aa8` |
| Original epoch22 LoRA `adapter_model.safetensors` | `bde9e864d7d3d3418557feb7efed4bc47f6af6d38ce958e5a56983878d016b8f` |
| Original epoch22 `metrics.json` | `4588f90cfe38ec8223d7f3b60018df22c7db47a689d0a3db1f6c1d02a078d1dd` |
| Reconstructed epoch9 `best_state.pt` | `4acfef86b2442bc0814bc94a52be216e89bbc306f030b8c64f882ba54a866d8c` |
| Reconstructed epoch9 `metrics.json` | `33b6420f7dd5a82a4868657874acbb8e266e093b4e6bef82b18d3dd5654d3e77` |

Weights remain separately distributed. This audit reads existing metadata and hashes existing files; it does not claim new training or inference. The saved MAE evidence can be checked with:

```bash
python scripts/reproduce_accessibility_mae.py
```

See [the reactivity comparison](ACCESSIBILITY_COMPARISON.md) for all four predictors, observed-position masking, external cohort construction and the scope of the saved-result verification.
