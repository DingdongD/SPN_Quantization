# Model-Level Semantic Quantization Adapters

This change adds fail-closed semantic adapters for CSPN, DySPN, NLSPN, and
CompletionFormer. The adapters sit on top of the logical-edge QDQ runtime and
translate implementation-specific module names and propagation interfaces into
stable tensor roles.

## Registered roles

All models register RGB, sparse-depth value/mask, encoder and decoder
activations, merge sites, propagation inputs, propagation outputs, and final
prediction. Model-specific roles include:

- CSPN: initial depth, guidance, normalized affinity, structural residual Add,
  decoder Concat, and propagation state.
- DySPN: RGB/depth stems, SE gates, initial depth, guidance, confidence logits
  and gate, offset/affinity logits, normalized sampling coordinates/affinity,
  and iterative states.
- NLSPN: initial depth, guidance, confidence, offset/affinity logits,
  normalized offset/affinity, and iterative states.
- CompletionFormer: PVT/MLP/QKV activations, CBAM channel/spatial gates,
  decoder/depth/guidance/confidence heads, depth-residual addition,
  offset/affinity logits, normalized propagation parameters, and states.

Executed `_concat` calls and CSPN structural Add/Concat sites are operational
merge sites. Direct functional merges and functional attention softmax sites
are declared with `operational=0` so coverage gaps remain visible rather than
being silently treated as quantized.

## Validation contract

Strict mode rejects a model when a required propagation module, producer,
semantic role, or expected number of decoder Concat calls is missing. The
manifest records role, producer, observed shape/dtype, non-finite values,
recommended escape bit width, operational coverage, and active merge policy.

The default transform for every site is `none`. LogNP remains available only as
an explicit per-site ablation and is not selected by any model adapter.

## Runner

```bash
python scripts/run_nyu_edge_quantization.py \
  --merge-policy independent \
  --run-dir <run-dir> \
  --sample-metrics <sample-metrics.csv> \
  --quant-backend hardware
```

The run writes `semantic_site_manifest.csv` beside the existing hardware and
accuracy tables. Use `--no-strict-semantic-sites` only for adapter bring-up.
