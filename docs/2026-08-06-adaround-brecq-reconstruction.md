# AdaRound and BRECQ Reconstruction

This change adds calibration-only weight rounding and semantic block
reconstruction on top of the semantic-site and edge-QDQ stack.

## Scope

- `AdaptiveRoundingParametrization` learns one up/down rounding variable for
  each weight while retaining a fixed per-output-channel uniform scale.
- Conv2d, ConvTranspose2d, and Linear are supported.
- `SemanticBlockReconstructor` uses the same rounding variables for either
  layer-wise AdaRound or joint block-wise BRECQ.
- Optional activation reconstruction learns uniform activation scales and can
  use QDrop during calibration.
- Reconstruction loss supports ordinary MSE and output-gradient-squared Fisher
  weighting.
- The best evaluated hard-rounding state is retained, so reconstruction cannot
  silently replace RTN with a worse hard solution.

## AdaRound example

```bash
python scripts/run_nyu_reconstruction.py \
  --method adaround \
  --run-dir output/nyu_converged_baselines/completionformer_iter18 \
  --checkpoint best.pt \
  --target backbone.dep_dec0.0 \
  --target backbone.conv1.0 \
  --w-bits 4 \
  --a-bits 0 \
  --calibration-samples 32 \
  --steps 1000
```

## BRECQ example

```bash
python scripts/run_nyu_reconstruction.py \
  --method brecq \
  --run-dir output/nyu_converged_baselines/completionformer_iter18 \
  --checkpoint best.pt \
  --target backbone.dep_dec0 \
  --w-bits 4 \
  --a-bits 4 \
  --qdrop-probability 0.5 \
  --loss fisher \
  --calibration-samples 32 \
  --steps 2000
```

Targets must not overlap. Use `--list-targets` to inspect candidate layer and
block names in the loaded checkpoint.

## Evaluation

The reconstruction command writes:

- `reconstructed.pt`
- `reconstruction_manifest.json`
- `weight_rounding_manifest.csv`
- `activation_reconstruction_manifest.csv`
- `reconstruction_summary.csv`
- `reconstruction_history.json`

Evaluate the hardened checkpoint and learned activation ranges through the
edge-aware runner:

```bash
python scripts/run_nyu_edge_quantization.py \
  --run-dir <run-dir> \
  --checkpoint <reconstruction-dir>/reconstructed.pt \
  --reconstruction-manifest <reconstruction-dir>/reconstruction_manifest.json \
  --merge-policy independent \
  --quant-backend hardware \
  --config-names HW_W4A4_full \
  --sample-metrics <sample-metrics.csv>
```

The learned activation maxima are applied only when the evaluated activation
bit width matches the reconstruction bit width. LogNP is not used.

## Recommended first experiments

1. AdaRound W4A8 on CompletionFormer `backbone.dep_dec0.0`,
   `backbone.conv1.0`, and `backbone.gd_dec0.0`.
2. AdaRound W4A8 on CSPN/NLSPN encoder and depth-head layers.
3. BRECQ W4A8 on decoder and depth-head blocks.
4. BRECQ W4A4 with learned activation scales after the W4A8 result is stable.

Propagation-domain tensors should remain A8 during these experiments.
