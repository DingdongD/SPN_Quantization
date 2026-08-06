# Strict AdaRound/BRECQ Deployment Closure

This path fixes the deployment mismatch in the earlier reconstruction prototype.
The previous flow hardened reconstructed floating-point weights and then allowed
the hardware runner to fold BatchNorm and perform a second RTN pass. That second
pass could change both the learned rounding decisions and their scales.

The strict flow has two separate artifacts:

1. The **original checkpoint**, used to rebuild the original model graph.
2. A **strict deployment contract**, containing the exact integer codes and
   scales learned on the deployment-equivalent folded graph.

The debug folded checkpoint written by the reconstruction script is not a valid
replacement for the original checkpoint in the standard model builder.

## Enforced Graph Contract

Both teacher and student are prepared with the same graph transformation used by
hardware evaluation:

```text
load original checkpoint
-> discover Conv-BN pairs
-> apply the configured exclusions
-> fold Conv-BN pairs
-> reconstruct folded weights
```

The deployment contract records:

- folded Conv-BN pairs;
- excluded fan-out pairs;
- explicitly unfolded pairs when folding is disabled;
- source-checkpoint SHA-256;
- pre-reconstruction folded weight and bias SHA-256 values.

Evaluation fails closed if any of these differ.

## Exact Weight Contract

Each reconstructed module exports:

```text
integer codes
per-output-channel scale
qmin / qmax
weight layout and group count
base weight and bias fingerprints
integer-code fingerprint
dequantized-weight fingerprint
```

At evaluation, the hardware instrumentor may initialize its normal RTN state,
but contracted modules are then overwritten with the exact exported code-scale
lattice. Their bias is recomputed with:

```text
bias_scale[o] = activation_input_scale * exact_weight_scale[o]
```

The contracted module is therefore not evaluated with a newly estimated W4
scale or a second set of rounding decisions.

Grouped and non-grouped `ConvTranspose2d` contracts are supported even though
the legacy hardware instrumentor originally covered only `Conv2d` and `Linear`.

## Strict Algorithm Names

The strict runner exposes:

```text
adaround_strict
brecq_strict
```

Both are weight-only reconstruction methods. This restriction is intentional.
The previous learnable-activation implementation does not yet own exactly the
same logical ReLU, Add, Concat, and producer-output edges as the deployment
runtime. It remains available through the older semantic reconstruction entry,
but its results must be named `semantic_brecq` or `semantic_brecq_qdrop`, not
strict BRECQ.

`adaround_strict` accepts exactly one supported weight module per target.
`brecq_strict` accepts a block containing one or more `Conv2d`,
`ConvTranspose2d`, or `Linear` modules.

The strict optimizer follows the original weight-reconstruction semantics:

- hard-sigmoid adaptive rounding;
- cached random mini-batches;
- sum-reduced rounding regularization;
- warm-up followed by beta decay;
- MSE, diagonal Fisher, or full Fisher reconstruction;
- optional asymmetric reconstruction using student inputs and teacher outputs;
- final hard rounding without a second RTN pass.

## Reconstruction Example

CompletionFormer depth head:

```bash
python scripts/run_nyu_strict_reconstruction.py \
  --method adaround_strict \
  --run-dir <run-dir> \
  --checkpoint best.pt \
  --target backbone.dep_dec0.0 \
  --w-bits 4 \
  --calibration-samples 1024 \
  --batch-size 32 \
  --steps 20000 \
  --loss mse
```

A complete semantic block:

```bash
python scripts/run_nyu_strict_reconstruction.py \
  --method brecq_strict \
  --run-dir <run-dir> \
  --checkpoint best.pt \
  --target backbone.dep_dec0 \
  --w-bits 4 \
  --calibration-samples 1024 \
  --batch-size 32 \
  --steps 20000 \
  --loss fisher_diag \
  --asymmetric
```

Outputs include:

```text
strict_deployment_contract.pt
strict_reconstruction_manifest.json
strict_weight_rounding_manifest.csv
strict_reconstruction_summary.csv
strict_reconstruction_history.json
reconstructed_folded.pt          # debug only
```

## Evaluation Example

Use the original checkpoint, not `reconstructed_folded.pt`:

```bash
python scripts/run_nyu_edge_quantization.py \
  --run-dir <run-dir> \
  --checkpoint best.pt \
  --reconstruction-manifest \
    <strict-output>/strict_reconstruction_manifest.json \
  --merge-policy independent \
  --quant-backend hardware \
  --config-names HW_W4A8_full \
  --sample-metrics <sample-metrics.csv>
```

A direct contract path is also accepted:

```bash
python scripts/run_nyu_edge_quantization.py \
  --run-dir <run-dir> \
  --checkpoint best.pt \
  --deployment-contract \
    <strict-output>/strict_deployment_contract.pt \
  --merge-policy independent \
  --quant-backend hardware \
  --config-names HW_W4A8_full \
  --sample-metrics <sample-metrics.csv>
```

## Required Validation

Before reporting real results, verify:

1. `metadata.json` contains `exact_weight_contract: 1`.
2. `hardware_manifest.csv` contains `exact_weight_contract` rows.
3. The source-checkpoint and graph-contract checks pass without overrides.
4. The W4A8 result is evaluated from the original checkpoint plus contract.
5. The contracted weight fingerprints are unchanged across repeated runs.

Local block reconstruction loss is diagnostic only. A contract is deployable
only after an end-to-end evaluation on the same sample set used by the RTN
baseline. The current acceptance rule requires zero nonfinite samples and mean
RMSE no worse than RTN W4A8.

Aggregate completed RTN, AdaRound, and BRECQ evaluations with:

```bash
python scripts/plot_nyu_strict_reconstruction.py \
  --root profile_logs/nyu_strict_w4a8_evaluation \
  --out-dir profile_logs/nyu_strict_w4a8_evaluation/summary
```

This writes the deployment decision table, a log-scale RMSE comparison, and a
GT/FP32/RTN/AdaRound/BRECQ prediction comparison for the sample with the largest
cross-method error spread in each model.

Strict activation reconstruction should be added only after activation ownership
is expressed by the same semantic edge graph during reconstruction and final
hardware evaluation.
