# FP4 Activation Validation Design

## Goal

Determine whether calibrated E2M1 FP4 activation QDQ preserves more useful
information than uniform INT4 activation QDQ in the official CSPN, DySPN,
NLSPN, and CompletionFormer NYU depth-completion models. The experiment must
separate activation-format error from weight error and from propagation-domain
error.

This is an accuracy and numerical-behavior experiment. The available A100 does
not execute native FP4 kernels, so the experiment makes no latency, throughput,
energy, or deployment-speed claim.

## Immutable Inputs

Use the existing converged official-model checkpoints and propagation counts:

| Model | Iterations | Checkpoint |
| --- | ---: | --- |
| CSPN | 24 | `/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24/best.pt` |
| DySPN | 6 | `/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/dyspn_iter6/best.pt` |
| NLSPN | 18 | `/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/nlspn_iter18/best.pt` |
| CompletionFormer | 18 | `/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/completionformer_iter18/best.pt` |

Model source, checkpoint loading, dataset preprocessing, calibration indices,
evaluation indices, and propagation iterations follow the unified PA-W4A8
evaluation contract. Source and checkpoint hashes must be recorded and
validated. No training or weight update is performed.

Calibration uses the existing 128 NYU training samples selected with seed
`20260804`. Evaluation uses the same ordered 64 NYU validation samples as the
unified PA-W4A8 evaluation.

## E2M1 QDQ Contract

The signed E2M1 finite-value codebook is:

```text
0, +/-0.5, +/-1, +/-1.5, +/-2, +/-3, +/-4, +/-6
```

Each activation site receives a frozen calibration scale. Ordinary Conv and
Linear sites use per-tensor scales. Existing `concat_conv` and fused
LayerNorm-output sites retain their current per-channel scale granularity. For
each tensor or channel, `scale = calibration_absmax / 6`; an all-zero range uses
scale one. Inference divides by the scale, rounds to the nearest E2M1 value
using round-to-nearest-even at exact midpoints, saturates outside the finite
codebook, and reconstructs in the original domain. Signed E2M1 is also used for
nonnegative ReLU values; the experiment must not invent an unsigned FP4 format.

Observers reject non-finite calibration values. Scales freeze before
evaluation and may not adapt per evaluation sample. The QDQ implementation
records MSE, SQNR, cosine similarity, zero ratio, saturation ratio, and
non-finite counts for every active site.

## Quantization Boundaries

E2M1 replaces uniform activation QDQ only at ordinary Conv/Linear inputs and
outputs, ReLU outputs, and fused LayerNorm output boundaries. It does not
change model modules or tensor shapes.

The following semantic tensors remain A8 in every comparison configuration:

- external sparse-depth input;
- the final initial-depth prediction passed into propagation;
- guidance, offset, and affinity tensors;
- confidence and gate tensors;
- normalization coefficients;
- every propagation state and final propagation-domain output.

The propagation backend continues to quantize neighbor affinity before Q13
INT16 normalization, reconstruct the center coefficient from quantized
neighbors, preserve sparse-depth anchors, and use the existing float-QDQ
sampling and propagation-MAC reference. Semantic-site resolution is strict: a
missing required A8-owned site aborts the run instead of silently applying
E2M1.

## Controlled Evaluation Matrix

Run a fresh FP32 baseline and two matched triplets:

| Configuration | Weights | Ordinary activations | Propagation signals |
| --- | ---: | --- | ---: |
| `FP4V_W8A4` | W8 | uniform A4 | A8 |
| `FP4V_W8E2M1` | W8 | E2M1 FP4 | A8 |
| `FP4V_W8A8` | W8 | uniform A8 | A8 |
| `FP4V_W4A4` | W4 | uniform A4 | A8 |
| `FP4V_W4E2M1` | W4 | E2M1 FP4 | A8 |
| `FP4V_W4A8` | W4 | uniform A8 | A8 |

Within each triplet, weight QDQ tensors are identical. To isolate activation
format in this first feasibility experiment, trained biases remain FP32 and
identical across A4, E2M1, and A8. This intentionally differs from a complete
integer backend and must be recorded as `bias_contract=fp32_isolation` in
metadata. A deployment-oriented FP4 bias/accumulator contract is out of scope
until E2M1 demonstrates an accuracy benefit.

The A4 and A8 controls use the existing symmetric quantizer for signed tensors
and unsigned quantizer for nonnegative ReLU/input tensors. E2M1 remains signed,
as required by the standard codebook.

## Execution Stages

First run unit and integration tests, then execute a two-sample smoke test for
all six configurations and all four models. The smoke test verifies hook
coverage, semantic A8 ownership, finite scales and outputs, source/checkpoint
identity, and propagation constraints. Smoke-test metrics are not included in
the accuracy conclusion.

Only after every model passes smoke testing, run the full 128-sample
calibration and 64-sample evaluation. Results are written to a new root:

```text
profile_logs/nyu_fp4_activation_validation
```

The run must not overwrite or append to the existing unified PA-W4A8 results.

## Metrics And Decision Rule

The primary metric is mean per-sample RMSE over the fixed 64 samples, matching
the existing report. Also report pooled-pixel RMSE, MAE, ABS_REL, non-finite
sample and pixel counts, and regional metrics.

For each weight width, compare E2M1 and INT4 on paired samples. Report the mean
per-sample RMSE difference and a deterministic 10,000-resample bootstrap 95%
confidence interval using seed `20260806`. Define A4-to-A8 recovery as:

```text
(RMSE_INT4 - RMSE_E2M1) / (RMSE_INT4 - RMSE_A8)
```

If the denominator is non-positive, report recovery as unavailable rather than
forcing an interpretation.

E2M1 is effective for a model and weight width only when all of these hold:

1. All 64 predictions are finite and propagation/anchor constraints do not
   regress.
2. E2M1 mean per-sample RMSE is lower than matched INT4 RMSE.
3. The upper bound of the paired RMSE-difference 95% confidence interval is
   below zero.
4. A4-to-A8 recovery is positive.
5. Initial-depth, final-prediction, and propagation-step diagnostics show no
   new numerical instability or error amplification.

Failure on one model does not invalidate another model's result. The final
report states effectiveness separately for all eight model/weight-width pairs.

## Diagnostic Outputs

Persist sample, regional, signal, layer, state, and propagation metrics;
calibration ranges; E2M1 manifests; metadata; and all 64 prediction payloads
for every configuration. Aggregate activation statistics by encoder,
attention, decoder, depth head, and propagation head where present.

Generate:

- one four-model summary figure comparing FP32, A4, E2M1, and A8 at W8 and W4;
- per-model paired RMSE-difference and recovery figures;
- per-model group-level SQNR, zero-ratio, and saturation-ratio figures;
- per-model propagation-step error figures;
- per-model visual comparisons containing GT, FP32, INT4, E2M1, A8, and their
  absolute-error maps for representative fixed samples.

## Testing And Failure Handling

Unit tests cover the complete E2M1 codebook, round-to-nearest-even midpoints,
negative symmetry, saturation, zero ranges, per-tensor and per-channel scales,
statistics, and rejection of non-finite calibration values.

Configuration tests verify the six-entry matrix, identical weight tensors and
FP32 biases within each triplet, propagation A8 settings, and strict A8
semantic exemptions. Integration tests verify that hooks do not requantize one
logical edge unexpectedly and that concat branches, ReLU boundaries,
LayerNorm fusion boundaries, and prediction export remain valid.

Run the complete test suite in both the base environment and
`completionformer-py37`. Formal evaluation aborts on source/checkpoint identity
mismatch, missing calibration observations, invalid scales, non-finite smoke
outputs, or propagation-constraint regression. Before reporting results,
verify that every model/configuration has exactly 64 metric rows and prediction
payloads, all models use identical calibration/evaluation indices, and every
metadata file records the E2M1 and bias contracts.

## Non-Goals

- Native FP4 kernels or A100 speed measurements.
- MXFP4, NVFP4, dynamic block scaling, or learned scales.
- Selective per-layer FP4 allocation before the dense E2M1 baseline is known.
- Quantization-aware training or checkpoint updates.
- FP4 affinity, confidence, offsets, sparse input, or propagation state.
