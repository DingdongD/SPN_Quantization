# CSPN Static and Dynamic Group-8 W4A4 QAT Design

## Objective

Implement end-to-end quantization-aware training for the official CSPN NYU
model and determine whether QAT can recover the accuracy lost by strict W4A4
quantization. Compare Static-G8 and Dynamic-G8 activation quantization under an
otherwise identical training and evaluation protocol.

The experiment starts both runs from the same converged official FP32
checkpoint. The only experimental variable is the source of ordinary
activation scales: calibrated static ranges or per-sample dynamic ranges.

## Scope

This work covers the official CSPN model only. It adds an isolated QAT path and
does not change the existing PTQ, BRECQ, QDrop, rotation, SmoothQuant, scale
permutation, or outlier-isolation implementations.

The QAT comparison does not enable FP4, mixed precision, progressive bit-width
schedules, distillation, or automatic FP fallback. Guidance and bias remain
FP32 to match the current strict CSPN PTQ baseline.

## Architecture

Add a dedicated `CSPNQATController` that installs and removes training-time
fake quantization without changing `HardwareAlignedInstrumentor`. The
controller owns four independent concerns:

1. W4 fake quantization for convolution weights.
2. A4 fake quantization for ordinary activation owners.
3. A4 fake quantization for the existing structural decoder boundaries.
4. A differentiable mirror of the strict CSPN A8/Q13/INT32 propagation path.

The proposed modules are:

```text
spn_quant/qat/ste.py
spn_quant/qat/quantizers.py
spn_quant/qat/cspn.py
scripts/train_nyu_cspn_group_a4_qat.py
scripts/evaluate_nyu_cspn_group_a4_qat.py
```

The QAT controller is explicit and reversible. Installation, calibration,
training, export, and removal are separate operations. Missing state or an
invalid transition is an error; no operation silently switches to FP32.

## Hard-Forward STE Contract

QAT forward values must equal the existing hard quantization path. STE affects
only backward propagation.

For ordinary rounding, the forward pass applies the same round, clamp, and
dequantize operations as PTQ while the backward pass uses the identity
gradient inside the supported tensor path. For CSPN propagation, a custom
autograd boundary returns the hard integer-path result in forward and routes
the gradient through a floating-point propagation proxy in backward. The hard
value is returned directly rather than reconstructed with an arithmetic
detach expression.

This contract is verified before end-to-end training. QAT evaluation results
are not accepted unless exported weights reproduce the QAT forward values
through the existing hard PTQ and propagation controllers.

## Quantization Contract

### Weights

All eligible convolution weights use signed symmetric W4 with one scale per
output channel. Each module retains an FP32 master weight for optimization.
The forward pass quantizes that master weight with STE. Export writes standard
FP32 master weights into an official-model-compatible state dictionary; the
existing hard evaluator performs final W4 quantization.

### Ordinary Activations

The ordinary owner registry is exactly the registry used by the strict CSPN
PTQ experiment. No QAT-only activation owner may be added.

- ReLU outputs use unsigned A4 codes `[0, 15]`.
- Signed activations use symmetric A4 codes `[-7, 7]`.
- Channels are partitioned into contiguous groups of eight.
- Existing tensor-granularity exceptions remain tensor-granularity and are
  identical between the two runs.

Static-G8 uses frozen MinMax ranges produced by the fixed stratified 128-sample
calibration set. Dynamic-G8 computes one scale per sample and contiguous group
from the current activation. Dynamic scales are detached from autograd. A
sample cannot influence another sample's dynamic scale.

### Structural Boundaries

The existing signed decoder boundaries use frozen calibrated Group-8 A4
scales in both experiments. They do not become dynamic. No random, Hadamard,
learned, or SVD rotation is enabled.

### CSPN Propagation

Guidance remains FP32. Raw affinity is quantized to A8 before normalization.
Neighbor coefficients are normalized into signed INT16 Q13 codes, and the
center coefficient is recomputed from the quantized neighbors so that the
coefficient sum remains exactly one in Q13. Propagation state uses A8. The hard
forward uses INT32 multiply-accumulate and preserves sparse-depth anchors.

The floating proxy follows the same neighborhood, center-residual, iteration,
and anchor data flow. It exists only to carry gradients through the hard
integer boundary. Confidence and offset settings remain A8 where applicable,
although official CSPN does not add generic confidence or offset owners.

### BN and Bias

Batch normalization is folded and frozen before QAT calibration and training.
Bias remains FP32 in both Static-G8 and Dynamic-G8 runs, matching the current
reference QDQ experiment. Bias quantization is outside this comparison.

## Training Protocol

Both experiments use:

- the same official converged FP32 checkpoint;
- the complete official NYU training split and augmentation pipeline;
- the same stratified 128-sample calibration manifest;
- the same training and validation manifests;
- the same data-loader ordering, batch size, seed, and CUDA determinism
  settings;
- SGD with momentum `0.9`, weight decay `1e-4`, and initial learning rate
  `1e-3`;
- `ReduceLROnPlateau` with the same configuration in both runs;
- the official masked L1 depth loss;
- W4A4 enabled from the first QAT step.

The initial learning rate is lower than the official from-scratch CSPN rate
because QAT fine-tunes a converged checkpoint. No auxiliary teacher,
intermediate-feature, or propagation-state loss is used. The task loss is
propagation-aware because its backward path crosses every quantized CSPN
iteration through the propagation STE proxy.

Training runs for at most 30 epochs. It stops when validation RMSE fails to
improve by at least 0.1% for six consecutive epochs. The best checkpoint is
selected exclusively by full-validation RMSE. The fixed 64-sample subset is
not used as an independent checkpoint-selection signal.

## Checkpoints and Resume

Each run writes `last.pt` and `best.pt`. A resumable checkpoint contains the
standard model state, optimizer state, scheduler state, epoch, convergence
tracker, quantization mode, calibration manifest digest, owner manifest, and
explicit quantization configuration.

Resume requires exact equality of the model, owner, calibration, and
quantization contracts. A mismatch raises an error. Exported best checkpoints
contain official-model-compatible FP32 master weights plus separate metadata;
they do not retain parametrization-specific state keys.

## Evaluation Protocol

After training, remove QAT instrumentation and load each exported best
checkpoint into a fresh official CSPN model. Reinstall the existing hard PTQ
instrumentor, structural-boundary controller, and A8/Q13/INT32 propagation
adapter. Final metrics are produced only by this hard path.

Evaluate these configurations:

```text
FP32
PTQ_STATIC_G8_W4A4
PTQ_DYNAMIC_G8_W4A4
QAT_STATIC_G8_W4A4
QAT_DYNAMIC_G8_W4A4
```

Report full-validation and paired fixed-64 results. Metrics include RMSE, MAE,
AbsRel, iRMSE, flat-region RMSE, boundary RMSE, per-sample RMSE deltas, better
and worse sample counts, activation SQNR, new-zero rate, saturation rate,
propagation-step error, anchor error, and affinity constraint error.

Generate prediction comparisons for the fixed 64 samples containing RGB,
sparse depth, ground truth, FP32 prediction, both PTQ predictions, both QAT
predictions, and absolute-error maps. Runtime artifacts remain under
`/workspace/SPN_Quantization/profile_logs/` and are not committed.

## Error Handling and Coding Rules

New configuration objects use direct attribute access. Dictionaries use direct
indexing. Required fields do not have code-level defaults. The implementation
does not catch configuration, checkpoint, calibration, numerical, or contract
errors merely to continue execution.

Non-finite input, activation, loss, gradient, parameter, scale, prediction, or
propagation result is an immediate error. Zero-range groups map exactly to the
zero code through an explicit quantizer rule; they are not treated as an error
and do not trigger a fallback scale policy.

## Verification

Unit tests verify:

- signed and unsigned W4/A4 code domains;
- per-output-channel W4 scale and gradient behavior;
- Static-G8 parity with the existing grouped PTQ quantizer;
- Dynamic-G8 per-sample scale independence and detached scale gradients;
- finite, nonzero STE gradients;
- zero-range and non-finite-input behavior;
- hard-forward propagation parity for one and multiple iterations;
- exact Q13 coefficient-sum and sparse-anchor invariants.

Integration tests verify:

- exact ordinary-owner and structural-boundary manifests;
- absence of generic quantization on guidance and propagation-only tensors;
- one complete QAT optimization step for each mode;
- checkpoint save, resume, export, and fresh-model reload;
- QAT forward versus exported hard-path prediction parity;
- unchanged outputs from the existing PTQ path when QAT is not installed.

Before the full run, execute a short training smoke test and the complete test
suite. Afterward, verify that both best checkpoints produce 64 finite paired
predictions through the hard evaluator.

## Acceptance Criteria

- Static-G8 and Dynamic-G8 differ only in ordinary activation scale source.
- Both runs start from identical official FP32 weights and use identical data,
  optimizer, scheduler, seed, epoch limit, and convergence criterion.
- The hard-forward parity tests pass for weights, ordinary activations,
  structural boundaries, and multi-step CSPN propagation.
- Guidance and bias remain FP32; propagation remains A8/Q13/INT32.
- Exported checkpoints load into a fresh official CSPN model without QAT-only
  parameter keys.
- Final claims use hard-path full-validation and paired fixed-64 metrics.
- QAT is considered beneficial only if hard-path RMSE improves over the
  matching PTQ baseline without non-finite outputs, anchor violations, or
  affinity-constraint regressions.
