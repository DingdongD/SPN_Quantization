# Propagation-Aware SPN Quantization Design

## Objective

Build a hardware-aligned W4A4 depth-completion baseline that keeps the CNN
backbone and heads at W4A4 while applying an explicit integer contract to the
SPN domain. The contract must preserve affinity normalization, sparse-depth
anchors, and bounded per-iteration error for CSPN, DySPN, NLSPN, and
CompletionFormer.

## Motivation

The current hardware-aligned instrumentor applies QDQ at Conv/Linear inputs
and outputs. Semantic adapters identify propagation tensors, but do not replace
the propagation arithmetic. Consequently, the output of an affinity or offset
projection is quantized as an ordinary activation and the following
normalization, center-coefficient identity, anchor update, and iterative state
transitions are not represented by one consistent integer contract.

For propagation state error `e_t`, a useful local bound is:

```
||e_t|| <= rho_t ||e_(t-1)|| + delta_t
```

`rho_t` is controlled by the quantized propagation coefficients and
`delta_t` contains new coefficient, sampling, accumulator, and state
requantization errors. Quantization can cause growth when it breaks the
normalization constraint, changes dominant neighbors, destroys a gate or
anchor, or repeatedly injects state error.

## Scope

The first implementation is an evaluation-grade integer simulation, not a
deployment kernel. It covers the official model structures already integrated
in this repository and evaluates 64 fixed NYU validation samples.

In scope:

- W4A4 CNN Conv/Linear paths using the existing hardware-aligned quantizer.
- A4 pre-normalization affinity values or logits.
- Unsigned A8 confidence and gate tensors.
- INT32 normalization sums and propagation accumulation.
- Signed Q13 INT16 normalized affinity coefficients.
- Exact center residuals in the normalized fixed-point domain.
- A4 and A8 offset/state ablations.
- Per-step propagation diagnostics and static prediction visualizations.

Out of scope:

- CUDA or vendor-specific integer kernels.
- QAT and weight updates.
- LogNP, SmoothQuant, AWQ, or percentile clipping experiments.
- A production nonlinear A4 confidence codebook. It can be added only after
  the unsigned A8 reference establishes the propagation contract.

## Quantization Boundary

Ordinary CNN quantization ends at the propagation-head projections. The SPN
adapter owns all tensors from the projection outputs through the final
propagated prediction:

```
CNN W4A4 -> raw affinity/offset/confidence outputs
          -> propagation-domain quantization
          -> integer normalization and coefficient construction
          -> iterative propagation and anchor restoration
          -> prediction
```

The propagation adapter must prevent the generic Conv output hook from
quantizing a tensor a second time when a propagation-domain quantizer owns that
boundary.

## Integer Contracts

### Affinity

Signed raw neighbor affinity uses symmetric A4 codes. Softmax logits use
symmetric A4 codes. Quantization always precedes normalization.

For absolute-sum normalization, the implementation accumulates absolute A4
codes in INT32, applies the model's original denominator rule, and converts the
normalized coefficients to a signed Q13 representation. Q13 is required
because signed neighbor sums can reach `-1`, so the derived center coefficient
can reach `2`; Q15 and Q14 would overflow an INT16 center coefficient. The
final neighbor is adjusted by the rounding residual when the model requires an
exact normalized sum.

For CSPN-style center coefficients, the center is never independently
quantized:

```
q_center = Q_ONE - sum(q_neighbor)
```

This preserves `eta_0 + sum(eta_k) = 1` in the coefficient domain.

For softmax, a lookup table maps each A4 logit difference to an exponential
value. INT32 sums and a fixed-point reciprocal produce nonnegative Q13
coefficients. A residual correction makes their integer sum exactly `Q_ONE`.

### Confidence And Gates

Confidence and gates use unsigned A8 with exact zero and one endpoints. A mask
is applied explicitly rather than inferred from a quantized nonzero value.
This keeps close/open semantics and sparse-anchor behavior separate from the
uniform quantization of ordinary activations.

### Offsets

Offsets are evaluated at A4 and A8. Their coordinate conversion and grid
sampling remain floating-point in the reference simulator, but the offset
values entering that conversion are reconstructed from integer codes. This
isolates offset precision from affinity constraints without claiming that
`grid_sample` is an integer kernel.

### Propagation State

Propagation multiplication uses Q13 coefficients and an INT32 conceptual
accumulator before reconstruction. State outputs are evaluated at A4 and A8.
After every state requantization, known sparse-depth positions are restored
from the sparse input using the model's original mask/confidence semantics.

## Model-Specific Semantics

### CSPN

Quantize the eight neighbor guidance values, perform the official absolute-sum
normalization, derive the center coefficient as the exact Q13 residual, and
restore sparse anchors after every iteration.

### NLSPN And CompletionFormer

Quantize raw affinity and offset values produced by `conv_offset_aff`.
Quantized A8 confidence is sampled at the quantized offsets and applied at the
same point as in the official implementation. The adapter then executes the
selected official normalization rule, including its denominator floor, before
deriving the center coefficient. Deformable sampling remains the official CUDA
operation in this evaluation reference.

### DySPN

Quantize per-iteration neighbor logits before softmax. Use LUT softmax and
residual-corrected Q13 coefficients whose sum is exactly one. Quantize the
sigmoid confidence with unsigned A8 and apply the sparse-depth blend after each
iteration.

## Experiment Matrix

All configurations use identical checkpoints, samples, preprocessing, and
calibration indices.

| Configuration | CNN | Affinity/logits | Confidence/gate | Offset | State |
|---|---|---|---|---|---|
| FP32 | FP32 | FP32 | FP32 | FP32 | FP32 |
| Generic W4A4 | W4A4 | generic A4 QDQ | generic A4 QDQ | generic A4 QDQ | FP32 loop |
| PA-Constraint | W4A4 | A4 then normalize | unsigned A8 | A4 | A4 |
| PA-OffsetA8 | W4A4 | A4 then normalize | unsigned A8 | A8 | A4 |
| PA-StateA8 | W4A4 | A4 then normalize | unsigned A8 | A8 | A8 |
| W8A8 | W8A8 | A8 | unsigned A8 | A8 | A8 |

The ablations are cumulative so that affinity constraints, offset precision,
and state precision can be attributed independently.

## Metrics

End-to-end metrics:

- RMSE, MAE, REL, delta thresholds, invalid ratio.
- Mean, median, p75, p95, p99, and maximum sample RMSE.
- Sparse-anchor and non-anchor region errors.

Propagation metrics:

- Per-step RMSE relative to the FP32 state.
- Step growth ratio and final-to-first error ratio.
- Sparse-anchor absolute error after each iteration.
- NaN and Inf ratio.

Coefficient metrics:

- `abs(center + sum(neighbors) - 1)` for residual-normalized models.
- Softmax coefficient-sum residual.
- Contraction violation rate under the model's local coefficient norm.
- Affinity zeroed, sign-flip, saturation, and dominant-neighbor-change rates.
- Confidence false-close, false-open, rank-change, and saturation rates.
- Offset zeroed and saturation rates.

## Visual Outputs

The evaluator writes static PNG figures and their source CSV/NPZ data:

- A 64-sample contact sheet comparing GT, FP32, Generic W4A4, and the best
  propagation-aware configuration.
- A detailed sheet for fixed random and worst-error samples with sparse input,
  GT, predictions, and absolute-error maps.
- Per-model propagation-step error curves.
- Per-model normalization, contraction, and anchor violation charts.

Depth panels share one metric color scale per sample. Error maps share one
scale per comparison row so that visual differences are not hidden by
independent autoscaling.

## Acceptance Criteria

- All existing tests continue to pass.
- Unit tests prove quantize-before-normalize ordering and exact fixed-point
  coefficient sums.
- CSPN/NLSPN/CompletionFormer center coefficients are derived rather than
  independently quantized.
- Sparse anchors remain exact where the official model requires hard anchors.
- DySPN Q13 softmax coefficients are nonnegative and sum to `Q_ONE`.
- Every evaluated configuration records finite/invalid, normalization,
  contraction, anchor, and per-step metrics.
- The 64-sample run exports both aggregate CSV/JSON and prediction figures.
- Results clearly distinguish Generic W4A4 from each propagation-aware
  ablation; no claim of improvement is made unless the measured metrics show
  it.
