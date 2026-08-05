# Per-Channel LogNP Activation Quantization Design

## Goal

Evaluate whether per-channel LogNP companding can reduce the accuracy loss of
A4 activation quantization for the official Vanilla CSPN, DySPN, NLSPN, and
CompletionFormer NYU models, while preserving a path toward a standard integer
backend.

The transform is

```text
z = sign(x) * log2(1 + |x| / alpha)
```

and its inverse is

```text
x_hat = sign(z_hat) * alpha * (2^|z_hat| - 1)
```

`alpha` is calibrated independently for each activation channel. The initial
implementation is a float QDQ reference model; it must not be reported as a
fused integer kernel.

## Scope

The experiment covers activation quantization and optional calibration-time
compensation. It reuses the existing hardware-aligned observer and prediction
export flow.

Included configurations:

1. FP32 baseline.
2. Uniform W8A4 MinMax baseline.
3. W8A4 LogNP per-tensor.
4. W8A4 LogNP per-channel.
5. W8A4 LogNP per-channel with bias correction.
6. W8A4 LogNP per-channel with per-output-channel weight correction.
7. W4A4 LogNP per-channel with weight correction.
8. Existing W4A8 control to isolate weight sensitivity.

The first run uses the existing fixed 128-sample calibration set and random
64-sample NYU evaluation set so that all configurations are directly
comparable. The four official full-size model structures and their existing
converged checkpoints remain unchanged.

Out of scope for the first implementation:

- changing model architecture or retraining from scratch;
- claiming exact standard-integer execution for `log2` or `2^x`;
- folding LogNP into Conv or Linear weights;
- compensating clipping or information that has already been lost;
- replacing the existing uniform W8A8/W4A8/W4A4 results.

## Alternatives

### A. Float LogNP QDQ reference model

Apply LogNP, quantize in the transformed domain, dequantize, and apply the
inverse transform around each activation boundary. This is simple to validate
and isolates quantization quality, but it includes floating-point transform
overhead and is not a hardware latency result.

### B. LUT or piecewise-linear integer approximation

Approximate the forward and inverse transforms with fixed-point LUTs or
piecewise-linear segments. This is the recommended follow-up for hardware
evaluation because it exposes memory, lookup, interpolation, and error costs.
It requires a separate integer-kernel contract and should not be mixed with the
reference accuracy experiment.

### C. Keep activations in the transformed domain through Conv

This would avoid an explicit inverse transform, but ordinary convolution is
not preserved because `Conv(Phi_inverse(z))` is not equal to `Conv(z)`. It would
require a new nonlinear convolution kernel and is rejected for this scope.

The implementation starts with A and records the requirements for B.

## Quantization Contract

For a tensor shaped `[N, C, H, W]`, `alpha` is broadcast as `[1, C, 1, 1]`.
The transformed tensor is quantized uniformly:

```text
q = clamp(round(z / sz), qmin, qmax)
z_hat = q * sz
```

Signed activations use symmetric signed A4 with `qmin=-7`, `qmax=7`.
Outputs known to be after ReLU use unsigned A4 with `qmin=0`, `qmax=15`.
The ReLU path must not waste half of its code range on negative values.

The reference implementation uses per-channel `alpha` and per-channel
transformed-domain `sz`. Calibration stores `alpha`, `sz`, signedness, bit
width, observed range, clipping rate, and finite-value checks for every site.

`alpha` is selected from calibration statistics and must be finite and positive.
Candidate policies are percentile-based robust fitting followed by a held-out
error check. The selected policy is fixed before evaluation and is not fitted
on the 64 evaluation samples.

The inverse uses a bounded exponent implementation such as `expm1(log(2) *
abs(z_hat))`, with an explicit upper clamp. Any non-finite output is recorded
as a failure and cannot silently become a valid prediction.

## Placement and Backend Boundary

The first implementation places the LogNP-QDQ module immediately after the
producer activation, before the consuming Conv or Linear operation. The model
therefore computes the normal floating-point operation on `x_hat`, preserving
the model's mathematical graph while making the activation error observable.

This placement is not a direct standard-integer MAC implementation. The
original hardware contract uses a single uniform input scale `sx` and bias
scale `sx * sw[o]`. LogNP produces a nonuniform reconstruction, so its
transformed-domain scale must not be used as `sx`. A future integer path must
either reconstruct into a common integer scale before the MAC or define a
dedicated LUT/dequantization datapath and a new bias contract.

Conv-BN folding remains before calibration. Weight quantization remains
per-output-channel signed symmetric. Merge sites retain their existing
requantization behavior; separate branches are calibrated independently before
the merge.

## Compensation

Compensation is evaluated only after the uncorrected LogNP reference is
validated.

1. Bias correction computes the mean output residual on calibration data and
   updates one bias value per output channel. It is the lowest-cost control.
2. Weight correction solves a regularized per-output-channel least-squares
   problem using the original output as the target and LogNP-reconstructed
   activation patches as the input. The correction is fitted only on
   calibration data and then frozen.
3. If least-squares correction is unstable or increases weight outliers, a
   short QAT experiment with frozen `alpha` and STE fake quantization is used as
   a secondary method. It must report whether W4 weight error increased.

Compensation is allowed to reduce average calibration error but is not allowed
to alter the evaluation set, calibration split, model topology, or clipping
policy. It cannot recover values clipped by the transformed-domain quantizer.

## Measurements

For every activation site and model configuration, record:

- original-domain MAE, RMSE, p50, p75, p99, and p99.9 absolute error;
- transformed-domain SQNR;
- zero-code rate, clipping rate, sign-flip rate, and non-finite count;
- per-channel `alpha`, transformed scale, and observed percentiles;
- layer/module error contribution and propagation-state error;
- end-to-end depth RMSE, MAE, ABS_REL, invalid-prediction count, and runtime.

The output package contains CSV summaries, JSON calibration metadata, NPZ
prediction payloads, and side-by-side FP32/quantized/GT/error visualizations
for the same 64 evaluation samples. Runtime for the float reference is marked
as reference-only; only the later LUT implementation may be used for integer
backend latency claims.

## Validation

Unit tests cover:

- forward/inverse transform round-trip on signed and unsigned inputs;
- per-channel broadcasting and channel-isolated calibration;
- monotonicity and finite output under extreme values;
- signed A4 and unsigned ReLU A4 code ranges;
- clipping and non-finite accounting;
- bias correction direction and least-squares correction shape;
- deterministic calibration metadata and prediction export.

Integration tests run one fixed batch through all four official models and
verify that FP32, uniform A4, LogNP per-tensor, and LogNP per-channel produce
finite outputs with matching tensor shapes. The full 64-sample experiment is
accepted only when all four models have complete manifests and no silent
non-finite predictions.

## Success Criteria

The method is considered promising if per-channel LogNP reduces W8A4
end-to-end error relative to uniform W8A4 on CSPN and NLSPN without increasing
invalid predictions, and if compensation provides an additional improvement on
the held-out 64 samples. W4A4 results must be reported separately because
weight and activation errors interact.

The method is not considered hardware-ready until a LUT or piecewise-linear
implementation is measured for approximation error, memory/lookup overhead,
and the corrected integer bias/requantization contract.
