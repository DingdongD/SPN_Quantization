# CSPN Scale-Aware Group-8 Design

## Scope

This study keeps the official CSPN architecture, converged checkpoint, fixed
128-sample NYU calibration set, fixed 64-sample evaluation set, per-output-
channel symmetric W4 weights, MinMax A4 activations, FP32 bias and guidance,
and the existing propagation-aware A8/Q13/INT32 path unchanged.

Scale-aware grouping applies only to strict Conv2d and ConvTranspose2d input
activation sites whose channel count is divisible by eight. Output, ReLU,
rotation, affinity, confidence, propagation state, propagation coefficients,
guidance, and non-divisible input sites retain the current contiguous Group-8
or tensor policy.

The experiment compares exactly:

1. `W4A4_G8_MINMAX`: current contiguous Group-8 MinMax baseline;
2. `W4A4_G8_SCALE_AWARE`: RMS-ranked input-channel Group-8 MinMax.

## Channel sensitivity and grouping

For every eligible consumer input, calibration accumulates the squared
activation sum and scalar count per channel over the same 128 ordered samples:

\[
r_c = \sqrt{\mathbb{E}[x_c^2]}.
\]

For a group `G`, scale dispersion is:

\[
D_G = \frac{\max_{c\in G} r_c}
            {\min_{c\in G} r_c + \epsilon}.
\]

The implementation uses a stable ascending sort by `(r_c, channel_index)` and
forms consecutive groups of eight in sorted order. For the stated objective
with fixed group cardinality and no channel-topology constraint, adjacent
sorted grouping is the deterministic one-dimensional range-matching policy.
It avoids introducing a combinatorial solver while directly placing channels
with similar RMS in the same group.

Exactly zero-RMS channels sort first. The reported dispersion uses
`epsilon=1e-12` in depth-model activation units; it does not alter MinMax
thresholds or quantized values.

## Consumer-side permutation

For each eligible consumer, let `P` be the fixed channel permutation:

\[
x' = Px, \qquad W' = WP^T.
\]

The input hook permutes the activation immediately before its consumer. The
consumer weight is permuted before W4 quantization:

- Conv2d: reorder weight dimension 1;
- ConvTranspose2d: reorder weight dimension 0.

The permutation is consumer-local, so different consumers of the same tensor
may use different groupings without changing residual or concatenation graph
semantics. Disabling activation rounding while retaining the paired input and
weight permutation must reproduce the unpermuted convolution output within the
existing floating-point tolerance.

The runtime reference path contains an explicit fixed channel shuffle. This
experiment does not claim that the shuffle has been removed from a deployed
kernel. A later graph-level pass may absorb a shared permutation into a
producer only when all consumers and structural branches satisfy the same
mapping.

## Quantization contract

The permuted activation uses contiguous Group-8 MinMax A4. Group thresholds
are derived from per-channel MinMax extents reordered by the same permutation.
Signed sites use symmetric A4 with `qmin=-7`, `qmax=7`; nonnegative input sites
use unsigned A4 with `qmin=0`, `qmax=15` and exact zero.

The paired permuted weight is quantized with the unchanged per-output-channel
symmetric W4 policy. Permuting an input dimension does not change an output
channel's weight value set, so its MinMax W4 scale and aggregate W4 error must
match the contiguous baseline up to numerical ordering effects. Bias remains
FP32 and is not recalibrated.

The input activation recorder maps quantized values and integer codes back to
original channel order before channel-sensitive error accounting. Module
execution still consumes the permuted quantized tensor. This prevents channel
shuffle from being counted as activation quantization error.

## Strict validation

The implementation rejects incomplete or duplicate permutation coverage,
non-bijective indices, channel-count changes, non-finite RMS values, unsupported
module types, and permutations applied to non-input sites. There is no fallback
from a declared scale-aware site to contiguous grouping.

Every eligible permutation and inverse permutation is serialized. The manifest
records module, channel count, group size, per-channel RMS, original channel
indices in grouped order, and per-group dispersion before and after grouping.

## Metrics and artifacts

Both configurations run on the same 64 evaluation identities and report RMSE,
MAE, AbsRel, iRMSE, flat RMSE, boundary RMSE, activation SQNR, new-zero rate,
saturation rate, block error, propagation-step error, and finite-value ratios.

Scale-aware analysis additionally reports:

- mean, median, maximum, and weighted mean group dispersion;
- dispersion reduction relative to contiguous Group-8;
- per-site activation SQNR and zero-collapse changes;
- whether W4 scales and weight reconstruction error remain invariant;
- per-sample RMSE deltas and the regression tail.

Prediction payloads are exported for both configurations. No new visualization
layout is required; the existing prediction/error plotting style is reused.

## Success criterion

Scale-aware grouping is useful only if it lowers aggregate RMSE without more
than 1% relative regression in MAE, AbsRel, boundary RMSE, or final propagation
block MSE, and without non-finite output. Lower dispersion or higher tensor
SQNR alone is not sufficient.

## Verification

Unit tests cover deterministic grouping, zero-RMS channels, bijective inverse
mapping, Conv2d and ConvTranspose2d permutation equivalence, exact Group-8
range construction, original-order activation accounting, strict contract
validation, and artifact coverage. The real CUDA experiment uses 128 NYU
calibration samples and the fixed 64-sample evaluation set. The complete
repository test suite must pass before results are reported.
