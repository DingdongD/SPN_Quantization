# CSPN Outlier Channel Isolation Design

## Scope

This study keeps the official converged CSPN model, contiguous Group-8
topology, static MinMax calibration, per-output-channel symmetric W4 weights,
unsigned A4 nonnegative activations, FP32 bias and guidance, and the existing
A8/Q13/INT32 propagation path. RMS clustering, maximum clustering, snake
grouping, channel permutation, and channel duplication are outside scope.

The method is called outlier channel isolation (OCI), not Outlier Channel
Splitting (OCS). OCI assigns an independent A4 scale to one selected outlier
channel while the other seven channels retain one shared A4 scale. It does not
change channel count, weights, or the floating-point graph.

## Outlier harm

For each original contiguous group `G`, calibration records each channel
maximum `M_c`. The candidate outlier and remaining maximum are:

\[
o=\arg\max_{c\in G}M_c,\qquad
M_{G\setminus o}=\max_{c\in G,c\ne o}M_c.
\]

For unsigned A4, the two zero thresholds are:

\[
T_G=M_o/30,\qquad T_{G\setminus o}=M_{G\setminus o}/30.
\]

On calibration activations, the harm score is:

\[
H_o=\sum_{c\ne o}
\left[P(0<x_c<T_G)-P(0<x_c<T_{G\setminus o})\right].
\]

The implementation also records the exact rescued element count, rescued
activation energy, number of affected victim channels, `M_o / M_second`, and
the per-victim rescue rate. Exact reference zeros are excluded.

## Quantization contract

The contiguous baseline assigns all eight channels scale `M_o / 15`. OCI
assigns channel `o` scale `M_o / 15` and the other seven channels scale
`M_second / 15`. Both scales use unsigned A4 with `qmin=0`, `qmax=15`, and
zero point zero. No channel is promoted to A8, split, reordered, or removed.

OCI is represented as an explicit channel-scale override on the local QDQ
owner that first quantizes the activation. Eligible owners are unsigned A4
Group-8 ReLU outputs and module sites whose input has not already lost the
candidate interval upstream. Declared isolation must identify an original
contiguous Group-8 member and its calibrated maximum. Missing or inconsistent
declarations fail directly; there is no fallback to ordinary Group-8.

Harm is collected with the contiguous W4A4 baseline active and immediately
before each local QDQ. This is required because a consumer-input requantizer
cannot recover values already rounded to zero by a producer-output or ReLU
QDQ. Input sites with no remaining positive harm are therefore not selected.

## Candidate selection

Candidate selection uses only the fixed 128-sample calibration set. Groups are
ranked by rescued activation energy, then rescued element count, then stable
site and channel identity. The experiment evaluates deterministic cumulative
budgets of isolated channels, including zero isolated channels and all
positive-harm candidates. Evaluation metrics never participate in selection.

Every budget reports the isolated-channel count and fraction, additional scale
count, affected site count, and calibration rescue statistics. This exposes
the accuracy versus scale-metadata overhead tradeoff.

## Evaluation

All budgets run on the same fixed 64 NYU evaluation samples. The primary
metrics are RMSE, MAE, AbsRel, iRMSE, flat RMSE, and boundary RMSE. Activation
diagnostics report new-zero rate, zero-collapse energy, total rounding energy,
block output error, and propagation-step error.

The result is useful only when a calibration-selected OCI budget improves
aggregate RMSE over contiguous Group-8 without non-finite output or material
regression in boundary RMSE and propagation error. A high `H_o` alone is not
sufficient.

## Deployment boundary

The reference experiment validates mixed `1+7` A4 scales. A standard backend
with exactly one scale per physical Group8 cannot execute OCI without a kernel
or packing extension. This study reports that scale overhead explicitly and
does not claim a standard Group8 kernel speedup.

## Verification

Unit tests cover exact outlier selection, threshold construction, harm counts,
energy accounting, independent outlier scale assignment, unchanged channel
order, unchanged W4 weights, strict declaration validation, calibration-only
selection, and complete artifact coverage. The real experiment uses the fixed
128 calibration and 64 evaluation identities from the previous CSPN studies.
