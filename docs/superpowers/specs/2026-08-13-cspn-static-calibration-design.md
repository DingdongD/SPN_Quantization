# CSPN Group-A4 Static Calibration Design

## Scope

This study compares static activation calibration methods on the official CSPN
model only. It keeps the converged checkpoint, W4 weight quantization, Group-8
activation granularity, propagation-aware A8/Q13 path, 128 NYU calibration
samples, and fixed 64 NYU evaluation samples unchanged.

The declared configurations are:

1. `W4A4_G8_MINMAX`
2. `W4A4_G8_PERCENTILE_P999`
3. `W4A4_G8_PERCENTILE_P9999`
4. `W4A4_G8_HIST_MSE`

All four configurations remain static PTQ. Inference does not compute sample
statistics or change a scale.

## Quantization boundary

The experiment covers the same 71 strict activation sites and 1673 activation
scales as the audited CSPN Group-8 baseline. Divisible channel dimensions use
contiguous Group-8 ranges; non-divisible sites retain their declared tensor
range. Ordinary Conv/Linear inputs, declared outputs, fused ReLU outputs, and
the two existing rotation boundaries use the selected calibration method.

Guidance, affinity, confidence, propagation state, propagation coefficients,
and structural tensors retain their current dedicated policy. Guidance remains
FP32. Propagation uses A8, signed INT16 Q13 coefficients, and INT32 accumulation.
Weights remain per-output-channel symmetric W4 and biases remain FP32.

## Two-pass calibration

The first pass runs the fixed 128 samples through the prepared FP model and
records the existing per-channel minimum and maximum. These values define each
tensor or Group-8 entity's absolute histogram range.

The second pass runs the same 128 samples in the same order and records 2048
uniform bins for every declared activation scale. For signed activation, the
histogram contains absolute magnitudes. For unsigned activation, it contains
the nonnegative value. A site reshapes its activation into groups, combines the
group index and bin index, and uses one `torch.bincount` update. It must not
launch one histogram operation per group.

Histogram counts are integer and deterministic for fixed activation tensors.
The recorder rejects non-finite tensors, channel-count changes, undeclared
sites, duplicate ownership, and incomplete site/scale coverage. A scale entity
that is exactly zero across calibration is represented explicitly as a
zero-range entity; it is not replaced by a different observer or neighboring
range.

## Threshold derivation

### MinMax

The threshold is the first-pass entity maximum. This reproduces the existing
Group-8 baseline.

### Percentile

For P99.9 and P99.99, the threshold is the upper boundary of the first
histogram bin whose cumulative count reaches the declared percentile. Signed
activation uses the absolute-value CDF; unsigned activation uses its
nonnegative CDF. Thresholds remain aligned to histogram boundaries.

### Histogram-MSE

For every nonzero histogram entity, all 2048 bin boundaries are candidate
thresholds. Candidate QDQ uses the real A4 code contract:

- signed symmetric: `qmin=-7`, `qmax=7`, zero point 0;
- unsigned affine: `qmin=0`, `qmax=15`, zero point 0.

For each candidate, histogram-bin centers are clipped, rounded, dequantized,
and compared with the original centers. Bin counts weight the squared error.
The lowest-MSE threshold wins; ties select the larger threshold to avoid
unnecessary saturation.

Because every entity's bin centers are normalized by its own maximum, the
candidate error matrix is shared by all entities of the same signedness. A
matrix multiplication between histogram counts and this error matrix searches
all thresholds without Python loops over entities.

## Experiment and selection

No calibration hyperparameter is tuned on the 64 evaluation samples. The two
percentiles and Histogram-MSE are fixed before evaluation. All four methods run
on exactly the same evaluation identities, and all outputs must be finite.

The study reports each method rather than declaring a calibration winner from
an evaluation-dependent search. A method is considered suitable for expansion
to Group-16 and the other models only when it improves RMSE over MinMax without
materially worsening MAE, AbsRel, boundary RMSE, or propagation error.

## Metrics and artifacts

The runner writes:

- per-sample RMSE, MAE, AbsRel, iRMSE, flat RMSE, and boundary RMSE;
- aggregate and regional depth metrics;
- activation SQNR, new-zero rate, saturation rate, clipping-error fraction,
  effective code count, and selected threshold/max-range ratio;
- per-block output MSE and SQNR;
- propagation-step error;
- a strict configuration manifest and metadata with calibration/evaluation
  identities;
- 64 prediction payloads for MinMax and the lowest-RMSE non-MinMax method,
  after all methods have already been evaluated.

Plots include calibration-method RMSE, activation error composition, and
GT/FP32/MinMax/candidate prediction and absolute-error comparisons. Plotting is
derived from completed CSV/NPZ artifacts and does not rerun inference.

The output root is
`profile_logs/nyu_cspn_static_calibration_group8`.

## Verification

Unit tests cover histogram updates, percentile boundaries, signed and unsigned
MSE threshold selection, deterministic tie-breaking, Group-8/tensor coverage,
configuration contracts, and strict artifact coverage. Integration verification
checks 4 x 64 finite metric rows and exact prediction identities. The complete
repository test suite must pass before the result is reported.
