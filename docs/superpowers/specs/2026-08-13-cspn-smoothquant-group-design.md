# CSPN SmoothQuant Group-A4 Design

## Objective

Evaluate whether selective input-channel SmoothQuant improves strict CSPN
Group-A4 by reducing within-group activation range imbalance without moving
excessive error into per-output-channel W4 weights.

## Quantization Contract

For each eligible Conv2d or Linear input with calibrated channel maxima `A_c`
and input-channel weight maxima `W_c`, compute:

```text
s_c = A_c^alpha / W_c^(1-alpha)
x'_c = x_c / s_c
W'[:, c] = W[:, c] * s_c
```

The transformed activation range is `A'_c = A_c / s_c`. Group-A4 receives
one range per contiguous channel group:

```text
A'_g = max(A'_c for c in group g)
```

The scale count must equal `C / group_size`. Sites whose channel count is not
divisible by the group size remain tensor sites and receive one transformed
range. ReLU remains unsigned A4; signed inputs remain symmetric A4.

SmoothQuant applies only to ordinary Conv2d/Linear input owners that are
quantized by the hardware instrumentor and whose input is not externally owned.
It does not apply to ReLU-output owners, structural merges, rotation boundaries,
guidance, confidence, affinity, offset, propagation state, or depth anchors.

## Experiment Matrix

Use the same official CSPN checkpoint, 128 calibration samples, fixed 64
evaluation samples, propagation A8/Q13/INT32 contract, FP32 guidance, and FP32
bias as the activation-resolution study.

- FP32 and Group16/Group8 W4A4 baselines;
- SmoothQuant alpha 0.25, 0.50, 0.75 with Group16 and Group8 W4A4;
- FP-weight plus SmoothQuant Group-A4 activation isolation;
- transformed W4 weight plus FP-activation isolation.

The two isolation configurations use alpha 0.50 for Group16 and Group8.

## Metrics

Record RMSE, MAE, AbsRel, iRMSE, flat RMSE, boundary RMSE, activation new-zero
rate and SQNR, transformed W4 weight SQNR and zero rate, block-output MSE/SQNR,
and per-site before/after channel imbalance. Export predictions for FP32,
Group8 baseline, and the best calibration-selected SmoothQuant Group8 result.

Configuration ranking uses calibration block-output MSE and SQNR only.
Evaluation is used for reporting, not selection.

## Outputs

Write runtime artifacts under:

```text
/workspace/SPN_Quantization/profile_logs/nyu_cspn_smoothquant_group/
```

The directory contains aggregate/sample metrics, activation and weight
diagnostics, block metrics, metadata, prediction payloads, and prediction/error
visualizations. Runtime artifacts remain outside Git.

## Acceptance Criteria

- Unsmooth configurations reproduce the current Group16/Group8 path.
- SmoothQuant FP32 algebraic equivalence is within the declared folding error.
- Group maxima have exactly `C / group_size` entries and are derived after
  channel scaling.
- The runner rejects missing or extra eligible SmoothQuant modules.
- Every configuration has 64 finite evaluation samples.
- Report activation benefit and W4 cost separately before claiming an
  end-to-end improvement.
