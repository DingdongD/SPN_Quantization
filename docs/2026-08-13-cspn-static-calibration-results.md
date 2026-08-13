# CSPN Group-A4 Static Calibration Results

## Protocol

The experiment uses the official CSPN architecture and converged
`cspn_iter24/best.pt` checkpoint. It performs two static calibration passes on
the same 128 real NYU training samples:

1. per-channel MinMax range observation;
2. 2048-bin grouped histogram collection within those ranges.

All methods use the same per-output-channel symmetric W4 weights, static
Group-8 A4 activation boundaries, FP32 bias and guidance, and propagation-aware
A8/INT16-Q13/INT32 path. The strict activation contract contains 71 sites and
1673 scales. Evaluation uses the existing fixed 64 NYU validation samples.

The compared calibration methods are MinMax, P99.9, P99.99, and
Histogram-MSE. No runtime range calculation or online scale update is used.

## End-to-end results

| Calibration | RMSE (m) | MAE (m) | AbsRel | iRMSE | Flat RMSE | Boundary RMSE |
|---|---:|---:|---:|---:|---:|---:|
| MinMax | **0.313318** | **0.228845** | 0.097618 | 0.072664 | **0.281930** | **0.501738** |
| P99.9 | 1.354118 | 1.016201 | 0.266681 | 0.165460 | 1.266145 | 1.812838 |
| P99.99 | 0.490240 | 0.297412 | **0.086061** | **0.066229** | 0.419497 | 0.840053 |
| Histogram-MSE | 0.988717 | 0.664688 | 0.161745 | 0.088809 | 0.903729 | 1.427456 |

MinMax exactly reproduces the preceding Group-8 baseline RMSE of 0.313318 m,
which verifies that explicit range overrides preserve the existing path.

P99.99 improves AbsRel and iRMSE and beats MinMax on 34 of 64 samples. Its
median per-sample RMSE delta is -0.010 m. The distribution has a severe positive
tail: the P90 delta is +0.884 m and the maximum regression is +1.200 m. Large
depth planes and boundaries are systematically underestimated in those cases,
so aggregate RMSE and boundary RMSE regress.

## Activation reconstruction

| Calibration | Mean threshold / MinMax | Activation SQNR | Clipping share | New-zero rate |
|---|---:|---:|---:|---:|
| MinMax | 1.000 | 13.74 dB | 0.008% | 31.71% |
| P99.9 | 0.516 | 18.57 dB | 4.37% | 15.16% |
| P99.99 | 0.699 | 18.28 dB | 0.67% | 18.29% |
| Histogram-MSE | 0.510 | **21.63 dB** | 6.75% | **11.37%** |

Histogram-MSE gives the best aggregate activation SQNR and substantially
reduces small-value collapse, but its RMSE is 0.989 m. Local activation MSE is
therefore not a valid CSPN end-to-end calibration objective by itself.

The activation error decomposition is numerically closed: clipping, rounding,
and zero-collapse energy sum to total error with relative discrepancy below
`8e-16` for every method. All predictions and propagation statistics are
finite. This rules out non-finite arithmetic and error-accounting defects as
the explanation for the regression.

## Error location

Relative to MinMax, Histogram-MSE increases initial-depth block MSE by 1.3020
and final propagation block MSE by 1.3945. P99.99 increases those blocks by
0.3294 and 0.3309 respectively. Several earlier encoder and decoder blocks have
lower local MSE, but that benefit does not survive the depth head.

The dedicated propagation arithmetic remains valid for all methods:

- anchor MAE is zero;
- coefficient-sum maximum error is zero;
- contraction-violation rate is zero;
- non-finite ratio is zero.

The increased propagation block error is injected through the quantized
initial-depth and feature tensors, not created by an affinity normalization or
fixed-point propagation defect.

## Conclusion

None of the tested non-MinMax static calibrators satisfies the expansion
criterion. P99.99 is the least damaging alternative and helps a slight majority
of samples, but its long regression tail makes it unsuitable as a global CSPN
Group-A4 policy. P99.9 and Histogram-MSE clip too aggressively.

The next useful direction is not another global percentile. Calibration must
be selective or depth-aware: preserve MinMax at the depth head and
propagation-sensitive upstream sites, and only clip sites whose output error is
stable under held-out calibration samples. Such a method needs a held-out
calibration objective based on block/depth error rather than tensor SQNR.

Artifacts are under
`profile_logs/nyu_cspn_static_calibration_group8`. They include 256 finite
sample rows, 6692 per-scale threshold rows, activation/block/propagation
diagnostics, two sets of 64 prediction payloads, and PNG/PDF figures.
