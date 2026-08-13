# CSPN Stratified SmoothQuant plus Group-A4 Results

## Protocol

- Official CSPN ResNet-18 architecture with 24 propagation steps.
- Converged `cspn_iter24/best.pt` checkpoint.
- Real NYU training split with the fixed 128-sample stratified calibration set:
  32 activation-tail samples plus 96 k-medoids coverage samples.
- The same fixed 64-sample NYU validation subset as the random-calibration
  SmoothQuant experiment.
- Ordinary encoder, decoder, and depth-head weights use signed symmetric W4.
- Ordinary activations use static MinMax Group-A4 with group sizes 16 or 8.
- Guidance remains FP. Propagation uses A8 activations, signed INT16 Q13
  coefficients, and INT32 accumulation.
- SmoothQuant uses `x' = x / s` and `W' = W * s` only at eligible ordinary
  module inputs. The alpha sweep is 0.25, 0.50, and 0.75.
- Alpha selection uses only 128-sample calibration block-output MSE.

The calibration manifest SHA-256 is
`a56eec33c7c18b251e1ee704fac1241e22a325bb18b5bd879b9cf6cfd7c93cdf`.

## End-to-end Results

| Configuration | Random-128 RMSE (m) | Stratified-128 RMSE (m) | Delta from stratified matching RTN |
|---|---:|---:|---:|
| FP32 | 0.166932 | 0.166932 | - |
| W4 only | 0.203795 | 0.204621 | - |
| SQ W4 only, alpha 0.50 | 0.214522 | 0.204136 | -0.24% vs W4 only |
| A4 only, Group 16 | 0.343671 | 0.399646 | - |
| SQ A4 only, Group 16, alpha 0.50 | 0.407222 | 0.442086 | +10.62% |
| A4 only, Group 8 | 0.318635 | 0.367289 | - |
| SQ A4 only, Group 8, alpha 0.50 | 0.383340 | 0.379035 | +3.20% |
| W4A4 Group 16 | 0.369955 | 0.417303 | - |
| SQ W4A4 Group 16, alpha 0.25 | 0.830848 | 1.060833 | +154.21% |
| SQ W4A4 Group 16, alpha 0.50 | 0.426425 | 0.507145 | +21.53% |
| SQ W4A4 Group 16, alpha 0.75 | 0.366676 | 0.407825 | -2.27% |
| W4A4 Group 8 | 0.313318 | 0.347124 | - |
| SQ W4A4 Group 8, alpha 0.25 | 0.619386 | 0.745447 | +114.75% |
| SQ W4A4 Group 8, alpha 0.50 | 0.364370 | 0.399747 | +15.16% |
| SQ W4A4 Group 8, alpha 0.75 | 0.329291 | 0.359289 | +3.50% |

Only Group 16 with alpha 0.75 improves over its matching stratified RTN
baseline. The 0.009478 m gain is present on 37 of 64 samples, but it does not
transfer to Group 8. Group 8 with alpha 0.75 is worse on 51 of 64 samples.

Calibration block-output MSE selects alpha 0.50 for both group sizes. Those
selected models regress by 21.53% and 15.16%, respectively. Alpha 0.75 is the
evaluation sweep winner and must not be reported as a calibration-selected
setting.

## Quantization Error

The stratified set exposes activation tails to static MinMax. It reduces
unseen-sample clipping but expands the A4 range and coarsens the quantization
step:

| Configuration | Activation SQNR random / stratified (dB) | New-zero rate random / stratified |
|---|---:|---:|
| W4A4 Group 16 | 13.18 / 12.57 | 34.18% / 36.46% |
| SQ W4A4 Group 16, alpha 0.75 | 12.66 / 12.22 | 33.37% / 34.79% |
| W4A4 Group 8 | 13.74 / 13.00 | 31.71% / 34.01% |
| SQ W4A4 Group 8, alpha 0.75 | 13.28 / 12.73 | 30.59% / 32.11% |

Baseline clipping contributes less than 0.003% of activation error after
stratified calibration. The dominant failure is therefore not clipping: tail
coverage increases the static MinMax scale, and more small nonzero activations
collapse to zero.

SmoothQuant improves aggregate transformed W4 SQNR. For stratified alpha 0.75,
weight SQNR rises from 11.93 dB to 13.15 dB. This does not imply better final
depth output because activation clipping is reintroduced and error is moved
between modules. At alpha 0.50, Group-8 initial-depth block SQNR falls from
15.21 dB to 13.27 dB and propagation-output SQNR falls from 20.41 dB to
18.77 dB, despite the improved transformed weight SQNR.

## Conclusion

Global SmoothQuant plus per-group A4 does not provide a robust CSPN W4A4
improvement. Group 16 with alpha 0.75 gives a small local gain, but the
calibration objective does not select it, Group 8 does not reproduce it, and
all stratified SmoothQuant variants remain worse than the random-calibrated
Group-8 RTN result of 0.313318 m.

The current default should remain Group-8 RTN with the random-128 calibration
artifact. A future SmoothQuant experiment needs module-selective scaling and a
held-out depth-aware selection objective; global alpha and aggregate block MSE
are insufficient.

Artifacts are under
`profile_logs/nyu_cspn_smoothquant_group_stratified128`.
