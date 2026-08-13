# CSPN SmoothQuant plus Group-A4 Results

## Protocol

- Official CSPN architecture and converged `cspn_iter24/best.pt` checkpoint.
- Real NYU training split for 128-sample calibration.
- Fixed 64-sample NYU validation subset used by the preceding CSPN studies.
- Ordinary encoder, decoder, and depth-head Conv/Linear weights use W4.
- Activation inputs and the existing strict activation boundaries use Group-A4.
- Guidance stays FP32. Propagation uses A8, signed INT16 Q13 coefficients,
  and INT32 accumulation.
- SmoothQuant is applied only at eligible ordinary module inputs. ReLU outputs,
  guidance, confidence, propagation state, and structural boundaries are not
  rescaled.
- The exact transform is `x' = x / s`, `W' = W * s`; group ranges are computed
  from the transformed per-channel calibration maxima.

The alpha sweep used `0.25`, `0.50`, and `0.75` for Group 16 and Group 8.
Alpha selection used only the 128-sample calibration block-output MSE. All
declared configurations were then evaluated on the same 64 samples.

## End-to-end results

| Configuration | RMSE (m) | Delta from matching RTN |
|---|---:|---:|
| FP32 | 0.166932 | - |
| PA only | 0.175221 | - |
| W4 only | 0.203795 | - |
| SQ W4 only, alpha 0.50 | 0.214522 | +0.010727 |
| A4 only, Group 16 | 0.343671 | - |
| SQ A4 only, Group 16, alpha 0.50 | 0.407222 | +0.063551 |
| A4 only, Group 8 | 0.318635 | - |
| SQ A4 only, Group 8, alpha 0.50 | 0.383340 | +0.064705 |
| W4A4 Group 16 | 0.369955 | - |
| SQ W4A4 Group 16, alpha 0.25 | 0.830848 | +0.460893 |
| SQ W4A4 Group 16, alpha 0.50 | 0.426425 | +0.056470 |
| SQ W4A4 Group 16, alpha 0.75 | 0.366676 | -0.003279 |
| W4A4 Group 8 | 0.313318 | - |
| SQ W4A4 Group 8, alpha 0.25 | 0.619386 | +0.306069 |
| SQ W4A4 Group 8, alpha 0.50 | 0.364370 | +0.051052 |
| SQ W4A4 Group 8, alpha 0.75 | 0.329291 | +0.015973 |

Group 16 with alpha 0.75 is the only full W4A4 case that improves over its
matching RTN baseline. The gain is 0.003279 m, or 0.89%, and does not transfer
to Group 8.

## Error analysis

At alpha 0.50, the mean channel-maximum imbalance across the 33 divisible
SmoothQuant sites falls from 2.50 to 2.01. The transformed W4 aggregate SQNR
also rises from 11.93 dB to 14.71 dB. Those two local statistics do not predict
the end-to-end result:

- Group-8 activation SQNR falls from 13.74 dB to 13.39 dB.
- Activation clipping-error fraction rises from 0.008% to 3.67%.
- A4-only Group-8 RMSE increases by 20.31%.
- SmoothQuant W4-only RMSE increases by 5.26%, despite its higher transformed
  weight SQNR.

The activation problem is therefore not just the calibration-set channel
imbalance. Rescaling changes the unseen-sample group ranges, and fixed MinMax
calibration clips transformed activations more often. On the weight side,
transformed-domain SQNR weights all coefficients equally and does not capture
the output sensitivity after multiplication by `x / s`.

Calibration selected alpha 0.50 for both group sizes because it minimized the
aggregate block MSE. Evaluation shows alpha 0.75 is better for final depth
RMSE, especially for Group 16. This is a proxy mismatch, not justification to
select alpha on the evaluation set. A useful follow-up would require a held-out
calibration validation subset and a depth-aware objective.

## Conclusion

The tested global SmoothQuant plus Group-A4 scheme is rejected as the CSPN
default. It reliably compresses channel imbalance but does not provide stable
W4A4 accuracy. Further work should be selective and output-aware: restrict
rescaling to modules with stable transformed ranges, use percentile or learned
clipping on a held-out calibration subset, and optimize module-output error
rather than transformed tensor SQNR alone.

Artifacts are under
`profile_logs/nyu_cspn_smoothquant_group`. They include all sample, regional,
block, activation, channel, weight, and propagation metrics; 64 prediction
payloads for each exported configuration; and PNG/PDF RMSE and prediction
comparisons.
