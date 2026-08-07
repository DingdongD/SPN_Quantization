# AdaRound Alignment Audit

## Confirmed Defects

The previous strict AdaRound path was not aligned with the reference objective:

- MSE averaged every output element instead of summing the output-channel
  dimension before averaging.
- A bare convolution target excluded its immediately following activation.
  This was critical for DySPN `base.gd_dec1_`, where ReLU support changes feed
  the guidance/output head and propagation loop.
- AdaRound and BRECQ shared one temperature schedule. AdaRound now uses the
  AIMET defaults `reg=0.01`, `warm_start=0.2`, and cosine beta decay. BRECQ uses
  `reg=0.01`, no warm start, and linear beta decay.
- The final hard state could be exported even when its local reconstruction
  loss was worse than the initial RTN state.

The integer deployment lattice remains the project-standard signed symmetric
W4 per-output-channel backend with codes in `[-7, 7]`. Exact contract code and
dequantized-weight fingerprints are checked during deployment replay.

## DySPN Diagnostic Evaluation

The diagnostic run used 64 calibration samples, 2000 reconstruction steps,
and the same 64 evaluation samples for every method.

| Method | Mean RMSE (m) | Mean MAE (m) | Mean ABS_REL | Invalid |
|---|---:|---:|---:|---:|
| RTN W4A8 | 0.125862 | 0.050793 | 0.016688 | 0 |
| Previous AdaRound W4A8 | 2.967507 | 2.760431 | 1.054564 | 0 |
| Aligned AdaRound W4A8 | 0.127643 | 0.053937 | 0.017779 | 0 |
| BRECQ W4A8 | 0.124312 | 0.049646 | 0.016456 | 0 |

The aligned AdaRound contract removes the catastrophic failure but is not
accepted because its mean RMSE is 0.001781 m worse than RTN in this diagnostic
configuration. A formal-quality AdaRound run should use the method default of
15000 steps and a larger calibration set before reassessing benefit.

## Propagation Error

| Signal | Previous RMSE | Aligned RMSE | Previous SQNR (dB) | Aligned SQNR (dB) |
|---|---:|---:|---:|---:|
| Affinity | 0.157021 | 0.094683 | 6.872 | 11.274 |
| Confidence logits | 4.012500 | 0.568286 | 9.075 | 26.103 |
| Initial prediction | 4.160151 | 0.386110 | -2.912 | 17.815 |
| Propagation states | 3.574304 | 0.128959 | -1.463 | 30.042 |

The old local rounding solution corrupted the guidance branch before the DySPN
loop. The propagation operator then accumulated and spread that injected error;
the CUDA deformable-convolution implementation was not the source of the
failure.

## Current Artifacts

The diagnostic artifacts used during defect isolation were removed after the
aligned implementation was validated. The retained strict reconstruction and
64-sample end-to-end evaluation are stored under:

- `profile_logs/nyu_strict_w4a8_reconstruction_current/dyspn/adaround_strict`
- `profile_logs/nyu_strict_w4a8_evaluation/adaround/dyspn`
