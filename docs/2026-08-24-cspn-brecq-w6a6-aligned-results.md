# CSPN BRECQ W6A6 Deployment-Aligned Results

## Protocol

- Model: official CSPN ResNet-18 with 24 propagation steps.
- Checkpoint: `nyu_converged_baselines/cspn_iter24/best.pt`, SHA256
  `482fb9532b27bdb0e529da14845d9a63ab546e28974d90dd1197b8f704066855`.
- Calibration: the persisted 128-sample stratified NYU train set, SHA256
  `a56eec33c7c18b251e1ee704fac1241e22a325bb18b5bd879b9cf6cfd7c93cdf`.
- Evaluation: the persisted fixed 64-sample NYU validation set, seed
  `20260812`.
- Reconstruction: 16 execution-ordered blocks, 20,000 steps per block,
  deterministic all-quantized student inputs.
- Deployment contract: `brecq_joint_strict`, per-output-channel W6 and exact
  calibrated A6 semantic-edge contracts. CSPN propagation remains A8 signals,
  INT16 Q13 coefficients and INT32 accumulation.
- No prediction clipping, invalid-value replacement or finite-only averaging.

## Strict Fixed-64 Results

| Configuration | RMSE (m) | MAE (m) | AbsRel | iRMSE | Invalid samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 0.159796 | 0.064791 | 0.021654 | - | 0/64 |
| PA-RTN W8A8 | 0.170966 | 0.084632 | 0.030678 | - | 0/64 |
| QDrop W6A6, three-seed mean | 0.195805 | - | - | - | 0/64 |
| BRECQ W6A6, deployment-aligned | **0.194704** | **0.106041** | **0.035882** | **0.027721** | **0/64** |

The repaired BRECQ result is `0.001101 m` (0.56%) better than the retained
three-seed QDrop W6A6 mean, but `0.023738 m` worse than PA-RTN W8A8. Its
sample RMSE standard deviation is `0.114123 m`, with a range of
`0.055230-0.523358 m`.

All evaluated valid-GT pixels are finite and greater than `1e-4 m`.
`nonfinite_pixels`, `nonpositive_pixels` and `invalid_pixels` are all zero;
the minimum predicted depth across the 64 samples is `0.432798 m`.

## Failure Repair

The retired BRECQ W6A6 path reconstructed weights against FP teacher inputs
and inserted static A6 QDQ only during deployment. Eight of 64 predictions
then contained invalid depth values, so strict RMSE was infinite. The repaired
path reuses the existing joint block reconstructor with quantization
probability `1.0`: each student block is optimized using the deployed A6 input
contract after all preceding reconstructed blocks. This removes the training
and deployment mismatch without clipping or fallback.

`layer4.1` remains the dominant local hotspot. Its all-quantized block loss was
`3.875583` before reconstruction and `0.482012` after reconstruction. The large
residual local loss does not create invalid final depth after downstream
decoder blocks are reconstructed, but it explains why W6A6 remains below the
W8A8 accuracy baseline.

## Artifacts

- Reconstruction contract and validation:
  `profile_logs/nyu_cspn_brecq_w6a6_aligned_64/reconstruction/formal_seed_20260812/`
- Fixed-64 deployment replay and predictions:
  `profile_logs/nyu_cspn_brecq_w6a6_aligned_64/evaluation/brecq/W6A6/cspn/`
- Strict aggregate:
  `profile_logs/nyu_cspn_brecq_w6a6_aligned_64/strict_summary.json`
