# CompletionFormer Joint Integer Quantization Results

## Evaluation contract

- Official full CompletionFormer model, source commit
  `2744eddee9b57595dc3064f7d342569736a6803b`.
- Strict checkpoint loading; checkpoint SHA256
  `161e016d83afb6b599494de5750f353d6d518fa2b566831f1ede99c46cf6e455`.
- 64 fixed NYU calibration samples and 64 fixed evaluation samples.
- Two-stage paired calibration: FP targets followed by quantized
  reconstruction on the same indices.
- 16 PVT Attention modules and 16 Transformer/CNN `concat_conv` modules.
- No retraining or evaluation-label scale selection.

## End-to-end RMSE

| Configuration | Mean | Median | P95 | Mean delta vs RTN W4A4 |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 0.119344 | 0.106372 | 0.249747 | -1.382720 |
| JIQ RTN W4A4 | 1.502064 | 1.448147 | 2.327764 | 0.000000 |
| JIQ Attention W4A4 | 1.501215 | 1.484284 | 2.324196 | -0.000849 |
| JIQ Concat W4A4 | 1.419858 | 1.392664 | 2.247959 | -0.082206 |
| JIQ Joint W4A4 | 1.420077 | 1.400728 | 2.272691 | -0.081988 |
| JIQ W4A8 | 0.618692 | 0.631565 | 1.113640 | -0.883373 |

Against RTN W4A4, Attention-only improves 32/64 samples, concat-only improves
59/64, joint W4A4 improves 61/64, and W4A8 improves 64/64. Joint W4A4 remains
1.300732 RMSE above FP32, so the tested joint optimization does not preserve
FP32 performance.

## Local error diagnosis

| Configuration | Attention probability KL | Attention context MSE | Concat block MSE | Concat partial-requant MSE |
| --- | ---: | ---: | ---: | ---: |
| Attention W4A4 | 2.3006e-4 | 9.76195e-3 | - | - |
| Joint W4A4 | 2.1630e-4 | 9.60454e-3 | 326759.11 | 0.336295 |
| Concat W4A4 | - | - | 327842.57 | 0.338272 |
| W4A8 | 5.663e-6 | 6.63727e-4 | 5077.23 | 0.00108615 |

Attention probability zero/saturation ratios are 0.1862/0.0001629 for
Attention W4A4, 0.1952/0.0001821 for joint W4A4, and
0.23678/0.0003168 for W4A8. Attention scale optimization has negligible
end-to-end impact. Concat optimization is consistently beneficial, but its
block-output error remains large and is dominated by high-magnitude early and
middle feature blocks. The primary W4A4 failure is therefore ordinary and
concat feature-path corruption, not the integer Attention probability path.

W4A8 substantially reduces both local and end-to-end error, but remains above
FP32 because weights and the ordinary feature path are still W4. The prediction
contact sheet shows the same result spatially: W4A4 variants introduce strong
texture and depth distortion, while W4A8 is closer to FP32 but still visibly
biased.

## Artifacts

Results are generated outside Git under
`profile_logs/nyu_completionformer_joint_integer_64`. The analysis directory
contains aggregate RMSE, local Attention/concat metrics, and a 64-sample
GT/FP32/quantized prediction contact sheet.
