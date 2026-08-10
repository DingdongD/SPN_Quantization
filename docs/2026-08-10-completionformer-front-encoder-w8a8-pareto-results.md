# CompletionFormer Front-Encoder W8A8 Pareto Results

## Protocol

- Official CompletionFormer source commit:
  `2744eddee9b57595dc3064f7d342569736a6803b`
- Converged checkpoint SHA256:
  `161e016d83afb6b599494de5750f353d6d518fa2b566831f1ede99c46cf6e455`
- Checkpoint load: zero ignored, missing, or unexpected keys
- Calibration: 64 NYU training samples, seed `20260804`
- Greedy search: 32 different NYU training samples, seed `20260810`
- Final evaluation: fixed 64 NYU validation samples
- Quantization base: joint W4A4 Attention/concat with propagation-aware A8
  boundaries; selected front-encoder units use W8A8
- Training or checkpoint updates: none

Calibration and search indices are disjoint. Every final configuration has 64
unique finite RMSE rows and zero non-finite prediction pixels. The 19 quantized
configurations produced 1,083 front-encoder bit-manifest rows with zero
actual/expected bit mismatches.

## Cost Contract

The denominator contains ordinary quantized Conv2d, ConvTranspose2d, and
Linear modules. It is 86,017,810,306 MACs, 82,422,346 weight parameters, and
248 operator sites per preparation forward. Custom Attention QK/AV and
propagation work is excluded.

The complete nine-unit front encoder accounts for 43.52% of ordinary MACs,
1.71% of ordinary parameters, and 7.66% of ordinary operator sites. Unit MACs
are:

| Unit | MACs | Parameters | Operators |
| --- | ---: | ---: | ---: |
| Stem | 2,654,926,848 | 38,304 | 3 |
| Embed1.0 | 5,110,235,136 | 73,728 | 2 |
| Embed1.1 | 5,110,235,136 | 73,728 | 2 |
| Embed1.2 | 5,110,235,136 | 73,728 | 2 |
| Embed2.0 | 3,974,627,328 | 229,376 | 3 |
| Embed2.1 | 5,110,235,136 | 294,912 | 2 |
| Embed2.2 | 5,110,235,136 | 294,912 | 2 |
| Embed2.3 | 5,110,235,136 | 294,912 | 2 |
| PatchEmbed1 | 141,950,976 | 32,768 | 1 |

## Greedy Search

Greedy selection maximizes search-set RMSE gain per incremental ordinary MAC.
When all remaining gains are negative, it continues with minimum RMSE, then
lower MAC, then official order. The selected path was:

| Step | Selected units | Search RMSE (m) |
| ---: | --- | ---: |
| 0 | W4A4 baseline | 1.1109 |
| 1 | E11 | 0.8954 |
| 2 | E11+E12 | 0.8498 |
| 3 | E11+E12+E20 | 0.8351 |
| 4 | E11+E12+E20+E21 | 0.8289 |
| 5 | E11+E12+E20+E21+P1 | 0.8266 |
| 6 | E11+E12+E20+E21+E22+P1 | 0.8251 |
| 7 | E11+E12+E20+E21+E22+E23+P1 | 0.8027 |
| 8 | E10+E11+E12+E20+E21+E22+E23+P1 | 0.8303 |
| 9 | Stem+E10+E11+E12+E20+E21+E22+E23+P1 | 1.3614 |

Stem and Embed1.0 are negative late additions. This behavior is also visible
in the official-prefix controls and shows that precision promotion is not
monotonic or additive.

## Final Results

W8A8 shares are percentages of the ordinary whole-model denominator. FP32 and
W4A8 are comparison-only rows and therefore do not receive front-unit W8A8
shares.

| Configuration | Mean RMSE | Median | P95 | MAC % | Param % | Op % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 0.1198 | 0.1046 | 0.2518 | - | - | - |
| W4A4 | 1.4201 | 1.4007 | 2.2727 | 0.00 | 0.00 | 0.00 |
| W4A8 | 0.6187 | 0.6316 | 1.1136 | - | - | - |
| E11 | 1.1911 | 1.1550 | 2.0261 | 5.94 | 0.09 | 0.81 |
| E11+E12 | 1.0922 | 1.0484 | 1.8286 | 11.88 | 0.18 | 1.61 |
| E11+E12+E20 | 1.0925 | 1.0442 | 1.8267 | 16.50 | 0.46 | 2.82 |
| E11+E12+E20+E21 | 1.0722 | 1.0126 | 1.7920 | 22.44 | 0.82 | 3.63 |
| E11+E12+E20+E21+P1 | 1.0829 | 1.0080 | 1.8421 | 22.61 | 0.85 | 4.03 |
| E11+E12+E20+E21+E22+P1 | 1.0817 | 1.0266 | 1.8519 | 28.55 | 1.21 | 4.84 |
| E11+E12+E20+E21+E22+E23+P1 | 1.0630 | 0.9802 | 1.8391 | 34.49 | 1.57 | 5.65 |
| E10+E11+E12+E20+E21+E22+E23+P1 | 1.1220 | 1.0499 | 1.8591 | 40.43 | 1.66 | 6.45 |
| All nine front units | 1.3902 | 1.4004 | 1.6663 | 43.52 | 1.71 | 7.66 |
| Stem | 1.5594 | 1.5608 | 2.2894 | 3.09 | 0.05 | 1.21 |
| Stem+E10 | 1.2253 | 1.1908 | 1.5643 | 9.03 | 0.14 | 2.02 |
| Stem+E10+E11 | 1.1069 | 1.1308 | 1.4483 | 14.97 | 0.23 | 2.82 |
| Stem+E10+E11+E12 | 1.1677 | 1.2189 | 1.4788 | 20.91 | 0.31 | 3.63 |
| Stem+E10+E11+E12+E20 | 1.3680 | 1.3917 | 1.5603 | 25.53 | 0.59 | 4.84 |
| Stem+E10+E11+E12+E20+E21 | 1.3625 | 1.3997 | 1.5947 | 31.47 | 0.95 | 5.65 |
| Stem+E10+E11+E12+E20+E21+E22 | 1.3392 | 1.3717 | 1.5770 | 37.41 | 1.31 | 6.45 |
| Stem+E10+E11+E12+E20+E21+E22+E23 | 1.3129 | 1.3142 | 1.5155 | 43.35 | 1.67 | 7.26 |

## Pareto Interpretation

The five non-dominated points are W4A4, E11, E11+E12,
E11+E12+E20+E21, and E11+E12+E20+E21+E22+E23+P1.

- Lowest-cost improvement: E11, 5.94% MAC and 1.1911 m. It reduces W4A4
  RMSE by 0.2289 m or 16.12%.
- Knee: E11+E12, 11.88% MAC and 1.0922 m. It reduces W4A4 RMSE by 0.3279 m
  or 23.09%.
- Lowest-RMSE front set: E11+E12+E20+E21+E22+E23+P1, 34.49% MAC and
  1.0630 m. It reduces W4A4 RMSE by 0.3571 m or 25.15%.

Moving from the knee to the lowest-RMSE set adds 22.61 percentage points of
ordinary W8A8 MAC share but improves RMSE by only 0.0292 m. The knee is the
more efficient allocation under this search contract.

Selective front W8A8 materially improves W4A4, but it does not preserve the
accuracy of W4A8 or FP32. The knee remains 0.4735 m above W4A8 and 0.9724 m
above FP32. The best searched front set remains 0.4443 m above W4A8. The
remaining error therefore lies outside these nine front units or requires a
different activation/propagation precision policy; simply promoting more
front units is not an effective solution.

## Artifacts

Generated artifacts are under
`profile_logs/nyu_completionformer_front_encoder_w8a8_pareto_64` and remain
untracked. The analysis directory contains three RMSE-versus-W8A8-share plots
and `prediction_comparison_64.png`, which compares GT, FP32, W4A4, W4A8, the
knee, and the lowest-RMSE front set with absolute-error maps.
