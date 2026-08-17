# CSPN Encoder-Prefix and Sensitive-Tail W8A8 Results

## Evaluation Contract

The experiment used the official CSPN ResNet-18 architecture with 24
propagation iterations and checkpoint
`output/nyu_converged_baselines/cspn_iter24/best.pt`. Each of the 24
configurations was reconstructed from the same checkpoint, calibrated on the
persisted stratified 128-sample NYU train set, and evaluated on the same fixed
64 NYU validation samples. This was post-training quantization; no checkpoint
training or fine-tuning was performed.

Ordinary CNN units used static contiguous Group-8 MinMax W4A4 unless promoted
as a complete W8A8 unit. Guidance remained FP32. CSPN propagation remained
A8 with INT16 Q13 coefficients and INT32 accumulation. Bias remained FP32 and
Conv-BN folding was reproduced before calibration.

The immutable measured output is stored at
`profile_logs/nyu_cspn_encoder_prefix_joint_w8a8_64`. The audit verified 24
aggregate rows, 1,536 sample rows, 64 unique samples per configuration, 37
executed Conv records per configuration, 704 prediction payloads for 11
selected configurations, and all 717 manifest artifact hashes.

## RMSE Matrix

All values are metres. Prefixes are cumulative: P1 is stem, P2 adds layer1,
P3 adds layer2, P4 adds layer3, and P5 adds layer4. T1 retains decoder4 in
W8A8, T2 retains the initial-depth head, and T3 retains both.

| Prefix | T0 None | T1 Decoder4 | T2 Depth | T3 Both |
|---|---:|---:|---:|---:|
| P0 None | 0.342178 | 0.284687 | 0.335302 | 0.263172 |
| P1 Stem | 0.343303 | 0.217950 | 0.341726 | 0.194379 |
| P2 Stem+L1 | 0.292750 | 0.207162 | 0.291339 | 0.176864 |
| P3 Stem+L1+L2 | 0.273342 | 0.204275 | 0.272149 | 0.172321 |
| P4 Stem+L1+L2+L3 | 0.274846 | 0.203947 | 0.273670 | **0.172269** |
| P5 Stem+L1+L2+L3+L4 | 0.275057 | 0.203788 | 0.273665 | 0.172296 |

The strict W4A4 reference is P0/T0 at 0.342178 m. The lowest measured RMSE is
P4/T3 at 0.172269 m, a reduction of 0.169909 m. P5/T3 does not improve it.

## Precision-Cost Tradeoff

| Configuration | W8 Conv count | W8 MAC | A8 activation elements | Added bit-element cost | RMSE |
|---|---:|---:|---:|---:|---:|
| P0/T0 | 0/37 | 0.00% | 0.00% | 0.00% | 0.342178 |
| P1/T3 | 6/37 | 38.20% | 58.99% | 30.98% | 0.194379 |
| P2/T3 | 10/37 | 42.47% | 67.89% | 35.85% | 0.176864 |
| P3/T3 | 15/37 | 46.33% | 74.65% | 40.40% | 0.172321 |
| P4/T3 | 20/37 | 50.32% | 78.13% | 46.51% | 0.172269 |
| P5/T3 | 25/37 | 54.80% | 80.03% | 64.79% | 0.172296 |

P2/T3 is the practical knee: it recovers 97.3% of the RMSE reduction achieved
by P4/T3 while using 10 instead of 20 W8 Conv operators and 10.66 percentage
points less normalized added cost. P3/T3 reduces RMSE by another 0.004542 m
over P2/T3 and wins on 49 of 64 paired samples. A fixed-seed paired bootstrap
of sample RMSE differences gives a 95% interval of [-0.006318, -0.002780] m,
so the layer2 promotion has a stable effect.

P4/T3 improves aggregate RMSE by only 0.000053 m over P3/T3. Its paired mean
sample-RMSE difference has a 95% bootstrap interval of
[-0.000800, 0.000696] m. P5/T3 versus P4/T3 is similarly unresolved, with an
interval of [-0.000884, 0.000989] m. Therefore P3/T3 is the smallest measured
configuration statistically indistinguishable from the raw minimum, while
P4/T3 remains the strict numerical minimum.

The normalized bit-element cost and W8 MAC fraction are logical precision
coverage metrics. They are not measured latency or energy.

## Error Flow

Stem W8A8 alone does not improve end-to-end RMSE: P1/T0 is 0.001125 m worse
than strict W4A4 even though stem SQNR rises from 9.10 dB to 32.81 dB. The
recovered stem signal is lost when layer1 remains W4A4. Promoting layer1 as
part of P2 reduces encoder-layer1 MSE from 0.097956 to 0.000379 and lowers
P2/T0 RMSE to 0.292750 m. Adding layer2 in P3 lowers encoder-layer2 MSE from
0.017274 to 0.000160 and lowers P3/T0 RMSE to 0.273342 m.

The tail is activation-sensitive. T1 alone lowers RMSE by 0.057490 m, while
T2 alone lowers it by only 0.006876 m. T3 lowers it by 0.079006 m. With P3/T3,
decoder4 output SQNR reaches 15.67 dB, initial-depth SQNR reaches 25.55 dB,
and propagation output SQNR reaches 32.96 dB. This confirms that retaining the
decoder4 and depth-head boundaries prevents early encoder recovery from being
destroyed before propagation.

The strongest interaction is P1/T3 at -0.069918 m, followed by P1/T1 at
-0.067863 m. Negative interaction means the joint promotion is substantially
better than the sum of its isolated effects. T2 has a small positive
interaction of approximately +0.0053 to +0.0057 m with every nonzero prefix,
so the initial-depth promotion is useful primarily together with decoder4,
not as an isolated encoder companion.

## Propagation and Inverse-Depth Checks

Across the inspected Pareto configurations, propagation recorded zero
non-finite values, zero contraction violations, zero coefficient-sum error,
zero anchor error, and INT32 accumulators. Maximum observed propagation
saturation was 0.0073% in strict W4A4 and at most 0.0015% in the selected T3
configurations.

Strict P0/T0 has iRMSE 59.429 because one of 4,435,968 valid prediction pixels
is exactly zero; finite-value checks alone do not expose this inverse-depth
failure. P1/T3 through P5/T3 have no non-positive or sub-0.1 m prediction
pixels. P3/T3 has RMSE 0.172321 m, MAE 0.086617 m, AbsRel 0.031247, iRMSE
0.027141, flat-region RMSE 0.124567 m, and boundary RMSE 0.422040 m.

## Selection

- Use P2/T3 when minimizing W8 coverage is the main objective. It contains
  stem, layer1, decoder4, and initial-depth W8A8 units.
- Use P3/T3 when accuracy is primary but layer3/layer4 W8 cost is not
  justified. It adds layer2 and is statistically indistinguishable from the
  raw minimum on this fixed 64-sample set.
- P4/T3 is the strict lowest-RMSE point, but its 0.000053 m gain over P3/T3 is
  not supported as a stable paired-sample improvement.
- Do not use stem-only P1/T0 or isolated depth-head T2 as the final mixed
  precision policy.

The generated RMSE and interaction heatmaps plus both Pareto views are stored
beside the measured CSVs. They use Arial, omit titles, keep tick labels
horizontal, and place grid lines behind the Pareto data.
