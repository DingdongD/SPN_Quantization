# CSPN Scale-Aware Group-8 Results

## Protocol

The experiment uses the official CSPN ResNet-18 architecture, converged
`cspn_iter24/best.pt` checkpoint, 128 fixed real NYU calibration samples, and
the existing fixed 64-sample NYU evaluation set.

Both configurations use per-output-channel symmetric W4 weights, MinMax A4
activations, Group-8 where channel dimensions are divisible, FP32 bias and
guidance, and A8/INT16-Q13/INT32 propagation. The only experimental change is
consumer-local RMS-ranked grouping at 33 Conv2d input sites. Conv input weights
are permuted before W4 using the paired inverse algebraic transform.

## End-to-end results

| Configuration | RMSE (m) | MAE (m) | AbsRel | iRMSE | Flat RMSE | Boundary RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Contiguous Group-8 MinMax | **0.313318** | **0.228845** | **0.097618** | **0.072664** | **0.281930** | **0.501738** |
| Scale-aware Group-8 | 0.316218 | 0.230545 | 0.098454 | 0.072826 | 0.284760 | 0.503640 |
| Relative change | +0.926% | +0.743% | +0.857% | +0.222% | +1.004% | +0.379% |

Scale-aware grouping does not meet the success criterion. It improves 21 of 64
samples and worsens 43. The mean RMSE delta is +0.00290 m, median is +0.00413 m,
P90 is +0.01391 m, maximum regression is +0.03358 m, and maximum gain is
-0.04133 m.

The contiguous result exactly reproduces the preceding static-calibration
MinMax baseline, confirming that the experiment did not change the reference
configuration.

## Grouping objective

The implementation groups 6208 channels across 33 eligible consumer inputs.
All 33 sites have lower summed RMS dispersion after sorting. The median site
dispersion reduction is 53.1%, with P10 21.3% and P90 67.6%. Six exact-zero RMS
channels occur at `layer1.0.conv1`; they make absolute epsilon-based dispersion
means very large, so median reduction is the meaningful aggregate statistic.

The grouping objective is therefore working as defined. Its failure is not due
to an incorrect sort or incomplete permutation.

## Core error source

The current graph already quantizes producer output/ReLU edges. At all 33
eligible contiguous consumer inputs, the second input QDQ is exactly idempotent:

| Consumer-input aggregate | Contiguous | Scale-aware |
|---|---:|---:|
| Added error energy | **0** | 1,616,111 |
| SQNR | infinity | 18.03 dB |
| New-zero rate | 0% | 0.693% |

The producer and contiguous consumer use an aligned quantized domain, so the
consumer input QDQ adds no error. RMS regrouping changes the consumer scale
assignment. The activation must then be requantized from the producer's code
domain into the new consumer-local Group-8 domain, injecting error even though
the new groups have much lower RMS dispersion.

The largest added consumer-input error energy occurs at:

| Input site | Added error energy |
|---|---:|
| `gud_up_proj_layer4.conv2` | 254,336 |
| `gud_up_proj_layer3.conv1_1` | 222,875 |
| `layer1.1.conv1` | 193,474 |
| `layer2.0.conv1` | 180,258 |
| `layer2.0.downsample.0` | 180,258 |

Across all 71 activation sites, aggregate SQNR drops from 13.74 dB to 12.85 dB
and activation error energy increases from 9,760,852 to 11,381,131. Almost the
entire increase is the newly non-idempotent consumer-input requantization.

## Weight and propagation checks

The paired weight permutation is correct. All 33 W4 module rows are present;
maximum W4 MSE delta is below `7e-21` and maximum error-energy delta is below
`2e-15`. Weight quantization is not the regression source.

The feature error reaches the depth head:

| Block | Contiguous MSE | Scale-aware MSE | Relative change |
|---|---:|---:|---:|
| Encoder layer 1 | 0.09338 | 0.10171 | +8.9% |
| Decoder layer 2 | 0.17144 | 0.19875 | +15.9% |
| Decoder layer 3 | 0.16141 | 0.19263 | +19.3% |
| Initial depth | 0.22695 | 0.25151 | +10.8% |
| Final propagation output | 0.07115 | 0.07283 | +2.36% |

Propagation arithmetic remains valid: anchor MAE, coefficient-sum error, and
contraction-violation rate are all zero, and every output is finite. Per-step
state quantization MSE is slightly lower under scale-aware grouping. CSPN
propagation therefore damps part of the injected feature/depth error rather
than creating the regression.

Activation clipping/rounding/zero-collapse energy closes to total error with
maximum relative discrepancy `6.25e-16`. Both configurations contain exactly
64 finite prediction payloads with identical sample identities.

## Conclusion

Global MinMax remains the best tested CSPN Group-8 policy. Consumer-local
RMS-ranked grouping reduces the requested dispersion objective but is not a
valid precision optimization in the current edge-quantized graph because it
breaks producer-consumer scale continuity and creates an extra requantization.

A useful next version must apply each scale-aware permutation once at the
producer edge and absorb it into every consumer, or optimize grouping jointly
over all consumers with an explicit requantization-error objective. Applying a
different permutation independently at every consumer should not be expanded
to the other models.

Artifacts are under `profile_logs/nyu_cspn_scale_aware_group8`. They include
the complete grouping/permutation manifest, 128 metric rows, two sets of 64
prediction payloads, activation/block/propagation diagnostics, weight
invariance checks, and PNG/PDF figures.
