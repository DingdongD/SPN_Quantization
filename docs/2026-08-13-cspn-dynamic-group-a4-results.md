# CSPN Dynamic Group-A4 Results

## Protocol

- Official CSPN ResNet-18 architecture with 24 propagation steps.
- Converged `cspn_iter24/best.pt` checkpoint and real NYU data.
- The same random-128 calibration indices and fixed 64 validation indices as
  the preceding static Group-8 experiment.
- W4 uses static signed symmetric per-output-channel weight scales.
- Static A4 uses calibrated MinMax Group-8 ranges.
- Dynamic A4 computes one range per sample and contiguous group of eight
  channels. ReLU outputs use unsigned `[0, 15]`; signed activations use
  symmetric `[-7, 7]`.
- Guidance remains FP. Propagation remains A8 with signed INT16 Q13
  coefficients and INT32 accumulation.
- SmoothQuant, rotation, OCI, BRECQ, QDrop, and activation permutation are not
  enabled.

Dynamic and static configurations quantize the same ordinary activation owners.
The two structural rotation boundaries retain their existing static ranges.

## End-to-end Results

| Configuration | RMSE (m) | MAE (m) | AbsRel | iRMSE | Flat RMSE (m) | Boundary RMSE (m) |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 0.166932 | 0.067154 | 0.022361 | 0.022990 | 0.115248 | 0.416973 |
| W4 only | 0.203795 | 0.117773 | 0.042022 | 0.032471 | 0.158178 | 0.446227 |
| A4 only, static Group-8 | 0.318635 | 0.237785 | 0.109981 | 0.080214 | 0.290062 | 0.493003 |
| A4 only, dynamic Group-8 | 0.284476 | 0.207811 | 0.085020 | 0.066065 | 0.253709 | 0.469314 |
| W4A4, static Group-8 | 0.313318 | 0.228845 | 0.097618 | 0.072664 | 0.281930 | 0.501738 |
| W4A4, dynamic Group-8 | 0.303394 | 0.223505 | 0.090444 | 0.077549 | 0.271832 | 0.492809 |

Dynamic Group-8 improves A4-only RMSE by 0.034160 m, or 10.72%. It is better
on 50 of 64 samples. In full W4A4 it improves RMSE by 0.009923 m, or 3.17%,
and is better on 43 of 64 samples. It reduces the static W4A4-to-FP32 RMSE gap
by 6.78%.

The improvement is not uniform. Compared with static W4A4, pixel-aggregated
regional RMSE changes are:

- near 0-2 m: -17.35%;
- sparse anchors: -7.54%;
- smooth regions: -3.58%;
- holes: -2.77%;
- far 5-10 m: -1.34%;
- boundaries: -0.86%;
- mid 2-5 m: +1.51%.

## Activation Error

| Configuration | Activation SQNR (dB) | New-zero rate | Clipping error share | Zero-collapse error share |
|---|---:|---:|---:|---:|
| A4 only, static Group-8 | 13.96 | 32.41% | 0.019% | 47.26% |
| A4 only, dynamic Group-8 | 16.26 | 24.01% | 0.0003% | 44.24% |
| W4A4, static Group-8 | 13.74 | 31.71% | 0.0078% | 47.40% |
| W4A4, dynamic Group-8 | 16.01 | 23.07% | 0.0004% | 44.77% |

The result supports the static-calibration diagnosis. Per-sample ranges remove
cross-sample tail inflation, reduce the W4A4 new-zero rate by 8.64 percentage
points, and improve aggregate activation SQNR by 2.27 dB. Clipping was already
small and is not the main gain.

The recorded `saturation_rate` is endpoint-code occupancy: it counts values
whose code equals `qmin` or `qmax`. It is not the clipping rate. Dynamic MinMax
naturally maps each group maximum to an endpoint, while the independently
computed clipping-error share remains approximately zero.

## Block Propagation

Dynamic W4A4 improves output SQNR throughout the ordinary network:

| Block | Static (dB) | Dynamic (dB) |
|---|---:|---:|
| Encoder stem | 9.47 | 11.81 |
| Encoder layer 1 | 6.41 | 9.78 |
| Encoder layer 2 | 1.92 | 5.05 |
| Encoder layer 3 | -2.49 | 2.10 |
| Encoder layer 4 | -0.38 | 1.50 |
| Decoder layer 1 | 1.41 | 4.25 |
| Decoder layer 2 | 3.75 | 6.53 |
| Decoder layer 3 | 5.05 | 7.95 |
| Decoder layer 4 | 5.95 | 8.57 |
| Initial depth | 16.35 | 18.46 |
| Propagation output | 21.53 | 22.14 |

The propagation output gain is smaller than the encoder/decoder gains. The
remaining W4 weight error and SPN sensitivity therefore limit how much local
activation improvement reaches final depth.

## Runtime Scale Cost

The ordinary dynamic path contains 69 activation sites:

| Group | Sites | Scales per 64 samples | Reduced elements per 64 samples |
|---|---:|---:|---:|
| Encoder | 41 | 73,280 | 426,721,280 |
| Decoder | 27 | 28,672 | 809,992,192 |
| Depth head | 1 | 512 | 283,901,952 |
| Total | 69 | 102,464 | 1,520,615,424 |

This corresponds to 1,601 ordinary online scales and 23,759,616 reduction
elements per sample. Two structural boundaries contribute another 72 static
scales per sample. These counts describe range-computation work; no optimized
kernel latency claim is made by this reference QDQ experiment.

## Conclusion

Per-sample Dynamic Group-8 is a valid CSPN W4A4 accuracy improvement and is
more effective than the tested global SmoothQuant variants. The best measured
W4A4 RMSE improves from 0.313318 m to 0.303394 m. It should remain an optional
accuracy-oriented policy until fused online reduction and quantization kernels
measure its latency and bandwidth cost.

Dynamic activation range selection does not solve the full low-bit problem.
The 0.303394 m result remains substantially worse than FP32 at 0.166932 m, and
21 of 64 samples regress. The next optimization should be selective dynamic
allocation at the encoder/decoder sites with the largest block-error benefit,
so most of the gain can be retained with fewer online reductions.

Artifacts are under
`profile_logs/nyu_cspn_dynamic_group_a4` and include all metric CSVs, dynamic
overhead rows, metadata, and prediction payloads for FP32 and both static and
dynamic A4-only/W4A4 configurations.
