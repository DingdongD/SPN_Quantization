# CSPN Stem Mixed-Precision Evaluation

## Protocol

- Model: official CSPN ResNet-18, 24 propagation steps.
- Checkpoint: `cspn_iter24/best.pt`, SHA256
  `482fb9532b27bdb0e529da14845d9a63ab546e28974d90dd1197b8f704066855`.
- Calibration: fixed stratified 128-sample NYU train subset.
- Evaluation: fixed 64-sample NYU validation subset, seed `20260812`.
- Non-stem network: static contiguous Group-8 W4A4 MinMax.
- Guidance: FP32. Propagation: A8/INT16-Q13/INT32.
- Every configuration was built from a fresh checkpoint load. All load reports
  contain no missing or unexpected keys.

The FP32 row below was recomputed from the reference predictions stored in the
`STRICT_W4A4` payloads. It is context only and was not used for calibration or
configuration selection.

## End-to-End Results

| Configuration | RMSE | Delta vs strict | Wins vs strict | MAE | AbsRel | Flat RMSE | Boundary RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 reference | 0.158092 | - | - | 0.064308 | 0.021359 | 0.107420 | 0.409986 |
| STRICT_W4A4 | 0.342178 | - | - | 0.257888 | 0.113045 | 0.316857 | 0.511444 |
| STEM_W8A8 | 0.343303 | +0.001125 | 34/64 | 0.264152 | 0.125913 | 0.319170 | 0.517037 |
| STEM_FP16 | 0.346071 | +0.003894 | 31/64 | 0.263966 | 0.127204 | 0.322372 | 0.506460 |
| STEM_BRANCH_A4 | 0.356008 | +0.013830 | 9/64 | 0.271264 | 0.120546 | 0.332212 | 0.517452 |

None of the three candidates satisfies the predefined acceptance condition of
lower aggregate RMSE and at least 33 improved samples. Keeping the stem in W8A8
or FP16 therefore does not recover end-to-end accuracy under this fixed W4A4
network contract. Branch-specific RGB/depth A4 scales also do not improve RMSE.

The inverse-depth metric requires separate interpretation. `STRICT_W4A4` and
`STEM_BRANCH_A4` produced respectively one and two zero-valued pixels among
4,435,968 predictions, so the `1e-6` denominator clamp inflated aggregate
iRMSE to 59.43 and 118.78. All predictions are finite. `STEM_W8A8` and
`STEM_FP16` contain no non-positive predictions and have iRMSE 0.0844 and
0.0896.

## Error Propagation

| Configuration | Stem output SQNR | Stem ReLU SQNR | Skip4 input SQNR | All-block SQNR | Initial-depth MSE | Propagation MSE |
|---|---:|---:|---:|---:|---:|---:|
| STRICT_W4A4 | 9.10 dB | 4.49 dB | 2.23 dB | 8.94 dB | 0.294950 | 0.091423 |
| STEM_W8A8 | 32.81 dB | 28.42 dB | 25.62 dB | 10.62 dB | 0.315091 | 0.095648 |
| STEM_FP16 | 73.66 dB | 6.26 dB | 3.19 dB | 9.97 dB | 0.325347 | 0.096089 |
| STEM_BRANCH_A4 | 14.73 dB | 5.71 dB | 2.92 dB | 9.61 dB | 0.304435 | 0.101183 |

The stem is locally sensitive to RGB/depth scale mismatch: independent A4
scales improve the combined stem-output SQNR by 5.63 dB, and W8A8 improves it
by 23.71 dB. This local recovery is not preserved by the remaining W4A4
network. In particular, initial-depth and propagation MSE are worse for every
candidate than for `STRICT_W4A4`. The result supports a later-layer/depth-head
bottleneck, not a conclusion that stem error is harmless.

The regional result is also non-monotonic. `STEM_W8A8` improves global
far-range RMSE from 0.52755 to 0.44888 and boundary RMSE from 0.60192 to
0.58941, but near-range RMSE increases from 0.37147 to 0.44962. These opposing
changes explain why better feature SQNR does not imply better aggregate depth
RMSE.

## Precision Cost

The official `conv1_1` stem accounts for 1.4511% of measured convolution MACs,
0.0532% of convolution weight elements, and 1.6856% of measured convolution
input elements. `STEM_W8A8` and `STEM_FP16` promote exactly this operation;
`STRICT_W4A4` and `STEM_BRANCH_A4` remain logically 100% W4A4.

Because a 1.45% MAC promotion does not improve primary RMSE, retaining the
whole stem at higher precision is not justified by this experiment. The next
precision search should target the downstream initial-depth head and the
specific encoder/decoder boundaries that feed it, while retaining the strict
same-sample protocol.

## Artifact Audit

- 256 unique sample rows: four configurations by 64 identities.
- 256 finite prediction payloads: 64 per configuration.
- 267 manifest hashes recomputed and matched.
- Full repository verification: `877 passed, 3 subtests passed`.
