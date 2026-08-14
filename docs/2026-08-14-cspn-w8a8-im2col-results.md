# CSPN W8A8 Im2Col Distribution Results

## Protocol

- Model: official CSPN ResNet-18 with 24 propagation steps.
- Checkpoint: `nyu_converged_baselines/cspn_iter24/best.pt`, SHA256
  `482fb9532b27bdb0e529da14845d9a63ab546e28974d90dd1197b8f704066855`.
- Data: the declared 128-sample stratified NYU train calibration set and fixed
  64-sample NYU validation set, seed `20260812`.
- Quantization: the existing propagation-aware W8A8 deployment contract.
  Ordinary Conv2d uses per-output-channel symmetric W8 and calibrated A8;
  CSPN propagation retains A8 signals, INT16 Q13 coefficients and INT32
  accumulation.
- Scope: no QAT, rotation, SmoothQuant, activation grouping or FP4.

The recorder collected all 37 Conv2d modules executed by the official forward
path. Thirteen legacy Conv2d modules retained in the model object but not
executed by this architecture are recorded as `excluded_unobserved_conv2d`.

## End-to-End Accuracy

| Configuration | RMSE | MAE | AbsRel |
| --- | ---: | ---: | ---: |
| FP32 | 0.159796 | 0.064791 | 0.021654 |
| PA-W8A8 | 0.170966 | 0.084632 | 0.030678 |
| Delta | +0.011170 (+6.99%) | +0.019841 (+30.62%) | +0.009024 (+41.67%) |

No evaluated sample contained NaN or Inf output pixels. Diagnostic attachment
was checked against an unrecorded W8A8 forward and did not change prediction
tensors.

## K-Axis Findings

The activation RMS variation is predominantly channel-wise rather than
kernel-offset-wise. For the most relevant layers, the coefficient of variation
of channel means divided by that of offset means is:

| Module | Channel CV | Offset CV | Channel/offset ratio |
| --- | ---: | ---: | ---: |
| `conv1_1` | 0.2544 | 0.0036 | 70.2 |
| `gud_up_proj_layer4.conv1_1` | 0.4241 | 0.0017 | 246.5 |
| `gud_up_proj_layer3.conv1_1` | 0.2838 | 0.0058 | 48.9 |
| `gud_up_proj_layer2.conv1_1` | 0.3222 | 0.0145 | 22.2 |
| `layer4.1.conv2` | 0.2841 | 0.0562 | 5.1 |

Kernel offsets become more relevant in deep 3x3 encoder convolutions, but
channel variation remains larger. In decoder fusion, the same bad channel is
consistently bad across all nine offsets. This argues for channel- or
branch-aware activation scaling before kernel-offset-specific treatment.

### RGBD Stem

`conv1_1` shares one A8 range across three normalized RGB channels and one
sparse-depth channel. The RGB channels have maximum magnitude 1.0 and RMS
0.539-0.567, while sparse depth reaches 9.904 with RMS 0.269. The shared
unsigned A8 step is therefore controlled by sparse depth. RGB A8 SQNR is only
34.0-34.4 dB and 1.30-3.11% of nonzero RGB values become zero; sparse depth
has 48.62 dB SQNR and no new zeros. The stem has the lowest local Conv output
SQNR of all collected layers, 32.81 dB.

This is a concrete mixed-input scale problem. A4 will enlarge the same step by
17 times relative to unsigned A8, so RGB and sparse depth should not share one
activation scale in a low-bit deployment.

### Decoder Fusion

Only four Conv inputs introduce nonzero incremental A8 error:
`conv1_1` and `gud_up_proj_layer{2,3,4}.conv1_1`. These are the external input
and decoder fusion boundaries. Their shares of total incremental A8 error
energy are 15.97%, 1.94%, 9.52% and 72.57%, respectively. The last fusion is
the dominant total-energy site because of both tensor size and scale mismatch;
its worst cells are channels 18 and 41 across every kernel offset, with about
27 dB A8 SQNR.

The total A8 error-energy decomposition is 90.73% ordinary rounding, 9.27%
new-zero collapse and 0.006% clipping. At W8A8, the measured issue is therefore
range resolution and branch/channel scale mismatch, not widespread clipping
by extreme outliers.

## Local Conv and Spatial Findings

The lowest local pre-output-QDQ Conv SQNR layers are:

| Module | Group | Local SQNR | Token error p99 |
| --- | --- | ---: | ---: |
| `conv1_1` | encoder | 32.81 dB | 0.004267 |
| `gud_up_proj_layer4.conv1_1` | decoder | 40.62 dB | 0.005022 |
| `layer4.0.downsample.0` | encoder | 41.16 dB | 0.004587 |
| `gud_up_proj_layer2.conv1_1` | decoder | 41.47 dB | 0.009858 |
| `layer2.1.conv1` | encoder | 41.65 dB | 0.002272 |
| `layer3.0.conv2` | encoder | 41.69 dB | 0.004699 |

`layer4.1.conv2` has the largest absolute token error p99, 0.1233, despite
42.51 dB aggregate local SQNR. Its local input A8 error is exactly zero and its
W8 weight SQNR is 36.99 dB. Its token error correlates strongly with patch RMS
(`r=0.971`), so this site is a high-energy W8 weight-error hotspot rather than
an incremental activation-rounding hotspot.

Across most encoder layers, local output error correlates strongly with patch
RMS (`r=0.70-0.97`). At `gud_up_proj_layer4.conv1_1`, local error instead
correlates more with A8 input error (`r=0.585`) than patch magnitude
(`r=-0.163`), matching the channel-scale mismatch finding. A 10%-width border
ring is not generally worse than the interior; only selected downsample sites
show a moderate border increase. The largest absolute errors are therefore
spatially concentrated high-energy patches or fusion-scale failures, not a
universal image-border artifact.

## Complete Unfolded Matrix Views

The follow-up capture renders the raw matrices requested for Conv-as-linear
inspection rather than channel/offset summary statistics. Each selected module
uses its own highest-local-error validation sample and produces:

- weight `Cout x (Cin Kh Kw)` FP32, W8-QDQ and absolute-error panels;
- activation `M x (Cin Kh Kw)` pre-QDQ, A8-QDQ and absolute-error panels.

No channels, offsets, output channels or spatial tokens are sampled. The 12
weight/activation matrix identities contain 29,093,312 elements; their three
FP/QDQ/error panels render 87,279,936 matrix points. All manifest rows satisfy
`sampling=none` and `rendered_elements == matrix_elements`.

The complete matrices make three structures visible:

1. The stem weight matrix has its strongest K ridge at sparse-depth channel 3,
   center offset `(3,3)`. The activation maximum is 9.882, while most RGB K
   rows remain below the sparse-depth spikes. A8 changes 74.05% of unfolded
   entries numerically, but the resulting activation SQNR remains 34.77 dB;
   2.35% of nonzero entries become zero. The error is spatially concentrated:
   token RMS imbalance is 2.69, compared with K-error imbalance 1.28.
2. `gud_up_proj_layer4.conv1_1` has the largest matrix,
   `17328 x 1152`. Its pre-QDQ and A8 matrices remain visually aligned, but
   rounding error is dense rather than isolated: 23.52% of entries change,
   A8 SQNR is 31.88 dB and error p99/max are 0.02903/0.03002. Error K
   imbalance is 2.36, led by channel 41 at the center kernel offset. This is
   the matrix-level evidence for a decoder-fusion channel-scale problem.
3. The selected ordinary encoder Conv inputs are already exactly on their
   local A8 grids, so their activation error panels are explicitly marked
   `all zero`. Their FP/QDQ activation matrices are identical at the current
   site, while W8 weight SQNR remains 36.04-39.69 dB. The deep encoder
   activation matrices still show channel@offset ridges: K-RMS imbalance is
   2.99 for `layer2.1.conv1` and 3.72 for `layer3.0.conv2`.

The figures confirm that full unfolded distributions are not well described
by one global outlier scalar. The stem combines a sparse high-amplitude depth
ridge with dense lower-amplitude RGB values; decoder fusion shows dense
rounding over selected K rows; deep encoder Conv error is currently dominated
by W8 weights because no additional A8 rounding occurs at those inputs.

## SQNR Semantics

Of 93,572 channel-offset cells, 91,072 have mathematically infinite
incremental activation SQNR because the current Conv input is already exactly
representable on that site's A8 grid. This does not mean that upstream W8A8
error is absent. The CSV retains `+inf`; figures map it to a labelled finite
display ceiling only so the 3D lines remain visible.

The local metric isolates the additional QDQ and W8 Conv error at each site.
It is not endpoint-gradient attribution and must not be interpreted as the
total contribution of that layer to final RMSE.

## Artifacts and Validation

Results are under
`profile_logs/nyu_cspn_w8a8_im2col_64`. The main tables are
`channel_offset_metrics.csv`, `channel_metrics.csv`,
`kernel_offset_metrics.csv`, `layer_metrics.csv` and
`top_spatial_tokens.csv`. K-axis and spatial figures are under `figures`.
Complete matrix captures and triptychs are under
`full_matrix_visualization/captures` and
`full_matrix_visualization/figures`; their identities are recorded in
`capture_manifest.csv` and `figure_manifest.csv`.

Strict validation passed for:

- exact 128/64 calibration and evaluation identities;
- all 37 active Conv2d modules and 13 explicit unobserved legacy modules;
- 2,368 module-sample spatial arrays and 24 PNG/PDF figure pairs;
- finite values except valid positive-infinite zero-error SQNR;
- exact channel and offset reductions back to layer counters and energies;
- nonblank inspected K-axis and spatial figures;
- direct Conv2d versus Im2Col matrix equivalence in unit tests.
- six complete native FP32 captures, 12 no-sampling triptychs and exact
  source-artifact digest preservation.

The W8A8 evidence prioritizes separate RGB/depth stem scales and separate
decoder-fusion branch scales before more elaborate kernel-offset treatment.
The next W4A4 experiment should preserve this same structured diagnostic so
that zero collapse and endpoint degradation can be compared at identical
sites.
