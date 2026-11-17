# Four compressed SPNs on RK3588: FP16 and calibrated W8A8

Date: 2026-11-17/18
Device: Orange Pi 5 Pro, RK3588S, RKNPU2
Runtime: RKNN API 2.5.0, RKNPU driver 0.9.8

## Scope and precision identity

This experiment executes the selected NAS + structured-pruning topology for
CSPN, DySPN, NLSPN, and CompletionFormer as complete RKNN forward graphs. Each
topology is compiled twice from the same RKNN-safe ONNX graph:

- `FP16`: `float_dtype=float16`, no RKNN quantization;
- `W8A8`: RKNN post-training quantization with per-channel weights, normal
  calibration, and 128 fixed NYU training samples.

The W8A8 variants are new deployment controls. They are **not** the original
research mixed-precision policies stored in the compressed checkpoints. In
particular, RKNN does not directly execute the checkpoint's W4/W6 assignments
or BF16 propagation state. DySPN, NLSPN, and CompletionFormer still execute
lowered `GridSample` through the registered floating-point CPU custom operator;
W8A8 applies to the compiler-supported NPU subgraph. CSPN propagation is
expressed entirely with exact fixed Conv2d shifts and reductions and therefore
remains in the NPU graph.

## Frozen candidates

| Model | NAS topology | Structured candidate | Stored research policy | RK3588 controls |
|---|---|---|---|---|
| CSPN | `drop_s4`, depths 2/2/2/1 | depth-pruned | decoder-stage2 W4, decoder-stage3 A6 | FP16, W8A8 |
| DySPN | `drop_s4`, depths 3/4/6/2 | bridge62, S5 hidden62, S4 hidden50 | S4 W4A6, S3 W6, BF16 propagation | FP16, W8A8 |
| NLSPN | `drop_s4`, depths 3/4/6/2 | bridge62, S5 hidden62, S4 hidden62.5 | S5 W4, S4 A6, tail A6, BF16 propagation | FP16, W8A8 |
| CompletionFormer | `drop_pvt_s3_s4`, depths 3/4/3/4/3/2 | MLP62, S4 hidden62, S3 hidden20 | QKV W6, output W6, MLP W4, BF16 propagation | FP16, W8A8 |

## Controlled latency protocol

- Batch 1, fixed NYU validation sample 386, input size 228 x 304.
- CPU affinity cores 6--7, `OMP_NUM_THREADS=2`, CPU governor `performance`;
  NPU governor `performance`, NPU core mask `AUTO`.
- Five independent runtime processes per model and precision.
- Each process performs 30 warmups, then 3 rounds x 50 timed complete forwards.
- A timed forward is synchronous `rknn_run + rknn_outputs_get`; model loading,
  input setup, output statistics, and output release are excluded.
- The reported mean is the mean of five process means. Process CV is computed
  across those five means; P05/P95 use all 750 timed forwards.
- Perf-detail is a separate, instrumented single forward and is never
  substituted for uninstrumented complete-forward latency.

## Complete-forward results

All eight candidates satisfy the five-process evidence contract. Values below
are measured complete-forward latency, not estimates or a sum of operator
times.

| Model | Precision | Mean (ms) | Median (ms) | P05--P95 (ms) | Process CV | W8A8 speedup |
|---|---|---:|---:|---:|---:|---:|
| CSPN | FP16 | 131.817 | 132.223 | 128.866--134.378 | 0.038% | -- |
| CSPN | W8A8 | 125.165 | 125.209 | 123.034--126.852 | 0.142% | 5.05% |
| DySPN | FP16 | 628.524 | 625.599 | 617.025--648.235 | 0.586% | -- |
| DySPN | W8A8 | 409.071 | 408.375 | 400.170--421.021 | 0.481% | 34.92% |
| NLSPN | FP16 | 718.046 | 714.574 | 707.680--743.396 | 0.139% | -- |
| NLSPN | W8A8 | 530.803 | 527.339 | 520.595--555.525 | 0.097% | 26.08% |
| CompletionFormer | FP16 | 782.551 | 779.778 | 771.642--803.382 | 0.340% | -- |
| CompletionFormer | W8A8 | 590.261 | 587.705 | 580.613--611.393 | 0.168% | 24.57% |

Thus DySPN and CompletionFormer **do use INT8** in this added experiment.
More precisely, RKNN quantizes the supported NPU subgraph to W8A8 while the
floating CPU custom operator remains floating point. It would be inaccurate to
describe these two executions as end-to-end integer-only graphs.

## Propagation callback and perf-detail breakdown

The callback values are direct counters accumulated inside the controlled
complete-forward runs. CSPN has no `GridSample` callback; its propagation is
represented by fixed NPU convolutions. The other models retain the exact
floating `GridSample` custom operator.

| Model | Precision | GridSample calls/forward | CPU GridSample (ms) | Share of complete forward |
|---|---|---:|---:|---:|
| CSPN | FP16 / W8A8 | 0 | 0 | 0% |
| DySPN | FP16 | 30 | 21.816 | 3.47% |
| DySPN | W8A8 | 30 | 21.319 | 5.21% |
| NLSPN | FP16 | 170 | 123.634 | 17.22% |
| NLSPN | W8A8 | 170 | 121.752 | 22.94% |
| CompletionFormer | FP16 | 170 | 123.810 | 15.82% |
| CompletionFormer | W8A8 | 170 | 121.677 | 20.61% |

The separate RKNN perf-detail pass gives the following exclusive category
shares. It is instrumented and therefore is used only for attribution; its
operator-time sum is not substituted for the latency table above.

| Model | Precision | Dense | Attention | Sampling | Propagation | Elem./norm | Layout/memory | CPU target share |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CSPN | FP16 | 45.5% | 0.0% | 2.5% | 43.3% | 6.5% | 2.3% | 5.3% |
| CSPN | W8A8 | 27.8% | 0.0% | 1.8% | 63.0% | 5.1% | 2.4% | 25.8% |
| DySPN | FP16 | 32.0% | 0.0% | 0.9% | 52.7% | 10.9% | 3.5% | 24.3% |
| DySPN | W8A8 | 23.6% | 0.0% | 0.7% | 64.4% | 9.5% | 1.8% | 23.4% |
| NLSPN | FP16 | 30.4% | 0.0% | 0.8% | 63.5% | 2.6% | 2.7% | 45.7% |
| NLSPN | W8A8 | 20.3% | 0.0% | 0.5% | 76.0% | 2.0% | 1.1% | 54.6% |
| CompletionFormer | FP16 | 28.7% | 1.7% | 1.2% | 56.0% | 9.7% | 2.6% | 45.4% |
| CompletionFormer | W8A8 | 18.9% | 1.6% | 1.0% | 67.8% | 9.1% | 1.6% | 52.5% |

The CPU-target share includes every compiler-placed CPU operation, not only
the directly timed `GridSample` callback. Because W8A8 shortens supported NPU
dense compute but does not similarly shorten floating propagation work, the
propagation and CPU shares increase even when their absolute time changes
little. CSPN's small 5.05% gain is likewise consistent with its iterative
propagation remaining on the critical path rather than dense compute alone.

## RKNN memory query

| Model | Precision | Weights (MiB) | Internal (MiB) | DMA allocated (MiB) |
|---|---|---:|---:|---:|
| CSPN | FP16 | 38.53 | 39.76 | 83.14 |
| CSPN | W8A8 | 20.61 | 29.99 | 54.38 |
| DySPN | FP16 | 29.03 | 199.97 | 241.26 |
| DySPN | W8A8 | 15.37 | 198.82 | 222.05 |
| NLSPN | FP16 | 79.12 | 99.95 | 193.00 |
| NLSPN | W8A8 | 41.12 | 56.32 | 105.25 |
| CompletionFormer | FP16 | 134.82 | 104.18 | 258.58 |
| CompletionFormer | W8A8 | 69.34 | 54.59 | 136.57 |

## Export validation

Before RKNN compilation, every final safe ONNX graph was executed on sample
386 and compared with its reconstructed PyTorch model. All outputs are finite.

| Model | Maximum absolute error (m) | Mean absolute error (m) | RMSE (m) |
|---|---:|---:|---:|
| CSPN | 0.002659 | 0.000424 | 0.000546 |
| DySPN | 0.000987 | 0.000022 | 0.000052 |
| NLSPN | 0.001621 | 0.000085 | 0.000131 |
| CompletionFormer | 0.001014 | 0.000028 | 0.000077 |

These are graph-export/lowering errors, not NYU task RMSE. They are reported
separately from both RKNN W8A8 error and full-validation task accuracy.

## Numerical and task-accuracy interpretation

The original compressed research policies already have 654-image NYU
validation evidence. Those values establish the accuracy of the stored W4/W6
+ BF16 policies, not of the RKNN-native W8A8 controls. This deployment study
reports direct FP16-versus-W8A8 output error on the fixed measured sample and
does not relabel that one-sample comparison as full-validation accuracy.

The direct RK3588 output comparison below uses the same sample 386 for FP16
and W8A8. These errors show that the current naive full-subgraph W8A8 control
is **not accuracy-qualified**. They do not quantify full-validation task loss;
that requires a new 654-image RKNN W8A8 evaluation.

| Model | W8A8 vs FP16 MAE (m) | RMSE (m) | Max abs. error (m) | Verdict |
|---|---:|---:|---:|---|
| CSPN | 0.309387 | 0.349562 | 1.073318 | not accuracy-qualified |
| DySPN | 0.065592 | 0.120512 | 2.263672 | not accuracy-qualified |
| NLSPN | 0.197317 | 0.258302 | 3.525622 | not accuracy-qualified |
| CompletionFormer | 0.227740 | 0.309714 | 4.171875 | not accuracy-qualified |

| Stored compressed policy | Full-validation mean per-image RMSE (m) | Images |
|---|---:|---:|
| CSPN selected W4/A6 policy | 0.151304 | 654 |
| DySPN selected W4A6/W6 + BF16 propagation | 0.108364 | 654 |
| NLSPN selected W4/A6 + BF16 propagation | 0.117435 | 654 |
| CompletionFormer selected W4/W6 + BF16 propagation | 0.109091 | 654 |

## CSPN graph correction

An initial CSPN export was rejected because the modern ONNX exporter expanded
each propagation iteration into eight Pads, Concat, and a CPU Transpose, and
constructed the first decoder's zero-insertion mask through ScatterND. That
graph was an exporter artifact rather than a fair compressed-model result.

The accepted graph applies algebraically exact deployment rewrites:

1. the eight guidance shifts are materialized once with explicit Pads;
2. per-iteration state shifts become fixed one-hot 3 x 3 Conv2d;
3. eight-channel reductions become fixed all-one 1 x 1 Conv2d, allowing the
   same ConvMul/ConvAdd fusion used by the validated vanilla graph.

The first decoder mask is frozen as a constant buffer. On sample 386, the
accepted rewritten PyTorch forward is bit-identical to the official
implementation (`max = mean = 0`). The rejected CSPN timing and the
accidentally single-threaded callback batch were deleted before the accepted
controlled run.

## Evidence

Machine-readable summaries, five process records per candidate, one output
dump per candidate, artifact hashes, and perf-detail breakdowns are stored
beside this report. `metrics.json` is the cross-precision summary;
`artifact_manifest.json` freezes every graph and input identity; and each
`raw/<model>/<precision>/` directory contains all five process records,
environment records, one output dump, the aggregate summary, and the parsed
single-run perf detail together with its raw runtime trace. No rejected or
single-threaded preliminary timing is included in this evidence directory.
