# Four vanilla SPN models: audited cross-platform forward latency

## Status and scope

This record contains measured—not estimated—batch-1 complete-forward latency
and hardware-oriented breakdowns for CSPN, DySPN, NLSPN, and
CompletionFormer. The fixed workload is NYU Depth V2 sample 386 at 228×304.
The timed boundary starts after preprocessing with inputs resident at the
execution interface and includes every model layer and all propagation
iterations. Dataset decode, preprocessing, host-to-device preparation, model
construction, checkpoint loading, and disk I/O are excluded.

The detected second NVIDIA device is an **NVIDIA GeForce RTX 3080 Ti**, not an
RTX 3090. NVIDIA measurements use the strict FP32 vanilla checkpoints. RK3588
uses RKNN FP16 graphs; host inputs and returned outputs are FP32. Cross-device
latencies therefore describe the deployed configurations and are not a
same-precision GPU-versus-NPU speedup claim.

The exact vanilla configurations are CSPN ResNet-18/k=24, DySPN
ResNet-34/k=6, NLSPN ResNet-34/k=18, and CompletionFormer-large/k=18. “Vanilla”
means the unchanged learned checkpoints and full forward semantics; RKNN-only
operator admission rewrites and the disclosed CompletionFormer stabilization
do not retrain or alter task parameters.

## Platforms and controls

| Platform | Measured execution device | Precision | Frequency/control state | Software |
|---|---|---|---|---|
| Jetson AGX Xavier 32 GB | 512-core Volta GPU with Tensor Cores | FP32 | MAXN; GPU fixed at 1377 MHz; benchmark pinned to CPU7 fixed at 2265.6 MHz | JetPack 5.1.2 / L4T R35.4.1, CUDA 11.4, PyTorch 2.1 NVIDIA build |
| AutoDL host | GeForce RTX 3080 Ti, 12 GiB, compute capability 8.6 | FP32 | observed P2, SM 1965 MHz and memory 9251 MHz in controlled files; 350 W limit | driver 595.71.05, CUDA 12.8, cuDNN 91002, PyTorch 2.8.0+cu128 |
| Orange Pi 5 Pro | RK3588S three-core NPU plus explicit CPU custom operators | FP16 internal | NPU performance governor at 1.0 GHz; DMC performance governor at 2.4 GHz; process pinned to CPU6–7 at up to 2.304 GHz; OMP=2 | Linux 6.1.43, RKNPU driver 0.9.8, RKNN API 2.5.0 |

For orientation only, vendor nominal peaks are approximately 32 TOPS for the
Jetson AGX Xavier, 34.1 TFLOP/s FP32 for the RTX 3080 Ti, and 6 TOPS INT8 for
RK3588. Those advertised peaks were not used to derive any latency or
breakdown value below.

The Jetson run retained unrelated user processes and is explicitly a
clock/affinity-controlled result under recorded background load, not an
exclusive idle-system baseline. The RTX clocks were observed rather than
administratively locked. RK3588 NPU and DMC governors were locked to
performance.

## Complete-forward latency

Each cell is aggregated over five independent processes. Every process used 30
warmups followed by 3 rounds × 50 timed forwards, so each model/platform has
750 timed forwards. The mean, median, range, two-sided Student-t 95% confidence
interval, and CV are calculated over the five independent process means—not by
pooling 750 correlated iterations.

| Platform | Model | Mean (ms) | Median process mean (ms) | Process-mean range (ms) | 95% CI (ms) | CV |
|---|---|---:|---:|---:|---:|---:|
| Jetson AGX Xavier | CSPN | 71.662 | 71.369 | 69.974–73.286 | 70.088–73.235 | 1.77% |
| Jetson AGX Xavier | DySPN | 139.333 | 139.177 | 139.034–139.959 | 138.877–139.789 | 0.26% |
| Jetson AGX Xavier | NLSPN | 157.024 | 156.789 | 155.768–158.318 | 155.726–158.321 | 0.67% |
| Jetson AGX Xavier | CompletionFormer | 193.359 | 193.868 | 191.245–195.903 | 191.016–195.702 | 0.98% |
| RTX 3080 Ti | CSPN | 17.133 | 16.987 | 16.890–17.697 | 16.727–17.539 | 1.91% |
| RTX 3080 Ti | DySPN | 12.905 | 12.933 | 12.827–12.941 | 12.844–12.966 | 0.38% |
| RTX 3080 Ti | NLSPN | 24.593 | 24.437 | 24.121–25.117 | 24.041–25.144 | 1.80% |
| RTX 3080 Ti | CompletionFormer | 38.540 | 38.371 | 38.084–39.227 | 37.978–39.102 | 1.18% |
| RK3588 mixed NPU+CPU | CSPN | 115.878 | 115.896 | 115.457–116.304 | 115.502–116.254 | 0.26% |
| RK3588 mixed NPU+CPU | DySPN | 654.234 | 653.766 | 650.396–657.725 | 650.386–658.082 | 0.47% |
| RK3588 mixed NPU+CPU | NLSPN | 782.708 | 782.360 | 779.473–786.004 | 779.481–785.934 | 0.33% |
| RK3588 mixed NPU+CPU | CompletionFormer (stabilized) | 754.761 | 754.752 | 753.315–755.942 | 753.530–755.991 | 0.13% |

CUDA Events delimit each NVIDIA forward and synchronize its completion. RKNN
uses `CLOCK_MONOTONIC` around synchronous `rknn_run` plus
`rknn_outputs_get`; this includes NPU execution, CPU custom-op callbacks, and
runtime boundary handling, while `rknn_outputs_release` and output statistics
remain outside the timed region.

## Complete-forward energy

Energy was measured on 2026-10-23 with the same fixed sample, checkpoints,
precision, GPU-resident inputs, and complete model-forward boundary as the
latency experiment. Each model has five fresh processes. Every process used 30
warmup forwards, a 15 s loaded-idle window, and at least 60 s of individually
synchronized complete forwards. Timestamped power was requested every 50 ms
and integrated with the trapezoidal rule at the exact monotonic-clock phase
boundaries.

`Gross` is the measured active-window energy divided by the number of completed
forwards. `Incremental` subtracts the loaded-idle mean power over the active
duration before division. The 95% intervals are two-sided Student-t intervals
over the five independent process-level energy values.

| Platform / physical sensor boundary | Model | Loaded idle (W) | Active (W) | Gross (J/forward) | Incremental (J/forward, 95% CI) |
|---|---|---:|---:|---:|---:|
| RTX 3080 Ti / NVML GPU board power | CSPN | 109.627 | 191.496 | 3.321 | 1.420 (1.345–1.496) |
| RTX 3080 Ti / NVML GPU board power | DySPN | 114.605 | 344.246 | 4.501 | 3.003 (2.952–3.053) |
| RTX 3080 Ti / NVML GPU board power | NLSPN | 122.268 | 332.619 | 5.684 | 3.595 (3.577–3.612) |
| RTX 3080 Ti / NVML GPU board power | CompletionFormer | 119.005 | 248.443 | 9.259 | 4.824 (4.790–4.858) |
| Jetson AGX Xavier / INA3221 module total | CSPN | 12.106 | 28.991 | 1.586 | 0.924 (0.881–0.967) |
| Jetson AGX Xavier / INA3221 module total | DySPN | 12.206 | 41.122 | 5.611 | 3.946 (3.814–4.077) |
| Jetson AGX Xavier / INA3221 module total | NLSPN | 12.361 | 39.938 | 5.935 | 4.098 (3.862–4.335) |
| Jetson AGX Xavier / INA3221 module total | CompletionFormer | 12.308 | 35.244 | 6.368 | 4.144 (3.984–4.304) |
| RK3588 / no physical power sensor exposed | all four models | unavailable | unavailable | unavailable | unavailable |

The NVIDIA values are hardware measurements, not latency × TDP estimates.
NVML exposes the RTX board-power boundary, whereas Xavier's `total_module` is
the sum reported by jtop from its INA3221-backed module rails. Because the
physical boundaries differ, the RTX and Xavier joule values must not be read as
a perfectly like-for-like silicon-only comparison. Within each platform,
incremental energy is the primary workload comparison; gross energy additionally
includes the loaded-idle baseline.

### Xavier incremental rail attribution

The individual Xavier rails below are integrated independently over the same
phase boundaries. They sum to the module-total result up to sampling and
rounding. Small signed auxiliary-rail changes are retained rather than clipped;
only `total_module` is used as the platform energy result.

| Model | GPU | CPU | SOC | VDDRQ | SYS5V | CV | Module total (J/forward) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CSPN | 0.679 | 0.073 | 0.070 | 0.069 | 0.032 | 0.000 | 0.924 |
| DySPN | 2.794 | 0.202 | 0.383 | 0.395 | 0.172 | -0.000 | 3.946 |
| NLSPN | 3.042 | 0.065 | 0.399 | 0.412 | 0.180 | 0.000 | 4.098 |
| CompletionFormer | 3.011 | 0.158 | 0.391 | 0.404 | 0.180 | 0.000 | 4.144 |

The continuous active-window CUDA-event means are retained in the raw energy
records as a boundary check, but they do not replace the independently
controlled headline latency table above.

### Why RK3588 has no strict energy number

The Orange Pi exposes temperature-only hwmon devices, a voltage-only SARADC,
and RKNPU utilization/frequency/voltage state. It exposes no current or power
channel, no power-supply telemetry, and no INA/current-monitor device-tree
node. Voltage, NPU load, 6-TOPS marketing peak, and TDP are insufficient to
integrate joules. A timestamped external board-input power meter or shunt is
therefore required before the four RKNN runs can be reported. The missing
measurement is intentionally shown as unavailable rather than zero or an
estimate.

Raw per-process JSON, timestamped JSONL samples, sensor inventories, hashes,
and aggregate statistics are in [`energy_summary.json`](energy_summary.json),
[`autodl_nvidia/energy`](autodl_nvidia/energy/),
[`jetson_xavier/energy`](jetson_xavier/energy/), and the
[`RK3588 capability audit`](rk3588/energy/capability_audit.json).

## Six-class NVIDIA GPU-activity breakdown

Nsight Systems captured five separately instrumented full forwards per model.
Each CUDA kernel, memcpy, and memset was mapped through CUPTI correlation ID to
its launching CUDA runtime call and then to the innermost `aten::` NVTX scope.
An activity launched inside the explicitly marked propagation module is
classified as Propagation before primitive-type classification. Every activity
is assigned once. Times are direct, unscaled milliseconds per instrumented
forward; the percentage in parentheses is its share of the direct activity
sum. `Other` is an audit residual outside the requested six classes.

### Jetson AGX Xavier

| Model | Dense compute | Attention | Sampling | Propagation | Elementwise / norm | Layout / memory | Other | Activity sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CSPN | 30.991 (71.84%) | 0.000 | 0.126 (0.29%) | 9.269 (21.49%) | 2.094 (4.85%) | 0.481 (1.12%) | 0.179 (0.42%) | 43.139 |
| DySPN | 101.304 (72.82%) | 0.000 | 0.000 | 7.043 (5.06%) | 24.831 (17.85%) | 5.745 (4.13%) | 0.191 (0.14%) | 139.114 |
| NLSPN | 112.010 (78.76%) | 0.000 | 0.000 | 6.151 (4.33%) | 18.370 (12.92%) | 5.691 (4.00%) | 0.000 | 142.223 |
| CompletionFormer | 98.619 (56.53%) | 18.834 (10.80%) | 0.267 (0.15%) | 6.342 (3.64%) | 38.915 (22.31%) | 9.583 (5.49%) | 1.887 (1.08%) | 174.445 |

### RTX 3080 Ti

| Model | Dense compute | Attention | Sampling | Propagation | Elementwise / norm | Layout / memory | Other | Activity sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CSPN | 4.865 (71.32%) | 0.000 | 0.013 (0.19%) | 1.505 (22.07%) | 0.266 (3.91%) | 0.048 (0.71%) | 0.123 (1.80%) | 6.821 |
| DySPN | 7.677 (62.82%) | 0.000 | 0.000 | 0.834 (6.83%) | 3.003 (24.57%) | 0.613 (5.02%) | 0.095 (0.77%) | 12.221 |
| NLSPN | 8.376 (68.95%) | 0.000 | 0.000 | 0.873 (7.19%) | 2.285 (18.81%) | 0.614 (5.05%) | 0.000 | 12.148 |
| CompletionFormer | 10.746 (45.73%) | 5.259 (22.38%) | 0.075 (0.32%) | 0.880 (3.74%) | 4.999 (21.27%) | 1.159 (4.93%) | 0.379 (1.61%) | 23.496 |

These data answer the propagation question directly: on both NVIDIA GPUs,
dense compute is the largest measured device-activity class for all four
models. Propagation is material for CSPN (about 22%), but only about 4–7% for
DySPN, NLSPN, and CompletionFormer. CompletionFormer additionally exposes a
large attention path (10.8% on Xavier and 22.4% on the RTX) and a roughly 21–22%
elementwise/normalization path.

## RK3588 six-class operator breakdown and placement

RKNN `RKNN_QUERY_PERF_DETAIL` was collected in a separate instrumented pass.
The categories are exclusive and unscaled; logical propagation scope takes
precedence over primitive type. Times below are instrumented operator time,
not the uninstrumented full-forward latency above.

| Model | Dense compute | Attention | Sampling | Propagation | Elementwise / norm | Layout / memory | Other | Operator sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CSPN | 57.193 (44.69%) | 0.000 | 3.525 (2.75%) | 46.312 (36.19%) | 16.317 (12.75%) | 4.631 (3.62%) | 0.003 | 127.981 |
| DySPN | 256.208 (38.47%) | 0.000 | 5.942 (0.89%) | 322.429 (48.41%) | 59.323 (8.91%) | 22.088 (3.32%) | 0.005 | 665.995 |
| NLSPN | 275.718 (35.00%) | 0.000 | 5.950 (0.76%) | 441.702 (56.06%) | 22.494 (2.86%) | 41.980 (5.33%) | 0.003 | 787.847 |
| CompletionFormer (stabilized) | 272.684 (31.15%) | 21.073 (2.41%) | 11.730 (1.34%) | 460.740 (52.63%) | 86.425 (9.87%) | 22.733 (2.60%) | 0.004 | 875.389 |

| Model | NPU operator time | CPU operator time | Direct custom GridSample CPU time in uninstrumented forward | Placement conclusion |
|---|---:|---:|---:|---|
| CSPN | 120.693 ms (94.31%) | 7.288 ms (5.69%) | 0.000 ms | Mixed: compiler places rewritten `Less` and `Greater` on CPU; propagation itself is NPU |
| DySPN | 518.345 ms (77.83%) | 147.650 ms (22.17%) | 33.247 ms (5.08% of full forward), 30 calls | Mixed NPU+CPU; GridSample callback and compiler-selected surrounding operators use CPU |
| NLSPN | 459.772 ms (58.36%) | 328.075 ms (41.64%) | 187.843 ms (24.00% of full forward), 170 calls | Mixed NPU+CPU; propagation is the dominant class |
| CompletionFormer (stabilized) | 506.343 ms (57.84%) | 369.046 ms (42.16%) | 117.515 ms (15.57% of full forward), 170 calls | Mixed NPU+CPU; propagation is the dominant class, while attention remains NPU-visible |

The direct custom-operator figures are means of the callback time measured
inside every uninstrumented forward, aggregated with the same five-process
protocol as the headline latency. They are not taken from `PERF_DETAIL`.

RKNN does support some ConvTranspose configurations, but the five fixed-shape
decoder operators in these graphs were decomposed exactly into
nearest-neighbor Resize, a checkerboard mask, and ordinary Conv to avoid the
observed RK3588 submission/optimizer failure. Those primitives remain on NPU;
ConvTranspose is **not** silently lowered to CPU in these results. The
unsupported native DCNv2 nodes were lowered exactly to GridSample-based graphs.
RKNN 2.5 places GridSample on CPU and provides no working native implementation
on this board, so the benchmark registers an exact bilinear custom CPU
operator. Its callbacks and all NPU↔CPU handling remain inside full-forward
timing.

The large difference between direct GridSample callback time and RKNN's total
CPU-target time is expected: the former instruments only the custom C++
bilinear kernels, while RKNN's placement table also assigns surrounding
layout/arithmetic operators and scheduling work to CPU.

## RK3588 numerical validation

All finite deployed outputs are compared against the FP32 ONNX Runtime result
for the same fixed preprocessed sample.

| Model | Finite | Max absolute error | Mean absolute error | P95 absolute error | P99 absolute error |
|---|---|---:|---:|---:|---:|
| CSPN | yes | 0.008663 | 0.001050 | 0.002760 | 0.004137 |
| DySPN | yes | 0.018241 | 0.001680 | 0.004296 | 0.006297 |
| NLSPN | yes | 0.073621 | 0.004880 | 0.014753 | 0.024488 |
| CompletionFormer (stabilized) | yes | 0.094077 | 0.008156 | 0.023739 | 0.036712 |

### CompletionFormer FP16 stabilization

The unmodified vanilla CompletionFormer cannot produce a valid strict FP16
forward: PyTorch `.half()` returns 69,312/69,312 NaNs, and the first default
RKNN build returned 43,283 NaNs, 424 infinities, and only 25,605 finite values.
That build and its timing were rejected and removed.

The overflow sites were measured at `patch_embed3.proj`, then
`patch_embed4.proj`, then decoder `dec4.0.0`. The accepted graph uses a positive
scale `s=0.5` and algebraically equivalent reparameterization:

- patch projection weight and bias are multiplied by `s`, and the following
  LayerNorm epsilon by `s²`;
- decoder ConvTranspose weight is multiplied by `s`, and the following
  BatchNorm running mean by `s`, running variance and epsilon by `s²`.

This changes FP16 rounding/dynamic range but not the real-valued function. The
stabilized and unmodified FP32 ONNX graphs were exactly equal on the fixed
input (maximum and mean absolute error both 0). No retraining, calibration,
pruning, or task-aware tuning is involved. The RK3588 row is therefore labeled
**vanilla-equivalent FP16-stabilized**, not silently presented as an untouched
FP16 conversion. Full failure and equivalence evidence is in
[`completionformer_fp16_stabilization.json`](rk3588/conversion/completionformer_fp16_stabilization.json).

## Why breakdown sums differ from complete-forward latency

The breakdown sum is not intended to equal the headline latency.

- NVIDIA activity sums contain direct GPU kernel/memcpy/memset service time.
  They exclude CPU launch gaps and idle intervals. Nsight/NVTX instrumentation
  also perturbs host dispatch, so its captured CUDA-event time is not used as
  the uninstrumented latency.
- RKNN operator sums come from one instrumented `PERF_DETAIL` pass. They use
  runtime/compiler operator accounting and can include instrumentation and
  placement overhead differently from the monotonic uninstrumented boundary.
- No category is multiplied by a residual ratio. Percentages are only within
  their own directly measured activity/operator sum.

## Architecture implications

- A dense MAC/Conv path remains the first-order optimization target on both
  NVIDIA GPUs. CSPN is the only model where propagation is already about one
  fifth of GPU activity.
- CompletionFormer needs an attention-aware path in addition to a dense array;
  attention plus elementwise/normalization accounts for roughly 44% of RTX
  device activity.
- On RK3588, DySPN and NLSPN become propagation/CPU-placement limited. A native
  DCNv2/GridSample or affinity-reduction engine, address generation, and local
  SRAM reuse would matter more than enlarging the Conv array alone.
- Decoder ConvTranspose can remain NPU-resident through exact static lowering;
  its hardware cost is represented by Resize/mask/Conv primitives rather than
  an unmeasured CPU fallback.
- Layout/memory is a visible but secondary direct activity class (roughly
  0.7–5.5% in these profiles). This does not replace a separate bandwidth and
  buffer-capacity DSE because profiler service time alone does not expose all
  queuing or capacity constraints.

## Evidence and reproducibility

- Jetson five-process results and clocks:
  [`controlled_latency_fp32_sample386_20261022`](jetson_xavier/controlled_latency_fp32_sample386_20261022/)
- Jetson breakdown JSON and original Nsight reports:
  [`strict_nsys_breakdown_fp32_sample386_20261022`](jetson_xavier/strict_nsys_breakdown_fp32_sample386_20261022/)
- RTX inventory, five-process results, breakdown JSON, and original Nsight
  reports: [`autodl_nvidia`](autodl_nvidia/)
- RK3588 inventory, five-process results, raw `PERF_DETAIL`, placement, and
  numerical comparisons: [`rk3588`](rk3588/)
- Exact RKNN model/input/benchmark-binary hashes and compiler flags:
  [`deployment_manifest.json`](rk3588/deployment_manifest.json)
- The generated [`manifest.json`](manifest.json) records deterministic
  aggregates and SHA-256 hashes of every result/breakdown file used by this
  report.
- Energy results, raw sample hashes, and sensor-boundary metadata:
  [`energy_summary.json`](energy_summary.json); RK3588's strict measurement
  gate is recorded separately because no physical power channel is available.

Checkpoint loading on NVIDIA is strict and invokes the complete official
forward path including propagation. CompletionFormer's compatibility shim only
suppresses optional initialization-time OpenMMLab/PVT bootstrap files that are
overwritten by the complete strict checkpoint; it does not stub any inference
operator. The fixed NVIDIA input bundle SHA-256 is
`a2991898a2737dd6a772961c4394f458e10ed95dca6f6ba03c5b849a3e1eeac5`.

These are fixed-input latency and deployment-numerics experiments, not a
full-dataset task-accuracy evaluation.
