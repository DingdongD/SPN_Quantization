# Vanilla SPN on Jetson AGX Xavier — controlled retest

## Measurement status

The four FP32 vanilla models have completed a five-process controlled
full-forward latency retest and a separate Nsight Systems device-activity
breakdown. Each model was constructed from the official source, loaded from
its complete checkpoint with `load_state_dict(..., strict=True)`, set to
evaluation mode, and invoked through the full `model(*inputs)` path, including
propagation.

The timed boundary starts with preprocessed inputs already resident on the GPU.
Dataset decoding, preprocessing, host-to-device transfer, model construction,
and checkpoint loading are intentionally outside the boundary.

## Full-model forward latency

Each independent process used 30 warmups followed by 3 rounds × 50 synchronized
CUDA-event measurements. Five processes were run for every model, for 750
timed forwards/model. The table reports statistics across the five process
means; the 95% CI is a two-sided Student-t interval for the mean of run means.

| Model | Mean of run means (ms) | Median run mean (ms) | Run-mean range (ms) | 95% CI (ms) | CV |
|---|---:|---:|---:|---:|---:|
| CSPN | 71.662 | 71.369 | 69.974–73.286 | 70.088–73.235 | 1.77% |
| DySPN | 139.333 | 139.177 | 139.034–139.959 | 138.877–139.789 | 0.26% |
| NLSPN | 157.024 | 156.789 | 155.768–158.318 | 155.726–158.321 | 0.67% |
| CompletionFormer | 193.359 | 193.868 | 191.245–195.903 | 191.016–195.702 | 0.98% |

TF32 was disabled and cuDNN benchmark selection was enabled. The complete
per-process summaries, control metadata, and telemetry are in
[`controlled_latency_fp32_sample386_20261022`](controlled_latency_fp32_sample386_20261022/).
## Directly measured GPU-activity breakdown

The table below reports mean CUDA kernel, memcpy, and memset service time from
five Nsight-captured full forwards. Every device activity is associated through
its CUPTI `correlationId` with the CUDA runtime launch, then with the innermost
`aten::` NVTX range. Activities launched inside the model's propagation-module
range are assigned to Propagation first, making all columns mutually exclusive.
No category was rescaled to the uninstrumented latency.

| Model | Dense compute | Attention | Sampling | Propagation | Elementwise / norm | Layout / memory | Other | GPU activity sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CSPN | 30.991 | 0.000 | 0.126 | 9.269 | 2.094 | 0.481 | 0.179 | 43.139 |
| DySPN | 101.304 | 0.000 | 0.000 | 7.043 | 24.831 | 5.745 | 0.191 | 139.114 |
| NLSPN | 112.010 | 0.000 | 0.000 | 6.151 | 18.370 | 5.691 | 0.000 | 142.223 |
| CompletionFormer | 98.619 | 18.834 | 0.267 | 6.342 | 38.915 | 9.583 | 1.887 | 174.445 |

All values are milliseconds per forward in the instrumented pass. Detailed
per-forward results and classification shares are in
[`strict_nsys_breakdown_fp32_sample386_20261022`](strict_nsys_breakdown_fp32_sample386_20261022/);
the original `.nsys-rep` traces and captured CUDA-event records are retained in
its `raw` subdirectory for audit.

The GPU-activity sum is **not** a second end-to-end latency number. It sums
direct device work and excludes CPU launch gaps and idle intervals. Nsight and
operator NVTX instrumentation also perturb host dispatch, so the instrumented
CUDA-event latency is retained in JSON for audit but must not replace the
uninstrumented latency table. This is why the six hardware classes are not
scaled to add up to the full-forward mean.

## Scope and reproducibility

- Board: NVIDIA Jetson AGX Xavier 32 GB in MAXN mode. During the controlled run,
  GPU `cur/min/max` were all 1377 MHz and CPU7 `cur/min/max` were all
  2265.6 MHz. The benchmark and its library threads were pinned to CPU7;
  `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` were one.
- The exact input bundle SHA-256 was
  `a2991898a2737dd6a772961c4394f458e10ed95dca6f6ba03c5b849a3e1eeac5`.
  Model order was rotated between repeats. A 500 ms tegrastats log contains
  2388 samples; GPU temperature was 48.0–70.5 °C and GR3D reached 99%.
- Unrelated user processes were not stopped. Other CPU cores averaged roughly
  68–73% utilization during the run, so this is a repeatable clock- and
  affinity-controlled result under recorded background load, not an exclusive
  idle-system baseline. EMC frequency could not be read with the available
  account and is not claimed as locked.
- Software: JetPack 5.1.2 / L4T R35.4.1, CUDA 11.4, Jetson PyTorch
  `2.1.0a0+41361538.nv23.06`, torchvision 0.16.1.
- Workload: batch 1, NYU sample index 386, resolution 228×304, FP32.
- Model revision: `587e9f7e22d474f013290e5a6348c7f55a8cb995`.
- CompletionFormer's mmcv/mmseg compatibility shim only bypassed its optional
  initialization-time PVT preload. The complete vanilla checkpoint was loaded
  strictly afterward; no inference operator or propagation path was stubbed.
- Results characterize this fixed input and software/power state; they are not
  a full-dataset latency distribution.

The earlier profiler result that proportionally rescaled operator shares into
the non-propagation latency budget has been removed. It was not a strict,
independent device-time measurement.
