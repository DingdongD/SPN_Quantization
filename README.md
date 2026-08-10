# SPN_Quantization

Unified post-training quantization and analysis tools for NYU depth-completion
models: vanilla CSPN, DySPN, NLSPN, and CompletionFormer.

## Scope

This repository contains the shared quantization instrumentation, propagation
state handling, model adapters, calibration analysis, prediction comparison,
and regression tests. Dataset files, checkpoints, profiler traces, and
generated experiment outputs are intentionally excluded from Git.

The quantization runner supports RTN, standard hardware-aligned QDQ, outlier
mitigation, mixed configurations, and propagation-aware integer QDQ. LogNP
support is retained as a general quantization method. The abandoned selective
LogNP implementation is not part of this repository.

## Layout

```text
scripts/       quantization runners, observers, adapters, analysis, plotting
models/        local CSPN and hardware-aligned model support
nlspn_test/    local NLSPN hardware-reference support
external/      official DySPN, NLSPN, and CompletionFormer submodules
tests/         quantization and analysis regression tests
docs/          experiment designs and execution plans
data/          local dataset mount point, ignored by Git
datalist/      local NYU list mount point, ignored CSV files
pretrained/   local checkpoints, ignored by Git
reports/       lightweight notes; generated reports are ignored
```

## Main entry point

Run one converged model using the shared interface. The model is inferred from
the training run metadata in `--run-dir`.

```bash
python scripts/run_nyu_rtn_quantization.py \
  --run-dir output/nyu_converged_baselines/cspn_iter24 \
  --sample-metrics profile_logs/nyu_activation_outliers/cspn/sample_metrics.csv \
  --data-root /path/to/nyu-workspace \
  --quant-backend hardware \
  --out-dir profile_logs/nyu_hardware_aligned_quantization/cspn
```

Supported backends are `rtn`, `hardware`, `outlier`, `mixed`, `lognp`,
`propagation`, and `fp4`. The same command is used for DySPN, NLSPN, and
CompletionFormer by changing `--run-dir` and the model-specific external
environment.

## Propagation-aware quantization

The `propagation` backend separates the SPN operator from ordinary CNN QDQ. It
quantizes affinity values before fixed-point normalization, reconstructs the
center coefficient from the quantized neighbors, uses signed Q13 INT16
coefficients with an integer normalization reference, and preserves
sparse-depth anchors. Confidence is unsigned A8. Offsets and propagation
states can be promoted to A8 independently. Deformable/grid sampling and the
propagation multiply-accumulate remain float QDQ references; this is not a
bit-exact integer DCN/grid-sample deployment kernel.

```bash
python scripts/run_nyu_rtn_quantization.py \
  --run-dir output/nyu_converged_baselines/cspn_iter24 \
  --checkpoint best.pt \
  --sample-metrics profile_logs/reference_64/cspn/sample_metrics.csv \
  --data-root /path/to/nyu-workspace \
  --quant-backend propagation \
  --calibration-samples 128 \
  --max-eval-samples 64 \
  --config-names FP32 PA_Generic_W4A4 PA_Constraint PA_OffsetA8 \
    PA_StateA8 PA_W4A8 PA_W8A8 \
  --export-prediction-configs FP32 PA_Generic_W4A4 PA_Constraint \
    PA_OffsetA8 PA_StateA8 PA_W4A8 PA_W8A8 \
  --out-dir profile_logs/nyu_propagation_aware_quantization
```

NLSPN and CompletionFormer must run in the Python environment containing their
compiled DCN extension. Render the 64-sample prediction, absolute-error,
propagation-step, and constraint comparisons with:

```bash
python scripts/plot_propagation_aware_quantization.py \
  --root profile_logs/nyu_propagation_aware_quantization
```

The unified evaluation is stored in
`profile_logs/nyu_propagation_aware_quantization_unified`. On its fixed
64-sample NYU evaluation set, mean per-sample RMSE in metres was:

| Model | FP32 | Generic W4A4 | Best propagation-aware W4A4 | W4A8 | W8A8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CSPN | 0.1669 | failed (64/64 non-finite) | 1.0745 | 0.2152 | 0.1785 |
| DySPN | 0.1202 | 2.7314 | 2.6306 | 0.1315 | 0.1271 |
| NLSPN | 0.1282 | 1.9476 | 1.4854 | 0.1764 | 0.1496 |
| CompletionFormer | 0.1193 | 3.5528 | 2.0429 | 0.5883 | 0.1292 |

The propagation-aware W4A4 variants enforce zero coefficient-sum error and
zero contraction violations, and remove CSPN's non-finite output failure.
Their remaining error is dominated by W4A4 corruption of the initial dense
prediction and coarse affinity, offset, and recurrent-state quantization.
W4A8 removes all non-finite outputs and is much better than W4A4, but its
FP32-relative RMSE degradation remains 28.9% for CSPN, 9.4% for DySPN, 37.6%
for NLSPN, and 392.9% for CompletionFormer. CompletionFormer's W4A8 error is
already present in the initial dense prediction; its propagation loop reduces
rather than amplifies that error, but cannot recover the W4-damaged feature
and depth heads. W8A8 remains close to FP32 for all four official structures,
with relative degradation of 6.9%, 5.7%, 16.7%, and 8.2%, respectively.

To dispatch all four models through the shared quantization interface:

```bash
SPN_DATA_ROOT=/path/to/dataset-root \
SPN_EXTERNAL_ROOT="$PWD/external" \
COMPLETIONFORMER_ROOT="$PWD/external/CompletionFormer" \
scripts/run_all_model_quantization.sh
```

Set `CSPN_PYTHON`, `DYSPN_PYTHON`, `NLSPN_PYTHON`, and
`COMPLETIONFORMER_PYTHON` when the four models use different environments.
Run `python scripts/check_migration.py` before the first call to inspect the
paths and Python packages in the target environment.

## FP4 activation validation

The `fp4` backend compares calibrated signed E2M1 activation QDQ with matched
uniform INT4 and A8 controls. Ordinary Conv/Linear, ReLU, concat, and
LayerNorm-output boundaries are quantized while the sparse-depth input, final
depth/guidance/confidence outputs, affinity, offsets, and propagation states
remain A8. Weights use per-output-channel RTN, biases remain FP32 for
activation-format isolation, and propagation keeps quantize-then-normalize Q13
coefficients.

Set every migration-dependent path and device explicitly, then run the smoke
stage before the formal 128-calibration/64-evaluation stage:

```bash
export SPN_DATA_ROOT=/path/to/cspn-training-workspace
export SPN_EXTERNAL_ROOT="$PWD/external"
export COMPLETIONFORMER_ROOT="$PWD/external/CompletionFormer"
export FP4_REFERENCE_ROOT="$PWD/profile_logs/nyu_propagation_aware_quantization_unified"
export CSPN_PYTHON=/path/to/python
export DYSPN_PYTHON=/path/to/python
export NLSPN_PYTHON=/path/to/dcn-python
export COMPLETIONFORMER_PYTHON=/path/to/dcn-python
export CSPN_DEVICE=cuda:1
export DYSPN_DEVICE=cuda:2
export NLSPN_DEVICE=cuda:0
export COMPLETIONFORMER_DEVICE=cuda:0

export FP4_OUTPUT_ROOT="$PWD/profile_logs/nyu_fp4_activation_validation_smoke"
scripts/run_fp4_activation_validation.sh smoke

export FP4_OUTPUT_ROOT="$PWD/profile_logs/nyu_fp4_activation_validation"
scripts/run_fp4_activation_validation.sh full
```

The four official models were rerun after adding complete `ConvTranspose2d`
coverage, explicit per-input-channel concat scales, and standard Conv-BN
folding. The corrected evaluation uses 128 calibration samples and the same
fixed 64-sample NYU evaluation set. Mean per-sample RMSE is reported in metres:

| Model | FP32 | W8 INT4 | W8 E2M1 | W8 A8 | W4 INT4 | W4 E2M1 | W4 A8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CSPN | 0.1669 | 0.9869 | 0.3925 | 0.1786 | 0.9920 | 0.4256 | 0.2171 |
| DySPN | 0.1202 | 0.3901 | 0.9016 | 0.1271 | 0.3904 | 0.8166 | 0.1313 |
| NLSPN | 0.1282 | 1.4652 | 0.9308 | 0.1449 | 1.3290 | 0.9812 | 0.1730 |
| CompletionFormer | 0.1193 | 0.9311 | 0.8277 | 0.1291 | 2.0297 | 1.8654 | 0.6636 |

DySPN uses the official `mode="dyspn"` `grid_sample` propagation path; deformable
convolution is not active. E2M1 improves over uniform INT4 in six of eight
weight/model comparisons, but remains substantially worse than A8 and is worse
than INT4 for both DySPN comparisons. CompletionFormer also has a separate W4
weight sensitivity: W4A8 reaches 0.6636 m while W8A8 reaches 0.1291 m. These
are float E2M1 QDQ accuracy results. The A100 run does not use native FP4
kernels and makes no latency or throughput claim. Metrics and prediction/error
figures are written under
`profile_logs/nyu_fp4_activation_validation_corrected`.

## Strict W4 reconstruction evaluation

The strict comparison replays frozen RTN, AdaRound, and BRECQ W4 weight
contracts on the same 64 calibration and 64 evaluation samples. Its primary
configurations are uniform W4A4, W4-E2M1, and W4A8 with FP32 bias and A8
semantic/propagation boundaries. `HW_W4A4_full` is kept as a separate integer
stress baseline.

```bash
STRICT_W4A4_FP4_OUTPUT_ROOT=/path/to/output \
scripts/run_strict_w4a4_fp4_evaluation.sh smoke

STRICT_W4A4_FP4_OUTPUT_ROOT=/path/to/output \
scripts/run_strict_w4a4_fp4_evaluation.sh full
```

The formal run found that no W4A4 or W4-E2M1 combination preserved FP32
performance under the predeclared 10% RMSE threshold. BRECQ improved several
same-format RTN results, but activation error remained dominant. Only DySPN
W4A8 with RTN (0.1312 m) and BRECQ (0.1304 m) met the preservation criterion.
See `docs/2026-08-07-strict-w4a4-fp4-reconstruction-results.md` for the full
matrix, paired-bootstrap interpretation, and artifact layout.

## W4A4 activation histograms

The activation histogram runner profiles every real uniform-QDQ activation
boundary in the official CSPN, DySPN, NLSPN, and CompletionFormer structures.
It reuses the strict RTN W4A4 policy and its fixed 64 NYU calibration samples.
Weights are per-output-channel W4. Ordinary signed activations use symmetric
A4, ReLU/nonnegative activations use unsigned A4, configured concat consumers
use per-input-channel scales, and sparse-depth plus depth/guidance/confidence
and propagation boundaries retain their declared A8 policy. Dense ground truth
stays FP32 and is never passed to the quantizer. This is profiling only; no
training or checkpoint update occurs.

Set every migration-dependent path explicitly. `smoke` performs the full
strict calibration and profiles one sample through both collection passes;
`full` profiles all 64 samples.

```bash
export SPN_DATA_ROOT=/path/to/cspn-training-workspace
export SPN_EXTERNAL_ROOT="$PWD/external"
export COMPLETIONFORMER_ROOT="$PWD/external/CompletionFormer"
export STRICT_W4A4_FP4_ROOT=/path/to/nyu_strict_w4a4_fp4_evaluation
export CSPN_PYTHON=/path/to/python
export DYSPN_PYTHON=/path/to/python
export NLSPN_PYTHON=/path/to/dcn-python
export COMPLETIONFORMER_PYTHON=/path/to/dcn-python
export CSPN_GPU=0
export DYSPN_GPU=1
export NLSPN_GPU=2
export COMPLETIONFORMER_GPU=3

export W4A4_HISTOGRAM_OUTPUT_ROOT="$PWD/profile_logs/nyu_w4a4_activation_histograms_smoke"
scripts/run_w4a4_activation_histograms.sh smoke

export W4A4_HISTOGRAM_OUTPUT_ROOT="$PWD/profile_logs/nyu_w4a4_activation_histograms_64"
scripts/run_w4a4_activation_histograms.sh full
```

Use a new `W4A4_HISTOGRAM_OUTPUT_ROOT` for each invocation. Each model directory
contains `histogram_data.npz`, `histogram_index.csv`, `outlier_summary.csv`, a
paginated `all_sites_histograms.pdf`, `critical_layers.png`,
`rgb_depth_input_histograms.png`, `group_outlier_distribution.png`, and strict
identity metadata. Synthetic RGB/depth slices are marked and excluded from
model/group error-energy aggregation. The root directory contains the
four-model comparison PNG and CSV. Saturation means pre-clamp out-of-range
values; endpoint-code occupancy and zero-code ratio are reported separately.

## Dependencies

The local code expects Python, PyTorch, NumPy, pandas, h5py, Pillow,
scikit-image, matplotlib, and torchvision. NLSPN and CompletionFormer require
their official external repositories and the CUDA deformable-convolution
extension. DySPN requires its official external repository.

Expected external paths can be overridden with environment variables:

```text
external/DySPN
external/NLSPN_ECCV20
external/CompletionFormer
```

The supported variables are `SPN_EXTERNAL_ROOT`, `COMPLETIONFORMER_ROOT`, and
`SPN_DATA_ROOT`; no source file needs to be edited during migration.

Place NYU HDF5 data under `data/nyudepth_hdf5` and the train/validation CSVs
under `datalist/` before running calibration or evaluation.

Clone this repository with the official model submodules:

```bash
git clone --recurse-submodules \
  https://github.com/DingdongD/SPN_Quantization.git
cd SPN_Quantization
git submodule update --init --recursive
```

## Tests

```bash
python -m pytest -q tests
```

Tests that import official external models require the corresponding external
repository and CUDA environment; the quantizer and analysis unit tests can be
run independently.

## License and provenance

The local CSPN model code is derived from the original CSPN project. DySPN,
NLSPN, and CompletionFormer are linked as submodules to their official source
repositories; use their original licenses and repository history.
