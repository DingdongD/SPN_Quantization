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
  --quant-backend hardware \
  --out-dir profile_logs/nyu_hardware_aligned_quantization/cspn
```

Supported backends are `rtn`, `hardware`, `outlier`, `mixed`, `lognp`, and
`propagation`. The same command is used for DySPN, NLSPN, and CompletionFormer
by changing `--run-dir` and the model-specific external environment.

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
SPN_DATA_ROOT=/path/to/dataset-root \
python scripts/run_nyu_rtn_quantization.py \
  --run-dir output/nyu_converged_baselines/cspn_iter24 \
  --checkpoint best.pt \
  --sample-metrics profile_logs/reference_64/cspn/sample_metrics.csv \
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
