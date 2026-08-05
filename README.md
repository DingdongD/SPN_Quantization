# SPN_Quantization

Unified post-training quantization and analysis tools for NYU depth-completion
models: vanilla CSPN, DySPN, NLSPN, and CompletionFormer.

## Scope

This repository contains the shared quantization instrumentation, propagation
state handling, model adapters, calibration analysis, prediction comparison,
and regression tests. Dataset files, checkpoints, profiler traces, and
generated experiment outputs are intentionally excluded from Git.

The quantization runner supports RTN, standard hardware-aligned QDQ, outlier
mitigation, and mixed configurations. LogNP support is retained as a general
quantization method. The abandoned selective LogNP implementation is not part
of this repository.

## Layout

```text
scripts/       quantization runners, observers, adapters, analysis, plotting
models/        local CSPN and hardware-aligned model support
nlspn_test/    local NLSPN hardware-reference support
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

Supported backends are `rtn`, `hardware`, `outlier`, and `mixed`.
The same command is used for DySPN, NLSPN, and CompletionFormer by changing
`--run-dir` and the model-specific external environment.

## Dependencies

The local code expects Python, PyTorch, NumPy, pandas, h5py, Pillow,
scikit-image, matplotlib, and torchvision. NLSPN and CompletionFormer require
their official external repositories and the CUDA deformable-convolution
extension. DySPN requires its official external repository.

Expected external paths can be overridden in the runner or environment:

```text
/workspace/external_depth_completion_models/DySPN
/workspace/external_depth_completion_models/NLSPN_ECCV20
/workspace/CompletionFormer
```

Place NYU HDF5 data under `data/nyudepth_hdf5` and the train/validation CSVs
under `datalist/` before running calibration or evaluation.

## Tests

```bash
python -m pytest -q tests
```

Tests that import official external models require the corresponding external
repository and CUDA environment; the quantizer and analysis unit tests can be
run independently.

## License and provenance

The local CSPN model code is derived from the original CSPN project. External
DySPN, NLSPN, and CompletionFormer source trees are not vendored here; use
their original licenses and repositories when installing them.
