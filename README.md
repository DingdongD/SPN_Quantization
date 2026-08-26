# SPN_Quantization

Unified post-training quantization and analysis tools for NYU depth-completion
models: vanilla CSPN, DySPN, NLSPN, and CompletionFormer.

## Scope

This repository contains the shared quantization instrumentation, propagation
state handling, model adapters, calibration analysis, prediction comparison,
and regression tests. Dataset files, checkpoints, profiler traces, and
generated experiment outputs are intentionally excluded from Git.

The quantization runner supports RTN, hardware-aligned QDQ, mixed precision,
propagation-aware integer QDQ, and CompletionFormer joint quantization. The
[quantization framework inventory](docs/2026-08-20-quantization-framework-inventory.md)
defines the active and retired methods, measured evidence, artifact roots, and
rerun commands.

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

Supported backends are `rtn`, `hardware`, `mixed`, `propagation`,
`completionformer_joint`, and `completionformer_front_pareto`. The same
command is used for DySPN, NLSPN, and CompletionFormer by changing `--run-dir`
and the model-specific external environment.

## Three-model selected quantization launch

The formal DySPN, NLSPN, and CompletionFormer matrix is defined by
`configs/three_model_selected_quantization.json`; exact interpreters,
environments, P3/T3 budgets, cost inputs, hard-deployment controls, and QAT
settings are defined separately by
`configs/three_model_quantization_launch.json`. Neither file permits automatic
Python or GPU selection. The configured lanes are DySPN on `cuda:0` with
`/opt/conda/bin/python`, NLSPN on `cuda:1` with the CompletionFormer Python 3.7
environment, and CompletionFormer on `cuda:3` with that same Python 3.7
environment. Hard deployment uses the explicit reviewed no-fold policy in
`hard_deployment.fold_conv_bn`; every applicable command carries
`--skip-conv-bn-fold`. This is a fixed matrix setting, not a runtime fallback.

The first two jobs in each model lane create and validate the five static
inputs under
`/workspace/SPN_Quantization/profile_logs/nyu_three_model_selected_quantization`
using `scripts/prepare_nyu_three_model_static_inputs.py` with that model's
exact interpreter, environment, and indexed CUDA device. The command declares
all five output paths and selects 128 train identities from 256 seeded
candidates as 32 distribution-tail samples plus 96 weighted k-medoids. It also
writes the configured ordered 64 validation identities and captures official
model weight-MAC and activation-traffic costs. The following validation job
checks exact schemas, split-list hashes, checkpoint/model/dataset identities,
ordered calibration and evaluation identities, descriptor schema, complete
positive cost coverage, and cross-file hashes. P3/T3, HAWQ trace, and every QAT
job depend on that validation receipt.

Generate only the reviewable DAG and per-job command manifests:

```bash
/opt/conda/bin/python \
  scripts/launch_nyu_three_model_quantization.py plan \
  --config "$PWD/configs/three_model_selected_quantization.json" \
  --launch-spec "$PWD/configs/three_model_quantization_launch.json"
```

This writes `launch/launch_plan.json` and 70 manifests under `launch/jobs/`,
one for every job. Each manifest records the exact command, full replacement
environment, configured CUDA device, input paths and revisions, output path,
UTC start/end times, and exit status. After reviewing that plan, formal
execution is an explicit second command:

```bash
/opt/conda/bin/python \
  scripts/launch_nyu_three_model_quantization.py execute \
  --config "$PWD/configs/three_model_selected_quantization.json" \
  --launch-spec "$PWD/configs/three_model_quantization_launch.json" \
  --plan /workspace/SPN_Quantization/profile_logs/nyu_three_model_selected_quantization/launch/launch_plan.json
```

Each model remains serial on its own GPU while the three model lanes run
concurrently. Static-input production and semantic validation precede all
artifact-producing method jobs. P3/T3 search precedes selected PTQ and mixed
task-aware QAT.
HAWQ trace precedes CPU allocation, which precedes HAWQ QAT. A validated formal
artifact index containing all five PTQ manifests and all four terminal QAT
checkpoints precedes the exact ten-method fixed-64 evaluation. Aggregation and
plots precede the final 30-row cross-model summary.

The committed one-sample harness is
`scripts/smoke_nyu_selected_quantization.py`. Run it separately with each
model's declared environment, interpreter, `--model`, `--device`, and a new
`--output` directory. It executes FP32, RTN W8A8/W4A4, QDrop/BRECQ W6A6 hard
deployment, one LSQ++ W4A4 step, one HAWQ probe, and one P3/T3 candidate while
requiring native CUDA execution, official propagation execution, finite
`[1,1,228,304]` output, and model-specific propagation invariants. DySPN HAWQ
uses the declared central block finite-difference HVP (`epsilon=0.001`);
NLSPN and CompletionFormer use the declared autograd block HVP. Trace artifacts
persist and validate this exact per-model choice.

## CompletionFormer joint integer quantization

The `completionformer_joint` backend keeps the official full CompletionFormer
PVT structure unchanged and validates all 16 Attention plus 16 `concat_conv`
blocks before execution. Calibration has two paired passes over identical
indices: an FP target pass, then an ordinary W4A4 plus semantic-A8
reconstruction pass. Evaluation samples and NYU ground truth are not used for
scale selection.

Q, K, and V use independent per-head signed A4 or A8 scales. Their codes are
stored in INT8 arithmetic lanes; A4 storage does not imply A8 precision. QK
and probability-V products use strict INT32 accumulation. Softmax remains
FP16 and its output is unsigned A8. Non-aligned Attention matrix dimensions
are zero-padded in the integer domain before `_int_mm` and sliced afterward.

Each `concat_conv` keeps Transformer and CNN branch scales separate, computes
two W4/activation integer partial convolutions, requantizes both INT32 results
to a declared common accumulator scale, and quantizes bias in that same scale.
This is a correctness reference backend and does not claim native CUDA kernel
speedup.

Run the fixed 64-calibration/64-evaluation experiment with explicit official
source and environment paths:

```bash
export COMPLETIONFORMER_PYTHON=/path/to/completionformer/python
export COMPLETIONFORMER_ROOT="$PWD/external/CompletionFormer"
export COMPLETIONFORMER_REFERENCE_METRICS=/path/to/fixed64/sample_metrics.csv
export SPN_DATA_ROOT=/path/to/nyu-workspace
scripts/run_completionformer_joint_quantization.sh
```

## CompletionFormer front-encoder W8A8 Pareto search

The `completionformer_front_pareto` backend starts from the validated joint
W4A4 contract and promotes selected early encoder units to W8A8. The atomic
units are `Stem`, the three official `embed_layer1` residual blocks, the four
official `embed_layer2` residual blocks, and `patch_embed1`. A unit owns its
complete Conv/Linear weight sites and calibrated activation boundaries; a
selected unit cannot be partially promoted.

The search uses 64 NYU training samples for calibration and 32 different
training samples for cost-aware greedy selection. The fixed 64 validation
samples are used only for final metrics and visualization. Strict
official-order prefixes are evaluated alongside the greedy path. This is an
approximate candidate frontier, not exhaustive enumeration of all 512 unit
subsets, and no training or checkpoint update occurs.

W8A8 MAC, parameter, and operator shares use all ordinary quantized
Conv2d/ConvTranspose2d/Linear modules in CompletionFormer as the denominator.
Custom Attention QK/AV and propagation work is deliberately excluded and is
identified in metadata. The runner records the actual per-site bit contract in
`front_encoder_bit_manifest.csv` and fails if a promoted or baseline site has
the wrong precision.

Run the fixed protocol with explicit migration-dependent paths:

```bash
export COMPLETIONFORMER_RUN_DIR=/path/to/completionformer_iter18
export COMPLETIONFORMER_REFERENCE_METRICS=/path/to/fixed64/sample_metrics.csv
export SPN_DATA_ROOT=/path/to/nyu-workspace
export COMPLETIONFORMER_ROOT="$PWD/external/CompletionFormer"
export COMPLETIONFORMER_DCN_PATH=/path/to/verified/dcn/lib
export COMPLETIONFORMER_PYTHON=/path/to/completionformer/python
export COMPLETIONFORMER_DEVICE=cuda:0
scripts/run_completionformer_front_encoder_pareto.sh
```

The output contains unit and cost manifests, every greedy candidate, final
per-sample and aggregate metrics, Pareto selections, strict prediction
payloads, three RMSE-versus-W8A8-share figures, and a 64-sample sheet comparing
GT, FP32, W4A4, W4A8, the knee, and the lowest-RMSE front set with absolute
error maps.

When that Python environment does not already provide the official modulated
DCN extension, build it against the active PyTorch/CUDA toolchain. The builder
copies only the official CompletionFormer DCN sources into the declared output
directory, applies checked PyTorch API migrations, and fails if the directory
already exists or the expected source pattern changes:

```bash
python scripts/build_completionformer_dcn_extension.py \
  --completionformer-root "$COMPLETIONFORMER_ROOT" \
  --out-dir "$PWD/profile_logs/runtime_extensions/completionformer_dcn" \
  --cuda-arch 8.0 \
  --jobs 2
export COMPLETIONFORMER_DCN_PATH="$PWD/profile_logs/runtime_extensions/completionformer_dcn/lib"
scripts/run_completionformer_joint_quantization.sh
```

The launcher verifies `COMPLETIONFORMER_DCN_PATH` when it is declared and does
not substitute a floating or torchvision deformable-convolution path. It sets
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` because the official initialization file
uses a trusted legacy PyTorch serialization format; the final experiment
checkpoint is still loaded strictly into the official architecture.

The output contains end-to-end sample metrics, Attention/concat manifests,
joint scale-search rows, local integer metrics, all six prediction sets, and
Arial figures under
`profile_logs/nyu_completionformer_joint_integer_64/analysis`.

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

## Strict W4A8 reconstruction evaluation

The active strict comparison replays frozen RTN, AdaRound, and BRECQ W4 weight
contracts under the same W4A8 activation and propagation contract. Generate
contracts with `scripts/run_nyu_strict_reconstruction.py`, evaluate them from
the original checkpoint with `scripts/run_nyu_edge_quantization.py`, and
aggregate results with `scripts/plot_nyu_strict_reconstruction.py`. See the
framework inventory for the required commands and current artifact roots.

## W4A4 activation histograms

The activation histogram runner profiles every real uniform-QDQ activation
boundary in the official CSPN, DySPN, NLSPN, and CompletionFormer structures.
It uses `PA_W4A4_PROP_A8` and a fixed-seed set of 64 NYU training samples.
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
`rgb_depth_input_histograms.png`, `group_outlier_distribution.png`, and model
provenance metadata. Synthetic RGB/depth slices are marked and excluded from
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

The strict CSPN activation-resolution study and its measured NYU results are
documented in
[`docs/2026-08-12-cspn-activation-resolution-results.md`](docs/2026-08-12-cspn-activation-resolution-results.md).

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
