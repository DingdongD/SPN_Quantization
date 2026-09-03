# Unified FP16 Propagation Task-Aware Quantization Results

## Protocol

- Models: official CSPN, DySPN, NLSPN, and CompletionFormer checkpoints.
- Calibration: 128 samples from the existing NYU calibration manifests.
- Evaluation: the same 64 NYU evaluation indices for every model.
- Ordinary modules: task-gradient W/A allocation over 4, 6, and 8 bits.
- Propagation: excluded from ordinary allocation and configured as FP16.
- Metrics: pooled RMSE over valid depth pixels; invalid means at least one
  evaluated sample violated finite, positive, or reproducible prediction checks.

## Results

| Model | Configuration | Pooled RMSE | Relative loss vs FP32 | Average W/A bits | Valid |
| --- | --- | ---: | ---: | ---: | :---: |
| CSPN | FP32 reference | 0.193794 | 0.00% | - / - | yes |
| CSPN | Uniform W8A8 + FP16 propagation | 0.194217 | 0.22% | 8.000 / 8.000 | yes |
| CSPN | Task-aware W4A4 | 0.455239 | 134.91% | 4.000 / 4.000 | yes |
| CSPN | Task-aware W5A5 | 0.464990 | 139.94% | 4.999 / 4.996 | yes |
| CSPN | Task-aware W6A6 | 0.493235 | 154.51% | 5.994 / 5.943 | yes |
| DySPN | FP32 reference | 0.141821 | 0.00% | - / - | yes |
| DySPN | Uniform W8A8 + FP16 propagation | 0.141714 | -0.08% | 8.000 / 8.000 | yes |
| DySPN | Task-aware W4A4 | 0.220216 | 55.28% | 4.000 / 4.000 | yes |
| DySPN | Task-aware W5A5 | 0.588510 | 314.97% | 4.994 / 4.990 | yes |
| DySPN | Task-aware W6A6 | 0.184679 | 30.22% | 5.990 / 5.998 | yes |
| NLSPN | FP32 reference | 0.150894 | 0.00% | - / - | yes |
| NLSPN | Uniform W8A8 + FP16 propagation | 0.160508 | 6.37% | 8.000 / 8.000 | yes |
| NLSPN | Task-aware W4A4 | invalid | - | 4.000 / 4.000 | no |
| NLSPN | Task-aware W5A5 | invalid | - | 4.996 / 4.999 | no |
| NLSPN | Task-aware W6A6 | invalid | - | 5.992 / 5.997 | no |
| CompletionFormer | FP32 reference | 0.140168 | 0.00% | - / - | yes |
| CompletionFormer | Uniform W8A8 + FP16 propagation | 0.140605 | 0.31% | 8.000 / 8.000 | yes |
| CompletionFormer | Task-aware W4A4 | invalid | - | 4.000 / 4.000 | no |
| CompletionFormer | Task-aware W5A5 | 0.382341 | 172.77% | 5.000 / 5.000 | yes |
| CompletionFormer | Task-aware W6A6 | 0.154126 | 9.96% | 6.000 / 6.000 | yes |

## Interpretation

The protocol isolates propagation precision from ordinary CNN/attention
quantization. FP16 propagation is not sufficient to make every low-bit
assignment usable: NLSPN still fails at W6A6, which indicates that the
dominant damage is injected before or at the propagation boundary by the
ordinary feature/depth-head path. CompletionFormer reaches a valid W6A6
point, while its W4A4 assignment is still unstable.

The non-monotonic W5/W6 results for CSPN and DySPN are evidence that the
current layerwise gradient score is only a first-order allocation heuristic;
it does not model cross-layer activation-scale interactions. These numbers
must therefore not be described as an optimal bit allocation. The next
optimization target is a boundary-aware allocation constrained by measured
W8A8-to-W6A6 degradation, followed by task-aware refinement.

## Artifacts

The complete machine-readable results are under:

`profile_logs/nyu_four_model_unified_fp16_task_aware_64_v4/`

`run_manifest.json` verifies that every model contains `summary.csv`,
`pareto.csv`, and `manifest.json` before the run is considered complete.
