# NYU RTN Quantization Analysis Design

## Scope

Evaluate post-training round-to-nearest quantization on the converged official
architectures and their best validation iteration settings:

- CSPN ResNet18, 24 propagation iterations
- DySPN ResNet34 v1, 5 neighbors, 9 propagation iterations
- NLSPN ResNet34, 12 propagation iterations
- CompletionFormer full model, 6 propagation iterations

All evaluations use the same 64 indices sampled without replacement from the
654-image official NYU validation split with seed `20260804`. Calibration uses a
separate deterministic subset of the NYU training split.

## Quantization Protocol

Weights use per-output-channel symmetric RTN quantization. Activations use
per-tensor asymmetric static quantization with min/max ranges collected from 128
training samples. Fake quantization uses quantize/dequantize operations on CUDA,
so custom propagation and deformable sampling kernels remain executable.

The primary configurations are FP32, W8A8, and W4A4. Conv2d and Linear module
inputs, weights, and outputs are quantized. BatchNorm, LayerNorm, nonlinearities,
softmax, sigmoid, interpolation, grid sampling, deformable sampling, sparse-depth
anchors, and propagation arithmetic stay in FP32. A separate stress experiment
requantizes the propagated depth state after every iteration to isolate recurrent
error accumulation.

## Module Groups

Each weighted module is assigned to exactly one architecture-aware group:

- `encoder`: RGB/depth stems and convolutional or Transformer feature encoders
- `attention`: CompletionFormer Q/K/V projections and attention output projections
- `decoder`: feature reconstruction and up-projection modules
- `depth_head`: initial dense-depth prediction branch
- `propagation_head`: guidance, affinity, offset, and confidence prediction branches

In addition to full-model W8A8/W4A4, each group is quantized independently at both
bit widths. This identifies sensitive information without conflating errors from
earlier groups.

## Diagnostics

End-to-end metrics are RMSE, MAE, and ABS_REL over valid depth pixels. Regional
metrics split pixels into depth boundaries and smooth regions, sparse-input anchor
locations and holes, and near/mid/far depth bins.

Layer diagnostics include signal-to-quantization-noise ratio, cosine similarity,
saturation rate, and sign-flip rate. Model-output diagnostics compare initial
depth, guidance, affinity, offset, confidence, and each available propagation
state. Affinity analysis also records dominant-neighbor changes; offset analysis
records endpoint error. Propagation diagnostics report drift by iteration.

## Artifacts

The analysis writes machine-readable CSV/JSON files, quantized prediction NPZs,
module-sensitivity plots, information-damage plots, and per-iteration propagation
drift plots under `profile_logs/nyu_rtn_quantization/`.

A separate high-resolution contact sheet under
`profile_logs/nyu_prediction_random64_fp32/` shows all 64 samples in one figure.
The layout contains 16 rows by 4 sample blocks; each block compares ground truth,
CSPN, DySPN, NLSPN, and CompletionFormer with a shared 0-10 m depth scale.

## Interpretation Limits

This is a controlled software QDQ study, not an integer-kernel latency benchmark.
RTN strictly describes the rounding used for weights and activation codes; static
activation scales still require calibration. Results therefore measure numerical
sensitivity and expected accuracy risk, while deployment latency and operator
coverage require a target-specific quantization backend.
