# CSPN Full Im2Col Matrix Visualization Design

## Objective

Add the missing raw unfolded-matrix visualizations for the official CSPN
PA-W8A8 experiment. The existing figures summarize activation and weight
statistics over channel and kernel-offset cells; the new figures must expose
the complete matrix structure corresponding to Conv2d as structured linear
algebra:

\[
W_{col} \in \mathbb{R}^{C_{out} \times K}, \qquad
X_{col} \in \mathbb{R}^{M \times K},
\]

where

\[
K=C_{in}K_hK_w, \qquad M=BH_{out}W_{out}.
\]

Every matrix element is retained. The capture and plotting paths must not
sample channels, kernel offsets, output channels or spatial tokens.

## Experimental Scope

- Reuse the completed official CSPN ResNet-18 PA-W8A8 experiment, converged
  checkpoint, 128-sample stratified calibration protocol and fixed 64-sample
  validation protocol.
- Capture the six Conv2d modules already selected by the layer-ranking
  contract in `run_manifest.json`.
- For each selected module, capture the validation sample with that module's
  largest recorded local Conv output error in `top_spatial_tokens.csv`.
- Capture ordinary Conv2d inputs and weights only. CSPN propagation tensors,
  grouped operations and ConvTranspose2d remain outside this visualization.
- Do not enable QAT, rotation, SmoothQuant, grouping, W4A4 or FP4.
- Do not change inference, quantizer configuration, calibration ranges or
  existing result files.

## Capture Contract

The capture command rebuilds the same official model and validates the
checkpoint digest, metadata digest, architecture, quantization contract and
selected module/sample identities against the existing experiment manifest.
It repeats the declared 128-sample calibration, freezes the existing W8A8
quantizers and evaluates only the required validation sample identities.

For each selected module, persist one compressed FP32 NPZ containing:

- the pre-input-QDQ activation feature map `[1, Cin, Hin, Win]`;
- the exact post-input-QDQ activation consumed by the Conv;
- the original FP32 Conv weight `[Cout, Cin, Kh, Kw]`;
- the deployed W8-QDQ Conv weight;
- module and sample identity;
- exact kernel, stride, padding and dilation geometry;
- activation quantizer format, signedness, bit width and scale.

The pre-input-QDQ tensor is the local floating-point input presented to the
current quantizer in the PA-W8A8 graph. It may already contain upstream
quantization error. The visualization therefore shows incremental local A8
rounding, not a paired full-FP32 network activation.

Persisting feature maps and native Conv weights avoids redundant expanded
storage. The plotting command reconstructs `X_col` with the strict existing
`ConvIm2ColLayout` implementation and reshapes `W_col` with the same K order.
It verifies matrix shapes and finite values before rendering.

## Full-Matrix Rendering

Generate two figures per selected module and sample.

### Weight Figure

Three 3D panels:

1. `|W_col|` for original weights;
2. `|Q_W(W_col)|` for deployed W8 weights;
3. `|W_col-Q_W(W_col)|`.

Axes:

- x: `K = input channel x kernel offset`;
- y: output channel;
- z: absolute weight or absolute weight error.

### Activation Figure

Three 3D panels:

1. `|X_col|` for the exact pre-input-QDQ activation;
2. `|Q_A(X_col)|` for the activation consumed by the Conv;
3. `|X_col-Q_A(X_col)|`.

Axes:

- x: spatial patch/token index `M`;
- y: `K = input channel x kernel offset`;
- z: absolute activation or absolute activation error.

FP and QDQ panels in each figure share z limits and color normalization.
The error panel uses its own range because its scale is substantially smaller.
Use Arial-first font configuration, no figure-level title and grid lines
behind data.

## No-Sampling Line Strategy

A separate 3D bar or stem object per element is not viable for matrices with
millions of entries. Render the complete matrix as line curtains:

- choose the shorter matrix dimension as the number of line objects;
- traverse the longer dimension within each line;
- include every matrix element in exactly one line;
- use a vectorized `Line3DCollection` where possible;
- rasterize dense line collections in PNG and PDF output.

This changes only the drawing primitive. It does not sample, pool, interpolate
or reduce matrix data. The figure manifest records matrix dimensions, element
count, line orientation and confirms `rendered_elements == matrix_elements`.

## Outputs

Create a new immutable subdirectory under the existing experiment root:

```text
full_matrix_visualization/
  captures/<module>/sample_<index>.npz
  figures/<module>/sample_<index>_weights.png
  figures/<module>/sample_<index>_weights.pdf
  figures/<module>/sample_<index>_activations.png
  figures/<module>/sample_<index>_activations.pdf
  capture_manifest.csv
  figure_manifest.csv
  run_manifest.json
```

Module path components remain reversible directory components. Existing
experiment tables, figures and prediction outputs are read-only.

## Validation

Unit tests must verify:

- native Conv feature maps reconstruct the same full `X_col` as `F.unfold`;
- native weights reconstruct the complete `W_col` in exact K order;
- no-sampling line conversion contains every matrix coordinate and value
  exactly once for both possible line orientations;
- FP/QDQ/error matrices and shared normalization are numerically correct;
- malformed identities, geometry, scales and non-finite values fail loudly;
- synthetic full-matrix artifacts produce nonblank PNG/PDF pairs and complete
  manifests without model execution.

Production acceptance requires:

- exactly six selected modules and one worst sample per module;
- capture identities matching the existing ranked tables;
- all capture tensors stored as FP32 and finite;
- exact unfolded dimensions and element counts in the figure manifest;
- `rendered_elements == matrix_elements` for every panel;
- inspected figures with readable axes and visible FP/QDQ/error structure;
- unchanged existing PA-W8A8 artifacts;
- complete repository tests passing.
