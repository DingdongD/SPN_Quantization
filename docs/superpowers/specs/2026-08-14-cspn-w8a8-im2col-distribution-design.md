# CSPN W8A8 Im2Col Distribution Design

## Objective

Measure how the official CSPN convolutions distribute signal and W8A8
quantization error across the two matrix dimensions exposed by Im2Col:

\[
M = B H_{out} W_{out}, \qquad
K = C_{in} K_h K_w.
\]

The experiment separates spatial patch variation from the structured K axis,
where each K element identifies both an input channel and a kernel offset. It
is a distribution diagnosis, not a QAT experiment and not yet an end-to-end
gradient attribution experiment.

## Experimental Contract

- Use the official CSPN ResNet-18 architecture with 24 propagation steps and
  its converged NYU checkpoint.
- Use the current 128-sample stratified train calibration set and the fixed
  64-sample validation set.
- Read the calibration indices, evaluation indices and sampling seed from the
  current stratified-set metadata; do not regenerate any of them.
- Evaluate only FP32 and the existing propagation-aware `PA_W8A8` contract.
- Reuse the deployed quantizers and frozen calibration ranges without defining
  a second W8A8 policy in the diagnostic code.
- Ordinary Conv2d weights use the exact deployed signed W8 weight QDQ.
- Ordinary Conv2d activations use the exact deployed A8 QDQ, including the
  existing signed/unsigned semantic ownership.
- CSPN propagation retains the existing A8 affinity, offset, confidence and
  state treatment with INT16 Q13 coefficients and INT32 accumulation.
- Do not enable rotation, QAT, SmoothQuant, grouping experiments or FP4.
- Do not change inference tensors, predictions or accuracy metrics while
  collecting diagnostics.

The run manifest records the checkpoint digest, sample indices, module
geometry, quantizer type, bit width, signedness, scale shape and propagation
contract. A missing required field or an unsupported convolution raises an
error; there is no fallback quantizer or alternate module path.

## Im2Col Representation

For each ordinary `torch.nn.Conv2d` with `groups == 1`, unfold its input with
the module's exact kernel size, dilation, padding and stride:

\[
X_{col} \in \mathbb{R}^{B \times K \times M}, \qquad
W_{col} \in \mathbb{R}^{C_{out} \times K}.
\]

The K index is decoded explicitly as:

\[
k \leftrightarrow (c, r, s),
\quad c \in [0,C_{in}),
\quad r \in [0,K_h),
\quad s \in [0,K_w).
\]

Both FP activation `X` and the exact activation-QDQ result `Q_A(X)` are
unfolded. Both FP weight `W` and the exact weight-QDQ result `Q_W(W)` are
reshaped with the same K ordering. Tests compare this ordering against direct
Conv2d output on asymmetric kernels, strides and padding.

ConvTranspose2d, Linear, grouped convolution and propagation kernels are not
silently treated as Conv2d. They are outside this first experiment and are
listed as excluded module types in the manifest.

## Streaming Statistics

The implementation must not persist full `M x K` tensors. Each batch is
processed in bounded spatial chunks and accumulated in float64 counters.
Bounded deterministic samples are retained only for percentile estimates and
plotting, and their capacity is recorded in the run manifest.

### Channel and Kernel-Offset Grid

For every `(input_channel, kernel_offset)` cell, aggregate over batch and
spatial patches:

- activation RMS, mean absolute value, p75, p99, p99.9 and maximum absolute
  value;
- activation reference-zero rate, quantized-zero rate and new-zero rate;
- activation signal energy, quantization-error energy and SQNR;
- activation clipping, rounding and zero-collapse error energy;
- weight RMS, mean absolute value and maximum absolute value across output
  channels;
- weight signal energy, W8 error energy and SQNR across output channels.

The resulting table has one row per `module x channel x kh x kw`. Channel-only
and offset-only tables are reductions of this grid, not separately sampled
measurements.

### Spatial Tokens

For every output patch `(sample, hout, wout)`, aggregate over K:

- FP patch RMS, p99 absolute value and maximum absolute value;
- A8 patch error energy, new-zero count and saturation count;
- exact local Conv output error

\[
E_Y(m)=\left\|
Q_A(X_{col,m})Q_W(W)^T-X_{col,m}W^T
\right\|_2^2.
\]

This pre-output-QDQ local error includes the joint W8 and A8 effect for the
Conv operation. Bias is excluded from the difference because the same deployed
bias is added to both sides. It identifies sensitive patches without requiring
an endpoint gradient; any separate output-activation QDQ remains outside this
metric and is reported by the existing activation diagnostics.

The full per-token table is written as compressed NPZ arrays per module and
sample. CSV output contains bounded summaries and the top-ranked tokens, not
millions of individual rows.

### Layer Ranking

Rank all collected Conv2d sites by:

1. aggregate local Conv output SQNR;
2. p99 token output-error energy;
3. worst `channel@offset` activation SQNR;
4. fraction of activation error caused by zero collapse;
5. channel and token imbalance ratios.

Ranking only selects which layers receive detailed plots. Statistics are
collected and exported for every supported Conv2d site.

## Visual Outputs

Use Arial when available through the repository's existing plotting setup,
with no figure title and with grid lines behind plotted data.

### K-Axis 3D Lines

For each selected layer, generate paired weight and activation figures:

- x axis: input channel;
- y axis: flattened kernel offset `r * Kw + s`;
- z axis: one declared metric;
- one colored line per kernel offset.

Separate panels report activation RMS, activation p99, activation SQNR,
new-zero rate, weight RMS and weight W8 SQNR. Metrics with incompatible units
are never placed on one z axis.

### Spatial Waterfall Lines

Preserve output geometry instead of flattening token position:

- x axis: output column `wout`;
- y axis: output row `hout`;
- z axis: patch RMS, A8 patch error or exact local Conv output error;
- one line per output row, with deterministic row thinning recorded in the
  figure manifest when the output is too dense.

Each selected sample receives matched FP magnitude, quantization error and
local output-error views with identical spatial axes. Plotting reads persisted
metrics and never reruns the model.

## Outputs

The experiment root contains:

- `run_manifest.json`;
- `module_manifest.csv`;
- `channel_offset_metrics.csv`;
- `channel_metrics.csv`;
- `kernel_offset_metrics.csv`;
- `layer_metrics.csv`;
- `top_spatial_tokens.csv`;
- `spatial_tokens/<module>/sample_<index>.npz`;
- `figures/k_axis_3d/<module>/*.png` and matching PDF files;
- `figures/spatial_3d/<module>/*.png` and matching PDF files;
- `figure_manifest.csv`.

Module names are encoded reversibly in paths and retained verbatim in every
table. Existing prediction and evaluation roots are read-only inputs.

## Validation

Unit tests must establish:

- exact `(c, kh, kw)` ordering for Im2Col and flattened weights;
- exact agreement between matrix multiplication and Conv2d output before and
  after QDQ within floating-point tolerance;
- correct statistics for a hand-constructed tensor with known channel,
  spatial and offset differences;
- chunk-size invariance of all exact counters and energies;
- deterministic bounded sampling and top-token ranking;
- explicit rejection of unsupported grouped or transposed convolutions;
- plotting from synthetic persisted metrics without model execution.

The production run is accepted only when:

- all expected ordinary CSPN Conv2d modules appear exactly once in the module
  manifest;
- all exported values are finite except mathematically valid infinite SQNR for
  zero error;
- channel and offset reductions reproduce layer totals;
- aggregate local Conv output errors match direct reference calculations on a
  deterministic validation subset;
- FP32 and PA-W8A8 depth metrics remain unchanged by diagnostic capture;
- no full `M x K` tensor or unbounded activation cache is written to disk.

## Deferred Work

After this distribution experiment identifies high-risk layers and patches, a
separate experiment may add endpoint sensitivity
`|Delta X_col * dL/dX_col|` for only those sites. W4A4, QAT and mixed-bit
allocation remain outside this design so that the first result cleanly answers
how W8A8 Conv structure is distributed across M and K.
