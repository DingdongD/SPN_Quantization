# CompletionFormer Joint Integer Quantization Design

## Objective

Implement and evaluate a deployable CompletionFormer W4A4 path that jointly
optimizes attention and CNN/Transformer concat boundaries. The implementation
must preserve the official model architecture, expose every non-float boundary
in a manifest, and separate scale optimization gains from weight-rounding or
training-based compensation.

The first implementation keeps RTN W4 weights fixed. AdaRound, BRECQ, and QAT
are outside this experiment so that changes can be attributed to integer data
flow and scale selection.

## Evaluation Contract

- Use the official CompletionFormer model and the converged NYU checkpoint
  already recorded by the strict evaluation metadata.
- Calibrate and evaluate on the same deterministic 64 NYU samples used by the
  existing W4A4 histogram run.
- Do not quantize ground truth.
- Compare FP32, existing RTN W4A4, attention-only optimization, concat-only
  optimization, joint attention+concat optimization, and W4A8.
- Reject non-finite tensors, missing quantization sites, incompatible shapes,
  or incomplete manifests. There is no float fallback for an enabled integer
  boundary.

## Selected Architecture

### Attention Data Flow

The official `Attention` implementation remains unchanged on disk. A
CompletionFormer-specific runtime adapter reproduces its forward data flow
using the model's existing `q`, `kv`, `sr`, `norm`, and `proj` modules. Adapter
installation validates module type, tensor shape, head count, and FP bypass
identity before quantization is enabled.

Q, K, and V use signed A4 codes with independent per-head scales. Four-bit
codes are unpacked into INT8 arithmetic lanes; this does not increase their
precision. Both matrix multiplications use INT32 accumulation:

```text
score_int32 = q_int8 @ transpose(k_int8)
score_fp16 = score_int32 * (s_q_head * s_k_head * head_scale)
probability_fp16 = softmax(score_fp16)
probability_u8 = requantize(probability_fp16, unsigned A8)
context_int32 = probability_u8 @ v_int8
context = context_int32 * (s_probability * s_v_head)
```

Softmax normalization remains FP16. Its output is quantized to unsigned A8
before the probability-by-V multiplication. Attention score tensors and
softmax probabilities are recorded separately; they are no longer inferred
from Linear input/output statistics.

The fused `kv` projection is not quantized with one shared output scale. It is
split into K and V first, then K and V are calibrated and quantized
independently. Q, K, and V scales are constant across each head's reduction
dimension so each dot product has one valid accumulator scale.

### Concat Data Flow

Each official `concat_conv` receives a `2C` tensor formed by concatenating the
Transformer and CNN branches. A consumer-side adapter splits this tensor into
the first and second `C` channels, so the official `torch.cat` call does not
need to be modified.

The branches retain independent signed A4 scales. The following convolution is
evaluated as two integer partial convolutions over the corresponding weight
slices:

```text
acc_transformer = conv_int32(x_transformer, weight[:, :C])
acc_cnn = conv_int32(x_cnn, weight[:, C:])
acc_output = requantize(acc_transformer, s_accumulator)
           + requantize(acc_cnn, s_accumulator)
output = requantize(acc_output + bias_int32, s_output)
```

This avoids collapsing the lower-range branch into a single shared A4 input
scale. The two partial accumulators are requantized to one explicitly selected
accumulator scale before addition. Bias is represented in that accumulator
domain. The existing per-input-channel fake-QDQ path remains available only as
an analysis upper bound and is not labeled as a standard integer result.

### Joint Scale Optimization

Calibration uses two paired passes over the same deterministic calibration
indices. The first pass disables every quantizer and stores each Attention
context and `concat_conv` output as an FP target. The second pass enables the
ordinary W4A4 and semantic A8 boundaries, then stores the actual W4A4-flow
Q/K/V and concat inputs together with their paired FP targets. Pairing is
strict by module, sample, and call order; missing or extra calls raise.

The reconstruction cache is bounded and contains no evaluation samples. Scale
candidates are deterministic clipping multipliers around the observed W4A4
flow range. Coordinate search minimizes normalized block-output MSE against
the paired FP target, not local tensor SQNR alone.

Attention optimization selects per-head Q, K, and V scales plus the unsigned
probability scale. Concat optimization selects the two branch scales, common
partial-accumulator scale, and output scale. Candidate selection must not use
NYU ground truth or evaluation samples.

Optimization order is:

1. Freeze W4 RTN weights and all ordinary activation quantizers.
2. Optimize attention blocks from shallow to deep.
3. Optimize concat blocks from shallow to deep using the selected upstream
   attention scales.
4. Run one joint validation pass. Do not iteratively tune on evaluation RMSE.

## Components

### Integer Attention Runtime

A focused module owns per-head observers, signed A4 Q/K/V quantizers,
unsigned A8 probability quantization, INT32 reference kernels, metrics, and
manifest serialization. It has explicit `observe`, `freeze`, `quantize`, and
`disable` phases matching existing quantization controllers.

### Split Concat Runtime

A focused module owns branch splitting, branch observers, W4 weight slicing,
partial INT32 convolution, accumulator requantization, bias conversion, output
quantization, and manifest serialization. Unsupported convolution parameters
fail during installation.

### CompletionFormer Adapter

The adapter discovers exactly 16 Attention modules and 16 `concat_conv`
modules. Counts are part of the contract. It installs the two runtimes without
editing the external CompletionFormer checkout and restores original forwards
when closed. Its target queues validate exact consumption during the paired
reconstruction pass so calibration cannot silently cross module or sample
boundaries.

### Runner and Reports

The NYU runner adds explicit attention-only, concat-only, and joint
configurations. It writes:

- `attention_integer_manifest.csv`
- `concat_integer_manifest.csv`
- `joint_scale_search.csv`
- `attention_metrics.csv`
- `concat_metrics.csv`
- `sample_metrics.csv`
- `regional_metrics.csv`
- prediction arrays and one comparison figure containing GT, FP32, existing
  W4A4, attention-only, concat-only, joint W4A4, and W4A8.

## Metrics

End-to-end metrics include RMSE, MAE, inverse RMSE, and delta RMSE from FP32.
Attention metrics include Q/K/V SQNR, score SQNR, probability KL divergence,
probability zero ratio, probability saturation ratio, and context-output MSE.
Concat metrics include branch SQNR, branch zero ratio, branch saturation ratio,
partial-accumulator requantization error, output SQNR, and block-output MSE.

All aggregate metrics are accompanied by per-sample rows. Local error energy is
reported separately from end-to-end sensitivity.

## Correctness Requirements

- A4 signed codes use the configured signed integer range; probability A8 is
  unsigned.
- Q/K/V scales are independent by tensor role and head.
- K and V never share one frozen scale.
- Every QK and probability-V product accumulates in INT32 in the reference
  implementation.
- Every concat partial sum declares its source scale and target accumulator
  scale.
- Bias scale matches the common concat accumulator domain.
- Quantized tensors and manifests contain no NaN or Inf.
- The adapter's disabled path matches the unmodified official model within the
  existing FP identity tolerance.

## Testing Strategy

Unit tests verify integer code ranges, per-head scale shapes, K/V separation,
INT32 accumulator identities, unsigned probability quantization, branch split
ordering, partial-convolution equivalence, accumulator requantization, bias
scale, state transitions, and fail-closed validation.

Integration tests use small official-shape Attention and concat blocks to
verify adapter installation, bypass identity, manifest completeness, and
attention-only/concat-only/joint configuration isolation. Runner contract tests
verify deterministic 64-sample indices and required outputs.

The final evaluation is accepted only if all tests pass, every configured site
has 64 updates, all outputs are finite, and the joint W4A4 RMSE is measured on
the fixed evaluation set. An accuracy improvement is a measured result, not a
precondition silently enforced by selecting evaluation-driven scales.

## Non-Goals

- Quantizing softmax normalization itself to A4.
- Modifying or retraining the official CompletionFormer model.
- Updating W4 rounding decisions during this first experiment.
- Claiming native CUDA kernel speedup from the integer reference path.
- Applying CompletionFormer-specific attention logic to CSPN, DySPN, or NLSPN.
