# Hardware-Aligned Depth Quantization Design

## Objective

Add a second PTQ path that follows a standard integer backend contract for the
official CSPN, DySPN, NLSPN, and CompletionFormer models. Preserve the existing
hook-based RTN results as the unfused baseline and report hardware-aligned
W4A4 and W4A8 separately.

## Quantization Contract

- Fold every executed `Conv2d -> BatchNorm2d` pair while the model is in eval
  mode, before collecting activation ranges.
- Quantize Conv2d and Linear weights per output channel with signed symmetric
  ranges: W4 `[-7, 7]`, W8 `[-127, 127]`.
- Quantize post-ReLU activations per tensor with unsigned ranges: A4 `[0, 15]`,
  A8 `[0, 255]`.
- Quantize signed activations per tensor with symmetric ranges: A4 `[-7, 7]`,
  A8 `[-127, 127]`.
- Use deterministic static MinMax calibration on the same 128 training samples
  as the existing experiment. Zero is always representable.
- Quantize Add inputs with their producer scales and requantize the Add result
  to one output scale, matching the usual integer Add contract. Requantize a
  Concat result to one common per-tensor scale before its consuming integer op;
  standard Concat therefore does not retain channel-specific activation scales.
- Quantize folded bias to INT32 with `s_bias[o] = s_x * s_weight[o]`. QDQ
  execution uses the dequantized INT32 bias so it remains comparable on CUDA.
- Keep softmax, normalization, interpolation, deformable sampling, SPN/LSPN
  propagation, and other unsupported operators as explicit floating-point
  islands.

## Graph Alignment

The implementation discovers Conv-BN producer-consumer pairs from one dry
forward pass and replaces them with a folded Conv plus Identity BN. Call-indexed
`_concat` helpers own one observer and output quantizer per decoder stage.
Direct `torch.cat` calls are requantized by the consuming Conv/Linear input QDQ,
which is numerically the same common-output-scale contract. Residual inputs keep
their producer scales and the explicit post-Add ReLU/output boundary supplies
the Add output scale.

The adapter emits a manifest containing folded pairs, explicit `_concat` sites,
activation signedness, integer ranges, and bias scales. Metadata separately
records the direct-Concat and Add requantization contracts so an explicit merge
count of zero is not interpreted as an unquantized decoder.

## Experiment

- Reuse the fixed 64 NYU validation indices and converged checkpoints.
- Run `HW_W4A8_full` first, then `HW_W4A4_full` after numerical preflight.
- Export sample, regional, signal, layer, nonfinite, and prediction artifacts
  without replacing the existing RTN rows.
- Compare FP32, existing W8A8/W4A8/W4A4, and hardware-aligned configurations.

## Acceptance Criteria

- Folded FP32 output differs from original FP32 by less than `0.05 m` maximum;
  the exact maximum is persisted so model sensitivity to folded-weight rounding
  remains visible. This threshold rejects structural errors while allowing
  accumulated FP32 rounding in iterative propagation. Conv-BN pairs with a
  pre-BN fan-out are kept explicit and recorded because folding them would
  change the official graph.
- Unit tests verify signed/unsigned code ranges, merge shared scale, fold-before-
  calibration ordering, and INT32 bias scale.
- All four models record 64 rows per hardware-aligned configuration.
- No configuration silently skips an expected Conv-BN pair; explicit and
  consuming-op merge boundaries are both documented.
- The report distinguishes quantization accuracy simulation from measured
  integer-kernel latency.

## Non-Goals

- Implementing packed INT4 CUDA kernels.
- Retaining independent branch scales with split partial convolutions.
- Quantizing custom deformable sampling or SPN propagation kernels.
- QAT or mixed-precision search in this phase.
