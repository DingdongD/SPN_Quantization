# CSPN BRECQ W6A6 Deployment Alignment Design

## Goal

Replace the invalid weight-only CSPN BRECQ W6A6 comparison with a strict,
deployment-aligned reconstruction and rerun the fixed 64-sample NYU
evaluation. The repaired result must use the official converged CSPN graph,
the persisted stratified calibration set, and the existing propagation-aware
A8/Q13/INT32 contract.

## Diagnosis

The historical BRECQ runner reconstructed each block from floating-point
teacher inputs and optimized weight rounding only. Deployment subsequently
inserted static A6 activation QDQ at the semantic module boundaries. This made
the reconstruction objective inconsistent with the deployed graph, especially
after quantization error accumulated into the encoder-tail `layer4.1` block.

The strict evaluator marked a sample invalid when any valid-GT pixel was
non-finite or no greater than `1e-4`. Eight of 64 BRECQ W6A6 samples were
invalid, so the aggregate RMSE was correctly reported as infinity. Output
clipping or invalid-value replacement is not an acceptable repair.

## Reconstruction Contract

The implementation reuses the existing joint BRECQ/QDrop block reconstructor.
QDrop remains stochastic with quantization probability `0.5`. BRECQ is the
deterministic endpoint with quantization probability `1.0`, so every optimized
block receives the actual quantized student input and all owned A6 boundaries
remain active during reconstruction.

Both algorithms use:

- W6 per-output-channel symmetric weight quantization;
- static A6 semantic activation boundaries initialized after Conv-BN folding;
- sequential block reconstruction in measured execution order;
- the same 128 train samples and 112/16 reconstruction-validation split;
- the same fixed 64 NYU validation samples;
- FP32 bias and protected sparse-depth boundaries;
- CSPN propagation with A8 affinity/state, Q13 coefficients, INT32
  accumulation, and sparse-anchor restoration.

The BRECQ contract records algorithm identity, probability, precision, sample
hashes, execution order, and exact weight and activation contracts. A BRECQ
contract cannot be loaded as QDrop, and precision mismatches fail directly.

## Evaluation And Diagnostics

The evaluator reports non-finite and non-positive valid-GT pixels separately.
Any affected sample still receives infinite strict RMSE. It also records the
minimum prediction so a finite negative-depth failure is not mislabeled as
floating-point overflow.

The formal comparison contains FP32, PA-RTN W8A8, repaired BRECQ W6A6, and the
retained three-seed QDrop W6A6 result. The primary acceptance criterion is
64/64 valid BRECQ predictions. Accuracy is reported without assuming that the
repair will beat QDrop or W8A8.

## Implementation Boundary

No new quantization framework or Python runner is added. The existing QDrop
reconstruction runner gains an explicit `qdrop` or `brecq` algorithm argument,
and existing joint contract/replay code accepts the matching strict method.
The historical weight-only strict runner remains available for W4A8 studies
but is not used to claim deployment-aligned BRECQ W6A6.

There is no fallback to RTN, QDrop, another seed, output clipping, or sample
dropping. Generated data is written under
`profile_logs/nyu_cspn_brecq_w6a6_aligned_64` outside the Git worktree.

## Verification

Tests cover deterministic all-quantized BRECQ inputs, strict method identity,
deployment-contract replay, separate invalid-depth counters, exact protocol
identity, and rejection of unsupported algorithm/probability combinations.
Focused tests run before the formal reconstruction, followed by the complete
repository test suite and the fixed-64 artifact audit.
