# Selected Quantization Adaptation for DySPN, NLSPN, and CompletionFormer

## Goal

Adapt the retained quantization methods to the official DySPN, NLSPN, and
CompletionFormer implementations under one strict NYU depth-completion
protocol. The selected method matrix is:

- RTN W8A8;
- RTN W4A4;
- QDrop W6A6;
- BRECQ W6A6;
- HAWQ mixed precision with average weight and activation precision no greater
  than 6 bits;
- LSQ++ W6A6 and W4A4;
- mixed task-aware QAT;
- model-specific P3/T3 mixed PTQ.

The work reuses shared quantizers and evaluation code, but does not copy CSPN
module names, protected sites, or bit assignments into another architecture.

## Scope

The target models are the repository's official integrations:

| Model | Required implementation | Propagation boundary |
| --- | --- | --- |
| DySPN | Official DySPN network and DCNv2 extension | DySPN offset, affinity, confidence/gate, anchors, and iterative state |
| NLSPN | Official NLSPN network and deformable propagation extension | NLSPN offset, affinity, confidence, anchors, and iterative state |
| CompletionFormer | Official full CompletionFormer network | Transformer attention/concat contracts plus NLSPN-style propagation |

Existing converged checkpoints, official tensor shapes, preprocessing,
propagation iteration counts, and CUDA execution paths remain unchanged. PTQ
methods do not retrain the source model. QAT methods start from the same
official checkpoint and produce method-specific checkpoints.

This work does not add FP4, LogNP, rotation, SmoothQuant, OCI/OCS, or discarded
historical experiments.

## Architecture

The implementation uses one method engine with explicit model contracts:

1. A shared experiment schema defines method, precision, calibration,
   optimization, evaluation, and artifact fields.
2. A model contract enumerates ordinary quantizable modules, activation
   owners, reconstruction blocks, protected semantic signals, propagation
   invariants, and mixed-precision search groups.
3. Shared RTN, QDrop, BRECQ, HAWQ, LSQ++, mixed-QAT, metric, and artifact code
   consumes the contract.
4. Each run validates the contract against the loaded official model before
   calibration or training starts.

Missing modules, unsupported operators, absent CUDA extensions, mismatched
checkpoints, incomplete calibration records, and invalid bit assignments are
errors. There is no backend, model, precision, or operator fallback.

## Quantization Contracts

### Ordinary CNN Layers

Conv2d weights use signed symmetric per-output-channel quantization. Activation
ownership follows the existing hardware-aligned edge graph so that one tensor
is quantized once at its producer-consumer boundary. Signed activations use
symmetric quantization; non-negative post-ReLU activations use unsigned
quantization. Add and concat edges requantize each input independently before
the merge. Bias remains FP32 where a unique integer bias scale cannot be
defined; otherwise integer bias uses the input scale multiplied by the
per-output-channel weight scale.

### DySPN

The ordinary encoder and decoder convolutions are eligible for the selected
methods. The following remain outside generic CNN quantization:

- DCNv2 sampling coordinates and grid construction;
- offset and affinity logits;
- normalized affinity;
- confidence/gate tensors;
- sparse-depth anchors;
- iterative propagation state and reductions.

The existing propagation-aware adapter owns these signals. Affinity is
quantized before normalization, coefficients use the established fixed-point
contract, state uses the configured protected precision, and accumulation uses
INT32. The official DCNv2 CUDA extension is mandatory.

### NLSPN

The ordinary ResNet encoder, decoder, and depth-estimation convolutions are
eligible. The propagation projection, offsets, affinity, confidence,
sparse-depth anchors, deformable sampling, and iterative state are owned by the
NLSPN propagation contract. Quantized affinity is normalized before use, the
center coefficient is derived from quantized neighbors, and anchors are
restored according to the official model behavior. The official deformable
convolution extension is mandatory.

### CompletionFormer

CNN edges, transformer projections, attention matrix multiplications, and
decoder merge edges have distinct owners.

- Q, K, and V projections use the method's configured weight and activation
  precision.
- QK and AV operands use explicit attention-edge quantizers with INT32
  accumulation where the integer contract applies.
- softmax, normalization, positional/deformable sampling metadata, and
  denominators remain in FP16 or FP32 according to the official path.
- concat inputs are quantized independently and requantized to the consumer
  scale; a shared pre-concat scale is forbidden.
- confidence, offset, affinity, anchors, and propagation state remain under the
  propagation adapter rather than generic attention or CNN quantization.

The full official CompletionFormer architecture is required. Tiny or reduced
variants are rejected.

## Method Matrix

### RTN

RTN W8A8 and W4A4 use static calibration and deterministic nearest rounding.
They establish the uniform PTQ baselines and do not perform optimization.

### QDrop W6A6

QDrop uses the existing official-style joint weight and activation
reconstruction. Reconstruction blocks come from the model contract. Protected
propagation signals are excluded. The hard deployment graph is evaluated after
reconstruction; fake-quant reconstruction output is not reported as the final
result.

### BRECQ W6A6

BRECQ uses deployment-aligned block reconstruction with the same block
partition and activation ownership as RTN. Its optimized rounding parameters
must be materialized into the hard quantized weights before formal evaluation.
Non-finite block outputs or predictions fail the run.

### HAWQ Mixed Precision

HAWQ traces model-specific ordinary blocks and ranks them with a Hessian-based
weight sensitivity score. Protected propagation and normalization operators
are excluded. Candidate weight and activation precisions are selected under
separate average-bit budgets no greater than 6 bits. The allocation report
records weighted average bits by parameter count, weight MACs, and activation
traffic.

All three models use the same explicitly declared central-finite-difference
generalized Gauss-Newton block trace. The converged depth checkpoints are not
local minima of the auxiliary masked MSE plus boundary MSE curvature loss, so
their full task-loss Hessians can be indefinite. DySPN's deformable-convolution
path also does not provide the required second derivative. The GGN quadratic
form is the exact output curvature of the configured masked depth MSE and
boundary MSE. It is positive-semidefinite by construction and is not a runtime
fallback; its mode and finite-difference epsilon are persisted and verified
with every artifact.

### LSQ++ W6A6 and W4A4

LSQ++ learns weight and activation scales from the official checkpoint. Scale
parameters are attached to contract-owned quantization edges. Guidance,
confidence, offset, affinity, normalization, anchor, and propagation state
cannot acquire generic LSQ++ activation scales.

### P3/T3 Mixed PTQ

P3/T3 is searched independently for each model. The labels describe positions
on that model's Pareto search and do not identify fixed CSPN layers.

The search starts from RTN W4A4 and evaluates:

1. individual W8A8 promotions for every eligible semantic block;
2. cumulative encoder prefixes;
3. decoder, initial-depth, attention, and pre-propagation tail groups;
4. interactions between the best prefix and tail groups;
5. Pareto ranking by pooled RMSE and normalized precision cost.

`P3` is the smallest stable prefix knee selected from the model's Pareto front.
`T3` is the best stable tail combination at the approved budget. A candidate is
stable only when all predictions are finite, propagation invariants pass, and
its paired 64-sample result is reproducible.

### Mixed Task-Aware QAT

Mixed task-aware QAT starts from the model-specific P3/T3 assignment. Weight
precision remains W4-dominant with W8 protection at sensitive sites.
Activation precision is selected from A4, A6, and A8 under an average
activation budget no greater than 6 bits. The task objective combines masked
depth loss, FP32 teacher loss, and model-specific pre-propagation or
propagation-state consistency. The final checkpoint is evaluated through the
same hard deployment path as PTQ.

## Calibration and Evaluation

Each model uses a train-only, 128-sample stratified calibration set. Selection
uses raw NYU descriptors and model-specific FP32 activation descriptors:

- GT depth mean, p50, p95, and maximum;
- RGB luminance, contrast, and edge density;
- sparse-depth spatial coverage;
- encoder stem, decoder fusion, initial-depth, and key signed-activation p99,
  maximum, and channel imbalance;
- CompletionFormer attention projection descriptors.

Calibration identities are persisted and cannot overlap the validation split.
All methods for one model use the same calibration identities.

Formal evaluation uses one fixed set of 64 NYU validation samples per model.
All methods for that model use identical identities, preprocessing, masks, and
FP32 references. Reports include:

- pooled RMSE;
- mean per-sample RMSE;
- pooled MAE, AbsRel, and iRMSE;
- absolute and percentage degradation relative to FP32;
- finite, non-positive, saturation, and zero-code ratios;
- per-block output error and SQNR;
- initial-depth and per-iteration propagation error;
- weight and activation average-bit costs and W4/W6/W8 shares.

Pooled RMSE is the primary accuracy objective. Mean per-sample RMSE is retained
as a diagnostic and is always labeled explicitly.

## Artifacts

Each model receives an isolated result directory containing:

- the immutable experiment configuration and model contract manifest;
- calibration and evaluation sample identities;
- aggregate and per-sample metric CSV files;
- block, attention, and propagation diagnostics;
- HAWQ and P3/T3 allocation tables with normalized cost;
- method checkpoints for QDrop, BRECQ, LSQ++, and mixed-QAT;
- aligned RGB, sparse depth, GT, FP32, and selected quantized predictions;
- a cross-method summary generated only from completed formal runs.

Historical result directories are not read as inputs. Cleanup of obsolete
artifacts occurs only after the new formal matrix passes validation.

## Execution

The three models may run concurrently on separate idle GPUs. A single model's
calibration, search, training, and formal evaluation remain sequential when
they share mutable state or checkpoints. GPU assignment is explicit in the
launch manifest. CompletionFormer and NLSPN use the CUDA-compatible environment
that contains their official extensions; DySPN uses its validated extension
environment.

The implementation sequence is:

1. generalize the experiment schema and model-contract interface;
2. add strict DySPN, NLSPN, and CompletionFormer contracts;
3. validate RTN W8A8/W4A4 parity;
4. adapt QDrop and BRECQ W6A6;
5. adapt HAWQ and LSQ++;
6. run model-specific P3/T3 searches;
7. train mixed task-aware QAT from each P3/T3 assignment;
8. run the fixed-64 formal comparison and generate final artifacts.

## Acceptance Criteria

- Every method runs against the official full architecture and expected
  checkpoint without fallback.
- All required CUDA extensions are loaded and exercised.
- Quantization owners are unique and protected semantic signals are excluded
  from generic quantization.
- CompletionFormer attention and concat scale contracts are validated.
- P3/T3 assignments differ by model when sensitivity results require it and
  include their search evidence.
- Every formal prediction is finite and propagation invariants pass.
- All methods use the same per-model calibration and evaluation identities.
- Reported RMSE values identify pooled or per-sample aggregation explicitly.
- Unit, contract, integration, and fixed-sample smoke tests pass before formal
  multi-GPU evaluation starts.
