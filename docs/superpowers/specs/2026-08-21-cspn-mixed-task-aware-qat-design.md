# CSPN W4-Dominant Mixed-Activation Task-Aware QAT Design

## Objective

Implement a CSPN-only mixed-precision quantization-aware training workflow
that protects the task-sensitive paths identified by P3/T3 while reducing the
activation-element-weighted average precision to at most 6 bits. The primary
quality target is RMSE at or below `0.175 m` on the fixed 64-sample NYU
evaluation set.

The experiment tests whether P3/T3's accuracy advantage over uniform W6A6 is
caused by preserving a small set of task-critical paths rather than increasing
precision uniformly. The first implementation changes activation allocation
and the QAT objective only. It keeps the P3/T3 W4/W8 weight assignment fixed so
that weight and activation effects remain separable.

## Scope

This work covers the official CSPN ResNet-18 model with 24 propagation steps.
It extends the existing task-sensitive bit assignment and Group-8 QAT paths
without changing the official model structure or the existing PTQ, BRECQ,
QDrop, propagation-aware, and strict W4A4 workflows.

The activation candidate set is `{4, 6, 8}`. Weight bits are inherited exactly
from P3/T3 and remain W4-dominant with W8 protection at the existing sensitive
sites. The first experiment does not search W2, W3, W6, FP4, non-uniform
quantization, dual-region activation codes, channel rotation, SmoothQuant, or
new activation owners.

Guidance remains FP32. Propagation retains the existing A8 affinity and state,
INT16 Q13 coefficients, INT32 accumulation, and exact sparse-depth anchors.
There is no automatic precision promotion, FP fallback, or numerical recovery
path.

## Alternatives

Three architectures were considered:

1. Seed from P3/T3, demote A8 sites to A6 or A4 under a strict activation
   budget, and then run mixed-bit task-aware QAT.
2. Train uniform W4A6 first and promote or demote selected sites afterward.
3. Learn soft A4/A6/A8 gates jointly with QAT and discretize them after
   training.

The first architecture is selected. It preserves the strongest measured CSPN
assignment, isolates activation precision as the experimental variable, and
keeps training and deployment on the same explicit discrete bit assignment.
The uniform approach discards known sensitivity structure. The differentiable
search approach adds optimization and discretization ambiguity without first
establishing whether the simpler measured demotion strategy works.

## Architecture

The workflow is:

```text
official checkpoint
  -> folded official CSPN model
  -> fixed P3/T3 W4/W8 and A4/A8 assignment
  -> 128-sample train-calibration A8-to-A6/A4 candidate measurements
  -> activation-budget-constrained assignment selection
  -> mixed-bit hard-forward QAT
  -> canonical checkpoint export
  -> fresh-model hard deployment evaluation
```

The implementation extends the existing modules instead of adding a parallel
quantization framework:

- `spn_quant/cspn_task_sensitive_bits.py` owns assignment validation, cost
  accounting, candidate demotion, and deterministic selection.
- `spn_quant/qat/quantizers.py` provides signed and unsigned STE quantizers for
  4, 6, and 8 bits while preserving hard-forward parity.
- `spn_quant/qat/cspn.py` applies per-module weight bits and per-owner
  activation bits from an explicit assignment.
- The existing Group-8 training and evaluation scripts are extended with a
  separate mixed-task-aware mode. Existing static and dynamic W4A4 commands
  and outputs remain unchanged.

The mixed assignment is a required serialized input to training, resume, and
evaluation. Every listed weight module and activation owner must have exactly
one bit value. Missing, duplicate, unsupported, or extra sites are errors.

## Precision Budget

Activation precision is measured using the existing activation-element cost
basis:

\[
\bar{b}_A = \frac{\sum_i b_i N_i}{\sum_i N_i},
\]

where `b_i` is the assigned activation bit width and `N_i` is the measured
number of activation elements at owner `i` under the fixed NYU input shape.
Layer-count averages and MAC-weighted activation averages are not acceptance
metrics.

Every searched, trained, exported, and evaluated assignment must satisfy
`average_activation_bits <= 6.0`. The auditor recomputes the numerator and
denominator from the serialized assignment and cost basis. Stored summary
values are reporting fields, not trusted inputs.

Weight reporting includes average bits by weight elements and the W8 fraction
by weight MACs. Because the P3/T3 weight assignment is fixed, those quantities
must remain identical across P3/T3 and mixed-QAT configurations.

## Assignment Search

Search starts from the existing P3/T3 assignment. Only A8 owners are eligible
for demotion. For each eligible block, the search measures A8-to-A6 and
A8-to-A4 candidates while weights and all other activation owners remain
unchanged.

Candidate ranking uses measured pooled calibration metrics in this order:

1. finite predictions and zero nonpositive prediction ratio;
2. pooled depth RMSE;
3. boundary RMSE;
4. final propagation-state MSE;
5. lower activation-element-weighted average bits;
6. stable block and owner order.

The selector builds assignments with A4, A6, and A8 and retains only those
that meet the 6-bit activation budget. Candidate effects are remeasured jointly
before selection; single-block deltas are not assumed to be additive. The
fixed 64-sample evaluation set is not read by search code.

## Quantization Contract

### Weights

Each eligible convolution uses the P3/T3-assigned W4 or W8 precision. Both use
signed symmetric quantization with one scale per output channel and an FP32
master weight for optimization. ConvTranspose2d uses the existing output
channel axis contract. Export writes canonical official-model parameter names;
QAT parametrization keys cannot appear in the deployment checkpoint.

### Ordinary Activations

The ordinary activation owner manifest is identical to the current strict
CSPN QAT manifest.

- ReLU outputs use unsigned A4, A6, or A8.
- Signed inputs and outputs use symmetric A4, A6, or A8.
- Contiguous Group-8 granularity is retained.
- Structural tensor-granularity exceptions retain their existing granularity.
- Static calibrated ranges are used for the first mixed-task-aware experiment.

The forward values must match the corresponding hard round, clamp, and
dequantize path exactly. STE changes gradients only.

### Decoder Merges

Upsample and skip branches retain independent observers and scales. Each branch
is quantized according to its own owner assignment before explicit merge
requantization. Add and concat sites cannot infer or share a range from the
other branch.

### CSPN Propagation

Guidance remains FP32 and cannot appear in the generic activation owner
manifest. Raw affinity is quantized to A8 before normalization. Neighbor
coefficients are converted to signed INT16 Q13 and the center coefficient is
recomputed from quantized neighbors. Propagation state is A8, accumulation is
INT32, and sparse-depth anchors are restored after every iteration.

The QAT propagation proxy follows all 24 official iterations and provides
gradients while the forward result remains the existing hard integer result.

### BN and Bias

Batch normalization is folded and frozen before calibration and QAT. Bias uses
FP32 exactly as in the current P3/T3 and strict Group-8 QAT contracts and is
not assigned an independent search bit width. The implementation does not fold
BN after calibration.

## Task-Aware QAT Loss

Training minimizes:

\[
L = L_{depth} + \lambda_b L_{boundary}
  + \lambda_t L_{teacher} + \lambda_p L_{propagation}.
\]

- `L_depth` is masked L1 against valid NYU ground-truth depth.
- `L_boundary` is masked L1 over a deterministic GT-derived depth boundary
  mask.
- `L_teacher` is masked L1 between the quantized prediction and the frozen
  official FP32 CSPN prediction for the same augmented input.
- `L_propagation` is the mean masked L1 between matching quantized-proxy and
  FP32 propagation states across the 24 iterations.

All four coefficients are required configuration fields and are accessed
directly. The training code contains no default coefficient values. The frozen
teacher runs in evaluation mode and receives no gradients.

The first formal run uses static Group-8 activation scales. Dynamic scales are
not combined with mixed precision until the static experiment passes its
budget, numerical, and hard-parity gates.

## Data Isolation

- The fixed stratified 128-sample NYU train set calibrates activation ranges.
- Assignment search uses only predictions and GT from the fixed stratified
  128-sample NYU train calibration set.
- QAT uses the official training split with the calibration identities allowed
  as ordinary training samples after calibration is frozen.
- Early stopping uses official validation identities excluding the fixed 64
  evaluation samples.
- The fixed 64 samples are evaluated only after assignment selection and QAT
  checkpoint selection are complete.

Every output records digests for the checkpoint, calibration manifest,
early-stopping manifest, fixed-64 manifest, owner manifest, assignment, and
cost basis.

## Training and Checkpointing

Training starts from the same official checkpoint for every comparison. The
optimizer, scheduler, maximum epochs, patience, gradient clipping, random seed,
and loss coefficients are required configuration values. Selection uses pooled
validation RMSE with finite and positive prediction gates.

`last.pt` and `best.pt` contain the canonical FP32 master weights, optimizer and
scheduler state, epoch, convergence tracker, assignment, cost basis digest,
owner manifest digest, data-manifest digests, and complete QAT configuration.
Resume requires exact equality of all contracts. A mismatch raises an error.

## Evaluation

Final evaluation loads each checkpoint into a fresh official CSPN model and
installs only the hard deployment quantizers. It evaluates these configurations
on identical samples:

```text
FP32
UNIFORM_W6A6
P3_T3
MIXED_TASK_AWARE_QAT
```

Reported metrics include RMSE, MAE, AbsRel, iRMSE, flat RMSE, boundary RMSE,
nonfinite ratio, nonpositive ratio, activation SQNR, new-zero ratio, saturation
ratio, per-step propagation error, anchor maximum error, Q13 coefficient-sum
error, contraction violations, average activation bits, average weight bits,
W8 weight-element fraction, and W8 weight-MAC fraction.

Prediction artifacts contain aligned RGB, sparse depth, GT, FP32, uniform
W6A6, P3/T3, mixed-QAT, and absolute-error maps for the fixed 64 samples.
Runtime artifacts remain under `profile_logs/` and are not committed.

## Error Handling and Coding Rules

Configuration objects use direct attribute access. Dictionaries use direct
indexing. Required configuration and metadata fields have no code-level
defaults. Configuration, calibration, checkpoint, manifest, numerical, and
hard-parity errors propagate immediately.

Non-finite inputs, scales, losses, gradients, parameters, predictions, or
propagation states are errors. Invalid bit assignments, budget violations,
nonpositive final predictions, anchor violations, and coefficient-sum
violations cannot trigger a precision promotion or FP fallback.

## Verification

Unit tests cover:

- signed and unsigned A4/A6/A8 code domains and gradients;
- per-module W4/W8 hard-forward weight parity;
- complete per-owner assignment validation;
- activation-element-weighted budget calculation and the `<= 6.0` gate;
- deterministic A8-to-A6/A4 candidate generation and ranking;
- independent decoder branch scales and explicit requantization;
- task loss terms, masks, teacher detachment, and propagation-state alignment;
- checkpoint, resume, and canonical export contracts.

Integration tests cover:

- unchanged existing W4A4 QAT behavior when mixed mode is not selected;
- one complete mixed-bit optimization step with finite gradients;
- QAT forward versus fresh-model hard-path parity;
- guidance exclusion and exact 24-step propagation invariants;
- rejection of evaluation-manifest use during search or early stopping;
- 64-sample output identity and prediction coverage.

Before formal training, run the complete repository test suite and a short CUDA
smoke run that calibrates, searches a restricted candidate set, performs one
QAT epoch, exports a checkpoint, and reloads it through the hard evaluator.

## Acceptance Criteria

The formal result is accepted only when all conditions hold:

- activation-element-weighted average precision is at most `6.0` bits;
- fixed-64 pooled RMSE is at most `0.175 m`;
- nonfinite and nonpositive prediction ratios are zero;
- sparse-anchor maximum error is zero;
- Q13 coefficient-sum maximum error is zero;
- contraction violation ratio is zero;
- QAT and exported hard-path predictions satisfy the parity tolerance defined
  by the existing hard quantizer tests;
- P3/T3 weight assignment and weight-cost statistics remain unchanged;
- existing PTQ, strict W4A4 QAT, BRECQ, and QDrop tests remain green.

Failure to meet the quality target is reported as a negative experimental
result. It does not change the bit budget, promote failing sites, or relax the
propagation contract automatically.
