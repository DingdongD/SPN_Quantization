# CSPN LSQ+ and HAWQ Migration Design

## Goal

Migrate LSQ+ and HAWQ to the official NYU CSPN model without changing the
model graph, checkpoint tensors, dataset preprocessing, or 24-step propagation
behavior. Evaluate LSQ+ at W4A4 and W6A6, and evaluate one HAWQ mixed-precision
configuration whose parameter-weighted average weight bits and
activation-element-weighted average activation bits are both no greater than
6.0.

The formal comparison uses the same official checkpoint, stratified 128-sample
training calibration set, held-out validation selection, and ordered fixed-64
NYU evaluation set as the existing CSPN quantization experiments.

## Reference Implementations

The migration is checked against these immutable upstream revisions:

- LSQplus: `ZouJiu1/LSQplus@f26c972c3175a74c0818e09da992feecfe9cc45c`
- HAWQ: `Zhen-Dong/HAWQ@1616df69fdd99f100a8d8a6e78742e0a89292634`

LSQplus is GPL-3.0. Its source is used as a behavioral reference and is not
copied or vendored. The implementation derives the quantization equations from
LSQ/LSQ+ and records the upstream revision in every formal artifact. HAWQ is
MIT-licensed, but the CSPN implementation still uses focused native modules so
that it shares the repository's existing graph, activation-boundary, cost, and
deployment contracts.

## Experiment Matrix

The formal fixed-64 evaluation contains:

| Configuration | Weight policy | Activation policy |
| --- | --- | --- |
| `FP32` | FP32 | FP32 |
| `PA_RTN_W4A4` | Per-output-channel symmetric W4 | Existing strict A4 deployment path |
| `PA_RTN_W6A6` | Per-output-channel symmetric W6 | Existing strict A6 deployment path |
| `LSQPLUS_W4A4` | Learned per-output-channel W4 step | Learned per-site A4 step and offset |
| `LSQPLUS_W6A6` | Learned per-output-channel W6 step | Learned per-site A6 step and offset |
| `HAWQ_MIXED_LE6` | Blockwise W4/W6/W8 | Tied blockwise A4/A6/A8 |
| `MIXED_TASK_AWARE_QAT` | Existing measured assignment | Existing measured assignment |

Historical metrics are not inserted into the formal result. Every row is
replayed from fresh model construction under the same code revision. The
existing mixed-task-aware checkpoint is allowed only after its source
checkpoint, sample manifests, graph contract, and hard deployment contract
match the new run exactly.

## Shared CSPN Contract

All configurations load the official CSPN ResNet18 structure and converged
24-iteration checkpoint with strict key validation. Batch normalization is
folded and frozen before quantizer initialization or training. The fold audit
must pass the existing maximum-error threshold.

Ordinary convolution and transposed-convolution boundaries use the current
semantic owner registry. Decoder upsample and skip branches retain separate
quantizers and separate scales before merge requantization. Bias remains FP32
in QDQ training and is audited against the integer deployment scale
`s_x * s_w,o`; it is not assigned an independent bit width.

Guidance remains FP32 and cannot enter an ordinary activation or weight
assignment. CSPN propagation retains the existing propagation-aware contract:
raw neighbor affinity is quantized before normalization, the center
coefficient is reconstructed from quantized neighbors, coefficients use signed
INT16 Q13, accumulation uses INT32, propagation state uses A8, and sparse-depth
anchors are restored after every iteration. Propagation precision is reported
separately and is excluded from ordinary CNN average-bit budgets.

No method may promote precision automatically, replace invalid values, clip a
prediction for metric computation, fall back to FP execution, or report a
finite-only metric.

## LSQ+ Quantization

LSQ+ uses the V2 min/max initialization behavior because the linked upstream
repository identifies V2 as its strongest implementation. For activation
tensor `x`, step `s > 0`, offset `beta`, and integer range `[Qn, Qp]`, the hard
forward is:

\[
\hat{x} = s\,\operatorname{clamp}
  \left(\operatorname{round}\left((x-\beta)/s\right),Q_n,Q_p\right)+\beta.
\]

ReLU outputs use unsigned codes `[0, 2^b-1]`. Signed boundaries use
`[-2^(b-1), 2^(b-1)-1]`. Each semantic activation owner has one learned scalar
step and one learned scalar offset. Separate decoder branches therefore cannot
share LSQ+ parameters. Sparse depth, guidance, affinity normalization,
confidence, and propagation state are not generic LSQ+ owners.

Weights use signed symmetric quantization, a learned step per output channel,
and no learned offset. ConvTranspose2d follows the existing logical output-axis
contract. FP32 master weights are retained during optimization. Rounding and
clamping use the LSQ+ straight-through gradients and gradient scaling
`1 / sqrt(N * Qp)`. The hard forward must match an independently computed QDQ
reference exactly before training and after checkpoint reload.

Initialization consumes every sample in the ordered stratified 128-sample
calibration manifest. The initialization batch size and number of initialization
batches are required configuration fields and must exactly cover the manifest.
Activation step and offset are initialized from the observed min/max range;
weight steps are initialized per output channel from the upstream V2
mean/standard-deviation rule. Non-positive or non-finite steps are errors.

`LSQPLUS_W4A4` and `LSQPLUS_W6A6` differ only in declared ordinary CNN weight
and activation bits. They use identical training data, task loss, optimizer
family, scheduler, epoch limit, early stopping, and checkpoint selection.

## HAWQ Trace Estimation

The upstream HAWQ repository stores normalized Hutchinson traces as constants
for ImageNet models. Those values are invalid for CSPN and are not reused.
The migration estimates one normalized Hessian trace for every eligible CSPN
quantization block on the stratified 128-sample NYU training set.

The curvature loss is valid-pixel masked depth MSE with an explicit boundary
MSE term. Masked L1 is not used for trace estimation because it has zero second
derivative almost everywhere. Loss coefficients, Hutchinson probe count,
probe seed, calibration batch size, and boundary threshold are required
configuration fields.

For each calibration batch and Rademacher probe, all eligible FP32 weight
blocks are probed in one Hessian-vector product. The estimator records
`v_l^T H_ll v_l / numel(W_l)` for each block. The final trace is the arithmetic
mean over all batches and probes. Individual negative stochastic estimates are
retained and reported. A non-finite or negative final mean trace is an error
and blocks assignment generation; the formal run must increase the configured
probe count and recompute the complete trace artifact.

Trace artifacts include the per-batch/per-probe estimates, mean, standard
error, coefficient of variation, parameter count, block owner, source
checkpoint digest, data manifest digest, and exact estimator configuration.

## HAWQ Mixed-Precision Assignment

Each eligible block selects one value from `{4, 6, 8}`. That value controls the
block's weight quantizer and its corresponding ordinary input/output activation
owners. Residual and decoder merge constraints require producers that enter a
shared integer add to expose compatible output precision, while each pre-merge
branch retains its own scale.

The first RGBD stem and final initial-depth head are fixed at W8A8, following
the upstream HAWQ first/last-layer policy and the measured CSPN sensitivity.
They are included in both budgets. Guidance and propagation are governed by the
shared CSPN contract and do not participate in the HAWQ search.

For block `l` and candidate bit width `b`, sensitivity cost is:

\[
C_{l,b} = \bar{T}_l
          \left\|W_l-Q_b(W_l)\right\|_2^2,
\]

where `bar(T_l)` is the measured normalized Hutchinson trace. The deterministic
integer optimizer minimizes `sum(C_l,b)` subject to:

\[
\frac{\sum_l b_l\,|W_l|}{\sum_l |W_l|} \le 6.0,
\qquad
\frac{\sum_o b_o\,N_o}{\sum_o N_o} \le 6.0.
\]

`N_o` is the measured activation-element count for owner `o` at the fixed NYU
shape. Assignment validation rejects uncovered, duplicated, extra, or
inconsistent modules and owners. The serialized assignment, cost basis, trace
table, perturbation table, objective value, constraints, and solver status are
required inputs to HAWQ QAT and deployment.

## HAWQ Quantization-Aware Training

After assignment, HAWQ QAT uses the linked implementation's quantization
semantics: symmetric per-output-channel weights and per-site asymmetric
activations with explicit running min/max ranges. The hard path uses the exact
selected bit width at every block. Branch rescaling uses the current CSPN merge
contract instead of HAWQ's classification-specific ResNet containers.

The selected assignment is immutable during QAT. HAWQ QAT and LSQ+ QAT share
the same CSPN task-aware objective used by the current mixed-task-aware QAT:
valid-depth loss, boundary loss, FP teacher loss, and 24-step propagation-state
loss. All coefficients are required configuration values. The teacher is the
frozen official FP32 model.

Training checkpoints contain canonical FP32 master weights, quantizer state,
optimizer and scheduler state, convergence state, graph digest, checkpoint
digest, data manifests, method configuration, and for HAWQ the trace and
assignment digests. Resume requires exact equality of every contract.

## Data Isolation and Convergence

- Quantizer initialization and HAWQ trace estimation use only the fixed
  stratified 128-sample training calibration manifest.
- QAT uses the official NYU training split.
- Early stopping uses validation identities excluding the fixed-64 final
  evaluation identities.
- The fixed-64 set is read only after checkpoint selection.
- All methods start from the same official FP32 checkpoint.

Maximum epochs, patience, minimum relative improvement, batch sizes, workers,
optimizer parameters, scheduler parameters, gradient limit, and seed are
required configuration values. Selection requires finite positive predictions
and uses pooled validation RMSE. Training continues until the configured
plateau or maximum-epoch criterion is met; a smoke run is never reported as a
formal result.

## Parallel GPU Execution

Formal jobs use exclusive visible devices and independent result roots:

- GPU 1: LSQ+ W4A4 initialization, QAT, and validation.
- GPU 2: LSQ+ W6A6 initialization, QAT, and validation.
- GPU 3: HAWQ trace estimation, assignment, QAT, and validation.
- GPU 0: reserved while occupied; otherwise used for FP/RTN replay and final
  fixed-64 evaluation after training jobs finish.

The launcher records physical GPU UUID, CUDA_VISIBLE_DEVICES, PyTorch/CUDA
versions, command, PID, start/end timestamps, and output root. Jobs never share
a mutable calibration cache or checkpoint path. Final evaluation may run in
parallel by configuration because each process writes a disjoint directory;
aggregation starts only after all expected manifests and 64 prediction files
per configuration pass validation.

## Evaluation and Artifacts

Every configuration is evaluated on the same ordered 64 samples. Metrics are
pooled over valid pixels and include RMSE, MAE, AbsRel, iRMSE, flat RMSE,
boundary RMSE, non-finite ratio, non-positive ratio, invalid sample/pixel
counts, anchor maximum error, and propagation-step drift. LSQ+ additionally
reports learned step/offset, saturation, zero-code ratio, and scale gradients.
HAWQ additionally reports trace uncertainty, candidate perturbation costs,
selected bits, budget utilization, and objective contributions.

Prediction payloads contain natural RGB, exact model RGBD input, sparse depth,
GT, FP32 prediction, quantized prediction, absolute error, validity mask,
sample identity, method identity, checkpoint digest, and assignment digest when
applicable. Figures compare RGB, sparse depth, GT, FP32, PA-RTN W4A4/W6A6,
LSQ+ W4A4/W6A6, HAWQ Mixed<=6, and existing Mixed-QAT with a shared 0-10 m
depth scale and shared error scale.

Outputs are written beneath dedicated method directories under
`profile_logs/nyu_cspn_lsqplus_hawq/`. Runtime outputs are not committed.
Code, tests, required JSON configurations, specifications, and result reports
are committed.

## Testing and Acceptance

Implementation follows test-driven development. Unit tests cover LSQ+ hard
forward and gradients, V2 initialization, signed/unsigned ranges, per-output
channel weights, HAWQ HVP trace estimation, perturbation costs, deterministic
budget solving, owner tying, average-bit audits, serialization, and strict
checkpoint reload.

Integration tests cover official CSPN construction, BN folding order, separate
decoder branch scales, guidance exclusion, A8/Q13/INT32 propagation ownership,
24 propagation states, resume contracts, and disjoint output roots. A CUDA
smoke test must complete one train step and hard replay for each new method.

Formal acceptance requires:

1. full repository tests pass;
2. hard-forward parity passes before and after checkpoint reload;
3. HAWQ average weight bits and activation bits are each no greater than 6.0;
4. every configuration has exactly 64 ordered prediction payloads;
5. no configuration has non-finite or non-positive predictions;
6. all metrics are computed without replacement, clipping, or finite filtering;
7. LSQ+ W4A4/W6A6 and HAWQ Mixed<=6 are compared with paired FP32 metrics and
   report absolute and relative degradation;
8. the final report distinguishes measured accuracy from software-QDQ runtime
   and makes no integer-kernel speed claim.

## Coding Constraints

Configuration attributes use direct dot access. Dictionary fields use direct
indexing. Required values have no code-level defaults. Errors propagate
directly; implementation code does not add broad exception handling, hidden
fallbacks, hash-based behavior, invalid-value repair, or temporary files under
`/tmp`.
