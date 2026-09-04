# Four-Model INT Mixed-Precision Quantization Within One Percent

## Objective

Build a strict, model-specific mixed-precision quantization workflow for the
official CSPN, DySPN, NLSPN, and CompletionFormer NYU depth-completion models.
For each model, find the nondominated configurations that minimize both
MAC-weighted average weight precision and activation-element-weighted average
activation precision subject to a hard pooled-RMSE loss limit of one percent.

This work uses integer W4/W6/W8 and A4/A6/A8 for ordinary Conv/Linear
computation. FP16 is permitted only for explicitly declared propagation and
task-sensitive boundaries. BF16 is evaluated only as a hardware-facing
replacement for selected FP16 boundaries after the INT assignment is fixed.

## Success Criteria

For each model and candidate `q`, define:

```text
relative_loss(q) = pooled_RMSE(q) / pooled_RMSE(FP32) - 1
average_W(q) = sum(MAC_i * weight_bits_i) / sum(MAC_i)
average_A(q) = sum(activation_elements_i * activation_bits_i) /
               sum(activation_elements_i)
```

A successful candidate must satisfy all of the following on the fixed 64 NYU
validation samples:

- `relative_loss <= 0.01`;
- every prediction is finite and strictly positive;
- two repeated forwards are bit-exact;
- the official propagation operator executes for the configured iteration
  count;
- every configured quantization owner executes exactly once per forward;
- effective weight and activation formats equal the declared assignment.

The result is the complete feasible Pareto frontier in `(average_W,
average_A)`. The report identifies the minimum-weight endpoint,
minimum-activation endpoint, and balanced knee, but does not hide the other
nondominated candidates behind a scalar score. If a model has no feasible
candidate, the run records `infeasible`; it does not relabel the lowest-RMSE
failure as successful.

## Fixed Data And Model Protocol

- Use the official converged checkpoint and architecture already declared in
  `configs/four_model_unified_fp16_task_aware.json`.
- Use each model's existing stratified 128-sample NYU train calibration set.
- Use the same ordered 64-sample NYU validation set for final evaluation.
- Calibration identities, evaluation identities, checkpoint path, checkpoint
  SHA-256, architecture class, native CUDA extension, and propagation
  iteration count are immutable run inputs.
- PTQ calibration observes the 128 calibration samples once before any
  candidate evaluation.
- QAT uses the complete NYU train split for a configured fixed number of
  epochs. Each epoch records hard-deployment metrics on an explicitly
  configured 64-sample validation subset disjoint from the final evaluation
  identities. The final scheduled checkpoint is evaluated; the final ordered
  64 samples are not used for training, early stopping, checkpoint selection,
  or bit assignment.
- No historical metric produced under another ownership, propagation, data,
  or checkpoint protocol may enter the final comparison.

## Numeric Contract

### Ordinary Weights

- Conv2d, ConvTranspose2d, and Linear weights use signed symmetric integer
  quantization.
- Weight scale is per output channel.
- Candidate precisions are W4, W6, and W8.
- A module may use FP16 only when its semantic policy explicitly permits it
  and task-level ablation proves that W8 cannot meet the accuracy constraint.

### Ordinary Activations

- Signed activations use symmetric integer quantization.
- ReLU outputs use unsigned integer quantization.
- Static calibrated per-tensor scale is the default.
- Candidate precisions are A4, A6, and A8.
- Online Dynamic-G8 scale is allowed only at an explicitly enumerated boundary
  whose static candidate cannot satisfy the one-percent constraint. Its scale
  reduction and metadata traffic are reported separately.
- FP16 activation is allowed only at a declared task-sensitive boundary after
  an A8 failure is measured.

### Integer Accumulation And Bias

- Conv/Linear products accumulate in INT32.
- For one-input consumers, bias uses `s_bias[o] = s_x * s_w[o]`.
- BatchNorm is folded before calibration and before weight quantization.
- Residual-add branches are independently quantized and explicitly
  requantized to the declared add output scale.
- Concat branches are independently calibrated and quantized. Each branch is
  explicitly requantized to the consumer convolution accumulation scale; a
  common pre-concat MinMax scale is forbidden.

### Attention

- CompletionFormer Q/K/V and projection weights participate in W4/W6/W8
  allocation.
- Q/K/V activation outputs have an A8 lower bound.
- QK and AV use A8 integer operands with INT32 accumulation.
- Attention softmax, normalization denominators, and probability tensors stay
  FP16.
- Attention residual and concat edges use independently calibrated branch
  scales followed by explicit requantization.

### Propagation

- The complete propagation path remains FP16 and is excluded from ordinary
  bit allocation and QAT quantizer insertion.
- Affinity normalization, offsets, confidence/gates, sparse-depth anchors,
  propagation state, denominator tensors, and probability tensors remain
  FP16.
- No propagation tensor may be quantized by a generic Conv/Linear owner.
- After an INT Pareto configuration is fixed, an additional run replaces its
  declared FP16 protected boundaries with BF16 to measure RTL-facing loss.

## Model-Specific Search Units

### CSPN

- RGBD stem: `conv1_1`.
- Encoder stages: `layer1`, `layer2`, `layer3`, and `layer4`.
- Decoder stages: `gud_up_proj_layer1` through `gud_up_proj_layer4`.
- Initial-depth head: `gud_up_proj_layer5`.
- Guidance and affinity path: `gud_up_proj_layer6` and
  `post_process_layer`, protected in FP16.
- Decoder Add/Concat edges are separate scale-policy units.

### DySPN

- RGB and sparse-depth stems: `base.conv1_rgb` and `base.conv1_dep`.
- Encoder stages: `base.conv2` through `base.conv6`.
- Decoder stages: `base.dec5` through `base.dec2`.
- Guidance decoder boundary: `base.gd_dec1_`.
- `base.gd_dec0`, `conv_offset_aff`, offset, affinity, confidence, sparse
  anchors, and the DySPN iteration state remain FP16.
- SE final gates remain FP16; the Conv/Linear layers before the final gate may
  participate only as a complete SE block.

### NLSPN

- RGB and sparse-depth stems: `conv1_rgb` and `conv1_dep`.
- Early boundary group: `conv2.0.conv1`, `conv2.0.conv2`, and
  `conv3.0.downsample.0`.
- Remaining encoder stages: `conv3` through `conv6`, excluding the early
  boundaries already owned by the early group.
- Shared decoder stages: `dec5.0` through `dec2.0`.
- Guidance decoder: `gd_dec1.0`.
- Initial-depth decoder: `id_dec1.0` and `id_dec0.0`.
- Guidance/offset/affinity/confidence heads, sparse anchors, and propagation
  state remain FP16.
- The known accuracy anchor uses FP16 for the early and initial-depth groups;
  each group is subsequently tested at W8/A8 and W6/A6 independently.

### CompletionFormer

- RGB and sparse-depth stems.
- CNN encoder stages.
- Transformer embedding and patch-embedding stages.
- Transformer Q/K/V, attention output, and MLP are distinct allocation units.
- Decoder stages, depth decoder, guidance decoder, and initial-depth head are
  distinct units.
- Every registered attention and concat edge is an explicit scale-policy unit.
- Confidence, guidance/offset/affinity output semantics and propagation remain
  FP16.

Contract construction fails if a required group is empty, overlaps a
protected propagation owner, or resolves to a different official module
shape. Module-name guessing and model-family fallback are forbidden.

## Search Procedure

### Phase 1: Feasible Anchors

1. Remeasure FP32 and strict uniform INT W8A8 with FP16 propagation.
2. If W8A8 exceeds one percent, promote model-specific boundary groups to
   FP16 in measured task-sensitivity order.
3. Stop promoting at the first configuration with at most `0.8%` relative
   loss, retaining `0.2%` search headroom.
4. If full permitted protection does not reach one percent, record the PTQ
   anchor as infeasible and send only the nearest finite candidate to QAT.

### Phase 2: Sensitivity And Factorial Ablation

For every ordinary search unit, independently evaluate:

```text
W6A8, W8A6, W6A6, W4A8, W8A4, W4A6, W6A4, W4A4
```

For each FP16-protected boundary, evaluate independent W16/A16 to W8/A8 and
W6/A6 demotions. Record local quantization error, task-gradient error, block
output error, initial-depth/guidance/offset/affinity/confidence error, first
and final propagation-state error, per-sample RMSE delta, and pooled-RMSE
delta.

Adjacent encoder/decoder pairs, skip/upsample concat boundaries,
initial-depth plus upstream decoder, and CompletionFormer attention plus MLP
receive explicit pairwise ablations. This separates isolated sensitivity from
error cancellation and amplification.

### Phase 3: Pareto Beam Search

- Start from a feasible anchor and move only toward lower W or A precision.
- Task-gradient scores prune clearly dominated single-unit moves; they never
  determine final acceptance.
- Rank surviving moves by measured bit reduction and paired 64-sample RMSE
  change.
- Retain all nondominated states up to the configured beam width.
- Cache a candidate only by its complete canonical assignment, scale policy,
  checkpoint identity, and calibration identity. Opaque hashes do not replace
  those fields in manifests.
- A move that exceeds one percent can remain only as a QAT input when its loss
  is at most `1.5%`; it cannot remain on the PTQ feasible frontier.

## Task-Aware QAT

- Select at most three nondominated or near-boundary candidates per model.
- Initialize from the PTQ integer assignment and calibrated scales.
- Train for the configured fixed 5-10 epochs over the complete NYU train
  split; evaluate the final scheduled checkpoint.
- Update ordinary model weights and explicit quantizer parameters only.
- Keep propagation and all protected semantic tensors frozen in FP16.
- Use model-specific configured weights for final-depth loss, initial-depth
  loss, propagation-entry distillation, and propagation-state distillation.
- Missing required semantic signals, non-finite losses, or incomplete epochs
  fail the run directly.
- Do not silently switch quantizers, disable a failing layer, restore FP32, or
  select a previous checkpoint.

## Scheduling

Run one official model per GPU when four devices are available. Device
assignments are explicit configuration values, not runtime fallbacks. Each
worker writes an immutable model result root. The coordinator publishes a
four-model summary only after all required model manifests pass validation.

## Artifacts

Each model writes:

- `fp32_reference.json` and `anchor_summary.csv`;
- `single_module_ablation.csv` and `interaction_ablation.csv`;
- `candidate_assignments.json` with explicit W/A format and scale policy;
- `pareto_ptq.csv` and `pareto_qat.csv`;
- `sample_metrics.csv`, `signal_metrics.csv`, and
  `propagation_state_metrics.csv`;
- `effective_quantization.csv` with owner call counts and effective formats;
- `qat_history.csv` and final checkpoint identity for each QAT candidate;
- `manifest.json` containing all data, model, numeric, and hardware contracts.

The global report compares pooled RMSE, relative loss, average W/A bits,
FP16/BF16 MAC fraction, FP16/BF16 activation fraction, dynamic-scale traffic,
and Pareto status. Old artifacts remain immutable and are not treated as
results under this protocol.

## Verification

- Unit tests cover quantizer ranges, signed/unsigned behavior, per-channel
  weights, bias scale, branch requantization, attention contracts, protected
  ownership, cost accounting, Pareto dominance, and exact candidate identity.
- Contract tests instantiate every official architecture and assert exact
  search/protected groups.
- Integration tests run deterministic calibration, one candidate, and QAT
  checkpoint serialization without real-data fallback.
- Official CUDA runs verify native extensions, fixed sample identities,
  effective formats, finite positive predictions, paired reproducibility, and
  complete propagation execution.
- Final publication requires `relative_loss <= 1%` independently for all four
  models. Partial success is reported per model and never promoted to a
  four-model success claim.
