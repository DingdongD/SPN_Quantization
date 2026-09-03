# Four-Model INT Mixed-Precision Within One Percent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a strict four-model INT W4/W6/W8 and A4/A6/A8 mixed-precision search that minimizes MAC-weighted weight bits and activation-element-weighted activation bits while enforcing at most one percent pooled-RMSE loss.

**Architecture:** Extend the existing model contracts, hardware-aligned quantizer, and hard-deployment evaluator instead of creating a second quantization stack. A small constrained-search module owns canonical assignments, cost accounting, feasibility, and Pareto selection; one unified runner performs anchor, factorial, interaction, beam-search, and optional fixed-epoch QAT phases for CSPN, DySPN, NLSPN, and CompletionFormer. Propagation and declared semantic tensors remain FP16 throughout the search, with a separate post-selection BF16 measurement.

**Tech Stack:** Python 3, PyTorch/CUDA, official model CUDA extensions, pytest, JSON/CSV artifacts, existing `spn_quant` adapters and NYU runtime loaders.

---

### Task 1: Canonical Assignments, Cost Accounting, And Pareto Rules

**Files:**
- Create: `spn_quant/constrained_mixed_precision.py`
- Create: `tests/test_constrained_mixed_precision.py`

- [ ] **Step 1: Write failing tests for strict assignment validation**

```python
def test_assignment_requires_complete_independent_weight_and_activation_maps():
    with pytest.raises(ValueError, match="weight assignment coverage"):
        PrecisionAssignment(
            weight_bits=(("encoder", 8),),
            activation_bits=(("encoder", 8), ("decoder", 8)),
            scale_policies=(("encoder", "static_tensor"),
                            ("decoder", "static_tensor")),
            expected_units=("encoder", "decoder"),
        )


def test_assignment_rejects_non_integer_search_precision():
    with pytest.raises(ValueError, match="4, 6, or 8"):
        PrecisionAssignment(
            weight_bits=(("encoder", 5),),
            activation_bits=(("encoder", 8),),
            scale_policies=(("encoder", "static_tensor"),),
            expected_units=("encoder",),
        )
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `pytest -q tests/test_constrained_mixed_precision.py`

Expected: collection fails because `spn_quant.constrained_mixed_precision` does not exist.

- [ ] **Step 3: Implement immutable assignments and measured candidates**

```python
INTEGER_BITS = (4, 6, 8)


@dataclass(frozen=True)
class PrecisionAssignment:
    weight_bits: Tuple[Tuple[str, int], ...]
    activation_bits: Tuple[Tuple[str, int], ...]
    scale_policies: Tuple[Tuple[str, str], ...]
    expected_units: Tuple[str, ...]


@dataclass(frozen=True)
class MeasuredCandidate:
    candidate_id: str
    assignment: PrecisionAssignment
    pooled_rmse: float
    reference_pooled_rmse: float
    average_weight_bits: float
    average_activation_bits: float
    fp16_mac_fraction: float
    fp16_activation_fraction: float
```

Validate exact unit coverage, duplicate keys, legal bit widths, finite positive RMSE, and canonical sorted serialization. Use direct dictionary indexing for required fields.

- [ ] **Step 4: Add failing tests for weighted averages, feasibility, and dominance**

```python
def test_weighted_costs_use_macs_and_activation_elements():
    costs = PrecisionCosts(
        weight_macs=(("encoder", 90), ("head", 10)),
        activation_elements=(("encoder", 10), ("head", 90)),
    )
    assignment = assignment_for(encoder=(4, 8), head=(8, 4))
    assert weighted_average_bits(assignment, costs) == pytest.approx((4.4, 4.4))


def test_pareto_frontier_enforces_one_percent_before_dominance():
    frontier = feasible_pareto_frontier(candidates, maximum_relative_loss=0.01)
    assert tuple(row.candidate_id for row in frontier) == ("min-a", "balanced")
```

- [ ] **Step 5: Implement cost and frontier functions and run GREEN**

Implement `relative_loss`, `weighted_average_bits`, `is_feasible`, `dominates`, `feasible_pareto_frontier`, and `balanced_knee`. The balanced knee normalizes W and A only over the feasible frontier and resolves ties by pooled RMSE then canonical candidate ID.

Run: `pytest -q tests/test_constrained_mixed_precision.py`

Expected: all tests pass.

- [ ] **Step 6: Commit the domain layer**

```bash
git add spn_quant/constrained_mixed_precision.py tests/test_constrained_mixed_precision.py
git commit -m "feat: add constrained mixed precision domain"
```

### Task 2: Exact Model-Specific Search Units And Precision Policies

**Files:**
- Modify: `spn_quant/model_contracts.py`
- Modify: `spn_quant/adapters/cspn.py`
- Modify: `spn_quant/adapters/dyspn.py`
- Modify: `spn_quant/adapters/nlspn.py`
- Modify: `spn_quant/adapters/completionformer.py`
- Modify: `tests/test_model_quantization_contracts.py`
- Modify: `tests/test_official_model_quantization_contracts.py`

- [ ] **Step 1: Write failing synthetic contract tests**

```python
def test_nlspn_contract_declares_initial_depth_and_early_boundary_units():
    contract = build_model_quantization_contract("nlspn", SyntheticNLSPN())
    policies = dict((unit.name, unit) for unit in contract.search_units)
    assert policies["early_boundary"].members == (
        "conv2.0.conv1", "conv2.0.conv2", "conv3.0.downsample.0")
    assert policies["initial_depth"].members == ("id_dec1.0", "id_dec0.0")
    assert policies["initial_depth"].allow_fp16 is True


def test_completionformer_attention_activation_floor_is_a8():
    contract = build_model_quantization_contract(
        "completionformer", SyntheticCompletionFormer())
    attention = tuple(unit for unit in contract.search_units
                      if unit.kind == "attention_qkv")
    assert attention
    assert all(unit.minimum_activation_bits == 8 for unit in attention)
```

- [ ] **Step 2: Run contract tests and verify RED**

Run: `pytest -q tests/test_model_quantization_contracts.py tests/test_official_model_quantization_contracts.py`

Expected: failures because `search_units` and precision-policy fields are absent.

- [ ] **Step 3: Add strict search-unit types and adapter declarations**

Add `PrecisionSearchUnit` with `name`, `members`, `activation_owners`, `kind`, `minimum_weight_bits`, `minimum_activation_bits`, and `allow_fp16`. Add adapter constants that enumerate each spec-approved unit with exact anchored regular expressions. Resolve each unit against contract blocks, reject empty units, duplicate members, protected overlap, and shape drift.

- [ ] **Step 4: Add official architecture assertions**

For each official architecture, assert exact membership for stems, encoder stages, decoder stages, fusion/concat, initial-depth, and attention units. Assert guidance, affinity, offset, confidence, sparse-anchor, state, probability, and propagation modules never occur in ordinary search units.

- [ ] **Step 5: Run all contract tests and commit**

Run: `pytest -q tests/test_model_quantization_contracts.py tests/test_official_model_quantization_contracts.py tests/test_propagation_fp16_contract.py`

Expected: all tests pass; Python-version-gated official tests remain explicitly deselected only under their existing marker.

```bash
git add spn_quant/model_contracts.py spn_quant/adapters tests/test_model_quantization_contracts.py tests/test_official_model_quantization_contracts.py
git commit -m "feat: define exact four-model precision units"
```

### Task 3: Apply Arbitrary INT Assignments In The Existing Hardware Evaluator

**Files:**
- Modify: `scripts/hardware_aligned_quantization.py`
- Modify: `scripts/run_nyu_model_p3t3_search.py`
- Modify: `tests/test_hardware_aligned_quantization.py`
- Modify: `tests/test_run_nyu_model_p3t3_search.py`

- [ ] **Step 1: Write failing tests for full owner-level assignments**

```python
def test_hard_evaluator_applies_independent_weight_and_activation_bits():
    result = evaluator.evaluate_assignment(assignment)
    assert result.effective_weight_bits == (("encoder.conv", 4),
                                            ("head.conv", 8))
    assert result.effective_activation_bits == (
        (("encoder.conv", "input"), 6),
        (("head.conv", "input"), 8),
    )
    assert result.owner_call_counts == (
        (("encoder.conv", "input"), 1),
        (("head.conv", "input"), 1),
    )
```

- [ ] **Step 2: Run focused evaluator tests and verify RED**

Run: `pytest -q tests/test_hardware_aligned_quantization.py tests/test_run_nyu_model_p3t3_search.py`

Expected: failure because the evaluator only accepts P3/T3 candidate wrappers.

- [ ] **Step 3: Add a strict assignment application API**

Add `configure_integer_assignment(...)` to `HardwareAlignedInstrumentor`. It must require complete explicit weight and activation owner maps for enabled groups, preserve per-output-channel signed weight quantization, preserve unsigned ReLU output quantization, and reject missing/extra owners. Do not introduce defaults or automatic owner disabling.

- [ ] **Step 4: Reuse calibration in an arbitrary-assignment evaluator**

Add `evaluate_precision_assignment(...)` to `HardDeploymentP3T3Evaluator`. Translate model search units to exact module and activation-owner bit overrides, bind existing concat/attention adapters, keep propagation adapter in FP16, and return effective-format, call-count, finite/positive, deterministic, semantic-signal, and propagation-iteration audits.

- [ ] **Step 5: Test strict failure behavior**

```python
def test_assignment_missing_one_calibrated_owner_fails():
    with pytest.raises(ValueError, match="activation assignment coverage"):
        evaluator.evaluate_precision_assignment(incomplete_assignment)


def test_protected_propagation_owner_cannot_enter_integer_assignment():
    with pytest.raises(ValueError, match="protected"):
        evaluator.evaluate_precision_assignment(propagation_assignment)
```

- [ ] **Step 6: Run tests and commit**

Run: `pytest -q tests/test_hardware_aligned_quantization.py tests/test_run_nyu_model_p3t3_search.py tests/test_propagation_fp16_contract.py`

Expected: all tests pass.

```bash
git add scripts/hardware_aligned_quantization.py scripts/run_nyu_model_p3t3_search.py tests/test_hardware_aligned_quantization.py tests/test_run_nyu_model_p3t3_search.py
git commit -m "feat: evaluate explicit integer precision assignments"
```

### Task 4: Enforce Branch-Aware Concat And CompletionFormer Attention Contracts

**Files:**
- Modify: `scripts/hardware_merge_adapters.py`
- Modify: `spn_quant/completionformer_concat.py`
- Modify: `spn_quant/adapters/completionformer_joint.py`
- Modify: `tests/test_hardware_merge_adapters.py`
- Modify: `tests/test_completionformer_concat.py`
- Modify: `tests/test_completionformer_joint_adapter.py`

- [ ] **Step 1: Write failing concat requantization tests**

```python
def test_concat_branches_keep_independent_scales_until_consumer_requantize():
    output, audit = adapter.quantize_concat((small_branch, large_branch), bits=6)
    assert audit.branch_scales[0] != audit.branch_scales[1]
    assert audit.consumer_scale > 0.0
    assert torch.count_nonzero(output[:, :small_channels]) > 0
```

- [ ] **Step 2: Write failing attention floor and protected-softmax tests**

```python
def test_completionformer_attention_rejects_qkv_below_a8():
    with pytest.raises(ValueError, match="QKV activation precision"):
        adapter.configure_attention({"stage.qkv": 6})


def test_completionformer_softmax_is_not_an_integer_owner():
    assert not set(adapter.integer_owners).intersection(adapter.softmax_owners)
```

- [ ] **Step 3: Run tests and verify RED**

Run: `pytest -q tests/test_hardware_merge_adapters.py tests/test_completionformer_concat.py tests/test_completionformer_joint_adapter.py`

Expected: at least the explicit scale-audit or A8-floor assertions fail.

- [ ] **Step 4: Implement explicit scale metadata and validation**

Retain independent branch observers, quantize each branch independently, and record the explicit integer multiplier/shift used to map each branch to the consumer scale. Enforce A8 Q/K/V and QK/AV inputs with INT32 accumulation while leaving softmax, normalization, and probability tensors FP16.

- [ ] **Step 5: Run tests and commit**

Run: `pytest -q tests/test_hardware_merge_adapters.py tests/test_completionformer_concat.py tests/test_completionformer_joint_adapter.py tests/test_attention_internal_quantization.py`

Expected: all tests pass.

```bash
git add scripts/hardware_merge_adapters.py spn_quant/completionformer_concat.py spn_quant/adapters/completionformer_joint.py tests/test_hardware_merge_adapters.py tests/test_completionformer_concat.py tests/test_completionformer_joint_adapter.py
git commit -m "fix: enforce branch and attention integer contracts"
```

### Task 5: Implement Anchor, Ablation, Interaction, And Pareto Beam Search

**Files:**
- Create: `scripts/run_nyu_four_model_int_mixed_precision.py`
- Create: `tests/test_run_nyu_four_model_int_mixed_precision.py`
- Create: `configs/four_model_int_mixed_precision_1pct.json`

- [ ] **Step 1: Write failing tests for search phase ordering and anchor rules**

```python
def test_search_builds_anchor_before_demotions():
    search = FakeSearch(measurements)
    result = search.run()
    assert search.calls[0].phase == "fp32"
    assert search.calls[1].phase == "uniform_w8a8"
    assert result.anchor.relative_loss <= 0.008


def test_search_records_infeasible_instead_of_accepting_best_failure():
    result = run_search(all_candidates_above_one_percent)
    assert result.status == "infeasible"
    assert result.pareto_frontier == ()
```

- [ ] **Step 2: Run the new test and verify RED**

Run: `pytest -q tests/test_run_nyu_four_model_int_mixed_precision.py`

Expected: import failure because the runner does not exist.

- [ ] **Step 3: Implement deterministic candidate generation**

Implement exact generators for uniform W8A8, ordered FP16 boundary promotions, the eight single-unit factorial points, declared pair interactions, and one-step W/A demotions. Candidate IDs must be readable canonical IDs; manifests contain full assignments and scale policies.

- [ ] **Step 4: Implement measured Pareto beam search**

Start at the measured feasible anchor, use gradient scores only to cap the evaluated move set, evaluate retained moves on the fixed ordered 64 samples, and preserve every nondominated feasible assignment. Permit `relative_loss <= 0.015` only in `qat_candidates`, never in the PTQ frontier.

- [ ] **Step 5: Implement artifact schemas and strict run manifests**

Write the exact files declared by the spec. `manifest.json` must include model class, checkpoint path and SHA-256, calibration/evaluation sample identities, native extension identity, iteration count, complete assignment, scale policy, call counts, deterministic audit, and status. Existing output directories cause `FileExistsError`.

- [ ] **Step 6: Add and validate the four-model configuration**

Configure each official model from `configs/four_model_unified_fp16_task_aware.json`, exact GPU IDs, stratified 128 calibration metadata, ordered 64 evaluation identities, beam width, interaction pairs, QAT threshold, and FP16 boundary permissions. Access required configuration with dot attributes or dictionary indexing only.

- [ ] **Step 7: Run tests and commit**

Run: `pytest -q tests/test_run_nyu_four_model_int_mixed_precision.py tests/test_constrained_mixed_precision.py tests/test_run_nyu_model_p3t3_search.py`

Expected: all tests pass.

```bash
git add scripts/run_nyu_four_model_int_mixed_precision.py tests/test_run_nyu_four_model_int_mixed_precision.py configs/four_model_int_mixed_precision_1pct.json
git commit -m "feat: add constrained four-model precision search"
```

### Task 6: Fixed-Epoch Task-Aware QAT Without Validation Selection

**Files:**
- Modify: `scripts/train_nyu_selected_qat.py`
- Modify: `spn_quant/qat/model_methods.py`
- Modify: `tests/test_train_nyu_selected_qat.py`
- Modify: `tests/test_qat_model_methods.py`

- [ ] **Step 1: Write failing tests for the new QAT protocol**

```python
def test_fixed_epoch_protocol_publishes_only_final_checkpoint(tmp_path):
    run_selected_qat(args_for_fixed_epoch(tmp_path, epochs=5))
    assert (tmp_path / "final.pt").is_file()
    assert not (tmp_path / "best.pt").exists()
    history = read_csv(tmp_path / "qat_history.csv")
    assert history[-1]["epoch"] == "5"


def test_fixed_evaluation_samples_do_not_select_or_stop_qat():
    tracker = FixedEpochProtocol(epochs=5)
    assert tuple(tracker.should_continue(epoch, rmse)
                 for epoch, rmse in enumerate((1.0, 2.0, 3.0, 4.0), 1)) == \
        (True, True, True, True)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest -q tests/test_train_nyu_selected_qat.py tests/test_qat_model_methods.py`

Expected: failures because the legacy convergence tracker writes `best.pt` and may stop early.

- [ ] **Step 3: Add an explicit fixed-epoch protocol**

Add a required `checkpoint_protocol` configuration. Preserve legacy behavior for legacy configs, but the new protocol must run every configured epoch, use the complete train loader, write `last.pt` each epoch and `final.pt` only at completion, and never write or read `best.pt`.

- [ ] **Step 4: Freeze propagation and require semantic losses**

Assert that propagation/protected tensors have no quantizer and no trainable quantizer parameter. Require configured initial-depth, propagation-entry, and state-distillation signals for each model; missing signals fail before the first optimizer step.

- [ ] **Step 5: Run tests and commit**

Run: `pytest -q tests/test_train_nyu_selected_qat.py tests/test_qat_model_methods.py tests/test_propagation_fp16_contract.py`

Expected: all tests pass and legacy checkpoint tests remain unchanged.

```bash
git add scripts/train_nyu_selected_qat.py spn_quant/qat/model_methods.py tests/test_train_nyu_selected_qat.py tests/test_qat_model_methods.py
git commit -m "feat: add fixed epoch mixed precision QAT"
```

### Task 7: Four-GPU Orchestration And Result Validation

**Files:**
- Create: `scripts/launch_nyu_four_model_int_mixed_precision.py`
- Create: `tests/test_launch_nyu_four_model_int_mixed_precision.py`
- Modify: `configs/four_model_int_mixed_precision_1pct.json`

- [ ] **Step 1: Write failing DAG and GPU assignment tests**

```python
def test_launcher_assigns_one_declared_model_to_each_gpu():
    graph = build_graph(load_config(CONFIG))
    assert tuple((job.model, job.cuda_device) for job in graph.root_jobs) == (
        ("cspn", 0), ("nlspn", 1), ("dyspn", 2),
        ("completionformer", 3))


def test_summary_waits_for_all_valid_manifests():
    with pytest.raises(RuntimeError, match="manifest validation"):
        publish_summary(graph_with_one_failed_model)
```

- [ ] **Step 2: Run launcher tests and verify RED**

Run: `pytest -q tests/test_launch_nyu_four_model_int_mixed_precision.py`

Expected: import failure because the launcher does not exist.

- [ ] **Step 3: Implement strict process orchestration**

Launch one model worker per explicitly configured GPU, stream logs to each immutable result root, wait for every worker, validate all manifests, and launch at most three QAT jobs per model after PTQ candidate publication. A failed process stops summary publication and is reported with its exact command and exit status; no GPU reassignment or method fallback is allowed.

- [ ] **Step 4: Implement global Pareto summary**

Write `four_model_summary.csv` and `manifest.json` with FP32, selected PTQ/QAT endpoints, pooled RMSE, relative loss, weighted W/A bits, FP16 fractions, dynamic-scale traffic, status, and model artifact links.

- [ ] **Step 5: Run tests and commit**

Run: `pytest -q tests/test_launch_nyu_four_model_int_mixed_precision.py tests/test_run_nyu_four_model_int_mixed_precision.py`

Expected: all tests pass.

```bash
git add scripts/launch_nyu_four_model_int_mixed_precision.py tests/test_launch_nyu_four_model_int_mixed_precision.py configs/four_model_int_mixed_precision_1pct.json
git commit -m "feat: orchestrate four-model precision evaluation"
```

### Task 8: Regression Verification And Official CUDA Evaluation

**Files:**
- Create: `docs/results/2026-09-03-four-model-int-mixed-precision-1pct.md`
- Create by command: `profile_logs/nyu_four_model_int_mixed_precision_1pct/`

- [ ] **Step 1: Run the complete focused regression suite**

Run:

```bash
pytest -q \
  tests/test_constrained_mixed_precision.py \
  tests/test_model_quantization_contracts.py \
  tests/test_official_model_quantization_contracts.py \
  tests/test_hardware_aligned_quantization.py \
  tests/test_hardware_merge_adapters.py \
  tests/test_completionformer_concat.py \
  tests/test_completionformer_joint_adapter.py \
  tests/test_run_nyu_model_p3t3_search.py \
  tests/test_run_nyu_four_model_int_mixed_precision.py \
  tests/test_train_nyu_selected_qat.py \
  tests/test_qat_model_methods.py \
  tests/test_launch_nyu_four_model_int_mixed_precision.py \
  tests/test_propagation_fp16_contract.py
```

Expected: all supported-environment tests pass; version-gated tests are listed explicitly.

- [ ] **Step 2: Validate official environments and CUDA extensions**

Run the repository's environment preflight for each configured interpreter and require the native CSPN/DySPN/NLSPN/CompletionFormer operator probes to pass. Record interpreter, PyTorch, CUDA, GPU, extension path, and architecture class in each manifest.

- [ ] **Step 3: Run FP32 and strict W8A8 anchors concurrently**

Run:

```bash
python scripts/launch_nyu_four_model_int_mixed_precision.py \
  --config configs/four_model_int_mixed_precision_1pct.json \
  --output profile_logs/nyu_four_model_int_mixed_precision_1pct \
  --phase anchors
```

Expected: four valid manifests containing finite positive deterministic FP32 and W8A8 measurements on the same ordered 64 samples.

- [ ] **Step 4: Run factorial, interaction, and PTQ Pareto phases**

Run:

```bash
python scripts/launch_nyu_four_model_int_mixed_precision.py \
  --config configs/four_model_int_mixed_precision_1pct.json \
  --output profile_logs/nyu_four_model_int_mixed_precision_1pct \
  --phase ptq-search
```

Expected: complete ablation tables, explicit assignments, and either a nonempty feasible frontier or an explicit `infeasible` status for each model.

- [ ] **Step 5: Run fixed-epoch QAT for published candidates**

Run:

```bash
python scripts/launch_nyu_four_model_int_mixed_precision.py \
  --config configs/four_model_int_mixed_precision_1pct.json \
  --output profile_logs/nyu_four_model_int_mixed_precision_1pct \
  --phase qat
```

Expected: every selected candidate completes the configured epoch count and publishes `final.pt`, `qat_history.csv`, and hard-deployment evaluation artifacts.

- [ ] **Step 6: Measure BF16 boundary substitution**

Evaluate only selected INT assignments with declared FP16 protected boundaries replaced by BF16. Keep this row separate from the INT Pareto frontier and report its pooled-RMSE delta.

- [ ] **Step 7: Write the results report from generated artifacts**

Document per-model FP32 RMSE, feasible endpoints, balanced knee, QAT result, weighted W/A bits, FP16/BF16 fractions, one-percent pass/fail, and dominant measured error sources. Do not copy historical metrics into this table.

- [ ] **Step 8: Run final repository verification and commit**

Run: `pytest -q`

Expected: the full supported test suite passes.

```bash
git add docs/results/2026-09-03-four-model-int-mixed-precision-1pct.md
git commit -m "docs: report four-model constrained quantization"
```
