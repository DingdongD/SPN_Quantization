# CSPN W4-Dominant Mixed-Activation Task-Aware QAT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a CSPN QAT workflow that preserves the P3/T3 W4/W8 weight assignment, allocates ordinary activations from `{4, 6, 8}`, satisfies activation-element-weighted average precision `<= 6.0` bits, and targets fixed-64 NYU RMSE `<= 0.175 m`.

**Architecture:** Reuse the existing task-sensitive assignment registry, hardware-aligned hard quantizers, Group-8 STE controllers, and propagation-aware CSPN adapter. Enumerate and jointly measure all budget-feasible blockwise demotions of P3/T3 on the fixed 128-sample train calibration set, train the selected discrete assignment with a task-aware loss, then reload canonical weights into a fresh official CSPN model for hard-path evaluation.

**Tech Stack:** Python 3.11, PyTorch, CUDA, NumPy, pytest, Matplotlib, official CSPN ResNet-18, existing `spn_quant` hardware and propagation adapters.

**Spec:** `docs/superpowers/specs/2026-08-21-cspn-mixed-task-aware-qat-design.md`

## Global Constraints

- The model is the official CSPN ResNet-18 with 24 propagation steps and checkpoint SHA-256 `482fb9532b27bdb0e529da14845d9a63ab546e28974d90dd1197b8f704066855`.
- Weight precision is exactly the existing P3/T3 W4/W8 assignment; this plan does not search weight bits.
- Ordinary activation precision is selected only from `{4, 6, 8}` with contiguous Group-8 scales and static calibrated ranges.
- The activation budget is `sum(bits_i * elements_i) / sum(elements_i) <= 6.0`.
- ReLU outputs use unsigned integer activation codes; signed activations use symmetric integer codes.
- Guidance and bias remain FP32.
- Affinity and propagation state are A8, coefficients are signed INT16 Q13, accumulation is INT32, and sparse-depth anchors are exact after every step.
- Calibration and assignment search use only the fixed stratified 128-sample NYU train set.
- The fixed 64 evaluation samples are excluded from assignment search and early stopping.
- Configuration fields are required and read directly; dictionaries use `[]`; no default precision promotion, FP fallback, or swallowed exception is allowed.
- Existing RTN, BRECQ, QDrop, task-sensitive PTQ, and static/dynamic W4A4 QAT behavior must remain unchanged.

## File Structure

- Modify `spn_quant/cspn_task_sensitive_bits.py`: P3/T3 seed assignment, activation-budget audit, blockwise demotion enumeration, measured candidate selection.
- Modify `spn_quant/qat/quantizers.py`: hard-forward per-output-channel W4/W8 STE.
- Modify `spn_quant/qat/cspn.py`: explicit per-site weight/activation assignments and differentiable propagation-state capture.
- Create `spn_quant/qat/cspn_task_loss.py`: deterministic boundary mask and composite task-aware loss.
- Modify `spn_quant/qat/__init__.py`: export the task-loss interfaces.
- Modify `spn_quant/propagation/adapters.py`: explicit on-device teacher-state capture mode.
- Create `scripts/run_nyu_cspn_mixed_activation_search.py`: strict 128-sample multi-GPU assignment search and artifact publication.
- Modify `scripts/train_nyu_cspn_group_a4_qat.py`: add an explicit `mixed_static` path while preserving existing modes.
- Modify `scripts/evaluate_nyu_cspn_group_a4_qat.py`: hard-path FP32/W6A6/P3-T3/Mixed-QAT evaluation.
- Modify `scripts/plot_nyu_cspn_group_a4_qat.py`: aligned mixed-QAT prediction and error panels.
- Create `configs/cspn_mixed_task_aware_qat.json`: complete search, loss, training, and acceptance values.
- Modify focused tests under `tests/`; do not introduce a second test harness.

---

### Task 1: P3/T3 Activation-Budget Assignment Contracts

**Files:**
- Modify: `spn_quant/cspn_task_sensitive_bits.py`
- Modify: `tests/test_cspn_task_sensitive_bits.py`

**Interfaces:**
- Consumes: existing `AllocationRegistry`, `BitAssignment`, `CostBasis`, `BLOCK_ORDER`, and `audit_budget()`.
- Produces: `p3_t3_assignment(registry) -> BitAssignment`, `audit_activation_budget(assignment, basis, maximum_bits) -> ActivationBudgetAudit`, `build_p3_t3_activation_candidates(registry, basis, maximum_bits) -> Tuple[BitAssignment, ...]`, and `select_measured_activation_candidate(candidates, rows, basis, maximum_bits) -> BitAssignment`.

- [ ] **Step 1: Write failing budget and seed-assignment tests**

Add tests that assert the exact five promoted P3/T3 blocks and an element-weighted budget independent of owner count:

```python
def test_p3_t3_seed_promotes_only_protected_blocks():
    current = registry()
    assignment = allocation.p3_t3_assignment(current)
    weights = dict(assignment.weight_bits)
    activations = dict(assignment.activation_bits)
    protected = {
        "stem", "encoder_layer1", "encoder_layer2",
        "decoder_layer4", "initial_depth",
    }
    for block in allocation.BLOCK_ORDER:
        expected = 8 if block in protected else 4
        assert {weights[name] for name in current.weights_by_block[block]} == {expected}
        assert {
            activations[owner]
            for owner in current.activations_by_block[block]
        } == {expected}


def test_activation_budget_uses_elements_and_accepts_exact_six():
    basis = allocation.CostBasis(
        weight_macs=(("w", 1),),
        activation_elements=((('large', 'input'), 3), (('small', 'input'), 1)),
    )
    assignment = allocation.BitAssignment(
        weight_bits=(("w", 4),),
        activation_bits=((('large', 'input'), 6), (('small', 'input'), 6)),
    )
    audit = allocation.audit_activation_budget(assignment, basis, 6.0)
    assert audit.activation_numerator == 24
    assert audit.activation_denominator == 4
    assert audit.average_activation_bits == 6.0
    assert audit.feasible
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python -m pytest tests/test_cspn_task_sensitive_bits.py -q`

Expected: FAIL because the four new public interfaces do not exist.

- [ ] **Step 3: Implement the activation-specific audit and deterministic seed**

Add an immutable audit type and preserve the legacy 4-bit `audit_budget()` behavior:

```python
P3_T3_PROTECTED_BLOCKS = (
    "stem", "encoder_layer1", "encoder_layer2",
    "decoder_layer4", "initial_depth",
)
MIXED_ACTIVATION_BITS = (4, 6, 8)


@dataclass(frozen=True)
class ActivationBudgetAudit:
    activation_numerator: int
    activation_denominator: int
    average_activation_bits: float
    maximum_activation_bits: float
    feasible: bool
    activation_element_fractions: Tuple[Tuple[int, float], ...]


def audit_activation_budget(assignment, basis, maximum_bits):
    activation_bits = dict(assignment.activation_bits)
    activation_elements = dict(basis.activation_elements)
    if set(activation_bits) != set(activation_elements):
        raise ValueError("activation assignment and cost coverage mismatch")
    if not math.isfinite(float(maximum_bits)) or float(maximum_bits) <= 0.0:
        raise ValueError("activation budget must be finite and positive")
    denominator = sum(activation_elements.values())
    numerator = sum(
        activation_bits[owner] * activation_elements[owner]
        for owner in activation_elements)
    average = numerator / float(denominator)
    return ActivationBudgetAudit(
        numerator, denominator, average, float(maximum_bits),
        average <= float(maximum_bits),
        tuple((bits, sum(
            elements for owner, elements in basis.activation_elements
            if activation_bits[owner] == bits) / float(denominator))
              for bits in MIXED_ACTIVATION_BITS),
    )
```

- [ ] **Step 4: Write failing enumeration and measured-selection tests**

Tests must prove that only P3/T3 A8 blocks are demoted, all candidates retain identical weight bits, every candidate meets `<= 6.0`, output order is deterministic, and selection rejects incomplete rows, non-finite predictions, nonpositive predictions, and evaluation-only fields.

```python
def test_measured_selection_prioritizes_valid_rmse_then_boundary():
    current = registry()
    basis = unit_basis(current)
    candidates = allocation.build_p3_t3_activation_candidates(
        current, basis, 6.0)
    rows = tuple({
        "assignment": candidate,
        "calibration_RMSE": 0.2 + index * 0.001,
        "boundary_RMSE": 0.4,
        "propagation_MSE": 0.01,
        "nonfinite_ratio": 0.0,
        "nonpositive_ratio": 0.0,
    } for index, candidate in enumerate(candidates))
    assert allocation.select_measured_activation_candidate(
        candidates, rows, basis, 6.0) == candidates[0]
```

- [ ] **Step 5: Implement complete blockwise enumeration and strict selection**

Use `itertools.product(MIXED_ACTIVATION_BITS, repeat=5)` over `P3_T3_PROTECTED_BLOCKS`, keep non-protected activation owners at A4, retain P3/T3 weight bits exactly, filter with `audit_activation_budget`, sort with `assignment_key`, and require exact measured-row coverage. Rank rows by:

```python
(
    float(row["nonfinite_ratio"]) != 0.0,
    float(row["nonpositive_ratio"]) != 0.0,
    float(row["calibration_RMSE"]),
    float(row["boundary_RMSE"]),
    float(row["propagation_MSE"]),
    audit.average_activation_bits,
    assignment_key(row["assignment"]),
)
```

Reject all candidates if the best row has a nonzero numerical-failure ratio.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest tests/test_cspn_task_sensitive_bits.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add spn_quant/cspn_task_sensitive_bits.py tests/test_cspn_task_sensitive_bits.py
git commit -m "feat: add CSPN mixed-activation budget search contracts"
```

---

### Task 2: Generalize Hard-Forward QAT to Explicit W4/W8 and A4/A6/A8 Assignments

**Files:**
- Modify: `spn_quant/qat/quantizers.py`
- Modify: `spn_quant/qat/cspn.py`
- Modify: `scripts/train_nyu_cspn_group_a4_qat.py`
- Modify: `tests/test_qat_quantizers.py`
- Modify: `tests/test_cspn_qat.py`
- Modify: `tests/test_train_nyu_cspn_group_a4_qat.py`

**Interfaces:**
- Consumes: hard activation quantizers already configured by `HardwareAlignedInstrumentor` and `CSPNActivationBoundaryController`.
- Produces: `CSPNQATConfig(mode, weight_bits, activation_bits, group_size, propagation)` where the bit fields are complete canonical tuples, and `CSPNWeightQATController(model, module_bits)`.

- [ ] **Step 1: Write failing W8 weight and mixed-manifest tests**

```python
@pytest.mark.parametrize("bits", (4, 8))
def test_weight_fake_quantizer_matches_symmetric_hard_path(bits):
    weight = torch.tensor([[[[-1.0, -0.1, 0.2, 0.9]]]], requires_grad=True)
    quantizer = PerOutputChannelWeightFakeQuantizer(bits, 0)
    actual = quantizer(weight)
    qmax = 2 ** (bits - 1) - 1
    scale = weight.detach().abs().amax() / qmax
    expected = torch.round(weight.detach() / scale).clamp(-qmax, qmax) * scale
    assert torch.equal(actual, expected)
    actual.sum().backward()
    assert torch.equal(weight.grad, torch.ones_like(weight))
```

Add a controller test with one W4 module, one W8 module, signed A6, unsigned A4, and boundary A8. Assert the manifest contains canonical `weight_bits` and `activation_bits` entries and guidance/bias remain FP32.

Add a decoder-merge test with distinct hard scales for the upsample branch,
signed skip boundary, and downstream merged convolution input. Assert STE
installation preserves all three scales and bit widths independently; neither
branch may inherit the other's scale.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_qat_quantizers.py tests/test_cspn_qat.py -q`

Expected: FAIL because W8 and per-site config are rejected.

- [ ] **Step 3: Generalize weight STE and configuration validation**

Change the weight quantizer validation to:

```python
if self.bits not in (4, 8):
    raise ValueError("CSPN QAT weight bits must be 4 or 8")
```

Replace scalar QAT bit fields with canonical complete tuples:

```python
@dataclass(frozen=True)
class CSPNQATConfig:
    mode: str
    weight_bits: Tuple[Tuple[str, int], ...]
    activation_bits: Tuple[Tuple[Tuple[str, str], int], ...]
    group_size: int
    propagation: PropagationQuantConfig
```

Validate exact uniqueness and allowed domains: weights `{4, 8}`, ordinary and structural activations `{4, 6, 8}`, static/dynamic mode, and Group-8. The controller must compare the declared activation assignment with the installed hard quantizers' actual `bits` values before wrapping them with STE.

- [ ] **Step 4: Update existing strict W4A4 call sites explicitly**

In `prepare_qat_model()`, construct all-W4 tuples from the observed module and owner manifests. Do not keep a scalar compatibility branch:

```python
qat_config = CSPNQATConfig(
    mode=mode,
    weight_bits=tuple((name, 4) for name in weight_modules),
    activation_bits=tuple(
        (owner, 4) for owner in sorted(all_activation_owners, key=str)),
    group_size=8,
    propagation=propagation_config,
)
```

- [ ] **Step 5: Run strict and mixed controller tests**

Run: `python -m pytest tests/test_qat_quantizers.py tests/test_cspn_qat.py tests/test_train_nyu_cspn_group_a4_qat.py -q`

Expected: PASS, including the existing CUDA parity test when CUDA is available.

- [ ] **Step 6: Commit**

```bash
git add spn_quant/qat/quantizers.py spn_quant/qat/cspn.py scripts/train_nyu_cspn_group_a4_qat.py tests/test_qat_quantizers.py tests/test_cspn_qat.py tests/test_train_nyu_cspn_group_a4_qat.py
git commit -m "feat: support explicit mixed precision in CSPN QAT"
```

---

### Task 3: Differentiable Propagation States and Task-Aware Loss

**Files:**
- Create: `spn_quant/qat/cspn_task_loss.py`
- Modify: `spn_quant/qat/cspn.py`
- Modify: `spn_quant/qat/__init__.py`
- Modify: `spn_quant/propagation/adapters.py`
- Create: `tests/test_cspn_task_loss.py`
- Modify: `tests/test_cspn_qat.py`
- Modify: `tests/test_propagation_aware_adapters.py`

**Interfaces:**
- Consumes: student prediction, GT, valid mask, frozen teacher prediction, 24 student proxy states, and 24 detached teacher states.
- Produces: `CSPNTaskLossWeights`, `depth_boundary_mask(target, valid, threshold_m)`, `cspn_task_aware_loss(...) -> Mapping[str, Tensor]`, `CSPNQATPropagationController.proxy_states()`, and `CSPNPropagationAdapter.capture_training_states()`.

- [ ] **Step 1: Write failing deterministic loss tests**

```python
def test_task_loss_is_exact_weighted_sum_and_teacher_is_detached():
    prediction = torch.tensor([[[[1.0, 3.0], [2.0, 4.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 2.0], [2.0, 5.0]]]])
    teacher = torch.tensor([[[[1.0, 2.5], [2.0, 4.5]]]], requires_grad=True)
    valid = torch.ones_like(target, dtype=torch.bool)
    weights = CSPNTaskLossWeights(1.0, 0.25, 0.5, 0.1)
    result = cspn_task_aware_loss(
        prediction, target, valid, teacher,
        (prediction,), (teacher.detach(),), weights, 0.1)
    expected = (
        result["depth"] + 0.25 * result["boundary"]
        + 0.5 * result["teacher"] + 0.1 * result["propagation"])
    assert torch.equal(result["total"], expected)
    result["total"].backward()
    assert prediction.grad is not None
    assert teacher.grad is None
```

Also test empty boundary masks, invalid state count, mismatched shapes, non-finite tensors, and a zero total gradient.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_cspn_task_loss.py -q`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement the focused loss module**

Use a deterministic finite-difference depth edge mask without learned or image-dependent thresholds:

```python
@dataclass(frozen=True)
class CSPNTaskLossWeights:
    depth: float
    boundary: float
    teacher: float
    propagation: float


def depth_boundary_mask(target, valid):
    horizontal = torch.zeros_like(valid)
    vertical = torch.zeros_like(valid)
    horizontal[..., :, 1:] = (
        (target[..., :, 1:] - target[..., :, :-1]).abs() >= 0.1
    ) & valid[..., :, 1:] & valid[..., :, :-1]
    vertical[..., 1:, :] = (
        (target[..., 1:, :] - target[..., :-1, :]).abs() >= 0.1
    ) & valid[..., 1:, :] & valid[..., :-1, :]
    return horizontal | vertical
```

The `0.1 m` boundary threshold is part of the explicit metric contract and must be serialized in the precision config rather than duplicated in training code. Pass it as a required argument to `depth_boundary_mask` in the implementation.

- [ ] **Step 4: Write failing 24-state capture and gradient tests**

Assert that `CSPNQATPropagationController.proxy_states()` returns exactly 24 GPU tensors retaining gradients, while `capture_training_states()` on the FP teacher returns exactly 24 detached tensors on the execution device. Existing `capture()` must continue returning CPU diagnostic states.

- [ ] **Step 5: Implement explicit training-state capture**

In the QAT proxy, append each post-anchor state to `self._proxy_states` and clear it at every forward. In the propagation adapter, add a separate `capture_training_states()` transition that selects float capture and retains detached states on device. `observe()`, `capture()`, `configure()`, and `disable()` must reset that mode explicitly.

- [ ] **Step 6: Run propagation and task-loss tests**

Run: `python -m pytest tests/test_cspn_task_loss.py tests/test_cspn_qat.py tests/test_propagation_aware_adapters.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add spn_quant/qat/cspn_task_loss.py spn_quant/qat/cspn.py spn_quant/qat/__init__.py spn_quant/propagation/adapters.py tests/test_cspn_task_loss.py tests/test_cspn_qat.py tests/test_propagation_aware_adapters.py
git commit -m "feat: add CSPN propagation-aware task QAT loss"
```

---

### Task 4: Strict 128-Sample Multi-GPU Activation Assignment Search

**Files:**
- Create: `scripts/run_nyu_cspn_mixed_activation_search.py`
- Create: `tests/test_run_nyu_cspn_mixed_activation_search.py`
- Create: `configs/cspn_mixed_task_aware_qat.json`

**Interfaces:**
- Consumes: official checkpoint, stratified calibration indices and metadata, existing evaluation protocol only for identity exclusion and digest checks, P3/T3 registry, and CUDA device list.
- Produces: `selected_assignment.json`, `cost_basis.json`, `candidate_metrics.csv`, `search_manifest.json`, and SHA-256 manifest under one atomic output root.

- [ ] **Step 1: Add the complete required configuration**

Create:

```json
{
  "model": "cspn",
  "search": {
    "activation_bits": [4, 6, 8],
    "activation_budget_bits": 6.0,
    "group_size": 8,
    "boundary_threshold_m": 0.1
  },
  "loss": {
    "depth": 1.0,
    "boundary": 0.25,
    "teacher": 0.5,
    "propagation": 0.1
  },
  "training": {
    "epochs": 30,
    "patience": 6,
    "min_relative_improvement": 0.001,
    "batch_size": 4,
    "val_batch_size": 1,
    "workers": 2,
    "learning_rate": 0.0001,
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "max_gradient_norm": 10.0,
    "seed": 20260812,
    "fold_max_error": 0.05,
    "log_interval": 50
  },
  "acceptance": {
    "rmse_m": 0.175,
    "average_activation_bits": 6.0,
    "nonfinite_ratio": 0.0,
    "nonpositive_ratio": 0.0,
    "anchor_max_error": 0.0,
    "coefficient_sum_max_error": 0.0,
    "contraction_violation_ratio": 0.0
  }
}
```

- [ ] **Step 2: Write failing CLI, isolation, and selection tests**

Tests must require every path and device argument, reject non-CUDA or duplicate devices, require exactly 128 unique train calibration indices, reject overlap between train calibration and fixed evaluation identities where identities are comparable, and prove that candidate evaluation calls `CSPNEvaluator.calibration()` only.

```python
def test_search_selects_only_from_calibration_rows(monkeypatch):
    calls = []
    class Evaluator:
        def calibration(self, phase, candidates):
            calls.append((phase, len(candidates)))
            return measured_rows(candidates)
        def validation(self, *args):
            raise AssertionError("validation entered assignment search")
    result = runner.run_activation_search(Evaluator(), registry(), basis(), 6.0)
    assert calls == [("mixed_activation", len(result.candidates))]
```

- [ ] **Step 3: Run tests and verify failure**

Run: `python -m pytest tests/test_run_nyu_cspn_mixed_activation_search.py -q`

Expected: FAIL because the runner does not exist.

- [ ] **Step 4: Implement search by reusing the existing evaluator**

Import `CSPNEvaluator`, `ParallelEvaluator`, `expected_registry`, `build_cost_basis`, `RuntimeCandidate`, and assignment serialization from `scripts/run_nyu_cspn_task_sensitive_bits.py`. Enumerate all blockwise budget-feasible assignments from Task 1, distribute complete candidates across the explicit CUDA devices, aggregate pooled metrics, select with `select_measured_activation_candidate`, and atomically publish only after every candidate row and hash is present.

The evaluation protocol may be read for checkpoint, seed, and fixed-64 identity validation, but its prediction or metric rows cannot be loaded by the search runner.

- [ ] **Step 5: Run search-runner and existing task-sensitive tests**

Run: `python -m pytest tests/test_run_nyu_cspn_mixed_activation_search.py tests/test_run_nyu_cspn_task_sensitive_bits.py tests/test_cspn_task_sensitive_bits.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add configs/cspn_mixed_task_aware_qat.json scripts/run_nyu_cspn_mixed_activation_search.py tests/test_run_nyu_cspn_mixed_activation_search.py
git commit -m "feat: add CSPN mixed-activation calibration search"
```

---

### Task 5: Mixed-Static Task-Aware QAT Training and Resume Contract

**Files:**
- Modify: `scripts/train_nyu_cspn_group_a4_qat.py`
- Modify: `tests/test_train_nyu_cspn_group_a4_qat.py`

**Interfaces:**
- Consumes: `configs/cspn_mixed_task_aware_qat.json`, selected assignment, cost basis, official checkpoint, and calibration metadata.
- Produces: `mixed_static/last.pt`, `mixed_static/best.pt`, `mixed_static/metrics.csv`, and `mixed_static/manifest.json` with complete digests and the exact hard-path assignment.

- [ ] **Step 1: Write failing configuration and data-isolation tests**

Add `mixed_static` to the explicit mode domain. Require `--precision-config`, `--assignment`, and `--cost-basis` in that mode. Verify direct JSON indexing raises on missing fields. Verify the validation loader excludes all 64 `metadata.evaluation_indices` and still covers every other official validation identity exactly once.

```python
def test_mixed_validation_excludes_fixed_evaluation_indices():
    metadata = CalibrationMetadata((1, 2), (3, 7))
    indices = validation_indices(10, metadata.evaluation_indices)
    assert indices == (0, 1, 2, 4, 5, 6, 8, 9)
```

- [ ] **Step 2: Run the training-script tests and verify failure**

Run: `python -m pytest tests/test_train_nyu_cspn_group_a4_qat.py -q`

Expected: FAIL because mixed mode, explicit loss config, and exclusion are absent.

- [ ] **Step 3: Configure the selected hard assignment before installing STE**

Use the existing `runtime_configuration()` and `_configure_quantized()` path to install exact weight and activation overrides. Validate configured hard bits against the serialized assignment, then pass canonical tuples into `CSPNQATConfig`. Do not recreate activation scales inside the QAT controller.

- [ ] **Step 4: Add the frozen teacher and composite training loss**

Load a second official CSPN model from the same checkpoint, set `eval()`, disable gradients, and configure its propagation adapter with `capture_training_states()`. Each training step must execute:

```python
with torch.no_grad():
    teacher_prediction = teacher(model_input)
    teacher_states = tuple(teacher_propagation.last_states())
prediction = model(model_input)
student_states = controller.propagation.proxy_states()
losses = cspn_task_aware_loss(
    prediction, target, target > 0.0, teacher_prediction,
    student_states, teacher_states, loss_weights, boundary_threshold_m)
losses["total"].backward()
```

Validate all predictions, states, loss terms, gradients, and parameters before the optimizer step.

- [ ] **Step 5: Extend checkpoint and resume equality checks**

Persist and compare exact values for precision-config SHA-256, assignment SHA-256, cost-basis SHA-256, calibration manifest, early-stopping identities, fixed-64 identities, owner manifest, loss weights, boundary threshold, and propagation contract. Resume with any mismatch must raise before loading optimizer state.

- [ ] **Step 6: Run training, controller, and loss tests**

Run: `python -m pytest tests/test_train_nyu_cspn_group_a4_qat.py tests/test_cspn_qat.py tests/test_cspn_task_loss.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/train_nyu_cspn_group_a4_qat.py tests/test_train_nyu_cspn_group_a4_qat.py
git commit -m "feat: train CSPN mixed-activation task-aware QAT"
```

---

### Task 6: Fresh-Model Hard Evaluation and Prediction Comparison

**Files:**
- Modify: `scripts/evaluate_nyu_cspn_group_a4_qat.py`
- Modify: `scripts/plot_nyu_cspn_group_a4_qat.py`
- Modify: `tests/test_evaluate_nyu_cspn_group_a4_qat.py`
- Modify: `tests/test_plot_nyu_cspn_group_a4_qat.py`

**Interfaces:**
- Consumes: official FP32 checkpoint, mixed-QAT canonical checkpoint, selected assignment, cost basis, precision config, and fixed evaluation protocol.
- Produces: full-validation and fixed-64 aggregate/sample/region/propagation metrics, 64 predictions per configuration, acceptance report, and aligned PNG/PDF figures.

- [ ] **Step 1: Write failing exact-configuration and hard-parity tests**

Require this order:

```python
EXPECTED_CONFIGS = (
    "FP32", "UNIFORM_W6A6", "P3_T3", "MIXED_TASK_AWARE_QAT",
)
```

Test that every quantized configuration is loaded into a separate fresh official model, P3/T3 and mixed QAT share identical weight assignments, only the mixed checkpoint changes master weights, and the evaluated assignment recomputes average activation bits `<= 6.0`.

- [ ] **Step 2: Run evaluator tests and verify failure**

Run: `python -m pytest tests/test_evaluate_nyu_cspn_group_a4_qat.py -q`

Expected: FAIL because the mixed protocol is absent.

- [ ] **Step 3: Implement fresh hard-path evaluation**

Refactor existing repeated model preparation into a helper that always reloads the requested canonical checkpoint, folds BN before calibration, configures the exact hard assignment, and enables propagation statistics. Recompute:

- pooled RMSE, MAE, AbsRel, and iRMSE;
- flat and boundary RMSE;
- nonfinite and nonpositive ratios;
- activation SQNR, new-zero, and saturation ratios;
- per-step propagation error;
- anchor, coefficient-sum, and contraction invariants;
- activation average bits and fractions;
- W8 weight element and MAC fractions.

The acceptance report must read every threshold from the precision config and list each failed gate separately.

- [ ] **Step 4: Write failing plotting coverage tests**

Verify exactly 64 identities per configuration and require figure columns `RGB`, `Sparse depth`, `GT`, `FP32`, `Uniform W6A6`, `P3/T3`, `Mixed QAT`, and `Mixed absolute error`. Plotting must recompute displayed RMSE from prediction arrays.

- [ ] **Step 5: Implement aligned prediction figures**

Extend the existing plot script without changing old static/dynamic figure behavior. Use Arial, fixed shared depth limits per sample, a separate fixed error range, non-rotated labels, no title, and no in-plot feature-description text.

- [ ] **Step 6: Run evaluator and plotting tests**

Run: `python -m pytest tests/test_evaluate_nyu_cspn_group_a4_qat.py tests/test_plot_nyu_cspn_group_a4_qat.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/evaluate_nyu_cspn_group_a4_qat.py scripts/plot_nyu_cspn_group_a4_qat.py tests/test_evaluate_nyu_cspn_group_a4_qat.py tests/test_plot_nyu_cspn_group_a4_qat.py
git commit -m "feat: evaluate CSPN mixed-task QAT hard path"
```

---

### Task 7: Regression Verification and CUDA Smoke Test

**Files:**
- Modify only when a failing test identifies a defect in files changed by Tasks 1-6.

**Interfaces:**
- Consumes: all code and tests from Tasks 1-6.
- Produces: green focused/full tests and one canonical hard-path CUDA smoke checkpoint.

- [ ] **Step 1: Run all focused CPU and CUDA-capable tests**

Run:

```bash
python -m pytest \
  tests/test_cspn_task_sensitive_bits.py \
  tests/test_qat_quantizers.py \
  tests/test_cspn_qat.py \
  tests/test_cspn_task_loss.py \
  tests/test_propagation_aware_adapters.py \
  tests/test_run_nyu_cspn_mixed_activation_search.py \
  tests/test_train_nyu_cspn_group_a4_qat.py \
  tests/test_evaluate_nyu_cspn_group_a4_qat.py \
  tests/test_plot_nyu_cspn_group_a4_qat.py -q
```

Expected: PASS with no newly introduced warnings.

- [ ] **Step 2: Run the complete repository suite**

Run: `python -m pytest -q`

Expected: at least the baseline `1176 passed, 16 subtests passed`; the only permitted warning is the existing old-`meshgrid` compatibility warning.

- [ ] **Step 3: Run one-epoch CUDA smoke training**

Use a generated budget-feasible assignment from Task 1 and execute mixed-static QAT with `epochs=1`, `max_train_samples=8`, and `max_val_samples=8` through an explicit smoke config copied from the production config with those three changed fields. Store it under `/workspace/SPN_Quantization/profile_logs/nyu_cspn_mixed_task_aware_qat_smoke`, not `/tmp`.

Expected checks:

- one finite optimization epoch;
- nonzero finite gradient norm;
- exactly 24 student and teacher states per batch;
- canonical checkpoint contains no parametrization keys;
- fresh hard evaluator prediction equals the QAT hard-forward prediction;
- activation budget remains `<= 6.0`.

- [ ] **Step 4: Commit any smoke-only configuration and verified fixes**

```bash
git add configs tests spn_quant scripts
git commit -m "test: verify CSPN mixed-task QAT deployment path"
```

Do not create an empty commit when no tracked files changed.

---

### Task 8: Formal Search, QAT, Fixed-64 Evaluation, and Results Record

**Files:**
- Create after measured completion: `docs/2026-08-21-cspn-mixed-task-aware-qat-results.md`
- Modify after measured completion: `docs/2026-08-20-quantization-framework-inventory.md`

**Interfaces:**
- Consumes: green Task 7 code, official checkpoint/data, fixed calibration/evaluation manifests, and four CUDA devices.
- Produces: measured search/training/evaluation artifacts and an evidence-backed result document.

- [ ] **Step 1: Run the formal 128-sample multi-GPU search**

```bash
python scripts/run_nyu_cspn_mixed_activation_search.py \
  --config configs/cspn_mixed_task_aware_qat.json \
  --run-dir /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24 \
  --checkpoint /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24/best.pt \
  --data-root /workspace/CSPN/cspn_pytorch \
  --calibration-indices /workspace/SPN_Quantization/profile_logs/nyu_cspn_stratified_calibration_128/calibration_indices.json \
  --calibration-metadata /workspace/SPN_Quantization/profile_logs/nyu_cspn_stratified_calibration_128/metadata.json \
  --evaluation-protocol /workspace/SPN_Quantization/profile_logs/nyu_cspn_activation_resolution/cspn/metadata.json \
  --out-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_mixed_task_aware_qat/search \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

Expected: all enumerated budget-feasible candidates have 128 calibration rows, the selected assignment is finite and positive, and its recomputed average activation precision is `<= 6.0`.

- [ ] **Step 2: Run formal mixed-static task-aware QAT**

```bash
python scripts/train_nyu_cspn_group_a4_qat.py \
  --mode mixed_static \
  --precision-config configs/cspn_mixed_task_aware_qat.json \
  --assignment /workspace/SPN_Quantization/profile_logs/nyu_cspn_mixed_task_aware_qat/search/selected_assignment.json \
  --cost-basis /workspace/SPN_Quantization/profile_logs/nyu_cspn_mixed_task_aware_qat/search/cost_basis.json \
  --checkpoint /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24/best.pt \
  --data-root /workspace/CSPN/cspn_pytorch \
  --calibration-metadata /workspace/SPN_Quantization/profile_logs/nyu_cspn_activation_resolution/cspn/metadata.json \
  --output-root /workspace/SPN_Quantization/profile_logs/nyu_cspn_mixed_task_aware_qat \
  --device cuda:0 \
  --epochs 30 --patience 6 --min-relative-improvement 0.001 \
  --batch-size 4 --val-batch-size 1 --workers 2 \
  --learning-rate 0.0001 --momentum 0.9 --weight-decay 0.0001 \
  --max-gradient-norm 10 --seed 20260812 \
  --max-train-samples 0 --max-val-samples 0 \
  --fold-max-error 0.05 --log-interval 50
```

Expected: training stops only by configured convergence or epoch limit and preserves the best finite positive-validation checkpoint.

- [ ] **Step 3: Run fresh-model hard evaluation and plotting**

```bash
python scripts/evaluate_nyu_cspn_group_a4_qat.py \
  --protocol mixed_task_aware \
  --precision-config configs/cspn_mixed_task_aware_qat.json \
  --device cuda:0 \
  --fp32-checkpoint /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24/best.pt \
  --mixed-checkpoint /workspace/SPN_Quantization/profile_logs/nyu_cspn_mixed_task_aware_qat/mixed_static/best.pt \
  --assignment /workspace/SPN_Quantization/profile_logs/nyu_cspn_mixed_task_aware_qat/search/selected_assignment.json \
  --cost-basis /workspace/SPN_Quantization/profile_logs/nyu_cspn_mixed_task_aware_qat/search/cost_basis.json \
  --data-root /workspace/CSPN/cspn_pytorch \
  --calibration-metadata /workspace/SPN_Quantization/profile_logs/nyu_cspn_activation_resolution/cspn/metadata.json \
  --output-root /workspace/SPN_Quantization/profile_logs/nyu_cspn_mixed_task_aware_qat/evaluation \
  --batch-size 1 --workers 4 --seed 20260812 \
  --fold-max-error 0.05 --sample-capacity 1000000
```

Expected: 64 predictions per configuration, zero numerical/invariant failures, a generated acceptance report, and PNG/PDF prediction panels.

- [ ] **Step 4: Independently recompute acceptance metrics**

Read prediction arrays and GT directly, recompute pooled fixed-64 RMSE in float64, recompute activation average bits from assignment plus cost basis, and compare both against the generated CSV/JSON with absolute tolerance `1e-12` for scalar aggregation fields.

The method passes only when RMSE `<= 0.175 m`, average activation bits `<= 6.0`, and every numerical/propagation invariant is zero. A failed quality gate is recorded without changing precision or rerunning with an automatic promotion.

- [ ] **Step 5: Write the measured results document and update inventory**

Record exact checkpoint/config/manifest hashes, selected block bits, weighted precision, FP32/W6A6/P3-T3/Mixed-QAT metrics, convergence history, propagation invariants, acceptance status, artifact paths, and reproducible commands. Update the inventory only after the hard-path artifacts and independent recomputation agree.

- [ ] **Step 6: Run final verification and commit**

Run: `python -m pytest -q`

Expected: full suite PASS.

```bash
git add docs/2026-08-21-cspn-mixed-task-aware-qat-results.md docs/2026-08-20-quantization-framework-inventory.md
git commit -m "docs: record CSPN mixed-task QAT results"
```
