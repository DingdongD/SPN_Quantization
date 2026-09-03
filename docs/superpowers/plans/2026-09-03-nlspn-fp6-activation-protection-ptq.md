# NLSPN FP6 Activation Protection PTQ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct NLSPN FP concat ownership, evaluate eight FP6-dominant activation-protection candidates on the fixed NYU 128/64 protocol, and retain only measured end-to-end improvements.

**Architecture:** Extend the existing NLSPN FP6 runner with an `activation-protection` mode. A dedicated evaluator disables the old concat execution adapter, returns initial-depth weights to the ordinary FP instrumentor, and installs exactly one common or branch-independent input QDQ. Diagnostic wrappers measure sparse-value damage, while paired FP32 captures provide depth and propagation-entry errors.

**Tech Stack:** Python 3.7/3.11, PyTorch, official NLSPN ResNet-34 and DCN extension, FP6 E3M2/FP8 E4M3FN fake quantization, pytest, CSV/JSON.

---

## File Structure

- Modify `spn_quant/selective_channel_smoothing.py`: reusable activation-damage accounting.
- Modify `scripts/run_nyu_nlspn_selective_channel_smoothing_fp6.py`: corrected evaluator, candidates, metrics, artifacts, and CLI mode.
- Create `tests/test_nlspn_fp6_activation_protection.py`: ownership, assignments, layouts, diagnostics, metrics, and schemas.
- Create `docs/results/2026-09-03-nlspn-fp6-activation-protection-results.md`: measured protocol and conclusions.
- Remove invalid runtime roots `profile_logs/nyu_nlspn_initial_depth_boundary_fp6_fp8_64_v1` through `v4` only after corrected artifacts pass validation.

### Task 1: Lock Single-QDQ Ownership

**Files:**
- Create: `tests/test_nlspn_fp6_activation_protection.py`
- Modify: `scripts/run_nyu_nlspn_selective_channel_smoothing_fp6.py`

- [ ] **Step 1: Write failing ownership tests**

Use test doubles to require concat-adapter disablement, empty active external ownership, and effective FP6 formats for both initial-depth weights:

```python
def test_corrected_fp_configuration_has_one_generic_owner():
    evaluator = _fake_evaluator()
    candidate = _candidate_for_test()
    _configure_corrected_fp_candidate(evaluator, candidate)
    assert evaluator.concat_adapter.calls == ["disable"]
    assert evaluator.instrumentor.external_ownership == (set(), set())
    assert evaluator.instrumentor.weight_formats["id_dec0.0"] == "fp6_e3m2"
    assert evaluator.instrumentor.weight_formats["id_dec1.0"] == "fp6_e3m2"
```

Add a second test requiring every official NLSPN activation owner used here to resolve to a module input/output boundary.

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=. pytest -q tests/test_nlspn_fp6_activation_protection.py \
  -k 'corrected_fp_configuration or generic_activation_configuration'
```

Expected: collection fails because the corrected helpers do not exist.

- [ ] **Step 3: Implement corrected configuration**

Add helpers that use direct attributes and mapping indexes:

```python
def _generic_fp_activation_configuration(evaluator, assignment):
    generic = {}
    for owner, format_name in assignment.activation_formats.items():
        site_name, role = owner
        site = evaluator.sites[site_name]
        if site.role != role:
            raise ValueError("FP activation role differs from contract")
        if site.owner_kind not in ("module_input", "module_output"):
            raise ValueError("NLSPN activation protection requires module sites")
        boundary = _site_boundary(site)
        if boundary in generic and generic[boundary] != format_name:
            raise ValueError("one FP boundary has multiple formats")
        generic[boundary] = format_name
    return generic


def _configure_corrected_fp_candidate(evaluator, candidate):
    evaluator.concat_adapter.disable()
    evaluator.instrumentor.set_external_ownership(set(), set())
    activation_formats = _generic_fp_activation_configuration(
        evaluator, candidate.assignment)
    evaluator.instrumentor.configure_floating_point(
        dict(candidate.assignment.weight_formats), activation_formats,
        set(evaluator.registry.blocks), external_output_ownership=True)
    evaluator.instrumentor.set_runtime_statistics(False)
    evaluator.instrumentor.relu_quantizers = {}
    evaluator.propagation_adapter.configure_fp16()
```

The new evaluator must not invoke `FPFormatEvaluator._configure_candidate`, because that method re-enables the old concat path. This is the enforced single-QDQ ownership boundary.

- [ ] **Step 4: Add an effective-weight regression**

With a real `Conv2d` and instrumentor, require the configured weight to equal direct per-output-channel FP6 QDQ and differ from the original:

```python
expected, _ = make_quantizer(
    "fp6_e3m2", maxima, broadcast_shape=(2, 1, 1, 1)
).quantize_with_codes(original)
torch.testing.assert_close(module.weight, expected, rtol=0.0, atol=0.0)
assert bool((module.weight != original).any().item())
```

- [ ] **Step 5: Run regressions**

```bash
PYTHONPATH=. pytest -q tests/test_nlspn_fp6_activation_protection.py \
  tests/test_fp_instrumentor.py tests/test_hardware_merge_adapters.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_nyu_nlspn_selective_channel_smoothing_fp6.py \
  tests/test_nlspn_fp6_activation_protection.py
git commit -m "fix: correct NLSPN FP concat ownership"
```

### Task 2: Measure Sparse And Branch Damage

**Files:**
- Modify: `spn_quant/selective_channel_smoothing.py`
- Modify: `tests/test_nlspn_fp6_activation_protection.py`

- [ ] **Step 1: Write failing accounting tests**

```python
def test_tracked_quantizer_separates_native_and_new_zero_codes():
    quantizer = TrackedFPQuantizer(
        make_quantizer("fp6_e3m2", torch.tensor(28.0)))
    quantizer.quantize_with_codes(torch.tensor([0.0, 0.01, 1.0, 40.0]))
    row = quantizer.diagnostics()
    assert row["native_zero_count"] == 1
    assert row["new_zero_count"] == 1
    assert row["reference_nonzero_count"] == 3
    assert row["saturation_count"] == 1
    assert math.isfinite(row["nonzero_sqnr_db"])
```

Extend the existing branch test to require the same fields per branch.

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=. pytest -q tests/test_nlspn_fp6_activation_protection.py \
  -k 'tracked_quantizer or branch_damage'
```

Expected: import or missing-key failure.

- [ ] **Step 3: Implement shared accounting**

Add `ActivationDamageAccumulator` and `TrackedFPQuantizer`. Proxy `format`, `bits`, `scale`, `spec`, `qmin`, `qmax`, `unsigned`, and `scale_for`. Update counts without changing reconstructed values:

```python
native = reference == 0
nonzero = ~native
normalized = reference / quantizer.scale_for(reference)
self.native_zero += int(native.sum().item())
self.reference_nonzero += int(nonzero.sum().item())
self.new_zero += int((nonzero & (codes == 0)).sum().item())
self.saturated += int(
    (normalized.abs() > quantizer.spec.maximum).sum().item())
self.nonzero_signal += float(
    reference[nonzero].double().square().sum().item())
error = reference[nonzero].double() - quantized[nonzero].double()
self.nonzero_error += float(error.square().sum().item())
```

Use the same accumulator for each branch inside `BranchIndependentFPQuantizer`.

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. pytest -q tests/test_nlspn_fp6_activation_protection.py \
  tests/test_nlspn_initial_depth_boundary.py \
  tests/test_selective_channel_smoothing.py
```

Expected: all tests pass and old quantized values remain bit-exact.

- [ ] **Step 5: Commit**

```bash
git add spn_quant/selective_channel_smoothing.py \
  tests/test_nlspn_fp6_activation_protection.py
git commit -m "feat: measure NLSPN activation quantization damage"
```

### Task 3: Implement The Eight Candidates

**Files:**
- Modify: `scripts/run_nyu_nlspn_selective_channel_smoothing_fp6.py`
- Modify: `tests/test_nlspn_fp6_activation_protection.py`

- [ ] **Step 1: Write failing matrix and layout tests**

Require exact candidate order:

```python
assert tuple(c.name for c in _activation_protection_candidates(base)) == (
    "BASE", "DEPTH_A8", "RGB_A8", "STEM_A8", "EARLY_A8",
    "STEM_EARLY_A8", "STEM_EARLY_ID_BRANCH", "FULL_BRANCH_AWARE")
```

Require exact official layouts:

```python
assert BRANCH_LAYOUTS == {
    "dec4.0": (256, 512), "dec3.0": (128, 256),
    "dec2.0": (64, 128), "id_dec1.0": (64, 64),
    "id_dec0.0": (64, 64),
}
```

Assert every weight stays FP6, both initial-depth inputs stay FP8, and candidates do not mutate earlier assignments.

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=. pytest -q tests/test_nlspn_fp6_activation_protection.py \
  -k 'candidate_matrix or branch_layout or assignment'
```

Expected: missing matrix failures.

- [ ] **Step 3: Implement constants and candidate type**

```python
STEM_MODULES = ("conv1_rgb.0", "conv1_dep.0")
EARLY_MODULES = (
    "conv2.0.conv1", "conv2.0.conv2", "conv3.0.downsample.0")
INITIAL_DEPTH_MODULES = ("id_dec1.0", "id_dec0.0")

@dataclass(frozen=True)
class ActivationProtectionCandidate:
    name: str
    assignment: FPFormatAssignment
    branch_independent_modules: tuple
```

Validate module existence, Conv type, activation owner, and exact input-channel sum.

- [ ] **Step 4: Install exactly one tracked input quantizer**

Wrap promoted common quantizers with `TrackedFPQuantizer`. Replace only declared branch-aware inputs with `BranchIndependentFPQuantizer`, deriving branch maxima from the existing channel observer and fixed layout. Require exactly two calls after paired evaluation.

- [ ] **Step 5: Add budget and Pareto tests**

Require average weight bits `6.0`, activation costs from `costs.activation_elements`, and the pooled-RMSE/`0.0001` m selection rule.

- [ ] **Step 6: Run and commit**

```bash
PYTHONPATH=. pytest -q tests/test_nlspn_fp6_activation_protection.py
git add scripts/run_nyu_nlspn_selective_channel_smoothing_fp6.py \
  tests/test_nlspn_fp6_activation_protection.py
git commit -m "feat: add NLSPN FP6 activation protection matrix"
```

Expected: tests pass before commit.

### Task 4: Add Paired Depth And Propagation Metrics

**Files:**
- Modify: `scripts/run_nyu_nlspn_selective_channel_smoothing_fp6.py`
- Modify: `tests/test_nlspn_fp6_activation_protection.py`

- [ ] **Step 1: Write failing metric tests**

Test pooled SSE, MAE, AbsRel, iRMSE, signal MSE/SQNR, iteration identity, and strict schemas:

```python
row = _depth_error_row(prediction, target, sample_index=7)
assert row["sample_index"] == 7
assert row["valid_pixels"] == 2
assert row["absolute_error_sum"] == pytest.approx(2.0)
signal = _signal_error_row("affinity", 0, reference, quantized)
assert signal["signal"] == "affinity"
assert signal["iteration"] == 0
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=. pytest -q tests/test_nlspn_fp6_activation_protection.py \
  -k 'depth_error or signal_error or metric_schema'
```

Expected: missing helper failures.

- [ ] **Step 3: Implement official-output capture**

Require `pred`, `pred_init`, `pred_inter`, `guidance`, `offset`, `aff`, and `confidence`. Require exactly 18 states. Cache one FP32 capture on CPU, execute every candidate twice, require exact signal/prediction reproducibility, and compare the first candidate pass against FP32:

```python
return {
    "prediction": output["pred"].detach().cpu(),
    "initial_depth": output["pred_init"].detach().cpu(),
    "guidance": output["guidance"].detach().cpu(),
    "offset": output["offset"].detach().cpu(),
    "affinity": output["aff"].detach().cpu(),
    "confidence": output["confidence"].detach().cpu(),
    "states": tuple(state.detach().cpu() for state in output["pred_inter"]),
}, ground_truth.detach().cpu()
```

- [ ] **Step 4: Implement strict artifacts**

Add `run_activation_protection()` and write:

```text
summary.csv
sample_metrics.csv
module_diagnostics.csv
branch_diagnostics.csv
effective_weight_metrics.csv
propagation_signal_metrics.csv
propagation_state_metrics.csv
pareto.csv
manifest.json
```

Include assignments, layouts, QDQ call counts, checkpoint identity, sample identities, FP32 metrics, and average bits. Reject an existing output root.

- [ ] **Step 5: Add CLI and schema tests**

Extend parser choices with `activation-protection` and map it directly to `run_activation_protection`. Empty or inconsistent rows must raise.

- [ ] **Step 6: Run and commit**

```bash
PYTHONPATH=. pytest -q tests/test_nlspn_fp6_activation_protection.py \
  tests/test_nlspn_initial_depth_boundary.py tests/test_fp_formats.py \
  tests/test_fp_instrumentor.py
git add scripts/run_nyu_nlspn_selective_channel_smoothing_fp6.py \
  tests/test_nlspn_fp6_activation_protection.py
git commit -m "feat: evaluate NLSPN FP6 activation protection"
```

Expected: tests pass before commit.

### Task 5: Verify Both Python Environments

**Files:**
- Modify only files already listed if a focused test exposes a defect.

- [ ] **Step 1: Run Python 3.11 regressions**

```bash
PYTHONPATH=. pytest -q \
  tests/test_nlspn_fp6_activation_protection.py \
  tests/test_nlspn_initial_depth_boundary.py \
  tests/test_selective_channel_smoothing.py tests/test_fp_formats.py \
  tests/test_fp_instrumentor.py tests/test_hardware_merge_adapters.py \
  tests/test_propagation_fp16_contract.py \
  tests/test_propagation_aware_adapters.py
```

Expected: all applicable tests pass.

- [ ] **Step 2: Run Python 3.7-compatible tests**

```bash
SPN_EXTERNAL_ROOT=/workspace/SPN_Quantization/external \
COMPLETIONFORMER_ROOT=/workspace/CompletionFormer PYTHONPATH=. \
/opt/conda/envs/completionformer-py37/bin/python -m pytest -q \
  tests/test_nlspn_fp6_activation_protection.py \
  tests/test_nlspn_initial_depth_boundary.py tests/test_fp_formats.py \
  tests/test_fp_instrumentor.py
```

Expected: all tests pass.

- [ ] **Step 3: Check syntax and whitespace**

```bash
python -m py_compile \
  scripts/run_nyu_nlspn_selective_channel_smoothing_fp6.py \
  spn_quant/selective_channel_smoothing.py \
  tests/test_nlspn_fp6_activation_protection.py
git diff --check
```

Expected: both commands exit zero.

### Task 6: Run And Validate The GPU Experiment

**Files:**
- Create runtime output: `profile_logs/nyu_nlspn_fp6_activation_protection_64_v1/`

- [ ] **Step 1: Verify GPU and DCN**

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader
SPN_EXTERNAL_ROOT=/workspace/SPN_Quantization/external \
COMPLETIONFORMER_ROOT=/workspace/CompletionFormer PYTHONPATH=. \
/opt/conda/envs/completionformer-py37/bin/python -c \
  'import torch, DCN; print(torch.cuda.device_count(), DCN.__file__)'
```

Expected: configured `cuda:1` exists and `DCN` resolves to the Python 3.7 extension.

- [ ] **Step 2: Run all candidates**

```bash
SPN_EXTERNAL_ROOT=/workspace/SPN_Quantization/external \
COMPLETIONFORMER_ROOT=/workspace/CompletionFormer PYTHONPATH=. \
/opt/conda/envs/completionformer-py37/bin/python \
  scripts/run_nyu_nlspn_selective_channel_smoothing_fp6.py \
  --config configs/four_model_unified_fp16_task_aware.json \
  --output-root /workspace/SPN_Quantization/profile_logs/nyu_nlspn_fp6_activation_protection_64_v1 \
  --experiment activation-protection
```

Expected: exit zero with eight valid rows.

- [ ] **Step 3: Validate artifacts**

Run a read-only Python check requiring eight finite results, 128 calibration identities, 64 evaluation identities, FP16 propagation, average weight bits exactly 6.0, two QDQ calls per tracked input, and changed FP6 weight elements for both initial-depth convolutions.

- [ ] **Step 4: Select the result**

Choose the lowest pooled RMSE; for differences below `0.0001` m choose lower average activation bits. Report every incremental delta from corrected `BASE`; retain no non-beneficial protection.

### Task 7: Clean Invalid Results And Document The Outcome

**Files:**
- Remove: `profile_logs/nyu_nlspn_initial_depth_boundary_fp6_fp8_64_v1/`
- Remove: `profile_logs/nyu_nlspn_initial_depth_boundary_fp6_fp8_64_v2/`
- Remove: `profile_logs/nyu_nlspn_initial_depth_boundary_fp6_fp8_64_v3/`
- Remove: `profile_logs/nyu_nlspn_initial_depth_boundary_fp6_fp8_64_v4/`
- Create: `docs/results/2026-09-03-nlspn-fp6-activation-protection-results.md`

- [ ] **Step 1: Delete old roots only after Task 6 passes**

```bash
rm -rf \
  /workspace/SPN_Quantization/profile_logs/nyu_nlspn_initial_depth_boundary_fp6_fp8_64_v1 \
  /workspace/SPN_Quantization/profile_logs/nyu_nlspn_initial_depth_boundary_fp6_fp8_64_v2 \
  /workspace/SPN_Quantization/profile_logs/nyu_nlspn_initial_depth_boundary_fp6_fp8_64_v3 \
  /workspace/SPN_Quantization/profile_logs/nyu_nlspn_initial_depth_boundary_fp6_fp8_64_v4
```

- [ ] **Step 2: Write the result document**

Record FP32, corrected `BASE`, all candidates, bit averages, incremental deltas, selected candidate, module/branch damage, propagation-entry/state errors, and the old ownership defect. State that this is PTQ fake quantization, not QAT or a native FP6 kernel.

- [ ] **Step 3: Final verification**

```bash
test -f profile_logs/nyu_nlspn_fp6_activation_protection_64_v1/manifest.json
test ! -e profile_logs/nyu_nlspn_initial_depth_boundary_fp6_fp8_64_v4
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 4: Commit tracked code and documentation**

```bash
git add scripts/run_nyu_nlspn_selective_channel_smoothing_fp6.py \
  spn_quant/selective_channel_smoothing.py \
  tests/test_nlspn_fp6_activation_protection.py \
  docs/results/2026-09-03-nlspn-fp6-activation-protection-results.md
git commit -m "results: validate NLSPN FP6 activation protection"
```

Runtime profile artifacts remain uncommitted unless the repository tracking policy already includes them.
