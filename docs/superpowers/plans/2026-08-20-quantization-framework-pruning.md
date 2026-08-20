# Quantization Framework Pruning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove retired quantization implementations and generated outputs while preserving the tested RTN, hardware-aligned, propagation-aware, Group8/QAT, AdaRound/BRECQ W4A8, QDrop W6A6, mixed-precision, and diagnostic workflows.

**Architecture:** Extract the signed activation-boundary quantizer currently coupled to channel rotation into a neutral module, then remove retired transformations from the shared runner and delete their dedicated entry points. Preserve conclusions in one tracked inventory before pruning ignored artifacts, and verify retained workflows with focused tests plus the complete suite.

**Tech Stack:** Python 3, PyTorch, NumPy, pytest, Bash, Git, CSV/JSON experiment manifests

**Spec:** `docs/superpowers/specs/2026-08-20-quantization-framework-pruning-design.md`

## Global Constraints

- Work only in `/workspace/SPN_Quantization/.worktrees/quantization-framework-pruning` on branch `cleanup/quantization-framework-pruning` until integration.
- Do not modify the pre-existing user changes in `/workspace/SPN_Quantization` or the `external/DySPN` submodule state.
- Access Python configuration attributes with `.` and mappings with `[]`; do not add `getattr`, mapping `.get()` defaults, `try/except` fallbacks, compatibility aliases, deprecated options, or hidden downgrade paths.
- Retired options must fail naturally as unknown values; they must not silently resolve to uniform MinMax quantization.
- Preserve MinMax calibration, activation diagnostics, contiguous Group8, dynamic Group8 QAT, propagation-aware SPN handling, and CompletionFormer attention contracts.
- Use repository paths for generated checks; do not write temporary files under `/tmp`.
- Delete ignored `profile_logs` data only after tracked metrics are copied to the inventory and checked against source CSV files.

---

### Task 1: Lock The Retained Quantization Contract

**Files:**
- Modify: `tests/test_quant_specs.py`
- Modify: `spn_quant/specs.py`

**Interfaces:**
- Consumes: `QuantSpec(bits, observer="minmax", transform="none")` and `run_nyu_rtn_quantization.SUPPORTED_QUANT_BACKENDS`.
- Produces: a quantization specification that accepts only `minmax` and `zero_aware` observers and only the `none` transform.

- [ ] **Step 1: Replace retired-transform tests with explicit rejection tests**

Add these assertions to `tests/test_quant_specs.py` and remove the test that calls `with_transform("lognp")`:

```python
def test_quant_spec_rejects_retired_observers():
    for observer in ("percentile", "mse"):
        with pytest.raises(ValueError, match="unknown observer"):
            QuantSpec.signed_tensor(4, observer=observer)


def test_quant_spec_rejects_retired_transforms():
    for transform in ("lognp", "smooth"):
        with pytest.raises(ValueError, match="unknown transform"):
            QuantSpec(bits=4, transform=transform)
```

- [ ] **Step 2: Run the contract tests and verify they fail against the current surface**

Run:

```bash
pytest -q tests/test_quant_specs.py
```

Expected: failures show that retired observers and transforms still exist.

- [ ] **Step 3: Narrow `QuantSpec` without adding compatibility behavior**

Change the constants and remove `with_transform` from `spn_quant/specs.py`:

```python
_VALID_OBSERVERS = frozenset(("minmax", "zero_aware"))
_VALID_TRANSFORMS = frozenset(("none",))
```

Keep the `transform` manifest field set to `none` so retained manifests remain explicit. Remove the LogNP wording from the class docstring.

- [ ] **Step 4: Run the focused specification tests**

Run: `pytest -q tests/test_quant_specs.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the retained contract**

```bash
git add spn_quant/specs.py tests/test_quant_specs.py
git commit -m "test: lock active quantization contract"
```

### Task 2: Extract Rotation-Independent Activation Boundaries

**Files:**
- Create: `spn_quant/activation_boundaries.py`
- Modify: `spn_quant/adapters/cspn.py`
- Modify: `spn_quant/adapters/__init__.py`
- Modify: `spn_quant/qat/quantizers.py`
- Modify: `scripts/run_nyu_cspn_activation_resolution.py`
- Modify: `scripts/evaluate_nyu_cspn_group_a4_qat.py`
- Modify: `scripts/train_nyu_cspn_group_a4_qat.py`
- Modify: `scripts/run_nyu_cspn_stem_precision.py`
- Modify: `scripts/run_nyu_cspn_dynamic_group_a4.py`
- Create: `tests/test_activation_boundaries.py`
- Modify: `tests/test_cspn_qat.py`
- Modify: `tests/test_qat_quantizers.py`
- Delete: `spn_quant/rotation.py`
- Delete: `scripts/run_nyu_cspn_rotation.py`
- Delete: `scripts/plot_nyu_cspn_rotation.py`
- Delete: `tests/test_rotation.py`
- Delete: `tests/test_run_nyu_cspn_rotation.py`
- Delete: `tests/test_plot_nyu_cspn_rotation.py`

**Interfaces:**
- Consumes: CSPN decoder boundary declarations and calibrated NCHW activations.
- Produces: `ActivationConsumer`, `ActivationBoundary`, `SignedActivationQuantizer`, `ActivationBoundaryObserver`, and `CSPNActivationBoundaryController` with identity-only observation and quantization.

- [ ] **Step 1: Define the neutral boundary API in tests**

Port only identity-path cases from `tests/test_rotation.py` into `tests/test_activation_boundaries.py`. Use these imports and object names:

```python
from spn_quant.activation_boundaries import (
    ActivationBoundaryObserver,
    CSPNActivationBoundaryController,
    SignedActivationQuantizer,
)
from spn_quant.adapters.cspn import ActivationBoundary, ActivationConsumer
```

The controller constructor and configuration calls must be:

```python
controller = CSPNActivationBoundaryController(model, boundaries)
controller.observe()
model(calibration_input)
controller.freeze()
controller.configure_specs(
    bit_widths={"decoder_entry": 4},
    group_sizes={"decoder_entry": 8},
    scale_factors={"decoder_entry": 1.0},
    quantize=True,
)
```

Assert that `manifest()` uses `boundary.decoder_entry`, reports `method == "identity"`, and contains no seed, rotation matrix, or transformed-weight entry.

- [ ] **Step 2: Run the new boundary tests and verify the module is missing**

Run: `pytest -q tests/test_activation_boundaries.py`

Expected: collection fails with `ModuleNotFoundError: spn_quant.activation_boundaries`.

- [ ] **Step 3: Implement the neutral boundary module**

Move `SignedActivationQuantizer` and the identity observer logic from `spn_quant/rotation.py` into `spn_quant/activation_boundaries.py`. Rename `RotationBoundaryObserver` to `ActivationBoundaryObserver`. Implement the controller with this public surface:

```python
class CSPNActivationBoundaryController:
    def __init__(self, model: nn.Module,
                 boundaries: Sequence[ActivationBoundary]) -> None: ...
    def observe(self) -> None: ...
    def freeze(self) -> None: ...
    def configure_specs(
            self, bit_widths: Mapping[str, int],
            group_sizes: Mapping[str, Optional[int]],
            scale_factors: Mapping[str, float],
            quantize: bool = True) -> None: ...
    def configure_specs_with_ranges(
            self, bit_widths: Mapping[str, int],
            group_sizes: Mapping[str, Optional[int]],
            scale_factors: Mapping[str, float],
            maximum_overrides: Mapping[str, torch.Tensor],
            quantize: bool = True) -> None: ...
    def set_activation_recorder(self, recorder: object) -> None: ...
    def set_calibration_recorder(self, recorder: object) -> None: ...
    def clear_recorders(self) -> None: ...
    def disable(self) -> None: ...
    def close(self) -> None: ...
    def manifest(self) -> Sequence[Mapping[str, object]]: ...
```

Hooks observe and quantize the boundary tensor directly. Recorder site names use the `boundary.decoder_entry` form. Do not carry `METHODS`, random/Hadamard matrices, seeds, input-weight transforms, `absorb_weights`, or `weight_source_overrides` into the new module.

- [ ] **Step 4: Rename the CSPN adapter declarations**

In `spn_quant/adapters/cspn.py`, rename the dataclasses and the adapter method:

```python
@dataclass(frozen=True)
class ActivationConsumer:
    module: str
    channel_start: int
    channel_count: Optional[int]


@dataclass(frozen=True)
class ActivationBoundary:
    name: str
    module: str
    argument_index: int
    consumers: Tuple[ActivationConsumer, ...]
```

Rename `rotation_boundaries()` to `activation_boundaries()` and export only the new names from `spn_quant/adapters/__init__.py`.

- [ ] **Step 5: Migrate retained callers and QAT quantizers**

Replace imports of `spn_quant.rotation` in retained files with `spn_quant.activation_boundaries`. Replace controller construction with the two-argument constructor, remove identity-method mappings and `absorb_weights`, and rename local variables from `rotation` to `boundaries` or `boundary_controller`. Update manifest and recorder keys from forms such as `rotation.decoder_entry` to `boundary.decoder_entry`.

- [ ] **Step 6: Remove the rotation implementation and dedicated workflow**

Delete the rotation module, runner, plotter, and dedicated tests listed above. Do not delete activation-resolution, stem-precision, dynamic Group8, or QAT workflows.

- [ ] **Step 7: Run the retained boundary and QAT tests**

Run:

```bash
pytest -q tests/test_activation_boundaries.py tests/test_qat_quantizers.py \
  tests/test_cspn_qat.py tests/test_run_nyu_cspn_activation_resolution.py \
  tests/test_run_nyu_cspn_stem_precision.py \
  tests/test_run_nyu_cspn_dynamic_group_a4.py
```

Expected: all tests pass and no retained file imports `spn_quant.rotation`.

- [ ] **Step 8: Commit the boundary extraction**

```bash
git add -A spn_quant scripts tests
git commit -m "refactor: separate activation boundaries from rotation"
```

### Task 3: Remove Retired Shared Activation Transformations

**Files:**
- Modify: `scripts/hardware_aligned_quantization.py`
- Modify: `scripts/run_nyu_rtn_quantization.py`
- Modify: `scripts/plot_activation_outlier_analysis.py`
- Modify: `tests/test_hardware_aligned_quantization.py`
- Modify: `tests/test_run_nyu_rtn_quantization.py`
- Modify: `tests/test_plot_activation_outlier_analysis.py`
- Delete: `scripts/lognp_quantization.py`
- Delete: `scripts/plot_lognp_quantization.py`
- Delete: `scripts/outlier_mitigation_quantization.py`
- Delete: `scripts/run_nyu_cspn_smoothquant_group.py`
- Delete: `scripts/plot_cspn_smoothquant_group.py`
- Delete: all tests dedicated to the five deleted scripts

**Interfaces:**
- Consumes: retained uniform signed/unsigned QDQ, per-tensor/contiguous Group8 activation contracts, and weight-source overrides used by AdaRound/BRECQ.
- Produces: a hardware instrumentor with no LogNP, SmoothQuant, AWQ clipping, percentile override, compensation, or retired backend dispatch.

- [ ] **Step 1: Add explicit shared-surface retirement tests**

Add this test to `tests/test_run_nyu_rtn_quantization.py`:

```python
def test_retired_backends_are_not_advertised():
    assert "lognp" not in runner.SUPPORTED_QUANT_BACKENDS
    assert "outlier" not in runner.SUPPORTED_QUANT_BACKENDS
    assert "fp4" not in runner.SUPPORTED_QUANT_BACKENDS
```

Import `inspect` and add this test to `tests/test_hardware_aligned_quantization.py`:

```python
def test_hardware_instrumentor_has_no_retired_activation_modes():
    source = inspect.getsource(HardwareAlignedInstrumentor)
    for retired in ("lognp", "smoothquant", "awq", "percentile"):
        assert retired not in source.lower()
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
pytest -q \
  tests/test_run_nyu_rtn_quantization.py::test_retired_backends_are_not_advertised \
  tests/test_hardware_aligned_quantization.py::test_hardware_instrumentor_has_no_retired_activation_modes
```

Expected: both tests fail because the retired surface is still present.

- [ ] **Step 3: Narrow existing shared-core tests to retained behavior**

Remove tests that configure `lognp`, `smoothquant`, AWQ, percentile clipping, or compensation. Keep and run tests for Conv-BN folding, unsigned ReLU outputs, Group8 scales, INT32 bias, Add/Concat requantization, propagation-aware exclusion, weight-source overrides, and activation recording.

- [ ] **Step 4: Remove retired imports and constructor parameters**

Delete imports from `scripts.lognp_quantization`, `scripts.outlier_mitigation_quantization`, `scripts.fp4_quantization`, and `spn_quant.scale_aware_grouping`. Remove constructor/configuration fields whose only consumers are `lognp`, `smoothquant`, `awq`, percentile clipping, scale-aware permutation, or OCI instrumentation.

The active activation quantizer selection must reduce to explicit uniform logic:

```python
if spec.signed:
    quantizer = SignedActivationQuantizer(
        bits=spec.bits,
        channel_maximum=channel_maximum,
        group_size=spec.group_size,
    )
else:
    quantizer = AffineActivationQuantizer.from_observer(observer, spec)
```

No branch may interpret an unknown transform or observer.

- [ ] **Step 5: Remove retired backend builders and CLI options**

In `scripts/run_nyu_rtn_quantization.py`, remove retired imports, `build_outlier_configurations`, `build_lognp_configurations`, LogNP compensation selection, FP4 configuration construction, retired CLI arguments, dispatch branches, and retired result metadata. Keep hardware, mixed, propagation, strict reconstruction, and CompletionFormer paths.

Set the backend tuple to the names still handled by the dispatch. Tests must compare the tuple to the actual retained set rather than checking a subset.

- [ ] **Step 6: Keep outlier diagnostics but remove mitigation ranking**

In `scripts/plot_activation_outlier_analysis.py`, retain plots/tables for p75, p99, p99.9, p99.99, maximum, zero ratio, saturation, kurtosis, and channel imbalance. Remove SmoothQuant/LogNP/AWQ recommendation labels and mitigation score columns.

- [ ] **Step 7: Delete dedicated retired transformation files**

Delete the LogNP, SmoothQuant, and outlier-mitigation scripts and tests listed in this task. Dedicated historical design/result documents are deleted in Task 7 after their metrics enter the inventory.

- [ ] **Step 8: Run shared quantization tests**

Run:

```bash
pytest -q tests/test_hardware_aligned_quantization.py \
  tests/test_run_nyu_rtn_quantization.py \
  tests/test_plot_activation_outlier_analysis.py \
  tests/test_adaptive_rounding.py \
  tests/test_strict_reconstruction.py
```

Expected: all tests pass; `rg -n 'lognp|smoothquant|awq' spn_quant scripts tests` returns no active implementation reference.

- [ ] **Step 9: Commit shared-core pruning**

```bash
git add -A scripts spn_quant tests
git commit -m "refactor: remove retired activation transformations"
```

### Task 4: Remove Retired Calibration And Grouping Experiments

**Files:**
- Delete: `spn_quant/static_calibration.py`
- Delete: `spn_quant/scale_aware_grouping.py`
- Delete: `spn_quant/outlier_channel_isolation.py`
- Delete: `scripts/run_nyu_cspn_static_calibration.py`
- Delete: `scripts/plot_cspn_static_calibration.py`
- Delete: `scripts/run_nyu_cspn_scale_aware_grouping.py`
- Delete: `scripts/plot_cspn_scale_aware_grouping.py`
- Delete: `scripts/run_nyu_cspn_outlier_channel_isolation.py`
- Delete: corresponding dedicated test files
- Modify: retained imports/tests found by the repository search below

**Interfaces:**
- Consumes: contiguous Group8 and MinMax calibration from the shared hardware path.
- Produces: no selectable percentile/histogram-MSE observer, scale-aware permutation, RMS/Max cluster, Snake spread, outlier-channel isolation, OCI, or OCS experiment.

- [ ] **Step 1: Prove all remaining references are either dedicated or shared-core leftovers**

Run:

```bash
rg -n "static_calibration|scale_aware|channel_isolation|OCI|OCS|snake|rms_cluster|max_cluster" \
  spn_quant scripts tests
```

Classify every match: delete dedicated files; remove shared-core branches; retain only natural-language historical conclusions until Task 7.

- [ ] **Step 2: Delete dedicated modules, runners, plotters, and tests**

Delete every file listed in this task. Do not replace them with stubs or aliases.

- [ ] **Step 3: Remove shared imports and instrumentation**

Remove OCI counters, split-channel manifests, permutations, cluster assignments, and caller arguments from retained files. Keep contiguous channel order for Group8 and the existing per-output-channel weight quantization.

- [ ] **Step 4: Verify retained Group8 and MinMax behavior**

Run:

```bash
pytest -q tests/test_hardware_aligned_quantization.py \
  tests/test_activation_resolution.py \
  tests/test_run_nyu_cspn_dynamic_group_a4.py \
  tests/test_run_nyu_cspn_stratified_calibration.py
```

Expected: all tests pass and the repository search from Step 1 returns no code matches.

- [ ] **Step 5: Commit calibration/grouping pruning**

```bash
git add -A spn_quant scripts tests
git commit -m "refactor: remove retired calibration and grouping experiments"
```

### Task 5: Remove E2M1 And Preserve Activation Diagnostics

**Files:**
- Delete: `scripts/fp4_quantization.py`
- Delete: `scripts/fp4_activation_validation.py`
- Delete: `scripts/analyze_fp4_activation_validation.py`
- Delete: `scripts/plot_fp4_activation_validation.py`
- Delete: `scripts/run_fp4_activation_validation.sh`
- Delete: `scripts/strict_w4a4_fp4_evaluation.py`
- Delete: `scripts/analyze_strict_w4a4_fp4_evaluation.py`
- Delete: `scripts/plot_strict_w4a4_fp4_evaluation.py`
- Delete: `scripts/run_strict_w4a4_fp4_evaluation.sh`
- Delete: dedicated FP4 and strict-W4A4-FP4 tests
- Modify: `scripts/run_w4a4_activation_histograms.py`
- Modify: `scripts/run_completionformer_joint_quantization.sh`
- Modify: `tests/test_w4a4_activation_histogram_runner.py`
- Create: `tests/test_completionformer_joint_shell_contract.py`

**Interfaces:**
- Consumes: standard hardware/propagation-aware W4A4 configuration and explicit CompletionFormer reference metrics.
- Produces: activation histograms that no longer depend on FP4 configuration helpers and a CompletionFormer script with no generated-artifact fallback path.

- [ ] **Step 1: Add histogram and shell contract tests**

The histogram test must assert that the profile configuration is a retained integer configuration and that no import from `fp4_activation_validation` or `fp4_quantization` exists. The shell test must assert this exact required environment expression:

```bash
REFERENCE_METRICS="${COMPLETIONFORMER_REFERENCE_METRICS:?COMPLETIONFORMER_REFERENCE_METRICS is required}"
```

- [ ] **Step 2: Run the new contract tests and verify they fail**

Run:

```bash
pytest -q tests/test_w4a4_activation_histogram_runner.py \
  tests/test_completionformer_joint_shell_contract.py
```

Expected: failures identify the current FP4 helper imports and default artifact path.

- [ ] **Step 3: Migrate histogram configuration to integer W4A4**

Move `resolve_per_channel_activation_inputs(model_name, module_names)` into `scripts/run_nyu_rtn_quantization.py` if it is still required by both callers. It must use direct model-name indexing and raise `ValueError` for unsupported models. Build the histogram runner from the retained propagation-aware/hardware W4A4 builder and name the profile `PA_W4A4_PROP_A8` rather than `FP4V_W4A4`.

- [ ] **Step 4: Require explicit CompletionFormer reference metrics**

Replace the default path in `scripts/run_completionformer_joint_quantization.sh` with the required environment expression from Step 1. Do not catch or replace the shell error.

- [ ] **Step 5: Delete E2M1 code and tests**

Delete all dedicated files listed above. Keep `scripts/strict_reconstruction.py`, `scripts/run_nyu_strict_reconstruction.py`, strict W4A8 orchestration, and their tests.

- [ ] **Step 6: Run retained diagnostics and reconstruction tests**

Run:

```bash
pytest -q tests/test_w4a4_activation_histogram_runner.py \
  tests/test_activation_histograms.py \
  tests/test_strict_reconstruction.py \
  tests/test_strict_reconstruction_runner.py \
  tests/test_cspn_selective_w4a8.py
```

Expected: all tests pass and `rg -n 'E2M1|fp4' spn_quant scripts tests` returns no active code reference.

- [ ] **Step 7: Commit E2M1 removal**

```bash
git add -A scripts tests spn_quant
git commit -m "refactor: retire E2M1 activation quantization"
```

### Task 6: Validate Every Retained Framework Before Documentation Cleanup

**Files:**
- Modify only files required by a failing retained test; do not broaden scope.

**Interfaces:**
- Consumes: the pruned implementation from Tasks 1-5.
- Produces: passing focused tests for every retained framework category.

- [ ] **Step 1: Search for retired implementation names**

Run:

```bash
rg -n -i "lognp|e2m1|smoothquant|awq|hadamard|random rotation|scale.aware|snake spread|outlier.channel.isolation|observer.*percentile|histogram.mse" \
  spn_quant scripts tests
```

Expected: no active implementation, CLI, import, or selectable configuration matches. Assertions that retired values are rejected may remain in tests.

- [ ] **Step 2: Run retained framework gates**

Run:

```bash
pytest -q \
  tests/test_quant_specs.py \
  tests/test_hardware_aligned_quantization.py \
  tests/test_edge_runtime.py \
  tests/test_propagation_aware_adapters.py \
  tests/test_propagation_quantization.py \
  tests/test_activation_resolution.py \
  tests/test_run_nyu_cspn_dynamic_group_a4.py \
  tests/test_cspn_qat.py \
  tests/test_qdrop_reconstruction.py \
  tests/test_adaptive_rounding.py \
  tests/test_strict_reconstruction.py \
  tests/test_cspn_selective_w4a8.py \
  tests/test_cspn_task_sensitive_bits.py \
  tests/test_completionformer_attention.py \
  tests/test_run_nyu_cspn_stratified_calibration.py \
  tests/test_activation_histograms.py \
  tests/test_im2col_diagnostics.py
```

Expected: all retained gates pass.

- [ ] **Step 3: Fix only direct regressions and rerun the failed gate**

For any failure, inspect the first traceback, remove the stale retired argument/import at its source, and rerun that exact test file. Do not add fallback branches or accept retired names.

- [ ] **Step 4: Commit retained-framework fixes**

```bash
git add -A spn_quant scripts tests
git commit -m "fix: preserve retained quantization workflows"
```

### Task 7: Consolidate The Active Inventory And Remove Retired Documentation

**Files:**
- Create: `docs/2026-08-20-quantization-framework-inventory.md`
- Modify: `README.md`
- Delete: dedicated retired method design, plan, and result pages listed by the inventory command
- Retain and annotate: cross-method result pages still supporting AdaRound/BRECQ W4A8, QDrop W6A6, QAT, or mixed precision

**Interfaces:**
- Consumes: final source metrics from the approved design and active rerun commands from retained scripts.
- Produces: one tracked source of truth for supported frameworks, retired frameworks, evidence, commands, and post-cleanup artifact locations.

- [ ] **Step 1: Verify source metrics before writing the inventory**

Use `rg --files profile_logs | rg '\\.(csv|json)$'` to locate source rows, then run a repository-local Python assertion script from standard input. It must assert these exact metre RMSE values before any artifact deletion:

```python
expected = {
    "smoothquant_w4_only_reference": 0.204621,
    "smoothquant_w4_only_best": 0.204136,
    "smoothquant_w4a4_group8_reference": 0.347124,
    "smoothquant_w4a4_group8_best": 0.359289,
    "smoothquant_w4a4_group16_reference": 0.417303,
    "smoothquant_w4a4_group16_best": 0.407825,
    "rotation_group_w4a4": 0.426802,
    "rotation_best": 0.430580,
    "contiguous_group8": 0.313318,
    "scale_aware_group8": 0.316218,
    "percentile_p999": 1.354118,
    "percentile_p9999": 0.490240,
    "histogram_mse": 0.988717,
}
```

Read the concrete CSV columns with `csv.DictReader`, index mappings with `[]`, and use `math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=5e-7)`. Let missing files, rows, or columns raise naturally.

- [ ] **Step 2: Write the active inventory**

Create `docs/2026-08-20-quantization-framework-inventory.md` with these sections:

1. Active deployment frameworks and bit-width configurations.
2. Active research/diagnostic frameworks.
3. Retired methods and the exact metrics above.
4. Retired configurations inside retained frameworks.
5. Current artifact roots retained after cleanup.
6. Exact rerun commands for RTN W8A8/W4A8, P3/T3, AdaRound/BRECQ W4A8, QDrop W6A6, static/dynamic Group8 QAT, CompletionFormer attention, activation histograms, and im2col diagnostics.
7. Disk cleanup table with before size, after size, and reclaimed bytes filled in Task 8.

State explicitly that SmoothQuant produced only a 0.24% W4-only gain, worsened Group8 W4A4, and is not active.

- [ ] **Step 3: Remove retired method-specific documentation**

Delete LogNP, E2M1/FP4-only, rotation-only, SmoothQuant-only, static-calibration-only, scale-aware-only, and OCI-only specs/plans/results after confirming every conclusion appears in the inventory. Keep `docs/2026-08-07-strict-w4a4-fp4-reconstruction-results.md` only if it also remains the source for retained W4A8 reconstruction conclusions; rename its heading to describe the cross-method reconstruction study and mark E2M1 retired.

- [ ] **Step 4: Link the inventory from the README**

Add one concise link in the existing quantization documentation section; do not create a second framework list in `README.md`.

- [ ] **Step 5: Validate documentation paths and commit**

Run:

```bash
python -m compileall -q spn_quant scripts
git diff --check
rg -n "quantization-framework-inventory" README.md
```

Expected: commands succeed.

```bash
git add -A README.md docs
git commit -m "docs: consolidate quantization framework inventory"
```

### Task 8: Prune Generated Artifacts And Record Storage Recovery

**Files:**
- Modify: `docs/2026-08-20-quantization-framework-inventory.md`
- Delete ignored paths under `/workspace/SPN_Quantization/profile_logs`

**Interfaces:**
- Consumes: checked inventory metrics and retained QDrop W6A6 outputs.
- Produces: pruned generated data, auditable retained QDrop W6A6 outputs, and exact storage recovery figures.

- [ ] **Step 1: Record initial size and exact root existence**

Run:

```bash
du -sb /workspace/SPN_Quantization/profile_logs
for path in \
  nyu_cspn_rotation_w4a4 \
  nyu_cspn_smoothquant_group \
  nyu_cspn_smoothquant_group_stratified128 \
  nyu_cspn_static_calibration_group8 \
  nyu_cspn_scale_aware_group8 \
  nyu_cspn_outlier_grouping_64 \
  nyu_cspn_outlier_channel_isolation_64 \
  nyu_cspn_selective_w4a8_boundary_search_64.incomplete \
  nyu_strict_w4a4_fp4_evaluation; do
  test -d "/workspace/SPN_Quantization/profile_logs/$path"
done
```

Expected: the byte count is printed and every retired root exists before deletion.

- [ ] **Step 2: Preserve active QDrop W6A6 data outside the mixed retired root**

Create `/workspace/SPN_Quantization/profile_logs/nyu_cspn_qdrop_w6a6_64` and move only QDrop W6A6 reconstruction contracts, evaluations, aggregate rows, and W6A6 prediction figures from `nyu_cspn_unified_brecq_qdrop_w4a4_w6a6_64`. Regenerate `manifest.json` as a sorted mapping from relative file path to SHA-256 for every retained file except the manifest itself. Do not preserve RTN/BRECQ W4A4 raw predictions or BRECQ W6A6 raw predictions.

The manifest root contains `algorithm` with value `sha256` and `files`, whose keys are relative paths and whose values are 64-character lowercase hexadecimal hashes. Use direct dictionary indexing when generating and auditing this manifest.

- [ ] **Step 3: Delete retired experiment roots**

Run the exact approved deletion after the retained move and manifest audit pass:

```bash
rm -rf \
  /workspace/SPN_Quantization/profile_logs/nyu_cspn_rotation_w4a4 \
  /workspace/SPN_Quantization/profile_logs/nyu_cspn_smoothquant_group \
  /workspace/SPN_Quantization/profile_logs/nyu_cspn_smoothquant_group_stratified128 \
  /workspace/SPN_Quantization/profile_logs/nyu_cspn_static_calibration_group8 \
  /workspace/SPN_Quantization/profile_logs/nyu_cspn_scale_aware_group8 \
  /workspace/SPN_Quantization/profile_logs/nyu_cspn_outlier_grouping_64 \
  /workspace/SPN_Quantization/profile_logs/nyu_cspn_outlier_channel_isolation_64 \
  /workspace/SPN_Quantization/profile_logs/nyu_cspn_selective_w4a8_boundary_search_64.incomplete \
  /workspace/SPN_Quantization/profile_logs/nyu_strict_w4a4_fp4_evaluation \
  /workspace/SPN_Quantization/profile_logs/nyu_cspn_unified_brecq_qdrop_w4a4_w6a6_64
```

- [ ] **Step 4: Check whether the older propagation-aware root is truly duplicated**

Compare relative path, size, and SHA-256 entries for `nyu_propagation_aware_quantization` against `nyu_propagation_aware_quantization_unified`. Delete the older root only if every old relative file has an identical unified counterpart. Otherwise retain both and record that the supersession proof failed.

- [ ] **Step 5: Audit retained data and record recovered bytes**

Run:

```bash
du -sb /workspace/SPN_Quantization/profile_logs
test -d /workspace/SPN_Quantization/profile_logs/nyu_cspn_qdrop_w6a6_64
test ! -e /workspace/SPN_Quantization/profile_logs/nyu_cspn_rotation_w4a4
test ! -e /workspace/SPN_Quantization/profile_logs/nyu_strict_w4a4_fp4_evaluation
```

Audit every retained manifest hash, then put the exact before bytes, after bytes, reclaimed bytes, and optional propagation-root decision into the inventory cleanup table.

- [ ] **Step 6: Commit the cleanup record**

```bash
git add docs/2026-08-20-quantization-framework-inventory.md
git commit -m "docs: record quantization artifact cleanup"
```

### Task 9: Full Verification And Integration Readiness

**Files:**
- Modify only direct defects discovered by verification.

**Interfaces:**
- Consumes: all completed cleanup tasks.
- Produces: a clean, fully tested branch ready to merge into `main`.

- [ ] **Step 1: Run syntax and repository hygiene checks**

Run:

```bash
python -m compileall -q spn_quant scripts tests
git diff --check main...HEAD
git status --short
```

Expected: compile and diff checks succeed; status contains no uncommitted changes before any final correction.

- [ ] **Step 2: Run the complete test suite**

Run: `pytest -q`

Expected: zero failures. The pass count is lower than the 1359-test baseline because dedicated retired-framework tests were intentionally deleted.

- [ ] **Step 3: Run final retired-surface and retained-surface searches**

Run:

```bash
rg -n -i "lognp|e2m1|smoothquant|awq|spn_quant\.rotation|scale_aware_grouping|outlier_channel_isolation" \
  spn_quant scripts tests
rg -n "AdaRound|BRECQ|QDrop|Group8|propagation|attention" \
  docs/2026-08-20-quantization-framework-inventory.md
```

Expected: the first command has only explicit rejection assertions if any; the second confirms retained frameworks are documented.

- [ ] **Step 4: Inspect branch scope**

Run:

```bash
git status --short --branch
git diff --stat main...HEAD
git submodule status
```

Expected: clean cleanup branch, no submodule changes, and no edits to the unrelated root-worktree files.

- [ ] **Step 5: Apply and commit any verification-only correction**

If verification required a direct correction, rerun its failing command and commit only those files:

```bash
git add -u
git add spn_quant scripts tests docs README.md
git commit -m "fix: complete quantization framework cleanup"
```

If no correction was required, do not create an empty commit.
