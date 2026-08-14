# CSPN Decoder and Initial-Depth Sensitivity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run a strict two-stage W4A8/W8A4/W8A8 sensitivity search over the official CSPN decoder and initial-depth head.

**Architecture:** Extend the existing hardware instrumentor with explicit per-module weight-bit overrides, keep candidate discovery/ranking/Pareto logic in a focused pure-Python module, and add a CSPN runner that reuses the official checkpoint, dataset protocol, stem contract, and propagation-aware backend. Every configuration gets a fresh model and calibration pass; Stage 2 is derived only from completed Stage-1 RMSE results.

**Tech Stack:** Python 3.11, PyTorch CUDA, NumPy, Matplotlib, pytest, existing CSPN quantization adapters.

---

### Task 1: Per-Module Weight-Bit Overrides

**Files:**
- Modify: `scripts/hardware_aligned_quantization.py`
- Modify: `scripts/run_nyu_cspn_activation_resolution.py`
- Test: `tests/test_hardware_aligned_quantization.py`
- Test: `tests/test_run_nyu_cspn_activation_resolution.py`

- [ ] **Step 1: Write a failing instrumentor test**

Configure two executed Conv modules through
`configure_components_with_ranges` with
`weight_bit_overrides={"decoder": 8}`
and assert `weight_bits_by_module()` reports W4 for the unselected module and
W8 for `decoder`.

- [ ] **Step 2: Run the test and verify RED**

```bash
python -m pytest -q tests/test_hardware_aligned_quantization.py -k components_with_weight_bit_overrides
```

Expected: fail because `configure_components_with_ranges` does not accept the
override.

- [ ] **Step 3: Pass the explicit override into `configure`**

Add `weight_bit_overrides=None` to
`configure_components_with_ranges` and pass it unchanged as the named
`weight_bit_overrides` argument in the existing `self.configure` call.

Keep the existing unknown-module and 4/8-bit validation authoritative.

- [ ] **Step 4: Write a failing CSPN configuration-schema test**

Assert `_configuration` with `weight_bit_overrides=(("module", 8),)` stores the
tuple, `_derived_configuration` preserves it, and `_configure_quantized`
forwards `{"module": 8}`.

- [ ] **Step 5: Run the schema test and verify RED**

```bash
python -m pytest -q tests/test_run_nyu_cspn_activation_resolution.py -k weight_bit_overrides
```

- [ ] **Step 6: Extend the strict CSPN configuration contract**

Add `weight_bit_overrides=()` to `_configuration`, preserve it in derived
configurations, and pass `dict(config["weight_bit_overrides"])` to the
instrumentor. Every locally generated configuration must contain this field;
do not use dictionary fallback access.

- [ ] **Step 7: Verify and commit**

```bash
python -m pytest -q tests/test_hardware_aligned_quantization.py tests/test_run_nyu_cspn_activation_resolution.py
git add scripts/hardware_aligned_quantization.py scripts/run_nyu_cspn_activation_resolution.py tests/test_hardware_aligned_quantization.py tests/test_run_nyu_cspn_activation_resolution.py
git commit -m "feat: support CSPN per-module weight precision"
```

### Task 2: Candidate Registry and Search Logic

**Files:**
- Create: `spn_quant/cspn_sensitivity.py`
- Create: `tests/test_cspn_sensitivity.py`

- [ ] **Step 1: Write failing registry tests**

Exercise this public contract:

```python
registry = build_candidate_registry(executed_modules, activation_owners)
stage1 = build_stage1_candidates(registry)
```

Assert five ordered blocks, 16 Stage-1 configurations including strict, exact
weight-module coverage, exact activation-owner coverage, and hard failure for
missing or unknown official candidates.

- [ ] **Step 2: Run registry tests and verify RED**

```bash
python -m pytest -q tests/test_cspn_sensitivity.py -k registry
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement immutable types and the exact registry**

```python
@dataclass(frozen=True)
class SensitivityCandidate:
    name: str
    stage: str
    block: str
    mode: str
    weight_modules: Tuple[str, ...]
    activation_owners: Tuple[Tuple[str, str], ...]

@dataclass(frozen=True)
class CandidateRegistry:
    weights_by_block: Mapping[str, Tuple[str, ...]]
    activations_by_block: Mapping[str, Tuple[Tuple[str, str], ...]]
    input_dependencies: Mapping[str, Tuple[Tuple[str, str], ...]]
```

Declare all 16 decoder/depth-head Conv modules and exact input dependencies.
Include both decoder branches for `gud_up_proj_layer4.conv1_1`.

- [ ] **Step 4: Write failing ranking and site tests**

Test that block ranking uses the best RMSE across W4A8/W8A4/W8A8, returns two
stable blocks, and selects the least-regressed blocks when no block improves.
Test individual activation W4A8, weight W8A4, and dependency-aware Conv W8A8
candidate generation.

- [ ] **Step 5: Implement Stage-2 selection**

Implement deterministic names, ranking, and site generation. Reject duplicate
names, owners, modules, and dependencies outside the strict registry.

- [ ] **Step 6: Write failing cumulative, cost, and Pareto tests**

Cover promotion union, duplicate elimination, improving-site ordering by RMSE
delta/cost/name, normalized bit-element cost, and non-dominated filtering,
including equal-cost and equal-RMSE cases.

- [ ] **Step 7: Implement cumulative and Pareto functions**

```python
build_cumulative_candidates(site_metrics, site_candidates, strict_rmse)
normalized_added_bit_cost(weight_rows, activation_rows, candidate)
pareto_rows(metric_rows)
```

Use finite-value checks and direct dictionary indexing. Return Pareto rows in
ascending cost then RMSE order.

- [ ] **Step 8: Verify and commit**

```bash
python -m pytest -q tests/test_cspn_sensitivity.py
git add spn_quant/cspn_sensitivity.py tests/test_cspn_sensitivity.py
git commit -m "feat: define CSPN decoder sensitivity search"
```

### Task 3: Strict NYU Sensitivity Runner

**Files:**
- Create: `scripts/run_nyu_cspn_decoder_sensitivity.py`
- Create: `tests/test_run_nyu_cspn_decoder_sensitivity.py`

- [ ] **Step 1: Write failing configuration translation tests**

Test that `hardware_configuration(candidate)` retains Group-8 W4A4 globally,
sets `promoted_owners` to the candidate owners, sets weight overrides to 8,
keeps the stem W4A4, excludes guidance, and retains `PROPAGATION_A8_Q13`.

- [ ] **Step 2: Run translation tests and verify RED**

```bash
python -m pytest -q tests/test_run_nyu_cspn_decoder_sensitivity.py -k hardware
```

- [ ] **Step 3: Implement configuration and protocol loading**

Reuse explicit source/checkpoint validation, `from_scratch=True`, stem
ownership, fold threshold, calibration, and fixed index protocol from
`run_nyu_cspn_stem_precision.py`. Do not add model/path fallbacks.

- [ ] **Step 4: Write failing metric and coverage tests**

Test exact 64-row coverage, finite metrics, paired wins, block capture,
operation precision rows, activation-element counts, and rejection of an
existing output directory.

- [ ] **Step 5: Implement one-candidate evaluation**

For each candidate: load a fresh checkpoint, prepare folding, calibrate 128
samples, apply weight and activation promotions, evaluate 64 samples against
the shared FP32 reference, return all metric rows, close hooks/controllers, and
release the model. This pass does not persist predictions.

- [ ] **Step 6: Write failing orchestration tests**

Use deterministic synthetic results to prove Stage 1 completes before Stage 2
construction, exactly two blocks are selected, cumulative candidates use only
improving sites, and prediction configuration names are deterministic.

- [ ] **Step 7: Implement two-stage orchestration and outputs**

Run strict plus 15 block candidates, derive and run Stage 2, run cumulative
candidates, and compute Pareto rows. Write sample, regional, block,
propagation, operation, activation, precision, stage summary, cumulative, and
Pareto CSVs plus a manifest. Rerun only selected prediction configurations to
write 64 payloads each.

- [ ] **Step 8: Verify and commit**

```bash
python -m pytest -q tests/test_run_nyu_cspn_decoder_sensitivity.py tests/test_cspn_sensitivity.py
git add scripts/run_nyu_cspn_decoder_sensitivity.py tests/test_run_nyu_cspn_decoder_sensitivity.py
git commit -m "feat: evaluate CSPN decoder precision sensitivity"
```

### Task 4: Pareto Plotter

**Files:**
- Create: `scripts/plot_nyu_cspn_decoder_sensitivity.py`
- Create: `tests/test_plot_nyu_cspn_decoder_sensitivity.py`

- [ ] **Step 1: Write and run failing plot validation tests**

Test required columns, unique names, finite RMSE/cost, Pareto membership, and
missing-file rejection.

```bash
python -m pytest -q tests/test_plot_nyu_cspn_decoder_sensitivity.py
```

- [ ] **Step 2: Implement the plotter**

Read `pareto_metrics.csv`; render RMSE against normalized added bit-element
cost with Arial-compatible font configuration, no title, grid behind points,
stable stage markers, and non-overlapping labels. Save PNG and PDF beside the
CSV.

- [ ] **Step 3: Verify and commit**

```bash
python -m pytest -q tests/test_plot_nyu_cspn_decoder_sensitivity.py
git add scripts/plot_nyu_cspn_decoder_sensitivity.py tests/test_plot_nyu_cspn_decoder_sensitivity.py
git commit -m "feat: plot CSPN decoder sensitivity Pareto set"
```

### Task 5: Full Verification and CUDA Evaluation

**Files:**
- Create: `docs/2026-08-14-cspn-decoder-depth-head-sensitivity-results.md`
- Generate: `/workspace/SPN_Quantization/profile_logs/nyu_cspn_decoder_depth_head_sensitivity_64/`

- [ ] **Step 1: Run focused and complete tests**

```bash
python -m pytest -q tests/test_hardware_aligned_quantization.py tests/test_run_nyu_cspn_activation_resolution.py tests/test_cspn_sensitivity.py tests/test_run_nyu_cspn_decoder_sensitivity.py tests/test_plot_nyu_cspn_decoder_sensitivity.py
python -m pytest -q
```

- [ ] **Step 2: Run the fixed CUDA experiment**

```bash
python scripts/run_nyu_cspn_decoder_sensitivity.py \
  --run-dir /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24 \
  --checkpoint /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24/best.pt \
  --data-root /workspace/CSPN/cspn_pytorch \
  --calibration-indices /workspace/SPN_Quantization/profile_logs/nyu_cspn_stratified_calibration_128/calibration_indices.json \
  --calibration-metadata /workspace/SPN_Quantization/profile_logs/nyu_cspn_stratified_calibration_128/metadata.json \
  --evaluation-protocol /workspace/SPN_Quantization/profile_logs/nyu_cspn_activation_resolution/cspn/metadata.json \
  --out-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_decoder_depth_head_sensitivity_64 \
  --device cuda:0 \
  --seed 20260812 \
  --fold-max-error 0.05
```

- [ ] **Step 3: Generate plots and audit artifacts**

```bash
python scripts/plot_nyu_cspn_decoder_sensitivity.py \
  --input-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_decoder_depth_head_sensitivity_64
```

Verify 64 unique finite sample rows per configuration, selected prediction
coverage, Stage-1 selection, cumulative unions, Pareto non-domination, and all
manifest hashes.

- [ ] **Step 4: Write, verify, and commit the report**

Report block and site rankings, cumulative/Pareto results, precision costs,
regional trade-offs, and the evidence locating dominant sensitivity. Run
`git diff --check` and `python -m pytest -q` again.

```bash
git add docs/2026-08-14-cspn-decoder-depth-head-sensitivity-results.md
git commit -m "docs: report CSPN decoder sensitivity results"
```
