# CSPN Selective W4A8 Boundary Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run a hierarchical CSPN search that keeps only the initial-depth weight at W8, searches W4A4/W4A8 activation units and boundaries, and finds the minimum-cost policy with NYU RMSE no greater than 0.1773 m.

**Architecture:** Extend the existing CSPN stem controller with one explicit W4A8 mode, then add a pure search module for deterministic unit masks, anchor selection, calibration-only boundary ranking, cumulative demotion, feasibility, and Pareto logic. A new strict runner reuses the official checkpoint, hardware-aligned Group-8 backend, fixed sample protocols, and diagnostics without changing prior experiment entry points; a separate plotter consumes only persisted CSV files.

**Tech Stack:** Python 3.11, PyTorch CUDA, NumPy, Matplotlib, pytest, official CSPN ResNet-18 adapter.

---

## File Responsibilities

- Modify `spn_quant/cspn_stem.py`: add the explicit W4-weight/A8-input stem contract.
- Modify `tests/test_cspn_stem.py`: prove W4A8 stem arithmetic and contract fields.
- Create `spn_quant/cspn_selective_w4a8.py`: own all pure candidate, ranking, path, feasibility, cost, winner, and Pareto logic.
- Create `tests/test_cspn_selective_w4a8.py`: test the pure search contract without CUDA.
- Create `scripts/run_nyu_cspn_selective_w4a8.py`: translate candidates into exact hardware configurations, run both stages, persist predictions and manifests, and reject contract drift.
- Create `tests/test_run_nyu_cspn_selective_w4a8.py`: test precision translation, orchestration, schemas, and immutable output behavior.
- Create `scripts/plot_nyu_cspn_selective_w4a8.py`: validate persisted CSVs and generate four fixed-format figures.
- Create `tests/test_plot_nyu_cspn_selective_w4a8.py`: test plot inputs and external script execution.
- Create `docs/2026-08-17-cspn-selective-w4a8-boundary-search-results.md`: report only measured CUDA results.

### Task 1: Explicit CSPN Stem W4A8 Contract

**Files:**
- Modify: `spn_quant/cspn_stem.py`
- Modify: `tests/test_cspn_stem.py`

- [ ] **Step 1: Write failing W4A8 configuration tests**

Add tests that configure a calibrated controller with `STEM_W4A8` and assert:

```python
controller.configure("STEM_W4A8")
contract = controller.contract()
self.assertEqual(contract["weight_bits"], 4)
self.assertEqual(contract["activation_bits"], 8)
self.assertEqual(contract["activation_scales"], 1)
self.assertEqual(controller.weight_codes.dtype, torch.int8)
```

Use a deterministic four-channel input and manually calculate W4
per-output-channel weight QDQ plus unsigned merged-input A8 QDQ. Assert the
controller output equals the manual convolution. Also assert `STEM_W8A8`
still reports W8/A8 and `STRICT_W4A4` still reports W4/A4.

- [ ] **Step 2: Run the tests and verify RED**

```bash
python -m pytest -q tests/test_cspn_stem.py -k w4a8
```

Expected: failure because `STEM_W4A8` is not in `STEM_CONFIGS`.

- [ ] **Step 3: Implement the minimal explicit mode**

Add `STEM_W4A8` to `STEM_CONFIGS`. Route it through
`_merged_forward(tensor, 8)`, quantize its weight with four bits in
`configure`, and report A8 in `contract` without changing any existing mode:

```python
if self.config in ("STEM_W4A8", "STEM_W8A8"):
    return self._merged_forward(tensor, 8)

weight_bits = 8 if name == "STEM_W8A8" else 4
self._quantize_weight(weight_bits)

bits = 8 if self.config in ("STEM_W4A8", "STEM_W8A8") else 4
```

- [ ] **Step 4: Verify focused stem regression**

```bash
python -m pytest -q tests/test_cspn_stem.py tests/test_run_nyu_cspn_stem_precision.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add spn_quant/cspn_stem.py tests/test_cspn_stem.py
git commit -m "feat: add CSPN stem W4A8 contract"
```

### Task 2: Pure Selective-W4A8 Search Model

**Files:**
- Create: `spn_quant/cspn_selective_w4a8.py`
- Create: `tests/test_cspn_selective_w4a8.py`

- [ ] **Step 1: Write failing Stage-1 candidate tests**

Test these immutable public types and constants:

```python
Owner = Tuple[str, str]

@dataclass(frozen=True)
class ActivationCandidate:
    name: str
    mask: int
    selected_units: Tuple[str, ...]
    activation_owners: Tuple[Owner, ...]

ACTIVATION_UNIT_ORDER = (
    "stem", "encoder_layer1", "encoder_layer2", "decoder_layer4")
INITIAL_DEPTH_WEIGHT = "gud_up_proj_layer5.conv1"
RMSE_LIMIT = 0.1773
```

Build the activation-unit registry from the exact validated
`cspn_encoder_prefix.UnitRegistry`. Assert `build_stage1_candidates` returns
all 16 masks in numeric order, mask zero has no A8 owners, mask 15 has the
stable union of all four units, shared skip4 ownership appears once, and every
candidate fixes only `INITIAL_DEPTH_WEIGHT` at W8.

- [ ] **Step 2: Run Stage-1 tests and verify RED**

```bash
python -m pytest -q tests/test_cspn_selective_w4a8.py -k stage1
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement candidate generation and anchor selection**

Implement exact registry validation, stable owner union, complete mask
generation, and:

```python
def select_stage1_anchor(rows, candidates, rmse_limit=RMSE_LIMIT):
    feasible = [row for row in rows if candidate_is_feasible(row, rmse_limit)]
    if not feasible:
        raise RuntimeError("no Stage-1 candidate satisfies the target")
    return min(feasible, key=lambda row: (
        float(row["normalized_added_bit_cost"]),
        float(row["a8_activation_element_fraction"]),
        float(row["RMSE"]),
        str(row["config"])))
```

Require exact candidate-row coverage and reject duplicate, missing, non-finite,
or context rows in primary selection.

- [ ] **Step 4: Write failing demotion-ranking and path tests**

Define and test:

```python
@dataclass(frozen=True)
class BoundaryDemotion:
    name: str
    owner: Owner
    activation_owners: Tuple[Owner, ...]

def build_single_demotions(anchor): ...
def rank_demotions(anchor_row, demotion_rows): ...
def build_cumulative_path(anchor, ranking): ...
```

Synthetic rows must prove the score is
`max(0, demotion_mse - anchor_mse) / saved_cost`, ties use downstream MSE,
larger saved cost, then canonical owner order, and cumulative candidates remove
exactly one additional owner while cost strictly decreases. Reject zero or
negative saved cost, duplicate owners, missing calibration rows, and a ranking
that contains an owner outside the anchor.

- [ ] **Step 5: Implement demotion ranking and cumulative construction**

Return ranking rows with explicit fields:

```python
{
    "rank": rank,
    "module": owner[0],
    "kind": owner[1],
    "propagation_mse_increase": increase,
    "downstream_mse_increase": downstream_increase,
    "saved_cost": saved_cost,
    "score": increase / saved_cost,
}
```

Build `PATH_000` as the unchanged anchor, then `PATH_001` through `PATH_N`
using the frozen owner order. Store owner lists directly; never reconstruct
them from names.

- [ ] **Step 6: Write and implement feasibility, winner, and Pareto tests**

Cover finite RMSE, `RMSE <= 0.1773`, zero non-positive ratio, zero coefficient
sum error, zero contraction violations, zero anchor error, complete 64-sample
coverage, and rerun equality. Implement lexicographic winner selection and
non-dominated rows for both normalized cost and A8 activation fraction.

- [ ] **Step 7: Verify and commit Task 2**

```bash
python -m pytest -q tests/test_cspn_selective_w4a8.py tests/test_cspn_encoder_prefix.py
git add spn_quant/cspn_selective_w4a8.py tests/test_cspn_selective_w4a8.py
git commit -m "feat: define CSPN selective W4A8 search"
```

### Task 3: Exact Hardware Translation and Calibration Diagnostics

**Files:**
- Create: `scripts/run_nyu_cspn_selective_w4a8.py`
- Create: `tests/test_run_nyu_cspn_selective_w4a8.py`

- [ ] **Step 1: Write failing precision-translation tests**

Test:

```python
def hardware_configuration(candidate): ...
def stem_configuration(candidate): ...
def validate_configured_precision(candidate, weight_bits, specs,
                                  rotation_specs, stem_contract): ...
```

Assert the primary configuration starts from static Group-8 W4A4, overrides
only `gud_up_proj_layer5.conv1` to W8, changes only declared owners to A8, and
uses `STEM_W4A8` exactly when the merged stem input owner is A8. Assert generic
ownership excludes the stem Conv/input but includes root ReLU and skip4. Test
missing, extra, and wrong-bit sites as hard failures.

- [ ] **Step 2: Run translation tests and verify RED**

```bash
python -m pytest -q tests/test_run_nyu_cspn_selective_w4a8.py -k "hardware or precision or stem"
```

Expected: import failure because the runner does not exist.

- [ ] **Step 3: Implement primary and context translation**

Reuse `run_nyu_cspn_encoder_prefix_joint` configuration builders for the
strict and P3/T3 context rows. For primary candidates, construct explicit
weight overrides and A8 `QuantSpec` replacements. Validate the actual union:

```python
expected_w8 = {INITIAL_DEPTH_WEIGHT}
actual_w8 = {name for name in weight_bits if weight_bits[name] == 8}
if actual_w8 != expected_w8:
    raise RuntimeError("configured W8 weight set does not match candidate")
```

Use direct dictionary indexing for every required field and no configuration
fallbacks.

- [ ] **Step 4: Write failing calibration-diagnostic tests**

Use synthetic FP32/quantized captures to test aggregation of final propagation
MSE, owner-block MSE/SQNR, downstream MSE, per-iteration propagation rows, and
saved activation cost. Require the same 128 unique calibration identities for
the anchor and every single demotion.

- [ ] **Step 5: Implement fresh calibration-only evaluation**

Implement a calibration evaluator that loads a fresh model/checkpoint,
reproduces Conv-BN folding, observes exactly the persisted 128 samples,
configures one candidate, and compares its configured forward outputs with a
shared FP32 calibration reference. Return only structured rows; close every
capture, counter, adapter, stem controller, and hook before releasing models.

- [ ] **Step 6: Verify and commit Task 3**

```bash
python -m pytest -q tests/test_run_nyu_cspn_selective_w4a8.py -k "hardware or precision or stem or calibration"
git add scripts/run_nyu_cspn_selective_w4a8.py tests/test_run_nyu_cspn_selective_w4a8.py
git commit -m "feat: configure CSPN selective W4A8 evaluation"
```

### Task 4: Two-Stage Evaluation and Immutable Outputs

**Files:**
- Modify: `scripts/run_nyu_cspn_selective_w4a8.py`
- Modify: `tests/test_run_nyu_cspn_selective_w4a8.py`

- [ ] **Step 1: Write failing Stage-1 orchestration tests**

Inject a synthetic evaluator and assert the runner evaluates 16 primary masks
plus two contexts from fresh model calls, validates 64 unique sample rows per
configuration, selects the exact lowest-cost feasible anchor, and stops with a
target-unsatisfied error rather than choosing a fallback.

- [ ] **Step 2: Implement Stage-1 orchestration**

Build the official registry after strict calibration, run the 18 declared
configurations, aggregate all diagnostics, and select the anchor using only
the pure module. Keep scalar rows in memory and write prediction payloads to a
sibling `<out-dir>.incomplete` staging root only after both stages succeed.
Reject an existing final or staging root and atomically rename the staging root
after artifact hashes pass, so an exception cannot leave a valid-looking final
result root.

- [ ] **Step 3: Write failing Stage-2 freeze tests**

Assert all single demotions use 128 calibration samples, the ranking is frozen
before validation evaluator calls, every cumulative point is evaluated in that
fixed order, and synthetic validation RMSE values cannot reorder or insert
path owners.

- [ ] **Step 4: Implement Stage-2 orchestration and winner selection**

Evaluate anchor plus all single demotions on calibration, call
`rank_demotions`, build the complete cumulative path, then evaluate every path
point on the fixed 64 samples. Select feasible rows and the winner only after
the path is complete. Record no implicit retry or alternate anchor.

- [ ] **Step 5: Write failing output and prediction tests**

Test exact CSV schemas, context/primary/path separation, artifact names,
manifest contracts, source/checkpoint/protocol hashes, rejection of an existing
final or `.incomplete` output root, deterministic prediction selection, 64
payloads per selected configuration, exact first-pass/rerun RMSE equality, and
atomic publication only after hash verification.

- [ ] **Step 6: Implement immutable persistence**

Write the files declared by the spec, including:

```text
stage1_aggregate_metrics.csv
stage1_sample_metrics_64.csv
stage1_block_metrics.csv
stage1_propagation_metrics.csv
stage1_operation_counts.csv
stage1_precision_coverage.csv
stage2_single_demotion_calibration.csv
stage2_boundary_ranking.csv
stage2_path_aggregate_metrics.csv
stage2_path_sample_metrics_64.csv
stage2_path_block_metrics.csv
stage2_path_propagation_metrics.csv
feasible_candidates.csv
pareto_normalized_cost.csv
pareto_a8_fraction.csv
manifest.json
```

Hash every artifact except the manifest itself after prediction reruns. The
manifest stores full owner arrays, not owner-name hashes as a replacement for
the actual contract.

- [ ] **Step 7: Verify and commit Task 4**

```bash
python -m pytest -q tests/test_run_nyu_cspn_selective_w4a8.py tests/test_cspn_selective_w4a8.py tests/test_run_nyu_cspn_encoder_prefix_joint.py
git add scripts/run_nyu_cspn_selective_w4a8.py tests/test_run_nyu_cspn_selective_w4a8.py
git commit -m "feat: run CSPN selective W4A8 boundary search"
```

### Task 5: CSV-Only Plots

**Files:**
- Create: `scripts/plot_nyu_cspn_selective_w4a8.py`
- Create: `tests/test_plot_nyu_cspn_selective_w4a8.py`

- [ ] **Step 1: Write failing plot-input tests**

Test required columns, exact 16-cell unit-mask coverage, unique frozen boundary
ranks, strictly decreasing cumulative A8 cost, valid Pareto membership,
finite values, missing-file rejection, and script import from outside the
repository root.

- [ ] **Step 2: Run plot tests and verify RED**

```bash
python -m pytest -q tests/test_plot_nyu_cspn_selective_w4a8.py
```

Expected: import failure because the plotter does not exist.

- [ ] **Step 3: Implement fixed-format plotting**

Read only persisted CSVs and generate:

```text
stage1_unit_mask_rmse.png/.pdf
stage2_boundary_demotion_sensitivity.png/.pdf
selective_w4a8_normalized_cost_pareto.png/.pdf
selective_w4a8_a8_fraction_pareto.png/.pdf
```

Use Arial, no title, horizontal tick labels, stable dimensions, explicit units,
and grid `zorder=0` below data. Render the 16 Stage-1 masks as a 4-by-4 matrix:
row state is `(stem, layer1)` and column state is `(layer2, decoder4)`. Label
only Pareto points and the winner to avoid overlap. Recompute manifest artifact
hashes after all figures are written.

- [ ] **Step 4: Verify and commit Task 5**

```bash
python -m pytest -q tests/test_plot_nyu_cspn_selective_w4a8.py
git add scripts/plot_nyu_cspn_selective_w4a8.py tests/test_plot_nyu_cspn_selective_w4a8.py
git commit -m "feat: plot CSPN selective W4A8 search"
```

### Task 6: Full Verification, CUDA Search, and Measured Report

**Files:**
- Create: `docs/2026-08-17-cspn-selective-w4a8-boundary-search-results.md`
- Generate: `/workspace/SPN_Quantization/profile_logs/nyu_cspn_selective_w4a8_boundary_search_64/`

- [ ] **Step 1: Run focused and complete tests**

```bash
python -m pytest -q \
  tests/test_cspn_stem.py \
  tests/test_cspn_selective_w4a8.py \
  tests/test_run_nyu_cspn_selective_w4a8.py \
  tests/test_plot_nyu_cspn_selective_w4a8.py \
  tests/test_cspn_encoder_prefix.py \
  tests/test_run_nyu_cspn_encoder_prefix_joint.py
python -m pytest -q
```

Expected: zero failures.

- [ ] **Step 2: Run the fixed CUDA experiment**

```bash
python scripts/run_nyu_cspn_selective_w4a8.py \
  --run-dir /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24 \
  --checkpoint /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24/best.pt \
  --data-root /workspace/CSPN/cspn_pytorch \
  --calibration-indices /workspace/SPN_Quantization/profile_logs/nyu_cspn_stratified_calibration_128/calibration_indices.json \
  --calibration-metadata /workspace/SPN_Quantization/profile_logs/nyu_cspn_stratified_calibration_128/metadata.json \
  --evaluation-protocol /workspace/SPN_Quantization/profile_logs/nyu_cspn_activation_resolution/cspn/metadata.json \
  --out-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_selective_w4a8_boundary_search_64 \
  --device cuda:0 \
  --seed 20260812 \
  --rmse-limit 0.1773 \
  --fold-max-error 0.05
```

Expected: 16 primary and two context configurations, a feasible Stage-1
anchor, all single-demotion calibration runs, every frozen cumulative path
point, selected prediction reruns, and exit code zero.

- [ ] **Step 3: Generate plots and audit artifacts**

```bash
python scripts/plot_nyu_cspn_selective_w4a8.py \
  --input-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_selective_w4a8_boundary_search_64
```

Audit exact candidate counts, 64 identities per validation candidate, 128
identities per calibration demotion, finite values, exact W8/A8 ownership,
strict path cost monotonicity, frozen ranking, propagation invariants,
non-positive prediction constraints, both Pareto sets, selected prediction
coverage, and every manifest hash. Visually inspect all four PNGs.

- [ ] **Step 4: Write the measured report**

Document the Stage-1 interaction result, selected anchor, complete demotion
ranking, minimum feasible policy, RMSE delta from P3/T3, A8/cost reduction,
block-error flow, propagation safety, inverse-depth behavior, and the difference
between logical precision coverage and measured hardware latency.

- [ ] **Step 5: Final verification and commit**

```bash
git diff --check
python -m pytest -q
git add docs/2026-08-17-cspn-selective-w4a8-boundary-search-results.md
git commit -m "docs: report CSPN selective W4A8 results"
```

Expected: clean feature worktree and zero test failures.
