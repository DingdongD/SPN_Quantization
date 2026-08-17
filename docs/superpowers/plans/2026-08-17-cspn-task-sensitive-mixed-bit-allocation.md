# CSPN Task-Sensitive Mixed-Bit Allocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a deterministic CSPN PTQ allocator that jointly assigns `{2, 4, 6, 8}` weight and activation bits while keeping MAC-weighted average weight bits and element-weighted average activation bits independently at or below four.

**Architecture:** Extend QDQ only where arbitrary bits are currently rejected. Keep registry, exact budgets, Beam search, local moves, and refinement in a pure module; keep official-model loading, calibration, CUDA inference, persistence, and predictions in a dedicated runner. Selection uses only 128 train calibration samples; the fixed 64 validation samples evaluate only the frozen allocation.

**Tech Stack:** Python 3, PyTorch/CUDA, NumPy, Matplotlib, pytest/unittest, existing CSPN hardware-aligned PTQ and propagation controllers.

---

## File Structure

- Modify `scripts/hardware_aligned_quantization.py` for `{2,4,6,8}` weight overrides.
- Modify `spn_quant/cspn_stem.py` for explicit merged RGBD W/A bits.
- Modify `scripts/run_nyu_cspn_activation_resolution.py` for complete activation-owner bit maps.
- Create `spn_quant/cspn_task_sensitive_bits.py` for allocation/search logic.
- Create `scripts/run_nyu_cspn_task_sensitive_bits.py` for CUDA orchestration and audit.
- Create `scripts/plot_nyu_cspn_task_sensitive_bits.py` for CSV-only plots.
- Create `tests/test_cspn_task_sensitive_bits.py`, `tests/test_run_nyu_cspn_task_sensitive_bits.py`, and `tests/test_plot_nyu_cspn_task_sensitive_bits.py`.
- Extend the three existing tests corresponding to modified production files.
- Create `docs/2026-08-17-cspn-task-sensitive-mixed-bit-allocation-results.md` after measurement.

### Task 1: Enable Explicit W2/W6 and Stem Mixed Bits

**Files:**
- Modify: `scripts/hardware_aligned_quantization.py:1360-1420`
- Modify: `spn_quant/cspn_stem.py:21-28,130-170,390-465`
- Test: `tests/test_hardware_aligned_quantization.py`
- Test: `tests/test_cspn_stem.py`

- [ ] **Step 1: Write failing W2/W6 weight tests**

Configure a calibrated instrumentor with `weight_bit_overrides={"0": 2,
"2": 6}`. Assert exact recorded bits and reconstructed code maxima 1 and 31.
Add a bit-3 rejection test.

```python
instrumentor.configure_components_with_ranges(
    4, 4, {"encoder"}, {"encoder"}, specs, False, {},
    weight_bit_overrides={"0": 2, "2": 6})
self.assertEqual(instrumentor.weight_bits_by_module()["0"], 2)
self.assertEqual(instrumentor.weight_bits_by_module()["2"], 6)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest -q tests/test_hardware_aligned_quantization.py -k 'two_and_six or rejects_three'`

Expected: failure because production validation accepts only 4 or 8.

- [ ] **Step 3: Implement the explicit weight-bit set**

```python
HARDWARE_INTEGER_BITS = (2, 4, 6, 8)

if weight_bits not in HARDWARE_INTEGER_BITS:
    raise ValueError(
        "weight bits must be one of %s: %s=%d" %
        (HARDWARE_INTEGER_BITS, name, weight_bits))
```

Keep `symmetric_weight_qdq`; it already gives qmax 1/7/31/127.

- [ ] **Step 4: Write failing stem tests**

Parameterize all 16 W/A pairs. Call `configure_integer(w, a)`, compare output
with manual `_unsigned_qdq()` plus `_quantized_weight()`, and assert contract
bits. Add bit-3 failures and preserve named-config tests.

```python
controller.configure_integer(weight_bits, activation_bits)
observed = module(tensor)
quantized, _, _ = controller._unsigned_qdq(
    tensor, controller.merged_maximum, activation_bits)
expected = controller._float_convolution(
    quantized, controller._quantized_weight(tensor))
torch.testing.assert_close(observed, expected)
```

- [ ] **Step 5: Run stem tests and verify RED**

Run: `python -m pytest -q tests/test_cspn_stem.py -k explicit_integer`

Expected: missing `configure_integer`.

- [ ] **Step 6: Implement explicit stem configuration**

Add `INTEGER_BITS`, `self.activation_bits`, `_activate_integer()`, and:

```python
def configure_integer(self, weight_bits: int, activation_bits: int) -> None:
    if self.observations == 0 or self.phase == "observe":
        raise RuntimeError("CSPN stem must be frozen before configuration")
    if int(weight_bits) not in INTEGER_BITS:
        raise ValueError("stem weight bits must be one of %s" %
                         (INTEGER_BITS,))
    if int(activation_bits) not in INTEGER_BITS:
        raise ValueError("stem activation bits must be one of %s" %
                         (INTEGER_BITS,))
    self._activate_integer(
        "STEM_W%dA%d" % (int(weight_bits), int(activation_bits)),
        int(weight_bits), int(activation_bits))
```

Make merged `_forward()` use `self.activation_bits`. Preserve named configs via
an explicitly indexed map; do not parse names or use `.get()`.

- [ ] **Step 7: Run focused regressions**

Run: `python -m pytest -q tests/test_hardware_aligned_quantization.py tests/test_cspn_stem.py`

- [ ] **Step 8: Commit**

Run: `git add scripts/hardware_aligned_quantization.py spn_quant/cspn_stem.py tests/test_hardware_aligned_quantization.py tests/test_cspn_stem.py && git commit -m "feat: support CSPN 2 4 6 8 bit QDQ"`

### Task 2: Add Complete Activation-Owner Bit Maps

**Files:**
- Modify: `scripts/run_nyu_cspn_activation_resolution.py:177-220,480-550,761-835`
- Test: `tests/test_run_nyu_cspn_activation_resolution.py`

- [ ] **Step 1: Write failing exact-owner tests**

Build a complete owner map cycling through 2/4/6/8. Assert ordinary and
rotation specs reproduce it. Add missing, extra, duplicate, and bit-3 failures.

```python
observed = dict(
    (runner.activation_owner(key), int(spec.bits))
    for key, spec in specs.items())
observed.update(dict(
    (tuple(owner), int(rotation_specs[owner].bits))
    for owner in rotation_specs))
self.assertEqual(observed, assignment)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest -q tests/test_run_nyu_cspn_activation_resolution.py -k owner_bit_assignment`

Expected: builders reject `bit_overrides`.

- [ ] **Step 3: Implement complete map application**

Add tuple-valued `activation_bit_overrides` to `_configuration()`. Extend both
spec builders with `bit_overrides=()`. When nonempty, require exact combined
ordinary/rotation owner coverage, validate values in `(2,4,6,8)`, and apply
`spec.with_bits(overrides[owner])`.

```python
overrides = dict((tuple(owner), int(bits))
                 for owner, bits in bit_overrides)
if overrides and set(overrides) != expected_owners:
    raise ValueError("activation bit assignment coverage mismatch")
```

Existing callers retain old promoted-owner behavior. The new runner always
supplies and independently validates a complete map.

- [ ] **Step 4: Run focused regressions**

Run: `python -m pytest -q tests/test_run_nyu_cspn_activation_resolution.py tests/test_run_nyu_cspn_selective_w4a8.py`

- [ ] **Step 5: Commit**

Run: `git add scripts/run_nyu_cspn_activation_resolution.py tests/test_run_nyu_cspn_activation_resolution.py && git commit -m "feat: configure exact CSPN activation bits"`

### Task 3: Implement Registry, Assignments, and Exact Budgets

**Files:**
- Create: `spn_quant/cspn_task_sensitive_bits.py`
- Test: `tests/test_cspn_task_sensitive_bits.py`

- [ ] **Step 1: Write failing registry/probe tests**

Assert the ten-block order, complete unique ownership, one baseline, and 15
non-W4A4 probes per block.

```python
probes = allocation.build_single_block_probes(registry)
assert len(probes) == 151
assert probes[0].name == "UNIFORM_W4A4"
assert len({probe.name for probe in probes}) == 151
```

Add missing/extra/duplicate coverage failures and assert the shared layer4 skip
owner is charged once.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest -q tests/test_cspn_task_sensitive_bits.py -k 'registry or probe'`

Expected: module import fails.

- [ ] **Step 3: Implement immutable contracts**

```python
BIT_OPTIONS = (2, 4, 6, 8)
BLOCK_ORDER = (
    "stem", "encoder_layer1", "encoder_layer2", "encoder_layer3",
    "encoder_layer4", "decoder_layer1", "decoder_layer2",
    "decoder_layer3", "decoder_layer4", "initial_depth",
)

@dataclass(frozen=True)
class BitAssignment:
    weight_bits: Tuple[Tuple[str, int], ...]
    activation_bits: Tuple[Tuple[Owner, int], ...]

@dataclass(frozen=True)
class AllocationCandidate:
    name: str
    stage: str
    block: str
    assignment: BitAssignment
    weight_bits: int
    activation_bits: int
```

Build encoder ownership from `cspn_encoder_prefix` and decoder ownership from
`cspn_sensitivity`. Validate exact observed coverage; never infer missing sites.

- [ ] **Step 4: Write failing weighted-budget tests**

```python
basis = allocation.CostBasis(
    weight_macs=(("large", 90), ("small", 10)),
    activation_elements=((('large', 'input'), 80),
                         (('small', 'input'), 20)))
assignment = allocation.BitAssignment(
    weight_bits=(("large", 2), ("small", 8)),
    activation_bits=((('large', 'input'), 2),
                     (('small', 'input'), 8)))
audit = allocation.audit_budget(assignment, basis)
assert audit.weight_numerator == 260
assert audit.activation_numerator == 320
assert audit.feasible
```

Cover exact equality, each independent failure, missing costs, duplicates,
nonpositive counts, and per-bit fraction sums.

- [ ] **Step 5: Run budget tests and verify RED**

Run: `python -m pytest -q tests/test_cspn_task_sensitive_bits.py -k 'budget or fraction'`

- [ ] **Step 6: Implement exact arithmetic**

Add `AllocationRegistry`, `CostBasis`, and `BudgetAudit`. Require assignment keys
to exactly equal basis keys. Compute integer numerators first:

```python
weight_feasible = weight_numerator <= 4 * weight_denominator
activation_feasible = activation_numerator <= 4 * activation_denominator
```

Return average bits and per-bit MAC/element fractions.

- [ ] **Step 7: Run pure tests**

Run: `python -m pytest -q tests/test_cspn_task_sensitive_bits.py`

- [ ] **Step 8: Commit**

Run: `git add spn_quant/cspn_task_sensitive_bits.py tests/test_cspn_task_sensitive_bits.py && git commit -m "feat: define CSPN mixed-bit allocation contracts"`

### Task 4: Implement Beam, Local Search, and Refinement

**Files:**
- Modify: `spn_quant/cspn_task_sensitive_bits.py`
- Test: `tests/test_cspn_task_sensitive_bits.py`

- [ ] **Step 1: Write failing sensitivity/Beam tests**

Use a miniature registry. Assert deterministic output under reversed input,
ignore `validation_RMSE`, enforce both budgets, cap width, reject incomplete or
duplicate probe rows, reject nonfinite calibration metrics, and use canonical
ties.

```python
left = allocation.search_block_assignments(
    registry, basis, measured_rows(validation_rmse=0.1), 8, 4)
right = allocation.search_block_assignments(
    registry, basis,
    list(reversed(measured_rows(validation_rmse=9.0))), 8, 4)
assert left == right
assert all(allocation.audit_budget(row.assignment, basis).feasible
           for row in left)
```

- [ ] **Step 2: Run Beam tests and verify RED**

Run: `python -m pytest -q tests/test_cspn_task_sensitive_bits.py -k 'beam or sensitivity or dominance'`

- [ ] **Step 3: Implement deterministic Beam**

Add `SearchState`, `build_sensitivity_table()`, and
`search_block_assignments()`. Read calibration fields with `[]`. Expand
canonical block order, prune states whose W2/A2 lower bound cannot meet final
budgets, deduplicate, dominance-prune, sort by estimated RMSE, propagation MSE,
budget slack, and canonical tuple, then slice to width.

- [ ] **Step 4: Write failing local/refinement tests**

Assert neighbors use ±2 steps, remain feasible, and are canonical; acceptance
requires lower calibration RMSE; stop at first non-improving round or three
rounds; top-four selection uses cheapest-demotion RMSE increase.

```python
neighbors = allocation.build_budget_preserving_neighbors(
    current, registry, basis)
assert neighbors
assert all(allocation.audit_budget(row, basis).feasible
           for row in neighbors)
```

- [ ] **Step 5: Run local tests and verify RED**

Run: `python -m pytest -q tests/test_cspn_task_sensitive_bits.py -k 'neighbor or local or refinement'`

- [ ] **Step 6: Implement local/refinement APIs**

Implement `build_budget_preserving_neighbors()`,
`select_local_improvement()`, `rank_refinement_blocks()`, and
`build_refinement_candidates()`. Generate only valid ±2 moves, require complete
measured-neighbor coverage, and split only four selected blocks into exact
modules/owners. Keep shared owners unique.

- [ ] **Step 7: Run pure search tests**

Run: `python -m pytest -q tests/test_cspn_task_sensitive_bits.py`

- [ ] **Step 8: Commit**

Run: `git add spn_quant/cspn_task_sensitive_bits.py tests/test_cspn_task_sensitive_bits.py && git commit -m "feat: search CSPN mixed-bit assignments"`

### Task 5: Build Strict Runtime Configuration and Precision Audit

**Files:**
- Create: `scripts/run_nyu_cspn_task_sensitive_bits.py`
- Test: `tests/test_run_nyu_cspn_task_sensitive_bits.py`

- [ ] **Step 1: Write failing runtime-map tests**

Convert a complete `BitAssignment` into an ordinary config and assert all
non-stem weight/activation bits plus propagation contract are explicit.

```python
config = runner.runtime_configuration(candidate)
self.assertEqual(config["w_bits"], 4)
self.assertEqual(config["a_bits"], 4)
self.assertEqual(dict(config["weight_bit_overrides"]),
                 expected_nonstem_weights)
self.assertEqual(dict(config["activation_bit_overrides"]),
                 expected_nonstem_activations)
self.assertEqual(config["propagation"], runner.base.PROPAGATION_A8_Q13)
```

Add omitted/extra owner, configured-bit mismatch, stem mismatch, and forbidden
guidance/propagation ownership failures. Use strict mappings that raise on a
missing `[]` access to prove no default insertion.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest -q tests/test_run_nyu_cspn_task_sensitive_bits.py -k 'runtime or precision'`

Expected: runner module is missing.

- [ ] **Step 3: Implement complete runtime configuration**

Create `RuntimeCandidate(name, stage, assignment)`. Build Group-8 ordinary
config with FP32 bias and `PROPAGATION_A8_Q13`; supply all non-stem overrides.
Configure stem with `configure_integer()`.

```python
def validate_configured_precision(candidate, weight_bits, specs,
                                  rotation_specs, stem_contract):
    expected_weights = dict(candidate.assignment.weight_bits)
    expected_activations = dict(candidate.assignment.activation_bits)
    actual_weights = dict((str(name), int(weight_bits[name]))
                          for name in weight_bits)
    actual_activations = dict(
        (base.activation_owner(key), int(specs[key].bits))
        for key in specs)
    actual_activations.update(dict(
        (tuple(owner), int(rotation_specs[owner].bits))
        for owner in rotation_specs))
    actual_weights[STEM_WEIGHT_MODULE] = int(stem_contract["weight_bits"])
    actual_activations[STEM_INPUT_OWNER] = int(
        stem_contract["activation_bits"])
    if actual_weights != expected_weights:
        raise RuntimeError("configured weight bits differ from assignment")
    if actual_activations != expected_activations:
        raise RuntimeError("configured activation bits differ from assignment")
```

Validate budget before loading and again from observed costs after calibration.

- [ ] **Step 4: Write failing validity tests**

Test exact failure reasons for NaN/Inf, nonpositive valid depth,
coefficient-sum error, contraction violation, anchor error, and a
Stage-2-or-later budget excess. Assert each invalid row retains the original
assignment. Add a Stage-1 probe test showing budget excess is recorded but does
not invalidate a sensitivity probe.

- [ ] **Step 5: Implement validity without exception swallowing**

Use pure `candidate_status(metrics, budget)` for numeric validity. Missing sites,
changed checkpoint/fold, or configured-bit drift remain hard exceptions. Do not
wrap inference in `try/except`, rerun at another bit, or insert FP32.

- [ ] **Step 6: Run runtime tests**

Run: `python -m pytest -q tests/test_run_nyu_cspn_task_sensitive_bits.py`

- [ ] **Step 7: Commit**

Run: `git add scripts/run_nyu_cspn_task_sensitive_bits.py tests/test_run_nyu_cspn_task_sensitive_bits.py && git commit -m "feat: configure CSPN mixed-bit candidates"`

### Task 6: Implement CUDA Search and Immutable Outputs

**Files:**
- Modify: `scripts/run_nyu_cspn_task_sensitive_bits.py`
- Test: `tests/test_run_nyu_cspn_task_sensitive_bits.py`

- [ ] **Step 1: Write failing orchestration tests**

Use a fake evaluator to assert 151 probes, 128 measured joint candidates, at
most three local rounds, 128 refinements, four frozen validation comparisons,
and complete assignments on every call.

```python
result = runner.run_search(protocol, registry, basis, evaluator)
self.assertEqual(len(result.single_block_rows), 151)
self.assertEqual(len(result.joint_rows), 128)
self.assertLessEqual(len(result.local_rounds), 3)
self.assertEqual(len(result.refined_rows), 128)
self.assertEqual(evaluator.validation_calls, 4)
self.assertEqual(
    evaluator.validation_names,
    ("FP32", "UNIFORM_W4A4", "CONTEXT_P3_T3_W8A8", "FINAL"))
self.assertTrue(result.final_budget.feasible)
```

Add tests proving shuffled evaluator results are deterministic, validation
metrics cannot select a winner, and incomplete phase coverage raises.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest -q tests/test_run_nyu_cspn_task_sensitive_bits.py -k 'orchestration or phase'`

- [ ] **Step 3: Implement fresh-model candidate evaluation**

Reuse `_prepare_model`, `_build_quantization_context`, `_calibrate`, `_forward`,
`ModuleOutputCapture`, `BlockErrorAccumulator`, propagation statistics, and
existing depth/region metrics. Every call must fresh-load the official model,
check architecture/checkpoint/fold equality, calibrate the exact identities,
configure and validate all bits, evaluate requested identities, close hooks and
controllers, move model to CPU, and empty CUDA cache.

- [ ] **Step 4: Implement the fixed phases**

`run_search()` must perform, in order:

1. 151 single-block calibration probes;
2. width-512 surrogate Beam and 128 real joint evaluations;
3. all valid neighbors for at most three accepted local rounds;
4. cheapest-demotion probes, four-block selection, width-128 refinement Beam,
   and 128 real refined evaluations;
5. freeze minimum-calibration-RMSE valid assignment;
6. evaluate FP32, uniform W4A4, P3/T3, and the frozen assignment exactly once
   each on the same 64 validation identities, writing prediction payloads.

None of the four validation rows may be read by a search or tie-break function.

Distribute independent candidates over `--devices` by stable round robin. One
candidate stays on one GPU and writes one immutable phase/config payload.

- [ ] **Step 5: Write failing output tests**

Test rejection of existing final or `.incomplete` roots, required CSV fields,
exact phase counts, four prediction configurations, sample identities, and
artifact hashes.

- [ ] **Step 6: Implement required CLI and atomic publication**

Require these arguments without code defaults:

```text
--run-dir --checkpoint --data-root --calibration-indices
--calibration-metadata --evaluation-protocol --out-dir --devices --seed
--fold-max-error --beam-width --joint-measured-limit --local-round-limit
--refinement-block-limit --refinement-width --refinement-measured-limit
```

Require production search values `512,128,3,4,128,128`. Persist each completed
phase under the sibling `.incomplete` root. Publish by one rename only after
coverage/hash audit. Never write under `/tmp`.

- [ ] **Step 7: Run runner tests**

Run: `python -m pytest -q tests/test_run_nyu_cspn_task_sensitive_bits.py`

- [ ] **Step 8: Commit**

Run: `git add scripts/run_nyu_cspn_task_sensitive_bits.py tests/test_run_nyu_cspn_task_sensitive_bits.py && git commit -m "feat: run CSPN task-sensitive bit allocation"`

### Task 7: Add CSV-Only Plots and Independent Audit

**Files:**
- Create: `scripts/plot_nyu_cspn_task_sensitive_bits.py`
- Create: `tests/test_plot_nyu_cspn_task_sensitive_bits.py`
- Modify: `scripts/run_nyu_cspn_task_sensitive_bits.py`
- Modify: `tests/test_run_nyu_cspn_task_sensitive_bits.py`

- [ ] **Step 1: Write failing plot-data tests**

Validate complete bit rows, exact fraction sums, budget limits, and missing CSV
errors. Tests use fixture rows and never invoke a model.

```python
rows = plotter.validate_allocation_rows(final_allocation_rows())
self.assertAlmostEqual(
    sum(float(row["weight_mac_fraction"]) for row in rows), 1.0)
self.assertAlmostEqual(
    sum(float(row["activation_element_fraction"]) for row in rows), 1.0)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest -q tests/test_plot_nyu_cspn_task_sensitive_bits.py`

Expected: plotter module is missing.

- [ ] **Step 3: Implement persisted-data plots**

Generate PNG/PDF for paired final block W/A bits, calibration RMSE versus both
budgets, per-bit MAC/activation fractions, and 64-sample prediction comparison.
Use Arial, no titles, horizontal labels, `set_axisbelow(True)`, grid zorder 0,
and data zorder at least 2. Plot code cannot select candidates.

- [ ] **Step 4: Write failing independent-audit tests**

Corrupt one field per test and require a hard failure for sample identity,
configured-bit equality, either integer budget, fraction sum, propagation
invariant, prediction count, and artifact hash.

- [ ] **Step 5: Implement `audit_result_root(root)`**

Read required fields with `[]`. Recompute both integer budget numerators from
cost bases and final assignment; verify bits, phase counts, uniqueness, samples,
finite metrics, propagation constraints, predictions, and every hash. Do not
repair malformed rows.

- [ ] **Step 6: Run plot/audit tests**

Run: `python -m pytest -q tests/test_plot_nyu_cspn_task_sensitive_bits.py tests/test_run_nyu_cspn_task_sensitive_bits.py`

- [ ] **Step 7: Commit**

Run: `git add scripts/plot_nyu_cspn_task_sensitive_bits.py scripts/run_nyu_cspn_task_sensitive_bits.py tests/test_plot_nyu_cspn_task_sensitive_bits.py tests/test_run_nyu_cspn_task_sensitive_bits.py && git commit -m "feat: audit and plot CSPN mixed-bit allocation"`

### Task 8: Verify, Run CUDA, and Report Results

**Files:**
- Create: `docs/2026-08-17-cspn-task-sensitive-mixed-bit-allocation-results.md`
- Generate: `profile_logs/nyu_cspn_task_sensitive_mixed_bits_w4a4_budget_64/`

- [ ] **Step 1: Run focused tests**

Run: `python -m pytest -q tests/test_hardware_aligned_quantization.py tests/test_cspn_stem.py tests/test_run_nyu_cspn_activation_resolution.py tests/test_cspn_task_sensitive_bits.py tests/test_run_nyu_cspn_task_sensitive_bits.py tests/test_plot_nyu_cspn_task_sensitive_bits.py`

Expected: all pass.

- [ ] **Step 2: Run full tests**

Run: `python -m pytest -q`

Expected: zero failures; record exact counts.

- [ ] **Step 3: Run source-quality checks**

Run: `git diff --check`

Run: `rg -n '\.get\(|getattr\(|try:|except |fallback|TODO|FIXME' spn_quant/cspn_task_sensitive_bits.py scripts/run_nyu_cspn_task_sensitive_bits.py scripts/plot_nyu_cspn_task_sensitive_bits.py`

Expected: no matches in new production files.

- [ ] **Step 4: Check GPUs and immutable output path**

Run: `nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader`

Run: `find /workspace/SPN_Quantization/profile_logs -maxdepth 1 -type d -name 'nyu_cspn_task_sensitive_mixed_bits_w4a4_budget_64*' -print`

Expected: GPUs available and no final/staging root. Inspect an exact stale root
before removing it; never broadly clean profile logs.

- [ ] **Step 5: Run the strict CUDA search**

```bash
python scripts/run_nyu_cspn_task_sensitive_bits.py \
  --run-dir /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24 \
  --checkpoint /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24/best.pt \
  --data-root /workspace/CSPN/cspn_pytorch \
  --calibration-indices /workspace/SPN_Quantization/profile_logs/nyu_cspn_stratified_calibration_128/calibration_indices.json \
  --calibration-metadata /workspace/SPN_Quantization/profile_logs/nyu_cspn_stratified_calibration_128/metadata.json \
  --evaluation-protocol /workspace/SPN_Quantization/profile_logs/nyu_cspn_activation_resolution/cspn/metadata.json \
  --out-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_task_sensitive_mixed_bits_w4a4_budget_64 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --seed 20260812 --fold-max-error 0.05 \
  --beam-width 512 --joint-measured-limit 128 --local-round-limit 3 \
  --refinement-block-limit 4 --refinement-width 128 \
  --refinement-measured-limit 128
```

Expected: all phases complete and final root publishes atomically. On failure,
use systematic debugging and a failing regression test before one root-cause
fix; remove only the inspected exact `.incomplete` root before rerun.

- [ ] **Step 6: Generate plots and audit**

Run: `python scripts/plot_nyu_cspn_task_sensitive_bits.py --input-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_task_sensitive_mixed_bits_w4a4_budget_64`

Run: `python scripts/run_nyu_cspn_task_sensitive_bits.py --audit-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_task_sensitive_mixed_bits_w4a4_budget_64`

Expected: hashes, exact budgets, bit equality, phase/sample coverage, finite
metrics, propagation invariants, and prediction coverage all pass.

- [ ] **Step 7: Inspect every PNG**

Use `view_image` and verify Arial, no title, horizontal labels, grid behind data,
readable legends, nonblank prediction panels, and no clipping/overlap. Fix a
plot defect only after adding a failing plot test.

- [ ] **Step 8: Write measured report**

Record final block/module/owner bits, budget integers and averages, per-bit
fractions, calibration/validation metrics, comparisons to FP32/W4A4/P3-T3/
ACT_MASK_15, regional errors, propagation safety, and prediction paths. Label
inferences and do not call logical bit cost measured latency or energy.

- [ ] **Step 9: Re-run verification and commit**

Run: `python -m pytest -q`

Run: `git diff --check && git status --short`

Run: `git add docs/2026-08-17-cspn-task-sensitive-mixed-bit-allocation-results.md && git commit -m "docs: report CSPN task-sensitive bit allocation"`
