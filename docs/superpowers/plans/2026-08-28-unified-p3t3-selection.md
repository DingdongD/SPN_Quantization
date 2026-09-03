# Unified P3/T3 Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make P3/T3 selection fair across the four SPN models while preserving W4A4 baseline and W8A8 sensitive-module promotion.

**Architecture:** Keep the existing measured P3/T3 candidate generator and hard deployment evaluator. Move common evaluation identities and the relative-RMSE gate into the shared launch contract, then select the minimum W8A8 protection cost among candidates that pass the gate.

**Tech Stack:** Python, pytest, JSON launch contracts, existing CUDA model runtimes.

---

### Task 1: Encode the shared evaluation protocol

**Files:**
- Modify: `configs/three_model_selected_quantization.json`
- Modify: `scripts/launch_nyu_three_model_quantization.py`
- Test: `tests/test_launch_nyu_three_model_quantization.py`

- [ ] **Step 1: Write the failing contract test**

Add assertions that every model launch input uses the shared CSPN evaluation
identity and that the P3/T3 quality gate is present in the launch spec.

- [ ] **Step 2: Run the focused launch tests**

Run: `pytest -q tests/test_launch_nyu_three_model_quantization.py`

Expected: FAIL because the current JSON contains model-local `0..63`
evaluation identities and no relative-RMSE gate.

- [ ] **Step 3: Implement the shared protocol fields**

Declare one shared evaluation metadata path and one P3/T3 policy object in
the JSON contract. The launcher must validate that every model's
`evaluation_protocol.json` resolves to the same ordered identity, sample
count, split, and metric aggregation policy before producing jobs.

- [ ] **Step 4: Run the focused launch tests again**

Run: `pytest -q tests/test_launch_nyu_three_model_quantization.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/three_model_selected_quantization.json scripts/launch_nyu_three_model_quantization.py tests/test_launch_nyu_three_model_quantization.py
git commit -m "fix: unify P3 T3 evaluation contract"
```

### Task 2: Select candidates by accuracy gate and protection cost

**Files:**
- Modify: `scripts/run_nyu_model_p3t3_search.py`
- Test: `tests/test_run_nyu_model_p3t3_search.py`

- [ ] **Step 1: Write the failing selection tests**

Add a measured evaluator where the lowest-RMSE candidate has higher W8A8 cost
than another candidate that passes a 10 percent paired-baseline gate. Assert
that the lower-cost passing candidate is selected. Add a second test asserting
that no passing candidate raises a clear quality-gate error.

- [ ] **Step 2: Run the focused search tests**

Run: `pytest -q tests/test_run_nyu_model_p3t3_search.py`

Expected: FAIL because selection currently minimizes RMSE inside only the
normalized-cost upper bound.

- [ ] **Step 3: Implement the gate and tie-break order**

Add an explicit relative-RMSE limit to the search API and artifact settings.
Compute each valid candidate's mean per-sample RMSE against the measured
uniform W4A4 baseline's paired FP32 reference, reject candidates above the
limit, and select by W8A8 protection cost followed by mean RMSE and pooled
RMSE. Keep W4/W8 and A4/A8 assignments paired.

- [ ] **Step 4: Run the focused search tests again**

Run: `pytest -q tests/test_run_nyu_model_p3t3_search.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_nyu_model_p3t3_search.py tests/test_run_nyu_model_p3t3_search.py
git commit -m "fix: gate P3 T3 selection by paired accuracy"
```

### Task 3: Validate artifact and cross-model invariants

**Files:**
- Modify: `scripts/evaluate_nyu_selected_quantization.py`
- Modify: `tests/test_evaluate_nyu_selected_quantization.py`
- Modify: `tests/test_launch_nyu_three_model_quantization.py`

- [ ] **Step 1: Add invariant tests**

Test that a published P3/T3 artifact rejects mismatched evaluation identities,
missing paired FP32 metrics, a candidate above the relative-RMSE threshold,
and any assignment containing a non-paired W4/W8 or A4/A8 promotion.

- [ ] **Step 2: Run the tests to verify the new failures**

Run: `pytest -q tests/test_evaluate_nyu_selected_quantization.py tests/test_launch_nyu_three_model_quantization.py`

Expected: FAIL on the newly required invariants.

- [ ] **Step 3: Implement strict artifact validation**

Read required fields directly from the JSON contract, compare ordered sample
identity hashes, and publish the selected candidate's actual W8A8 cost and
relative RMSE gate result. Do not add fallback values or catch validation
errors.

- [ ] **Step 4: Run the focused invariant tests**

Run: `pytest -q tests/test_evaluate_nyu_selected_quantization.py tests/test_launch_nyu_three_model_quantization.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate_nyu_selected_quantization.py tests/test_evaluate_nyu_selected_quantization.py tests/test_launch_nyu_three_model_quantization.py
git commit -m "test: enforce cross-model P3 T3 invariants"
```

### Task 4: Run the full regression suite and launch validation

**Files:**
- Verify: `scripts/run_nyu_model_p3t3_search.py`
- Verify: `scripts/launch_nyu_three_model_quantization.py`
- Verify: `configs/three_model_selected_quantization.json`

- [ ] **Step 1: Run all relevant tests**

Run: `pytest -q tests/test_run_nyu_model_p3t3_search.py tests/test_evaluate_nyu_selected_quantization.py tests/test_launch_nyu_three_model_quantization.py tests/test_train_nyu_selected_qat.py`

Expected: all tests pass.

- [ ] **Step 2: Validate the launch contract without starting GPU jobs**

Run: `python scripts/launch_nyu_three_model_quantization.py --config configs/three_model_selected_quantization.json --validate-only`

Expected: exit 0 and report one shared evaluation identity for all models.

- [ ] **Step 3: Inspect the generated protocol metadata**

Verify that every model artifact records 64 identical evaluation indices,
128 calibration indices, paired FP32 metrics, the 10 percent gate, and the
actual W8A8 protection costs.

- [ ] **Step 4: Commit the verified integration**

```bash
git status --short
git log -4 --oneline
```

