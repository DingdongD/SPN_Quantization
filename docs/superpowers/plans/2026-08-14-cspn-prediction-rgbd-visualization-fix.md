# CSPN Prediction RGBD Visualization Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct CSPN prediction RGB and sparse-depth visualization without changing inference or evaluation results.

**Architecture:** Add an explicit post-evaluation payload upgrader that preserves the official model RGB as `model_rgb` and attaches natural HDF5 RGB as `rgb`. Keep sparse depth unchanged and mask its zero background only in the plotter.

**Tech Stack:** Python, PyTorch datasets, NumPy NPZ, Matplotlib, pytest.

---

### Task 1: Upgrade Prediction Payload RGB

**Files:**
- Modify: `scripts/evaluate_nyu_cspn_group_a4_qat.py`
- Modify: `tests/test_evaluate_nyu_cspn_group_a4_qat.py`

- [ ] **Step 1: Write a failing payload-upgrade test**

Create five old-schema payloads for one sample, provide a natural RGB dataset
sample, invoke `upgrade_prediction_visuals`, and assert that `model_rgb` exactly
equals the old `rgb` while the new `rgb` equals the natural image.

- [ ] **Step 2: Verify the test fails**

Run: `python -m pytest -q tests/test_evaluate_nyu_cspn_group_a4_qat.py -k visualization`

Expected: fail because `upgrade_prediction_visuals` is missing.

- [ ] **Step 3: Implement the strict atomic upgrader**

Implement `visualization_dataset(saved_args)` with `NyuHdf5Dataset` and
`upgrade_prediction_visuals(root, indices, dataset)`. Require the old exact
schema, finite same-shaped HWC RGB, `[0, 1]` natural values, and matching sample
indices. Write through a sibling `.pending` file and replace the source only
after serialization succeeds. Write `prediction_visualization_manifest.json`.

- [ ] **Step 4: Integrate the upgrader after five-configuration evaluation**

Call the upgrader before prediction coverage validation. Do not alter the model
dataset or any inference input.

- [ ] **Step 5: Run evaluator tests**

Run: `python -m pytest -q tests/test_evaluate_nyu_cspn_group_a4_qat.py`

Expected: all tests pass.

### Task 2: Correct RGB and Sparse Plotting

**Files:**
- Modify: `scripts/plot_nyu_cspn_group_a4_qat.py`
- Modify: `tests/test_plot_nyu_cspn_group_a4_qat.py`

- [ ] **Step 1: Write failing schema and sparse-mask tests**

Require `model_rgb` in the payload, assert `_rgb_image` returns natural RGB
unchanged, and assert `_sparse_image` masks exactly the zero pixels.

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest -q tests/test_plot_nyu_cspn_group_a4_qat.py`

Expected: fail because the plotter still inverse-normalizes RGB and has no
sparse-mask helper.

- [ ] **Step 3: Implement strict visualization helpers**

Validate natural RGB as finite HWC `[0, 1]` data and return it unchanged.
Validate sparse depth as finite nonnegative 2D data and return a masked array
whose mask is exactly `sparse == 0`.

- [ ] **Step 4: Render sparse points over a neutral background**

Use a copied colormap with a neutral `bad` color and the existing GT depth
range. Do not change the sparse payload values.

- [ ] **Step 5: Run plotter tests**

Run: `python -m pytest -q tests/test_plot_nyu_cspn_group_a4_qat.py`

Expected: all tests pass.

### Task 3: Migrate Artifacts and Verify

**Files:**
- Modify: `docs/2026-08-13-cspn-static-dynamic-g8-qat-results.md`

- [ ] **Step 1: Upgrade the existing 320 payloads**

Invoke the production upgrader with the official checkpoint metadata, real NYU
data root, and the fixed 64 indices. Do not rerun inference.

- [ ] **Step 2: Regenerate four figures**

Run `scripts/plot_nyu_cspn_group_a4_qat.py` with the existing evaluation and
figure directories.

- [ ] **Step 3: Inspect the detailed PNG**

Confirm natural RGB colors, 500 visible sparse points, aligned GT/predictions,
and unchanged depth/error panels.

- [ ] **Step 4: Update result documentation**

Document `rgb` as natural display data, `model_rgb` as the exact official input,
and sparse depth as the exact 500-point model input.

- [ ] **Step 5: Run complete verification**

Run: `python -m pytest -q`

Run: `git diff --check`

Verify five finite aggregate rows, 320 new-schema payloads, unchanged metrics
file hashes, and four nonempty figures.
