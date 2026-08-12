# CSPN Activation Prediction Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate strict, comparable detail and 64-sample visualization figures from the audited CSPN activation-resolution NPZ payloads.

**Architecture:** A focused plotting script loads the exact four configuration directories, validates cross-configuration payload identity, computes sample RMSE and global display ranges, and renders two fixed-layout figures. Unit tests exercise contracts and rendering with small synthetic arrays; the production command consumes the existing 64-sample artifacts.

**Tech Stack:** Python, NumPy, Matplotlib, Pillow, unittest/pytest.

---

### Task 1: Define payload and selection contracts

**Files:**
- Create: `scripts/plot_cspn_activation_resolution_predictions.py`
- Create: `tests/test_plot_cspn_activation_resolution_predictions.py`

- [ ] **Step 1: Write failing contract tests**

Create synthetic NPZ payloads for the four exact configurations. Assert that
the loader returns shared sample indices and rejects missing configuration
directories. Assert deterministic representative selection from known RMSE
values.

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_plot_cspn_activation_resolution_predictions.py
```

Expected: collection fails because the plotting module does not exist.

- [ ] **Step 3: Implement strict loading and selection**

Implement `load_predictions()`, `sample_rmse()`,
`select_representative_samples()`, and `global_error_limit()`. Use required
dictionary keys with `[]` and raise on every contract mismatch.

- [ ] **Step 4: Verify contract tests pass**

Run the test file and expect all contract tests to pass.

### Task 2: Render detail and contact-sheet figures

**Files:**
- Modify: `scripts/plot_cspn_activation_resolution_predictions.py`
- Modify: `tests/test_plot_cspn_activation_resolution_predictions.py`

- [ ] **Step 1: Write failing render tests**

Assert that detail and contact-sheet render functions create non-empty PNG
files and that PDF export creates a non-empty PDF from each PNG.

- [ ] **Step 2: Verify the tests fail**

Run the render tests and expect missing render functions.

- [ ] **Step 3: Implement rendering and CLI**

Use shared depth and error normalizations, fixed panel order, masked invalid
pixels, external colorbars, and an `Agg` backend. Add CLI arguments for input,
output, expected sample count, and DPI.

- [ ] **Step 4: Verify the test file passes**

Run the complete plotting test file and expect all tests to pass.

### Task 3: Generate and inspect real outputs

**Files:**
- Runtime output: `profile_logs/nyu_cspn_activation_resolution/figures/`

- [ ] **Step 1: Run the production plotting command**

```bash
PYTHONPATH=. python scripts/plot_cspn_activation_resolution_predictions.py \
  --experiment-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_activation_resolution/cspn \
  --out-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_activation_resolution/figures \
  --expected-samples 64 --dpi 150
```

- [ ] **Step 2: Inspect both PNG figures**

Use image inspection to verify nonblank panels, readable labels, consistent
color scales, and no overlap.

- [ ] **Step 3: Run complete verification**

Run:

```bash
PYTHONPATH=. pytest -q
```

Expected: the complete repository suite passes.

- [ ] **Step 4: Commit source, tests, and documentation**

Commit only the plotting source, tests, design, and plan. Do not commit NPZ,
PNG, or PDF runtime artifacts.
