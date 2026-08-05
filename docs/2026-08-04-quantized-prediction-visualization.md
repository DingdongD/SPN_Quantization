# Quantized Prediction Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export fixed-64 predictions for FP32 and four hardware-aligned quantization configurations, then render deterministic GT/prediction/error comparisons for CSPN, DySPN, NLSPN, and CompletionFormer.

**Architecture:** Extend the mixed-precision builder with W8A8 and make prediction export an explicit runner contract. Keep numerical selection/validation in a focused analysis module and plotting in a separate script that consumes persisted NPZ files and formal metrics. Re-run only the five requested configurations with the existing calibration seed and append them to the formal result root.

**Tech Stack:** Python 3.7/3.11, PyTorch, NumPy, Matplotlib, unittest, existing NYU quantization runner and official CUDA/DCN extensions.

---

### Task 1: Hardware-Aligned W8A8 and Explicit Export Selection

**Files:**
- Modify: `scripts/activation_bit_allocation.py`
- Modify: `scripts/run_nyu_rtn_quantization.py`
- Modify: `tests/test_activation_bit_allocation.py`
- Modify: `tests/test_run_nyu_rtn_quantization.py`

- [ ] **Step 1: Write failing W8A8 and export-selection tests**

Add assertions that the mixed configuration list contains a full-model W8A8
entry and that explicit CLI selections replace the legacy name-based defaults:

```python
by_name = dict((row["name"], row) for row in
               allocation.build_mixed_configurations({"enc": "encoder"}, []))
self.assertEqual((by_name["MP_W8A8_full"]["w_bits"],
                  by_name["MP_W8A8_full"]["a_bits"]), (8, 8))
self.assertEqual(by_name["MP_W8A8_full"]["selection"], "full_w8a8")

self.assertTrue(runner.should_export_predictions(
    "MP_W4A4_base", {"MP_W4A4_base"}))
self.assertFalse(runner.should_export_predictions(
    "W4A4_full", {"MP_W4A4_base"}))
self.assertTrue(runner.should_export_predictions("W4A4_full", None))
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python -m unittest tests.test_activation_bit_allocation tests.test_run_nyu_rtn_quantization
```

Expected: failure because `MP_W8A8_full` and the explicit export argument do
not exist.

- [ ] **Step 3: Implement W8A8 and the CLI contract**

Allow `_configuration` to accept `w_bits`, add this configuration immediately
after `MP_W4A8_full`, and preserve its values in the manifest:

```python
configs.append(_configuration(
    "MP_W8A8_full", all_groups, w_bits=8, a_bits=8,
    selection="full_w8a8"))
```

Change the runner helper to:

```python
def should_export_predictions(config_name, explicit_names=None):
    if explicit_names is not None:
        return config_name in explicit_names
    return config_name in LEGACY_PREDICTION_CONFIGS
```

Add `--export-prediction-configs` with `nargs="*"`, validate that every name is
present in the selected configuration list, and pass the resulting set to the
helper.

- [ ] **Step 4: Run focused tests in both environments and verify GREEN**

Run:

```bash
python -m unittest tests.test_activation_bit_allocation tests.test_run_nyu_rtn_quantization
/opt/conda/envs/completionformer-py37/bin/python -m unittest tests.test_activation_bit_allocation tests.test_run_nyu_rtn_quantization
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add scripts/activation_bit_allocation.py scripts/run_nyu_rtn_quantization.py tests/test_activation_bit_allocation.py tests/test_run_nyu_rtn_quantization.py
git commit -m "feat: add aligned W8A8 prediction export"
```

### Task 2: Rich FP32 and Quantized Prediction Payloads

**Files:**
- Modify: `scripts/run_nyu_rtn_quantization.py`
- Modify: `tests/test_run_nyu_rtn_quantization.py`

- [ ] **Step 1: Write failing payload tests**

Use 2x2 NumPy arrays containing one invalid-GT pixel and one nonfinite
prediction. Assert that `prediction_payload` stores raw prediction, GT, FP32,
absolute error, valid-GT mask, and nonfinite mask without converting NaN to a
valid depth:

```python
payload = runner.prediction_payload(
    gt=np.array([[1., 0.], [2., 3.]], np.float32),
    fp32=np.array([[1.1, 4.], [2.1, 3.1]], np.float32),
    pred=np.array([[1.2, 5.], [np.nan, 2.5]], np.float32),
    sample_index=7, model="cspn", config="MP_W4A4_base")
self.assertTrue(np.array_equal(payload["valid_gt"],
                               [[True, False], [True, True]]))
self.assertTrue(payload["nonfinite"][1, 0])
self.assertAlmostEqual(payload["abs_err"][0, 0], 0.2)
self.assertTrue(np.isnan(payload["abs_err"][1, 0]))
```

Also test that explicit `FP32` export writes one payload per captured record.

- [ ] **Step 2: Run the runner tests and verify RED**

Run:

```bash
python -m unittest tests.test_run_nyu_rtn_quantization
```

Expected: failure because `prediction_payload` and FP32 export do not exist.

- [ ] **Step 3: Implement one payload writer used by both paths**

Implement:

```python
def prediction_payload(gt, fp32, pred, sample_index, model, config):
    valid_gt = np.isfinite(gt) & (gt > 1e-4)
    nonfinite = valid_gt & ~np.isfinite(pred)
    abs_err = np.abs(pred - gt).astype(np.float32)
    abs_err[~valid_gt] = np.nan
    return {
        "gt": gt.astype(np.float32),
        "fp32": fp32.astype(np.float32),
        "pred": pred.astype(np.float32),
        "abs_err": abs_err,
        "valid_gt": valid_gt,
        "nonfinite": nonfinite,
        "sample_index": np.array(sample_index),
        "model": np.array(model),
        "config": np.array(config),
    }
```

Add a small `write_prediction_payload` helper and call it for FP32 records when
explicitly selected and for each requested quantized configuration. Write to
`<model-out>/predictions/<config>/sample_<index>.npz`. Remove stale NPZ files
for a selected configuration before exporting so interrupted older runs cannot
masquerade as a complete set.

- [ ] **Step 4: Run runner tests in both environments**

Run:

```bash
python -m unittest tests.test_run_nyu_rtn_quantization
/opt/conda/envs/completionformer-py37/bin/python -m unittest tests.test_run_nyu_rtn_quantization
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add scripts/run_nyu_rtn_quantization.py tests/test_run_nyu_rtn_quantization.py
git commit -m "feat: persist quantized prediction payloads"
```

### Task 3: Prediction Validation and Representative Selection

**Files:**
- Create: `scripts/quantized_prediction_analysis.py`
- Create: `tests/test_quantized_prediction_analysis.py`

- [ ] **Step 1: Write failing metric and selection tests**

Create tests for `prediction_metrics`, `validate_sample_sets`, and
`select_representative_samples`. The selection fixture must prove median, P90,
maximum, and sparse-recovery choices are unique and deterministic. A CSPN
fixture with nonfinite counts must select the highest-invalid sample for the
fourth slot.

```python
selected = analysis.select_representative_samples(rows, model="cspn")
self.assertEqual([row["reason"] for row in selected],
                 ["median_w4a4", "p90_w4a4", "maximum_w4a4",
                  "maximum_nonfinite"])
self.assertEqual(selected[-1]["sample_index"], 9)
```

Metric tests must verify that finite RMSE uses the same mask as
`regional_depth_metrics`, while nonfinite rate uses valid GT pixels as the
denominator.

- [ ] **Step 2: Run the new test module and verify RED**

Run:

```bash
python -m unittest tests.test_quantized_prediction_analysis
```

Expected: import failure because the analysis module does not exist.

- [ ] **Step 3: Implement the focused analysis module**

Provide these public functions:

```python
def prediction_metrics(gt, pred):
    valid_gt = np.isfinite(gt) & (gt > 1e-4)
    finite = valid_gt & np.isfinite(pred)
    diff = pred[finite] - gt[finite]
    return {
        "RMSE": float(np.sqrt(np.mean(diff ** 2))),
        "MAE": float(np.mean(np.abs(diff))),
        "nonfinite_pixels": int(np.count_nonzero(
            valid_gt & ~np.isfinite(pred))),
        "num_pixels": int(np.count_nonzero(finite)),
        "valid_gt_pixels": int(np.count_nonzero(valid_gt)),
    }

def validate_sample_sets(prediction_root, model_configs, expected_indices):
    expected = set(int(index) for index in expected_indices)
    for config in model_configs:
        paths = sorted((Path(prediction_root) / config).glob("sample_*.npz"))
        observed = set(int(np.load(str(path), allow_pickle=False)[
            "sample_index"]) for path in paths)
        if observed != expected or len(paths) != len(expected):
            raise ValueError("prediction sample mismatch for %s" % config)

def select_representative_samples(rows, model):
    by_role = {}
    for row in rows:
        by_role.setdefault(row["role"], {})[int(row["sample_index"])] = row
    base = by_role["w4a4"]
    sparse = by_role["sparse_a8"]
    ordered = sorted(base)
    finite_ordered = [index for index in ordered
                      if np.isfinite(float(base[index]["RMSE"]))]
    selected = []
    for reason, quantile in (("median_w4a4", 0.5), ("p90_w4a4", 0.9)):
        target = float(np.quantile([float(base[index]["RMSE"])
                                   for index in finite_ordered], quantile))
        choice = min((index for index in finite_ordered
                      if index not in selected),
                     key=lambda index: (abs(float(base[index]["RMSE"])-target),
                                        index))
        selected.append(choice)
    selected.append(max((index for index in finite_ordered
                         if index not in selected),
                        key=lambda index: (float(base[index]["RMSE"]), -index)))
    remaining = [index for index in ordered if index not in selected]
    if model == "cspn":
        fourth = max(remaining, key=lambda index: (
            float(base[index]["nonfinite_rate"]), -index))
        fourth_reason = "maximum_nonfinite"
    else:
        fourth = max(remaining, key=lambda index: (
            (float(base[index]["RMSE"])-float(sparse[index]["RMSE"])
             if np.isfinite(float(base[index]["RMSE"]))
             and np.isfinite(float(sparse[index]["RMSE"])) else -np.inf),
            -index))
        fourth_reason = "maximum_sparse_recovery"
    selected.append(fourth)
    reasons = ["median_w4a4", "p90_w4a4", "maximum_w4a4", fourth_reason]
    return [{"model": model, "sample_index": index, "reason": reason}
            for index, reason in zip(selected, reasons)]

def cross_validate_metrics(computed_rows, formal_rows, tolerance=1e-5):
    formal = dict(((row["model"], row["config"], int(row["sample_index"])), row)
                  for row in formal_rows)
    for row in computed_rows:
        key = (row["model"], row["config"], int(row["sample_index"]))
        if key not in formal:
            raise ValueError("missing formal metric for %s" % (key,))
        for field in ("RMSE", "MAE"):
            if not np.isclose(float(row[field]), float(formal[key][field]),
                              rtol=tolerance, atol=tolerance, equal_nan=True):
                raise ValueError("metric mismatch for %s %s" % (key, field))
```

Use stable sorting by `(distance_to_quantile, sample_index)`, prevent duplicate
selection by taking the nearest unselected candidate, and raise `ValueError`
with the model/config/sample in every mismatch message.

- [ ] **Step 4: Run analysis tests in both environments**

Run:

```bash
python -m unittest tests.test_quantized_prediction_analysis
/opt/conda/envs/completionformer-py37/bin/python -m unittest tests.test_quantized_prediction_analysis
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add scripts/quantized_prediction_analysis.py tests/test_quantized_prediction_analysis.py
git commit -m "feat: select quantized prediction examples"
```

### Task 4: Comparison Rendering

**Files:**
- Create: `scripts/plot_quantized_prediction_comparison.py`
- Create: `tests/test_plot_quantized_prediction_comparison.py`

- [ ] **Step 1: Write failing render-mask and layout tests**

Test `depth_rgba` and `error_rgba` with valid, invalid-GT, and nonfinite pixels.
Assert the exact semantic colors:

```python
self.assertTrue(np.allclose(rgba[0, 1], [0.85, 0.85, 0.85, 1.0]))
self.assertTrue(np.allclose(rgba[1, 0], [1.0, 0.0, 1.0, 1.0]))
```

Test that `panel_specifications` returns six prediction columns, two rows per
sample, the correct model-specific sparse config, and unrotated labels.

- [ ] **Step 2: Run plotting tests and verify RED**

Run:

```bash
python -m unittest tests.test_plot_quantized_prediction_comparison
```

Expected: import failure because the plotting module does not exist.

- [ ] **Step 3: Implement loading, semantic RGBA conversion, and plots**

The script accepts:

```text
--root profile_logs/nyu_activation_bit_allocation
--out-dir profile_logs/nyu_activation_bit_allocation/prediction_comparison
```

It validates all NPZ files before drawing, writes
`representative_samples.csv` and `prediction_metrics.csv`, and creates one
8x6 panel figure per model plus the compact overview. Use `viridis` for 0-10 m
depth, `magma` for 0-3 m absolute error, light gray `[0.85, 0.85, 0.85, 1]` for
invalid GT, and magenta `[1, 0, 1, 1]` for nonfinite predictions. Add separate
depth and error colorbars, Arial-compatible fonts, no figure title, and no tick
label rotation.

- [ ] **Step 4: Run plotting and analysis tests in both environments**

Run:

```bash
python -m unittest tests.test_quantized_prediction_analysis tests.test_plot_quantized_prediction_comparison
/opt/conda/envs/completionformer-py37/bin/python -m unittest tests.test_quantized_prediction_analysis tests.test_plot_quantized_prediction_comparison
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add scripts/plot_quantized_prediction_comparison.py tests/test_plot_quantized_prediction_comparison.py
git commit -m "feat: plot quantized depth predictions"
```

### Task 5: Four-Model Fixed-64 Export

**Files:**
- Generate: `profile_logs/nyu_activation_bit_allocation/<model>/predictions/*`

- [ ] **Step 1: Export CSPN and DySPN on separate GPUs**

Run the mixed backend with `--append`, calibration 128, evaluation 64, seed
`20260804`, and explicit export names. CSPN configurations are `FP32
MP_W8A8_full MP_W4A4_base MP_heads_A8 MP_W4A8_full`; DySPN replaces
`MP_heads_A8` with `MP_encoder_A8`.

```bash
python scripts/run_nyu_rtn_quantization.py \
  --run-dir output/nyu_iteration_sweep_converged/cspn_iter24 \
  --checkpoint best.pt \
  --sample-metrics profile_logs/nyu_activation_outliers/cspn/sample_metrics.csv \
  --out-dir profile_logs/nyu_activation_bit_allocation --device cuda:1 \
  --seed 20260804 --calibration-samples 128 --max-eval-samples 64 \
  --quant-backend mixed --sensitivity-root profile_logs/nyu_activation_outliers \
  --candidate-limit 4 --append \
  --config-names FP32 MP_W8A8_full MP_W4A4_base MP_heads_A8 MP_W4A8_full \
  --export-prediction-configs FP32 MP_W8A8_full MP_W4A4_base MP_heads_A8 MP_W4A8_full

python scripts/run_nyu_rtn_quantization.py \
  --run-dir output/nyu_iteration_sweep_converged/dyspn_iter9 \
  --checkpoint best.pt \
  --sample-metrics profile_logs/nyu_activation_outliers/dyspn/sample_metrics.csv \
  --out-dir profile_logs/nyu_activation_bit_allocation --device cuda:2 \
  --seed 20260804 --calibration-samples 128 --max-eval-samples 64 \
  --quant-backend mixed --sensitivity-root profile_logs/nyu_activation_outliers \
  --candidate-limit 4 --append \
  --config-names FP32 MP_W8A8_full MP_W4A4_base MP_encoder_A8 MP_W4A8_full \
  --export-prediction-configs FP32 MP_W8A8_full MP_W4A4_base MP_encoder_A8 MP_W4A8_full
```

- [ ] **Step 2: Export NLSPN and CompletionFormer with official DCN**

Use `/opt/conda/envs/completionformer-py37/bin/python`,
`CUDA_LAUNCH_BLOCKING=1`, and `cuda:0`. NLSPN uses `MP_site01_A8` and
CompletionFormer uses `MP_heads_A8`; all other names match Task 5 Step 1.

```bash
CUDA_LAUNCH_BLOCKING=1 /opt/conda/envs/completionformer-py37/bin/python \
  scripts/run_nyu_rtn_quantization.py \
  --run-dir output/nyu_iteration_sweep_converged/nlspn_iter12 \
  --checkpoint best.pt \
  --sample-metrics profile_logs/nyu_activation_outliers/nlspn/sample_metrics.csv \
  --out-dir profile_logs/nyu_activation_bit_allocation --device cuda:0 \
  --seed 20260804 --calibration-samples 128 --max-eval-samples 64 \
  --quant-backend mixed --sensitivity-root profile_logs/nyu_activation_outliers \
  --candidate-limit 4 --append \
  --config-names FP32 MP_W8A8_full MP_W4A4_base MP_site01_A8 MP_W4A8_full \
  --export-prediction-configs FP32 MP_W8A8_full MP_W4A4_base MP_site01_A8 MP_W4A8_full

CUDA_LAUNCH_BLOCKING=1 /opt/conda/envs/completionformer-py37/bin/python \
  scripts/run_nyu_rtn_quantization.py \
  --run-dir output/nyu_iteration_sweep_converged/completionformer_iter6 \
  --checkpoint best.pt \
  --sample-metrics profile_logs/nyu_activation_outliers/completionformer/sample_metrics.csv \
  --out-dir profile_logs/nyu_activation_bit_allocation --device cuda:0 \
  --seed 20260804 --calibration-samples 128 --max-eval-samples 64 \
  --quant-backend mixed --sensitivity-root profile_logs/nyu_activation_outliers \
  --candidate-limit 4 --append \
  --config-names FP32 MP_W8A8_full MP_W4A4_base MP_heads_A8 MP_W4A8_full \
  --export-prediction-configs FP32 MP_W8A8_full MP_W4A4_base MP_heads_A8 MP_W4A8_full
```

- [ ] **Step 3: Verify complete and common sample sets**

Run a Python assertion that every `(model, config)` directory has exactly 64
NPZ files, NPZ sample IDs equal metadata evaluation indices, and metadata
calibration/evaluation indices are common across all models.

Expected: 5 configurations x 64 samples for each of four models.

### Task 6: Generate, Inspect, and Verify Formal Outputs

**Files:**
- Generate: `profile_logs/nyu_activation_bit_allocation/prediction_comparison/*`

- [ ] **Step 1: Generate formal CSVs and figures**

Run:

```bash
python scripts/plot_quantized_prediction_comparison.py \
  --root profile_logs/nyu_activation_bit_allocation \
  --out-dir profile_logs/nyu_activation_bit_allocation/prediction_comparison
```

Expected: two CSV files, four model figures, and one overview figure.

- [ ] **Step 2: Inspect all five figures**

Use `view_image` for every PNG. Confirm depth/error scales are consistent,
magenta appears only for nonfinite valid-GT predictions, invalid GT is gray,
labels fit, no panels overlap, and figures contain no overall title.

- [ ] **Step 3: Run the complete relevant test set freshly**

Run in both Python environments:

```bash
python -m unittest tests.test_hardware_aligned_quantization tests.test_activation_bit_allocation tests.test_run_nyu_rtn_quantization tests.test_quantized_prediction_analysis tests.test_plot_quantized_prediction_comparison tests.test_plot_activation_bit_allocation
```

Expected: zero failures in both environments.

- [ ] **Step 4: Validate generated artifacts**

Use Pillow to decode every PNG and NumPy to open every NPZ with
`allow_pickle=False`. Assert all expected CSV files are nonempty and no GPU
process remains running.

- [ ] **Step 5: Commit implementation-only files**

Generated `profile_logs` remain untracked. Commit only scripts and tests not
already committed in Tasks 1-4.
