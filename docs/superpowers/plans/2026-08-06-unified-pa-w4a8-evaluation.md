# Unified Propagation-Aware W4A8 Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the missing propagation-aware W4A8 control, prove that every run uses the pinned official model source and an exactly compatible converged checkpoint, and regenerate one clean four-model NYU evaluation.

**Architecture:** Keep every official model class and forward path unchanged. Extend the existing propagation configuration matrix and plotting contract, then add source/checkpoint provenance at the shared model-loading boundary so an incompatible architecture fails before calibration. Run all configurations from one revision into a new result root using fixed calibration and validation indices.

**Tech Stack:** Python 3.7-compatible Python, PyTorch, NumPy, Matplotlib, pytest/unittest, CUDA model environments, Git submodules.

---

## File Map

- Modify `scripts/run_nyu_rtn_quantization.py`: add `PA_W4A8`, persist and validate run identity during append.
- Modify `scripts/export_nyu_predictions.py`: validate checkpoint keys and collect model/checkpoint provenance.
- Modify `scripts/plot_propagation_aware_quantization.py`: include W4A8 in metric and prediction plots.
- Modify `tests/test_run_nyu_rtn_quantization.py`: specify W4A8 bit contracts and append identity rejection.
- Modify `tests/test_export_nyu_predictions.py`: specify strict checkpoint compatibility and provenance behavior.
- Modify `tests/test_plot_propagation_aware_quantization.py`: require W4A8 payloads in plot generation.
- Modify `README.md`: document the unified command and replace the result table after evaluation.
- Create `profile_logs/nyu_propagation_aware_quantization_unified/`: ignored runtime metrics, predictions, figures, and summary.

### Task 1: Specify The Missing PA W4A8 Contract

**Files:**
- Modify: `tests/test_run_nyu_rtn_quantization.py`
- Modify: `scripts/run_nyu_rtn_quantization.py:102-135`

- [ ] **Step 1: Write the failing configuration test**

Change the expected propagation matrix to include `PA_W4A8` before
`PA_W8A8`, then assert its complete bit contract:

```python
w4a8 = dict((config["name"], config) for config in configs)["PA_W4A8"]
self.assertEqual((w4a8["w_bits"], w4a8["a_bits"]), (4, 8))
self.assertEqual(w4a8["propagation"], {
    "affinity_bits": 8,
    "confidence_bits": 8,
    "offset_bits": 8,
    "state_bits": 8,
    "coefficient_fraction_bits": 13,
})
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m pytest -q tests/test_run_nyu_rtn_quantization.py::RTNExperimentRunnerTest::test_propagation_backend_has_cumulative_ablation_matrix
```

Expected: fail because `PA_W4A8` is absent.

- [ ] **Step 3: Add the minimal configuration**

Insert this entry immediately before `PA_W8A8`:

```python
dict(base, name="PA_W4A8", w_bits=4, a_bits=8,
     propagation=dict(
         propagation_a4, affinity_bits=8, offset_bits=8,
         state_bits=8)),
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the command from Step 2. Expected: pass.

- [ ] **Step 5: Commit the configuration change**

```bash
git add scripts/run_nyu_rtn_quantization.py tests/test_run_nyu_rtn_quantization.py
git commit -m "feat: add propagation-aware W4A8 control"
```

### Task 2: Enforce Official Source And Checkpoint Compatibility

**Files:**
- Create: `tests/test_export_nyu_predictions.py`
- Modify: `scripts/export_nyu_predictions.py:20-95`
- Modify: `scripts/run_nyu_rtn_quantization.py:660-665,956-1015`

- [ ] **Step 1: Write failing checkpoint-validation tests**

Use a small real `torch.nn.Module` and assert that the helper accepts an exact
state dict, rejects unexpected keys, and ignores only CSPN's validated dynamic
fixed kernel:

```python
def test_checkpoint_validation_rejects_architecture_mismatch():
    model = torch.nn.Linear(2, 1)
    state = dict(model.state_dict())
    state["unexpected"] = torch.ones(1)
    with pytest.raises(RuntimeError, match="unexpected checkpoint keys"):
        predictions.load_model_state(model, state, "dyspn")
```

Add a provenance test that rejects a source path outside the expected official
root and verifies SHA256 output for a temporary checkpoint.

- [ ] **Step 2: Run the new test file and verify RED**

```bash
python -m pytest -q tests/test_export_nyu_predictions.py
```

Expected: fail because the validation/provenance helpers do not exist.

- [ ] **Step 3: Implement strict loading and provenance**

Add Python 3.7-compatible helpers that:

```python
ALLOWED_MISSING_KEYS = {
    "cspn": {"post_process_layer.sum_conv.weight"},
    "dyspn": set(),
    "nlspn": set(),
    "completionformer": set(),
}
```

Validate and remove only CSPN's dynamic all-one sum kernel, load with
`strict=False`, compare the returned missing/unexpected sets exactly, and raise
on any mismatch. Resolve `inspect.getfile(type(model))`, require it
to be below `models/`, `external/DySPN`, `external/NLSPN_ECCV20`, or
`external/CompletionFormer` as appropriate, calculate source/checkpoint
SHA256, and read the pinned submodule commit with `git rev-parse HEAD`.

Attach this record to the builder metadata returned by `build_model` and write
it as `model_provenance` in result metadata. On append, reject a provenance or
checkpoint hash mismatch before calibration/evaluation rows are merged.

- [ ] **Step 4: Run focused tests and verify GREEN**

```bash
python -m pytest -q tests/test_export_nyu_predictions.py tests/test_run_nyu_rtn_quantization.py
```

Expected: all pass.

- [ ] **Step 5: Audit actual model sources without quantization**

Instantiate each current run directory in its required environment and print
the recorded class, resolved source path, source hash, submodule commit, and
checkpoint hash. Expected roots and commits:

```text
CSPN: models/, source identical to /workspace/CSPN/cspn_pytorch/models/
DySPN: external/DySPN, d4871eeabc8797d821873a2a41daf359466a7255
NLSPN: external/NLSPN_ECCV20, ba33fa5d9ea62ca970026a145ab18fab76d79d4a
CompletionFormer: external/CompletionFormer, 2744eddee9b57595dc3064f7d342569736a6803b
```

- [ ] **Step 6: Commit strict provenance enforcement**

```bash
git add scripts/export_nyu_predictions.py scripts/run_nyu_rtn_quantization.py tests/test_export_nyu_predictions.py tests/test_run_nyu_rtn_quantization.py
git commit -m "fix: enforce official model evaluation provenance"
```

### Task 3: Include W4A8 In Plotting And Documentation

**Files:**
- Modify: `tests/test_plot_propagation_aware_quantization.py`
- Modify: `scripts/plot_propagation_aware_quantization.py:17-35,179-250`
- Modify: `README.md:56-100`

- [ ] **Step 1: Write the failing plotting test**

Add `PA_W4A8` to the synthetic configuration set and assert it is a required
prediction configuration with the label `W4A8`. Extend comparison panels so
the detailed output includes FP32, W4A8, W8A8, and the selected W4A4 result.

- [ ] **Step 2: Run the plotting tests and verify RED**

```bash
python -m pytest -q tests/test_plot_propagation_aware_quantization.py
```

Expected: fail because plotting does not load or label `PA_W4A8`.

- [ ] **Step 3: Update plotting constants and panels**

Add:

```python
"PA_W4A8",
```

to `CONFIGS`, add `"PA_W4A8": "W4A8"` to `LABELS`, and include W4A8/W8A8
predictions and absolute-error maps in comparison output while retaining the
best W4A4 diagnostic panel.

- [ ] **Step 4: Verify plots and update the run command**

Run the focused test and update README configuration/export lists to include
`PA_W4A8`. Do not update measured result values before the real run.

- [ ] **Step 5: Commit plotting support**

```bash
git add scripts/plot_propagation_aware_quantization.py tests/test_plot_propagation_aware_quantization.py README.md
git commit -m "feat: compare PA W4A8 predictions"
```

### Task 4: Run Software Verification

**Files:**
- Verify only

- [ ] **Step 1: Run the base-environment suite**

```bash
python -m pytest -q tests
```

Expected: all tests pass.

- [ ] **Step 2: Run the CompletionFormer CUDA-extension suite**

```bash
conda run -n completionformer-py37 python -m pytest -q tests
```

Expected: all tests pass with the official deformable-convolution extension
available.

- [ ] **Step 3: Verify repository diff hygiene**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; unrelated pre-existing edge-runner changes and
the DySPN untracked cache remain untouched.

### Task 5: Execute One Clean Four-Model Evaluation

**Files:**
- Create (ignored): `profile_logs/nyu_propagation_aware_quantization_unified/`
- Modify after measurements: `README.md`

- [ ] **Step 1: Confirm GPU and environment availability**

Run `nvidia-smi`, list conda environments, and execute
`scripts/check_migration.py` in each selected environment. Abort rather than
fall back to CPU or a non-official replacement implementation.

- [ ] **Step 2: Remove only a stale target root if present**

Use a previously unused output path. If interrupted output exists, inspect it
and remove only `profile_logs/nyu_propagation_aware_quantization_unified/`;
never remove historical reference logs or source files.

- [ ] **Step 3: Run all seven configurations for each model**

Use `scripts/run_nyu_rtn_quantization.py` with:

```text
--quant-backend propagation
--seed 20260804
--calibration-samples 128
--max-eval-samples 64
--config-names FP32 PA_Generic_W4A4 PA_Constraint PA_OffsetA8 PA_StateA8 PA_W4A8 PA_W8A8
--export-prediction-configs FP32 PA_Generic_W4A4 PA_Constraint PA_OffsetA8 PA_StateA8 PA_W4A8 PA_W8A8
```

Run CSPN iter24, DySPN iter6, NLSPN iter18, and CompletionFormer iter18 with
their configured Python executables and the existing shared sample-metrics
files.

- [ ] **Step 4: Render all figures**

```bash
python scripts/plot_propagation_aware_quantization.py \
  --root profile_logs/nyu_propagation_aware_quantization_unified
```

Expected: four figures per model with W4A8 comparisons present.

- [ ] **Step 5: Validate artifact cardinality and numerical status**

For each model/configuration assert 64 sample rows and 64 NPZ files, identical
ordered calibration/evaluation indices, the expected checkpoint hash and
source provenance, and explicit finite/non-finite counts. Calculate aggregate
RMSE from sample-level MSE rather than averaging rounded display values.

- [ ] **Step 6: Update README with measured results and commit**

Replace the result table with FP32, generic/best W4A4, W4A8, and W8A8 columns,
record the unified result root, and commit only source/docs changes:

```bash
git add README.md
git commit -m "docs: report unified PA W4A8 evaluation"
```

Do not add datasets, checkpoints, NPZ predictions, or generated figures to Git.
