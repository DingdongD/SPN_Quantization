# CSPN Unified BRECQ and QDrop W4A4/W6A6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify the existing BRECQ and QDrop paths to reconstruct and evaluate CSPN W4A4 and W6A6 on the same stratified 128-sample calibration set and deterministic 64-sample evaluation protocol used by P3/T3.

**Architecture:** Parameterize the existing QDrop contract and runner for matched W4A4/W6A6 variants, make both reconstruction runners consume persisted sample protocols, and replay BRECQ/QDrop through the existing propagation-aware edge evaluator with CSPN propagation fixed at A8/Q13/INT32. Extend the existing QDrop orchestrator and plotter; do not add a Python runner or a second quantization framework.

**Tech Stack:** Python 3.11, PyTorch 2.7.1+cu118, CUDA, NumPy, Matplotlib, pytest, existing strict BRECQ/QDrop contracts, official CSPN NYU checkpoint.

---

## File Map

- Modify `configs/qdrop_w4a4_official.json` and `spn_quant/qdrop_config.py` for explicit W4A4/W6A6 variants.
- Modify `spn_quant/qdrop_contract.py` for exact 4-bit or 6-bit contracts.
- Modify `scripts/run_nyu_qdrop_reconstruction.py` and `scripts/run_nyu_strict_reconstruction.py` to consume persisted protocols.
- Modify `scripts/run_nyu_rtn_quantization.py` and `scripts/run_nyu_edge_quantization.py` for current CSPN propagation semantics.
- Modify `scripts/run_nyu_qdrop_w4a4.py` and `scripts/plot_nyu_qdrop_w4a4.py` for the four reconstructed configurations.
- Modify existing corresponding tests; add no Python module.

### Task 1: Parameterize Existing QDrop Configuration

**Files:**
- Modify: `tests/test_qdrop_config.py`
- Modify: `configs/qdrop_w4a4_official.json`
- Modify: `spn_quant/qdrop_config.py`

- [ ] **Step 1: Write failing precision tests**

Change the test payload to require:

```python
"quantization": {
    "variants": [
        {"name": "W4A4", "weight_bits": 4,
         "activation_bits": 4, "official": 1},
        {"name": "W6A6", "weight_bits": 6,
         "activation_bits": 6, "official": 0},
    ],
    "weight_clip_ratio": 1.0,
    "activation_scale_minimum": 1.0e-8,
},
```

Assert direct lookup of both variants. Reject duplicate names, W4A6, W8A8,
missing variants, and an official W6A6 declaration.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_qdrop_config.py`

Expected: failures because the loader requires scalar W4A4 fields.

- [ ] **Step 3: Implement strict parsing**

Add `QDropPrecisionConfig`, parse `quantization.variants` with direct `[]`
access, and expose exact `precision(name)` lookup. Accept only `(4, 4)` and
`(6, 6)`. Require W4A4 to be official and W6A6 to be an extension. Update the
existing JSON and set formal evaluation seed to `20260812`.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest -q tests/test_qdrop_config.py`

```bash
git add configs/qdrop_w4a4_official.json spn_quant/qdrop_config.py tests/test_qdrop_config.py
git commit -m "feat: parameterize strict QDrop precision"
```

### Task 2: Generalize Existing Exact QDrop Contract

**Files:**
- Modify: `tests/test_qdrop_contract.py`
- Modify: `spn_quant/qdrop_contract.py`
- Modify: `spn_quant/qdrop_activation.py`

- [ ] **Step 1: Write failing W6A6 contract tests**

Create a 6-bit version of the smallest existing contract fixture. Assert
build/save/load and `QDropContractInstrumentor.configure(6, 6, ...)` preserve
integer codes, scales, bits, and edge ownership. Assert W4A6 and runtime versus
contract bit mismatch fail.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_qdrop_contract.py tests/test_qdrop_activation.py`

Expected: W6A6 fails at the current W4A4 constants.

- [ ] **Step 3: Derive bits from the strict contract**

Validate every weight and activation entry against the declared matched pair,
restricted to `(4, 4)` or `(6, 6)`. Build quantizers from those fields and
reject runtime mismatch. Do not change QDrop masks, scale optimization, or hard
rounding.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest -q tests/test_qdrop_contract.py tests/test_qdrop_activation.py`

```bash
git add spn_quant/qdrop_contract.py spn_quant/qdrop_activation.py tests/test_qdrop_contract.py
git commit -m "feat: replay exact QDrop W4 and W6 contracts"
```

### Task 3: Bind QDrop Reconstruction to Persisted Samples

**Files:**
- Modify: `tests/test_qdrop_reconstruction_runner.py`
- Modify: `scripts/run_nyu_qdrop_reconstruction.py`

- [ ] **Step 1: Write failing protocol tests**

Replace random split tests with:

```python
split = build_calibration_split(
    calibration_indices=tuple(range(1000, 1128)),
    reconstruction_samples=112,
    validation_samples=16,
)
assert split.reconstruction == tuple(range(1000, 1112))
assert split.validation == tuple(range(1112, 1128))
```

Assert optimizer seeds do not change identities. Require CLI precision,
calibration-indices, calibration-metadata, and evaluation-protocol. Require the
manifest to record actual bits and all source hashes.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_qdrop_reconstruction_runner.py`

Expected: random sampling and W4A4 manifest assumptions fail.

- [ ] **Step 3: Implement exact protocol and precision use**

Use `run_nyu_cspn_stem_precision.index_protocol` and source metadata validation.
Split the persisted ordered 128 indices into 112/16. Remove random permutation
and numeric train/validation index overlap logic because they are different
dataset splits. Resolve the requested config precision and propagate its bits
through weight rounding, activation banks, contracts, and manifests.

Replace old propagation A4 validation with affinity/confidence/offset/state A8,
Q13 coefficients, and INT32 accumulation.

- [ ] **Step 4: Verify GREEN, style, and commit**

Run: `python -m pytest -q tests/test_qdrop_reconstruction_runner.py tests/test_qdrop_config.py`

Run: `rg -n "getattr\(|\.get\(|except |fallback" scripts/run_nyu_qdrop_reconstruction.py spn_quant/qdrop_*.py`

Expected: tests pass and no new defensive fallback patterns appear.

```bash
git add scripts/run_nyu_qdrop_reconstruction.py tests/test_qdrop_reconstruction_runner.py
git commit -m "fix: align QDrop reconstruction data protocol"
```

### Task 4: Bind Existing BRECQ to the Same Calibration Set

**Files:**
- Modify: `tests/test_strict_reconstruction_runner.py`
- Modify: `scripts/run_nyu_strict_reconstruction.py`

- [ ] **Step 1: Write failing persisted-index tests**

Test exact ordered 128-index loading and rejection of duplicate, wrong-count,
metadata-hash, checkpoint, and seed mismatches. Require the three protocol paths
in parser tests.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_strict_reconstruction_runner.py`

Expected: the current random `RandomState` calibration path fails the contract.

- [ ] **Step 3: Implement shared protocol use**

Replace random selection with the persisted ordered 128 indices and record all
hashes in the deployment contract and strict manifest. Keep existing
`--w-bits`, invoked as 4 and 6. Require `--eval-samples 0`; common evaluation
uses the fixed 64 samples.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest -q tests/test_strict_reconstruction_runner.py tests/test_strict_reconstruction.py`

```bash
git add scripts/run_nyu_strict_reconstruction.py tests/test_strict_reconstruction_runner.py
git commit -m "fix: align BRECQ reconstruction data protocol"
```

### Task 5: Align Runtime Quantization and Propagation

**Files:**
- Modify: `tests/test_run_nyu_rtn_quantization.py`
- Modify: `tests/test_semantic_edge_runner.py`
- Modify: `scripts/run_nyu_rtn_quantization.py`
- Modify: `scripts/run_nyu_edge_quantization.py`

- [ ] **Step 1: Write failing runtime tests**

Require `PA_W4A4_PROP_A8` and `PA_W6A6_PROP_A8` with ordinary bits `(4,4)` and
`(6,6)`, Group-8 static MinMax, FP32 bias, and propagation `(8,8,8,8,13)`.
Require edge-loader manifest, contract, and runtime bits to match for QDrop and
BRECQ.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_run_nyu_rtn_quantization.py tests/test_semantic_edge_runner.py`

Expected: missing W6 and W4-only loader failures.

- [ ] **Step 3: Add explicit current-protocol configurations**

Add both named configurations to the existing propagation registry with
`quantize_bias=False`. Generalize strict QDrop loading from fixed 4 to the
contract pair. BRECQ remains exact weight-contract plus evaluator-owned A4/A6.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest -q tests/test_run_nyu_rtn_quantization.py tests/test_semantic_edge_runner.py tests/test_qdrop_contract.py`

```bash
git add scripts/run_nyu_rtn_quantization.py scripts/run_nyu_edge_quantization.py tests/test_run_nyu_rtn_quantization.py tests/test_semantic_edge_runner.py
git commit -m "feat: align reconstructed CSPN runtime precision"
```

### Task 6: Extend Existing Orchestration and Aggregation

**Files:**
- Modify: `tests/test_qdrop_w4a4_evaluation.py`
- Modify: `scripts/run_nyu_qdrop_w4a4.py`

- [ ] **Step 1: Write failing matrix tests**

Use `models=("cspn",)` and `precisions=("W4A4", "W6A6")`. Require six QDrop
jobs, two BRECQ jobs, all three seeds, explicit protocol arguments, and no GPU
collision per wave. Aggregation must contain FP32, both RTN, both BRECQ, both
QDrop, and P3/T3 on identical 64 samples. Non-finite output is an explicit
negative result, not a dropped row or fallback.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_qdrop_w4a4_evaluation.py`

Expected: four-model W4A4-only assumptions fail.

- [ ] **Step 3: Parameterize the existing orchestrator**

Retain the current file. Add required model, precisions, calibration paths,
evaluation protocol, P3/T3 root, and device arguments. Build BRECQ commands
with the existing strict runner and QDrop commands with the existing QDrop
runner. Key every output by method, precision, and seed. Reject incomplete
coverage and mismatched GT/FP32 hashes.

Select the displayed QDrop seed by median 16-sample reconstruction-validation
RMSE before reading test metrics. Write one artifact-hashed manifest.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest -q tests/test_qdrop_w4a4_evaluation.py`

```bash
git add scripts/run_nyu_qdrop_w4a4.py tests/test_qdrop_w4a4_evaluation.py
git commit -m "feat: orchestrate unified CSPN reconstruction matrix"
```

### Task 7: Extend Existing Prediction Figures

**Files:**
- Modify: `tests/test_plot_nyu_qdrop_w4a4.py`
- Modify: `scripts/plot_nyu_qdrop_w4a4.py`

- [ ] **Step 1: Write failing figure tests**

Require W4 columns `GT, FP32, RTN, BRECQ, QDrop` and W6 columns `GT, FP32,
RTN, BRECQ, QDrop, P3/T3`. Require byte-identical identity, GT, FP32, RGB, and
sparse arrays across sources. Require the preselected validation seed rather
than a test-selected or averaged prediction.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_plot_nyu_qdrop_w4a4.py`

Expected: current W4-only and prediction-mean behavior fails.

- [ ] **Step 3: Render strict aligned outputs**

Generate W4A4, W6A6, and aggregate RMSE PNG/PDF files. Retain Arial, no title,
horizontal labels, grids behind bars, shared depth/error limits, and exact
sample validation.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest -q tests/test_plot_nyu_qdrop_w4a4.py`

```bash
git add scripts/plot_nyu_qdrop_w4a4.py tests/test_plot_nyu_qdrop_w4a4.py
git commit -m "feat: plot unified CSPN reconstruction results"
```

### Task 8: Formal GPU Evaluation and Safe Historical Cleanup

**Files:**
- Modify after measurement: `docs/2026-08-07-strict-w4a4-fp4-reconstruction-results.md`
- Generated: `profile_logs/nyu_cspn_unified_brecq_qdrop_w4a4_w6a6_64/`
- Delete after audit: `profile_logs/nyu_qdrop_w4a4/`
- Delete after audit: `profile_logs/nyu_brecq_cspn_pa_w4a4/`

- [ ] **Step 1: Run focused and full tests**

```bash
python -m pytest -q tests/test_qdrop_config.py tests/test_qdrop_activation.py tests/test_qdrop_contract.py tests/test_qdrop_reconstruction_runner.py tests/test_strict_reconstruction_runner.py tests/test_semantic_edge_runner.py tests/test_qdrop_w4a4_evaluation.py tests/test_plot_nyu_qdrop_w4a4.py
python -m pytest -q
```

Expected: zero failures.

- [ ] **Step 2: Verify immutable inputs and GPUs**

Run: `sha256sum /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24/best.pt`

Expected checkpoint hash: `482fb9532b27bdb0e529da14845d9a63ab546e28974d90dd1197b8f704066855`.

Run: `nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader`

Expected: four available A100 GPUs.

- [ ] **Step 3: Run existing orchestrator phases**

Run the modified existing script four times, replacing `PHASE` in this exact
command with `brecq`, `formal`, `evaluate`, and `audit` in that order:

```bash
python scripts/run_nyu_qdrop_w4a4.py \
  --config configs/qdrop_w4a4_official.json \
  --phase PHASE \
  --model cspn \
  --precisions W4A4 W6A6 \
  --run-dir /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24 \
  --checkpoint /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24/best.pt \
  --data-root /workspace/CSPN/cspn_pytorch \
  --calibration-indices /workspace/SPN_Quantization/profile_logs/nyu_cspn_stratified_calibration_128/calibration_indices.json \
  --calibration-metadata /workspace/SPN_Quantization/profile_logs/nyu_cspn_stratified_calibration_128/metadata.json \
  --evaluation-protocol /workspace/SPN_Quantization/profile_logs/nyu_cspn_activation_resolution/cspn/metadata.json \
  --p3-t3-root /workspace/SPN_Quantization/profile_logs/nyu_cspn_task_sensitive_mixed_bits_w4a4_budget_64 \
  --out-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_unified_brecq_qdrop_w4a4_w6a6_64 \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
```

Expected: two BRECQ contracts, six QDrop contracts, eight aligned comparison
configurations, and 64 identities each. An explicit non-finite result is valid
negative evidence; a missing artifact is a pipeline failure.

- [ ] **Step 4: Plot and inspect**

Run the existing plotter on the new root. Inspect every PNG with `view_image`
for nonblank aligned RGB/depth/error panels, readable labels, no title, and no
overlap.

- [ ] **Step 5: Audit, then clean only confirmed old roots**

Require checkpoint/protocol hashes, exact 112/16 and 64 identities, contract
bits, propagation A8/Q13, complete seeds, aligned predictions, and artifact
hashes. Record old root sizes and contents, then delete only:

```bash
rm -rf /workspace/SPN_Quantization/profile_logs/nyu_qdrop_w4a4
rm -rf /workspace/SPN_Quantization/profile_logs/nyu_brecq_cspn_pa_w4a4
```

- [ ] **Step 6: Update measured report and verify**

Mark old values superseded and report all four new results, QDrop seed spread,
and deltas versus FP32, RTN, and P3/T3.

Run: `python -m pytest -q`

Run: `git diff --check`

Run: `git status --short`

Expected: zero test failures, no whitespace errors, and only intentional source,
test, config, and documentation changes. Generated profile logs remain ignored.

- [ ] **Step 7: Commit measured documentation**

```bash
git add docs/2026-08-07-strict-w4a4-fp4-reconstruction-results.md
git commit -m "docs: report unified CSPN reconstruction evaluation"
```
