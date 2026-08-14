# CSPN Stem Mixed-Precision and Branch-Aware Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and evaluate four official-CSPN stem precision contracts on the fixed NYU calibration/evaluation protocol.

**Architecture:** A focused stem controller owns the official `conv1_1` input and weight while the existing hardware instrumentor owns every non-stem W4A4 site. A dedicated runner builds a fresh model per configuration, reuses the persisted stratified-128 calibration identities and fixed-64 validation identities, and writes stem diagnostics beside the existing depth metrics.

**Tech Stack:** Python, PyTorch CUDA, existing CSPN NYU loader, existing Group-8/propagation-aware quantization framework, pytest.

---

### Task 1: Stem Quantization Controller

**Files:**
- Create: `spn_quant/cspn_stem.py`
- Create: `tests/test_cspn_stem.py`

- [ ] **Step 1: Write failing tests for strict contracts**

Add tests that construct a bias-free `Conv2d(4, 3, 3)` and assert:

```python
controller = CSPNStemController(conv)
controller.observe()
conv(torch.cat((rgb, depth), dim=1))
controller.freeze()
controller.configure("STEM_BRANCH_A4")
result = conv(torch.cat((rgb, depth), dim=1))
assert result.shape == (1, 3, 4, 4)
assert controller.contract()["rgb_scale"] != \
    controller.contract()["depth_scale"]
```

Cover merged A4, W8A8, FP16, branch A4, exact RGB/depth channel slicing,
shared per-output-channel weight codes, INT32 partial convolution, Q31
requantization, finite outputs, overflow rejection, and restoration on close.

- [ ] **Step 2: Verify the tests fail because the module is absent**

Run: `pytest -q tests/test_cspn_stem.py`

Expected: collection fails with `ModuleNotFoundError: spn_quant.cspn_stem`.

- [ ] **Step 3: Implement the minimal controller**

Implement explicit modes and configurations without fallback behavior:

```python
STEM_CONFIGS = (
    "STRICT_W4A4", "STEM_W8A8", "STEM_FP16", "STEM_BRANCH_A4",
)

class CSPNStemController(object):
    def observe(self) -> None: ...
    def freeze(self) -> None: ...
    def configure(self, name: str) -> None: ...
    def reset_statistics(self) -> None: ...
    def statistics(self) -> list[dict[str, object]]: ...
    def contract(self) -> dict[str, object]: ...
    def close(self) -> None: ...
```

Use existing `integer_im2col`, `int8_mm_int32`, `requantize_int32`, and
`symmetric_weight_qdq`. Keep one shared W4 scale per output channel across the
full RGBD weight filter. Branch A4 uses independent unsigned RGB/depth scales,
two INT32 partial convolutions, the declared maximum accumulator scale, and an
INT32 checked sum. FP16 uses FP16 operands and returns FP32 output.

- [ ] **Step 4: Run focused tests to green**

Run: `pytest -q tests/test_cspn_stem.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the controller**

```bash
git add spn_quant/cspn_stem.py tests/test_cspn_stem.py
git commit -m "feat: add CSPN branch-aware stem quantization"
```

### Task 2: Configuration Isolation and Existing Framework Integration

**Files:**
- Create: `tests/test_run_nyu_cspn_stem_precision.py`
- Create: `scripts/run_nyu_cspn_stem_precision.py`

- [ ] **Step 1: Write failing configuration tests**

Test exact configuration order, stem ownership, Group-8 non-stem specs, A8
promotion owners, and strict index loading:

```python
assert [row.name for row in build_configurations()] == [
    "STRICT_W4A4", "STEM_W8A8", "STEM_FP16", "STEM_BRANCH_A4",
]
assert promoted_owners("STEM_W8A8") == {
    ("relu#0", "relu_output"),
    ("rotation.layer4_signed_skip", "boundary"),
}
```

Assert missing keys, duplicate calibration identities, split-aware identity
overlap, non-128 calibration sets, and non-64 evaluation sets raise directly.
Numeric train and validation indices are not treated as the same identity.

- [ ] **Step 2: Verify the runner tests fail**

Run: `pytest -q tests/test_run_nyu_cspn_stem_precision.py`

Expected: collection fails because the runner module is absent.

- [ ] **Step 3: Implement configuration and context construction**

Create a runner that requires all paths explicitly and accesses dictionaries
with `[]`. Build a fresh reference and quantized model for each configuration.
Construct `HardwareAlignedInstrumentor` with `conv1_1` added to externally owned
inputs; keep the existing externally owned outputs, rotation boundaries, and
propagation adapter. Calibrate the instrumentor, rotation controller,
propagation adapter, and stem controller together on the exact stratified 128
indices.

- [ ] **Step 4: Implement per-configuration quantization**

Configure all non-stem ordinary layers as static contiguous Group-8 W4A4. The
stem controller applies the selected contract. Promote only the stem ReLU and
`layer4_signed_skip` rotation boundary for `STEM_W8A8`. Keep guidance FP and
propagation A8/INT16-Q13/INT32 for every configuration.

- [ ] **Step 5: Run runner unit tests to green**

Run: `pytest -q tests/test_run_nyu_cspn_stem_precision.py`

Expected: all tests pass.

- [ ] **Step 6: Commit runner structure**

```bash
git add scripts/run_nyu_cspn_stem_precision.py \
  tests/test_run_nyu_cspn_stem_precision.py
git commit -m "feat: add CSPN stem precision evaluation runner"
```

### Task 3: Evaluation Metrics and Artifact Contract

**Files:**
- Modify: `scripts/run_nyu_cspn_stem_precision.py`
- Modify: `tests/test_run_nyu_cspn_stem_precision.py`

- [ ] **Step 1: Write failing metric and manifest tests**

Test aggregate RMSE/MAE/AbsRel/iRMSE, regional rows, 33-of-64 acceptance,
precision coverage arithmetic, prediction payload coverage, and required
manifest fields. Assert every configuration has exactly 64 unique evaluation
rows.

- [ ] **Step 2: Verify the new tests fail for missing output functions**

Run: `pytest -q tests/test_run_nyu_cspn_stem_precision.py`

Expected: failures identify missing aggregation and artifact functions.

- [ ] **Step 3: Implement evaluation and outputs**

Reuse existing depth metrics and prediction payload helpers. Write:

```text
aggregate_metrics.csv
sample_metrics_64.csv
regional_metrics.csv
stem_activation_metrics.csv
stem_partial_conv_metrics.csv
block_metrics.csv
propagation_metrics.csv
precision_coverage.csv
predictions/<config>/*.npz
manifest.json
```

The manifest records checkpoint and index-file SHA256 values, source identity,
all four contracts, coverage counts, and artifact hashes. No generated file is
written under `/tmp`.

- [ ] **Step 4: Run focused tests to green**

Run: `pytest -q tests/test_run_nyu_cspn_stem_precision.py`

Expected: all tests pass.

- [ ] **Step 5: Commit metrics and artifact handling**

```bash
git add scripts/run_nyu_cspn_stem_precision.py \
  tests/test_run_nyu_cspn_stem_precision.py
git commit -m "feat: report CSPN stem quantization ablation"
```

### Task 4: Verification and Real NYU CUDA Evaluation

**Files:**
- Create: `docs/2026-08-14-cspn-stem-mixed-precision-results.md`
- Generate outside Git: `profile_logs/nyu_cspn_stem_mixed_precision_branch_a4_64/`

- [ ] **Step 1: Run focused and full tests**

Run:

```bash
pytest -q tests/test_cspn_stem.py \
  tests/test_run_nyu_cspn_stem_precision.py
pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run the strict CUDA experiment**

Run with the converged official checkpoint, explicit run/data paths, persisted
stratified calibration indices, and existing fixed evaluation protocol:

```bash
python scripts/run_nyu_cspn_stem_precision.py \
  --run-dir /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24 \
  --checkpoint /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24/best.pt \
  --data-root /workspace/CSPN/cspn_pytorch \
  --calibration-indices /workspace/SPN_Quantization/profile_logs/nyu_cspn_stratified_calibration_128/calibration_indices.json \
  --evaluation-protocol /workspace/SPN_Quantization/profile_logs/nyu_cspn_activation_resolution/cspn/metadata.json \
  --out-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_stem_mixed_precision_branch_a4_64 \
  --device cuda:0 \
  --seed 20260812 \
  --fold-max-error 1e-5
```

Expected: four finite 64-sample evaluations and a complete manifest.

- [ ] **Step 3: Audit artifacts and write results**

Verify unique sample/config coverage, no calibration/evaluation overlap,
finite predictions, metric aggregation, branch scales, INT32 overflow count,
and artifact hashes. Write the results document with absolute and relative RMSE
changes, wins out of 64, regional regressions, stem diagnostic changes, and
high-precision coverage.

- [ ] **Step 4: Run final verification and commit the report**

Run: `git diff --check && pytest -q`

Expected: no whitespace errors and all tests pass.

```bash
git add docs/2026-08-14-cspn-stem-mixed-precision-results.md
git commit -m "docs: report CSPN stem precision ablation"
```
