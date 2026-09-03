# NLSPN Propagation Float-Dtype Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Measure NLSPN W8A8 with FP32, BF16-state, and FP16-state propagation using one strict paired evaluation protocol.

**Architecture:** Extend the existing propagation controller with an explicit float mode and state-storage dtype. Reuse `HardDeploymentP3T3Evaluator` to configure every ordinary Conv2d, ConvTranspose2d, and Linear module as W8A8, while a focused runner selects propagation mode and writes an independent result root.

**Tech Stack:** Python 3, PyTorch, existing official NLSPN runtime, CUDA, pytest, CSV/JSON.

---

### Task 1: Add Explicit Float Propagation Modes

**Files:**
- Modify: `spn_quant/propagation/controller.py`
- Modify: `spn_quant/propagation/adapters.py`
- Test: `tests/test_propagation_aware_adapters.py`

- [ ] **Step 1: Add failing controller/adaptor tests**

```python
def test_float_propagation_modes_are_explicit():
    controller = PropagationQuantController()
    controller.configure_float("bf16")
    assert controller.mode == "float"
    assert controller.float_state_dtype == torch.bfloat16


def test_invalid_float_propagation_dtype_is_rejected():
    controller = PropagationQuantController()
    with pytest.raises(ValueError):
        controller.configure_float("float8")
```

Add an NLSPN adapter test that configures `"fp32"`, `"bf16"`, and `"fp16"`
after observation/freeze and verifies the returned prediction is finite and
the recorded state tensors use the requested storage dtype for the half modes.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_propagation_aware_adapters.py -k float
```

Expected: failure because `configure_float` and `float_state_dtype` do not
exist.

- [ ] **Step 3: Implement the explicit controller mode**

Add the strict mapping and no implicit fallback:

```python
FLOAT_STATE_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

def configure_float(self, state_dtype):
    self.float_state_dtype = FLOAT_STATE_DTYPES[state_dtype]
    self.mode = "float"
    self.config = None
    self._statistics = []
```

Unknown names must raise through indexed mapping access. Keep the existing
`disable()` bypass mode unchanged for ordinary callers.

- [ ] **Step 4: Implement FP32 accumulation with explicit state casts**

In `NLSPNPropagationAdapter._forward`, treat `mode == "float"` as the
uncalibrated float-coefficient path. For the propagation loop, compute each
round from `state.float()`, perform affinity/anchor arithmetic in FP32, and
cast only the recurrent state back to `controller.float_state_dtype` after the
round. The FP32 mode remains float32 throughout. Do not quantize guidance,
offset, affinity, confidence, initial depth, or state in float mode.

- [ ] **Step 5: Run propagation regression tests**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_propagation_aware_adapters.py tests/test_propagation_quantization.py
```

Expected: all existing tests and the new float-mode tests pass.

### Task 2: Add the NLSPN Ablation Runner

**Files:**
- Create: `scripts/run_nyu_nlspn_propagation_dtype_ablation.py`
- Create: `configs/nlspn_propagation_dtype_ablation.json`
- Test: `tests/test_nyu_nlspn_propagation_dtype_ablation.py`

- [ ] **Step 1: Write runner contract tests**

```python
def test_ablation_config_has_exact_modes():
    assert propagation_modes() == (
        "PA_W8A8", "W8A8_FP32_PROP",
        "W8A8_BF16_STATE", "W8A8_FP16_STATE")


def test_ablation_requires_exactly_64_evaluation_samples(config):
    with pytest.raises(ValueError):
        validate_evaluation_protocol(config, tuple(range(63)))
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run: `PYTHONPATH=. pytest -q tests/test_nyu_nlspn_propagation_dtype_ablation.py`

Expected: import failure because the runner does not exist.

- [ ] **Step 3: Implement strict runner configuration**

Load the existing selected-quantization config and require the NLSPN entry,
checkpoint, calibration metadata, and exact fixed evaluation identities.
Create one `HardDeploymentSettings` with `base_weight_bits=8` and
`base_activation_bits=8`, then configure all contract blocks as promoted W8A8.
Use the existing propagation adapter for each explicit mode and never catch
configuration or CUDA dtype exceptions.

- [ ] **Step 4: Implement paired metric and diagnostic writers**

Write `sample_metrics.csv`, `propagation_signal_metrics.csv`,
`propagation_state_metrics.csv`, `summary.json`, and a protocol manifest under
`profile_logs/nyu_nlspn_propagation_dtype_ablation_64/nlspn/`. Pooled RMSE is
computed from float64 SSE over valid pixels; mean per-sample RMSE is stored as a
separate field. Failed BF16/FP16 execution is recorded in the manifest and
does not create a successful metric row.

- [ ] **Step 5: Run runner unit tests**

Run: `PYTHONPATH=. pytest -q tests/test_nyu_nlspn_propagation_dtype_ablation.py`

Expected: all runner contract tests pass.

### Task 3: Execute and Verify the CUDA Experiment

**Files:**
- Runtime output: `/workspace/SPN_Quantization/profile_logs/nyu_nlspn_propagation_dtype_ablation_64/nlspn/`
- Modify: `docs/2026-08-30-nlspn-propagation-float-dtype-ablation-results.md`

- [ ] **Step 1: Run the four paired configurations on one GPU**

Run the runner with the official NLSPN Python environment, fixed seed,
existing 128-sample calibration metadata, and the common 64-sample protocol.

- [ ] **Step 2: Independently recompute pooled metrics**

Read the emitted per-sample rows, recompute pooled RMSE from SSE and valid
pixels, and require exact equality with `summary.json`.

- [ ] **Step 3: Verify propagation invariants**

Require zero non-finite/non-positive predictions, zero contraction violations,
zero coefficient-sum error, and finite state errors for every successful mode.

- [ ] **Step 4: Write the results report**

Report absolute and relative differences against FP32 and PA-W8A8, plus the
interpretation of FP32 propagation recovery versus BF16/FP16 state-storage
effects. Clearly identify any unsupported custom-kernel dtype as a failed
configuration.

- [ ] **Step 5: Run final regression checks**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_propagation_aware_adapters.py tests/test_propagation_quantization.py tests/test_nyu_nlspn_propagation_dtype_ablation.py
python -m compileall -q scripts spn_quant tests
git diff --check
```
