# Propagation-Aware SPN Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a calibrated, propagation-aware integer simulation for CSPN, DySPN, NLSPN, and CompletionFormer, then evaluate and visualize its behavior on the fixed 64-sample NYU set.

**Architecture:** Keep the existing hardware-aligned W4A4 CNN instrumentor, but assign propagation-head outputs to a new SPN-domain controller. The controller quantizes raw affinity/logits before model-specific normalization, represents normalized coefficients as signed Q13 INT16 values, uses unsigned A8 confidence/gates and INT32 conceptual accumulators, restores sparse anchors per iteration, and records invariant and propagation-error metrics.

**Tech Stack:** Python 3, PyTorch, NumPy, Matplotlib, existing official SPN model integrations, `unittest`/`pytest`, CUDA deformable convolution extensions.

---

## File Map

- Create `spn_quant/propagation/__init__.py`: public propagation-aware API.
- Create `spn_quant/propagation/fixed_point.py`: A4/A8 QDQ, Q13 residual normalization, and LUT softmax.
- Create `spn_quant/propagation/controller.py`: calibration, configuration, statistics, and per-forward captures.
- Create `spn_quant/propagation/adapters.py`: CSPN, DySPN, NLSPN, and CompletionFormer runtime adapters.
- Modify `scripts/hardware_aligned_quantization.py`: externally owned output boundaries.
- Modify `scripts/run_nyu_rtn_quantization.py`: propagation backend, configurations, metrics, and prediction export.
- Create `scripts/plot_propagation_aware_quantization.py`: aggregate and detailed static figures.
- Create `tests/test_propagation_fixed_point.py`: integer primitive tests.
- Create `tests/test_propagation_aware_adapters.py`: model-semantic propagation tests.
- Modify `tests/test_hardware_aligned_quantization.py`: owned-output tests.
- Modify `tests/test_run_nyu_rtn_quantization.py`: configuration and persistence tests.
- Create `tests/test_plot_propagation_aware_quantization.py`: plot selection and rendering tests.

### Task 1: Fixed-Point Propagation Primitives

**Files:**
- Create: `spn_quant/propagation/__init__.py`
- Create: `spn_quant/propagation/fixed_point.py`
- Test: `tests/test_propagation_fixed_point.py`

- [ ] **Step 1: Write failing tests for Q13 range and exact residual identities**

```python
import torch

from spn_quant.propagation.fixed_point import (
    Q13_ONE,
    normalize_signed_q13,
    softmax_logits_q13,
    unsigned_unit_qdq,
)


def test_signed_normalization_derives_center_without_int16_overflow():
    raw = torch.tensor([[[[-1.0]], [[0.0]], [[0.0]]]])
    neighbor, center, codes = normalize_signed_q13(
        raw, denominator_floor=False, eps=0.0)
    assert codes.dtype == torch.int16
    assert int(center.item()) == 2 * Q13_ONE
    assert int(center.item() + codes.sum().item()) == Q13_ONE
    torch.testing.assert_close(neighbor, codes.float() / Q13_ONE)


def test_softmax_q13_is_nonnegative_and_has_exact_sum():
    logits = torch.tensor([[[[3.0]], [[1.0]], [[-2.0]]]])
    values, codes = softmax_logits_q13(logits, dim=1, bits=4, maximum=3.0)
    assert torch.all(codes >= 0)
    assert int(codes.sum(dim=1).item()) == Q13_ONE
    torch.testing.assert_close(values.sum(dim=1), torch.ones(1, 1, 1))


def test_unsigned_unit_a8_preserves_endpoints():
    values, codes = unsigned_unit_qdq(torch.tensor([0.0, 0.5, 1.0]), bits=8)
    assert codes.tolist() == [0, 128, 255]
    assert float(values[0]) == 0.0
    assert float(values[-1]) == 1.0
```

- [ ] **Step 2: Run the primitive tests and verify the import failure**

Run: `pytest -q tests/test_propagation_fixed_point.py`

Expected: FAIL because `spn_quant.propagation.fixed_point` does not exist.

- [ ] **Step 3: Implement calibrated QDQ, residual normalization, and LUT softmax**

```python
Q13_FRACTION_BITS = 13
Q13_ONE = 1 << Q13_FRACTION_BITS


def symmetric_qdq(tensor, bits, maximum):
    qmax = (1 << (int(bits) - 1)) - 1
    scale = max(float(maximum) / qmax, torch.finfo(torch.float32).eps)
    codes = torch.clamp(torch.round(tensor / scale), -qmax, qmax).to(torch.int8)
    return codes.to(tensor.dtype) * scale, codes


def unsigned_unit_qdq(tensor, bits=8):
    qmax = (1 << int(bits)) - 1
    codes = torch.clamp(torch.round(tensor * qmax), 0, qmax).to(torch.uint8)
    return codes.to(tensor.dtype) / qmax, codes


def normalize_signed_q13(raw, denominator_floor, eps=1e-4):
    denominator = raw.abs().sum(dim=1, keepdim=True) + eps
    if denominator_floor:
        denominator = torch.maximum(denominator, torch.ones_like(denominator))
    normalized = raw / denominator
    codes32 = torch.round(normalized * Q13_ONE).to(torch.int32)
    center = Q13_ONE - codes32.sum(dim=1, keepdim=True)
    if int(codes32.abs().max()) > 32767 or int(center.abs().max()) > 32767:
        raise OverflowError("Q13 propagation coefficient exceeds INT16")
    return codes32.to(raw.dtype) / Q13_ONE, center.to(torch.int16), codes32.to(torch.int16)
```

Implement `softmax_logits_q13` with a 15-entry exponential LUT indexed by the
difference between each A4 code and the maximum code:

```python
def softmax_logits_q13(logits, dim, bits, maximum):
    _, raw_codes = symmetric_qdq(logits, bits, maximum)
    qmax = (1 << (int(bits) - 1)) - 1
    scale = max(float(maximum) / qmax, torch.finfo(torch.float32).eps)
    differences = raw_codes.to(torch.int16) - raw_codes.max(
        dim=dim, keepdim=True).values.to(torch.int16)
    exp_one = 1 << 20
    lut = torch.round(torch.exp(
        torch.arange(-2 * qmax, 1, device=logits.device,
                     dtype=torch.float32) * scale) * exp_one)
    lut = torch.clamp(lut, min=1, max=exp_one).to(torch.int32)
    weights = lut[(differences + 2 * qmax).long()]
    weight_sum = weights.sum(dim=dim, keepdim=True, dtype=torch.int32)
    reciprocal_q30 = torch.round(
        float(1 << 30) / weight_sum.to(torch.float64)).to(torch.int64)
    codes32 = ((weights.to(torch.int64) * reciprocal_q30 * Q13_ONE +
                (1 << 29)) >> 30).to(torch.int32)
    residual = Q13_ONE - codes32.sum(dim=dim, keepdim=True)
    winner = weights.argmax(dim=dim, keepdim=True)
    codes32.scatter_add_(dim, winner, residual)
    if int(codes32.max()) > 32767 or int(codes32.min()) < 0:
        raise OverflowError("Q13 softmax coefficient exceeds INT16")
    codes = codes32.to(torch.int16)
    return codes.to(logits.dtype) / Q13_ONE, codes
```

- [ ] **Step 4: Run primitive tests**

Run: `pytest -q tests/test_propagation_fixed_point.py`

Expected: PASS.

- [ ] **Step 5: Commit fixed-point primitives**

```bash
git add spn_quant/propagation tests/test_propagation_fixed_point.py
git commit -m "feat: add fixed-point SPN propagation primitives"
```

### Task 2: Assign Propagation Outputs To One Quantization Owner

**Files:**
- Modify: `scripts/hardware_aligned_quantization.py:287`
- Modify: `tests/test_hardware_aligned_quantization.py`

- [ ] **Step 1: Write a failing externally-owned-output test**

```python
def test_externally_owned_output_keeps_weight_and_input_qdq_only(self):
    model = nn.Sequential(nn.Conv2d(1, 3, 1)).eval()
    instrumentor = haq.HardwareAlignedInstrumentor(
        model, lambda name, module: "propagation_head",
        externally_owned_outputs={"0"})
    assert ("0", "input") in instrumentor.observers
    assert ("0", "output") not in instrumentor.observers
    assert instrumentor.externally_owned_outputs() == ["0"]
    instrumentor.close()
```

- [ ] **Step 2: Run the focused test and verify the constructor failure**

Run: `pytest -q tests/test_hardware_aligned_quantization.py -k externally_owned`

Expected: FAIL because the constructor has no `externally_owned_outputs` argument.

- [ ] **Step 3: Add the ownership contract**

```python
def __init__(self, model, group_fn, fused_relu_producers=None,
             fuse_layernorm=True, externally_owned_outputs=None):
    self._externally_owned_outputs = set(externally_owned_outputs or ())
    unknown = self._externally_owned_outputs - set(dict(model.named_modules()))
    if unknown:
        raise ValueError("unknown externally owned outputs: %s" % sorted(unknown))
```

Treat these names like skipped Conv/Linear outputs while retaining input,
weight, and INT32 bias quantization. Add:

```python
def externally_owned_outputs(self):
    return sorted(self._externally_owned_outputs)
```

Include the ownership list in `metadata()` and the hardware manifest.

- [ ] **Step 4: Run hardware quantizer tests**

Run: `pytest -q tests/test_hardware_aligned_quantization.py`

Expected: PASS.

- [ ] **Step 5: Commit the boundary contract**

```bash
git add scripts/hardware_aligned_quantization.py tests/test_hardware_aligned_quantization.py
git commit -m "feat: support SPN-owned projection outputs"
```

### Task 3: Calibrated Propagation Controller And Diagnostics

**Files:**
- Create: `spn_quant/propagation/controller.py`
- Modify: `spn_quant/propagation/__init__.py`
- Modify: `tests/test_propagation_fixed_point.py`

- [ ] **Step 1: Write failing lifecycle and metric tests**

```python
from spn_quant.propagation.controller import PropagationQuantConfig, PropagationQuantController


def test_controller_requires_calibration_and_records_constraint_metrics():
    controller = PropagationQuantController()
    controller.observe()
    controller.observe_signal("affinity_raw", torch.tensor([-2.0, 1.0]))
    controller.observe_signal("offset", torch.tensor([-0.5, 0.75]))
    controller.freeze()
    controller.configure(PropagationQuantConfig(
        affinity_bits=4, confidence_bits=8, offset_bits=4, state_bits=4))
    _, center, codes = controller.signed_affinity(
        torch.tensor([[[[-2.0]], [[1.0]]]]), denominator_floor=False)
    controller.record_constraints(codes, center, iteration=0)
    row = controller.statistics()[0]
    assert row["coefficient_sum_max_error"] == 0.0
    assert row["contraction_violation_rate"] == 0.0
```

- [ ] **Step 2: Run and verify the missing controller failure**

Run: `pytest -q tests/test_propagation_fixed_point.py -k controller`

Expected: FAIL because the controller module does not exist.

- [ ] **Step 3: Implement lifecycle, calibrated maxima, and statistics**

Define immutable configuration fields:

```python
@dataclass(frozen=True)
class PropagationQuantConfig:
    affinity_bits: int = 4
    confidence_bits: int = 8
    offset_bits: int = 4
    state_bits: int = 4
    coefficient_fraction_bits: int = 13
```

The controller modes are `bypass`, `observe`, `quantize`, and `capture`.
Calibration stores finite maximum absolute values for `affinity_raw`, `offset`,
and `state`; confidence is fixed to `[0, 1]`. Reject configure-before-freeze,
non-finite calibration ranges, unsupported bit widths, and fraction bits other
than 13. Store one metrics row per signal and iteration, including zeroed,
saturation, coefficient-sum error, contraction violation, anchor error, and
NaN/Inf ratios.

- [ ] **Step 4: Run controller and primitive tests**

Run: `pytest -q tests/test_propagation_fixed_point.py`

Expected: PASS.

- [ ] **Step 5: Commit the controller**

```bash
git add spn_quant/propagation tests/test_propagation_fixed_point.py
git commit -m "feat: add calibrated propagation quantization controller"
```

### Task 4: CSPN Propagation-Aware Adapter

**Files:**
- Create: `spn_quant/propagation/adapters.py`
- Modify: `spn_quant/propagation/__init__.py`
- Create: `tests/test_propagation_aware_adapters.py`

- [ ] **Step 1: Write failing CSPN identity and anchor tests**

```python
def test_cspn_adapter_quantizes_neighbors_before_normalization():
    module = ToyCSPN(prop_time=2)
    adapter = CSPNPropagationAdapter(module)
    calibrate_adapter(adapter, module)
    adapter.configure(PropagationQuantConfig(state_bits=4))
    output = module(GUIDANCE, INITIAL, SPARSE)
    metrics = adapter.statistics()
    assert max(row["coefficient_sum_max_error"] for row in metrics) == 0.0
    assert torch.equal(output[SPARSE > 0], SPARSE[SPARSE > 0])


def test_cspn_center_is_derived_from_quantized_neighbor_codes():
    adapter = CSPNPropagationAdapter(ToyCSPN(prop_time=1))
    neighbor_codes, center_code = adapter.normalized_codes(NEGATIVE_GUIDANCE)
    assert torch.equal(center_code + neighbor_codes.sum(1, keepdim=True),
                       torch.full_like(center_code, Q13_ONE))
```

- [ ] **Step 2: Run and verify the missing adapter failure**

Run: `pytest -q tests/test_propagation_aware_adapters.py -k cspn`

Expected: FAIL because `CSPNPropagationAdapter` does not exist.

- [ ] **Step 3: Implement CSPN propagation replacement**

Patch only `post_process_layer.forward`. In observe mode, execute the original
path and collect raw guidance plus each state. In quantize mode:

```python
raw_q = controller.quantize_affinity(guidance)
neighbor, center_code, neighbor_code = controller.signed_affinity(
    raw_q, denominator_floor=False)
center = center_code.to(guidance.dtype) / Q13_ONE
for iteration in range(1, module.prop_time + 1):
    neighbor_sum = propagate_neighbors(state, neighbor)
    state = neighbor_sum + center * initial
    state = controller.quantize_state(state, iteration)
    state = torch.where(sparse_mask, sparse_depth, state)
    controller.record_anchor(state, sparse_depth, sparse_mask, iteration)
```

Reuse the official padding and sum modules, preserve `norm_type`, and make
`disable()` an exact original-forward bypass. `close()` restores the original
method.

- [ ] **Step 4: Run adapter tests**

Run: `pytest -q tests/test_propagation_aware_adapters.py -k cspn`

Expected: PASS.

- [ ] **Step 5: Commit CSPN support**

```bash
git add spn_quant/propagation tests/test_propagation_aware_adapters.py
git commit -m "feat: preserve CSPN propagation invariants under W4A4"
```

### Task 5: NLSPN And CompletionFormer Adapters

**Files:**
- Modify: `spn_quant/propagation/adapters.py`
- Modify: `tests/test_propagation_aware_adapters.py`

- [ ] **Step 1: Write failing model-rule and center-residual tests**

```python
@pytest.mark.parametrize("affinity,denominator_floor", [
    ("AS", False), ("ASS", True), ("TC", False), ("TGASS", True),
])
def test_nlspn_adapter_preserves_official_normalization_rule(
        affinity, denominator_floor):
    module = ToyNLSPNModule(affinity=affinity, preserve_input=True)
    adapter = NLSPNPropagationAdapter(module)
    calibrate_adapter(adapter, module)
    adapter.configure(PropagationQuantConfig(offset_bits=4, state_bits=4))
    pred, states, offset, coefficients, scale = module(
        INITIAL, GUIDANCE, CONFIDENCE, SPARSE)
    codes = adapter.last_coefficient_codes()
    assert torch.all(codes.sum(1) == Q13_ONE)
    anchor_rows = [row for row in adapter.statistics()
                   if row["signal"] == "anchor_injection"]
    assert anchor_rows
    assert max(row["anchor_max_error"] for row in anchor_rows) == 0.0
    assert adapter.denominator_floor is denominator_floor
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest -q tests/test_propagation_aware_adapters.py -k nlspn`

Expected: FAIL because `NLSPNPropagationAdapter` does not exist.

- [ ] **Step 3: Implement the shared NLSPN-family adapter**

Support both official modules by their runtime interface rather than import
path. Reproduce the official order:

```python
offset_raw, affinity_raw = split_projection(module.conv_offset_aff(guidance))
offset = controller.quantize_offset(offset_raw)
affinity_raw = apply_official_affinity_transform(module, affinity_raw)
confidence_q, confidence_codes = controller.quantize_confidence(confidence)
affinity_raw = apply_official_sampled_confidence(
    module, affinity_raw, offset, confidence_q)
affinity_q = controller.quantize_affinity(affinity_raw)
neighbor, center_code, neighbor_code = controller.signed_affinity(
    affinity_q, denominator_floor=module.affinity in ("ASS", "TGASS"))
coefficients = insert_center(neighbor, center_code, module.idx_ref)
```

For `TC`, retain the official tanh and scale semantics but do not divide by the
AS denominator. For hard `preserve_input`, restore sparse anchors after state
requantization and before the next deformable propagation. Keep the official
CUDA `_propagate_once` implementation.

- [ ] **Step 4: Run NLSPN-family tests**

Run: `pytest -q tests/test_propagation_aware_adapters.py -k 'nlspn or completionformer'`

Expected: PASS.

- [ ] **Step 5: Commit NLSPN-family support**

```bash
git add spn_quant/propagation/adapters.py tests/test_propagation_aware_adapters.py
git commit -m "feat: add invariant-aware NLSPN propagation quantization"
```

### Task 6: DySPN Adapter And LUT Softmax

**Files:**
- Modify: `spn_quant/propagation/adapters.py`
- Modify: `tests/test_propagation_aware_adapters.py`

- [ ] **Step 1: Write failing DySPN softmax and confidence tests**

```python
def test_dyspn_adapter_uses_exact_sum_softmax_and_unsigned_a8_confidence():
    module = ToyDySPN(iteration=3, num=3)
    adapter = DySPNPropagationAdapter(module)
    calibrate_adapter(adapter, module)
    adapter.configure(PropagationQuantConfig(offset_bits=4, state_bits=4))
    result = module(INITIAL, GUIDANCE, SPARSE, CONFIDENCE_LOGITS)
    codes = adapter.last_coefficient_codes()
    assert torch.all(codes >= 0)
    assert torch.all(codes.sum(2) == Q13_ONE)
    assert adapter.confidence_code_range() == (0, 255)
    assert len(result["list_feat"]) == 3
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest -q tests/test_propagation_aware_adapters.py -k dyspn`

Expected: FAIL because `DySPNPropagationAdapter` does not exist.

- [ ] **Step 3: Implement DySPN ordering**

Split `conv_offset_aff` into offset and per-iteration logits. Quantize offsets
with the configured A4/A8 scale, quantize logits to A4, apply Q13 LUT softmax
over neighbors, and quantize `sigmoid(confidence_logits)` as unsigned A8.
For every iteration, use the official grid and `grid_sample`, apply Q13
affinity values, blend sparse depth with A8 confidence, then quantize state.
Record coefficient, confidence, offset, state, and anchor metrics.

- [ ] **Step 4: Run all propagation adapter tests**

Run: `pytest -q tests/test_propagation_aware_adapters.py`

Expected: PASS.

- [ ] **Step 5: Commit DySPN support**

```bash
git add spn_quant/propagation/adapters.py tests/test_propagation_aware_adapters.py
git commit -m "feat: add propagation-aware DySPN quantization"
```

### Task 7: Runner Configurations, Metrics, And Prediction Export

**Files:**
- Modify: `scripts/run_nyu_rtn_quantization.py`
- Modify: `tests/test_run_nyu_rtn_quantization.py`

- [ ] **Step 1: Write failing configuration and table tests**

```python
def test_propagation_backend_has_cumulative_ablation_matrix(self):
    configs = runner.build_propagation_configurations(["encoder", "propagation_head"])
    assert [row["name"] for row in configs] == [
        "FP32", "PA_Generic_W4A4", "PA_Constraint",
        "PA_OffsetA8", "PA_StateA8", "PA_W8A8",
    ]
    assert configs[2]["propagation"]["confidence_bits"] == 8
    assert configs[2]["propagation"]["offset_bits"] == 4
    assert configs[3]["propagation"]["offset_bits"] == 8
    assert configs[4]["propagation"]["state_bits"] == 8


def test_persist_tables_writes_propagation_metrics(tmp_path):
    runner.persist_tables(tmp_path, [], [], [], [], [], [{
        "model": "cspn", "config": "PA_Constraint", "signal": "affinity",
        "iteration": 0, "coefficient_sum_max_error": 0.0,
    }])
    assert (tmp_path / "propagation_quantization_metrics.csv").exists()
```

- [ ] **Step 2: Run and verify missing backend failures**

Run: `pytest -q tests/test_run_nyu_rtn_quantization.py -k propagation`

Expected: FAIL because the builder and persistence argument do not exist.

- [ ] **Step 3: Add the propagation backend and owned projection names**

Add `propagation` to `--quant-backend`. Build exactly the six configurations
from the test. Map externally owned projection outputs as follows:

```python
def propagation_projection_outputs(model_name, model):
    if model_name == "cspn":
        return {"gud_up_proj_layer6.conv1"}
    if model_name == "dyspn":
        return {"dyspn_%d_%d.conv_offset_aff" % (model.iteration, model.num_sample)}
    return {"prop_layer.conv_offset_aff"}
```

Install the propagation-aware adapter for this backend and retain the current
state-only adapter for other backends. During calibration call both the CNN
instrumentor and propagation controller in observe mode. During each ablation
configure the controller with `PropagationQuantConfig` and export its rows to
`propagation_quantization_metrics.csv`.

- [ ] **Step 4: Add FP32-relative per-step metrics**

Use captured FP32 states already stored in each record. For every quantized
state, compute RMSE, MAE, SQNR, anchor MAE, step growth ratio, and
final-to-first ratio. Include model, config, sample index, signal, and iteration
keys so partial reruns can replace selected configurations safely.

- [ ] **Step 5: Run runner tests**

Run: `pytest -q tests/test_run_nyu_rtn_quantization.py tests/test_nyu_quantization_analysis.py`

Expected: PASS.

- [ ] **Step 6: Commit runner integration**

```bash
git add scripts/run_nyu_rtn_quantization.py tests/test_run_nyu_rtn_quantization.py
git commit -m "feat: run propagation-aware NYU quantization ablations"
```

### Task 8: Static Prediction And Propagation Visualizations

**Files:**
- Create: `scripts/plot_propagation_aware_quantization.py`
- Create: `tests/test_plot_propagation_aware_quantization.py`

- [ ] **Step 1: Write failing selection and rendering tests**

```python
def test_select_details_contains_fixed_random_and_worst_samples():
    rows = make_metric_rows(64)
    selected = plotting.select_detail_samples(rows, seed=20260804, random_count=4, worst_count=4)
    assert len(selected) == 8
    assert {row["reason"] for row in selected} == {"random", "worst_generic_w4a4"}


def test_render_writes_contact_detail_and_step_figures(tmp_path):
    fixture = build_prediction_fixture(tmp_path, sample_count=4)
    outputs = plotting.render_model(fixture, tmp_path / "figures", "cspn")
    assert {path.name for path in outputs} == {
        "cspn_prediction_contact_sheet.png",
        "cspn_prediction_details.png",
        "cspn_propagation_step_error.png",
        "cspn_constraint_violations.png",
    }
```

- [ ] **Step 2: Run and verify the module import failure**

Run: `pytest -q tests/test_plot_propagation_aware_quantization.py`

Expected: FAIL because the plotting module does not exist.

- [ ] **Step 3: Implement deterministic static figures**

Use Arial with Liberation Sans fallback, no figure titles, unrotated labels,
grid lines behind plotted data, and bars/lines at a higher z-order. Produce:

```python
CONTACT_CONFIGS = ("FP32", "PA_Generic_W4A4", "PA_StateA8")
DETAIL_COLUMNS = (
    "Sparse", "GT", "FP32", "Generic W4A4", "PA W4A4",
    "Generic abs error", "PA abs error",
)
```

Use one `[0, 10] m` depth scale and one robust shared error scale per detailed
sample. Validate identical sample sets before rendering. Write
`selected_visual_samples.csv` beside the figures.

- [ ] **Step 4: Run plot tests**

Run: `pytest -q tests/test_plot_propagation_aware_quantization.py`

Expected: PASS.

- [ ] **Step 5: Commit visualization support**

```bash
git add scripts/plot_propagation_aware_quantization.py tests/test_plot_propagation_aware_quantization.py
git commit -m "feat: visualize propagation-aware quantization errors"
```

### Task 9: Full Verification And 64-Sample NYU Evaluation

**Files:**
- Modify: `README.md`
- Output: `profile_logs/nyu_propagation_aware_quantization/`

- [ ] **Step 1: Run focused and full test suites**

Run:

```bash
pytest -q tests/test_propagation_fixed_point.py \
  tests/test_propagation_aware_adapters.py \
  tests/test_hardware_aligned_quantization.py \
  tests/test_run_nyu_rtn_quantization.py \
  tests/test_plot_propagation_aware_quantization.py
pytest -q
```

Expected: all focused tests and the complete repository suite PASS.

- [ ] **Step 2: Verify CUDA extensions and checkpoints before long runs**

Run the repository environment checker for all four official models, then run
one calibration and one evaluation sample per model with `PA_Constraint`.

Expected: all model imports, checkpoints, NYU files, deformable-convolution
extensions, and CUDA forwards succeed with finite predictions.

- [ ] **Step 3: Run the fixed 64-sample experiment for each model**

Use the existing converged run directories, checkpoints, and fixed formal
64-sample list:

```bash
for run_dir in \
  /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24 \
  /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/dyspn_iter6 \
  /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/nlspn_iter18 \
  /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/completionformer_iter18
do
  python scripts/run_nyu_rtn_quantization.py \
    --run-dir "$run_dir" --checkpoint best.pt \
    --sample-metrics profile_logs/completionformer_w4a4_error_source_64/completionformer/sample_metrics.csv \
    --quant-backend propagation --calibration-samples 128 \
    --max-eval-samples 64 \
    --export-prediction-configs FP32 PA_Generic_W4A4 PA_Constraint PA_OffsetA8 PA_StateA8 PA_W8A8 \
    --out-dir profile_logs/nyu_propagation_aware_quantization || exit 1
done
```

The runner deduplicates the formal sample rows into the same 64 indices used by
the existing CompletionFormer analysis. The loop runs CSPN, DySPN, NLSPN, and
CompletionFormer sequentially so GPU memory and extension state are released
between models.

- [ ] **Step 4: Generate figures and validate artifacts**

Run:

```bash
python scripts/plot_propagation_aware_quantization.py \
  --root profile_logs/nyu_propagation_aware_quantization
```

Expected per model: 64 prediction NPZ files for every exported configuration,
finite aggregate metrics, propagation metrics for every iteration, and four
PNG figures. Check that all configurations share exactly the same 64 indices.

- [ ] **Step 5: Document commands and measured conclusions**

Add a README section describing the propagation backend, mixed-precision
contract, exact run command, artifact paths, and measured conclusions. State
whether each model's error is dominated by affinity constraints, offsets, or
state requantization based on the cumulative ablations; do not infer a cause
when adjacent configurations do not show a measurable difference.

- [ ] **Step 6: Commit verified documentation**

```bash
git add README.md
git commit -m "docs: report propagation-aware W4A4 evaluation"
```
