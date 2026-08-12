# CSPN Activation Resolution Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run a strict CSPN W4A4 study that attributes propagation, weight, activation, and interaction error, then tests Group-A4, residual-aware integer merge, and calibration-learned activation scales.

**Architecture:** Extend the current hardware-aligned instrumentor instead of adding a second quantization path. A focused activation-resolution module owns exact error partitions and per-channel summaries; the CSPN runner owns fixed dataset indices, configuration selection, block attribution, and artifact writing. Existing semantic adapters and propagation controllers retain ownership of guidance and SPN arithmetic.

**Tech Stack:** Python 3, PyTorch, NumPy, existing `spn_quant` semantic/propagation APIs, unittest/pytest, NYU Depth V2, CUDA.

---

### Task 1: Exact Activation Resolution Metrics

**Files:**
- Create: `spn_quant/activation_resolution.py`
- Create: `tests/test_activation_resolution.py`

- [ ] **Step 1: Write failing tests for mutually exclusive error attribution**

```python
def test_error_partition_conserves_total_energy():
    quantizer = SymmetricActivationQuantizer(4, 7.0)
    reference = torch.tensor([-9.0, -0.2, 0.0, 0.4, 9.0])
    quantized, codes = quantizer.quantize_with_codes(reference)
    accumulator = ActivationResolutionAccumulator(channel_dim=0, capacity=32)
    accumulator.update(reference, quantized, codes, quantizer)
    row = accumulator.tensor_summary()
    partition = (row["zero_collapse_error_energy"] +
                 row["rounding_error_energy"] +
                 row["clipping_error_energy"])
    assert partition == pytest.approx(row["total_error_energy"])
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest -q tests/test_activation_resolution.py`

Expected: FAIL because `spn_quant.activation_resolution` does not exist.

- [ ] **Step 3: Implement exact tensor/per-channel accumulation**

Implement three public interfaces. `BoundedChannelSampler(channels, capacity)`
accepts channel-first samples through `update(values)` and returns one
percentile vector per requested probability through `percentiles(values)`.
`ActivationResolutionAccumulator(channel_dim, capacity)` accepts one real QDQ
event through `update(reference, quantized, codes, quantizer)` and exposes
`tensor_summary()` plus `channel_summaries()`. `ActivationResolutionRecorder`
implements the instrumentor recorder protocol and adds required site metadata
and the declared `split` field to `tensor_rows()` and `channel_rows()`.

Use `zero_mask = (reference != 0) & (codes == 0)`. Define clipping as an out-of-range unrounded code excluding `zero_mask`, then assign all remaining elements to rounding. Accumulate exact counts and float64 squared-error energies. Compute `effective_code_count` as `exp(H(q))` from accumulated integer-code counts.

- [ ] **Step 4: Add tests for natural zeros, per-channel attribution, percentiles, and effective code count**

```python
def test_new_zero_rate_excludes_reference_zeros():
    quantizer = SymmetricActivationQuantizer(4, 7.0)
    reference = torch.tensor([0.0, 0.2, 1.0])
    quantized, codes = quantizer.quantize_with_codes(reference)
    accumulator = ActivationResolutionAccumulator(0, 16)
    accumulator.update(reference, quantized, codes, quantizer)
    assert accumulator.tensor_summary()["new_zero_rate"] == 0.5

def test_channel_error_shares_sum_to_one():
    quantizer = SymmetricActivationQuantizer(4, 7.0)
    reference = torch.tensor([[[[0.2]]], [[[1.4]]]])
    quantized, codes = quantizer.quantize_with_codes(reference)
    accumulator = ActivationResolutionAccumulator(1, 16)
    accumulator.update(reference, quantized, codes, quantizer)
    assert sum(row["error_energy_share"]
               for row in accumulator.channel_summaries()) == pytest.approx(1.0)

def test_bounded_sampler_is_deterministic():
    first = BoundedChannelSampler(2, 4)
    second = BoundedChannelSampler(2, 4)
    values = torch.arange(20, dtype=torch.float32).reshape(2, 10)
    first.update(values)
    second.update(values)
    torch.testing.assert_close(first.percentiles((0.75, 0.99)),
                               second.percentiles((0.75, 0.99)))

def test_effective_code_count_uses_code_entropy():
    quantizer = SymmetricActivationQuantizer(4, 1.0)
    reference = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    quantized, codes = quantizer.quantize_with_codes(reference)
    accumulator = ActivationResolutionAccumulator(0, 16)
    accumulator.update(reference, quantized, codes, quantizer)
    assert accumulator.tensor_summary()["effective_code_count"] == pytest.approx(2.0)

def test_recorder_keeps_evaluation_split():
    recorder = ActivationResolutionRecorder("evaluation", 16)
    quantizer = SymmetricActivationQuantizer(4, 1.0)
    reference = torch.ones(1, 1, 1, 1)
    quantized, codes = quantizer.quantize_with_codes(reference)
    recorder.record("conv", "input", 0, "encoder", reference,
                    quantized, codes, quantizer, 1)
    assert recorder.tensor_rows()[0]["split"] == "evaluation"
```

- [ ] **Step 5: Run tests and commit**

Run: `pytest -q tests/test_activation_resolution.py tests/test_hardware_aligned_quantization.py`

Expected: PASS.

Commit: `git commit -m "feat: add activation resolution diagnostics"`

### Task 2: Generic Tensor, Group, and Channel Activation QDQ

**Files:**
- Modify: `scripts/hardware_aligned_quantization.py`
- Modify: `spn_quant/specs.py`
- Modify: `tests/test_hardware_aligned_quantization.py`
- Modify: `tests/test_quant_specs.py`

- [ ] **Step 1: Write failing tests for Group-A4 scales**

```python
def test_group_a4_uses_one_scale_per_contiguous_channel_group():
    observer = ChannelMinMaxObserver(channel_dim=1)
    observer.update(torch.tensor([[[[1.0]], [[2.0]], [[10.0]], [[20.0]]]]))
    quantizer = observer.quantizer_for(
        QuantSpec(bits=4, scheme="affine", granularity="group",
                  axis=1, group_size=2, signed=False, preserve_zero=True))
    assert quantizer.scale_count == 2
    assert quantizer.scale.tolist() == pytest.approx([2.0 / 15.0, 20.0 / 15.0])
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_hardware_aligned_quantization.py -k group_a4`

Expected: FAIL because `quantizer_for` and generic grouped QDQ are absent.

- [ ] **Step 3: Implement spec-driven activation observers and quantizers**

Add `ChannelMinMaxObserver.quantizer_for(spec)` and `GroupedActivationQuantizer`. The grouped quantizer stores one scale per group, expands scales only for arithmetic, exposes `scale_for(tensor)`, `quantize_with_codes(tensor)`, `scale_count`, and a manifest. Tensor and channel specs must produce the current strict RTN and per-channel behavior exactly.

- [ ] **Step 4: Add instrumentor activation-spec configuration**

Add an explicit `activation_specs` mapping to the component-aware configuration path. Every quantized activation key must resolve to a declared `QuantSpec`; unknown keys and non-divisible groups raise directly. ReLU sites require unsigned affine specs and ordinary signed sites require signed symmetric specs.

- [ ] **Step 5: Verify baseline compatibility and granularity ownership**

Run: `pytest -q tests/test_quant_specs.py tests/test_hardware_aligned_quantization.py tests/test_activation_histograms.py`

Expected: PASS, including existing tensor/per-channel tests.

Commit: `git commit -m "feat: add group activation quantization"`

### Task 3: Separate Propagation, Weight, and Activation Attribution

**Files:**
- Modify: `scripts/hardware_aligned_quantization.py`
- Modify: `tests/test_hardware_aligned_quantization.py`

- [ ] **Step 1: Write failing component-ownership tests**

```python
def test_weight_only_keeps_w4_weights_and_bypasses_activation_qdq():
    model, instrumentor, sample = calibrated_conv_instrumentor()
    reference_weight = model.conv.weight.detach().clone()
    instrumentor.configure_components(4, 4, {"encoder"}, set(), {}, False)
    assert not torch.equal(model.conv.weight, reference_weight)
    assert instrumentor.quantizers == {}

def test_activation_only_restores_fp_weights_and_applies_a4_qdq():
    model, instrumentor, sample = calibrated_conv_instrumentor()
    reference_weight = model.conv.weight.detach().clone()
    specs = instrumentor.tensor_activation_specs(4)
    instrumentor.configure_components(4, 4, set(), {"encoder"}, specs, False)
    torch.testing.assert_close(model.conv.weight, reference_weight)
    assert not torch.equal(model(sample), model.conv(sample))

def test_component_configuration_rejects_undeclared_groups():
    model, instrumentor, sample = calibrated_conv_instrumentor()
    with pytest.raises(ValueError, match="unknown activation groups"):
        instrumentor.configure_components(4, 4, set(), {"missing"}, {}, False)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_hardware_aligned_quantization.py -k component`

Expected: FAIL because weights and activations currently share `enabled_groups`.

- [ ] **Step 3: Add explicit component configuration**

Implement:

`configure_components(w_bits, a_bits, weight_groups, activation_groups,
activation_specs, quantize_bias)` is an explicit public method; all six
arguments are required.

The existing `configure(...)` delegates with the same group set for both components. `configure_components` quantizes weights only for `weight_groups`, builds/runs activation QDQ only for `activation_groups`, and restores FP weights for activation-only runs. Bias remains FP32 in this study.

- [ ] **Step 4: Run regression tests and commit**

Run: `pytest -q tests/test_hardware_aligned_quantization.py tests/test_rtn_quantization.py tests/test_strict_w4a4_fp4_evaluation.py`

Expected: PASS.

Commit: `git commit -m "feat: separate weight and activation quantization"`

### Task 4: Residual-Aware INT32 Add

**Files:**
- Modify: `spn_quant/integer_ops.py`
- Modify: `spn_quant/merge.py`
- Modify: `spn_quant/adapters/cspn.py`
- Modify: `tests/test_integer_ops.py`
- Modify: `tests/test_hardware_merge_adapters.py`
- Modify: `tests/test_model_semantic_adapters.py`

- [ ] **Step 1: Write failing integer merge tests**

```python
def test_independent_branch_codes_requantize_before_int32_add():
    left_codes = torch.tensor([1, 2], dtype=torch.int32)
    right_codes = torch.tensor([10, 20], dtype=torch.int32)
    output = add_requantized_int32(
        ((left_codes, 0.1), (right_codes, 1.0)),
        output_scale=0.25, qmin=-127, qmax=127)
    assert output.dtype == torch.int32
    assert output.tolist() == [40, 81]
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_integer_ops.py tests/test_hardware_merge_adapters.py -k requant`

Expected: FAIL because integer branch merge is absent.

- [ ] **Step 3: Implement fixed-point branch requantization and merge policy**

Add `add_requantized_int32` using the existing Q31 requantizer, retaining branch codes and scales independently. Extend `MergeSiteController` with an explicit `residual` policy: branch bit widths `(4, 8)`, independent branch observers, A8 output observer, INT32 add, and output dequantization. Do not add raw integer codes with unequal scales.

- [ ] **Step 4: Expose selected CSPN residual sites**

Allow `CSPNStructuralMergeAdapter` to receive an indexed mapping from merge site to residual policy. Unselected sites retain the declared baseline policy; unknown selected sites raise after calibration.

- [ ] **Step 5: Run tests and commit**

Run: `pytest -q tests/test_integer_ops.py tests/test_hardware_merge_adapters.py tests/test_model_semantic_adapters.py tests/test_propagation_fixed_point.py`

Expected: PASS.

Commit: `git commit -m "feat: add residual-aware integer merge"`

### Task 5: CSPN Attribution and Group Sweep Runner

**Files:**
- Create: `scripts/run_nyu_cspn_activation_resolution.py`
- Create: `tests/test_run_nyu_cspn_activation_resolution.py`

- [ ] **Step 1: Write failing configuration and selection tests**

```python
def test_attribution_configs_share_propagation_contract():
    configs = build_attribution_configurations()
    quantized = [row for row in configs if row["name"] != "FP32"]
    assert {tuple(sorted(row["propagation"].items())) for row in quantized} == {
        tuple(sorted(PROPAGATION_A8_Q13.items()))}

def test_interaction_delta_uses_pa_only_baseline():
    values = {"PA_ONLY": 1.0, "W4_ONLY": 1.2,
              "A4_ONLY": 1.5, "W4A4_RTN": 2.0}
    assert attribution_interaction(values) == pytest.approx(0.3)

def test_sensitive_sites_are_selected_from_calibration_only():
    rows = [{"split": "calibration", "site": "a", "error": 2.0},
            {"split": "evaluation", "site": "b", "error": 100.0}]
    assert select_candidate_sites(rows, 1) == ("a",)

def test_group_selection_uses_block_mse_then_sqnr():
    rows = [{"name": "g16", "block_mse": 1.0, "block_sqnr": 3.0},
            {"name": "g8", "block_mse": 1.0, "block_sqnr": 4.0}]
    assert select_calibration_configuration(rows)["name"] == "g8"

def test_guidance_group_is_never_quantized():
    for config in build_attribution_configurations():
        assert "guidance" not in config["activation_groups"]
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_run_nyu_cspn_activation_resolution.py`

Expected: FAIL because the runner does not exist.

- [ ] **Step 3: Implement fixed experiment configurations**

Declare `FP32`, `PA_ONLY`, `W4_ONLY`, `A4_ONLY`, and `W4A4_RTN`, all with explicit weight groups, activation groups, activation specs, bias format, and propagation configuration. Define the CSPN block list from `conv1_1`, `layer1` through `layer4`, `gud_up_proj_layer1` through `gud_up_proj_layer5`, and `post_process_layer`.

- [ ] **Step 4: Implement calibration/evaluation collection and CSV output**

Reuse dataset/checkpoint/sample helpers from `run_nyu_rtn_quantization.py` and model loading from the strict CSPN rotation runner. Write the exact artifact layout in the design, include the required split column, and reject non-finite predictions. Candidate ranking and A4-to-A8 single-site interventions use calibration block outputs only.

- [ ] **Step 5: Implement global and selective Group-A4 sweep**

Generate Tensor, Group128/64/32/16/8, and Per-channel specs only where channel counts permit. Select by calibration block MSE then SQNR. Record activation element/site fractions and number of scales.

- [ ] **Step 6: Run runner unit tests and commit**

Run: `pytest -q tests/test_run_nyu_cspn_activation_resolution.py tests/test_run_nyu_cspn_rotation.py`

Expected: PASS.

Commit: `git commit -m "feat: add CSPN activation resolution runner"`

### Task 6: Calibration-Learned Activation Scales

**Files:**
- Modify: `spn_quant/scale_search.py`
- Modify: `scripts/run_nyu_cspn_activation_resolution.py`
- Modify: `tests/test_scale_search.py`
- Modify: `tests/test_run_nyu_cspn_activation_resolution.py`

- [ ] **Step 1: Write failing bounded scale-search tests**

```python
def test_scale_search_can_trade_bounded_clipping_for_lower_block_mse():
    rows = [{"split": "calibration", "factor": 1.0, "block_mse": 2.0,
             "clipping_error_ratio": 0.0},
            {"split": "calibration", "factor": 0.75, "block_mse": 1.0,
             "clipping_error_ratio": 0.2}]
    assert select_activation_scale(rows)["factor"] == 0.75

def test_scale_search_rejects_clipping_dominated_candidate():
    rows = [{"split": "calibration", "factor": 1.0, "block_mse": 2.0,
             "clipping_error_ratio": 0.0},
            {"split": "calibration", "factor": 0.5, "block_mse": 0.5,
             "clipping_error_ratio": 0.6}]
    assert select_activation_scale(rows)["factor"] == 1.0

def test_scale_selection_never_reads_evaluation_rows():
    rows = [{"split": "calibration", "factor": 1.0, "block_mse": 2.0,
             "clipping_error_ratio": 0.0},
            {"split": "evaluation", "factor": 0.5, "block_mse": 0.1,
             "clipping_error_ratio": 0.0}]
    assert select_activation_scale(rows)["factor"] == 1.0
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_scale_search.py tests/test_run_nyu_cspn_activation_resolution.py -k scale`

Expected: FAIL on the missing activation-scale selection contract.

- [ ] **Step 3: Add calibration-only scale candidate selection**

Use `CoordinateScaleSearch` with explicit factors `(1.0, 0.95, 0.9, 0.85, 0.75, 0.625, 0.5)` on selected tensor/group scales. The objective is selected block-output MSE. Persist every candidate's clipping energy ratio, new-zero rate, SQNR, objective, and selected flag; reject candidates whose clipping error exceeds half of total activation error.

- [ ] **Step 4: Run tests and commit**

Run: `pytest -q tests/test_scale_search.py tests/test_run_nyu_cspn_activation_resolution.py`

Expected: PASS.

Commit: `git commit -m "feat: add learned CSPN activation scales"`

### Task 7: Real NYU CUDA Evaluation and Result Report

**Files:**
- Create: `docs/2026-08-12-cspn-activation-resolution-results.md`
- Modify: `README.md`

- [ ] **Step 1: Run focused and full verification**

Run: `pytest -q tests/test_activation_resolution.py tests/test_hardware_aligned_quantization.py tests/test_integer_ops.py tests/test_hardware_merge_adapters.py tests/test_run_nyu_cspn_activation_resolution.py tests/test_scale_search.py`

Expected: PASS.

Run: `pytest -q`

Expected: all tests pass; the pre-existing whitespace change in `tests/test_qdrop_reconstruction.py` remains unstaged and untouched.

- [ ] **Step 2: Run strict CSPN experiment on CUDA**

Run:

```bash
python scripts/run_nyu_cspn_activation_resolution.py \
  --run-dir /workspace/SPN_Quantization/checkpoints/nyu_converged/cspn_iter24 \
  --checkpoint /workspace/SPN_Quantization/checkpoints/nyu_converged/cspn_iter24/best.pt \
  --indices /workspace/SPN_Quantization/profile_logs/nyu_random64/sample_indices.json \
  --output /workspace/SPN_Quantization/profile_logs/nyu_cspn_activation_resolution \
  --device cuda:0 \
  --seed 2026
```

Expected: fixed 128-sample calibration and 64-sample evaluation complete with no non-finite predictions.

- [ ] **Step 3: Audit artifacts and write result report**

Check that every declared configuration has 64 sample metric rows, calibration/evaluation activation rows are disjoint, attribution arithmetic closes, error partitions conserve energy, and prediction payloads contain GT/FP32/quantized outputs. Summarize the dominant sites/channels, Group-A4 Pareto, residual result, learned-scale result, and whether RMSE beats `0.426802 m`.

- [ ] **Step 4: Commit results and documentation**

Commit: `git commit -m "docs: report CSPN activation resolution study"`
