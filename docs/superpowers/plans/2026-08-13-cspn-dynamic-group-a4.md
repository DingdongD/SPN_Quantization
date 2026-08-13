# CSPN Dynamic Group-A4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add true per-sample online Group-8 activation QDQ and evaluate it against the existing static CSPN W4A4 path on 64 fixed NYU samples.

**Architecture:** Dedicated dynamic tensor and grouped quantizers compute ranges from each current activation without reading observer extrema. `QuantSpec.dynamic` selects this path inside `HardwareAlignedInstrumentor`; the existing static path remains unchanged. A focused CSPN runner reuses the official checkpoint, propagation adapter, metric recorder, and fixed evaluation subset.

**Tech Stack:** Python 3, PyTorch 2.7, CUDA, unittest/pytest, existing CSPN hardware-aligned quantization framework.

---

### Task 1: Dynamic activation quantization primitives

**Files:**
- Modify: `tests/test_hardware_aligned_quantization.py`
- Modify: `scripts/hardware_aligned_quantization.py`

- [ ] **Step 1: Write failing primitive tests**

Add tests that construct `DynamicTensorActivationQuantizer` and
`DynamicGroupedActivationQuantizer` directly:

```python
def test_dynamic_signed_group_a4_uses_independent_sample_ranges(self):
    quantizer = haq.DynamicGroupedActivationQuantizer(
        bits=4, channel_dim=1, group_size=2, channels=4,
        unsigned=False)
    values = torch.tensor([
        [-7.0, -1.0, -14.0, -2.0],
        [70.0, 10.0, 140.0, 20.0],
    ]).reshape(2, 4, 1, 1)

    quantized, codes = quantizer.quantize_with_codes(values)

    self.assertEqual((int(codes.min()), int(codes.max())), (-7, 7))
    torch.testing.assert_close(quantized[0], values[0])
    torch.testing.assert_close(quantized[1], values[1])
    self.assertEqual(tuple(quantizer.scale_for(values).shape), (2, 4, 1, 1))

def test_dynamic_unsigned_group_a4_preserves_zero_range(self):
    quantizer = haq.DynamicGroupedActivationQuantizer(
        bits=4, channel_dim=1, group_size=2, channels=4,
        unsigned=True)
    values = torch.tensor([[[[0.0]], [[0.0]], [[1.0]], [[15.0]]]])

    quantized, codes = quantizer.quantize_with_codes(values)

    torch.testing.assert_close(quantized[:, :2], torch.zeros(1, 2, 1, 1))
    self.assertEqual((int(codes.min()), int(codes.max())), (0, 15))

def test_dynamic_tensor_a4_uses_one_scale_per_sample(self):
    quantizer = haq.DynamicTensorActivationQuantizer(bits=4, unsigned=False)
    values = torch.tensor([[-7.0, 1.0], [-70.0, 10.0]])

    scale = quantizer.scale_for(values)

    torch.testing.assert_close(scale.reshape(-1), torch.tensor([1.0, 10.0]))

def test_dynamic_activation_rejects_nonfinite_input(self):
    quantizer = haq.DynamicTensorActivationQuantizer(bits=4, unsigned=False)

    with self.assertRaisesRegex(ValueError, "finite"):
        quantizer(torch.tensor([[float("inf")]]))
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest tests/test_hardware_aligned_quantization.py \
  -k 'dynamic_signed_group or dynamic_unsigned_group or dynamic_tensor_a4 or dynamic_activation_rejects' -q
```

Expected: FAIL because both dynamic quantizer classes are absent.

- [ ] **Step 3: Implement minimal dynamic quantizers**

Add `DynamicTensorActivationQuantizer` and
`DynamicGroupedActivationQuantizer` beside the static activation quantizers.
Both expose `format`, `unsigned`, `granularity`, `qmin`, `qmax`,
`quantize_with_codes`, `scale_for`, and `__call__`. The grouped implementation
resolves the channel axis, reshapes the current input to
`[N, groups, group_size, spatial_elements]`, reduces over the last two axes,
and broadcasts `[N, groups]` scales back to the input shape. Use scale 1 only
for an exactly zero range so zero remains exact. Reject non-finite tensors and
channel-shape changes.

Track online overhead only in `quantize_with_codes`:

```python
self.invocations += 1
self.runtime_scale_count += int(extent.numel())
self.reduction_elements += int(tensor.numel())
```

`scale_for` recomputes the mathematical scale without updating counters so
statistics collection cannot double-count overhead.

- [ ] **Step 4: Run primitive tests and static regression tests**

Run:

```bash
python -m pytest tests/test_hardware_aligned_quantization.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit primitive implementation**

```bash
git add scripts/hardware_aligned_quantization.py \
  tests/test_hardware_aligned_quantization.py
git commit -m "feat: add dynamic activation quantizers"
```

### Task 2: QuantSpec and instrumentor dynamic path

**Files:**
- Modify: `spn_quant/specs.py`
- Modify: `scripts/hardware_aligned_quantization.py`
- Modify: `scripts/run_nyu_cspn_activation_resolution.py`
- Modify: `tests/test_quant_specs.py`
- Modify: `tests/test_hardware_aligned_quantization.py`
- Modify: `tests/test_run_nyu_cspn_activation_resolution.py`

- [ ] **Step 1: Write failing QuantSpec and instrumentor tests**

Add a `QuantSpec.with_dynamic()` test:

```python
def test_with_dynamic_preserves_quantization_contract(self):
    source = QuantSpec.unsigned_group(4, axis=1, group_size=8)
    dynamic = source.with_dynamic()

    self.assertTrue(dynamic.dynamic)
    self.assertEqual(dynamic.granularity, "group")
    self.assertEqual(dynamic.group_size, 8)
    self.assertFalse(source.dynamic)
```

Add instrumentor tests proving that a dynamic spec builds a dynamic grouped
quantizer, ignores observer magnitude, rejects a static maximum override, and
leaves a matching static spec on `GroupedActivationQuantizer`:

```python
dynamic = QuantSpec.unsigned_group(4, axis=1, group_size=2).with_dynamic()
instrumentor.configure_components_with_ranges(
    4, 4, set(), {"encoder"}, {("0", "input"): dynamic},
    False, activation_maxima={})
self.assertIsInstance(
    instrumentor.quantizers[("0", "input")],
    haq.DynamicGroupedActivationQuantizer)
```

Add configuration tests that call `build_activation_specs(..., dynamic=True)`
and assert all returned ordinary specs are dynamic while the default call
returns static specs.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest tests/test_quant_specs.py \
  tests/test_hardware_aligned_quantization.py \
  tests/test_run_nyu_cspn_activation_resolution.py \
  -k dynamic -q
```

Expected: FAIL because `with_dynamic`, dynamic spec construction, and
instrumentor routing are absent.

- [ ] **Step 3: Implement QuantSpec routing**

Add:

```python
def with_dynamic(self, dynamic: bool = True) -> "QuantSpec":
    return replace(self, dynamic=bool(dynamic))
```

Update `ChannelMinMaxObserver.quantizer_for` so static MinMax validation remains
unchanged, while `spec.dynamic` rejects supplied `maximum`, returns
`DynamicTensorActivationQuantizer` for tensor granularity, returns
`DynamicGroupedActivationQuantizer` for group granularity, and rejects dynamic
channel granularity because it is outside this experiment. The dynamic
constructor must not read observer extrema to form a runtime range.

- [ ] **Step 4: Propagate the dynamic configuration flag**

Extend `build_activation_specs` with `dynamic: bool = False` and apply
`spec.with_dynamic(dynamic)` after granularity and bit promotion are resolved.
Extend `_configuration` with `dynamic: bool = False`, store the key directly,
and carry it through `_derived_configuration`. In `_configure_quantized` and
`_configuration_manifest`, pass `config["dynamic"]` to
`build_activation_specs`. Add `dynamic` to manifest rows.

Do not set dynamic specs for rotation boundaries. Guidance and propagation
ownership remain unchanged.

- [ ] **Step 5: Expose online overhead rows**

Add `HardwareAlignedInstrumentor.dynamic_activation_rows()` that returns one
row per dynamic ordinary/relu quantizer with module, kind, group, granularity,
group size, invocations, runtime scale count, and reduction elements. It must
return an empty list for static configurations.

- [ ] **Step 6: Run focused and full component tests**

Run:

```bash
python -m pytest tests/test_quant_specs.py \
  tests/test_hardware_aligned_quantization.py \
  tests/test_run_nyu_cspn_activation_resolution.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit integration**

```bash
git add spn_quant/specs.py scripts/hardware_aligned_quantization.py \
  scripts/run_nyu_cspn_activation_resolution.py tests/test_quant_specs.py \
  tests/test_hardware_aligned_quantization.py \
  tests/test_run_nyu_cspn_activation_resolution.py
git commit -m "feat: route dynamic activation specs"
```

### Task 3: Focused CSPN evaluation runner

**Files:**
- Create: `scripts/run_nyu_cspn_dynamic_group_a4.py`
- Create: `tests/test_run_nyu_cspn_dynamic_group_a4.py`

- [ ] **Step 1: Write failing runner contract tests**

Test the exact six-configuration order and contracts:

```python
self.assertEqual(
    tuple(config["name"] for config in runner.build_configurations()),
    ("FP32", "W4_ONLY", "A4_ONLY_G8_STATIC",
     "A4_ONLY_G8_DYNAMIC", "W4A4_G8_STATIC",
     "W4A4_G8_DYNAMIC"))
```

Assert static/dynamic pairs have equal weight groups, activation groups,
Group-8 granularity, propagation policy, and owner selection, differing only
in `dynamic`. Assert guidance is absent from ordinary groups and propagation is
`PROPAGATION_A8_Q13`. Test exact 64-index validation and exact prediction
coverage.

- [ ] **Step 2: Run runner tests and verify RED**

Run:

```bash
python -m pytest tests/test_run_nyu_cspn_dynamic_group_a4.py -q
```

Expected: FAIL because the runner module is absent.

- [ ] **Step 3: Implement the runner**

Follow `run_nyu_cspn_smoothquant_group.py` for official CSPN loading,
Conv-BN preparation, strict semantic ownership, random-128 calibration, fixed
64 evaluation indices, paired block capture, CSV writing, metadata, and
prediction payloads. Build only the six declared configurations. Use
`sample_capacity=256`, seed `20260812`, and fold error threshold `0.05` as
required CLI arguments rather than hidden defaults.

After each configuration, collect `dynamic_activation_rows()` before the next
configuration replaces quantizers. Write aggregate, sample, regional, block,
activation resolution/channel, layer quantization, propagation, dynamic
overhead, configuration manifest, and metadata files under `cspn/`.

Export prediction payloads for FP32, both W4A4 configurations, and both
A4-only configurations. Validate all six configurations contain all 64 finite
samples and each exported configuration contains exactly the fixed indices.

- [ ] **Step 4: Run runner and compatibility tests**

Run:

```bash
python -m pytest tests/test_run_nyu_cspn_dynamic_group_a4.py \
  tests/test_run_nyu_cspn_smoothquant_group.py \
  tests/test_run_nyu_cspn_scale_aware_grouping.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the runner**

```bash
git add scripts/run_nyu_cspn_dynamic_group_a4.py \
  tests/test_run_nyu_cspn_dynamic_group_a4.py
git commit -m "feat: evaluate CSPN dynamic Group-A4"
```

### Task 4: CUDA evaluation and result report

**Files:**
- Create: `docs/2026-08-13-cspn-dynamic-group-a4-results.md`

- [ ] **Step 1: Run focused tests and syntax checks**

Run:

```bash
python -m py_compile scripts/hardware_aligned_quantization.py \
  scripts/run_nyu_cspn_activation_resolution.py \
  scripts/run_nyu_cspn_dynamic_group_a4.py
python -m pytest tests/test_quant_specs.py \
  tests/test_hardware_aligned_quantization.py \
  tests/test_run_nyu_cspn_activation_resolution.py \
  tests/test_run_nyu_cspn_dynamic_group_a4.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the six-configuration CUDA evaluation**

Run from `/workspace/CSPN/cspn_pytorch`:

```bash
PYTHONPATH=/workspace/SPN_Quantization/.worktrees/main \
CUDA_VISIBLE_DEVICES=0 python \
  /workspace/SPN_Quantization/.worktrees/main/scripts/run_nyu_cspn_dynamic_group_a4.py \
  --run-dir /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24 \
  --checkpoint /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24/best.pt \
  --sample-metrics /workspace/SPN_Quantization/profile_logs/nyu_strict_w4a4_fp4_evaluation/primary/rtn/cspn/sample_metrics.csv \
  --data-root /workspace/CSPN/cspn_pytorch \
  --out-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_dynamic_group_a4 \
  --device cuda:0 --seed 20260812 --calibration-samples 128 \
  --sample-capacity 256 --fold-max-error 0.05
```

Expected: exit 0 and six configurations with 64 finite sample rows each.

- [ ] **Step 3: Validate artifacts and analyze paired results**

Check checkpoint/evaluation-index identity, `6 * 64` unique sample rows, zero
non-finite ratios, and prediction coverage. Report static-to-dynamic deltas for
A4-only and W4A4, paired better/worse counts, activation SQNR/new-zero/error
composition, block SQNR, runtime scale counts, and reduction elements.

- [ ] **Step 4: Write the result report**

Create `docs/2026-08-13-cspn-dynamic-group-a4-results.md` with protocol,
aggregate metrics, paired stability, activation error mechanism, online scale
overhead, and a decision on whether dynamic Group-8 should replace static
Group-8. Do not call the method successful solely because zero-code rate falls.

- [ ] **Step 5: Run full verification**

Run:

```bash
python -m pytest -q
git diff --check
git status --short --branch
```

Expected: all tests pass and only intentional result documentation is
uncommitted.

- [ ] **Step 6: Commit verified results**

```bash
git add docs/2026-08-13-cspn-dynamic-group-a4-results.md
git commit -m "docs: evaluate CSPN dynamic Group-A4"
```
