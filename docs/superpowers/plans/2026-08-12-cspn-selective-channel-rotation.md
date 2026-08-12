# CSPN Selective Channel Rotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and evaluate strict selective Random/Hadamard channel rotation at the two approved signed CSPN decoder boundaries under CNN W4A4 and propagation-aware A8/Q13.

**Architecture:** Add one focused rotation module that owns orthogonal transforms, convolution weight absorption, boundary observation, and boundary QDQ. Extend the CSPN semantic adapter only with declarative topology metadata, then add a CSPN-only NYU runner that reuses existing model loading, calibration, propagation, metric, and prediction APIs. The ordinary hardware instrumentor continues to own CNN W4A4 outside the two rotation boundaries and excludes the complete guidance head.

**Tech Stack:** Python, PyTorch forward hooks, NumPy, existing hardware-aligned QDQ, existing CSPN propagation adapter, pytest/unittest, Matplotlib.

---

## File Structure

- Create `spn_quant/rotation.py`: orthogonal matrices, channel transforms,
  convolution weight absorption, CSPN boundary controller, boundary QDQ, and
  metrics.
- Modify `spn_quant/adapters/cspn.py`: declare the two official CSPN rotation
  boundaries and exact consumer modules/channel slices.
- Modify `spn_quant/adapters/__init__.py`: export the CSPN rotation-boundary
  declaration type through the existing adapter package.
- Create `scripts/run_nyu_cspn_rotation.py`: CSPN-only calibration and evaluation
  runner using existing NYU/model/metric helpers.
- Create `scripts/plot_nyu_cspn_rotation.py`: concise analysis plots and aggregate
  table from runner outputs.
- Create `tests/test_rotation.py`: mathematical transforms, fanout, concat slice,
  QDQ, and metric tests.
- Modify `tests/test_model_semantic_adapters.py`: exact CSPN topology declaration
  and protected-path tests.
- Create `tests/test_run_nyu_cspn_rotation.py`: configuration, ownership, output
  schema, and selection tests.
- Create `tests/test_plot_nyu_cspn_rotation.py`: aggregate and plot generation
  tests.

### Task 1: Orthogonal Rotation Primitives

**Files:**
- Create: `spn_quant/rotation.py`
- Create: `tests/test_rotation.py`

- [ ] **Step 1: Write failing tests for Random and Hadamard matrices**

```python
def test_random_rotation_is_deterministic_and_orthogonal():
    first = random_orthogonal_matrix(8, seed=17)
    second = random_orthogonal_matrix(8, seed=17)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first @ first.t(), torch.eye(8),
                               rtol=1e-5, atol=1e-6)

def test_hadamard_rotation_is_orthogonal():
    rotation = hadamard_rotation_matrix(8, seed=17)
    torch.testing.assert_close(rotation @ rotation.t(), torch.eye(8),
                               rtol=1e-5, atol=1e-6)

def test_hadamard_requires_power_of_two_channels():
    with pytest.raises(ValueError, match="power of two"):
        hadamard_rotation_matrix(6, seed=17)
```

- [ ] **Step 2: Run the primitive tests and verify failure**

Run: `pytest -q tests/test_rotation.py`

Expected: collection fails because `spn_quant.rotation` does not exist.

- [ ] **Step 3: Implement deterministic orthogonal matrices and channel transforms**

```python
def random_orthogonal_matrix(channels: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(channels, channels, generator=generator,
                         dtype=torch.float64)
    q, r = torch.linalg.qr(matrix)
    signs = torch.where(torch.diag(r) < 0, -1.0, 1.0)
    return (q * signs.unsqueeze(0)).to(torch.float32)

def hadamard_rotation_matrix(channels: int, seed: int) -> torch.Tensor:
    if channels <= 0 or channels & (channels - 1):
        raise ValueError("Hadamard channels must be a power of two")
    matrix = torch.ones(1, 1, dtype=torch.float32)
    while matrix.shape[0] < channels:
        matrix = torch.cat((torch.cat((matrix, matrix), 1),
                            torch.cat((matrix, -matrix), 1)), 0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    signs = torch.randint(0, 2, (channels,), generator=generator,
                          dtype=torch.int64).float().mul_(2).sub_(1)
    return matrix.mul(signs.unsqueeze(0)).div(math.sqrt(channels))

def rotate_channels(tensor: torch.Tensor,
                    rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum("oc,nchw->nohw", rotation.to(tensor), tensor)
```

- [ ] **Step 4: Add convolution weight absorption tests**

```python
def test_absorbed_conv_matches_rotated_input():
    conv = nn.Conv2d(8, 5, 3, padding=1, bias=True)
    value = torch.randn(2, 8, 7, 9)
    rotation = random_orthogonal_matrix(8, 3)
    reference = conv(value)
    absorb_input_rotation(conv, rotation)
    actual = conv(rotate_channels(value, rotation))
    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)

def test_absorb_rotation_updates_only_concat_slice():
    conv = nn.Conv2d(12, 5, 1, bias=False)
    left = torch.rand(2, 4, 3, 3)
    right = torch.randn(2, 8, 3, 3)
    rotation = random_orthogonal_matrix(8, 5)
    reference = conv(torch.cat((left, right), 1))
    absorb_input_rotation(conv, rotation, channel_start=4,
                          channel_count=8)
    actual = conv(torch.cat((left, rotate_channels(right, rotation)), 1))
    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)
```

- [ ] **Step 5: Implement full and sliced weight absorption**

```python
def absorb_input_rotation(module: nn.Conv2d, rotation: torch.Tensor,
                          channel_start: int = 0,
                          channel_count: Optional[int] = None) -> None:
    if module.groups != 1:
        raise ValueError("rotation requires groups=1")
    count = rotation.shape[0] if channel_count is None else int(channel_count)
    start = int(channel_start)
    stop = start + count
    weight = module.weight.data[:, start:stop]
    transformed = torch.einsum("oihw,ji->ojhw", weight,
                               rotation.to(weight))
    module.weight.data[:, start:stop].copy_(transformed)
```

- [ ] **Step 6: Run tests and commit**

Run: `pytest -q tests/test_rotation.py`

Expected: all primitive and absorption tests pass.

```bash
git add spn_quant/rotation.py tests/test_rotation.py
git commit -m "feat: add channel rotation primitives"
```

### Task 2: Declare the Two CSPN Rotation Boundaries

**Files:**
- Modify: `spn_quant/adapters/cspn.py`
- Modify: `spn_quant/adapters/__init__.py`
- Modify: `tests/test_model_semantic_adapters.py`

- [ ] **Step 1: Write the failing CSPN declaration test**

```python
def test_cspn_declares_only_approved_rotation_boundaries(self):
    adapter = install_model_semantic_adapter(CModel(), "cspn", strict=True)
    boundaries = adapter.rotation_boundaries()
    self.assertEqual([item.name for item in boundaries], [
        "decoder_entry", "layer4_signed_skip",
    ])
    self.assertEqual(boundaries[0].module, "gud_up_proj_layer1")
    self.assertEqual(boundaries[0].consumers, (
        RotationConsumer("gud_up_proj_layer1.conv1", 0, None),
        RotationConsumer("gud_up_proj_layer1.sc_conv1", 0, None),
    ))
    self.assertEqual(boundaries[1].module, "gud_up_proj_layer4")
    self.assertEqual(boundaries[1].argument_index, 1)
    self.assertEqual(boundaries[1].consumers, (
        RotationConsumer("gud_up_proj_layer4.conv1_1", 64, 64),
    ))
```

- [ ] **Step 2: Run the declaration test and verify failure**

Run: `pytest -q tests/test_model_semantic_adapters.py::Tests::test_cspn_declares_only_approved_rotation_boundaries`

Expected: fails because `rotation_boundaries` is absent.

- [ ] **Step 3: Add immutable topology declarations**

```python
@dataclass(frozen=True)
class RotationConsumer:
    module: str
    channel_start: int
    channel_count: Optional[int]

@dataclass(frozen=True)
class RotationBoundary:
    name: str
    module: str
    argument_index: int
    consumers: Tuple[RotationConsumer, ...]
```

`CSPNSemanticAdapter.rotation_boundaries()` resolves the real layer4 skip width
from `gud_up_proj_layer4.conv1.out_channels` and returns exactly:

```python
return (
    RotationBoundary(
        "decoder_entry", "gud_up_proj_layer1", 0,
        (RotationConsumer("gud_up_proj_layer1.conv1", 0, None),
         RotationConsumer("gud_up_proj_layer1.sc_conv1", 0, None))),
    RotationBoundary(
        "layer4_signed_skip", "gud_up_proj_layer4", 1,
        (RotationConsumer("gud_up_proj_layer4.conv1_1", skip_channels,
                          skip_channels),)),
)
```

- [ ] **Step 4: Run adapter tests and commit**

Run: `pytest -q tests/test_model_semantic_adapters.py`

Expected: all adapter tests pass.

```bash
git add spn_quant/adapters/cspn.py spn_quant/adapters/__init__.py tests/test_model_semantic_adapters.py
git commit -m "feat: declare CSPN rotation boundaries"
```

### Task 3: CSPN Boundary Controller and Quantization Metrics

**Files:**
- Modify: `spn_quant/rotation.py`
- Modify: `tests/test_rotation.py`

- [ ] **Step 1: Write failing fanout and concat controller tests**

```python
def test_controller_preserves_fp_for_both_decoder_entry_consumers():
    model = CModel()
    controller = CSPNRotationController(model, boundaries(model), seed=7)
    value = torch.randn(1, 4, 5, 5)
    reference = model(value)
    controller.configure({"decoder_entry": "random"}, quantize=False)
    actual = model(value)
    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)

def test_controller_preserves_fp_for_layer4_skip_slice():
    model = CModel()
    controller = CSPNRotationController(model, boundaries(model), seed=7)
    value = torch.randn(1, 4, 5, 5)
    reference = model(value)
    controller.configure({"layer4_signed_skip": "hadamard"}, quantize=False)
    actual = model(value)
    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)
```

- [ ] **Step 2: Run tests and verify controller failure**

Run: `pytest -q tests/test_rotation.py -k controller`

Expected: fails because `CSPNRotationController` is absent.

- [ ] **Step 3: Implement observe/freeze/configure/disable ownership**

The controller registers one forward pre-hook on each declared block. It stores
original consumer weights once, restores them before each configuration, rotates
only the declared positional argument, and absorbs the selected matrix into all
declared consumers. Its modes are exactly `bypass`, `observe`, and `quantize`.

```python
def configure(self, methods: Mapping[str, str], bits: int,
              group_size: Optional[int]) -> None:
    self._restore_weights()
    for boundary in self.boundaries:
        method = methods[boundary.name]
        rotation = self._rotation(boundary, method)
        for consumer in boundary.consumers:
            absorb_input_rotation(self.modules[consumer.module], rotation,
                                  consumer.channel_start,
                                  consumer.channel_count)
        self.states[boundary.name].configure(rotation, bits, group_size)
    self.mode = "quantize"
```

No method key defaults are used. A configuration explicitly maps both boundaries
to one of `identity`, `random`, or `hadamard`.

- [ ] **Step 4: Add boundary observation and QDQ metric tests**

```python
def test_boundary_observer_reports_requested_tail_and_qdq_metrics():
    observer = RotationBoundaryObserver(channels=4)
    observer.update(torch.tensor([[[[-8.0]], [[-1.0]], [[2.0]], [[4.0]]]]))
    quantizer = observer.quantizer(bits=4, group_size=None)
    row = observer.statistics(quantizer)
    assert set(("maximum", "p75", "p99", "p99_9", "p99_99",
                "kurtosis", "channel_imbalance", "sqnr",
                "zero_code_ratio", "saturation_ratio")) <= set(row)

def test_group_quantizer_requires_divisible_channel_count():
    observer = RotationBoundaryObserver(channels=6)
    observer.update(torch.randn(1, 6, 2, 2))
    with pytest.raises(ValueError, match="divide"):
        observer.quantizer(bits=4, group_size=4)
```

- [ ] **Step 5: Implement signed calibration, group QDQ, and metrics**

Use one bounded deterministic sample for percentiles/SQNR and exact accumulated
per-channel L2 energy. `freeze()` rejects a boundary whose observed minimum is
nonnegative. Group QDQ uses one signed symmetric scale per contiguous group and
records codes directly for zero and saturation ratios.

- [ ] **Step 6: Run focused tests and commit**

Run: `pytest -q tests/test_rotation.py tests/test_model_semantic_adapters.py`

Expected: all tests pass.

```bash
git add spn_quant/rotation.py tests/test_rotation.py
git commit -m "feat: add CSPN rotation boundary controller"
```

### Task 4: CSPN Rotation Configuration and Ownership

**Files:**
- Create: `scripts/run_nyu_cspn_rotation.py`
- Create: `tests/test_run_nyu_cspn_rotation.py`

- [ ] **Step 1: Write failing experiment matrix tests**

```python
def test_build_configurations_covers_identity_single_and_joint_rotation():
    names = [row["name"] for row in build_configurations(32)]
    assert names == [
        "FP32", "RTN_W4A4", "GROUP_W4A4",
        "RANDOM_decoder_entry", "RANDOM_layer4_signed_skip", "RANDOM_both",
        "HADAMARD_decoder_entry", "HADAMARD_layer4_signed_skip",
        "HADAMARD_both", "HADAMARD_GROUP_both",
    ]

def test_cspn_group_function_excludes_complete_guidance_head():
    assert cspn_quant_group("gud_up_proj_layer6.conv1", nn.Conv2d(4, 8, 1)) is None
    assert cspn_quant_group("gud_up_proj_layer5.conv1", nn.Conv2d(4, 1, 1)) == "depth_head"
```

- [ ] **Step 2: Run runner tests and verify failure**

Run: `pytest -q tests/test_run_nyu_cspn_rotation.py`

Expected: collection fails because the runner does not exist.

- [ ] **Step 3: Implement explicit configurations and CLI**

The CLI requires `--run-dir`, `--sample-metrics`, and `--data-root`. It defaults
to `--checkpoint best.pt`, `--device cuda:0`, `--calibration-samples 128`, and
`--out-dir profile_logs/nyu_cspn_rotation_w4a4`. It accepts one explicit
`--group-size` from `16`, `32`, or `64` for final runs; calibration-only group
selection is implemented in Task 6.

Each non-FP configuration contains explicit values for `w_bits`, `a_bits`, both
boundary methods, `group_size`, and the propagation dictionary:

```python
PROPAGATION_A8_Q13 = {
    "affinity_bits": 8,
    "confidence_bits": 8,
    "offset_bits": 8,
    "state_bits": 8,
    "coefficient_fraction_bits": 13,
}
```

- [ ] **Step 4: Implement strict module ownership**

`cspn_quant_group()` returns `None` for every module under
`gud_up_proj_layer6`, preserving the complete guidance head in FP. It delegates
all remaining modules to `classify_module("cspn", name, module)`.

The two boundary consumer inputs are declared as externally owned by the
rotation controller, and `gud_up_proj_layer5.conv1` output is externally owned so
the depth prediction head output is not passed through ordinary A4 QDQ.

For a rotated configuration, build `W'=WR^T` from the folded FP weight and pass
that tensor as the explicit hardware-instrumentor weight source. The existing
per-output-channel signed W4 QDQ then quantizes `W'`. The runtime rotation hook
must not absorb the matrix into the already quantized weight a second time.

- [ ] **Step 5: Run tests and commit**

Run: `pytest -q tests/test_run_nyu_cspn_rotation.py`

Expected: configuration and ownership tests pass.

```bash
git add scripts/run_nyu_cspn_rotation.py tests/test_run_nyu_cspn_rotation.py
git commit -m "feat: define CSPN rotation experiment contract"
```

### Task 5: Calibration, FP Equivalence, and End-to-End Evaluation

**Files:**
- Modify: `scripts/run_nyu_cspn_rotation.py`
- Modify: `tests/test_run_nyu_cspn_rotation.py`

- [ ] **Step 1: Write failing calibration and schema tests**

```python
def test_fp_equivalence_rejects_wrong_rotation():
    with pytest.raises(RuntimeError, match="FP equivalence"):
        validate_fp_equivalence(torch.ones(1), torch.zeros(1),
                                site="decoder_entry")

def test_metric_schema_contains_depth_and_rotation_metrics():
    assert END_TO_END_FIELDS == (
        "model", "config", "sample_index", "RMSE", "MAE", "ABS_REL",
        "IRMSE", "flat_RMSE", "boundary_RMSE", "nonfinite_ratio",
    )
    assert "channel_imbalance" in BOUNDARY_FIELDS
    assert "block_output_sqnr" in BOUNDARY_FIELDS
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest -q tests/test_run_nyu_cspn_rotation.py`

Expected: new tests fail because evaluation helpers are absent.

- [ ] **Step 3: Implement the calibration flow**

Reuse `build_model`, `prepare_args`, `calibration_dataset`, `seeded_sample`,
`batch_from_sample`, and `sweep.batch_to_model_input`. Perform Conv-BN folding
before constructing the rotation controller. Observe 128 fixed samples once for
the ordinary instrumentor, propagation adapter, and both rotation boundaries.
Freeze all observers, then evaluate each explicit configuration without
recalibrating from evaluation data.

For each rotation configuration, run one calibration input in bypass and rotated
FP modes before QDQ. Compare both boundary block outputs and final prediction
with normalized RMS error at most `5e-4` and maximum error divided by reference
maximum at most `1e-3`; convert a failure to a direct `RuntimeError` naming the
boundary/configuration. The scale-aware criterion covers measured FP32 CUDA
reduction reordering without accepting a materially changed block.

- [ ] **Step 4: Implement evaluation and output files**

Reuse existing `regional_depth_metrics`, prediction payload, and propagation
adapter statistics. Add iRMSE from valid positive depth pixels and emit:

```text
metadata.json
config_manifest.csv
boundary_metrics.csv
sample_metrics.csv
regional_metrics.csv
propagation_metrics.csv
predictions/{configuration_name}/*.npz
```

Reject non-finite calibration tensors and predictions. Export predictions for
FP32, RTN W4A4, and every rotation configuration so the later analysis can
select a named best result without rerunning inference.

- [ ] **Step 5: Run runner tests and focused integration tests**

Run: `pytest -q tests/test_rotation.py tests/test_model_semantic_adapters.py tests/test_run_nyu_cspn_rotation.py`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_nyu_cspn_rotation.py tests/test_run_nyu_cspn_rotation.py
git commit -m "feat: evaluate CSPN selective rotation"
```

### Task 6: Calibration-Only Group Selection and Analysis

**Files:**
- Modify: `scripts/run_nyu_cspn_rotation.py`
- Create: `scripts/plot_nyu_cspn_rotation.py`
- Create: `tests/test_plot_nyu_cspn_rotation.py`

- [ ] **Step 1: Write failing group selection and aggregate tests**

```python
def test_select_group_size_uses_block_error_then_sqnr_then_size():
    rows = [
        {"group_size": 16, "block_output_mse": 0.4, "block_output_sqnr": 20.0},
        {"group_size": 32, "block_output_mse": 0.2, "block_output_sqnr": 18.0},
        {"group_size": 64, "block_output_mse": 0.2, "block_output_sqnr": 21.0},
    ]
    assert select_group_size(rows) == 64

def test_aggregate_selects_named_best_rotation_only():
    rows = fixture_sample_rows()
    summary, best = aggregate_results(rows)
    assert best == "HADAMARD_both"
    assert all(row["config"] != "FP32" for row in summary if row["is_rotation"])
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest -q tests/test_plot_nyu_cspn_rotation.py`

Expected: collection fails because analysis helpers are absent.

- [ ] **Step 3: Implement calibration-only group-size selection**

Evaluate `{16, 32, 64}` on cached calibration boundary inputs and FP block
outputs. Keep only sizes dividing both configured boundary channel counts. Rank
by mean block-output MSE, then descending block-output SQNR, then smaller runtime
group count. Write all candidates to `group_size_search.csv` and the selected
size to `metadata.json`. Do not read evaluation metrics during selection.

- [ ] **Step 4: Implement concise analysis outputs**

Aggregate sample metrics by configuration, select the lowest mean-RMSE rotation
configuration by its explicit name, and produce:

```text
analysis/summary.csv
analysis/activation_rotation_comparison.png
analysis/rmse_comparison.png
analysis/prediction_comparison.png
```

The prediction figure contains GT, FP32, RTN W4A4, and the selected rotation for
the same fixed sample indices. Plotting uses Arial when installed, has no title,
keeps grid lines behind bars, and uses unrotated readable labels.

- [ ] **Step 5: Run analysis tests and commit**

Run: `pytest -q tests/test_plot_nyu_cspn_rotation.py tests/test_run_nyu_cspn_rotation.py`

Expected: all tests pass.

```bash
git add scripts/run_nyu_cspn_rotation.py scripts/plot_nyu_cspn_rotation.py tests/test_plot_nyu_cspn_rotation.py
git commit -m "feat: analyze CSPN rotation results"
```

### Task 7: Remove Invalid Guidance Results and Run CSPN Study

**Files:**
- Modify: `scripts/strict_w4a4_fp4_evaluation.py`
- Modify: `tests/test_strict_w4a4_fp4_evaluation.py`
- Delete runtime artifacts:
  `profile_logs/nyu_brecq_cspn_pa_w4a4/smoke`,
  `profile_logs/nyu_brecq_cspn_pa_w4a4/smoke_v2`
- Generate: `profile_logs/nyu_cspn_rotation_w4a4/`

- [ ] **Step 1: Write a failing invalid-stress exclusion test**

```python
def test_cspn_brecq_nonfinite_stress_is_excluded_from_active_analysis(self):
    path = self.root / "stress" / "brecq" / "cspn" / "sample_metrics.csv"
    rows = read_csv(path)
    for row in rows:
        if row["config"] == "HW_W4A4_full":
            row["RMSE"] = "inf"
            row["nonfinite_pixels"] = "69312"
    write_csv(path, rows)
    tables = analyze_result_root(self.root, expected_samples=2,
                                 baseline_method="rtn")
    assert not any(row["model"] == "cspn" and row["method"] == "brecq"
                   for row in tables["stress"])
```

- [ ] **Step 2: Run the comparison test and verify failure**

Run: `pytest -q tests/test_strict_w4a4_fp4_evaluation.py::StrictW4A4FP4AggregationTest::test_cspn_brecq_nonfinite_stress_is_excluded_from_active_analysis`

Expected: fails because the invalid row is still aggregated.

- [ ] **Step 3: Exclude the invalid stress row and remove obsolete artifacts**

Keep the historical stress files as diagnostic evidence, but exclude exactly
`model == "cspn" and method == "brecq"` from active stress aggregates and plots.
Do not add path probing or a fallback list. Delete only the two approved smoke
directories.

- [ ] **Step 4: Run focused and existing affected tests**

Run:

```bash
pytest -q \
  tests/test_rotation.py \
  tests/test_model_semantic_adapters.py \
  tests/test_hardware_aligned_quantization.py \
  tests/test_propagation_fixed_point.py \
  tests/test_propagation_aware_adapters.py \
  tests/test_run_nyu_cspn_rotation.py \
  tests/test_plot_nyu_cspn_rotation.py
```

Expected: all selected tests pass. Each test maps to changed rotation,
instrumentation, propagation, runner, or analysis behavior.

- [ ] **Step 5: Run the real CSPN calibration and 64-sample evaluation**

Run with the existing CSPN checkpoint, NYU root, and fixed 64-sample metric
source:

```bash
python scripts/run_nyu_cspn_rotation.py \
  --run-dir /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24 \
  --checkpoint best.pt \
  --sample-metrics /workspace/SPN_Quantization/profile_logs/nyu_propagation_aware_quantization_unified/cspn/sample_metrics.csv \
  --data-root /workspace/CSPN/cspn_pytorch \
  --device cuda:0 \
  --calibration-samples 128 \
  --out-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_rotation_w4a4

python scripts/plot_nyu_cspn_rotation.py \
  --input-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_rotation_w4a4/cspn \
  --out-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_rotation_w4a4/analysis
```

Expected: 128 calibration samples and all ten configurations complete with zero
non-finite predictions; analysis outputs name the selected rotation explicitly.

- [ ] **Step 6: Inspect generated metrics for the research decision**

Compare RTN, Group-A4, Random, Hadamard, and Hadamard+Group on activation tails,
block SQNR, RMSE, MAE, AbsRel, iRMSE, flat error, and boundary error. Proceed to
Learned Rotation/BRECQ/QDrop only if the recorded results demonstrate a useful
improvement; do not alter the configured boundary when the result is negative.

- [ ] **Step 7: Commit code and artifact cleanup**

```bash
git add scripts/strict_w4a4_fp4_evaluation.py tests \
  profile_logs/nyu_cspn_rotation_w4a4
git add -u profile_logs/nyu_brecq_cspn_pa_w4a4
git commit -m "results: evaluate CSPN selective rotation"
```

Do not stage the pre-existing whitespace-only change in
`tests/test_qdrop_reconstruction.py` unless this task requires changing the same
line for the invalid-source correction.
