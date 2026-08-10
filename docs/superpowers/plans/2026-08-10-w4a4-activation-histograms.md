# W4A4 Activation Histograms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate count-conserving activation, tail, quantization-error, and code-occupancy histograms for every strict W4A4 activation boundary in CSPN, DySPN, NLSPN, and CompletionFormer on the fixed 64-sample NYU set.

**Architecture:** Add an explicit recorder callback to the existing hardware-aligned instrumentor so diagnostics observe the exact reference tensor, dequantized tensor, codes, and frozen quantizer at each real QDQ boundary. A standalone profiler performs one profiled-flow range pass and one histogram pass, persists compact NPZ/CSV data, and a separate plotting module generates paginated PDFs and cross-model PNG summaries.

**Tech Stack:** Python 3.7/3.11, PyTorch 1.10/current, NumPy, Matplotlib, PyMuPDF, Pillow, unittest/pytest, official CUDA extensions for NLSPN and CompletionFormer.

---

## File Structure

- Create `scripts/activation_histograms.py`: streaming range/histogram accumulators, site recorder, count validation, NPZ/CSV persistence, and RGB/depth slice definitions.
- Create `scripts/run_w4a4_activation_histograms.py`: official-model setup, strict W4A4 calibration/configuration, two profiling passes, manifest coverage, and metadata output.
- Create `scripts/plot_w4a4_activation_histograms.py`: per-model PDF, critical-site figures, input RGB/depth comparison, group distribution, and root comparison outputs.
- Create `scripts/run_w4a4_activation_histograms.sh`: reproducible four-model launcher with explicit run directories, data root, environments, devices, and output root.
- Modify `scripts/hardware_aligned_quantization.py`: explicit activation recorder API and callback emission at input/output/ReLU/LayerNorm QDQ sites.
- Create `tests/test_activation_histograms.py`: streaming statistics, persistence, coverage, and RGB/depth slicing tests.
- Create `tests/test_plot_w4a4_activation_histograms.py`: PDF/PNG generation and summary-selection tests.
- Modify `tests/test_hardware_aligned_quantization.py`: recorder payload and shared-call-index tests.
- Create `tests/test_w4a4_activation_histogram_runner.py`: configuration identity and manifest-coverage tests without loading NYU or CUDA models.
- Modify `README.md`: command, output contract, and semantic A8 disclosure.

### Task 1: Emit Exact QDQ Records From The Instrumentor

**Files:**
- Modify: `scripts/hardware_aligned_quantization.py`
- Modify: `tests/test_hardware_aligned_quantization.py`

- [ ] **Step 1: Write a failing recorder test**

Add a recorder with a strict method signature and assert that a shared Conv emits distinct call indices with exact tensors and codes:

```python
class RecordingSink(object):
    def __init__(self):
        self.rows = []

    def record(self, module, kind, call_index, group, reference,
               quantized, codes, quantizer):
        self.rows.append({
            "module": module,
            "kind": kind,
            "call_index": call_index,
            "group": group,
            "reference": reference.detach().clone(),
            "quantized": quantized.detach().clone(),
            "codes": codes.detach().clone(),
            "quantizer": quantizer,
        })


def test_activation_recorder_observes_real_qdq_and_shared_calls(self):
    model = SharedConv().eval()
    sink = RecordingSink()
    instrumentor = HardwareAlignedInstrumentor(
        model, lambda name, module: "encoder")
    instrumentor.set_activation_recorder(sink)
    instrumentor.observe()
    model(torch.ones(1, 1, 2, 2))
    instrumentor.freeze()
    instrumentor.configure(4, 4, {"encoder"})
    model(torch.ones(1, 1, 2, 2))

    inputs = [row for row in sink.rows if row["kind"] == "input"]
    assert [row["call_index"] for row in inputs] == [0, 1]
    assert all(row["codes"].dtype == torch.int32 for row in inputs)
    torch.testing.assert_close(
        inputs[0]["quantized"], inputs[0]["quantizer"](
            inputs[0]["reference"]))
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m pytest -q tests/test_hardware_aligned_quantization.py \
  -k activation_recorder_observes_real_qdq_and_shared_calls
```

Expected: FAIL because `set_activation_recorder` does not exist.

- [ ] **Step 3: Implement the recorder contract**

Add explicit state and methods to `HardwareAlignedInstrumentor`:

```python
self.activation_recorder = None
self.activation_call_counts = {}

def set_activation_recorder(self, recorder):
    if recorder is None or not callable(recorder.record):
        raise TypeError("activation recorder must define record")
    self.activation_recorder = recorder

def clear_activation_recorder(self):
    self.activation_recorder = None

def _next_activation_call(self, name, kind):
    key = (name, kind)
    index = self.activation_call_counts[key] \
        if key in self.activation_call_counts else 0
    self.activation_call_counts[key] = index + 1
    return index

def _record_activation(self, name, kind, reference, quantized,
                       codes, quantizer):
    if self.activation_recorder is None:
        return
    self.activation_recorder.record(
        name, kind, self._next_activation_call(name, kind),
        self.groups[name], reference, quantized, codes, quantizer)
```

Reset `activation_call_counts` in the model forward-pre-hook. In uniform input,
output, ReLU, and LayerNorm QDQ paths, retain the returned codes and invoke
`_record_activation` after existing statistics update. Do not emit records in
observe or bypass mode.

- [ ] **Step 4: Verify input, output, ReLU, LayerNorm, and per-channel records**

Add focused assertions that all emitted payloads preserve shape, code domain,
group, signedness, and vector scale. Run:

```bash
python -m pytest -q tests/test_hardware_aligned_quantization.py
```

Expected: all tests PASS.

- [ ] **Step 5: Commit the recorder API**

```bash
git add scripts/hardware_aligned_quantization.py \
  tests/test_hardware_aligned_quantization.py
git commit -m "feat: expose activation QDQ records"
```

### Task 2: Implement Streaming Range And Histogram Accumulators

**Files:**
- Create: `scripts/activation_histograms.py`
- Create: `tests/test_activation_histograms.py`

- [ ] **Step 1: Write failing range-collection tests**

Cover deterministic magnitude sampling, exact zero counts, signal/error energy,
channel maxima, and quantizer identity:

```python
def test_range_collector_preserves_exact_counts_and_energies():
    quantizer = UnsignedActivationQuantizer(4, 3.0)
    reference = torch.tensor([[[[0.0, 0.1, 1.0, 3.5]]]])
    quantized, codes = quantizer.quantize_with_codes(reference)
    collector = RangeCollector(capacity=32, per_update=32)
    collector.update(reference, quantized, codes, quantizer, channel_dim=1)
    row = collector.summary()

    assert row["elements"] == 4
    assert row["reference_zeros"] == 1
    assert row["zero_codes"] == 1
    assert row["saturated_values"] == 1
    assert row["endpoint_codes"] == 1
    assert row["signal_energy"] == pytest.approx(float((reference ** 2).sum()))
    assert row["error_energy"] == pytest.approx(
        float(((reference - quantized) ** 2).sum()))
```

- [ ] **Step 2: Run the range tests and verify RED**

```bash
python -m pytest -q tests/test_activation_histograms.py \
  -k range_collector
```

Expected: FAIL because `scripts.activation_histograms` does not exist.

- [ ] **Step 3: Implement `RangeCollector` and quantizer descriptors**

Implement exact counters in float64/int64 and reuse the deterministic bounded
sampler pattern from `activation_outlier_analysis.py`. The collector API is:

```python
collector.update(reference, quantized, codes, quantizer, channel_dim)
collector.summary()
collector.signed_edges(bin_count)
collector.magnitude_edges(bin_count)
collector.error_edges(bin_count)
```

Quantizer descriptors explicitly serialize `bits`, `unsigned`, `qmin`, `qmax`,
`scale`, and channel dimension. Scalar and vector scales are separate code paths;
missing attributes raise naturally.

- [ ] **Step 4: Write failing histogram count-conservation tests**

```python
def test_histogram_accumulator_conserves_reference_error_and_code_counts():
    quantizer = SymmetricActivationQuantizer(4, 4.0)
    reference = torch.linspace(-5.0, 5.0, 101).reshape(1, 1, 1, -1)
    quantized, codes = quantizer.quantize_with_codes(reference)
    ranges = RangeCollector(capacity=256, per_update=256)
    ranges.update(reference, quantized, codes, quantizer, 1)
    hist = HistogramAccumulator.from_range(ranges, bin_count=64)
    hist.update(reference, quantized, codes)

    assert hist.reference_total == reference.numel()
    assert hist.magnitude_total == reference.numel() - 1
    assert hist.error_total == reference.numel()
    assert hist.code_total == reference.numel()
    assert hist.reference_underflow > 0
    assert hist.reference_overflow > 0
```

- [ ] **Step 5: Implement GPU-side bucketization and code occupancy**

Use `torch.bucketize` and `torch.bincount` on the source device, transfer only
the count vectors to CPU, and keep explicit underflow/overflow and zero buckets.
Reject nonfinite reference, quantized, error, edge, and scale values.

- [ ] **Step 6: Run the complete accumulator tests**

```bash
python -m pytest -q tests/test_activation_histograms.py
```

Expected: all current tests PASS.

- [ ] **Step 7: Commit the streaming core**

```bash
git add scripts/activation_histograms.py tests/test_activation_histograms.py
git commit -m "feat: add streaming activation histograms"
```

### Task 3: Add Site Recording, RGB/Depth Slices, Persistence, And Coverage

**Files:**
- Modify: `scripts/activation_histograms.py`
- Modify: `tests/test_activation_histograms.py`

- [ ] **Step 1: Write failing call-index and RGB/depth split tests**

```python
def test_cspn_input_is_split_into_rgb_and_sparse_depth_records():
    recorder = ActivationHistogramRecorder(
        model_name="cspn", phase="range", capacity=64, per_update=64)
    value = torch.cat((torch.ones(1, 3, 2, 2),
                       torch.full((1, 1, 2, 2), 8.0)), dim=1)
    quantizer = UnsignedActivationQuantizer(8, 8.0)
    quantized, codes = quantizer.quantize_with_codes(value)
    recorder.record("conv1_1", "input", 0, "encoder",
                    value, quantized, codes, quantizer)

    assert set(recorder.site_names()) == {
        "conv1_1#0:input", "input_rgb#0:input", "input_depth#0:input"}
```

Add equivalent separated-stem tests for DySPN, NLSPN, and CompletionFormer.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
python -m pytest -q tests/test_activation_histograms.py \
  -k 'split or call_index or coverage'
```

Expected: FAIL because the recorder and coverage validator are absent.

- [ ] **Step 3: Implement strict site keys and phase transitions**

Implement:

```python
recorder = ActivationHistogramRecorder(..., phase="range")
recorder.record(...)
range_rows = recorder.freeze_ranges(bin_count=128)
recorder.begin_histogram_pass()
recorder.record(...)
recorder.validate(expected_manifest_sites, expected_updates=64)
```

Site keys are `module#call_index:kind`. Synthetic input slices use stable module
names `input_rgb` and `input_depth`, retain the parent call index, and are marked
`synthetic_slice=1` so they are excluded from manifest coverage.

- [ ] **Step 4: Write failing NPZ/CSV round-trip tests**

Assert every `histogram_index.csv` array key exists in `histogram_data.npz`, all
count sums survive serialization, and no unexpected array remains unindexed.

- [ ] **Step 5: Implement persistence and hard coverage checks**

Write CSV through `csv.DictWriter`, NPZ through `numpy.savez_compressed`, and
metadata through `json.dumps(..., allow_nan=False)`. Validate:

```python
observed_manifest_sites == expected_manifest_sites
sum(reference_counts) + underflow + overflow + zero_bucket == elements
sum(error_counts) + error_underflow + error_overflow == elements
sum(code_counts) == elements
```

- [ ] **Step 6: Run and commit site/persistence behavior**

```bash
python -m pytest -q tests/test_activation_histograms.py
git add scripts/activation_histograms.py tests/test_activation_histograms.py
git commit -m "feat: persist complete activation profiles"
```

### Task 4: Build The Strict W4A4 Histogram Runner

**Files:**
- Create: `scripts/run_w4a4_activation_histograms.py`
- Create: `tests/test_w4a4_activation_histogram_runner.py`

- [ ] **Step 1: Write failing configuration-identity tests**

Test that the runner selects only `FP4V_W4A4`, preserves semantic A8 overrides,
uses all instrumentor groups, and derives expected activation sites from the
configured instrumentor manifest:

```python
def test_select_w4a4_config_rejects_missing_or_duplicate_config():
    configs = [{"name": "FP4V_W4A4", "w_bits": 4, "a_bits": 4}]
    assert select_w4a4_config(configs)["name"] == "FP4V_W4A4"
    with pytest.raises(RuntimeError):
        select_w4a4_config([])
```

- [ ] **Step 2: Run the runner tests and verify RED**

```bash
python -m pytest -q tests/test_w4a4_activation_histogram_runner.py
```

Expected: FAIL because the runner module does not exist.

- [ ] **Step 3: Implement setup and calibration**

Reuse, without copying quantization formulas:

- `build_model`, `load_run_args`, `prepare_args`;
- `calibration_dataset`, `seeded_sample`, `batch_from_sample`;
- `prepare_hardware_model`, `HardwareAlignedInstrumentor`;
- `build_fp4_runner_configurations`;
- `resolve_per_channel_activation_inputs`;
- `install_propagation_adapter`, `configure_runtime_adapter`.

The CLI requires `--run-dir`, `--data-root`, `--strict-root`, `--out-dir`,
`--device`, `--seed`, and `--calibration-samples`. Defaults are not embedded for
checkpoint identity, data root, or strict result root.

- [ ] **Step 4: Implement the two profiling passes**

After calibration and `instrumentor.freeze()`:

```python
config = select_w4a4_config(configs)
instrumentor.configure(...)
instrumentor.set_activation_recorder(range_recorder)
run_indices(model, dataset, indices, saved_args, device)
range_recorder.freeze_ranges(bin_count=args.histogram_bins)

instrumentor.configure(...)
instrumentor.set_activation_recorder(histogram_recorder)
run_indices(model, dataset, indices, saved_args, device)
histogram_recorder.validate(expected_sites, expected_updates=len(indices))
```

Reconfigure the propagation adapter identically before each profiling pass.
Validate checkpoint SHA256, source SHA256, calibration indices, and semantic A8
rows against `--strict-root/primary/rtn/<model>/metadata.json` and manifests.

- [ ] **Step 5: Write results and immutable metadata**

Write the model directory only after all coverage/count checks pass. Metadata
records the exact run args, strict identity, calibration indices, observer and
histogram pass counts, manifest site count, observed call-index site count,
synthetic input slices, and elapsed seconds.

- [ ] **Step 6: Run unit tests and commit the runner**

```bash
python -m pytest -q tests/test_w4a4_activation_histogram_runner.py \
  tests/test_activation_histograms.py
git add scripts/run_w4a4_activation_histograms.py \
  tests/test_w4a4_activation_histogram_runner.py
git commit -m "feat: run strict W4A4 histogram profiling"
```

### Task 5: Generate Paginated PDFs And Summary Figures

**Files:**
- Create: `scripts/plot_w4a4_activation_histograms.py`
- Create: `tests/test_plot_w4a4_activation_histograms.py`

- [ ] **Step 1: Write failing summary-selection and plotting tests**

Use a synthetic two-site NPZ/CSV fixture and assert that independent selectors
choose highest error share, lowest SQNR, strongest spatial tail, and strongest
channel imbalance. Render one PDF and three PNGs, then assert page count and
nonblank pixels.

- [ ] **Step 2: Run plotting tests and verify RED**

```bash
python -m pytest -q tests/test_plot_w4a4_activation_histograms.py
```

Expected: FAIL because the plotting module does not exist.

- [ ] **Step 3: Implement common style and site panels**

Set:

```python
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 14,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})
```

Each PDF site panel contains signed reference, normalized magnitude, error, and
code occupancy. Use no figure title, keep grid at z-order 0 and data at z-order
3, and keep categorical tick labels unrotated.

- [ ] **Step 4: Implement model and cross-model summaries**

Generate `critical_layers.png`, `rgb_depth_input_histograms.png`,
`group_outlier_distribution.png`, root `all_models_outlier_comparison.png`, and
root `all_models_outlier_summary.csv`. The group figure displays site prevalence
and median/maximum tail severity separately from error-energy share.

- [ ] **Step 5: Verify plots and commit**

```bash
python -m pytest -q tests/test_plot_w4a4_activation_histograms.py
git add scripts/plot_w4a4_activation_histograms.py \
  tests/test_plot_w4a4_activation_histograms.py
git commit -m "feat: plot W4A4 activation distributions"
```

### Task 6: Add Portable Four-Model Launching And Documentation

**Files:**
- Create: `scripts/run_w4a4_activation_histograms.sh`
- Modify: `README.md`
- Modify: `tests/test_strict_w4a4_fp4_shell_contract.py`

- [ ] **Step 1: Write a failing shell-contract test**

Assert the launcher contains all four official run directories, passes
`--data-root`, `--strict-root`, `--calibration-samples 64`, assigns explicit
CUDA devices, invokes the plotting script only after all model runs, and uses
the Python 3.7 CompletionFormer environment where DCN requires it.

- [ ] **Step 2: Run the shell test and verify RED**

```bash
python -m pytest -q tests/test_strict_w4a4_fp4_shell_contract.py \
  -k activation_histogram
```

Expected: FAIL because the launcher is absent.

- [ ] **Step 3: Implement the launcher**

Use `set -euo pipefail`, resolve repository root from the script path, validate
every run directory/checkpoint/data directory/DCN extension before launch, run
one model per GPU, wait for every PID, and invoke plotting only after successful
profiles. Do not write to `/tmp` and do not download dependencies at runtime.

- [ ] **Step 4: Document invocation and quantization semantics**

Document that W4A4 is nominal internal precision with A8 sparse-depth,
confidence, initial-depth/guidance, and propagation exceptions. Include output
paths and clarify that group error-energy share is not outlier prevalence.

- [ ] **Step 5: Verify and commit portability changes**

```bash
bash -n scripts/run_w4a4_activation_histograms.sh
python -m pytest -q tests/test_strict_w4a4_fp4_shell_contract.py
git add scripts/run_w4a4_activation_histograms.sh README.md \
  tests/test_strict_w4a4_fp4_shell_contract.py
git commit -m "docs: add activation histogram workflow"
```

### Task 7: CUDA Smoke Tests And Full 64-Sample Profiling

**Files:**
- Generate: `profile_logs/nyu_w4a4_activation_histograms_64/**`

- [ ] **Step 1: Initialize and verify official submodules**

```bash
git submodule update --init --recursive
git submodule status
```

Expected: CSPN source is in the main repository and all three external model
submodules show the commits recorded by the strict metadata without a leading
`-` or `+`.

- [ ] **Step 2: Run one-sample CUDA smoke tests**

Run each model with `--calibration-samples 1` into
`profile_logs/nyu_w4a4_activation_histograms_smoke`. Expected: zero coverage,
count, checkpoint, source, DCN, or nonfinite errors, and one model directory per
model.

- [ ] **Step 3: Run the full four-model launcher**

```bash
bash scripts/run_w4a4_activation_histograms.sh
```

Expected: 64 range-pass and 64 histogram-pass forwards per model, followed by
successful PDF/PNG generation.

- [ ] **Step 4: Validate generated artifacts**

Run the profiler's `--validate-only` mode for all model directories and verify:

```text
CSPN: 76 manifest boundaries plus synthetic input slices
DySPN: 169 manifest boundaries plus synthetic input slices
NLSPN: 97 manifest boundaries plus synthetic input slices
CompletionFormer: 537 manifest boundaries plus synthetic input slices
```

Shared-call sites may increase observed site rows but may not reduce manifest
coverage. Confirm every PDF is nonempty and every PNG has nonzero pixel
variance.

- [ ] **Step 5: Compare recomputed metrics with strict outputs**

Join histogram summary rows to
`profile_logs/nyu_strict_w4a4_fp4_evaluation/primary/rtn/<model>/layer_quantization_metrics.csv`
by model/module/kind. Compare aggregate signal/error energy, SQNR, zero-code
ratio, and saturation ratio within `1e-6` relative tolerance for deterministic
operators and a documented `1e-4` tolerance for DCN/grid-sample paths.

### Task 8: Final Regression, Review, And Commit

**Files:**
- Modify only files found defective during review.

- [ ] **Step 1: Run focused and full tests**

```bash
python -m pytest -q tests/test_activation_histograms.py \
  tests/test_hardware_aligned_quantization.py \
  tests/test_w4a4_activation_histogram_runner.py \
  tests/test_plot_w4a4_activation_histograms.py \
  tests/test_strict_w4a4_fp4_shell_contract.py
python -m pytest -q tests
```

Expected: all focused tests and all repository tests PASS.

- [ ] **Step 2: Run static and repository checks**

```bash
python -m py_compile scripts/activation_histograms.py \
  scripts/run_w4a4_activation_histograms.py \
  scripts/plot_w4a4_activation_histograms.py
bash -n scripts/run_w4a4_activation_histograms.sh
git diff --check
git status --short
```

Expected: no syntax errors, shell errors, whitespace errors, or unrelated files.

- [ ] **Step 3: Review implementation against the design**

Confirm all manifest boundaries, output files, plotting rules, semantic input
slices, strict identity checks, and non-goals are satisfied. Remove smoke output
after the full run succeeds; retain only the formal output directory.

- [ ] **Step 4: Commit final corrections**

```bash
git add scripts tests README.md docs/superpowers
git commit -m "test: verify complete W4A4 histogram workflow"
```

- [ ] **Step 5: Record final evidence**

Report commit IDs, full test count, model/site counts, artifact paths, aggregate
outlier distributions, and any metric-tolerance exceptions. Do not claim a
GitHub push unless `git ls-remote origin refs/heads/main` confirms the remote
commit.
