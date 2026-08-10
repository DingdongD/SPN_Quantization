# CompletionFormer Front-Encoder W8A8 Pareto Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Search and evaluate the smallest useful set of early CompletionFormer encoder blocks promoted from W4A4 to W8A8, then report RMSE versus W8A8 cost on the fixed NYU protocol.

**Architecture:** Extend the existing hardware-aligned instrumentor with strict per-module weight-bit overrides. Add a focused module for the nine-unit CompletionFormer contract, runtime Conv/Linear cost accounting, cost-aware greedy selection, and Pareto analysis. Integrate a dedicated completionformer_front_pareto mode into the existing runner so model loading, calibration, joint Attention/concat, propagation, metrics, and prediction export continue through the validated path.

**Tech Stack:** Python 3.11, PyTorch 2.7/CUDA 11.8, NumPy, Matplotlib, CSV/JSON, pytest/unittest, official CompletionFormer modulated DCN.

---

### Task 1: Add Strict Per-Module Weight Bits

**Files:**
- Modify: scripts/hardware_aligned_quantization.py
- Modify: scripts/run_nyu_rtn_quantization.py
- Test: tests/test_hardware_aligned_quantization.py
- Test: tests/test_run_nyu_rtn_quantization.py

- [ ] **Step 1: Write failing instrumentor tests**

Add tests proving one selected Conv uses W8 while another remains W4, unknown
module overrides fail, invalid bits fail, and INT32 bias uses the selected
module's weight scale.

~~~python
def test_selected_module_uses_w8_while_other_module_remains_w4(self):
    model, instrumentor, sample = self._calibrated_model()
    instrumentor.configure(
        4, 4, {"encoder"}, weight_bit_overrides={"0": 8})
    bits = instrumentor.weight_bits_by_module()
    self.assertEqual(bits["0"], 8)
    self.assertEqual(bits["2"], 4)
    model(sample)
    instrumentor.close()

def test_unknown_weight_bit_override_fails(self):
    model, instrumentor, _ = self._calibrated_model()
    with self.assertRaisesRegex(ValueError, "unknown weight bit overrides"):
        instrumentor.configure(
            4, 4, {"encoder"}, weight_bit_overrides={"missing": 8})
    instrumentor.close()
~~~

- [ ] **Step 2: Verify RED**

Run:

~~~bash
PYTHONPATH=. pytest -q tests/test_hardware_aligned_quantization.py   -k 'weight_bit_override or uses_w8_while_other'
~~~

Expected: fail because configure does not accept weight_bit_overrides and
weight_bits_by_module does not exist.

- [ ] **Step 3: Implement per-module weight bits**

Add weight_bit_overrides=None to configure. Validate keys against self.modules,
require bits in (4, 8), and resolve one bit width per enabled module.

~~~python
unknown = set(weight_bit_overrides) - set(self.modules)
if unknown:
    raise ValueError("unknown weight bit overrides: %s" % sorted(unknown))
self.weight_bits = {}
for name, module in self.modules.items():
    if self.groups[name] not in self.enabled_groups:
        continue
    bits = int(weight_bit_overrides[name])         if name in weight_bit_overrides else self.w_bits
    if bits not in (4, 8):
        raise ValueError(
            "weight bits must be 4 or 8: %s=%d" % (name, bits))
    self.weight_bits[name] = bits
~~~

Use the resolved bits in weight QDQ and compensation re-quantization. Bias
continues to use input_scale times the selected weight_scale. Expose a copy
through weight_bits_by_module(). Add weight_bit_overrides to
instrumentor_options. New code uses direct required dictionary access and
contains no hidden fallback behavior.

- [ ] **Step 4: Verify GREEN and regression**

~~~bash
PYTHONPATH=. pytest -q tests/test_hardware_aligned_quantization.py   tests/test_run_nyu_rtn_quantization.py
~~~

Expected: all tests pass.

- [ ] **Step 5: Commit**

~~~bash
git add scripts/hardware_aligned_quantization.py   scripts/run_nyu_rtn_quantization.py   tests/test_hardware_aligned_quantization.py   tests/test_run_nyu_rtn_quantization.py
git commit -m "feat: support per-module weight bit allocation"
~~~

### Task 2: Resolve The Nine Atomic Front-Encoder Units

**Files:**
- Create: spn_quant/completionformer_front_encoder.py
- Create: tests/test_completionformer_front_encoder.py

- [ ] **Step 1: Write failing unit-contract tests**

Build a fake official-name manifest and assert exact unit order, module
membership, activation owners, no overlaps, and strict failure for a missing
downsample or unexpected matching module.

~~~python
def test_resolve_units_preserves_official_atomic_order():
    units = resolve_front_encoder_units(
        official_weight_names(), official_activation_names())
    assert list(units) == [
        "Stem", "Embed1.0", "Embed1.1", "Embed1.2",
        "Embed2.0", "Embed2.1", "Embed2.2", "Embed2.3",
        "PatchEmbed1",
    ]
    assert units["Stem"]["weight_modules"] == (
        "backbone.conv1_rgb.0",
        "backbone.conv1_dep.0",
        "backbone.conv1.0",
    )
    assert "backbone.former.embed_layer2.0.downsample.0" in         units["Embed2.0"]["weight_modules"]
    assert units["PatchEmbed1"]["activation_owners"] == (
        "backbone.former.patch_embed1.proj",
        "backbone.former.patch_embed1.norm",
    )
~~~

- [ ] **Step 2: Verify RED**

~~~bash
PYTHONPATH=. pytest -q tests/test_completionformer_front_encoder.py -k resolve
~~~

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement exact discovery and promotion overrides**

Define FRONT_ENCODER_UNIT_ORDER and exact expected module sets. Implement:

~~~python
def resolve_front_encoder_units(weight_modules, activation_owners):
    ...

def promotion_overrides(units, selected_units):
    return {
        "weight_bit_overrides": weight_overrides,
        "activation_bit_overrides": activation_overrides,
    }

def unit_manifest_rows(units):
    ...
~~~

Require every selected unit to be known and preserve official order. Every
declared weight and activation owner must exist. Reject a weight module owned
by multiple units. Every promoted override is exactly 8 bits.

- [ ] **Step 4: Verify GREEN**

~~~bash
PYTHONPATH=. pytest -q tests/test_completionformer_front_encoder.py   -k 'resolve or promotion'
~~~

Expected: pass.

- [ ] **Step 5: Commit**

~~~bash
git add spn_quant/completionformer_front_encoder.py   tests/test_completionformer_front_encoder.py
git commit -m "feat: define CompletionFormer front encoder units"
~~~

### Task 3: Add Runtime Cost Accounting

**Files:**
- Modify: spn_quant/completionformer_front_encoder.py
- Modify: tests/test_completionformer_front_encoder.py

- [ ] **Step 1: Write failing Conv/Linear MAC tests**

Use a toy model containing grouped Conv2d, ConvTranspose2d, repeated Linear,
and an unused module. Assert invocation counts, exact MACs, parameters, unit
aggregation, whole-model shares, and front-encoder shares.

~~~python
def test_profile_quantized_costs_counts_runtime_shapes_and_invocations():
    rows = profile_quantized_costs(
        model, (sample,), quantized_modules, module_to_unit)
    by_name = {row["module"]: row for row in rows}
    assert by_name["conv"]["macs"] == 1 * 4 * 4 * 6 * 2 * 3 * 3
    assert by_name["linear"]["invocations"] == 2
    assert by_name["unused"]["macs"] == 0
~~~

- [ ] **Step 2: Verify RED**

~~~bash
PYTHONPATH=. pytest -q tests/test_completionformer_front_encoder.py -k cost
~~~

Expected: fail because cost functions are missing.

- [ ] **Step 3: Implement profiling and aggregation**

Implement forward hooks for only supplied ordinary quantized modules. Remove
all hooks after the forward. Use output shapes for Conv/ConvTranspose and
output vector count for Linear.

~~~python
def profile_quantized_costs(model, model_args, quantized_modules,
                            module_to_unit):
    ...

def aggregate_unit_costs(cost_rows, unit_order):
    ...

def configuration_cost(unit_costs, selected_units, totals):
    ...
~~~

Return integer macs, parameters, operators and floating whole-model and
front-encoder shares. The whole-model parameter/operator denominator includes
declared ordinary quantized modules even if one preparation forward does not
invoke them. Custom QK/AV and propagation MACs are excluded and labeled.

- [ ] **Step 4: Verify GREEN**

~~~bash
PYTHONPATH=. pytest -q tests/test_completionformer_front_encoder.py -k cost
~~~

Expected: pass.

- [ ] **Step 5: Commit**

~~~bash
git add spn_quant/completionformer_front_encoder.py   tests/test_completionformer_front_encoder.py
git commit -m "feat: profile W8A8 layer allocation cost"
~~~

### Task 4: Implement Greedy And Pareto Algorithms

**Files:**
- Modify: spn_quant/completionformer_front_encoder.py
- Modify: tests/test_completionformer_front_encoder.py

- [ ] **Step 1: Write failing algorithm tests**

Test gain-per-MAC selection, all-negative selection, tie-breaking by MAC and
official order, strict prefixes, duplicate removal, Pareto dominance, and
normalized knee selection.

~~~python
def test_greedy_selects_largest_rmse_gain_per_incremental_mac():
    rmse = {
        (): 1.5,
        ("Stem",): 0.9,
        ("Embed1.0",): 1.1,
    }
    rows, winners = greedy_search(
        ("Stem", "Embed1.0"),
        {"Stem": 30, "Embed1.0": 10},
        100,
        lambda units: rmse[tuple(units)])
    assert winners[1]["selected_units"] == ("Embed1.0",)

def test_pareto_front_rejects_equal_cost_worse_rmse():
    frontier = pareto_front(rows, "mac_share", "mean_rmse")
    assert [row["name"] for row in frontier] == [
        "base", "efficient", "best"]
~~~

- [ ] **Step 2: Verify RED**

~~~bash
PYTHONPATH=. pytest -q tests/test_completionformer_front_encoder.py   -k 'greedy or prefix or pareto or knee'
~~~

Expected: fail because algorithms are missing.

- [ ] **Step 3: Implement deterministic algorithms**

Implement greedy_search, strict_prefix_sets, deduplicate_sets, pareto_front,
and pareto_knee. Candidate rows record round, candidate, selected set, mean
RMSE, delta, incremental MAC share, score, and winner. When no candidate has
positive gain, choose minimum RMSE, then lower MAC, then official order. Read
all required row fields directly.

- [ ] **Step 4: Verify GREEN**

~~~bash
PYTHONPATH=. pytest -q tests/test_completionformer_front_encoder.py
~~~

Expected: pass.

- [ ] **Step 5: Commit**

~~~bash
git add spn_quant/completionformer_front_encoder.py   tests/test_completionformer_front_encoder.py
git commit -m "feat: search front encoder W8A8 Pareto sets"
~~~

### Task 5: Integrate Search Into The Official Joint Runner

**Files:**
- Modify: scripts/run_nyu_rtn_quantization.py
- Modify: tests/test_run_nyu_rtn_quantization.py

- [ ] **Step 1: Write failing backend and index tests**

Add tests for the backend, deterministic 32 search indices disjoint from
calibration, W8A8 override construction, final-set de-duplication, and rejection
of append/manual config filters.

~~~python
def test_front_search_indices_are_deterministic_and_disjoint():
    first = runner.select_disjoint_training_indices(
        100, 32, 20260810, tuple(range(16)))
    second = runner.select_disjoint_training_indices(
        100, 32, 20260810, tuple(range(16)))
    self.assertEqual(first, second)
    self.assertFalse(set(first) & set(range(16)))

def test_front_pareto_config_promotes_selected_units_only():
    config = runner.build_front_pareto_config(
        "FE_G01_Stem", ("Stem",), groups, units)
    self.assertEqual(
        config["weight_bit_overrides"]["backbone.conv1.0"], 8)
    self.assertNotIn(
        "backbone.former.embed_layer1.0.conv1",
        config["weight_bit_overrides"])
~~~

- [ ] **Step 2: Verify RED**

~~~bash
PYTHONPATH=. pytest -q tests/test_run_nyu_rtn_quantization.py   -k front_pareto
~~~

Expected: fail because the backend and helpers are missing.

- [ ] **Step 3: Add backend and shared configuration**

Add completionformer_front_pareto to backend choices, model validation, and
propagation-adapter selection. Extract repeated configuration into:

~~~python
def configure_quantized_model(config, instrumentor, adapter,
                              joint_adapter, propagation_outputs,
                              model_name):
    ...
~~~

Both final evaluation and search RMSE evaluation call this helper. Front configs
start from Joint W4A4 and add only explicit W8A8 weight/activation overrides.

- [ ] **Step 4: Add isolated search records**

Implement direct helpers:

~~~python
def select_disjoint_training_indices(length, count, seed,
                                     excluded_indices):
    ...

def capture_rmse_records(model, saved_args, dataset, indices, device, seed):
    ...

def evaluate_mean_rmse(model, saved_args, records, device, config,
                       instrumentor, adapter, joint_adapter,
                       propagation_outputs):
    ...
~~~

Use seed 20260810 and exactly 32 training indices. Reject insufficient remaining
samples. Search records retain only sample, GT, and index and collect no layer
or propagation diagnostics.

- [ ] **Step 5: Run greedy and prefix paths**

After joint calibration and runtime cost profiling, resolve units, evaluate all
greedy additions on search records, generate official-order prefixes, and
deduplicate final configurations. Reject append, config-names, and manual
prediction config lists for this data-dependent backend.

Persist:
- front_encoder_units.csv
- front_encoder_costs.csv
- front_encoder_search.csv
- front_encoder_final_configs.csv

- [ ] **Step 6: Evaluate fixed 64 and export selected predictions**

Evaluate FP32, JIQ_Joint_W4A4, JIQ_W4A8, greedy winners, and strict prefixes on
the existing fixed validation indices. Persist per-sample metrics plus
front_encoder_final_aggregate.csv and front_encoder_pareto.csv.

After aggregate selection, export predictions only for FP32, W4A4, W4A8, the
knee, and the lowest-RMSE front set. Re-run only selected quantized
configurations for payload export and do not merge second-pass diagnostics into
metric tables.

- [ ] **Step 7: Add strict metadata and table validation**

Record search seed/count/indices, unit order, cost denominators, greedy winners,
prefixes, frontier, knee, and best config under front_encoder_w8a8_pareto.
Assert calibration/search disjointness, 64 unique finite final rows per config,
exact promoted bits, and 64 prediction payloads per exported config.

- [ ] **Step 8: Verify GREEN and regression**

~~~bash
PYTHONPATH=. pytest -q tests/test_run_nyu_rtn_quantization.py   tests/test_completionformer_front_encoder.py
~~~

Expected: pass.

- [ ] **Step 9: Commit**

~~~bash
git add scripts/run_nyu_rtn_quantization.py   tests/test_run_nyu_rtn_quantization.py   spn_quant/completionformer_front_encoder.py   tests/test_completionformer_front_encoder.py
git commit -m "feat: run CompletionFormer front encoder Pareto search"
~~~

### Task 6: Add Pareto And Prediction Figures

**Files:**
- Create: scripts/plot_completionformer_front_encoder_pareto.py
- Create: tests/test_plot_completionformer_front_encoder_pareto.py

- [ ] **Step 1: Write failing plot-contract tests**

Create complete temporary aggregate, Pareto, unit, and prediction payloads.
Assert missing configs/samples fail and these files are produced:

~~~text
rmse_vs_w8a8_mac_share.png
rmse_vs_w8a8_parameter_share.png
rmse_vs_w8a8_operator_share.png
prediction_comparison_64.png
~~~

Also assert non-dominated points and the knee are labeled without rotating x
tick labels.

- [ ] **Step 2: Verify RED**

~~~bash
PYTHONPATH=. pytest -q tests/test_plot_completionformer_front_encoder_pareto.py
~~~

Expected: import failure because the plotter does not exist.

- [ ] **Step 3: Implement strict plotting**

Read required CSV fields by direct indexing. Validate 64 unique sample IDs for
each prediction column and finite aggregate values. Use Arial-compatible font,
grid below artists, no title, non-rotated labels, and promoted-unit
annotations. Keep dominated points visible in gray and connect only the Pareto
frontier.

The prediction sheet uses GT, FP32, W4A4, W4A8, knee, and best columns with
shared depth and absolute-error ranges.

- [ ] **Step 4: Verify GREEN**

~~~bash
PYTHONPATH=. pytest -q tests/test_plot_completionformer_front_encoder_pareto.py
~~~

Expected: pass.

- [ ] **Step 5: Commit**

~~~bash
git add scripts/plot_completionformer_front_encoder_pareto.py   tests/test_plot_completionformer_front_encoder_pareto.py
git commit -m "feat: plot front encoder W8A8 Pareto results"
~~~

### Task 7: Add Reproducible Launcher And Documentation

**Files:**
- Create: scripts/run_completionformer_front_encoder_pareto.sh
- Modify: README.md

- [ ] **Step 1: Add the exact launcher**

Follow the joint launcher environment checks and use:

~~~text
backend: completionformer_front_pareto
calibration: 64 NYU train samples, seed 20260804
search: 32 disjoint NYU train samples, seed 20260810
evaluation: existing fixed 64 NYU validation samples
output: profile_logs/nyu_completionformer_front_encoder_w8a8_pareto_64
~~~

Require official source, checkpoint, reference sample metrics, and declared DCN
extension. Set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 only for the trusted official
legacy initialization file. Do not write to /tmp.

- [ ] **Step 2: Document semantics and outputs**

Explain atomic blocks, whole-model ordinary-op MAC denominator, search/evaluation
isolation, approximate greedy frontier, and that no retraining occurs.

- [ ] **Step 3: Verify shell and focused tests**

~~~bash
bash -n scripts/run_completionformer_front_encoder_pareto.sh
PYTHONPATH=. pytest -q tests/test_completionformer_front_encoder.py   tests/test_plot_completionformer_front_encoder_pareto.py
~~~

Expected: pass.

- [ ] **Step 4: Commit**

~~~bash
git add scripts/run_completionformer_front_encoder_pareto.sh README.md
git commit -m "docs: add front encoder Pareto experiment launcher"
~~~

### Task 8: Full Verification And Fixed Evaluation

**Files:**
- Generate only: profile_logs/nyu_completionformer_front_encoder_w8a8_pareto_64/
- Create: docs/2026-08-10-completionformer-front-encoder-w8a8-pareto-results.md

- [ ] **Step 1: Run complete tests and CUDA integer checks**

~~~bash
PYTHONPATH=. pytest -q
PYTHONPATH=. pytest -q tests/test_integer_ops.py -k cuda
~~~

Expected: all tests pass and both CUDA integer tests execute rather than skip.

- [ ] **Step 2: Run the fixed experiment**

~~~bash
export COMPLETIONFORMER_ROOT=/workspace/CompletionFormer
export COMPLETIONFORMER_DCN_PATH=/workspace/SPN_Quantization/profile_logs/runtime_extensions/completionformer_dcn_verified/lib
export COMPLETIONFORMER_DEVICE=cuda:0
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
bash scripts/run_completionformer_front_encoder_pareto.sh
~~~

Do not retrain.

- [ ] **Step 3: Audit outputs**

Verify calibration/search/evaluation counts and isolation, complete unit
membership, cost totals, every search candidate row, 64 finite rows per final
config, 64 payloads per exported config, exact bit ownership, source commit,
checkpoint SHA256, and no generated files tracked by Git.

- [ ] **Step 4: Generate and inspect figures**

~~~bash
PYTHONPATH=. python scripts/plot_completionformer_front_encoder_pareto.py   --root /workspace/SPN_Quantization/profile_logs/nyu_completionformer_front_encoder_w8a8_pareto_64/completionformer   --out-dir /workspace/SPN_Quantization/profile_logs/nyu_completionformer_front_encoder_w8a8_pareto_64/analysis   --expected-samples 64   --dpi 120
~~~

Inspect all four images for blank panels, overlaps, clipping, readable unit
annotations, correct frontier ordering, and populated predictions.

- [ ] **Step 5: Write measured results**

Document every final configuration's mean/median/p95 RMSE and W8A8
MAC/parameter/operator shares. State greedy order, prefix comparison,
non-dominated sets, knee, best set, improvement versus W4A4, gap to FP32/W4A8,
and whether a small W8A8 front set materially preserves accuracy.

- [ ] **Step 6: Final regression and commit**

~~~bash
PYTHONPATH=. pytest -q
git diff --check
git status --short
~~~

Commit only source, tests, launcher, README, and result document. Do not add
generated profile_logs.

~~~bash
git add docs/2026-08-10-completionformer-front-encoder-w8a8-pareto-results.md
git commit -m "docs: report front encoder W8A8 Pareto evaluation"
~~~

