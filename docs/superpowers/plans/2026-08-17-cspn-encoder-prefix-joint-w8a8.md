# CSPN Encoder-Prefix Joint W8A8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run the fixed 24-configuration CSPN encoder-prefix by sensitive-tail W4A4/W8A8 experiment on NYU.

**Architecture:** Add one pure candidate/analysis module that owns exact unit registries, Cartesian-product generation, cost, interaction, and Pareto logic. Add a separate runner that reuses the official checkpoint, stem controller, hardware-aligned Group-8 backend, fixed sample protocol, and decoder diagnostics without changing existing experiments. Add a CSV-only plotter and retain all measured outputs under a new immutable result root.

**Tech Stack:** Python 3.11, PyTorch CUDA, NumPy, Matplotlib, pytest, official CSPN ResNet-18 adapter.

---

### Task 1: Exact Prefix and Tail Candidate Model

**Files:**
- Create: `spn_quant/cspn_encoder_prefix.py`
- Create: `tests/test_cspn_encoder_prefix.py`

- [ ] **Step 1: Write failing registry tests**

Test the public API:

```python
registry = build_unit_registry(executed_modules, activation_owners)
```

Assert ordered units `stem`, `encoder_layer1` through `encoder_layer4`,
`decoder_layer4`, and `initial_depth`; exact official Conv coverage; exact
activation-owner coverage; `conv1_1` stem ownership; and shared
`rotation.layer4_signed_skip` membership in both stem and decoder4. Assert
missing, extra, and duplicate executed modules or owners raise errors.

- [ ] **Step 2: Run the registry tests and verify RED**

```bash
python -m pytest -q tests/test_cspn_encoder_prefix.py -k registry
```

Expected: import failure because `spn_quant.cspn_encoder_prefix` does not
exist.

- [ ] **Step 3: Implement immutable unit and candidate contracts**

Create:

```python
Owner = Tuple[str, str]

@dataclass(frozen=True)
class UnitRegistry:
    weights_by_unit: Mapping[str, Tuple[str, ...]]
    activations_by_unit: Mapping[str, Tuple[Owner, ...]]

@dataclass(frozen=True)
class PrefixTailCandidate:
    name: str
    prefix_index: int
    tail_index: int
    encoder_units: Tuple[str, ...]
    tail_units: Tuple[str, ...]
    weight_modules: Tuple[str, ...]
    activation_owners: Tuple[Owner, ...]
    stem_w8a8: bool
```

Declare the official ResNet-18 Conv names explicitly: four Conv modules in
layer1 and five each in layers2-4. Reuse the exact decoder4 and initial-depth
registries from `spn_quant.cspn_sensitivity`, and define stem cost ownership as
`conv1_1` plus `conv1_1::input`, root ReLU output, and retained skip4 boundary.
Use stable unions that preserve registry order while charging shared owners
once.

- [ ] **Step 4: Write failing matrix tests**

Assert `build_candidates(registry)` returns exactly 24 unique candidates in
prefix-major/tail-minor order. Assert P0/T0 has no promotions, P5/T3 contains
all five encoder units and both tails, every prefix is cumulative, and only
P1-P5 activate `stem_w8a8`.

- [ ] **Step 5: Implement Cartesian-product generation**

Generate six prefix tuples and four tail tuples. Build deterministic names
`PREFIX_P%d__TAIL_T%d`, ordered weight and activation unions, and explicit unit
lists. Reject duplicate registry entries rather than silently deduplicating
invalid source data.

- [ ] **Step 6: Write failing cost, interaction, and Pareto tests**

Cover:

```python
precision_cost(weight_rows, activation_rows, candidate)
interaction_rows(aggregate_rows)
pareto_rows(aggregate_rows, cost_field="normalized_added_bit_cost")
pareto_rows(aggregate_rows, cost_field="w8_weight_mac_fraction")
prediction_candidate_names(aggregate_rows, normalized_pareto_rows)
```

Use synthetic values that prove shared skip4 costs once, interaction follows
`R(P,T)-R(P,T0)-R(P0,T)+R(P0,T0)`, equal points remain deterministic, dominated
points are removed, and prediction selection contains strict, the cheapest
improving point, every normalized-cost Pareto point, and P5/T3 without
duplicates.

- [ ] **Step 7: Implement analysis functions and verify**

Require exact row coverage, unique names, finite RMSE/cost values, nonnegative
counts, and complete 6-by-4 interaction coverage. Do not use dictionary
fallback reads.

```bash
python -m pytest -q tests/test_cspn_encoder_prefix.py
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 1**

```bash
git add spn_quant/cspn_encoder_prefix.py tests/test_cspn_encoder_prefix.py
git commit -m "feat: define CSPN encoder prefix W8A8 search"
```

### Task 2: Strict Joint Evaluation Runner

**Files:**
- Create: `scripts/run_nyu_cspn_encoder_prefix_joint.py`
- Create: `tests/test_run_nyu_cspn_encoder_prefix_joint.py`

- [ ] **Step 1: Write failing hardware-translation tests**

Test that `hardware_configuration(candidate)` retains global static Group-8
W4A4 and A8/Q13 propagation, forwards only non-stem W8 weight overrides, and
promotes only generic/rotation activation owners. Assert
`stem_configuration(candidate)` returns exactly `STRICT_W4A4` for P0 and
`STEM_W8A8` for P1-P5. Assert `conv1_1` weight/input are absent from generic
ownership.

- [ ] **Step 2: Run translation tests and verify RED**

```bash
python -m pytest -q tests/test_run_nyu_cspn_encoder_prefix_joint.py -k "hardware or stem"
```

Expected: import failure because the runner does not exist.

- [ ] **Step 3: Implement configuration translation and exact validation**

Reuse `base._configuration`, `stem_runner._build_quantization_context`, and the
fixed propagation contract. Implement:

```python
def hardware_configuration(candidate): ...
def stem_configuration(candidate): ...
def validate_configured_precision(candidate, weight_bits, specs,
                                  rotation_specs, stem_contract): ...
```

Validate generic W8 modules and A8 owners separately from the stem controller,
then validate their union against the candidate. Require the stem contract to
report matching weight and activation bits.

- [ ] **Step 4: Write failing discovery and cost-basis tests**

Construct synthetic executed instrumentor/rotation objects and assert the
runner discovers only official encoder/decoder4/depth-head candidates. Assert
the operation basis includes `conv1_1` exactly once even though it is not a
generic instrumentor module. Assert the activation basis includes the stem
input and deduplicates the shared skip4 owner.

- [ ] **Step 5: Implement registry discovery and operation accounting**

Use observed executed modules and activation owners, then pass them into the
explicit registry validator. Extend the operation counter module list with
`conv1_1`; verify operation shapes and counts are identical across all
candidates. Reuse the existing per-sample activation count observers and add
the stem input count explicitly.

- [ ] **Step 6: Write failing aggregation and orchestration tests**

Assert exactly 64 unique sample rows per candidate, finite metrics, complete
24-candidate coverage, deterministic prefix/tail fields, exact paired wins,
complete interaction rows, and rejection of an existing output directory.
Use synthetic evaluator results to prove every candidate gets a fresh model and
calibration call before aggregate/Pareto construction.

- [ ] **Step 7: Implement one-candidate evaluation**

For one candidate, load a fresh checkpoint, reproduce fold metadata, calibrate
128 samples, configure and validate precision, evaluate 64 samples against the
shared FP32 model, and return sample/regional/block/propagation/operation/layer
rows. Close all captures, counters, adapters, and stem hooks before releasing
the model. The first pass must not write predictions.

- [ ] **Step 8: Implement 24-candidate orchestration and immutable outputs**

Build the registry from a calibrated strict context, run all candidates,
calculate cost and interactions, produce both Pareto tables, select prediction
configurations, and rerun only those configurations to write payloads. Require
prediction-rerun RMSE equality. Write:

```text
aggregate_metrics.csv
sample_metrics_64.csv
regional_metrics.csv
block_metrics.csv
propagation_metrics.csv
operation_counts.csv
layer_quantization_metrics.csv
activation_cost_basis.csv
precision_coverage.csv
interaction_metrics.csv
pareto_normalized_cost.csv
pareto_w8_mac.csv
manifest.json
```

Hash every artifact except `manifest.json` and store exact candidate contracts,
source/checkpoint/protocol hashes, sample identities, and fold metadata.

- [ ] **Step 9: Verify and commit Task 2**

```bash
python -m pytest -q tests/test_run_nyu_cspn_encoder_prefix_joint.py tests/test_cspn_encoder_prefix.py
git add scripts/run_nyu_cspn_encoder_prefix_joint.py tests/test_run_nyu_cspn_encoder_prefix_joint.py
git commit -m "feat: evaluate CSPN encoder prefix W8A8 combinations"
```

### Task 3: CSV-Only Plots

**Files:**
- Create: `scripts/plot_nyu_cspn_encoder_prefix_joint.py`
- Create: `tests/test_plot_nyu_cspn_encoder_prefix_joint.py`

- [ ] **Step 1: Write failing plot-input tests**

Test required columns, exact 24 aggregate rows, exact six prefixes and four
tails, unique cells, finite RMSE/interaction/cost values, valid Pareto
membership, missing-file rejection, and CLI imports from outside the repository
root.

- [ ] **Step 2: Run plot tests and verify RED**

```bash
python -m pytest -q tests/test_plot_nyu_cspn_encoder_prefix_joint.py
```

Expected: import failure because the plotter does not exist.

- [ ] **Step 3: Implement plotting**

Read only persisted CSVs and generate:

```text
encoder_prefix_tail_rmse_heatmap.png/.pdf
encoder_prefix_tail_interaction_heatmap.png/.pdf
encoder_prefix_normalized_cost_pareto.png/.pdf
encoder_prefix_w8_mac_pareto.png/.pdf
```

Use Arial, no title, unrotated tick labels, stable dimensions, grid below
Pareto points, explicit units on axes/color bars, and annotations short enough
to avoid overlap. Update manifest artifact hashes after all files are written.

- [ ] **Step 4: Verify and commit Task 3**

```bash
python -m pytest -q tests/test_plot_nyu_cspn_encoder_prefix_joint.py
git add scripts/plot_nyu_cspn_encoder_prefix_joint.py tests/test_plot_nyu_cspn_encoder_prefix_joint.py
git commit -m "feat: plot CSPN encoder prefix W8A8 Pareto results"
```

### Task 4: Full Verification and CUDA Evaluation

**Files:**
- Create: `docs/2026-08-17-cspn-encoder-prefix-joint-w8a8-results.md`
- Generate: `/workspace/SPN_Quantization/profile_logs/nyu_cspn_encoder_prefix_joint_w8a8_64/`

- [ ] **Step 1: Run focused and complete tests**

```bash
python -m pytest -q \
  tests/test_cspn_encoder_prefix.py \
  tests/test_run_nyu_cspn_encoder_prefix_joint.py \
  tests/test_plot_nyu_cspn_encoder_prefix_joint.py \
  tests/test_cspn_sensitivity.py \
  tests/test_run_nyu_cspn_decoder_sensitivity.py
python -m pytest -q
```

Expected: zero failures.

- [ ] **Step 2: Run the fixed CUDA experiment**

```bash
python scripts/run_nyu_cspn_encoder_prefix_joint.py \
  --run-dir /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24 \
  --checkpoint /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24/best.pt \
  --data-root /workspace/CSPN/cspn_pytorch \
  --calibration-indices /workspace/SPN_Quantization/profile_logs/nyu_cspn_stratified_calibration_128/calibration_indices.json \
  --calibration-metadata /workspace/SPN_Quantization/profile_logs/nyu_cspn_stratified_calibration_128/metadata.json \
  --evaluation-protocol /workspace/SPN_Quantization/profile_logs/nyu_cspn_activation_resolution/cspn/metadata.json \
  --out-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_encoder_prefix_joint_w8a8_64 \
  --device cuda:0 \
  --seed 20260812 \
  --fold-max-error 0.05
```

Expected: 24 completed configurations followed by deterministic selected
prediction reruns and an exit code of zero.

- [ ] **Step 3: Generate plots and audit artifacts**

```bash
python scripts/plot_nyu_cspn_encoder_prefix_joint.py \
  --input-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_encoder_prefix_joint_w8a8_64
```

Audit 1,536 sample rows, 64 unique identities per configuration, finite
predictions and metrics, complete 6-by-4 matrices, exact configured precision,
both Pareto sets, selected prediction coverage, and every manifest hash.
Visually inspect all four PNGs for font, clipping, overlap, and unrotated ticks.

- [ ] **Step 4: Write the measured report**

Document the prefix/tail RMSE table, minimum-cost Pareto points, W8 MAC shares,
block-error flow, strongest positive/negative interactions, iRMSE/minimum-depth
risks, and whether decoder4/initial-depth retention preserves early encoder
recovery. Do not describe logical cost as measured latency.

- [ ] **Step 5: Final verification and commit**

```bash
git diff --check
python -m pytest -q
git add docs/2026-08-17-cspn-encoder-prefix-joint-w8a8-results.md
git commit -m "docs: report CSPN encoder prefix W8A8 results"
```

Expected: clean worktree and zero test failures.
