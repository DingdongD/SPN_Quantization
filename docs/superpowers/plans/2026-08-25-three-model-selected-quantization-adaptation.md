# Three-Model Selected Quantization Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt the retained RTN, QDrop, BRECQ, HAWQ, LSQ++, model-specific P3/T3 PTQ, and mixed task-aware QAT protocols to official DySPN, NLSPN, and CompletionFormer models.

**Architecture:** Introduce a strict model quantization contract that maps semantic blocks, activation owners, reconstruction boundaries, protected propagation signals, and mixed-precision search groups. Reuse the existing method implementations and official model loaders through this contract, retaining CSPN compatibility wrappers while adding one multi-model training, search, evaluation, and launch surface.

**Tech Stack:** Python 3, PyTorch, official DySPN/NLSPN/CompletionFormer CUDA extensions, NumPy, SciPy MILP, pytest, Matplotlib.

**Spec:** `docs/superpowers/specs/2026-08-25-three-model-selected-quantization-adaptation-design.md`

## Global Constraints

- Target only official full DySPN, NLSPN, and CompletionFormer architectures and their existing converged checkpoints.
- Selected methods are RTN W8A8, RTN W4A4, QDrop W6A6, BRECQ W6A6, HAWQ mixed <=6, LSQ++ W6A6/W4A4, mixed task-aware QAT, and model-specific P3/T3 mixed PTQ.
- Use 128 train-only stratified calibration samples and a fixed 64-sample validation set per model.
- Report pooled RMSE as the primary metric and label mean per-sample RMSE separately.
- Preserve official propagation iterations, CUDA extensions, preprocessing, tensor shapes, and checkpoints.
- Guidance, confidence, offsets, affinity, anchors, normalization, and propagation state follow model-specific protected contracts.
- Do not add backend, model, precision, operator, checkpoint, or artifact fallback behavior.
- Configuration fields use attribute access and dictionaries use indexed access so missing fields fail naturally.
- Avoid broad `try`/`except` blocks and do not write intermediate files under `/tmp`.

---

### Task 1: Strict Model Quantization Contracts

**Files:**
- Create: `spn_quant/model_contracts.py`
- Modify: `spn_quant/adapters/base.py`
- Modify: `spn_quant/adapters/dyspn.py`
- Modify: `spn_quant/adapters/nlspn.py`
- Modify: `spn_quant/adapters/completionformer.py`
- Test: `tests/test_model_quantization_contracts.py`

**Interfaces:**
- Consumes: `ModelSemanticAdapter.module_manifest()` and official model `named_modules()`.
- Produces: `QuantizationModelContract`, `QuantizationBlock`, `SearchTopology`, and `build_model_quantization_contract(model_name, model)`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_dyspn_contract_protects_dcn_and_propagation_signals():
    contract = build_model_quantization_contract("dyspn", dyspn_model)
    assert "offset" in contract.protected_roles
    assert "affinity" in contract.protected_roles
    assert all("conv_offset_aff" not in name for name in contract.weight_modules)


def test_completionformer_contract_has_attention_and_concat_edges():
    contract = build_model_quantization_contract(
        "completionformer", completionformer_model)
    assert contract.attention_edges
    assert contract.concat_edges
```

- [ ] **Step 2: Run the tests and verify missing contract failures**

Run: `pytest -q tests/test_model_quantization_contracts.py`

Expected: collection or import failure for `spn_quant.model_contracts`.

- [ ] **Step 3: Implement immutable contract records and strict validation**

```python
@dataclass(frozen=True)
class QuantizationBlock:
    name: str
    weight_modules: Tuple[str, ...]
    activation_owners: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class QuantizationModelContract:
    model_name: str
    blocks: Tuple[QuantizationBlock, ...]
    prefix_groups: Tuple[Tuple[str, ...], ...]
    tail_groups: Tuple[Tuple[str, ...], ...]
    protected_roles: Tuple[str, ...]
    attention_edges: Tuple[str, ...]
    concat_edges: Tuple[str, ...]
```

Build each contract from explicit module patterns and fail when any required
block is empty, any module appears in two weight blocks, any activation owner
appears twice, or any protected semantic role is assigned to a generic block.

- [ ] **Step 4: Run contract and semantic-adapter tests**

Run: `pytest -q tests/test_model_quantization_contracts.py tests/test_model_semantic_adapters.py tests/test_propagation_aware_adapters.py`

Expected: all pass.

- [ ] **Step 5: Commit the contract layer**

```bash
git add spn_quant/model_contracts.py spn_quant/adapters tests/test_model_quantization_contracts.py
git commit -m "feat: add strict model quantization contracts"
```

### Task 2: Shared Experiment Configuration and Model Runtime

**Files:**
- Create: `spn_quant/experiment_config.py`
- Create: `scripts/nyu_model_runtime.py`
- Create: `configs/three_model_selected_quantization.json`
- Test: `tests/test_experiment_config.py`
- Test: `tests/test_nyu_model_runtime.py`

**Interfaces:**
- Consumes: existing `scripts.train_nyu_iteration_sweep` builders and run directories.
- Produces: `SelectedQuantizationConfig`, `ModelExperimentConfig`, `load_selected_quantization_config(path)`, and `NYUModelRuntime`.

- [ ] **Step 1: Write failing strict-config tests**

```python
def test_config_requires_all_three_models(tmp_path):
    path = write_config_without_completionformer(tmp_path)
    with pytest.raises(KeyError):
        load_selected_quantization_config(path)


def test_runtime_rejects_model_checkpoint_mismatch(runtime_args):
    with pytest.raises(ValueError, match="checkpoint model"):
        NYUModelRuntime.from_args(runtime_args)
```

- [ ] **Step 2: Verify the tests fail**

Run: `pytest -q tests/test_experiment_config.py tests/test_nyu_model_runtime.py`

Expected: imports fail because the modules do not exist.

- [ ] **Step 3: Implement strict dataclass parsing**

Parse JSON with indexed dictionary access and construct frozen dataclasses.
Require explicit fields for run directory, checkpoint, Python executable,
device, propagation iterations, calibration metadata, calibration count,
evaluation indices, all method hyperparameters, and output root.

- [ ] **Step 4: Implement the official runtime facade**

```python
class NYUModelRuntime:
    def build_model(self, device: torch.device) -> nn.Module: ...
    def build_dataset(self, split: str): ...
    def model_input(self, sample, device: torch.device): ...
    def prediction(self, output) -> torch.Tensor: ...
    def close(self) -> None: ...
```

Use the existing official loader functions. Assert the saved model name,
checkpoint iteration, expected architecture class, and required CUDA extension.

- [ ] **Step 5: Run config and runtime tests**

Run: `pytest -q tests/test_experiment_config.py tests/test_nyu_model_runtime.py tests/test_run_nyu_rtn_quantization.py`

Expected: all pass.

- [ ] **Step 6: Commit the shared runtime**

```bash
git add spn_quant/experiment_config.py scripts/nyu_model_runtime.py configs/three_model_selected_quantization.json tests/test_experiment_config.py tests/test_nyu_model_runtime.py
git commit -m "feat: add strict multi-model experiment runtime"
```

### Task 3: Generic Allocation Registry and P3/T3 Search

**Files:**
- Create: `spn_quant/mixed_precision.py`
- Create: `scripts/run_nyu_model_p3t3_search.py`
- Modify: `spn_quant/cspn_task_sensitive_bits.py`
- Test: `tests/test_mixed_precision.py`
- Test: `tests/test_run_nyu_model_p3t3_search.py`

**Interfaces:**
- Consumes: `QuantizationModelContract` and hardware-aligned layer cost rows.
- Produces: `AllocationRegistry`, `BitAssignment`, `P3T3SearchResult`, `build_registry(contract, costs)`, and per-model `p3_t3_assignment.json`.

- [ ] **Step 1: Write failing generic registry tests**

```python
def test_registry_uses_contract_blocks_without_cspn_names(contract, costs):
    registry = build_registry(contract, costs)
    assert registry.blocks == tuple(block.name for block in contract.blocks)


def test_p3t3_search_selects_model_specific_prefix_and_tail(search_inputs):
    result = search_p3_t3(**search_inputs)
    assert result.assignment.model_name == search_inputs["contract"].model_name
    assert result.prefix in search_inputs["contract"].prefix_groups
    assert result.tail in search_inputs["contract"].tail_groups
```

- [ ] **Step 2: Run and verify failures**

Run: `pytest -q tests/test_mixed_precision.py tests/test_run_nyu_model_p3t3_search.py`

Expected: missing generic allocation module and runner.

- [ ] **Step 3: Move architecture-neutral allocation logic**

Move the generic data records, budget audits, dominance pruning, sensitivity
ranking, and assignment search from `cspn_task_sensitive_bits.py` into
`mixed_precision.py`. Keep imports and aliases in the CSPN module so existing
CSPN tests and serialized assignments remain valid.

- [ ] **Step 4: Implement the model-specific search runner**

Evaluate uniform W4A4, all single-block W8A8 promotions, cumulative prefixes,
tail combinations, and prefix-tail interactions. Persist pooled RMSE,
per-sample RMSE, normalized weight/activation cost, validity, and paired sample
differences for every candidate. Select the smallest stable Pareto knee as P3
and the best budget-valid tail as T3.

- [ ] **Step 5: Run generic and CSPN regression tests**

Run: `pytest -q tests/test_mixed_precision.py tests/test_run_nyu_model_p3t3_search.py tests/test_cspn_task_sensitive_bits.py tests/test_run_nyu_cspn_task_sensitive_bits.py`

Expected: all pass.

- [ ] **Step 6: Commit generic mixed precision**

```bash
git add spn_quant/mixed_precision.py spn_quant/cspn_task_sensitive_bits.py scripts/run_nyu_model_p3t3_search.py tests/test_mixed_precision.py tests/test_run_nyu_model_p3t3_search.py
git commit -m "feat: generalize model-specific P3 T3 search"
```

### Task 4: Selected PTQ Matrix Runner

**Files:**
- Create: `scripts/run_nyu_selected_ptq.py`
- Modify: `scripts/run_nyu_rtn_quantization.py`
- Modify: `scripts/run_nyu_qdrop_reconstruction.py`
- Test: `tests/test_run_nyu_selected_ptq.py`
- Test: `tests/test_qdrop_contract.py`

**Interfaces:**
- Consumes: `NYUModelRuntime`, `QuantizationModelContract`, and P3/T3 assignment files.
- Produces: RTN W8A8/W4A4, QDrop W6A6, BRECQ W6A6, and P3/T3 hard-deployment artifacts.

- [ ] **Step 1: Write failing method-matrix tests**

```python
def test_selected_ptq_matrix_is_exact():
    assert selected_ptq_methods() == (
        "rtn_w8a8", "rtn_w4a4", "qdrop_w6a6",
        "brecq_w6a6", "p3_t3_mixed_ptq")


def test_reconstruction_excludes_protected_contract_roles(contract, plan):
    protected = set(contract.protected_modules)
    assert protected.isdisjoint(plan.module_names)
```

- [ ] **Step 2: Run and verify failures**

Run: `pytest -q tests/test_run_nyu_selected_ptq.py tests/test_qdrop_contract.py`

Expected: missing selected PTQ runner or contract arguments.

- [ ] **Step 3: Add explicit precision and contract inputs to existing runners**

Do not duplicate RTN or reconstruction algorithms. Add a contract-driven path
that supplies quantizable modules, protected outputs, attention/concat owners,
and exact W/A bits. Remove no existing CSPN command-line behavior.

- [ ] **Step 4: Implement selected PTQ orchestration**

The runner dispatches only the five selected PTQ configurations, validates the
hard deployment manifest, and rejects incomplete reconstruction checkpoints.
QDrop and BRECQ use W6A6 and share calibration identities but write separate
optimization states.

- [ ] **Step 5: Run PTQ tests**

Run: `pytest -q tests/test_run_nyu_selected_ptq.py tests/test_run_nyu_rtn_quantization.py tests/test_qdrop_reconstruction_runner.py tests/test_qdrop_contract.py tests/test_completionformer_joint_adapter.py`

Expected: all pass.

- [ ] **Step 6: Commit selected PTQ support**

```bash
git add scripts/run_nyu_selected_ptq.py scripts/run_nyu_rtn_quantization.py scripts/run_nyu_qdrop_reconstruction.py tests/test_run_nyu_selected_ptq.py tests/test_qdrop_contract.py
git commit -m "feat: add selected PTQ matrix for official SPN models"
```

### Task 5: Generic HAWQ Allocation and Trace

**Files:**
- Create: `scripts/run_nyu_model_hawq_trace.py`
- Modify: `spn_quant/hawq_trace.py`
- Modify: `spn_quant/hawq_allocation.py`
- Test: `tests/test_run_nyu_model_hawq_trace.py`
- Test: `tests/test_hawq_allocation.py`

**Interfaces:**
- Consumes: model contract blocks, fixed calibration batches, and separate weight/activation cost tables.
- Produces: `hawq_mixed_le6_assignment.json` with average weight and activation bits no greater than 6.

- [ ] **Step 1: Write failing contract-driven HAWQ tests**

```python
def test_hawq_trace_uses_only_contract_blocks(contract, traces):
    assert tuple(row.block for row in traces) == contract.block_names


def test_hawq_assignment_meets_both_bit_budgets(assignment):
    assert assignment.average_weight_bits <= 6.0
    assert assignment.average_activation_bits <= 6.0
```

- [ ] **Step 2: Verify failures**

Run: `pytest -q tests/test_run_nyu_model_hawq_trace.py tests/test_hawq_allocation.py`

Expected: the trace runner is CSPN-specific.

- [ ] **Step 3: Generalize trace block construction**

Build trace blocks from `QuantizationModelContract.blocks`. Compute masked depth
and boundary curvature using `NYUModelRuntime.prediction()`. Reject protected
modules and non-finite Hessian-vector products.

- [ ] **Step 4: Enforce separate <=6 budgets**

Retain the MILP solver but provide independent weight parameter/MAC costs and
activation traffic costs from the contract. Persist objective components and
constraint residuals.

- [ ] **Step 5: Run HAWQ tests**

Run: `pytest -q tests/test_run_nyu_model_hawq_trace.py tests/test_hawq_trace.py tests/test_hawq_allocation.py tests/test_run_nyu_cspn_hawq_trace.py`

Expected: all pass, including CSPN regression.

- [ ] **Step 6: Commit generic HAWQ**

```bash
git add scripts/run_nyu_model_hawq_trace.py spn_quant/hawq_trace.py spn_quant/hawq_allocation.py tests/test_run_nyu_model_hawq_trace.py tests/test_hawq_allocation.py
git commit -m "feat: generalize HAWQ allocation across SPN models"
```

### Task 6: Multi-Model LSQ++ and Mixed Task-Aware QAT

**Files:**
- Create: `spn_quant/qat/model_methods.py`
- Create: `spn_quant/qat/task_loss.py`
- Create: `scripts/train_nyu_selected_qat.py`
- Modify: `spn_quant/qat/cspn_methods.py`
- Modify: `spn_quant/qat/cspn_task_loss.py`
- Test: `tests/test_model_method_qat.py`
- Test: `tests/test_model_task_loss.py`
- Test: `tests/test_train_nyu_selected_qat.py`

**Interfaces:**
- Consumes: strict runtime, model contract, LSQ++/HAWQ/P3T3 assignment, and FP32 teacher.
- Produces: LSQ++ W4A4/W6A6, HAWQ mixed <=6, and mixed task-aware QAT checkpoints.

- [ ] **Step 1: Write failing controller and loss tests**

```python
def test_model_qat_controller_owns_each_activation_once(controller):
    owners = controller.activation_owner_manifest()
    assert len(owners) == len(set(owners))


def test_task_loss_aligns_model_specific_propagation_states(runtime, batch):
    loss = model_task_aware_loss(runtime, batch)
    assert torch.isfinite(loss.total)
    assert loss.propagation.ndim == 0
```

- [ ] **Step 2: Verify failures**

Run: `pytest -q tests/test_model_method_qat.py tests/test_model_task_loss.py tests/test_train_nyu_selected_qat.py`

Expected: missing generic QAT modules and runner.

- [ ] **Step 3: Extract the architecture-neutral QAT controller**

Move LSQ++ quantizer attachment, hard activation configuration, assignment
loading, owner manifests, and checkpoint method state into
`ModelMethodQATController`. Keep `CSPNMethodQATController` as a compatibility
wrapper around the generic controller.

- [ ] **Step 4: Implement model-aware task loss**

Compute masked depth, boundary, detached FP32 teacher, initial-depth, and
propagation-state consistency. Obtain semantic tensors from each model adapter;
do not index model-specific output tuples in the loss implementation.

- [ ] **Step 5: Implement the selected QAT runner**

Accept exactly `lsqplus_w4a4`, `lsqplus_w6a6`, `hawq_mixed_le6`, and
`mixed_task_aware`. Use canonical FP32 master weights, explicit optimizer and
scheduler state, deterministic resume metadata, and hard-deployment validation
after every evaluation epoch.

- [ ] **Step 6: Run QAT and CSPN regression tests**

Run: `pytest -q tests/test_model_method_qat.py tests/test_model_task_loss.py tests/test_train_nyu_selected_qat.py tests/test_lsqplus_quantizers.py tests/test_hawq_quantizers.py tests/test_train_nyu_cspn_lsqplus_hawq.py`

Expected: all pass.

- [ ] **Step 7: Commit multi-model QAT**

```bash
git add spn_quant/qat/model_methods.py spn_quant/qat/task_loss.py scripts/train_nyu_selected_qat.py spn_quant/qat/cspn_methods.py spn_quant/qat/cspn_task_loss.py tests/test_model_method_qat.py tests/test_model_task_loss.py tests/test_train_nyu_selected_qat.py
git commit -m "feat: adapt LSQ and task aware QAT to SPN models"
```

### Task 7: Unified Formal Evaluation and Prediction Exports

**Files:**
- Create: `scripts/evaluate_nyu_selected_quantization.py`
- Create: `scripts/plot_nyu_selected_quantization.py`
- Modify: `scripts/evaluate_nyu_cspn_lsqplus_hawq.py`
- Test: `tests/test_evaluate_nyu_selected_quantization.py`
- Test: `tests/test_plot_nyu_selected_quantization.py`

**Interfaces:**
- Consumes: completed PTQ/QAT artifacts and fixed per-model evaluation indices.
- Produces: aggregate metrics, relative-loss table, cost table, diagnostics, and aligned prediction panels.

- [ ] **Step 1: Write failing pooled-metric and artifact tests**

```python
def test_pooled_rmse_uses_global_squared_error_sum(records):
    result = aggregate_predictions(records)
    assert result.pooled_rmse == pytest.approx(
        np.sqrt(result.squared_error_sum / result.valid_pixel_count))


def test_summary_rejects_missing_selected_method(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_method_summary(tmp_path, SELECTED_METHODS)
```

- [ ] **Step 2: Verify failures**

Run: `pytest -q tests/test_evaluate_nyu_selected_quantization.py tests/test_plot_nyu_selected_quantization.py`

Expected: missing multi-model evaluator and plotter.

- [ ] **Step 3: Implement strict method preparation and aggregation**

Prepare every method through its hard deployment contract. Accumulate global
squared error and valid pixels for pooled RMSE, and separately average sample
RMSE. Record FP32-relative absolute and percentage degradation.

- [ ] **Step 4: Implement aligned prediction export**

Render fixed identities with RGB, sparse depth, GT, FP32, RTN W8A8, RTN W4A4,
QDrop W6A6, BRECQ W6A6, HAWQ, LSQ++ W6A6/W4A4, mixed-QAT, and P3/T3. Use shared
depth and error ranges within each sample.

- [ ] **Step 5: Run evaluation tests**

Run: `pytest -q tests/test_evaluate_nyu_selected_quantization.py tests/test_plot_nyu_selected_quantization.py tests/test_evaluate_nyu_cspn_lsqplus_hawq.py`

Expected: all pass.

- [ ] **Step 6: Commit unified evaluation**

```bash
git add scripts/evaluate_nyu_selected_quantization.py scripts/plot_nyu_selected_quantization.py scripts/evaluate_nyu_cspn_lsqplus_hawq.py tests/test_evaluate_nyu_selected_quantization.py tests/test_plot_nyu_selected_quantization.py
git commit -m "feat: add unified selected quantization evaluation"
```

### Task 8: Multi-GPU Launch and End-to-End Validation

**Files:**
- Create: `scripts/launch_nyu_three_model_quantization.py`
- Create: `tests/test_launch_nyu_three_model_quantization.py`
- Modify: `README.md`
- Modify: `docs/2026-08-20-quantization-framework-inventory.md`

**Interfaces:**
- Consumes: all runners and the strict experiment config.
- Produces: reproducible per-model launch manifests and the final cross-model summary.

- [ ] **Step 1: Write failing launch-graph tests**

```python
def test_launch_graph_respects_method_dependencies(config):
    graph = build_launch_graph(config)
    assert graph.predecessors("mixed_task_aware") == {"p3_t3_mixed_ptq"}
    assert graph.predecessors("hawq_mixed_le6_qat") == {"hawq_trace"}


def test_each_job_has_an_explicit_cuda_device(config):
    assert all(job.device.startswith("cuda:") for job in build_jobs(config))
```

- [ ] **Step 2: Verify failures**

Run: `pytest -q tests/test_launch_nyu_three_model_quantization.py`

Expected: missing launch module.

- [ ] **Step 3: Implement explicit dependency-aware launch**

Schedule independent model jobs on separate configured GPUs. Persist command,
environment, input revisions, output path, start/end time, and exit status.
Never choose another GPU or Python environment automatically.

- [ ] **Step 4: Run the focused and complete test suites**

Run: `pytest -q tests/test_model_quantization_contracts.py tests/test_experiment_config.py tests/test_nyu_model_runtime.py tests/test_mixed_precision.py tests/test_run_nyu_selected_ptq.py tests/test_run_nyu_model_hawq_trace.py tests/test_model_method_qat.py tests/test_model_task_loss.py tests/test_train_nyu_selected_qat.py tests/test_evaluate_nyu_selected_quantization.py tests/test_plot_nyu_selected_quantization.py tests/test_launch_nyu_three_model_quantization.py`

Run: `pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Run one-sample CUDA smoke tests**

Run each model in its configured environment for FP32, RTN W8A8, RTN W4A4,
QDrop W6A6 hard deployment, BRECQ W6A6 hard deployment, one LSQ++ training
step, one HAWQ trace probe, and one P3/T3 candidate. Assert CUDA extension use,
finite output, expected shape, and propagation invariants.

- [ ] **Step 6: Run formal searches, training, and fixed-64 evaluation**

Launch the three models concurrently according to the explicit GPU map. Do not
publish a cross-method row until its checkpoint, hard-deployment manifest, all
64 predictions, and metric tables are complete.

- [ ] **Step 7: Update inventory and usage documentation**

Document exact commands, environments, checkpoints, calibration identities,
P3/T3 assignments, bit-cost definitions, pooled metrics, and artifact paths.

- [ ] **Step 8: Commit launch and documentation**

```bash
git add scripts/launch_nyu_three_model_quantization.py tests/test_launch_nyu_three_model_quantization.py README.md docs/2026-08-20-quantization-framework-inventory.md
git commit -m "feat: launch three-model quantization evaluation"
```

### Task 9: Final Verification and Cleanup

**Files:**
- Modify: only newly generated formal artifact manifests and retained documentation.

**Interfaces:**
- Consumes: completed formal runs.
- Produces: verified final comparison without stale-result contamination.

- [ ] **Step 1: Validate repository state and artifacts**

Run: `git status --short`

Run the artifact validator over every selected method and assert exact model,
checkpoint, evaluation identities, method contract, prediction count, and
finite pooled metrics.

- [ ] **Step 2: Remove obsolete selected-method artifacts only after validation**

Delete superseded result directories identified by their manifests. Do not
remove checkpoints, datasets, external model sources, current formal results,
or unrelated user files.

- [ ] **Step 3: Re-run summary generation from retained formal roots**

Expected: summary row count is exactly 3 models multiplied by 10 selected
configurations, with no duplicate model-method keys.

- [ ] **Step 4: Run final regression verification**

Run: `pytest -q`

Expected: all tests pass with no pending execution sessions.

- [ ] **Step 5: Commit verified cleanup metadata**

```bash
git add README.md docs profile_logs
git commit -m "docs: publish selected SPN quantization results"
```
