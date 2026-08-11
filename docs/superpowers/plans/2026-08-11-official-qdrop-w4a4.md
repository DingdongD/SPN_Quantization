# Official QDrop INT W4A4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an official-aligned, deployment-equivalent QDrop W4A4 path for CSPN, DySPN, NLSPN, and CompletionFormer without changing existing quantization methods.

**Architecture:** Add a separate learnable activation-QDQ bank and QDrop block reconstructor on top of the existing folded graph, adaptive rounding, semantic edge ownership, and exact weight contract. Export a versioned QDrop contract containing exact W4 weight codes and deterministic A4 edge parameters, then replay it through a dedicated strict instrumentor while the existing propagation-aware path retains exclusive ownership of SPN tensors.

**Tech Stack:** Python 3.11, PyTorch 2.7.1+cu118, CUDA, NumPy, Matplotlib, pytest, official QDrop commit `4a9ca007ce91b66620b911de97df36d5109ecae0`, official CSPN/DySPN/NLSPN/CompletionFormer sources, NYU Depth V2.

---

## File Structure

- Create `configs/qdrop_w4a4.json`: every search, reconstruction, seed, and evaluation value required by the new path.
- Create `spn_quant/qdrop_config.py`: strict JSON-to-dataclass parsing with direct dictionary indexing.
- Create `spn_quant/qdrop_activation.py`: official QDrop learnable A4 fake quantization, mask control, metrics, and deterministic hard quantizers.
- Create `spn_quant/qdrop_targets.py`: model-specific eligible block rules, activation ownership, and propagation exclusions.
- Create `spn_quant/qdrop_edges.py`: bind learnable activation quantizers to existing hardware and CompletionFormer joint boundaries.
- Create `spn_quant/qdrop_reconstruction.py`: cached input mixing, weight/activation joint optimization, loss, and result manifests.
- Create `spn_quant/qdrop_contract.py`: version-2 exact W4+A4 deployment contract and strict replay instrumentor.
- Create `scripts/run_nyu_qdrop_reconstruction.py`: probability search and formal per-model/per-seed reconstruction.
- Create `scripts/run_nyu_qdrop_w4a4.py`: four-model, three-seed orchestration and fixed-sample evaluation.
- Create `scripts/plot_nyu_qdrop_w4a4.py`: aggregate tables and aligned prediction/error figures.
- Modify `scripts/run_nyu_edge_quantization.py`: accept `qdrop_strict` manifests without changing legacy contract behavior.
- Modify `spn_quant/__init__.py`: export only stable QDrop public types.
- Add focused tests under `tests/` and update `README.md` plus a results document after formal evaluation.

### Task 1: Strict Configuration And Official Reference Contract

**Files:**
- Create: `configs/qdrop_w4a4.json`
- Create: `spn_quant/qdrop_config.py`
- Create: `tests/test_qdrop_config.py`

- [ ] **Step 1: Write failing configuration tests**

```python
def test_loads_complete_qdrop_configuration(tmp_path):
    path = write_valid_config(tmp_path)
    config = load_qdrop_config(path)
    assert config.reference.commit == (
        "4a9ca007ce91b66620b911de97df36d5109ecae0")
    assert config.quantization.weight_bits == 4
    assert config.quantization.activation_bits == 4
    assert config.search.quant_probabilities == (0.25, 0.5, 0.75)
    assert config.formal.seeds == (1005, 1006, 1007)


def test_missing_required_field_fails(tmp_path):
    payload = valid_payload()
    del payload["reconstruction"]["activation_learning_rate"]
    path = write_payload(tmp_path, payload)
    with pytest.raises(KeyError, match="activation_learning_rate"):
        load_qdrop_config(path)
```

Add an AST test that rejects `getattr` calls, dictionary `.get` calls, and
`try/except` nodes in all newly created `qdrop_*.py` production files. This
locks in the requested fail-closed coding style.

- [ ] **Step 2: Run RED tests**

Run: `PYTHONPATH=. pytest -q tests/test_qdrop_config.py`

Expected: import failure because `spn_quant.qdrop_config` does not exist.

- [ ] **Step 3: Implement immutable nested configuration types**

Create frozen dataclasses with no field defaults:

```python
@dataclass(frozen=True)
class QDropReferenceConfig:
    repository: str
    branch: str
    commit: str


@dataclass(frozen=True)
class QDropQuantizationConfig:
    weight_bits: int
    activation_bits: int
    weight_clip_ratio: float
    activation_scale_minimum: float


@dataclass(frozen=True)
class QDropSearchConfig:
    calibration_samples: int
    reconstruction_samples: int
    validation_samples: int
    steps: int
    quant_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class QDropReconstructionConfig:
    batch_size: int
    steps: int
    weight_learning_rate: float
    activation_learning_rate: float
    round_loss_weight: float
    warmup_fraction: float
    beta_start: float
    beta_end: float
    loss_power: float


@dataclass(frozen=True)
class QDropFormalConfig:
    seeds: tuple[int, ...]
    evaluation_samples: int
    evaluation_seed: int


@dataclass(frozen=True)
class QDropConfig:
    reference: QDropReferenceConfig
    quantization: QDropQuantizationConfig
    search: QDropSearchConfig
    reconstruction: QDropReconstructionConfig
    formal: QDropFormalConfig
```

`load_qdrop_config(path)` uses `json.loads`, direct `payload["field"]`
indexing, explicit type conversion, and validation. It rejects any bit setting
other than W4A4, requires exactly 1024 calibration samples split as 896+128,
requires probability `0.5` among the candidates, requires 20,000 formal steps,
and requires three unique seeds.

- [ ] **Step 4: Add the complete versioned experiment JSON**

Record the official anchor and explicit values:

```json
{
  "reference": {
    "repository": "https://github.com/wimh966/QDrop.git",
    "branch": "qdrop",
    "commit": "4a9ca007ce91b66620b911de97df36d5109ecae0"
  },
  "quantization": {
    "weight_bits": 4,
    "activation_bits": 4,
    "weight_clip_ratio": 1.0,
    "activation_scale_minimum": 1e-8
  },
  "search": {
    "calibration_samples": 1024,
    "reconstruction_samples": 896,
    "validation_samples": 128,
    "steps": 2000,
    "quant_probabilities": [0.25, 0.5, 0.75]
  },
  "reconstruction": {
    "batch_size": 32,
    "steps": 20000,
    "weight_learning_rate": 0.001,
    "activation_learning_rate": 0.00004,
    "round_loss_weight": 0.01,
    "warmup_fraction": 0.2,
    "beta_start": 20.0,
    "beta_end": 2.0,
    "loss_power": 2.0
  },
  "formal": {
    "seeds": [1005, 1006, 1007],
    "evaluation_samples": 64,
    "evaluation_seed": 20260804
  }
}
```

- [ ] **Step 5: Verify and commit**

Run: `PYTHONPATH=. pytest -q tests/test_qdrop_config.py`

Expected: all tests pass.

Commit:

```bash
git add configs/qdrop_w4a4.json spn_quant/qdrop_config.py tests/test_qdrop_config.py
git commit -m "feat: add strict QDrop W4A4 configuration"
```

### Task 2: Official QDrop Learnable Activation Quantizer

**Files:**
- Create: `spn_quant/qdrop_activation.py`
- Create: `tests/test_qdrop_activation.py`

- [ ] **Step 1: Write failing signed, unsigned, and asymmetric tests**

```python
def test_signed_symmetric_a4_has_fixed_zero_point():
    quantizer = QDropActivationQuantizer(
        site="activation::conv#0", bits=4, signed=True,
        symmetric=True, scale_minimum=1e-8, seed=7)
    quantizer.initialize(torch.tensor([-2.0, 0.0, 2.0]))
    quantizer.start_reconstruction(quant_probability=1.0)
    quantized, codes = quantizer.quantize_with_codes(
        torch.tensor([-2.0, 0.0, 2.0]))
    assert codes.tolist() == [-7, 0, 7]
    assert quantizer.zero_point_parameter is None


def test_unsigned_relu_a4_preserves_zero():
    quantizer = make_unsigned_quantizer(seed=11)
    quantizer.initialize(torch.tensor([0.0, 1.0, 3.0]))
    quantizer.start_reconstruction(quant_probability=1.0)
    quantized, codes = quantizer.quantize_with_codes(
        torch.tensor([0.0, 1.0, 3.0]))
    assert codes.min().item() == 0
    assert codes.max().item() == 15
    assert quantized[0].item() == 0.0
```

Also test asymmetric zero-point gradient, integer hardening, clamp bounds,
positive finite scale, and rejection of unsupported bits.

- [ ] **Step 2: Run RED tests**

Run: `PYTHONPATH=. pytest -q tests/test_qdrop_activation.py -k 'signed or unsigned or asymmetric'`

Expected: import failure for `spn_quant.qdrop_activation`.

- [ ] **Step 3: Implement STE and learnable quantization parameters**

Implement exact helpers and the following public methods:

```python
def round_ste(value: torch.Tensor) -> torch.Tensor:
    return (torch.round(value) - value).detach() + value


def gradient_scale(value: torch.Tensor, factor: float) -> torch.Tensor:
    return (value - value * factor).detach() + value * factor
```

`QDropActivationQuantizer` exposes `initialize(tensor)`,
`start_reconstruction(quant_probability)`, `quantize_with_codes(tensor)`,
`freeze()`, `contract()`, and `statistics()`.

Use a device-matched seeded `torch.Generator`. During reconstruction compute
the deterministic QDQ value first, then return
`torch.where(mask, quantized, tensor)` with
`mask = rand < quant_probability`. Codes always describe the deterministic
QDQ value. Calibration and frozen phases contain no random sampling.

- [ ] **Step 4: Write failing official mask-semantics tests**

Test probabilities 0 and 1 exactly. For probability 0.5, compare output to a
mask generated by an independent generator with the same seed and shape. Test
same-seed equality and different-seed inequality.

- [ ] **Step 5: Implement deterministic hard replay quantizer**

Add `ExactActivationQuantizer.from_contract(entry)` implementing the same
`quantize_with_codes` interface as hardware quantizers. It validates every
required field using `entry["field"]`, verifies the contract fingerprint, and
contains no trainable parameter or random generator.

- [ ] **Step 6: Verify and commit**

Run: `PYTHONPATH=. pytest -q tests/test_qdrop_activation.py`

Commit:

```bash
git add spn_quant/qdrop_activation.py tests/test_qdrop_activation.py
git commit -m "feat: add official QDrop activation quantizer"
```

### Task 3: Explicit Four-Model Targets And Propagation Exclusions

**Files:**
- Create: `spn_quant/qdrop_targets.py`
- Create: `tests/test_qdrop_targets.py`

- [ ] **Step 1: Write failing target-resolution tests**

Build compact models carrying representative official module names. Assert:

```python
def test_dyspn_targets_exclude_propagation_modules():
    plan = resolve_qdrop_targets("dyspn", make_toy_dyspn())
    assert "base.conv2.0" in plan.blocks
    assert "base.gd_dec1_" in plan.blocks
    assert all(not name.startswith("dyspn_") for name in plan.blocks)
    assert "signal::propagation_state" in plan.excluded_sites


def test_completionformer_targets_include_attention_and_concat():
    plan = resolve_qdrop_targets(
        "completionformer", make_toy_completionformer())
    assert any("former.block" in name for name in plan.blocks)
    assert any(site.owner_kind == "attention_qkv"
               for site in plan.activation_sites)
    assert any(site.owner_kind == "concat_input"
               for site in plan.activation_sites)
```

Add tests for CSPN and NLSPN, duplicate activation ownership, overlapping
blocks, missing required roots, unsupported weight modules, and every excluded
role listed in the design.

- [ ] **Step 2: Run RED tests**

Run: `PYTHONPATH=. pytest -q tests/test_qdrop_targets.py`

- [ ] **Step 3: Implement immutable target descriptors**

```python
@dataclass(frozen=True)
class QDropActivationSite:
    site: str
    owner_name: str
    owner_kind: str
    role: str
    signed: bool
    symmetric: bool


@dataclass(frozen=True)
class QDropTargetPlan:
    model: str
    blocks: tuple[str, ...]
    activation_sites: tuple[QDropActivationSite, ...]
    excluded_sites: tuple[str, ...]
```

Implement one resolver per model and one exact dispatch dictionary indexed as
`RESOLVERS[model_name]`. Resolvers use explicit model-specific path patterns,
official root names, and class contracts. They form non-overlapping
residual/decoder/attention blocks and explicitly named leaf-layer blocks.
There is no generic single-layer rule: an eligible Conv/ConvTranspose/Linear
module outside the model-specific patterns raises. Paths under CSPN
`post_process_layer`, DySPN `dyspn_*`, and NLSPN/CompletionFormer `prop_layer`
are excluded before target construction and are never accepted as activation
sites.

- [ ] **Step 4: Validate against all official model builders**

Add a CUDA-marked integration test that builds all four official checkpoint
architectures, folds them with `prepare_hardware_model`, resolves targets, and
asserts:

```python
assert plan.blocks == tuple(sorted(set(plan.blocks)))
assert len(plan.activation_sites) == len(
    set(site.site for site in plan.activation_sites))
assert not propagation_collisions(plan)
assert all_eligible_supported_weights_are_owned(model, plan)
```

- [ ] **Step 5: Verify and commit**

Run: `PYTHONPATH=. pytest -q tests/test_qdrop_targets.py`

Commit:

```bash
git add spn_quant/qdrop_targets.py tests/test_qdrop_targets.py
git commit -m "feat: define strict QDrop model targets"
```

### Task 4: Bind QDrop To Existing Semantic Edges

**Files:**
- Create: `spn_quant/qdrop_edges.py`
- Create: `tests/test_qdrop_edges.py`
- Modify: `spn_quant/adapters/completionformer_joint.py`
- Modify: `tests/test_completionformer_joint_adapter.py`

- [ ] **Step 1: Write failing hardware-boundary ownership tests**

Construct a toy Conv-ReLU-Conv block with a configured
`HardwareAlignedInstrumentor`. Bind a `QDropActivationBank`, run one forward,
and assert the instrumentor calls the learnable quantizer exactly once per
logical edge. Assert generic QDQ does not run in addition to the QDrop edge.

- [ ] **Step 2: Run RED tests**

Run: `PYTHONPATH=. pytest -q tests/test_qdrop_edges.py`

- [ ] **Step 3: Implement the activation bank**

`QDropActivationBank` exposes `observe()`, `initialize()`,
`reconstruct(target, probability)`, `freeze_target(target)`,
`disable_randomness()`, `parameters_for(target)`, `contracts()`, and
`manifest()`.

Map each explicit site to exactly one existing boundary:

- `module_input` and `module_output` map to `instrumentor.quantizers[(name, kind)]`;
- `relu_output` maps to `instrumentor.relu_quantizers[call_name]`;
- merge sites map to the existing independent branch controller;
- CompletionFormer Q/K/V and concat sites map to the joint adapter controllers.

The bank snapshots the replaced objects and restores them on `close()`. Unknown
or multiply owned boundaries raise. It never builds an observer or derives a
new scale during reconstruction.

- [ ] **Step 4: Add CompletionFormer failing QDrop phase tests**

Extend the toy CompletionFormer tests to assert Q/K/V and concat controllers
accept externally supplied learnable quantizers during reconstruction, expose
their parameters, consume exactly the captured teacher target count, and switch
to deterministic contract quantizers at freeze. Softmax probability remains
A8/FP16 and is not a QDrop site.

- [ ] **Step 5: Add the narrow joint-adapter integration**

Add explicit `bind_qdrop_sites(sites, quantizers)`,
`qdrop_parameters(target)`, `freeze_qdrop_sites(target)`, and
`unbind_qdrop_sites()` methods to `CompletionFormerJointAdapter`.

Existing `capture_targets`, `observe_reconstruction`, `freeze`, and `configure`
behavior must remain byte-for-byte unchanged when no QDrop binding exists.

- [ ] **Step 6: Verify regression safety and commit**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_qdrop_edges.py tests/test_completionformer_joint_adapter.py tests/test_hardware_aligned_quantization.py
```

Commit:

```bash
git add spn_quant/qdrop_edges.py spn_quant/adapters/completionformer_joint.py tests/test_qdrop_edges.py tests/test_completionformer_joint_adapter.py
git commit -m "feat: bind QDrop to semantic activation edges"
```

### Task 5: QDrop Block Reconstruction Engine

**Files:**
- Create: `spn_quant/qdrop_reconstruction.py`
- Create: `tests/test_qdrop_reconstruction.py`

- [ ] **Step 1: Write failing official input-mixing tests**

```python
def test_mix_inputs_matches_official_elementwise_semantics():
    quantized = torch.tensor([[1.0, 2.0, 3.0]])
    full_precision = torch.tensor([[10.0, 20.0, 30.0]])
    generator = torch.Generator().manual_seed(5)
    mixed = mix_qdrop_inputs(
        quantized, full_precision, 0.5, generator)
    expected_mask = torch.rand(
        quantized.shape, generator=torch.Generator().manual_seed(5)) < 0.5
    torch.testing.assert_close(
        mixed, torch.where(expected_mask, quantized, full_precision))
```

Cover nested tuple/list/dictionary inputs, shape mismatch, non-tensor mismatch,
probabilities 0/1, and deterministic seed behavior.

- [ ] **Step 2: Run RED tests**

Run: `PYTHONPATH=. pytest -q tests/test_qdrop_reconstruction.py -k mix`

- [ ] **Step 3: Implement records, configuration, and result types**

```python
@dataclass
class QDropCalibrationRecord:
    quantized_inputs: tuple[object, ...]
    full_precision_inputs: tuple[object, ...]
    reference: object


@dataclass(frozen=True)
class QDropOptimizerConfig:
    steps: int
    batch_size: int
    weight_learning_rate: float
    activation_learning_rate: float
    round_loss_weight: float
    warmup_fraction: float
    beta_start: float
    beta_end: float
    loss_power: float
    quant_probability: float
    seed: int


@dataclass
class QDropReconstructionResult:
    before_loss: float
    after_loss: float
    history: list[dict[str, float]]
    weight_contracts: dict[str, dict[str, object]]
    activation_contracts: dict[str, dict[str, object]]
```

All optimizer fields are required. Invalid probability, empty records, missing
activation parameters, overlapping targets, or non-finite tensors raise.

- [ ] **Step 4: Write the failing joint-gradient and hard-result tests**

Use a two-layer block and one activation site. Assert after backward that only
AdaRound alpha and activation scale/zero-point parameters have gradients. Save
ordinary parameters before fitting and assert exact equality after fitting.
Assert a finite hard result is exported even when its local loss is worse than
the initial state, matching official QDrop without RTN/BRECQ fallback.

- [ ] **Step 5: Implement official reconstruction optimization**

`QDropBlockReconstructor.fit(records)` performs:

```text
freeze ordinary parameters
install soft adaptive rounding
activate only target-owned QDrop sites
sample cached mini-batch indices with a CPU generator
mix quantized and FP block inputs element by element
forward block with per-edge QDrop masks
compute channel-sum LP reconstruction loss
add sum-reduced rounding regularization after warm-up
step Adam(weight alpha, lr=1e-3)
step Adam(activation params, lr=4e-5)
step cosine activation scheduler
harden weights and activation parameters
evaluate deterministic hard loss
export whenever hard loss is finite
```

Reuse `_rounding_regularization`, `strict_reconstruction_loss`, and the existing
temperature schedule rather than duplicating their formulas.

- [ ] **Step 6: Verify and commit**

Run: `PYTHONPATH=. pytest -q tests/test_qdrop_reconstruction.py tests/test_strict_reconstruction.py`

Commit:

```bash
git add spn_quant/qdrop_reconstruction.py tests/test_qdrop_reconstruction.py
git commit -m "feat: add strict QDrop block reconstruction"
```

### Task 6: Exact W4+A4 Contract And Deterministic Replay

**Files:**
- Create: `spn_quant/qdrop_contract.py`
- Create: `tests/test_qdrop_contract.py`
- Modify: `scripts/run_nyu_edge_quantization.py`
- Modify: `tests/test_edge_runner.py`
- Modify: `spn_quant/__init__.py`

- [ ] **Step 1: Write failing version-2 contract round-trip tests**

Build a payload with one weight and one activation entry. Save/load it and
assert exact integer codes, scale tensor, zero point, site owner, checkpoint
hash, graph contract, and activation fingerprint survive. Tampering with each
field must raise before model execution.

- [ ] **Step 2: Run RED tests**

Run: `PYTHONPATH=. pytest -q tests/test_qdrop_contract.py`

- [ ] **Step 3: Implement a separate QDrop contract format**

Expose `QDROP_CONTRACT_VERSION = 2`,
`build_qdrop_contract(source_checkpoint, graph_contract, weight_contracts,
activation_contracts, targets, metadata)`,
`save_qdrop_contract(path, payload)`, `load_qdrop_contract(path)`, and
`QDropContractInstrumentor` with
`configure(w_bits, a_bits, enabled_groups, **kwargs)`, `manifest()`, and
`metadata()`.

`QDropContractInstrumentor` composes `StrictContractInstrumentor` for exact W4
replay, then replaces every active generic/semantic A4 quantizer with
`ExactActivationQuantizer` from the contract. It requires `w_bits == 4`,
`a_bits == 4`, uniform activation mode, matching enabled ownership, and zero
observer updates after binding. Legacy version-1 AdaRound/BRECQ loading remains
unchanged.

- [ ] **Step 4: Add failing edge-runner dispatch tests**

Test `load_reconstruction_manifest` accepts exactly:

```json
{
  "method": "qdrop_strict",
  "weight_bits": 4,
  "activation_bits": 4,
  "activation_policy": "exact_semantic_edge_contract"
}
```

It must reject QDrop with activation bits 0/8, absent activation contracts,
E2M1, direct activation overrides, or a version-1 contract.

- [ ] **Step 5: Integrate explicit QDrop dispatch**

Keep the current weight-only path unchanged. Add a separate exact method branch
indexed by `manifest["method"]`; it calls `load_qdrop_contract` and constructs
`QDropContractInstrumentor`. Do not catch validation errors and do not convert a
QDrop failure to RTN/BRECQ.

- [ ] **Step 6: Verify and commit**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_qdrop_contract.py tests/test_deployment_contract.py tests/test_edge_runner.py
```

Commit:

```bash
git add spn_quant/qdrop_contract.py spn_quant/__init__.py scripts/run_nyu_edge_quantization.py tests/test_qdrop_contract.py tests/test_edge_runner.py
git commit -m "feat: replay exact QDrop W4A4 contracts"
```

### Task 7: NYU Probability Search And Formal Reconstruction Runner

**Files:**
- Create: `scripts/run_nyu_qdrop_reconstruction.py`
- Create: `tests/test_qdrop_reconstruction_runner.py`

- [ ] **Step 1: Write failing split and selection tests**

Test that 1024 unique calibration indices split deterministically into 896
reconstruction and 128 validation indices, neither overlaps the fixed 64
evaluation indices, and candidate selection orders by validation loss then
probability. Non-finite candidate loss, failed target, or missing candidate
must make selection fail.

- [ ] **Step 2: Run RED tests**

Run: `PYTHONPATH=. pytest -q tests/test_qdrop_reconstruction_runner.py`

- [ ] **Step 3: Implement strict command interface**

Require every path and mode explicitly:

```text
--config
--run-dir
--checkpoint
--data-root
--model
--phase probability-search|formal
--seed
--out-dir
```

The parser provides no values for these arguments. The loaded config supplies
only versioned experiment values. `--model` must match checkpoint provenance.

- [ ] **Step 4: Implement capture and candidate isolation**

For each probability candidate, rebuild teacher and student from the original
checkpoint, prepare and validate identical folded graphs, create the same target
plan, calibrate the exact activation boundaries, and reconstruct every block in
plan order. No candidate reuses optimized weights or activation parameters from
another candidate.

Write these artifacts per candidate:

```text
qdrop_configuration.json
qdrop_target_manifest.csv
qdrop_activation_manifest.csv
qdrop_reconstruction_summary.csv
qdrop_reconstruction_history.json
qdrop_candidate_metrics.csv
qdrop_strict_contract.pt
qdrop_strict_manifest.json
```

- [ ] **Step 5: Implement formal reconstruction**

Load the selected probability from the completed search artifact, rebuild from
the original checkpoint, use all 1024 calibration samples, run 20,000 steps per
block with the explicit seed, and export one immutable contract. Verify a fresh
process can load the contract and reproduce contract fingerprints before marking
the run complete.

- [ ] **Step 6: Verify CPU tests and one-block CUDA smoke**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_qdrop_reconstruction_runner.py
python scripts/run_nyu_qdrop_reconstruction.py --config configs/qdrop_w4a4.json --run-dir /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/dyspn_iter6 --checkpoint best.pt --data-root /workspace/datasets/nyu --model dyspn --phase probability-search --seed 1005 --out-dir profile_logs/nyu_qdrop_w4a4_smoke
```

For the smoke invocation, use a test-only configuration file created under the
worktree with 64 calibration samples and 20 steps, then remove that file after
the run. Do not place artifacts under `/tmp`.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_nyu_qdrop_reconstruction.py tests/test_qdrop_reconstruction_runner.py
git commit -m "feat: add strict NYU QDrop reconstruction runner"
```

### Task 8: Four-Model Evaluation, Aggregation, And Predictions

**Files:**
- Create: `scripts/run_nyu_qdrop_w4a4.py`
- Create: `scripts/plot_nyu_qdrop_w4a4.py`
- Create: `tests/test_qdrop_w4a4_evaluation.py`
- Create: `tests/test_plot_nyu_qdrop_w4a4.py`

- [ ] **Step 1: Write failing orchestration-contract tests**

Assert the run matrix contains exactly four models, three seeds, W4A4 only, and
the same 64 evaluation indices for FP32, RTN, BRECQ, and QDrop. Assert a missing
seed, mismatched checkpoint hash, repeated evaluation index, or non-finite
prediction fails aggregation.

- [ ] **Step 2: Run RED tests**

Run: `PYTHONPATH=. pytest -q tests/test_qdrop_w4a4_evaluation.py tests/test_plot_nyu_qdrop_w4a4.py`

- [ ] **Step 3: Implement orchestration without subprocess error masking**

`run_nyu_qdrop_w4a4.py` validates all run directories, checkpoints, BRECQ
contracts, config fields, CUDA extensions, and output directories before the
first reconstruction. It writes an explicit command manifest, executes each
command with `check=True`, and stops on the first failure.

- [ ] **Step 4: Implement strict aggregate tables**

Write:

```text
qdrop_w4a4_sample_metrics.csv
qdrop_w4a4_seed_summary.csv
qdrop_w4a4_model_summary.csv
qdrop_w4a4_layer_metrics.csv
qdrop_w4a4_propagation_metrics.csv
qdrop_w4a4_acceptance.csv
```

For QDrop, report per-model mean, standard deviation, minimum, and maximum RMSE
over seeds. Include RMSE, MAE, ABS_REL, non-finite ratios, zero ratio,
saturation ratio, activation SQNR, and propagation-step error. Acceptance
requires finite output and lower mean RMSE than BRECQ W4A4; separately report
whether degradation from FP32 is at most 10%.

- [ ] **Step 5: Implement aligned prediction figures**

For one median-error and one worst-error sample per model, render a single figure
with RGB, sparse depth, GT, FP32, RTN W4A4, BRECQ W4A4, each QDrop seed, QDrop
seed-mean prediction, and absolute-error maps. Use Arial, no title, unrotated
model labels, shared depth/error ranges, grid below image artists, and explicit
finite-value masks.

- [ ] **Step 6: Verify and commit**

Run:

```bash
PYTHONPATH=. pytest -q tests/test_qdrop_w4a4_evaluation.py tests/test_plot_nyu_qdrop_w4a4.py
```

Commit:

```bash
git add scripts/run_nyu_qdrop_w4a4.py scripts/plot_nyu_qdrop_w4a4.py tests/test_qdrop_w4a4_evaluation.py tests/test_plot_nyu_qdrop_w4a4.py
git commit -m "feat: evaluate QDrop W4A4 across SPN models"
```

### Task 9: Regression Verification And Formal GPU Campaign

**Files:**
- Modify: `README.md`
- Create after evaluation: `docs/2026-08-11-official-qdrop-w4a4-results.md`

- [ ] **Step 1: Run the focused QDrop suite**

```bash
PYTHONPATH=. pytest -q tests/test_qdrop_config.py tests/test_qdrop_activation.py tests/test_qdrop_targets.py tests/test_qdrop_edges.py tests/test_qdrop_reconstruction.py tests/test_qdrop_contract.py tests/test_qdrop_reconstruction_runner.py tests/test_qdrop_w4a4_evaluation.py tests/test_plot_nyu_qdrop_w4a4.py
```

Expected: all tests pass with no warnings or skipped non-CUDA unit tests.

- [ ] **Step 2: Run existing quantization regressions**

```bash
PYTHONPATH=. pytest -q tests/test_strict_reconstruction.py tests/test_strict_reconstruction_runner.py tests/test_deployment_contract.py tests/test_edge_runner.py tests/test_hardware_aligned_quantization.py tests/test_completionformer_joint_adapter.py tests/test_propagation_quantization.py
```

Expected: existing RTN, AdaRound, BRECQ, PA, Attention, and Concat tests remain
green.

- [ ] **Step 3: Run four-model probability search**

```bash
python scripts/run_nyu_qdrop_w4a4.py --config configs/qdrop_w4a4.json --phase probability-search --data-root /workspace/datasets/nyu --out-dir profile_logs/nyu_qdrop_w4a4
```

Expected: each model has three complete candidate contracts and one selected
probability based only on the 128 held-out calibration samples.

- [ ] **Step 4: Run formal 1024-sample, 20,000-step reconstruction**

```bash
python scripts/run_nyu_qdrop_w4a4.py --config configs/qdrop_w4a4.json --phase formal --data-root /workspace/datasets/nyu --out-dir profile_logs/nyu_qdrop_w4a4
```

Expected: 12 immutable QDrop contracts, one for every model/seed pair, all with
fresh-process replay verification.

- [ ] **Step 5: Evaluate the fixed 64 samples and render figures**

```bash
python scripts/run_nyu_qdrop_w4a4.py --config configs/qdrop_w4a4.json --phase evaluate --data-root /workspace/datasets/nyu --out-dir profile_logs/nyu_qdrop_w4a4
python scripts/plot_nyu_qdrop_w4a4.py --root profile_logs/nyu_qdrop_w4a4 --out-dir profile_logs/nyu_qdrop_w4a4/analysis
```

Expected: all method/model rows use identical evaluation indices, contain no
non-finite predictions, and include GT/FP32/RTN/BRECQ/QDrop comparisons.

- [ ] **Step 6: Document measured results**

Update `README.md` with the QDrop command and artifact index. Write the results
document from generated CSV files, including exact hashes, selected
probabilities, seed variance, acceptance decisions, layer bottlenecks, and
propagation error behavior. Do not enter values manually before the formal CSV
files exist.

- [ ] **Step 7: Run final verification and commit**

```bash
git diff --check
PYTHONPATH=. pytest -q
git status --short
```

Inspect all generated summary CSV files and prediction figures, then commit only
source, tests, configuration, README, and the concise results document:

```bash
git add README.md docs/2026-08-11-official-qdrop-w4a4-results.md
git commit -m "docs: report official QDrop W4A4 evaluation"
```
