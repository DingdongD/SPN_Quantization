# CSPN LSQ+ and HAWQ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native LSQ+ W4A4/W6A6 and HAWQ mixed-precision <=6-bit QAT for the official 24-step NYU CSPN model, then produce one strict fixed-64 comparison and prediction visualization.

**Architecture:** New algorithm modules implement LSQ+ learned step/offset quantization, HAWQ running-range quantization, CSPN Hessian trace estimation, and deterministic mixed-bit allocation. A focused CSPN method controller reuses the existing semantic owner registry and A8/Q13/INT32 propagation controller. Separate training/evaluation workers write disjoint artifacts, while one launcher assigns LSQ+ W4A4, LSQ+ W6A6, HAWQ, and baseline replay to four GPUs.

**Tech Stack:** Python 3.11, PyTorch, NumPy, SciPy `optimize.milp`, pytest, Matplotlib, official CSPN CUDA execution, NYU Depth V2.

---

## File Map

- `configs/cspn_lsqplus_hawq.json`: all required method, training, trace, budget, GPU, and plotting values.
- `spn_quant/qat/method_config.py`: strict typed loading of the JSON configuration.
- `spn_quant/qat/lsqplus.py`: LSQ+ activation and per-output-channel weight quantizers.
- `spn_quant/qat/hawq.py`: HAWQ running-range activation and weight fake quantizers.
- `spn_quant/hawq_trace.py`: masked curvature loss and Hutchinson block-trace estimator.
- `spn_quant/hawq_allocation.py`: candidate perturbation costs, CSPN owner tying, MILP solve, and budget audit.
- `spn_quant/qat/cspn_methods.py`: LSQ+/HAWQ installation, state export/reload, and propagation composition.
- `scripts/run_nyu_cspn_hawq_trace.py`: official CSPN trace capture and mixed-bit assignment.
- `scripts/train_nyu_cspn_lsqplus_hawq.py`: shared strict QAT runner for three formal training jobs.
- `scripts/evaluate_nyu_cspn_lsqplus_hawq.py`: per-configuration fixed-64 evaluator and strict aggregator.
- `scripts/plot_nyu_cspn_lsqplus_hawq.py`: metric comparison and aligned prediction/error figures.
- `scripts/launch_nyu_cspn_lsqplus_hawq.py`: two-phase four-GPU orchestration.
- `tests/test_lsqplus_hawq_config.py`: configuration contract tests.
- `tests/test_lsqplus_quantizers.py`: LSQ+ forward, initialization, gradient, and state tests.
- `tests/test_hawq_quantizers.py`: HAWQ range and QDQ tests.
- `tests/test_hawq_trace.py`: curvature and Hutchinson estimator tests.
- `tests/test_hawq_allocation.py`: objective, tying, MILP, and budget tests.
- `tests/test_cspn_method_qat.py`: official owner/controller integration tests.
- `tests/test_run_nyu_cspn_hawq_trace.py`: trace runner protocol tests.
- `tests/test_train_nyu_cspn_lsqplus_hawq.py`: training and checkpoint contract tests.
- `tests/test_evaluate_nyu_cspn_lsqplus_hawq.py`: strict evaluation/aggregation tests.
- `tests/test_plot_nyu_cspn_lsqplus_hawq.py`: figure data and shape tests.
- `tests/test_launch_nyu_cspn_lsqplus_hawq.py`: GPU/stage command tests.
- `docs/2026-08-24-cspn-lsqplus-hawq-results.md`: measured formal results and interpretation.

### Task 1: Strict Method Configuration

**Files:**
- Create: `configs/cspn_lsqplus_hawq.json`
- Create: `spn_quant/qat/method_config.py`
- Modify: `spn_quant/qat/__init__.py`
- Test: `tests/test_lsqplus_hawq_config.py`

- [ ] **Step 1: Write the failing configuration tests**

```python
from pathlib import Path

import pytest

from spn_quant.qat.method_config import load_method_config


CONFIG = Path(__file__).resolve().parents[1] / \
    "configs/cspn_lsqplus_hawq.json"


def test_formal_config_declares_exact_methods_and_gpu_owners():
    config = load_method_config(CONFIG)
    assert config.model == "cspn"
    assert config.lsqplus.bits == (4, 6)
    assert config.hawq.bits == (4, 6, 8)
    assert config.hawq.maximum_average_weight_bits == 6.0
    assert config.hawq.maximum_average_activation_bits == 6.0
    assert config.gpus == (
        ("baselines", 0), ("lsqplus_w4a4", 1),
        ("lsqplus_w6a6", 2), ("hawq_mixed_le6", 3))


def test_missing_required_field_is_not_defaulted(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"model": "cspn"}', encoding="utf-8")
    with pytest.raises(KeyError):
        load_method_config(path)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `PYTHONPATH=. pytest -q tests/test_lsqplus_hawq_config.py`

Expected: collection fails with `ModuleNotFoundError: spn_quant.qat.method_config`.

- [ ] **Step 3: Add the complete required JSON contract**

```json
{
  "model": "cspn",
  "source_revisions": {
    "lsqplus": "f26c972c3175a74c0818e09da992feecfe9cc45c",
    "hawq": "1616df69fdd99f100a8d8a6e78742e0a89292634"
  },
  "lsqplus": {
    "bits": [4, 6],
    "initialization_batch_size": 4,
    "initialization_batches": 32
  },
  "hawq": {
    "bits": [4, 6, 8],
    "maximum_average_weight_bits": 6.0,
    "maximum_average_activation_bits": 6.0,
    "fixed_blocks": ["encoder_stem", "initial_depth"],
    "activation_range_momentum": 0.95,
    "trace": {
      "batch_size": 4,
      "probes_per_batch": 8,
      "seed": 20260824,
      "depth_mse_weight": 1.0,
      "boundary_mse_weight": 0.25,
      "boundary_threshold_m": 0.1
    }
  },
  "loss": {
    "depth": 1.0,
    "boundary": 0.25,
    "teacher": 0.5,
    "propagation": 0.1
  },
  "training": {
    "epochs": 30,
    "patience": 6,
    "min_relative_improvement": 0.001,
    "batch_size": 4,
    "val_batch_size": 1,
    "workers": 2,
    "learning_rate": 0.0001,
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "max_gradient_norm": 10.0,
    "seed": 20260824,
    "fold_max_error": 0.05,
    "log_interval": 50
  },
  "gpus": {
    "baselines": 0,
    "lsqplus_w4a4": 1,
    "lsqplus_w6a6": 2,
    "hawq_mixed_le6": 3
  },
  "evaluation": {
    "samples": 64,
    "batch_size": 1,
    "workers": 2,
    "depth_min_m": 0.0,
    "depth_max_m": 10.0,
    "detail_samples": 8,
    "font_size": 13
  }
}
```

- [ ] **Step 4: Implement direct-indexed dataclass loading**

Create frozen dataclasses `LSQPlusMethodConfig`, `HAWQTraceConfig`,
`HAWQMethodConfig`, `TrainingMethodConfig`, `EvaluationMethodConfig`, and
`CSPNMethodExperimentConfig`. `load_method_config(path)` must use
`payload["field"]` for every field, validate exact bit tuples and positive
numeric values, and return GPU rows sorted as shown in the test. Do not use
`.get`, `getattr`, a broad exception, or code-level defaults.

- [ ] **Step 5: Run the focused tests and full config-adjacent tests**

Run: `PYTHONPATH=. pytest -q tests/test_lsqplus_hawq_config.py tests/test_quant_specs.py tests/test_cspn_task_sensitive_bits.py`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add configs/cspn_lsqplus_hawq.json spn_quant/qat/method_config.py \
  spn_quant/qat/__init__.py tests/test_lsqplus_hawq_config.py
git commit -m "feat: define CSPN LSQ+ and HAWQ contracts"
```

### Task 2: LSQ+ Learned Quantizers

**Files:**
- Create: `spn_quant/qat/lsqplus.py`
- Modify: `spn_quant/qat/__init__.py`
- Test: `tests/test_lsqplus_quantizers.py`

- [ ] **Step 1: Write failing hard-forward and initialization tests**

```python
import pytest
import torch

from spn_quant.qat.lsqplus import (
    LSQPlusActivationQuantizer,
    LSQPlusWeightParametrization,
)


def test_lsqplus_activation_matches_affine_hard_reference():
    quantizer = LSQPlusActivationQuantizer(bits=4, unsigned=False)
    quantizer.initialize(torch.tensor([-2.0, -0.3, 0.7, 3.0]))
    current = torch.tensor([-2.5, -0.2, 0.8, 3.5], requires_grad=True)
    scale = quantizer.step.detach()
    offset = quantizer.offset.detach()
    expected = torch.round((current.detach() - offset) / scale).clamp(-8, 7)
    expected = expected * scale + offset
    torch.testing.assert_close(quantizer(current).detach(), expected)


def test_lsqplus_unsigned_uses_all_sixteen_a4_codes():
    quantizer = LSQPlusActivationQuantizer(bits=4, unsigned=True)
    quantizer.initialize(torch.tensor([0.0, 15.0]))
    _, codes = quantizer.quantize_with_codes(torch.arange(16.0))
    assert torch.equal(codes, torch.arange(16.0))


def test_lsqplus_weight_step_is_per_logical_output_channel():
    weight = torch.tensor([[[[1.0]]], [[[4.0]]]])
    quantizer = LSQPlusWeightParametrization(
        bits=4, channel_dim=0, initial_weight=weight)
    assert quantizer.step.shape == (2, 1, 1, 1)
    assert quantizer(weight).shape == weight.shape


def test_lsqplus_requires_explicit_initialization():
    quantizer = LSQPlusActivationQuantizer(bits=4, unsigned=False)
    with pytest.raises(RuntimeError, match="not initialized"):
        quantizer(torch.ones(2))
```

- [ ] **Step 2: Run and verify RED**

Run: `PYTHONPATH=. pytest -q tests/test_lsqplus_quantizers.py`

Expected: import fails because `spn_quant.qat.lsqplus` does not exist.

- [ ] **Step 3: Implement LSQ+ STE primitives and quantizers**

Implement:

```python
def grad_scale(value: torch.Tensor, scale: float) -> torch.Tensor:
    scaled = value * float(scale)
    return (value - scaled).detach() + scaled


def round_pass(value: torch.Tensor) -> torch.Tensor:
    rounded = torch.round(value)
    return (rounded - value).detach() + value
```

`LSQPlusActivationQuantizer` must expose `bits`, `unsigned`, `qmin`, `qmax`,
`granularity="tensor"`, `group_size`, `scale_count`, `step`, `offset`,
`initialize(tensor)`, `scale_for(tensor)`, `quantize_with_codes(tensor)`, and
`forward(tensor)`. Initialization uses exact min/max affine V2 equations. The
forward uses `g=1/sqrt(numel(x)*qmax)` and fails on non-positive/non-finite
step or non-finite input.

`LSQPlusWeightParametrization` initializes one step per logical output channel
from `max(abs(mean-3*std), abs(mean+3*std))/(2^bits-1)`, applies signed
symmetric codes, and retains gradients for both master weight and step.

- [ ] **Step 4: Add gradient and round-trip state tests**

```python
def test_lsqplus_step_offset_and_master_weight_receive_gradients():
    activation = LSQPlusActivationQuantizer(bits=4, unsigned=False)
    activation.initialize(torch.tensor([-1.0, 2.0]))
    weight = torch.tensor([[[[0.5]]]], requires_grad=True)
    weight_quantizer = LSQPlusWeightParametrization(4, 0, weight.detach())
    loss = activation(torch.tensor([0.25], requires_grad=True)).sum()
    loss = loss + weight_quantizer(weight).sum()
    loss.backward()
    assert activation.step.grad is not None
    assert activation.offset.grad is not None
    assert weight_quantizer.step.grad is not None
    assert weight.grad is not None


def test_lsqplus_state_reload_preserves_hard_output():
    source = LSQPlusActivationQuantizer(6, False)
    source.initialize(torch.tensor([-3.0, 5.0]))
    state = source.state_dict()
    target = LSQPlusActivationQuantizer(6, False)
    target.load_state_dict(state)
    current = torch.linspace(-4.0, 6.0, 101)
    torch.testing.assert_close(source(current), target(current))
```

- [ ] **Step 5: Run focused and existing QAT tests**

Run: `PYTHONPATH=. pytest -q tests/test_lsqplus_quantizers.py tests/test_qat_quantizers.py tests/test_qat_ste.py`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add spn_quant/qat/lsqplus.py spn_quant/qat/__init__.py \
  tests/test_lsqplus_quantizers.py
git commit -m "feat: implement LSQ+ learned quantizers"
```

### Task 3: HAWQ Fake Quantizers

**Files:**
- Create: `spn_quant/qat/hawq.py`
- Modify: `spn_quant/qat/quantizers.py`
- Modify: `spn_quant/qat/__init__.py`
- Test: `tests/test_hawq_quantizers.py`
- Test: `tests/test_qat_quantizers.py`

- [ ] **Step 1: Write failing range and W6 tests**

```python
import torch

from spn_quant.qat.hawq import HAWQActivationQuantizer
from spn_quant.qat.quantizers import PerOutputChannelWeightFakeQuantizer


def test_hawq_activation_updates_then_freezes_running_range():
    quantizer = HAWQActivationQuantizer(
        bits=4, unsigned=False, range_momentum=0.5)
    quantizer.train()
    quantizer(torch.tensor([-2.0, 4.0]))
    quantizer(torch.tensor([-4.0, 2.0]))
    observed = (quantizer.minimum.clone(), quantizer.maximum.clone())
    quantizer.freeze_range()
    quantizer(torch.tensor([-100.0, 100.0]))
    assert torch.equal(quantizer.minimum, observed[0])
    assert torch.equal(quantizer.maximum, observed[1])


def test_hawq_affine_codes_include_zero_point():
    quantizer = HAWQActivationQuantizer(4, False, 0.5)
    quantizer.initialize_range(torch.tensor([-1.0, 3.0]))
    _, codes = quantizer.quantize_with_codes(torch.tensor([0.0]))
    assert 0 <= int(codes.item()) <= 15
    assert int(quantizer.zero_point.item()) != 0


def test_existing_weight_fake_quantizer_accepts_w6():
    quantizer = PerOutputChannelWeightFakeQuantizer(6, 0)
    output = quantizer(torch.randn(3, 2, 1, 1))
    assert output.shape == (3, 2, 1, 1)
```

- [ ] **Step 2: Run and verify RED**

Run: `PYTHONPATH=. pytest -q tests/test_hawq_quantizers.py`

Expected: import fails for `spn_quant.qat.hawq`, and the existing weight
quantizer rejects 6 bits.

- [ ] **Step 3: Implement the running-range quantizer and W6 support**

`HAWQActivationQuantizer` must implement one affine per-tensor range per semantic
owner, unsigned ReLU or signed input behavior, EMA updates only while
`training and running_range`, `freeze_range()`, `scale_for`,
`quantize_with_codes`, hard-forward STE, and state serialization. First update
copies measured min/max; subsequent updates apply the required momentum.

Change the existing weight fake quantizer validation from `(4, 8)` to
`(4, 6, 8)` without changing its W4/W8 output.

- [ ] **Step 4: Verify parity and regressions**

Run: `PYTHONPATH=. pytest -q tests/test_hawq_quantizers.py tests/test_qat_quantizers.py tests/test_cspn_qat.py`

Expected: all tests pass, including unchanged W4/W8 reference values.

- [ ] **Step 5: Commit**

```bash
git add spn_quant/qat/hawq.py spn_quant/qat/quantizers.py \
  spn_quant/qat/__init__.py tests/test_hawq_quantizers.py \
  tests/test_qat_quantizers.py
git commit -m "feat: implement HAWQ fake quantizers"
```

### Task 4: CSPN Hutchinson Trace Estimator

**Files:**
- Create: `spn_quant/hawq_trace.py`
- Test: `tests/test_hawq_trace.py`

- [ ] **Step 1: Write exact quadratic trace tests**

```python
import torch
import torch.nn as nn

from spn_quant.hawq_trace import (
    HutchinsonTraceConfig,
    estimate_block_traces,
    masked_curvature_loss,
)


def test_masked_curvature_loss_uses_depth_and_boundary_mse():
    prediction = torch.tensor([[[[1.0, 3.0]]]])
    target = torch.tensor([[[[1.0, 1.0]]]])
    valid = torch.ones_like(target, dtype=torch.bool)
    loss = masked_curvature_loss(prediction, target, valid, 1.0, 0.25, 0.5)
    assert float(loss) == 2.5


def test_hutchinson_trace_matches_diagonal_quadratic_hessian():
    model = nn.Linear(2, 1, bias=False)
    model.weight.data.copy_(torch.tensor([[1.0, 2.0]]))
    config = HutchinsonTraceConfig(probes=8, seed=7)

    def loss_fn():
        return (model.weight.square() * torch.tensor([[2.0, 5.0]])).sum()

    result = estimate_block_traces(
        (("linear", model.weight),), loss_fn, config)
    assert result[0].block == "linear"
    assert result[0].mean == 7.0
    assert result[0].normalized_mean == 3.5
```

- [ ] **Step 2: Run and verify RED**

Run: `PYTHONPATH=. pytest -q tests/test_hawq_trace.py`

Expected: `ModuleNotFoundError: spn_quant.hawq_trace`.

- [ ] **Step 3: Implement one-HVP all-block estimation**

Implement frozen `HutchinsonTraceConfig(probes, seed)` and
`BlockTraceEstimate(block, estimates, mean, standard_error,
normalized_mean, coefficient_of_variation, parameters)` dataclasses.

For each probe:

```python
loss = loss_fn()
gradients = torch.autograd.grad(
    loss, parameters, create_graph=True, retain_graph=True)
vectors = tuple(
    torch.empty_like(parameter).bernoulli_(0.5, generator=generator)
    .mul_(2.0).sub_(1.0)
    for parameter in parameters)
inner = sum((gradient * vector).sum()
            for gradient, vector in zip(gradients, vectors))
hessian_vectors = torch.autograd.grad(inner, parameters)
estimate = tuple((vector * hv).sum()
                 for vector, hv in zip(vectors, hessian_vectors))
```

The implementation validates exact block/parameter coverage, finite loss and
estimates, and a non-negative final mean. It does not clamp or replace trace
values. `masked_curvature_loss` uses direct tensor checks and the existing
GT-derived boundary mask.

- [ ] **Step 4: Add reproducibility and negative-mean tests**

```python
def test_hutchinson_seed_is_reproducible():
    def estimate():
        model = nn.Linear(2, 1, bias=False)
        model.weight.data.copy_(torch.tensor([[1.0, -2.0]]))
        return estimate_block_traces(
            (("linear", model.weight),),
            lambda: (model.weight.square() *
                     torch.tensor([[2.0, 5.0]])).sum(),
            HutchinsonTraceConfig(8, 11))[0]

    first = estimate()
    second = estimate()
    assert first.estimates == second.estimates
    assert first.mean == second.mean


def test_negative_final_trace_is_rejected():
    model = nn.Linear(1, 1, bias=False)
    with pytest.raises(ValueError, match="negative mean"):
        estimate_block_traces(
            (("linear", model.weight),),
            lambda: -model.weight.square().sum(),
            HutchinsonTraceConfig(2, 3))
```

- [ ] **Step 5: Run focused tests**

Run: `PYTHONPATH=. pytest -q tests/test_hawq_trace.py tests/test_cspn_task_loss.py`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add spn_quant/hawq_trace.py tests/test_hawq_trace.py
git commit -m "feat: estimate CSPN HAWQ traces"
```

### Task 5: HAWQ Mixed-Bit Allocation

**Files:**
- Create: `spn_quant/hawq_allocation.py`
- Test: `tests/test_hawq_allocation.py`

- [ ] **Step 1: Write failing objective and budget tests**

```python
from spn_quant.hawq_allocation import (
    HAWQBlock,
    HAWQCandidate,
    solve_hawq_assignment,
)


def test_hawq_solver_selects_sensitive_block_at_eight_bits():
    blocks = (
        HAWQBlock("a", 10, 10, False),
        HAWQBlock("b", 10, 10, False),
    )
    candidates = (
        HAWQCandidate("a", 4, 100.0), HAWQCandidate("a", 6, 10.0),
        HAWQCandidate("a", 8, 0.0), HAWQCandidate("b", 4, 2.0),
        HAWQCandidate("b", 6, 1.0), HAWQCandidate("b", 8, 0.0),
    )
    assignment = solve_hawq_assignment(blocks, candidates, 6.0, 6.0, ())
    assert assignment.block_bits == (("a", 8), ("b", 4))
    assert assignment.average_weight_bits == 6.0
    assert assignment.average_activation_bits == 6.0


def test_fixed_stem_and_depth_head_are_counted_in_budget():
    blocks = (
        HAWQBlock("encoder_stem", 1, 1, True),
        HAWQBlock("body", 3, 3, False),
        HAWQBlock("initial_depth", 1, 1, True),
    )
    candidates = (
        HAWQCandidate("encoder_stem", 4, 8.0),
        HAWQCandidate("encoder_stem", 6, 4.0),
        HAWQCandidate("encoder_stem", 8, 0.0),
        HAWQCandidate("body", 4, 8.0),
        HAWQCandidate("body", 6, 4.0),
        HAWQCandidate("body", 8, 0.0),
        HAWQCandidate("initial_depth", 4, 8.0),
        HAWQCandidate("initial_depth", 6, 4.0),
        HAWQCandidate("initial_depth", 8, 0.0),
    )
    assignment = solve_hawq_assignment(
        blocks, candidates, 6.0, 6.0, ())
    assert assignment.block_bits == (
        ("encoder_stem", 8), ("body", 4), ("initial_depth", 8))
    assert assignment.average_weight_bits == 5.6
    assert assignment.average_activation_bits == 5.6
```

- [ ] **Step 2: Run and verify RED**

Run: `PYTHONPATH=. pytest -q tests/test_hawq_allocation.py`

Expected: `ModuleNotFoundError: spn_quant.hawq_allocation`.

- [ ] **Step 3: Implement candidate costs and SciPy MILP solve**

Create frozen dataclasses `HAWQBlock`, `HAWQCandidate`, and `HAWQAssignment`.
Build one binary variable for every `(block, bit)` row, then call:

```python
result = scipy.optimize.milp(
    c=objective,
    integrality=np.ones(variable_count, dtype=np.int8),
    bounds=scipy.optimize.Bounds(0.0, 1.0),
    constraints=scipy.optimize.LinearConstraint(matrix, lower, upper),
)
```

Constraints require exactly one bit per block, fixed blocks at 8, declared
coupled blocks at equal bit, and the two <=6 weighted budgets. Reject any
unsuccessful solver status or non-integral result. Tie-break equal objectives
by adding a deterministic machine-safe secondary coefficient that prefers
lower block-order bits; record the unmodified HAWQ objective separately.

Implement `candidate_cost(trace, weight, bits, channel_dim)` as normalized
trace times per-output-channel symmetric weight perturbation squared.

- [ ] **Step 4: Add coverage, coupling, and infeasible tests**

Tests must show missing candidate bits raise `ValueError`, a coupled residual
pair receives identical bits, and an impossible fixed-block budget raises
`RuntimeError` with solver status.

- [ ] **Step 5: Run allocation tests**

Run: `PYTHONPATH=. pytest -q tests/test_hawq_allocation.py tests/test_cspn_task_sensitive_bits.py`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add spn_quant/hawq_allocation.py tests/test_hawq_allocation.py
git commit -m "feat: solve CSPN HAWQ mixed precision"
```

### Task 6: CSPN LSQ+/HAWQ Method Controller

**Files:**
- Create: `spn_quant/qat/cspn_methods.py`
- Modify: `spn_quant/qat/__init__.py`
- Test: `tests/test_cspn_method_qat.py`

- [ ] **Step 1: Write failing controller tests**

Build a two-convolution toy instrumentor using the same quantizer dictionaries
as `HardwareAlignedInstrumentor`, then test:

```python
def test_lsqplus_controller_installs_learned_quantizers_and_exports_state():
    controller = build_toy_controller(
        method="lsqplus", weight_bits=(("conv", 4),),
        activation_bits=((('conv', 'input'), 4),))
    controller.initialize_activations((torch.tensor([-1.0, 2.0]),))
    controller.install()
    state = controller.method_state_dict()
    assert "weight.conv.step" in state
    assert "activation.('conv', 'input').step" in state
    assert "activation.('conv', 'input').offset" in state


def test_hawq_controller_uses_exact_assignment_and_freezes_ranges():
    controller = build_toy_controller(
        method="hawq", weight_bits=(("conv", 6),),
        activation_bits=((('conv', 'input'), 6),))
    controller.install()
    controller.freeze_activation_ranges()
    assert controller.manifest()["weight_bits"] == (("conv", 6),)
    assert controller.manifest()["guidance"] == "fp32"
```

Add a test using the official expected registry that asserts guidance modules
are absent and propagation is exactly A8/Q13/INT32 with 24 proxy states.

- [ ] **Step 2: Run and verify RED**

Run: `PYTHONPATH=. pytest -q tests/test_cspn_method_qat.py`

Expected: import fails for `spn_quant.qat.cspn_methods`.

- [ ] **Step 3: Implement focused method controllers**

Implement `CSPNMethodQATConfig` and `CSPNMethodQATController` without changing
the existing `CSPNQATController`. Reuse `CSPNQATPropagationController`.

The controller must:

- install LSQ+ or HAWQ weight parametrizations on declared Conv2d and
  ConvTranspose2d modules;
- replace every declared ordinary, ReLU, and structural boundary quantizer
  with the method quantizer while preserving independent owner instances;
- validate exact owner and bit coverage;
- keep guidance outside the assignment;
- export canonical FP32 master model state separately from learned method state;
- reload method state before hard replay;
- expose manifest, gradient validation, range freeze, and clean removal.

Do not add a generic plugin registry or fallback factory. Use an explicit
`if config.method == "lsqplus"` / `elif config.method == "hawq"` branch and
raise on any other value.

- [ ] **Step 4: Verify controller and legacy QAT tests**

Run: `PYTHONPATH=. pytest -q tests/test_cspn_method_qat.py tests/test_cspn_qat.py tests/test_qat_quantizers.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add spn_quant/qat/cspn_methods.py spn_quant/qat/__init__.py \
  tests/test_cspn_method_qat.py
git commit -m "feat: integrate LSQ+ and HAWQ with CSPN QAT"
```

### Task 7: Official CSPN HAWQ Trace and Assignment Runner

**Files:**
- Create: `scripts/run_nyu_cspn_hawq_trace.py`
- Test: `tests/test_run_nyu_cspn_hawq_trace.py`

- [ ] **Step 1: Write failing protocol tests**

```python
from argparse import Namespace

import pytest

from scripts import run_nyu_cspn_hawq_trace as runner


def test_trace_cli_requires_every_formal_path():
    with pytest.raises(SystemExit):
        runner.parse_args([])


def test_trace_blocks_exclude_guidance_and_propagation():
    blocks = runner.hawq_block_contract(runner.expected_registry())
    names = tuple(block.name for block in blocks)
    assert names[0] == "encoder_stem"
    assert names[-1] == "initial_depth"
    assert not any("guidance" in name or "propagation" in name for name in names)


def test_trace_manifest_requires_128_unique_train_indices():
    with pytest.raises(ValueError, match="128"):
        runner.validate_calibration_indices(tuple(range(127)))
```

- [ ] **Step 2: Run and verify RED**

Run: `PYTHONPATH=. pytest -q tests/test_run_nyu_cspn_hawq_trace.py`

Expected: import fails because the runner does not exist.

- [ ] **Step 3: Implement the strict runner**

Required CLI fields: `--config`, `--checkpoint`, `--data-root`,
`--calibration-metadata`, `--output-root`, and `--device`.

The runner must load/fold the official CSPN once, use the existing registry and
cost capture, evaluate all 128 calibration identities in ordered batches,
compute and append per-probe trace rows, calculate W4/W6/W8 perturbation costs,
solve the assignment, and write:

- `trace_rows.csv`
- `trace_summary.csv`
- `candidate_costs.csv`
- `cost_basis.json`
- `selected_assignment.json`
- `run_manifest.json`

The output directory must be new or empty. All dictionaries use direct
indexing. Trace and solver errors propagate.

- [ ] **Step 4: Add a CPU toy end-to-end runner test**

Factor `estimate_and_allocate(model, batches, blocks, config)` so a two-layer
quadratic toy model produces a complete assignment without NYU or CUDA. Assert
the six artifact payload schemas and exact selected bits.

- [ ] **Step 5: Run focused tests**

Run: `PYTHONPATH=. pytest -q tests/test_run_nyu_cspn_hawq_trace.py tests/test_hawq_trace.py tests/test_hawq_allocation.py`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_nyu_cspn_hawq_trace.py \
  tests/test_run_nyu_cspn_hawq_trace.py
git commit -m "feat: run CSPN HAWQ trace allocation"
```

### Task 8: Shared LSQ+/HAWQ QAT Runner

**Files:**
- Create: `scripts/train_nyu_cspn_lsqplus_hawq.py`
- Test: `tests/test_train_nyu_cspn_lsqplus_hawq.py`

- [ ] **Step 1: Write failing method and checkpoint tests**

```python
import pytest

from scripts import train_nyu_cspn_lsqplus_hawq as runner


def test_training_methods_are_exact():
    assert runner.METHODS == (
        "lsqplus_w4a4", "lsqplus_w6a6", "hawq_mixed_le6")


def test_lsqplus_method_has_uniform_declared_bits():
    assignment = runner.uniform_method_assignment(
        runner.expected_registry(), 4)
    assert set(bits for _, bits in assignment.weight_bits) == {4}
    assert set(bits for _, bits in assignment.activation_bits) == {4}


def test_hawq_training_requires_assignment_path():
    args = runner.parse_args(required_cli("hawq_mixed_le6"))
    with pytest.raises(ValueError, match="assignment"):
        runner.validate_method_paths(args)
```

Add checkpoint tests asserting `model_state`, `method_state`, optimizer,
scheduler, epoch, convergence, assignment, method config, and owner manifest
are required, and any method/assignment mismatch is rejected on resume.

- [ ] **Step 2: Run and verify RED**

Run: `PYTHONPATH=. pytest -q tests/test_train_nyu_cspn_lsqplus_hawq.py`

Expected: import fails because the training runner does not exist.

- [ ] **Step 3: Implement official model preparation and initialization**

Reuse these existing functions rather than duplicating dataset/model logic:

- `saved_checkpoint_args`
- `load_calibration_metadata`
- `build_loaders`
- `prepare_teacher`
- `task_aware_forward`
- `clip_gradients`
- `QATConvergenceTracker`

Prepare the ordinary strict owner path, including `conv1_1`, and exclude
guidance. LSQ+ initialization observes all 128 samples and initializes each
activation owner once from its complete observed min/max. HAWQ loads the exact
serialized assignment, initializes running ranges on the same calibration
manifest, then updates ranges during QAT until the configured final epoch
boundary, where they are frozen explicitly.

- [ ] **Step 4: Implement shared convergence training and checkpointing**

Use the existing task-aware depth/boundary/teacher/propagation loss and
hard-forward propagation proxy. Save `last.pt` every epoch and replace
`best.pt` only on lower finite positive validation RMSE. Method state and FP32
master model state are separate fields. Resume restores both before optimizer
state and requires exact contract equality.

- [ ] **Step 5: Add CUDA one-step smoke test marker**

Add a test guarded only by `torch.cuda.is_available()` that constructs official
CSPN with the existing smoke fixture, runs one LSQ+ train step and one HAWQ
train step, then compares pre-export and reloaded hard predictions. The test
must fail rather than switch to CPU.

- [ ] **Step 6: Run training-focused and legacy tests**

Run: `PYTHONPATH=. pytest -q tests/test_train_nyu_cspn_lsqplus_hawq.py tests/test_train_nyu_cspn_group_a4_qat.py tests/test_cspn_method_qat.py`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/train_nyu_cspn_lsqplus_hawq.py \
  tests/test_train_nyu_cspn_lsqplus_hawq.py
git commit -m "feat: train CSPN LSQ+ and HAWQ"
```

### Task 9: Strict Fixed-64 Evaluation and Figures

**Files:**
- Create: `scripts/evaluate_nyu_cspn_lsqplus_hawq.py`
- Create: `scripts/plot_nyu_cspn_lsqplus_hawq.py`
- Test: `tests/test_evaluate_nyu_cspn_lsqplus_hawq.py`
- Test: `tests/test_plot_nyu_cspn_lsqplus_hawq.py`

- [ ] **Step 1: Write failing matrix and strict aggregation tests**

```python
from scripts import evaluate_nyu_cspn_lsqplus_hawq as evaluator


def test_formal_matrix_is_fixed():
    assert evaluator.CONFIGURATIONS == (
        "FP32", "PA_RTN_W4A4", "PA_RTN_W6A6",
        "LSQPLUS_W4A4", "LSQPLUS_W6A6",
        "HAWQ_MIXED_LE6", "MIXED_TASK_AWARE_QAT")


def test_aggregate_rejects_missing_or_nonpositive_prediction(tmp_path):
    write_complete_shards(tmp_path)
    remove_prediction(tmp_path, "LSQPLUS_W4A4", 63)
    with pytest.raises(RuntimeError, match="coverage"):
        evaluator.aggregate_shards(tmp_path, tuple(range(64)))
```

Add tests for exact sample order, GT identity across shards, strict invalid
pixel handling, paired FP deltas, average HAWQ budgets, and method-state reload.

- [ ] **Step 2: Run and verify RED**

Run: `PYTHONPATH=. pytest -q tests/test_evaluate_nyu_cspn_lsqplus_hawq.py`

Expected: import fails because the evaluator does not exist.

- [ ] **Step 3: Implement per-configuration evaluation workers**

The CLI requires one `--configuration`. Reuse the existing official dataset,
metric, region, propagation, RTN, and prediction-payload helpers. FP/RTN build
fresh official models. LSQ+/HAWQ build fresh official models, load canonical
master weights, install method controllers, load method state, and run hard
QDQ. Existing Mixed-QAT uses its current hard deployment loader after exact
manifest equality.

Each worker writes only `evaluation/shards/<configuration>/` and emits 64 NPZ
files plus sample, region, propagation, activation, and precision rows.

- [ ] **Step 4: Implement strict aggregation**

Aggregation validates seven complete shards, identical ordered sample IDs and
GT arrays, zero non-finite/non-positive predictions, and HAWQ W/A averages
<=6.0. It writes `aggregate_metrics.csv`, `sample_metrics.csv`,
`precision_summary.csv`, `relative_fp_loss.csv`, and `strict_summary.json`.

- [ ] **Step 5: Write failing plotting tests**

Tests create 64 small synthetic payloads, run the plotter, and assert:

- `quantization_rmse_comparison.png` is nonblank;
- `prediction_details.png` contains eight rows and ten visual columns;
- `prediction_contact_sheet.png` uses the shared 0-10 m scale;
- labels are Arial, unrotated, and use `LSQ+ W4A4`, `LSQ+ W6A6`, and
  `HAWQ Mixed<=6`.

- [ ] **Step 6: Implement plotting and run tests**

Run: `PYTHONPATH=. pytest -q tests/test_evaluate_nyu_cspn_lsqplus_hawq.py tests/test_plot_nyu_cspn_lsqplus_hawq.py tests/test_plot_nyu_cspn_group_a4_qat.py`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/evaluate_nyu_cspn_lsqplus_hawq.py \
  scripts/plot_nyu_cspn_lsqplus_hawq.py \
  tests/test_evaluate_nyu_cspn_lsqplus_hawq.py \
  tests/test_plot_nyu_cspn_lsqplus_hawq.py
git commit -m "feat: evaluate CSPN LSQ+ and HAWQ"
```

### Task 10: Four-GPU Orchestration

**Files:**
- Create: `scripts/launch_nyu_cspn_lsqplus_hawq.py`
- Test: `tests/test_launch_nyu_cspn_lsqplus_hawq.py`

- [ ] **Step 1: Write failing stage and GPU tests**

```python
from scripts import launch_nyu_cspn_lsqplus_hawq as launcher


def test_phase_one_uses_all_four_declared_gpus():
    commands = launcher.phase_one_commands(arguments_fixture())
    assert tuple(command.gpu for command in commands) == (0, 1, 2, 3)
    assert tuple(command.name for command in commands) == (
        "baselines", "lsqplus_w4a4", "lsqplus_w6a6", "hawq_mixed_le6")
    assert len(set(command.output_root for command in commands)) == 4


def test_phase_two_evaluation_shards_are_disjoint():
    commands = launcher.phase_two_commands(arguments_fixture())
    outputs = tuple(command.output_root for command in commands)
    assert len(outputs) == len(set(outputs)) == 4
```

- [ ] **Step 2: Run and verify RED**

Run: `PYTHONPATH=. pytest -q tests/test_launch_nyu_cspn_lsqplus_hawq.py`

Expected: import fails because the launcher does not exist.

- [ ] **Step 3: Implement explicit two-phase scheduling**

Phase one starts four `subprocess.Popen` commands:

- GPU 0: fresh FP32 and PA-RTN W4A4/W6A6 shards;
- GPU 1: LSQ+ W4A4 training;
- GPU 2: LSQ+ W6A6 training;
- GPU 3: HAWQ trace, assignment, then HAWQ training.

Phase two starts four evaluation workers for LSQ+ W4A4, LSQ+ W6A6, HAWQ,
and existing Mixed-QAT. Every child receives a single-device
`CUDA_VISIBLE_DEVICES` and uses `--device cuda:0`. Wait for every child and
raise with the exact failed command if any exit code is nonzero. Do not retry,
move a job to another GPU, or reuse an incomplete output.

After phase two, run aggregation and plotting once. Write `jobs.csv` with name,
physical GPU index/UUID, command, PID, timestamps, exit status, and output root.

- [ ] **Step 4: Run launcher tests**

Run: `PYTHONPATH=. pytest -q tests/test_launch_nyu_cspn_lsqplus_hawq.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/launch_nyu_cspn_lsqplus_hawq.py \
  tests/test_launch_nyu_cspn_lsqplus_hawq.py
git commit -m "feat: orchestrate CSPN quantization on four GPUs"
```

### Task 11: Verification, Formal Run, and Result Report

**Files:**
- Create: `docs/2026-08-24-cspn-lsqplus-hawq-results.md`
- Modify only if verification exposes a defect: files introduced in Tasks 1-10

- [ ] **Step 1: Run all focused tests**

Run:

```bash
PYTHONPATH=. pytest -q \
  tests/test_lsqplus_hawq_config.py \
  tests/test_lsqplus_quantizers.py \
  tests/test_hawq_quantizers.py \
  tests/test_hawq_trace.py \
  tests/test_hawq_allocation.py \
  tests/test_cspn_method_qat.py \
  tests/test_run_nyu_cspn_hawq_trace.py \
  tests/test_train_nyu_cspn_lsqplus_hawq.py \
  tests/test_evaluate_nyu_cspn_lsqplus_hawq.py \
  tests/test_plot_nyu_cspn_lsqplus_hawq.py \
  tests/test_launch_nyu_cspn_lsqplus_hawq.py
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the full repository suite**

Run: `PYTHONPATH=. pytest -q`

Expected: at least the baseline `1226 passed, 1 warning, 16 subtests passed`
plus every newly added test, with no failures.

- [ ] **Step 3: Run one-sample CUDA smoke jobs**

Run each method on its assigned GPU with an explicit smoke output beneath
`profile_logs/nyu_cspn_lsqplus_hawq_smoke/`. Require one initialization batch,
one training step, checkpoint reload parity, one evaluation sample, and finite
positive predictions. Delete the smoke runtime directory only after the formal
run succeeds.

- [ ] **Step 4: Launch the formal four-GPU experiment**

Run:

```bash
PYTHONPATH=. python scripts/launch_nyu_cspn_lsqplus_hawq.py \
  --config configs/cspn_lsqplus_hawq.json \
  --checkpoint output/nyu_converged_baselines/cspn_iter24/best.pt \
  --data-root data/nyu_depth_v2 \
  --calibration-metadata profile_logs/nyu_cspn_mixed_task_aware_qat/calibration_metadata.json \
  --mixed-checkpoint profile_logs/nyu_cspn_mixed_task_aware_qat/training/best.pt \
  --mixed-assignment profile_logs/nyu_cspn_mixed_task_aware_qat/search/selected_assignment.json \
  --mixed-cost-basis profile_logs/nyu_cspn_mixed_task_aware_qat/search/cost_basis.json \
  --output-root profile_logs/nyu_cspn_lsqplus_hawq
```

Before execution, resolve the checkpoint/data paths from the existing run
metadata and substitute the exact real paths if these repository-relative
examples differ. Do not add path discovery or fallback logic to production
code.

- [ ] **Step 5: Verify formal artifacts**

Run a deterministic audit command that reads the strict summary and asserts:

- seven configurations;
- exactly 64 ordered rows and prediction files per configuration;
- identical GT arrays;
- zero invalid samples and pixels;
- HAWQ average W <=6 and average A <=6;
- LSQ+ methods report learned step/offset state;
- prediction figures are nonblank and have the expected dimensions.

- [ ] **Step 6: Write the measured result report**

Record exact FP32-relative RMSE/MAE/AbsRel/iRMSE losses, HAWQ selected bits and
budget use, trace ranking and uncertainty, LSQ+ saturation/zero-code behavior,
convergence epochs, and any method that fails its strict numerical gate. Link
the aggregate CSV, strict JSON, and prediction figures by absolute local path.

- [ ] **Step 7: Run final verification and commit**

Run:

```bash
git diff --check
PYTHONPATH=. pytest -q
git status --short
```

Then commit only source, tests, config, and result documentation:

```bash
git add docs/2026-08-24-cspn-lsqplus-hawq-results.md
git commit -m "docs: report CSPN LSQ+ and HAWQ results"
```

Do not commit `profile_logs/` runtime artifacts.
