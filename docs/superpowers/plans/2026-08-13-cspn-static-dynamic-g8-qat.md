# CSPN Static and Dynamic Group-8 W4A4 QAT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add strict CSPN Static-G8 and Dynamic-G8 W4A4 quantization-aware training, then compare exported best checkpoints through the existing hard evaluator on the full NYU validation split and the fixed 64-sample subset.

**Architecture:** Keep `HardwareAlignedInstrumentor` unchanged and wrap its calibrated hard activation quantizers with exact-forward STE adapters. Install W4 as a per-output-channel parametrization over FP32 master weights, and wrap the existing A8/Q13/INT32 CSPN propagation adapter with a differentiable floating proxy. Final metrics come only from canonical checkpoints loaded into fresh official CSPN models with the existing hard path.

**Tech Stack:** Python 3, PyTorch 2.7, CUDA, pytest, NumPy, Matplotlib, official CSPN NYU HDF5 loader.

---

## File Map

- Create `spn_quant/qat/__init__.py`: public QAT API.
- Create `spn_quant/qat/ste.py`: exact hard-forward autograd primitives.
- Create `spn_quant/qat/quantizers.py`: activation STE adapter and W4 parametrization.
- Create `spn_quant/qat/cspn.py`: CSPN weight, activation, export, and propagation controllers.
- Create `scripts/train_nyu_cspn_group_a4_qat.py`: calibration, QAT, convergence, resume, and export.
- Create `scripts/evaluate_nyu_cspn_group_a4_qat.py`: fresh hard-path evaluation.
- Create `scripts/plot_nyu_cspn_group_a4_qat.py`: paired prediction and error figures.
- Create `tests/test_qat_ste.py`, `tests/test_qat_quantizers.py`, `tests/test_cspn_qat.py`, `tests/test_train_nyu_cspn_group_a4_qat.py`, `tests/test_evaluate_nyu_cspn_group_a4_qat.py`, and `tests/test_plot_nyu_cspn_group_a4_qat.py`.
- Create `docs/2026-08-13-cspn-static-dynamic-g8-qat-results.md`: measured results.

The existing PTQ, rotation-boundary, and propagation files are imported but not behaviorally modified.

### Task 1: Exact Hard-Forward STE Primitives

**Files:**
- Create: `spn_quant/qat/__init__.py`
- Create: `spn_quant/qat/ste.py`
- Test: `tests/test_qat_ste.py`

- [ ] **Step 1: Write failing forward and gradient tests**

```python
import pytest
import torch

from spn_quant.qat.ste import hard_forward_proxy, round_ste


def test_hard_forward_proxy_returns_hard_and_routes_proxy_gradient():
    hard = torch.tensor([1.0, -2.0])
    proxy = torch.tensor([0.25, 0.5], requires_grad=True)
    output = hard_forward_proxy(hard, proxy)
    assert torch.equal(output, hard)
    output.sum().backward()
    assert torch.equal(proxy.grad, torch.ones_like(proxy))


def test_round_ste_matches_round_and_has_identity_gradient():
    value = torch.tensor([-1.6, -0.4, 0.4, 1.6], requires_grad=True)
    output = round_ste(value)
    assert torch.equal(output, torch.round(value.detach()))
    output.sum().backward()
    assert torch.equal(value.grad, torch.ones_like(value))


def test_hard_forward_proxy_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="matching shapes"):
        hard_forward_proxy(torch.zeros(2), torch.zeros(3, requires_grad=True))
```

- [ ] **Step 2: Verify the tests fail before implementation**

Run: `pytest -q tests/test_qat_ste.py`

Expected: collection fails with `ModuleNotFoundError: No module named 'spn_quant.qat'`.

- [ ] **Step 3: Implement exact-forward autograd functions**

```python
class _HardForwardProxy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hard, proxy):
        if hard.shape != proxy.shape:
            raise ValueError("hard and proxy tensors require matching shapes")
        return hard

    @staticmethod
    def backward(ctx, gradient):
        return None, gradient


class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return torch.round(value)

    @staticmethod
    def backward(ctx, gradient):
        return gradient


def hard_forward_proxy(hard, proxy):
    if not torch.isfinite(hard).all():
        raise FloatingPointError("hard-forward tensor contains non-finite values")
    if not torch.isfinite(proxy).all():
        raise FloatingPointError("proxy tensor contains non-finite values")
    return _HardForwardProxy.apply(hard, proxy)


def round_ste(value):
    if not torch.isfinite(value).all():
        raise FloatingPointError("rounded tensor contains non-finite values")
    return _RoundSTE.apply(value)
```

Export both functions from `spn_quant/qat/__init__.py` with direct imports and an explicit `__all__` tuple.

- [ ] **Step 4: Run tests and commit**

Run: `pytest -q tests/test_qat_ste.py`

Expected: `3 passed`.

```bash
git add spn_quant/qat/__init__.py spn_quant/qat/ste.py tests/test_qat_ste.py
git commit -m "feat: add exact QAT STE primitives"
```

### Task 2: Activation STE Adapter and W4 Weight Parametrization

**Files:**
- Create: `spn_quant/qat/quantizers.py`
- Modify: `spn_quant/qat/__init__.py`
- Test: `tests/test_qat_quantizers.py`

- [ ] **Step 1: Write failing PTQ parity tests**

```python
import torch

from scripts.hardware_aligned_quantization import (
    DynamicGroupedActivationQuantizer,
    GroupedActivationQuantizer,
    symmetric_weight_qdq,
)
from spn_quant.qat.quantizers import (
    ActivationSTEQuantizer,
    PerOutputChannelWeightFakeQuantizer,
)


def test_static_activation_ste_matches_hard_grouped_ptq():
    hard = GroupedActivationQuantizer(
        4, torch.zeros(2), torch.tensor([3.0, 6.0]), 1, 8, 16, True)
    qat = ActivationSTEQuantizer(hard)
    value = torch.linspace(0.0, 7.0, 32).reshape(1, 16, 1, 2).requires_grad_()
    expected, codes = hard.quantize_with_codes(value.detach())
    actual, actual_codes = qat.quantize_with_codes(value)
    assert torch.equal(actual.detach(), expected)
    assert torch.equal(actual_codes, codes)
    actual.sum().backward()
    assert torch.equal(value.grad, torch.ones_like(value))


def test_dynamic_activation_ste_keeps_per_sample_hard_values():
    hard = DynamicGroupedActivationQuantizer(4, 1, 8, 8, False)
    qat = ActivationSTEQuantizer(hard)
    value = torch.cat((torch.ones(1, 8, 2, 2),
                       torch.full((1, 8, 2, 2), 100.0))).requires_grad_()
    expected, codes = hard.quantize_with_codes(value.detach())
    actual, actual_codes = qat.quantize_with_codes(value)
    assert torch.equal(actual.detach(), expected)
    assert torch.equal(actual_codes, codes)
    actual.square().mean().backward()
    assert torch.isfinite(value.grad).all()


def test_w4_fake_quant_matches_existing_qdq():
    weight = torch.tensor([[[[-3.0, 1.0]]], [[[0.25, 2.0]]]],
                          requires_grad=True)
    quantizer = PerOutputChannelWeightFakeQuantizer(4, 0)
    actual = quantizer(weight)
    expected, scale = symmetric_weight_qdq(weight.detach(), 4, 0)
    assert torch.equal(actual.detach(), expected)
    assert torch.equal(quantizer.scale.detach(), scale)
    actual.sum().backward()
    assert torch.equal(weight.grad, torch.ones_like(weight))
```

- [ ] **Step 2: Verify missing implementation**

Run: `pytest -q tests/test_qat_quantizers.py`

Expected: collection fails because `spn_quant.qat.quantizers` is missing.

- [ ] **Step 3: Implement wrappers without duplicating scale logic**

```python
class ActivationSTEQuantizer(nn.Module):
    def __init__(self, hard_quantizer):
        super().__init__()
        self.hard_quantizer = hard_quantizer
        self.bits = hard_quantizer.bits
        self.format = hard_quantizer.format
        self.unsigned = hard_quantizer.unsigned
        self.qmin = hard_quantizer.qmin
        self.qmax = hard_quantizer.qmax
        self.group_size = hard_quantizer.group_size
        self.zero_point = hard_quantizer.zero_point

    @property
    def scale(self):
        return self.hard_quantizer.scale

    def scale_for(self, tensor):
        return self.hard_quantizer.scale_for(tensor)

    def quantize_with_codes(self, tensor):
        hard, codes = self.hard_quantizer.quantize_with_codes(tensor)
        return hard_forward_proxy(hard, tensor), codes

    def forward(self, tensor):
        return self.quantize_with_codes(tensor)[0]


class PerOutputChannelWeightFakeQuantizer(nn.Module):
    def __init__(self, bits, channel_dim):
        super().__init__()
        self.bits = int(bits)
        self.channel_dim = int(channel_dim)
        if self.bits != 4:
            raise ValueError("CSPN QAT weight quantization requires W4")
        self.register_buffer("scale", torch.empty(0), persistent=False)

    def forward(self, weight):
        hard, scale = symmetric_weight_qdq(
            weight, self.bits, self.channel_dim)
        self.scale = scale.detach()
        return hard_forward_proxy(hard, weight)
```

The activation adapter must wrap existing static or dynamic hard quantizers. Dynamic scale tensors therefore cannot receive gradients, while the current sample still controls forward scale selection.

- [ ] **Step 4: Run tests and commit**

Run: `pytest -q tests/test_qat_quantizers.py tests/test_hardware_aligned_quantization.py -k 'Dynamic or Grouped or qat'`

Expected: all selected tests pass.

```bash
git add spn_quant/qat/__init__.py spn_quant/qat/quantizers.py tests/test_qat_quantizers.py
git commit -m "feat: add strict W4A4 QAT quantizers"
```

### Task 3: CSPN Weight and Activation Controllers

**Files:**
- Create: `spn_quant/qat/cspn.py`
- Test: `tests/test_cspn_qat.py`

- [ ] **Step 1: Write failing lifecycle, owner, and export tests**

```python
from torch.nn.utils import parametrize

from spn_quant.qat.cspn import (
    CSPNActivationQATController,
    CSPNWeightQATController,
)


def test_weight_controller_exports_master_weight_under_standard_key():
    model = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1))
    reference = model[0].weight.detach().clone()
    controller = CSPNWeightQATController(model, ("0",))
    controller.install()
    model(torch.randn(2, 4, 8, 8)).sum().backward()
    assert parametrize.is_parametrized(model[0], "weight")
    assert torch.isfinite(model[0].parametrizations.weight.original.grad).all()
    state = controller.canonical_state_dict()
    assert torch.equal(state["0.weight"], reference)
    assert "0.parametrizations.weight.original" not in state
    controller.remove()
    assert not parametrize.is_parametrized(model[0], "weight")


def test_activation_controller_wraps_exact_strict_owners(strict_fixture):
    ordinary = set(strict_fixture.instrumentor.quantizers) | set(
        strict_fixture.instrumentor.relu_quantizers)
    structural = set(strict_fixture.rotation.active_quantizers)
    controller = CSPNActivationQATController(
        strict_fixture.instrumentor, strict_fixture.rotation)
    controller.install()
    assert controller.ordinary_owners == ordinary
    assert controller.structural_owners == structural
    assert all(isinstance(value, ActivationSTEQuantizer)
               for value in strict_fixture.instrumentor.quantizers.values())
    controller.remove()
```

Build `strict_fixture` from the small CSPN-like model patterns in `tests/test_run_nyu_cspn_activation_resolution.py`; it must not require NYU files or CUDA.

- [ ] **Step 2: Verify the tests fail**

Run: `pytest -q tests/test_cspn_qat.py -k 'weight_controller or activation_controller'`

Expected: collection fails because both controllers are missing.

- [ ] **Step 3: Implement W4 parametrization and canonical export**

```python
class CSPNWeightQATController:
    def __init__(self, model, module_names):
        self.model = model
        self.module_names = tuple(module_names)
        self.modules = {}
        self.installed = False

    def install(self):
        if self.installed:
            raise RuntimeError("CSPN W4 QAT is already installed")
        named = dict(self.model.named_modules())
        for name in self.module_names:
            module = named[name]
            if not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                raise TypeError("unsupported CSPN weight module: %s" % name)
            channel_dim = 1 if isinstance(module, nn.ConvTranspose2d) else 0
            parametrize.register_parametrization(
                module, "weight",
                PerOutputChannelWeightFakeQuantizer(4, channel_dim), unsafe=True)
            self.modules[name] = module
        self.installed = True

    def canonical_state_dict(self):
        if not self.installed:
            raise RuntimeError("CSPN W4 QAT is not installed")
        state = dict((key, value.detach().cpu().clone())
                     for key, value in self.model.state_dict().items())
        for name in self.module_names:
            source = "%s.parametrizations.weight.original" % name
            state["%s.weight" % name] = state[source]
            del state[source]
        return state

    def remove(self):
        if not self.installed:
            raise RuntimeError("CSPN W4 QAT is not installed")
        for name in self.module_names:
            parametrize.remove_parametrizations(
                self.modules[name], "weight", leave_parametrized=False)
        self.modules = {}
        self.installed = False
```

Use direct dictionary indexing for unknown module names so `KeyError` surfaces naturally. Add `qat_state_dict()` to return the parametrized state used only by resumable checkpoints.

- [ ] **Step 4: Implement activation wrapping and restoration**

```python
class CSPNActivationQATController:
    def __init__(self, instrumentor, rotation):
        self.instrumentor = instrumentor
        self.rotation = rotation
        self.installed = False

    def install(self):
        if self.installed:
            raise RuntimeError("CSPN activation QAT is already installed")
        self.original_quantizers = dict(self.instrumentor.quantizers)
        self.original_relu_quantizers = dict(self.instrumentor.relu_quantizers)
        self.original_structural = dict(self.rotation.active_quantizers)
        self.ordinary_owners = set(self.original_quantizers) | set(
            self.original_relu_quantizers)
        if any("gud_up_proj_layer6" in str(owner)
               for owner in self.ordinary_owners):
            raise RuntimeError("guidance cannot be an ordinary QAT owner")
        self.instrumentor.quantizers = {
            key: ActivationSTEQuantizer(value)
            for key, value in self.original_quantizers.items()}
        self.instrumentor.relu_quantizers = {
            key: ActivationSTEQuantizer(value)
            for key, value in self.original_relu_quantizers.items()}
        self.rotation.active_quantizers = {
            key: ActivationSTEQuantizer(value)
            for key, value in self.original_structural.items()}
        self.structural_owners = set(self.original_structural)
        self.installed = True
```

`remove()` restores all three original dictionaries and rejects a second removal. Before installation call `validate_strict_site_contract`; the official model must have 69 ordinary owners and exactly `decoder_entry` and `layer4_signed_skip` structural owners.

- [ ] **Step 5: Run regressions and commit**

Run: `pytest -q tests/test_cspn_qat.py tests/test_run_nyu_cspn_dynamic_group_a4.py tests/test_run_nyu_cspn_activation_resolution.py`

Expected: all selected tests pass.

```bash
git add spn_quant/qat/cspn.py tests/test_cspn_qat.py
git commit -m "feat: control CSPN W4A4 QAT sites"
```

### Task 4: Differentiable A8/Q13/INT32 Propagation

**Files:**
- Modify: `spn_quant/qat/cspn.py`
- Modify: `tests/test_cspn_qat.py`

- [ ] **Step 1: Write failing hard-parity and gradient tests**

```python
@pytest.mark.parametrize("steps", (1, 4, 24))
def test_qat_propagation_matches_hard_and_backpropagates(steps):
    hard, module = configured_hard_cspn_adapter(steps)
    qat = CSPNQATPropagationController(hard)
    qat.install()
    guidance = torch.randn(2, 8, 5, 6, requires_grad=True)
    initial = torch.rand(2, 1, 5, 6, requires_grad=True)
    sparse = torch.zeros_like(initial)
    sparse[:, :, 2, 3] = initial.detach()[:, :, 2, 3]
    actual = module(guidance, initial, sparse)
    expected = qat.hard_result(guidance.detach(), initial.detach(), sparse)
    assert torch.equal(actual.detach(), expected)
    actual.mean().backward()
    assert torch.isfinite(guidance.grad).all()
    assert torch.isfinite(initial.grad).all()
    assert float(guidance.grad.abs().sum()) > 0.0
    assert float(initial.grad.abs().sum()) > 0.0
```

Add assertions that hard adapter statistics report zero anchor max error, zero Q13 coefficient-sum error, and zero contraction violation.

- [ ] **Step 2: Verify missing propagation controller**

Run: `pytest -q tests/test_cspn_qat.py -k propagation`

Expected: tests fail because `CSPNQATPropagationController` is missing.

- [ ] **Step 3: Implement hard forward plus quantized floating proxy**

```python
class CSPNQATPropagationController:
    def __init__(self, hard_adapter):
        self.hard_adapter = hard_adapter
        self.module = hard_adapter.module
        self.hard_forward = hard_adapter.patched_forward
        self.installed = False

    def _proxy(self, guidance, initial, sparse):
        controller = self.hard_adapter.controller
        config = controller.config
        raw = _pad_cspn_channels(guidance)
        raw_hard = symmetric_qdq(
            raw, config.affinity_bits,
            controller.maximum["affinity_raw"])[0]
        raw_qat = hard_forward_proxy(raw_hard, raw)
        denominator = raw_qat.abs().sum(dim=1, keepdim=True)
        neighbor = raw_qat / denominator.clamp_min(torch.finfo(raw.dtype).tiny)
        center = 1.0 - neighbor.sum(dim=1, keepdim=True)
        state = initial
        mask = sparse != 0
        for iteration in range(1, int(self.module.prop_time) + 1):
            propagated = _crop_cspn(
                (neighbor * _pad_cspn_state(state)).sum(dim=1, keepdim=True))
            propagated = propagated + _crop_cspn(center) * initial
            state_hard = symmetric_qdq(
                propagated, config.state_bits,
                controller.maximum["state"])[0]
            state = hard_forward_proxy(state_hard, propagated)
            state = torch.where(mask, initial, state)
        return state

    def _forward(self, guidance, initial, sparse=None):
        hard = self.hard_forward(guidance, initial, sparse)
        proxy = self._proxy(guidance, initial, sparse)
        return hard_forward_proxy(hard, proxy)
```

`install()` replaces only `module.forward`; `remove()` restores `hard_adapter.patched_forward`. `hard_result()` calls the existing hard adapter directly and never switches it to bypass. The final forward tensor must be bit-exact with the current affinity-before-normalization A8, INT16 Q13 coefficient, A8 state, and INT32 accumulator implementation.

- [ ] **Step 4: Run fixed-point regressions and commit**

Run: `pytest -q tests/test_cspn_qat.py -k propagation tests/test_propagation_fixed_point.py tests/test_propagation_aware_adapters.py -k 'cspn or q13'`

Expected: all selected tests pass.

```bash
git add spn_quant/qat/cspn.py tests/test_cspn_qat.py
git commit -m "feat: add propagation-aware CSPN QAT"
```

### Task 5: Unified Controller and Training Runner

**Files:**
- Modify: `spn_quant/qat/cspn.py`
- Modify: `spn_quant/qat/__init__.py`
- Create: `scripts/train_nyu_cspn_group_a4_qat.py`
- Create: `tests/test_train_nyu_cspn_group_a4_qat.py`

- [ ] **Step 1: Write failing configuration, convergence, and resume tests**

```python
from scripts import train_nyu_cspn_group_a4_qat as runner


def test_training_config_requires_every_field():
    values = {
        "epochs": 30, "patience": 6,
        "min_relative_improvement": 0.001, "batch_size": 4,
        "val_batch_size": 1, "workers": 2, "learning_rate": 0.001,
        "momentum": 0.9, "weight_decay": 0.0001, "seed": 20260812,
    }
    assert runner.TrainingConfig(**values).epochs == 30
    del values["patience"]
    with pytest.raises(TypeError):
        runner.TrainingConfig(**values)


def test_tracker_stops_after_six_insignificant_epochs():
    tracker = runner.QATConvergenceTracker(30, 6, 0.001)
    values = (0.3000, 0.2998, 0.29975, 0.2997, 0.29968, 0.29967, 0.29966)
    stopped = [tracker.update(epoch + 1, value)
               for epoch, value in enumerate(values)]
    assert stopped[-1]
    assert tracker.reason == "validation_plateau"


def test_resume_rejects_changed_mode():
    payload = runner.checkpoint_contract("static", "digest", ("owner",))
    with pytest.raises(ValueError, match="quantization mode"):
        runner.validate_resume_contract(
            payload, "dynamic", "digest", ("owner",))
```

- [ ] **Step 2: Verify runner tests fail**

Run: `pytest -q tests/test_train_nyu_cspn_group_a4_qat.py`

Expected: collection fails because the training runner is missing.

- [ ] **Step 3: Implement explicit config and calibration loading**

```python
@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    patience: int
    min_relative_improvement: float
    batch_size: int
    val_batch_size: int
    workers: int
    learning_rate: float
    momentum: float
    weight_decay: float
    seed: int


def load_calibration_indices(path):
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    indices = tuple(int(index) for index in metadata["calibration_indices"])
    if len(indices) != 128:
        raise ValueError("CSPN QAT calibration requires 128 indices")
    if metadata["calibration_source"]["selection"] != "32_tail_96_kmedoids":
        raise ValueError("CSPN QAT requires stratified calibration indices")
    return indices
```

Every CLI experiment value is required: `--mode`, `--checkpoint`, `--data-root`, `--calibration-metadata`, `--output-root`, `--device`, epochs, patience, relative improvement, batch sizes, workers, learning rate, momentum, weight decay, and seed. Do not use parser defaults for these fields.

- [ ] **Step 4: Prepare the exact Static/Dynamic graph**

Load with `_load_cspn`, fold Conv-BN with `prepare_hardware_model`, and calibrate the ordinary instrumentor, identity structural boundaries, and hard propagation adapter on the exact 128 indices. Configure ordinary specs with:

```python
specs = build_activation_specs(
    instrumentor, ORDINARY_GROUPS, 4, 8,
    dynamic=config.mode == "dynamic")
instrumentor.configure_components(
    4, 4, set(), ORDINARY_GROUPS, specs, quantize_bias=False)
hard_propagation.configure(PropagationQuantConfig(
    affinity_bits=8, confidence_bits=8, offset_bits=8,
    state_bits=8, coefficient_fraction_bits=13))
```

Configure both structural boundaries as identity, static Group-8 A4 with `absorb_weights=False`. Select W4 modules from strict instrumentor modules whose group belongs to `ORDINARY_GROUPS`; guidance remains excluded.

- [ ] **Step 5: Implement training, checkpoint, and export**

```python
def train_epoch(model, controller, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for sample in loader:
        model_input, target = batch_to_model_input("cspn", sample, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = extract_pred(model(*model_input))
        loss = masked_l1(prediction, target)
        validate_batch_numerics(prediction, target, loss)
        loss.backward()
        controller.assert_finite_gradients()
        optimizer.step()
        for parameter in model.parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError("QAT parameter contains non-finite values")
        total_loss += float(loss.detach().item()) * target.shape[0]
        total_samples += int(target.shape[0])
    return total_loss / total_samples
```

Use SGD and `ReduceLROnPlateau(mode="min", factor=0.1, patience=3, threshold=1e-4, min_lr=1e-6)`. Stop after six epochs below 0.1% relative validation-RMSE improvement or at epoch 30. Write `last.pt`, `best_qat.pt`, canonical `best.pt`, `metrics.csv`, `manifest.json`, and `run_summary.json`. Resume requires exact mode, calibration SHA-256, owner manifest, and QAT config equality.

- [ ] **Step 6: Run unit and one-batch CUDA smoke tests**

Run: `pytest -q tests/test_qat_ste.py tests/test_qat_quantizers.py tests/test_cspn_qat.py tests/test_train_nyu_cspn_group_a4_qat.py`

Expected: all tests pass.

Run each mode for one epoch with 8 train and 4 validation samples using the Task 8 command arguments plus the sample limits.

Expected: finite loss and validation metrics, nonzero gradients, and both checkpoint formats.

- [ ] **Step 7: Commit controller and trainer**

```bash
git add spn_quant/qat/__init__.py spn_quant/qat/cspn.py scripts/train_nyu_cspn_group_a4_qat.py tests/test_cspn_qat.py tests/test_train_nyu_cspn_group_a4_qat.py
git commit -m "feat: train CSPN Static and Dynamic Group-A4 QAT"
```

### Task 6: Fresh Hard Evaluator and Prediction Plotter

**Files:**
- Create: `scripts/evaluate_nyu_cspn_group_a4_qat.py`
- Create: `scripts/plot_nyu_cspn_group_a4_qat.py`
- Create: `tests/test_evaluate_nyu_cspn_group_a4_qat.py`
- Create: `tests/test_plot_nyu_cspn_group_a4_qat.py`

- [ ] **Step 1: Write failing matrix and payload tests**

```python
EXPECTED = (
    "FP32", "PTQ_STATIC_G8_W4A4", "PTQ_DYNAMIC_G8_W4A4",
    "QAT_STATIC_G8_W4A4", "QAT_DYNAMIC_G8_W4A4",
)


def test_evaluation_matrix_is_fixed():
    assert evaluator.CONFIGURATIONS == EXPECTED


def test_prediction_coverage_requires_same_64_indices(tmp_path):
    indices = tuple(range(64))
    write_prediction_fixture(tmp_path, EXPECTED, indices)
    evaluator.validate_prediction_coverage(tmp_path, indices)
    (tmp_path / EXPECTED[-1] / "sample_00063.npz").unlink()
    with pytest.raises(RuntimeError, match="prediction coverage"):
        evaluator.validate_prediction_coverage(tmp_path, indices)


def test_plot_loader_rejects_nonfinite_payload(tmp_path):
    path = write_prediction_payload(tmp_path, nonfinite=True)
    with pytest.raises(ValueError, match="non-finite"):
        plotter.load_payload(path, "QAT_STATIC_G8_W4A4")
```

- [ ] **Step 2: Verify missing scripts**

Run: `pytest -q tests/test_evaluate_nyu_cspn_group_a4_qat.py tests/test_plot_nyu_cspn_group_a4_qat.py`

Expected: collection fails because both scripts are missing.

- [ ] **Step 3: Implement fresh hard evaluation**

For every configuration build a fresh official CSPN model. FP32/PTQ load the original checkpoint; QAT configurations load canonical `best.pt`. Reject any key containing `.parametrizations.weight.`. Install only the existing hard instrumentor, identity structural boundaries, and hard propagation adapter. The evaluator must not import `spn_quant.qat`.

```python
CONFIGURATIONS = (
    "FP32", "PTQ_STATIC_G8_W4A4", "PTQ_DYNAMIC_G8_W4A4",
    "QAT_STATIC_G8_W4A4", "QAT_DYNAMIC_G8_W4A4",
)


def configuration_mode(name):
    return {
        "FP32": None,
        "PTQ_STATIC_G8_W4A4": "static",
        "PTQ_DYNAMIC_G8_W4A4": "dynamic",
        "QAT_STATIC_G8_W4A4": "static",
        "QAT_DYNAMIC_G8_W4A4": "dynamic",
    }[name]
```

Evaluate the complete validation split and the metadata's exact 64 indices. Save aggregate, sample, activation, block, and propagation CSVs plus 64 NPZ payloads per configuration. Reuse metric and prediction helpers from `run_nyu_cspn_activation_resolution.py`.

- [ ] **Step 4: Implement plots**

Use Arial with Liberation Sans and DejaVu Sans fallbacks. The detail figure shows RGB, sparse depth, GT, FP32, both PTQ predictions/errors, and both QAT predictions/errors. Generate PNG/PDF detail and 64-sample contact-sheet files.

```python
def set_style(font_size):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
    })
```

- [ ] **Step 5: Run tests and commit**

Run: `pytest -q tests/test_evaluate_nyu_cspn_group_a4_qat.py tests/test_plot_nyu_cspn_group_a4_qat.py`

Expected: all tests pass.

```bash
git add scripts/evaluate_nyu_cspn_group_a4_qat.py scripts/plot_nyu_cspn_group_a4_qat.py tests/test_evaluate_nyu_cspn_group_a4_qat.py tests/test_plot_nyu_cspn_group_a4_qat.py
git commit -m "feat: evaluate CSPN Group-A4 QAT on hard path"
```

### Task 7: Hard-Parity Gate and Full Regression

**Files:**
- Modify: `tests/test_cspn_qat.py`

- [ ] **Step 1: Add official-model CUDA parity tests**

```python
@pytest.mark.cuda
@pytest.mark.parametrize("mode", ("static", "dynamic"))
def test_official_cspn_qat_matches_fresh_hard_path(mode, official_fixture):
    qat, hard, input_tensor = official_fixture(mode)
    with torch.no_grad():
        qat_prediction = qat(input_tensor)
        hard_prediction = hard(input_tensor)
    torch.testing.assert_close(
        qat_prediction, hard_prediction, rtol=0.0, atol=0.0)
```

- [ ] **Step 2: Run CUDA parity and complete suite**

Run: `CUDA_VISIBLE_DEVICES=0 pytest -q -m cuda tests/test_cspn_qat.py`

Expected: both official-model parity cases pass bit-exactly.

Run: `pytest -q`

Expected: the previous 748-test baseline plus all new tests passes without regression.

- [ ] **Step 3: Commit the gate**

```bash
git add tests/test_cspn_qat.py
git commit -m "test: verify CSPN QAT hard-path parity"
```

### Task 8: Train, Evaluate, Plot, and Report

**Files:**
- Create: `docs/2026-08-13-cspn-static-dynamic-g8-qat-results.md`

- [ ] **Step 1: Train Static-G8 on GPU 0**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_nyu_cspn_group_a4_qat.py \
  --mode static --device cuda:0 \
  --checkpoint /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24/best.pt \
  --data-root /workspace/CSPN/cspn_pytorch \
  --calibration-metadata /workspace/SPN_Quantization/profile_logs/nyu_cspn_stratified_w4a4_128/stratified128/cspn/metadata.json \
  --output-root /workspace/SPN_Quantization/profile_logs/nyu_cspn_group_a4_qat \
  --epochs 30 --patience 6 --min-relative-improvement 0.001 \
  --batch-size 4 --val-batch-size 1 --workers 2 \
  --learning-rate 0.001 --momentum 0.9 --weight-decay 0.0001 \
  --seed 20260812
```

- [ ] **Step 2: Train Dynamic-G8 concurrently on GPU 1**

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_nyu_cspn_group_a4_qat.py \
  --mode dynamic --device cuda:0 \
  --checkpoint /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24/best.pt \
  --data-root /workspace/CSPN/cspn_pytorch \
  --calibration-metadata /workspace/SPN_Quantization/profile_logs/nyu_cspn_stratified_w4a4_128/stratified128/cspn/metadata.json \
  --output-root /workspace/SPN_Quantization/profile_logs/nyu_cspn_group_a4_qat \
  --epochs 30 --patience 6 --min-relative-improvement 0.001 \
  --batch-size 4 --val-batch-size 1 --workers 2 \
  --learning-rate 0.001 --momentum 0.9 --weight-decay 0.0001 \
  --seed 20260812
```

Run both in managed terminal sessions and wait for completion. Record best epoch, best validation RMSE, and stop reason from each `run_summary.json`; reaching epoch 30 alone is not convergence.

- [ ] **Step 3: Evaluate all five configurations on GPU 2**

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_nyu_cspn_group_a4_qat.py \
  --device cuda:0 \
  --fp32-checkpoint /workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24/best.pt \
  --static-checkpoint /workspace/SPN_Quantization/profile_logs/nyu_cspn_group_a4_qat/static/best.pt \
  --dynamic-checkpoint /workspace/SPN_Quantization/profile_logs/nyu_cspn_group_a4_qat/dynamic/best.pt \
  --data-root /workspace/CSPN/cspn_pytorch \
  --calibration-metadata /workspace/SPN_Quantization/profile_logs/nyu_cspn_stratified_w4a4_128/stratified128/cspn/metadata.json \
  --output-root /workspace/SPN_Quantization/profile_logs/nyu_cspn_group_a4_qat/evaluation \
  --batch-size 1 --workers 2 --seed 20260812
```

Expected: five finite full-validation rows and exactly 320 paired NPZ payloads.

- [ ] **Step 4: Generate figures**

```bash
python scripts/plot_nyu_cspn_group_a4_qat.py \
  --experiment-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_group_a4_qat/evaluation \
  --output-dir /workspace/SPN_Quantization/profile_logs/nyu_cspn_group_a4_qat/figures \
  --expected-samples 64
```

Expected: nonempty detail and contact-sheet PNG/PDF outputs.

- [ ] **Step 5: Write measured results**

Create the result document with these exact sections and fill every table from generated CSV/JSON artifacts:

```markdown
# CSPN Static and Dynamic Group-8 W4A4 QAT Results

## Contract
## Convergence
## Full-Validation Hard-Path Accuracy
## Paired 64-Sample Comparison
## Quantization and Propagation Diagnostics
## Prediction Figures
## Conclusion
```

Report RMSE, MAE, AbsRel, iRMSE, flat RMSE, boundary RMSE, paired better/worse counts, activation SQNR, new-zero and saturation ratios, propagation-step error, anchor max error, and Q13 sum error. Compare each QAT mode only with its matching PTQ baseline and call out Dynamic-G8 iRMSE even if RMSE improves.

- [ ] **Step 6: Run final verification and commit the report**

Run: `pytest -q && git diff --check && git status --short`

Expected: all tests pass; checkpoints, CSVs, NPZs, and figures remain outside Git; only the result document is pending.

```bash
git add docs/2026-08-13-cspn-static-dynamic-g8-qat-results.md
git commit -m "docs: evaluate CSPN Group-A4 QAT"
```

- [ ] **Step 7: Verify final artifact counts and metrics**

```bash
python - <<'PY'
import csv
from pathlib import Path

root = Path('/workspace/SPN_Quantization/profile_logs/nyu_cspn_group_a4_qat/evaluation')
with (root / 'aggregate_metrics.csv').open(newline='') as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 5:
    raise RuntimeError('aggregate configuration count is not five')
for row in rows:
    for key in ('RMSE', 'MAE', 'ABS_REL', 'IRMSE'):
        value = float(row[key])
        if not (value >= 0.0 and value < float('inf')):
            raise FloatingPointError('%s %s is non-finite' % (row['config'], key))
for config in (
        'FP32', 'PTQ_STATIC_G8_W4A4', 'PTQ_DYNAMIC_G8_W4A4',
        'QAT_STATIC_G8_W4A4', 'QAT_DYNAMIC_G8_W4A4'):
    count = len(tuple((root / 'predictions' / config).glob('sample_*.npz')))
    if count != 64:
        raise RuntimeError('%s prediction count is %d' % (config, count))
print('five finite aggregates and 320 paired predictions verified')
PY
```

Expected: `five finite aggregates and 320 paired predictions verified` and a clean worktree.
