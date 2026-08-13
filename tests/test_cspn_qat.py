import importlib.util
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from scripts.hardware_aligned_quantization import (
    HardwareAlignedInstrumentor,
    SymmetricActivationQuantizer,
    UnsignedActivationQuantizer,
    prepare_hardware_model,
)
from scripts import run_nyu_cspn_activation_resolution as resolution
from spn_quant.adapters import install_model_semantic_adapter
from spn_quant.propagation import install_propagation_adapter
from spn_quant.rotation import CSPNRotationController
from spn_quant.qat.cspn import (
    CSPNActivationQATController,
    CSPNQATConfig,
    CSPNQATController,
    CSPNQATPropagationController,
    CSPNWeightQATController,
)
from spn_quant.qat.quantizers import ActivationSTEQuantizer
from spn_quant.propagation.adapters import CSPNPropagationAdapter
from spn_quant.propagation.controller import PropagationQuantConfig


_CSPN_PATH = Path(__file__).resolve().parents[1] / "models" / "cspn.py"
_CSPN_SPEC = importlib.util.spec_from_file_location(
    "qat_official_cspn", _CSPN_PATH)
_CSPN_MODULE = importlib.util.module_from_spec(_CSPN_SPEC)
_CSPN_SPEC.loader.exec_module(_CSPN_MODULE)
Affinity_Propagate = _CSPN_MODULE.Affinity_Propagate


def test_weight_controller_exports_master_weight_under_standard_key():
    model = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1))
    reference = model[0].weight.detach().clone()
    controller = CSPNWeightQATController(model, ("0",))

    controller.install()
    model(torch.randn(2, 4, 8, 8)).sum().backward()

    assert parametrize.is_parametrized(model[0], "weight")
    master = model[0].parametrizations.weight.original
    assert torch.isfinite(master.grad).all()
    state = controller.canonical_state_dict()
    assert torch.equal(state["0.weight"], reference)
    assert "0.parametrizations.weight.original" not in state

    fresh = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1))
    fresh.load_state_dict(state)
    assert torch.equal(fresh[0].weight, reference)

    controller.remove()
    assert not parametrize.is_parametrized(model[0], "weight")
    assert torch.equal(model[0].weight, reference)


def test_weight_controller_rejects_unknown_and_duplicate_install():
    model = nn.Sequential(nn.Conv2d(4, 8, 1))
    with pytest.raises(KeyError):
        CSPNWeightQATController(model, ("missing",)).install()

    controller = CSPNWeightQATController(model, ("0",))
    controller.install()
    with pytest.raises(RuntimeError, match="already installed"):
        controller.install()
    controller.remove()


def _activation_fixture():
    input_quantizer = SymmetricActivationQuantizer(bits=4, maximum=2.0)
    relu_quantizer = UnsignedActivationQuantizer(bits=4, maximum=3.0)
    boundary_quantizer = SymmetricActivationQuantizer(bits=4, maximum=4.0)
    instrumentor = SimpleNamespace(
        quantizers={("encoder.conv", "input"): input_quantizer},
        relu_quantizers={"encoder.relu#0": relu_quantizer},
    )
    rotation = SimpleNamespace(
        active_quantizers={"decoder_entry": boundary_quantizer})
    return instrumentor, rotation


def test_activation_controller_wraps_and_restores_exact_owners():
    instrumentor, rotation = _activation_fixture()
    original_quantizers = dict(instrumentor.quantizers)
    original_relu_quantizers = dict(instrumentor.relu_quantizers)
    original_structural = dict(rotation.active_quantizers)
    controller = CSPNActivationQATController(instrumentor, rotation)

    controller.install()

    assert controller.ordinary_owners == {
        ("encoder.conv", "input"), "encoder.relu#0"}
    assert controller.structural_owners == {"decoder_entry"}
    assert all(isinstance(value, ActivationSTEQuantizer)
               for value in instrumentor.quantizers.values())
    assert all(isinstance(value, ActivationSTEQuantizer)
               for value in instrumentor.relu_quantizers.values())
    assert all(isinstance(value, ActivationSTEQuantizer)
               for value in rotation.active_quantizers.values())

    controller.remove()

    assert instrumentor.quantizers == original_quantizers
    assert instrumentor.relu_quantizers == original_relu_quantizers
    assert rotation.active_quantizers == original_structural
    with pytest.raises(RuntimeError, match="not installed"):
        controller.remove()


def test_activation_controller_rejects_guidance_owner():
    instrumentor, rotation = _activation_fixture()
    instrumentor.quantizers[("gud_up_proj_layer6", "output")] = \
        SymmetricActivationQuantizer(bits=4, maximum=1.0)
    controller = CSPNActivationQATController(instrumentor, rotation)

    with pytest.raises(RuntimeError, match="guidance"):
        controller.install()


def _configured_hard_cspn_adapter(steps):
    torch.manual_seed(31)
    module = Affinity_Propagate(steps, 3, "8sum").eval()
    adapter = CSPNPropagationAdapter(module)
    guidance = torch.randn(2, 8, 5, 6)
    initial = torch.rand(2, 1, 5, 6) * 2.0
    sparse = torch.zeros_like(initial)
    sparse[:, :, 2, 3] = initial[:, :, 2, 3]
    adapter.observe()
    module(guidance, initial, sparse)
    adapter.freeze()
    adapter.configure(PropagationQuantConfig(
        affinity_bits=8,
        confidence_bits=8,
        offset_bits=8,
        state_bits=8,
        coefficient_fraction_bits=13,
    ))
    return adapter, module, guidance, initial, sparse


@pytest.mark.parametrize("steps", (1, 4, 24))
def test_qat_propagation_matches_hard_and_backpropagates(steps):
    adapter, module, guidance, initial, sparse = \
        _configured_hard_cspn_adapter(steps)
    expected = module(guidance, initial, sparse).detach().clone()
    qat = CSPNQATPropagationController(adapter)
    qat.install()
    guidance = guidance.requires_grad_()
    initial = initial.requires_grad_()

    actual = module(guidance, initial, sparse)

    assert torch.equal(actual.detach(), expected)
    actual.mean().backward()
    assert torch.isfinite(guidance.grad).all()
    assert torch.isfinite(initial.grad).all()
    assert float(guidance.grad.abs().max().item()) < 1e6
    assert float(guidance.grad.abs().sum().item()) > 0.0
    assert float(initial.grad.abs().sum().item()) > 0.0
    qat.remove()
    adapter.close()


def test_qat_propagation_preserves_q13_and_anchor_constraints():
    adapter, module, guidance, initial, sparse = \
        _configured_hard_cspn_adapter(4)
    qat = CSPNQATPropagationController(adapter)
    qat.install()

    output = module(guidance, initial, sparse)

    mask = sparse != 0
    assert torch.equal(output[mask], initial[mask])
    constraints = [
        row for row in adapter.statistics()
        if row["signal"] == "affinity_constraints"
    ]
    anchors = [
        row for row in adapter.statistics()
        if row["signal"] == "anchor"
    ]
    assert constraints[0]["coefficient_sum_max_error"] == 0.0
    assert constraints[0]["contraction_violation_rate"] == 0.0
    assert all(row["anchor_max_error"] == 0.0 for row in anchors)
    qat.remove()
    adapter.close()


def test_qat_propagation_zero_affinity_has_finite_gradients():
    adapter, module, _, initial, _ = _configured_hard_cspn_adapter(24)
    qat = CSPNQATPropagationController(adapter)
    qat.install()
    quantization_step = adapter.controller.maximum["affinity_raw"] / 127.0
    guidance = torch.full(
        (2, 8, 5, 6), quantization_step * 0.25,
        requires_grad=True)
    initial = initial.requires_grad_()

    output = module(guidance, initial)
    output.square().mean().backward()

    assert torch.isfinite(guidance.grad).all()
    assert torch.isfinite(initial.grad).all()
    assert float(guidance.grad.abs().max().item()) < 1e6
    qat.remove()
    adapter.close()


def test_qat_config_rejects_non_strict_precision():
    propagation = PropagationQuantConfig(
        affinity_bits=8, confidence_bits=8, offset_bits=8,
        state_bits=8, coefficient_fraction_bits=13)
    with pytest.raises(ValueError, match="W4A4"):
        CSPNQATConfig(
            mode="static", weight_bits=8, activation_bits=4,
            group_size=8, propagation=propagation)
    with pytest.raises(ValueError, match="static or dynamic"):
        CSPNQATConfig(
            mode="other", weight_bits=4, activation_bits=4,
            group_size=8, propagation=propagation)


def test_unified_qat_controller_lifecycle_and_manifest():
    model = nn.Sequential(nn.Conv2d(4, 8, 1))
    instrumentor, rotation = _activation_fixture()
    hard, _, _, _, _ = _configured_hard_cspn_adapter(1)
    config = CSPNQATConfig(
        mode="static", weight_bits=4, activation_bits=4,
        group_size=8, propagation=PropagationQuantConfig(
            affinity_bits=8, confidence_bits=8, offset_bits=8,
            state_bits=8, coefficient_fraction_bits=13))
    controller = CSPNQATController(
        model, instrumentor, rotation, hard, ("0",), config)

    controller.install()
    model(torch.randn(1, 4, 2, 2)).sum().backward()
    controller.assert_finite_gradients()
    manifest = controller.manifest()

    assert manifest["mode"] == "static"
    assert manifest["guidance"] == "fp32"
    assert manifest["bias"] == "fp32"
    assert manifest["weight_modules"] == ["0"]
    assert "0.weight" in controller.canonical_state_dict()
    controller.remove()
    hard.close()


def _official_hard_and_qat_models(mode):
    import torch_resnet_cspn_nyu

    torch.manual_seed(77)
    device = torch.device("cuda:0")
    source = torch_resnet_cspn_nyu.resnet18(
        pretrained=False,
        cspn_config={"step": 24, "kernel": 3, "norm_type": "8sum"},
    ).to(device).eval()
    model_input = torch.randn(1, 4, 228, 304, device=device)
    model_input[:, 3].abs_().mul_(0.1)
    preparation = prepare_hardware_model(
        source, (model_input,), excluded_pairs=(("conv1_1", "bn1"),))
    hard_model = copy.deepcopy(source)
    qat_model = copy.deepcopy(source)

    def components(model):
        semantic = install_model_semantic_adapter(
            model, "cspn", strict=True)
        boundaries = semantic.rotation_boundaries()
        semantic.close()
        instrumentor = HardwareAlignedInstrumentor(
            model, resolution.cspn_quant_group,
            preparation["fused_relu_producers"],
            externally_owned_outputs=resolution.strict_owned_outputs(),
            externally_owned_inputs=resolution.strict_owned_inputs())
        rotation = CSPNRotationController(model, boundaries, seed=77)
        propagation = install_propagation_adapter("cspn", model)
        instrumentor.observe()
        rotation.observe()
        propagation.observe()
        with torch.no_grad():
            model(model_input)
        instrumentor.freeze()
        rotation.freeze()
        propagation.freeze()
        resolution.validate_strict_site_contract(instrumentor, rotation)
        return instrumentor, rotation, propagation

    hard_instrumentor, hard_rotation, hard_propagation = \
        components(hard_model)
    qat_instrumentor, qat_rotation, qat_propagation = components(qat_model)
    config = resolution._configuration(
        "hard", resolution.ORDINARY_GROUPS, resolution.ORDINARY_GROUPS,
        resolution.PROPAGATION_A8_Q13,
        granularity="hybrid_group_tensor", group_size=8,
        dynamic=mode == "dynamic")
    resolution._configure_quantized(
        config, hard_instrumentor, hard_rotation, hard_propagation, {})

    specs = resolution.build_activation_specs(
        qat_instrumentor, resolution.ORDINARY_GROUPS, 4, 8,
        dynamic=mode == "dynamic")
    qat_instrumentor.configure_components(
        4, 4, set(), resolution.ORDINARY_GROUPS,
        specs, quantize_bias=False)
    rotation_specs = resolution.build_rotation_activation_specs(
        qat_rotation, 4, 8)
    methods = {
        "decoder_entry": "identity",
        "layer4_signed_skip": "identity",
    }
    bit_widths = {}
    group_sizes = {}
    scale_factors = {}
    for name in qat_rotation.channels:
        spec = rotation_specs[("rotation.%s" % name, "boundary")]
        bit_widths[name] = int(spec.bits)
        group_sizes[name] = int(spec.group_size)
        scale_factors[name] = 1.0
    qat_rotation.configure_specs(
        methods, bit_widths, group_sizes, scale_factors,
        quantize=True, absorb_weights=False)
    propagation_config = PropagationQuantConfig(
        affinity_bits=8, confidence_bits=8, offset_bits=8,
        state_bits=8, coefficient_fraction_bits=13)
    qat_propagation.configure(propagation_config)
    weight_modules = tuple(sorted(
        name for name in qat_instrumentor.modules
        if qat_instrumentor.groups[name] in resolution.ORDINARY_GROUPS))
    controller = CSPNQATController(
        qat_model, qat_instrumentor, qat_rotation, qat_propagation,
        weight_modules, CSPNQATConfig(
            mode=mode, weight_bits=4, activation_bits=4,
            group_size=8, propagation=propagation_config))
    controller.install()
    return hard_model, qat_model, model_input, controller, (
        hard_instrumentor, hard_rotation, hard_propagation,
        qat_instrumentor, qat_rotation, qat_propagation,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("mode", ("static", "dynamic"))
def test_official_cspn_qat_matches_fresh_hard_path(mode):
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hard, qat, model_input, controller, components = \
        _official_hard_and_qat_models(mode)

    with torch.no_grad():
        hard_prediction = hard(model_input)
        qat_prediction = qat(model_input)

    assert torch.equal(qat_prediction, hard_prediction)
    controller.remove()
    for component in components:
        component.close()
