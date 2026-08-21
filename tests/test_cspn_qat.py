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
from spn_quant.activation_boundaries import CSPNActivationBoundaryController
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
    controller = CSPNWeightQATController(model, (("0", 4),))

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
        CSPNWeightQATController(model, (("missing", 4),)).install()

    controller = CSPNWeightQATController(model, (("0", 4),))
    controller.install()
    with pytest.raises(RuntimeError, match="already installed"):
        controller.install()
    controller.remove()


def _activation_fixture():
    input_quantizer = SymmetricActivationQuantizer(bits=6, maximum=2.0)
    relu_quantizer = UnsignedActivationQuantizer(bits=4, maximum=3.0)
    boundary_quantizer = SymmetricActivationQuantizer(bits=8, maximum=4.0)
    instrumentor = SimpleNamespace(
        quantizers={("encoder.conv", "input"): input_quantizer},
        relu_quantizers={"encoder.relu#0": relu_quantizer},
    )
    boundary_controller = SimpleNamespace(
        active_quantizers={"decoder_entry": boundary_quantizer})
    return instrumentor, boundary_controller


def _activation_bits():
    return (
        (("boundary_controller.decoder_entry", "boundary"), 8),
        (("encoder.conv", "input"), 6),
        (("encoder.relu#0", "relu_output"), 4),
    )


def test_activation_controller_wraps_and_restores_exact_owners():
    instrumentor, boundary_controller = _activation_fixture()
    original_quantizers = dict(instrumentor.quantizers)
    original_relu_quantizers = dict(instrumentor.relu_quantizers)
    original_structural = dict(boundary_controller.active_quantizers)
    controller = CSPNActivationQATController(
        instrumentor, boundary_controller, _activation_bits())

    controller.install()

    assert controller.ordinary_owners == {
        ("encoder.conv", "input"),
        ("encoder.relu#0", "relu_output")}
    assert controller.structural_owners == {"decoder_entry"}
    assert all(isinstance(value, ActivationSTEQuantizer)
               for value in instrumentor.quantizers.values())
    assert all(isinstance(value, ActivationSTEQuantizer)
               for value in instrumentor.relu_quantizers.values())
    assert all(isinstance(value, ActivationSTEQuantizer)
               for value in boundary_controller.active_quantizers.values())

    controller.remove()

    assert instrumentor.quantizers == original_quantizers
    assert instrumentor.relu_quantizers == original_relu_quantizers
    assert boundary_controller.active_quantizers == original_structural
    with pytest.raises(RuntimeError, match="not installed"):
        controller.remove()


def test_activation_controller_rejects_guidance_owner():
    instrumentor, boundary_controller = _activation_fixture()
    instrumentor.quantizers[("gud_up_proj_layer6", "output")] = \
        SymmetricActivationQuantizer(bits=4, maximum=1.0)
    activation_bits = _activation_bits() + (
        (("gud_up_proj_layer6", "output"), 4),)
    controller = CSPNActivationQATController(
        instrumentor, boundary_controller, activation_bits)

    with pytest.raises(RuntimeError, match="guidance"):
        controller.install()


def test_activation_controller_rejects_declared_hard_bit_mismatch():
    instrumentor, boundary_controller = _activation_fixture()
    mismatched = tuple(
        (owner, 4 if owner == ("encoder.conv", "input") else bits)
        for owner, bits in _activation_bits())
    controller = CSPNActivationQATController(
        instrumentor, boundary_controller, mismatched)

    with pytest.raises(ValueError, match="hard quantizers"):
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


def test_qat_propagation_exposes_all_proxy_states_with_gradients():
    adapter, module, guidance, initial, sparse = \
        _configured_hard_cspn_adapter(24)
    qat = CSPNQATPropagationController(adapter)
    qat.install()
    guidance = guidance.requires_grad_()
    initial = initial.requires_grad_()

    output = module(guidance, initial, sparse)
    states = qat.proxy_states()

    assert len(states) == 24
    assert all(state.device == output.device for state in states)
    assert all(state.requires_grad and state.grad_fn is not None
               for state in states)
    states[-1].mean().backward()
    assert guidance.grad is not None
    assert initial.grad is not None
    qat.remove()
    adapter.close()


def test_qat_config_rejects_invalid_mixed_precision():
    propagation = PropagationQuantConfig(
        affinity_bits=8, confidence_bits=8, offset_bits=8,
        state_bits=8, coefficient_fraction_bits=13)
    with pytest.raises(ValueError, match="weight bits"):
        CSPNQATConfig(
            mode="static", weight_bits=(("0", 6),),
            activation_bits=_activation_bits(),
            group_size=8, propagation=propagation)
    with pytest.raises(ValueError, match="static, dynamic, or mixed_static"):
        CSPNQATConfig(
            mode="other", weight_bits=(("0", 4),),
            activation_bits=_activation_bits(),
            group_size=8, propagation=propagation)


def test_mixed_controller_preserves_independent_decoder_scales():
    model = nn.Sequential(
        nn.Conv2d(4, 8, 1),
        nn.ConvTranspose2d(8, 8, 2, stride=2),
    )
    upsample = SymmetricActivationQuantizer(bits=4, maximum=2.0)
    merged = SymmetricActivationQuantizer(bits=6, maximum=12.0)
    skip = SymmetricActivationQuantizer(bits=8, maximum=0.5)
    instrumentor = SimpleNamespace(
        quantizers={
            ("decoder.up", "output"): upsample,
            ("decoder.merge", "input"): merged,
        },
        relu_quantizers={},
    )
    boundary_controller = SimpleNamespace(
        active_quantizers={"layer4_signed_skip": skip})
    hard, _, _, _, _ = _configured_hard_cspn_adapter(1)
    activation_bits = (
        (("boundary_controller.layer4_signed_skip", "boundary"), 8),
        (("decoder.merge", "input"), 6),
        (("decoder.up", "output"), 4),
    )
    config = CSPNQATConfig(
        mode="static",
        weight_bits=(("0", 4), ("1", 8)),
        activation_bits=activation_bits,
        group_size=8,
        propagation=PropagationQuantConfig(
            affinity_bits=8, confidence_bits=8, offset_bits=8,
            state_bits=8, coefficient_fraction_bits=13),
    )
    expected_scales = {
        "upsample": upsample.scale_for(torch.zeros(1)),
        "merged": merged.scale_for(torch.zeros(1)),
        "skip": skip.scale_for(torch.zeros(1)),
    }
    controller = CSPNQATController(
        model, instrumentor, boundary_controller, hard, config)

    controller.install()

    assert instrumentor.quantizers[("decoder.up", "output")].bits == 4
    assert instrumentor.quantizers[("decoder.merge", "input")].bits == 6
    assert boundary_controller.active_quantizers[
        "layer4_signed_skip"].bits == 8
    assert model[0].parametrizations.weight[0].bits == 4
    assert model[1].parametrizations.weight[0].bits == 8
    assert instrumentor.quantizers[(
        "decoder.up", "output")].scale_for(
            torch.zeros(1)) == expected_scales["upsample"]
    assert instrumentor.quantizers[(
        "decoder.merge", "input")].scale_for(
            torch.zeros(1)) == expected_scales["merged"]
    assert boundary_controller.active_quantizers[
        "layer4_signed_skip"].scale_for(
            torch.zeros(1)) == expected_scales["skip"]
    manifest = controller.manifest()
    assert manifest["weight_bits"] == config.weight_bits
    assert manifest["activation_bits"] == config.activation_bits
    assert manifest["guidance"] == "fp32"
    assert manifest["bias"] == "fp32"
    controller.remove()
    hard.close()


def test_unified_qat_controller_lifecycle_and_manifest():
    model = nn.Sequential(nn.Conv2d(4, 8, 1))
    instrumentor, boundary_controller = _activation_fixture()
    hard, _, _, _, _ = _configured_hard_cspn_adapter(1)
    config = CSPNQATConfig(
        mode="static", weight_bits=(("0", 4),),
        activation_bits=_activation_bits(),
        group_size=8, propagation=PropagationQuantConfig(
            affinity_bits=8, confidence_bits=8, offset_bits=8,
            state_bits=8, coefficient_fraction_bits=13))
    controller = CSPNQATController(
        model, instrumentor, boundary_controller, hard, config)

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
        boundaries = semantic.activation_boundaries()
        semantic.close()
        instrumentor = HardwareAlignedInstrumentor(
            model, resolution.cspn_quant_group,
            preparation["fused_relu_producers"],
            externally_owned_outputs=resolution.strict_owned_outputs(),
            externally_owned_inputs=resolution.strict_owned_inputs())
        boundary_controller = CSPNActivationBoundaryController(model, boundaries)
        propagation = install_propagation_adapter("cspn", model)
        instrumentor.observe()
        boundary_controller.observe()
        propagation.observe()
        with torch.no_grad():
            model(model_input)
        instrumentor.freeze()
        boundary_controller.freeze()
        propagation.freeze()
        resolution.validate_strict_site_contract(instrumentor, boundary_controller)
        return instrumentor, boundary_controller, propagation

    hard_instrumentor, hard_boundary, hard_propagation = \
        components(hard_model)
    qat_instrumentor, qat_boundary, qat_propagation = components(qat_model)
    config = resolution._configuration(
        "hard", resolution.ORDINARY_GROUPS, resolution.ORDINARY_GROUPS,
        resolution.PROPAGATION_A8_Q13,
        granularity="hybrid_group_tensor", group_size=8,
        dynamic=mode == "dynamic")
    resolution._configure_quantized(
        config, hard_instrumentor, hard_boundary, hard_propagation, {})

    specs = resolution.build_activation_specs(
        qat_instrumentor, resolution.ORDINARY_GROUPS, 4, 8,
        dynamic=mode == "dynamic")
    qat_instrumentor.configure_components(
        4, 4, set(), resolution.ORDINARY_GROUPS,
        specs, quantize_bias=False)
    boundary_specs = resolution.build_boundary_activation_specs(
        qat_boundary, 4, 8)
    bit_widths = {}
    group_sizes = {}
    scale_factors = {}
    for name in qat_boundary.channels:
        spec = boundary_specs[("boundary_controller.%s" % name, "boundary")]
        bit_widths[name] = int(spec.bits)
        group_sizes[name] = int(spec.group_size)
        scale_factors[name] = 1.0
    qat_boundary.configure_specs(
        bit_widths, group_sizes, scale_factors, quantize=True)
    propagation_config = PropagationQuantConfig(
        affinity_bits=8, confidence_bits=8, offset_bits=8,
        state_bits=8, coefficient_fraction_bits=13)
    qat_propagation.configure(propagation_config)
    weight_modules = tuple(sorted(
        name for name in qat_instrumentor.modules
        if qat_instrumentor.groups[name] in resolution.ORDINARY_GROUPS))
    activation_bits = tuple(sorted(
        tuple((owner, quantizer.bits)
              for owner, quantizer in qat_instrumentor.quantizers.items()) +
        tuple(((owner, "relu_output"), quantizer.bits)
              for owner, quantizer in
              qat_instrumentor.relu_quantizers.items()) +
        tuple((("boundary_controller.%s" % owner, "boundary"),
               quantizer.bits)
              for owner, quantizer in
              qat_boundary.active_quantizers.items()),
        key=str))
    controller = CSPNQATController(
        qat_model, qat_instrumentor, qat_boundary, qat_propagation,
        CSPNQATConfig(
            mode=mode,
            weight_bits=tuple((name, 4) for name in weight_modules),
            activation_bits=activation_bits,
            group_size=8, propagation=propagation_config))
    controller.install()
    return hard_model, qat_model, model_input, controller, (
        hard_instrumentor, hard_boundary, hard_propagation,
        qat_instrumentor, qat_boundary, qat_propagation,
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
