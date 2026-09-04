from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

from spn_quant.model_contracts import (
    QuantizationBlock,
    QuantizationModelContract,
)
from spn_quant.propagation import PropagationQuantConfig
from spn_quant.propagation.controller import PropagationQuantController
from spn_quant.qdrop_targets import QDropActivationSite, QDropTargetPlan
from spn_quant.qat.model_methods import (
    FixedEpochQATProtocol,
    ModelHardDeploymentController,
    ModelMethodQATConfig,
    ModelMethodQATController,
)


def test_gradient_validation_uses_one_aggregate_device_barrier(monkeypatch):
    first = nn.Parameter(torch.ones(2))
    second = nn.Parameter(torch.ones(3))
    first.grad = torch.full_like(first, 2.0)
    second.grad = torch.full_like(second, 3.0)
    controller = SimpleNamespace(
        installed=True,
        named_parameters=lambda: (("first", first), ("second", second)),
    )
    monkeypatch.setattr(
        torch.Tensor, "item",
        lambda self: (_ for _ in ()).throw(
            AssertionError("per-gradient device synchronization")))

    norm = ModelMethodQATController.assert_finite_gradients(controller)

    assert norm == pytest.approx((2 * 4.0 + 3 * 9.0) ** 0.5)


def test_gradient_validation_identifies_nonfinite_parameter():
    parameter = nn.Parameter(torch.ones(2))
    parameter.grad = torch.tensor([1.0, float("nan")])
    controller = SimpleNamespace(
        installed=True,
        named_parameters=lambda: (("broken", parameter),),
    )

    with pytest.raises(FloatingPointError, match="broken"):
        ModelMethodQATController.assert_finite_gradients(controller)


def test_fixed_epoch_protocol_never_stops_or_selects_by_validation():
    protocol = FixedEpochQATProtocol(epochs=5)

    assert tuple(protocol.should_continue(epoch)
                 for epoch in range(1, 6)) == \
        (True, True, True, True, False)
    assert not protocol.publish_best_checkpoint
    protocol.validate_final_epoch(5)


def test_fixed_epoch_protocol_rejects_incomplete_training():
    protocol = FixedEpochQATProtocol(epochs=5)

    with pytest.raises(RuntimeError, match="configured final epoch"):
        protocol.validate_final_epoch(4)


def test_model_qat_fp16_propagation_has_no_integer_qparams():
    propagation = PropagationQuantController()
    propagation.observe()
    propagation.observe_signal("state", torch.ones(1))
    propagation.freeze()
    propagation.configure_fp16()
    adapter = SimpleNamespace(controller=propagation)
    base = _config()
    config = ModelMethodQATConfig(
        method=base.method,
        weight_bits=base.weight_bits,
        activation_bits=base.activation_bits,
        fp16_weight_modules=(),
        fp16_activation_owners=(),
        propagation=None,
        propagation_mode="fp16",
        hawq_range_momentum=base.hawq_range_momentum,
    )
    controller = ModelMethodQATController(
        ToyModel(), _contract(), _sites(), config,
        propagation_adapter=adapter)
    controller.initialize_activations(_initialization_rows())

    controller.install()

    assert controller.deployment_qparams()["propagation"] == {
        "mode": "fp16"}
    assert not tuple(
        name for name in controller.method_state_dict()
        if name.startswith("propagation."))
    controller.remove()


def test_hard_controller_restores_fp16_propagation_without_calibration():
    base = _config()
    config = ModelMethodQATConfig(
        method=base.method,
        weight_bits=base.weight_bits,
        activation_bits=base.activation_bits,
        fp16_weight_modules=(),
        fp16_activation_owners=(),
        propagation=None,
        propagation_mode="fp16",
        hawq_range_momentum=base.hawq_range_momentum,
    )
    training_propagation = PropagationQuantController()
    training_propagation.observe()
    training_propagation.observe_signal("state", torch.ones(1))
    training_propagation.freeze()
    training_propagation.configure_fp16()
    training = ModelMethodQATController(
        ToyModel(), _contract(), _sites(), config,
        propagation_adapter=SimpleNamespace(
            controller=training_propagation))
    training.initialize_activations(_initialization_rows())
    training.install()
    qparams = training.deployment_qparams()
    deployment_propagation = PropagationQuantController()

    deployed = ModelHardDeploymentController(
        ToyModel(), _contract(), _sites(), config, qparams,
        propagation_adapter=SimpleNamespace(
            controller=deployment_propagation))

    assert deployment_propagation.mode == "float"
    assert deployment_propagation.float_state_dtype == torch.float16
    training.remove()


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(1, 2, 1)
        self.decoder = nn.Conv2d(2, 1, 1)
        self.propagation = nn.Conv2d(1, 1, 1)

    def forward(self, value):
        return self.decoder(torch.relu(self.encoder(value)))


def _contract():
    return QuantizationModelContract(
        model_name="nlspn",
        blocks=(
            QuantizationBlock(
                "encoder", ("encoder",),
                (("activation::encoder::input", "module_input"),)),
            QuantizationBlock(
                "decoder", ("decoder",),
                (("activation::decoder::input", "module_input"),)),
        ),
        prefix_groups=(("encoder",),),
        tail_groups=(("decoder",),),
        protected_roles=(
            "guidance_logits", "confidence", "offset", "affinity",
            "normalization", "sparse_anchor", "propagation_state",
        ),
        attention_edges=(),
        concat_edges=(),
        protected_modules=("propagation",),
        module_roles=(("propagation", "affinity"),),
    )


def _sites():
    return QDropTargetPlan(
        model="nlspn",
        blocks=("decoder", "encoder"),
        activation_sites=(
            QDropActivationSite(
                "activation::decoder::input", "decoder", "module_input",
                "module_input", True, True),
            QDropActivationSite(
                "activation::encoder::input", "encoder", "module_input",
                "module_input", True, True),
        ),
        excluded_sites=(),
    )


def _propagation():
    return PropagationQuantConfig(
        affinity_bits=8,
        confidence_bits=8,
        offset_bits=8,
        state_bits=8,
        coefficient_fraction_bits=13,
    )


def _config(method="lsqplus", bits=4):
    contract = _contract()
    return ModelMethodQATConfig(
        method=method,
        weight_bits=tuple((name, bits) for name in contract.weight_modules),
        activation_bits=tuple(
            (owner, bits) for block in contract.blocks
            for owner in block.activation_owners),
        fp16_weight_modules=(),
        fp16_activation_owners=(),
        propagation=_propagation(),
        propagation_mode="integer",
        hawq_range_momentum=0.9,
    )


def _initialization_rows():
    return (
        (("activation::encoder::input", "module_input"),
         torch.tensor([-1.0, 1.0])),
        (("activation::decoder::input", "module_input"),
         torch.tensor([-2.0, 2.0])),
    )


def test_model_qat_controller_owns_each_contract_activation_once():
    controller = ModelMethodQATController(
        ToyModel(), _contract(), _sites(), _config())

    owners = controller.activation_owner_manifest()

    assert owners == (
        ("activation::encoder::input", "module_input"),
        ("activation::decoder::input", "module_input"),
    )
    assert len(owners) == len(set(owners))


def test_model_qat_keeps_explicit_fp16_unit_out_of_quantizers():
    config = ModelMethodQATConfig(
        method="task_aware",
        weight_bits=(("encoder", 6),),
        activation_bits=((
            ("activation::encoder::input", "module_input"), 6),),
        fp16_weight_modules=("decoder",),
        fp16_activation_owners=(
            ("activation::decoder::input", "module_input"),),
        propagation=_propagation(),
        propagation_mode="integer",
        hawq_range_momentum=0.9,
    )
    model = ToyModel()
    controller = ModelMethodQATController(
        model, _contract(), _sites(), config)
    controller.initialize_activations((
        (("activation::encoder::input", "module_input"),
         torch.tensor([-1.0, 1.0])),
    ))

    controller.install()

    assert parametrize.is_parametrized(model.encoder, "weight")
    assert not parametrize.is_parametrized(model.decoder, "weight")
    assert controller.activation_owner_manifest() == (
        ("activation::encoder::input", "module_input"),
        ("activation::decoder::input", "module_input"),
    )
    assert len(controller.activation_quantizers()) == 1
    manifest = controller.manifest()
    assert manifest["fp16_weight_modules"] == ("decoder",)
    assert manifest["fp16_activation_owners"] == (
        ("activation::decoder::input", "module_input"),)
    controller.remove()


def test_model_qat_controller_preserves_fp32_master_and_materializes_hard_state():
    model = ToyModel()
    reference = model.encoder.weight.detach().clone()
    controller = ModelMethodQATController(
        model, _contract(), _sites(), _config())
    controller.initialize_activations(_initialization_rows())
    controller.install()

    model(torch.tensor([[[[0.75]]]]))
    canonical = controller.canonical_model_state_dict()
    hard = controller.hard_model_state_dict()

    assert torch.equal(canonical["encoder.weight"], reference)
    assert not any("parametrizations" in name for name in canonical)
    assert not any("parametrizations" in name for name in hard)
    assert not torch.equal(hard["encoder.weight"], reference)
    deployment = controller.hard_deployment_manifest()
    assert deployment["validated"] == 1
    assert deployment["weight_bits"] == _config().weight_bits
    assert deployment["activation_bits"] == _config().activation_bits
    assert len(deployment["activation_qparams"]) == 2
    assert deployment["protected_scale_roles_excluded"] == 1
    controller.remove()


def test_model_qat_controller_rejects_protected_or_incomplete_ownership():
    contract = _contract()
    incomplete = ModelMethodQATConfig(
        method="lsqplus",
        weight_bits=(("encoder", 4),),
        activation_bits=_config().activation_bits,
        fp16_weight_modules=(),
        fp16_activation_owners=(),
        propagation=_propagation(),
        propagation_mode="integer",
        hawq_range_momentum=0.9,
    )
    with pytest.raises(ValueError, match="weight assignment coverage"):
        ModelMethodQATController(
            ToyModel(), contract, _sites(), incomplete)

    protected = ModelMethodQATConfig(
        method="hawq",
        weight_bits=(
            ("encoder", 4), ("decoder", 4), ("propagation", 4)),
        activation_bits=_config().activation_bits,
        fp16_weight_modules=(),
        fp16_activation_owners=(),
        propagation=_propagation(),
        propagation_mode="integer",
        hawq_range_momentum=0.9,
    )
    with pytest.raises(ValueError, match="protected"):
        ModelMethodQATController(
            ToyModel(), contract, _sites(), protected)


def test_model_qat_controller_rejects_disguised_protected_signal_owner():
    contract = QuantizationModelContract(
        model_name="nlspn",
        blocks=(QuantizationBlock(
            "encoder", ("encoder",),
            (("signal::affinity", "module_input"),)),),
        prefix_groups=(("encoder",),),
        tail_groups=(("encoder",),),
        protected_roles=("affinity",),
        attention_edges=(),
        concat_edges=(),
        protected_modules=("propagation",),
        module_roles=(("propagation", "affinity"),),
    )
    plan = QDropTargetPlan(
        model="nlspn",
        blocks=("encoder",),
        activation_sites=(QDropActivationSite(
            "signal::affinity", "encoder", "module_input",
            "module_input", True, True),),
        excluded_sites=(),
    )
    config = ModelMethodQATConfig(
        method="lsqplus",
        weight_bits=(("encoder", 4),),
        activation_bits=((('signal::affinity', 'module_input'), 4),),
        fp16_weight_modules=(),
        fp16_activation_owners=(),
        propagation=_propagation(),
        propagation_mode="integer",
        hawq_range_momentum=0.9,
    )

    with pytest.raises(ValueError, match="protected"):
        ModelMethodQATController(ToyModel(), contract, plan, config)


def test_mixed_task_aware_uses_static_a4_a6_a8_ranges():
    config = ModelMethodQATConfig(
        method="mixed_task_aware",
        weight_bits=(("encoder", 4), ("decoder", 8)),
        activation_bits=(
            (("activation::encoder::input", "module_input"), 6),
            (("activation::decoder::input", "module_input"), 8),
        ),
        fp16_weight_modules=(),
        fp16_activation_owners=(),
        propagation=_propagation(),
        propagation_mode="integer",
        hawq_range_momentum=0.9,
    )
    controller = ModelMethodQATController(
        ToyModel(), _contract(), _sites(), config)

    controller.initialize_activations(_initialization_rows())

    assert all(not quantizer.running_range
               for quantizer in controller.activation_quantizers())


def test_model_qat_interfaces_are_exported_from_package():
    from spn_quant import qat

    assert qat.ModelHardDeploymentController is ModelHardDeploymentController
    assert qat.ModelMethodQATConfig is ModelMethodQATConfig
    assert qat.ModelMethodQATController is ModelMethodQATController


def test_generic_lsqplus_rejects_mixed_weight_activation_precision():
    with pytest.raises(ValueError, match="uniform W4A4 or W6A6"):
        ModelMethodQATConfig(
            method="lsqplus",
            weight_bits=(("encoder", 4), ("decoder", 4)),
            activation_bits=(
                (("activation::encoder::input", "module_input"), 6),
                (("activation::decoder::input", "module_input"), 6),
            ),
            fp16_weight_modules=(),
            fp16_activation_owners=(),
            propagation=_propagation(),
            propagation_mode="integer",
            hawq_range_momentum=0.9,
        )


def test_hard_controller_uses_materialized_weights_and_frozen_qparams():
    training_model = ToyModel()
    training = ModelMethodQATController(
        training_model, _contract(), _sites(), _config())
    training.initialize_activations(_initialization_rows())
    training.install()
    hard_state = training.hard_model_state_dict()
    weight_diagnostic_qparams = \
        training.deployment_weight_diagnostic_qparams()
    qparams = training.deployment_qparams()

    deployed_model = ToyModel()
    deployed_model.load_state_dict(hard_state, strict=True)
    deployed = ModelHardDeploymentController(
        deployed_model, _contract(), _sites(), _config(), qparams)
    deployed.configure_weight_code_statistics(weight_diagnostic_qparams)
    deployed.install()
    value = torch.tensor([[[[0.75]]]])

    with torch.no_grad():
        expected = training_model(value)
        actual = deployed_model(value)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert not parametrize.is_parametrized(deployed_model.encoder, "weight")
    assert not parametrize.is_parametrized(deployed_model.decoder, "weight")
    assert torch.equal(deployed_model.encoder.weight, hard_state["encoder.weight"])
    code_rows = deployed.statistics()
    activation_rows = tuple(
        row for row in code_rows if row["owner_kind"] == "activation")
    weight_rows = tuple(
        row for row in code_rows if row["owner_kind"] == "weight")
    assert tuple(row["owner"] for row in activation_rows) == \
        deployed.activation_owner_manifest()
    assert tuple(row["owner"] for row in weight_rows) == \
        tuple(name for name, bits in deployed.config.weight_bits)
    assert all(row["calls"] == 1 and row["numel"] > 0 for row in code_rows)
    assert all(0 <= row["zero_code_count"] <= row["numel"]
               for row in code_rows)
    assert all(0 <= row["saturation_count"] <= row["numel"]
               for row in code_rows)
    assert all(0.0 <= row["zero_code_rate"] <= 1.0 for row in code_rows)
    assert all(0.0 <= row["saturation_rate"] <= 1.0 for row in code_rows)
    deployed.remove()
    training.remove()


def test_hard_controller_rejects_modified_activation_grid_contract():
    training = ModelMethodQATController(
        ToyModel(), _contract(), _sites(), _config())
    training.initialize_activations(_initialization_rows())
    training.install()
    qparams = deepcopy(training.deployment_qparams())
    qparams["activation"][0]["qmin"] = -7

    with pytest.raises(ValueError, match="activation qparam grid"):
        ModelHardDeploymentController(
            ToyModel(), _contract(), _sites(), _config(), qparams)

    training.remove()


def test_generic_method_state_serializes_exact_propagation_qparams():
    propagation = PropagationQuantController()
    propagation.observe()
    propagation.observe_signal("state", torch.tensor([2.5]))
    propagation.freeze()
    propagation.configure(_propagation())
    controller = ModelMethodQATController(
        ToyModel(),
        _contract(),
        _sites(),
        _config(),
        propagation_adapter=SimpleNamespace(controller=propagation),
    )
    controller.initialize_activations(_initialization_rows())
    controller.install()

    state = controller.method_state_dict()
    propagation.maximum["state"] = 9.0
    controller.load_method_state_dict(state)

    assert "propagation.maximum.state" in state
    assert propagation.maximum == {"state": 2.5}
    assert controller.deployment_qparams()["propagation"] == {
        "maximum": (("state", 2.5),),
        "config": {
            "affinity_bits": 8,
            "confidence_bits": 8,
            "offset_bits": 8,
            "state_bits": 8,
            "coefficient_fraction_bits": 13,
        },
        "frozen": True,
    }
    controller.remove()
