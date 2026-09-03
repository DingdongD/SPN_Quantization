import pytest
import torch
import torch.nn as nn

from spn_quant.model_contracts import (
    QuantizationBlock,
    QuantizationModelContract,
    propagation_owned_modules,
    validate_propagation_ownership,
)
from spn_quant.propagation.controller import PropagationQuantController


def _contract(protected_modules=("propagation",)):
    return QuantizationModelContract(
        model_name="nlspn",
        blocks=(
            QuantizationBlock(
                "encoder", ("encoder",),
                (("activation::encoder::input", "module_input"),)),
        ),
        prefix_groups=(("encoder",),),
        tail_groups=(("encoder",),),
        protected_roles=("propagation_state",),
        attention_edges=(),
        concat_edges=(),
        protected_modules=protected_modules,
        module_roles=(("propagation", "propagation_state"),),
    )


def test_controller_has_explicit_fp16_mode():
    controller = PropagationQuantController()
    controller.observe()
    controller.observe_signal("state", torch.ones(1))
    controller.freeze()

    controller.configure_fp16()

    assert controller.mode == "float"
    assert controller.float_state_dtype is torch.float16


def test_propagation_ownership_is_disjoint_from_ordinary_modules():
    model = nn.Module()
    model.add_module("encoder", nn.Conv2d(1, 1, 1))
    model.add_module("propagation", nn.Conv2d(1, 1, 1))
    contract = _contract()

    validate_propagation_ownership(contract, model)

    assert propagation_owned_modules(contract) == ("propagation",)
    assert set(contract.weight_modules).isdisjoint(
        propagation_owned_modules(contract))


def test_propagation_ownership_rejects_missing_module():
    model = nn.Module()
    model.add_module("encoder", nn.Conv2d(1, 1, 1))

    with pytest.raises(KeyError, match="missing propagation modules"):
        validate_propagation_ownership(_contract(), model)
