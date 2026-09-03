import torch
import torch.nn as nn

from spn_quant.qat.model_methods import ModelMethodQATConfig
from spn_quant.qat.lsqplus import (
    LSQPlusActivationQuantizer,
    LSQPlusWeightParametrization,
)
from spn_quant.propagation.controller import PropagationQuantConfig


def test_task_aware_config_accepts_independent_w4_w6_w8_assignments():
    config = ModelMethodQATConfig(
        method="task_aware",
        weight_bits=(("conv4", 4), ("conv6", 6), ("conv8", 8)),
        activation_bits=(
            (("activation::conv4::input", "module_input"), 4),
            (("activation::conv6::input", "module_input"), 6),
            (("activation::conv8::input", "module_input"), 8),
        ),
        fp16_weight_modules=(),
        fp16_activation_owners=(),
        propagation=PropagationQuantConfig(),
        propagation_mode="integer",
        hawq_range_momentum=0.9,
    )

    assert config.method == "task_aware"
    assert tuple(bits for name, bits in config.weight_bits) == (4, 6, 8)


def test_task_aware_uses_learnable_lsqplus_quantizers_at_eight_bits():
    activation = LSQPlusActivationQuantizer(8, unsigned=False)
    weight = LSQPlusWeightParametrization(
        8, 0, torch.randn(3, 2, 1, 1))

    assert isinstance(activation.step, nn.Parameter)
    assert isinstance(weight.step, nn.Parameter)
