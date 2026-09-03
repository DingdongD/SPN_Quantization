import pytest

from spn_quant.mixed_precision import (
    BitAssignment,
    validate_assignment_ownership,
)
from spn_quant.model_contracts import (
    QuantizationBlock,
    QuantizationModelContract,
)


def _contract():
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
        protected_modules=("propagation",),
        module_roles=(("propagation", "propagation_state"),),
    )


def test_assignment_rejects_propagation_weight_module():
    assignment = BitAssignment(
        weight_bits=(("encoder", 4), ("propagation", 8)),
        activation_bits=((("activation::encoder::input", "module_input"), 4),),
    )

    with pytest.raises(ValueError, match="propagation"):
        validate_assignment_ownership(_contract(), assignment)


def test_assignment_rejects_propagation_activation_role():
    assignment = BitAssignment(
        weight_bits=(("encoder", 4),),
        activation_bits=(
            (("activation::encoder::input", "module_input"), 4),
            (("signal::state", "propagation_state"), 4),
        ),
    )

    with pytest.raises(ValueError, match="propagation"):
        validate_assignment_ownership(_contract(), assignment)


def test_assignment_accepts_only_ordinary_sites():
    assignment = BitAssignment(
        weight_bits=(("encoder", 4),),
        activation_bits=((("activation::encoder::input", "module_input"), 4),),
    )

    validate_assignment_ownership(_contract(), assignment)
