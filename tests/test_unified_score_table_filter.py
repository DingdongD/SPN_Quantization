import pytest

from scripts.run_nyu_unified_fp16_task_aware_allocation import (
    ordinary_score_tables,
    sensitivity_module_from_owner,
)
from spn_quant import mixed_precision
from spn_quant.model_contracts import (
    QuantizationBlock,
    QuantizationModelContract,
)


def _contract():
    return QuantizationModelContract(
        model_name="nlspn",
        blocks=(QuantizationBlock(
            "encoder", ("encoder",),
            (("activation::encoder::input", "module_input"),)),),
        prefix_groups=(("encoder",),),
        tail_groups=(("encoder",),),
        protected_roles=("propagation_state",),
        attention_edges=(),
        concat_edges=(),
        protected_modules=("propagation",),
        module_roles=(("propagation", "propagation_state"),),
    )


def _scores():
    return {4: 4.0, 6: 2.0, 8: 0.0}


def test_ordinary_score_tables_exclude_declared_propagation_rows():
    costs = mixed_precision.CostBasis(
        weight_macs=(("encoder", 2),),
        activation_elements=((("activation::encoder::input",
                               "module_input"), 3),))
    weights, activations = ordinary_score_tables(
        _contract(), costs,
        {"encoder": _scores(), "propagation": _scores()},
        {"encoder": _scores(), "propagation": _scores()})

    assert tuple(weights) == ("encoder",)
    assert tuple(activations) == (
        ("activation::encoder::input", "module_input"),)


def test_ordinary_score_tables_reject_unknown_extra_rows():
    costs = mixed_precision.CostBasis(
        weight_macs=(("encoder", 2),),
        activation_elements=((("activation::encoder::input",
                               "module_input"), 3),))

    with pytest.raises(ValueError, match="unknown weight sensitivity"):
        ordinary_score_tables(
            _contract(), costs,
            {"encoder": _scores(), "unrelated": _scores()},
            {"encoder": _scores()})


def test_sensitivity_module_from_completionformer_joint_owner():
    assert sensitivity_module_from_owner(
        ("attention::backbone.former.block1.0.attn::q", "attention_q")) == \
        "backbone.former.block1.0.attn.q"
    assert sensitivity_module_from_owner(
        ("attention::backbone.former.block1.0.attn::k", "attention_k")) == \
        "backbone.former.block1.0.attn.kv"
    assert sensitivity_module_from_owner(
        ("attention::backbone.former.block1.0.attn::v", "attention_v")) == \
        "backbone.former.block1.0.attn.kv"
    assert sensitivity_module_from_owner(
        ("concat::backbone.former.block1.0.concat_conv::cnn_input",
         "concat_cnn_input")) == "backbone.former.block1.0.concat_conv"
