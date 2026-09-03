import pytest

from spn_quant.fp_mixed_precision import (
    GROUP_NAMES,
    build_format_assignment,
    build_grouped_three_level_format_assignment,
    build_grouped_format_assignment,
    classify_activation_owner,
    classify_weight_module,
    weighted_format_fractions,
)
from spn_quant.model_contracts import QuantizationBlock, QuantizationModelContract


def _contract():
    return QuantizationModelContract(
        model_name="test",
        blocks=(
            QuantizationBlock(
                "first", ("conv1", "conv2"),
                (("activation::conv1::input", "module_input"),
                 ("activation::conv2::input", "module_input"))),
        ),
        prefix_groups=(("first",),),
        tail_groups=(("first",),),
        protected_roles=("propagation_state",),
        attention_edges=(),
        concat_edges=(),
        protected_modules=("propagation",),
        module_roles=(("propagation", "propagation_state"),),
    )


def test_uniform_format_assignment_covers_only_contract_owners():
    assignment = build_format_assignment(
        _contract(), "fp4_e2m1", "fp8_e4m3fn", {}, 0.5)
    assert set(assignment.weight_formats) == {"conv1", "conv2"}
    assert set(assignment.activation_formats) == {
        ("activation::conv1::input", "module_input"),
        ("activation::conv2::input", "module_input"),
    }
    assert set(assignment.activation_formats.values()) == {"fp8_e4m3fn"}


def test_uniform_format_assignment_accepts_fp6():
    assignment = build_format_assignment(
        _contract(), "fp6_e3m2", "fp6_e3m2", {}, 0.0)
    assert set(assignment.weight_formats.values()) == {"fp6_e3m2"}
    assert set(assignment.activation_formats.values()) == {"fp6_e3m2"}


def test_three_level_assignment_uses_fp6_as_an_intermediate_promotion():
    assignment, audit = build_grouped_three_level_format_assignment(
        _contract(),
        {
            "conv1": {4: 1.0, 6: 1.0, 8: 1.0},
            "conv2": {4: 1.0, 6: 1.0, 8: 1.0},
        },
        {
            ("activation::conv1::input", "module_input"):
            {4: 10.0, 6: 9.0, 8: 0.0},
            ("activation::conv2::input", "module_input"):
            {4: 10.0, 6: 1.0, 8: 0.0},
        },
        {"conv1": 1, "conv2": 1},
        {
            ("activation::conv1::input", "module_input"): 1,
            ("activation::conv2::input", "module_input"): 1,
        },
        dict(
            (name, {"weight_average_bits": 4.0,
                    "activation_average_bits": 6.0,
                    "activation_minimum_fp8_fraction": 0.0})
            for name in GROUP_NAMES),
        {"weight_average_bits": 4.0, "activation_average_bits": 6.0},
    )
    assert set(assignment.activation_formats.values()) == {"fp6_e3m2"}
    assert audit["encoder"]["activation"]["format_unit_counts"] == {
        "fp6_e3m2": 2,
    }


def test_mixed_activation_promotion_uses_weighted_sensitivity_order():
    assignment = build_format_assignment(
        _contract(), "fp4_e2m1", "fp4_e2m1",
        {
            ("activation::conv1::input", "module_input"): 3.0,
            ("activation::conv2::input", "module_input"): 1.0,
        },
        0.5,
        {(
            "activation::conv1::input", "module_input"): 1.0,
         ("activation::conv2::input", "module_input"): 1.0},
    )
    assert assignment.activation_formats[
        ("activation::conv1::input", "module_input")] == "fp8_e4m3fn"
    assert assignment.activation_formats[
        ("activation::conv2::input", "module_input")] == "fp4_e2m1"


def test_format_assignment_rejects_missing_sensitivity_and_invalid_fraction():
    owner = ("activation::conv1::input", "module_input")
    with pytest.raises(ValueError, match="sensitivity"):
        build_format_assignment(_contract(), "fp4_e2m1", "fp4_e2m1",
                                {owner: 1.0}, 0.5,
                                {owner: 1.0,
                                 ("activation::conv2::input", "module_input"): 1.0})
    with pytest.raises(ValueError, match="fraction"):
        build_format_assignment(_contract(), "fp4_e2m1", "fp4_e2m1",
                                {owner: 1.0, ("activation::conv2::input", "module_input"): 1.0}, 1.5)


def test_weighted_format_fractions_use_explicit_costs():
    formats = {"a": "fp4_e2m1", "b": "fp8_e4m3fn"}
    fractions = weighted_format_fractions(
        formats, {"a": 1.0, "b": 3.0})
    assert fractions == {"fp4_e2m1": 0.25, "fp8_e4m3fn": 0.75}


def test_semantic_group_classification_is_explicit():
    assert classify_weight_module("layer3.0.conv1") == "encoder"
    assert classify_weight_module("backbone.dec4.1.conv2") == "decoder"
    assert classify_weight_module("gud_up_proj_layer2.conv1") == "fusion"
    assert classify_weight_module(
        "backbone.former.block2.0.attn.q") == "attention"
    assert classify_weight_module(
        "backbone.former.block2.0.concat_conv") == "concat"
    assert classify_activation_owner(
        ("attention::block::q", "attention_q")) == "attention"
    assert classify_activation_owner(
        ("concat::block::cnn_input", "concat_cnn_input")) == "concat"
    with pytest.raises(ValueError, match="semantic group"):
        classify_weight_module("unclassified_module")


def test_grouped_assignment_obeys_weight_and_activation_budgets_separately():
    contract = QuantizationModelContract(
        model_name="test",
        blocks=(
            QuantizationBlock(
                "first", ("layer1.0.conv1", "backbone.dec2.0.0"),
                (("activation::layer1.0.conv1::output", "module_output"),
                 ("activation::backbone.dec2.0.0::input", "module_input"))),
        ),
        prefix_groups=(("first",),),
        tail_groups=(("first",),),
        protected_roles=("propagation_state",),
        attention_edges=(),
        concat_edges=(),
        protected_modules=("propagation",),
        module_roles=(("propagation", "propagation_state"),),
    )
    assignment, audit = build_grouped_format_assignment(
        contract,
        weight_scores={"layer1.0.conv1": 1.0, "backbone.dec2.0.0": 8.0},
        activation_scores={
            ("activation::layer1.0.conv1::output", "module_output"): 1.0,
            ("activation::backbone.dec2.0.0::input", "module_input"): 8.0,
        },
        weight_costs={"layer1.0.conv1": 100, "backbone.dec2.0.0": 1},
        activation_costs={
            ("activation::layer1.0.conv1::output", "module_output"): 100,
            ("activation::backbone.dec2.0.0::input", "module_input"): 1,
        },
        group_budgets=dict(
            (name, {"weight_average_bits": 4.0,
                    "activation_average_bits": 4.0,
                    "activation_minimum_fp8_fraction": 0.0})
            for name in GROUP_NAMES),
        global_budgets={"weight_average_bits": 4.0,
                        "activation_average_bits": 4.0},
    )
    assert assignment.weight_formats["layer1.0.conv1"] == "fp4_e2m1"
    assert assignment.weight_formats["backbone.dec2.0.0"] == "fp4_e2m1"
    assert assignment.activation_formats[
        ("activation::layer1.0.conv1::output", "module_output")] == "fp4_e2m1"
    assert assignment.activation_formats[
        ("activation::backbone.dec2.0.0::input", "module_input")] == "fp4_e2m1"
    assert audit["encoder"]["weight_average_bits"] == 4.0
    assert audit["decoder"]["activation_average_bits"] == 4.0


def test_grouped_assignment_requires_budget_for_every_group():
    with pytest.raises(KeyError, match="semantic group budget"):
        build_grouped_format_assignment(
            _contract(), {"conv1": 1.0, "conv2": 1.0},
            {
                ("activation::conv1::input", "module_input"): 1.0,
                ("activation::conv2::input", "module_input"): 1.0,
            },
            {"conv1": 1, "conv2": 1},
            {
                ("activation::conv1::input", "module_input"): 1,
                ("activation::conv2::input", "module_input"): 1,
            },
            {"encoder": {"weight_average_bits": 4.0,
                          "activation_average_bits": 4.0,
                          "activation_minimum_fp8_fraction": 0.0}},
            {"weight_average_bits": 4.0, "activation_average_bits": 4.0},
        )


def test_grouped_assignment_allows_a_group_without_activation_owner():
    contract = QuantizationModelContract(
        model_name="test",
        blocks=(QuantizationBlock("first", ("backbone.conv1",), ()),),
        prefix_groups=(("first",),),
        tail_groups=(("first",),),
        protected_roles=("propagation_state",),
        attention_edges=(),
        concat_edges=(),
        protected_modules=("propagation",),
        module_roles=(("propagation", "propagation_state"),),
    )
    assignment, audit = build_grouped_format_assignment(
        contract,
        {"backbone.conv1": 1.0}, {}, {"backbone.conv1": 1}, {},
        dict((name, {"weight_average_bits": 4.0,
                     "activation_average_bits": 4.0,
                     "activation_minimum_fp8_fraction": 0.0})
             for name in GROUP_NAMES),
        {"weight_average_bits": 4.0, "activation_average_bits": 4.0},
    )
    assert assignment.weight_formats["backbone.conv1"] == "fp4_e2m1"
    assert assignment.activation_formats == {}
    assert audit["encoder"]["present"] is True
    assert audit["encoder"]["activation_average_bits"] is None


def test_grouped_assignment_protects_decoder_and_fusion_before_global_promotions():
    contract = QuantizationModelContract(
        model_name="test",
        blocks=(
            QuantizationBlock(
                "encoder", ("layer1.0.conv1", "layer1.0.conv2"),
                (("activation::layer1.0.conv1::input", "module_input"),
                 ("activation::layer1.0.conv2::input", "module_input"))),
            QuantizationBlock(
                "decoder", ("backbone.dec2.0.0",),
                (("activation::backbone.dec2.0.0::input", "module_input"),)),
            QuantizationBlock(
                "fusion", ("gud_up_proj_layer1.conv1",),
                (("activation::gud_up_proj_layer1.conv1::input", "module_input"),)),
        ),
        prefix_groups=(("encoder",), ("decoder",), ("fusion",)),
        tail_groups=(("encoder",), ("decoder",), ("fusion",)),
        protected_roles=("propagation_state",),
        attention_edges=(),
        concat_edges=(),
        protected_modules=("propagation",),
        module_roles=(("propagation", "propagation_state"),),
    )
    budgets = dict(
        (name, {"weight_average_bits": 8.0,
                "activation_average_bits": 8.0,
                "activation_minimum_fp8_fraction":
                1.0 if name == "fusion" else
                0.5 if name == "decoder" else 0.0})
        for name in GROUP_NAMES)
    assignment, audit = build_grouped_format_assignment(
        contract,
        {"layer1.0.conv1": 1.0, "layer1.0.conv2": 1.0,
         "backbone.dec2.0.0": 1.0, "gud_up_proj_layer1.conv1": 1.0},
        {
            ("activation::layer1.0.conv1::input", "module_input"): 5.0,
            ("activation::layer1.0.conv2::input", "module_input"): 5.0,
            ("activation::backbone.dec2.0.0::input", "module_input"): 1.0,
            ("activation::gud_up_proj_layer1.conv1::input", "module_input"): 1.0,
        },
        {"layer1.0.conv1": 1, "layer1.0.conv2": 1,
         "backbone.dec2.0.0": 1, "gud_up_proj_layer1.conv1": 1},
        {
            ("activation::layer1.0.conv1::input", "module_input"): 1,
            ("activation::layer1.0.conv2::input", "module_input"): 1,
            ("activation::backbone.dec2.0.0::input", "module_input"): 1,
            ("activation::gud_up_proj_layer1.conv1::input", "module_input"): 1,
        },
        budgets,
        {"weight_average_bits": 8.0, "activation_average_bits": 7.0},
    )
    assert assignment.activation_formats[
        ("activation::gud_up_proj_layer1.conv1::input", "module_input")] == "fp8_e4m3fn"
    assert assignment.activation_formats[
        ("activation::backbone.dec2.0.0::input", "module_input")] == "fp8_e4m3fn"
    assert audit["fusion"]["activation_floor_satisfied"] is True
    assert audit["decoder"]["activation_floor_satisfied"] is True
    assert audit["encoder"]["activation"]["fp8_unit_count"] == 1
