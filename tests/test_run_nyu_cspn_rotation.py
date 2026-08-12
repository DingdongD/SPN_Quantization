import pytest
import torch
import torch.nn as nn

from scripts.run_nyu_cspn_rotation import (
    BOUNDARY_FIELDS,
    END_TO_END_FIELDS,
    PROPAGATION_A8_Q13,
    build_configurations,
    cspn_quant_group,
    rotation_owned_inputs,
    rotation_owned_outputs,
    validate_fp_equivalence,
)


def test_build_configurations_covers_identity_single_and_joint_rotation():
    names = [row["name"] for row in build_configurations(group_size=32)]

    assert names == [
        "FP32",
        "RTN_W4A4",
        "GROUP_W4A4",
        "RANDOM_decoder_entry",
        "RANDOM_layer4_signed_skip",
        "RANDOM_both",
        "HADAMARD_decoder_entry",
        "HADAMARD_layer4_signed_skip",
        "HADAMARD_both",
        "HADAMARD_GROUP_both",
    ]
    for config in build_configurations(group_size=32)[1:]:
        assert set(config["rotation_methods"]) == {
            "decoder_entry", "layer4_signed_skip"}
        assert config["propagation"] == PROPAGATION_A8_Q13


def test_cspn_group_function_excludes_complete_guidance_head():
    assert cspn_quant_group(
        "gud_up_proj_layer6.conv1", nn.Conv2d(4, 8, 1)) is None
    assert cspn_quant_group(
        "gud_up_proj_layer5.conv1", nn.Conv2d(4, 1, 1)) == "depth_head"


def test_rotation_ownership_covers_raw_signed_boundaries_and_consumers():
    assert rotation_owned_outputs() == {
        "conv1_1", "conv2", "gud_up_proj_layer5.conv1"}
    assert rotation_owned_inputs() == {
        "gud_up_proj_layer1.conv1",
        "gud_up_proj_layer1.sc_conv1",
        "gud_up_proj_layer4.conv1_1",
    }


def test_fp_equivalence_rejects_wrong_rotation():
    with pytest.raises(RuntimeError, match="FP equivalence"):
        validate_fp_equivalence(
            torch.ones(1), torch.zeros(1), site="decoder_entry")


def test_metric_schema_contains_depth_and_rotation_metrics():
    assert END_TO_END_FIELDS == (
        "model", "config", "sample_index", "RMSE", "MAE", "ABS_REL",
        "IRMSE", "flat_RMSE", "boundary_RMSE", "nonfinite_ratio",
    )
    assert "channel_imbalance" in BOUNDARY_FIELDS
    assert "block_output_sqnr" in BOUNDARY_FIELDS
