import pytest
import numpy as np
import torch
import torch.nn as nn

from scripts.run_nyu_cspn_rotation import (
    BOUNDARY_FIELDS,
    END_TO_END_FIELDS,
    PROPAGATION_A8_Q13,
    build_configurations,
    depth_sample_metrics,
    cspn_quant_group,
    rotation_owned_inputs,
    rotation_owned_outputs,
    select_group_size,
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


def test_fp_equivalence_accepts_low_normalized_rotation_roundoff():
    reference = torch.tensor([0.0, 1.0, 2.0, 4.0])
    candidate = reference + torch.tensor([5e-4, 0.0, -2e-4, 3e-4])

    validate_fp_equivalence(
        reference, candidate, site="decoder_entry")


def test_metric_schema_contains_depth_and_rotation_metrics():
    assert END_TO_END_FIELDS == (
        "model", "config", "sample_index", "RMSE", "MAE", "ABS_REL",
        "IRMSE", "flat_RMSE", "boundary_RMSE", "nonfinite_ratio",
    )
    assert "channel_imbalance" in BOUNDARY_FIELDS
    assert "block_output_sqnr" in BOUNDARY_FIELDS


def test_depth_sample_metrics_reports_inverse_and_regions():
    gt = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    pred = torch.tensor([[1.0, 2.5], [2.5, 4.0]]).numpy()
    sparse = torch.zeros(2, 2).numpy()

    row, regions = depth_sample_metrics(gt, pred, sparse)

    assert row["RMSE"] == pytest.approx((0.5 / 4.0) ** 0.5)
    expected_inverse_mse = (
        (1.0 / 2.5 - 1.0 / 2.0) ** 2
        + (1.0 / 2.5 - 1.0 / 3.0) ** 2
    ) / 4.0
    assert row["IRMSE"] == pytest.approx(expected_inverse_mse ** 0.5)
    by_region = dict((item["region"], item) for item in regions)
    assert np.isnan(row["flat_RMSE"])
    assert row["boundary_RMSE"] == by_region["boundary"]["RMSE"]
    assert row["nonfinite_ratio"] == 0.0


def test_select_group_size_uses_block_error_then_sqnr_then_group_count():
    rows = [
        {"group_size": 16, "block_output_mse": 0.4,
         "block_output_sqnr": 20.0},
        {"group_size": 32, "block_output_mse": 0.2,
         "block_output_sqnr": 18.0},
        {"group_size": 64, "block_output_mse": 0.2,
         "block_output_sqnr": 21.0},
    ]

    assert select_group_size(rows) == 64
