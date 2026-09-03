import pytest
import torch

from scripts.run_nyu_nlspn_selective_channel_smoothing_fp6 import (
    _attribution_rows,
    _boundary_candidates,
    _boundary_module_bits,
    _initial_depth_branch_channels,
    _pareto_rows,
)
from spn_quant.fp_formats import make_quantizer
from spn_quant.fp_mixed_precision import FPFormatAssignment
from spn_quant.selective_channel_smoothing import (
    BranchIndependentFPQuantizer,
)


ID_DEC0_OWNER = ("activation::id_dec0.0::input", "module_input")
ID_DEC1_OWNER = ("activation::id_dec1.0::input", "module_input")


def _base_assignment():
    return FPFormatAssignment(
        {
            "conv2.0.conv1": "fp6_e3m2",
            "id_dec0.0": "fp6_e3m2",
            "id_dec1.0": "fp6_e3m2",
        },
        {
            ("activation::conv2.0.conv1::input", "module_input"):
            "fp6_e3m2",
            ID_DEC0_OWNER: "fp6_e3m2",
            ID_DEC1_OWNER: "fp6_e3m2",
        },
    )


def test_boundary_candidates_form_individual_and_joint_factorial_grid():
    candidates = _boundary_candidates(_base_assignment())
    assert tuple(candidate.name for candidate in candidates) == (
        "BOTH_W8A8",
        "ID_DEC0_W6A6",
        "ID_DEC0_W6A8",
        "ID_DEC0_W8A6",
        "ID_DEC0_W8A6_BRANCH",
        "ID_DEC1_W6A6",
        "ID_DEC1_W6A8",
        "ID_DEC1_W8A6",
        "ID_DEC1_W8A6_BRANCH",
        "JOINT_W6A6",
        "JOINT_W6A8",
        "JOINT_W8A6",
        "JOINT_W8A6_BRANCH",
    )
    selected = dict((candidate.name, candidate) for candidate in candidates)
    candidate = selected["ID_DEC0_W6A8"]
    assert candidate.assignment.weight_formats["id_dec0.0"] == "fp6_e3m2"
    assert candidate.assignment.activation_formats[ID_DEC0_OWNER] == \
        "fp8_e4m3fn"
    assert candidate.assignment.weight_formats["id_dec1.0"] == \
        "fp8_e4m3fn"
    assert candidate.assignment.activation_formats[ID_DEC1_OWNER] == \
        "fp8_e4m3fn"
    assert candidate.assignment.weight_formats["conv2.0.conv1"] == \
        "fp6_e3m2"
    assert selected["JOINT_W8A6_BRANCH"].branch_independent_modules == (
        "id_dec0.0", "id_dec1.0")
    for candidate in candidates:
        assert candidate.assignment.weight_formats["conv2.0.conv1"] == \
            "fp6_e3m2"
        assert candidate.assignment.activation_formats[
            ("activation::conv2.0.conv1::input", "module_input")] == \
            "fp6_e3m2"
    assert selected["ID_DEC0_W8A6"].assignment == \
        selected["ID_DEC0_W8A6_BRANCH"].assignment
    assert selected["ID_DEC1_W8A6"].assignment == \
        selected["ID_DEC1_W8A6_BRANCH"].assignment
    assert selected["JOINT_W8A6"].assignment == \
        selected["JOINT_W8A6_BRANCH"].assignment


def test_boundary_module_bits_report_actual_fixed_module_precision():
    selected = dict(
        (candidate.name, candidate)
        for candidate in _boundary_candidates(_base_assignment()))
    candidate = selected["ID_DEC0_W6A6"]
    assert _boundary_module_bits(candidate, "id_dec0.0") == (6, 6)
    assert _boundary_module_bits(candidate, "id_dec1.0") == (8, 8)


def test_initial_depth_boundary_requires_official_64_plus_64_channels():
    assert _initial_depth_branch_channels(128) == (64, 64)
    with pytest.raises(ValueError, match="128 channels"):
        _initial_depth_branch_channels(64)


def test_branch_independent_fp_quantizer_prevents_large_branch_from_zeroing_small_branch():
    tensor = torch.tensor([[[[0.05, 0.04]], [[100.0, 80.0]]]])
    quantizer = BranchIndependentFPQuantizer(
        "fp6_e3m2", torch.tensor([0.05, 100.0]),
        branch_channels=(1, 1), channel_dim=1)
    quantized, codes = quantizer.quantize_with_codes(tensor)
    assert bool((codes[:, 0] != 0).all().item())
    assert bool((quantized[:, 0] != 0).all().item())
    assert quantizer.diagnostics()["branch_0_zero_code_ratio"] == 0.0


def test_branch_histogram_percentile_uses_histogram_population():
    tensor = torch.tensor([[[[0.05, 0.04]], [[100.0, 80.0]]]])
    quantizer = BranchIndependentFPQuantizer(
        "fp6_e3m2", torch.tensor([100.0, 100.0]),
        branch_channels=(1, 1), channel_dim=1)
    quantizer.quantize_with_codes(tensor)
    diagnostics = quantizer.diagnostics()
    assert diagnostics["branch_0_histogram_count"] == 2
    assert diagnostics["branch_0_reference_p99"] < 1.0


def test_common_branch_scale_is_bit_exact_with_scalar_fp_quantization():
    tensor = torch.tensor([[[[0.05, 0.04]], [[100.0, 80.0]]]])
    branch = BranchIndependentFPQuantizer(
        "fp6_e3m2", torch.tensor([100.0, 100.0]),
        branch_channels=(1, 1), channel_dim=1)
    scalar = make_quantizer("fp6_e3m2", torch.tensor(100.0))
    branch_values, branch_codes = branch.quantize_with_codes(tensor)
    scalar_values, scalar_codes = scalar.quantize_with_codes(tensor)
    torch.testing.assert_close(branch_values, scalar_values, rtol=0.0, atol=0.0)
    torch.testing.assert_close(branch_codes, scalar_codes, rtol=0.0, atol=0.0)


def test_reference_p99_is_invariant_to_quantization_calibration_scale():
    tensor = torch.tensor([[[[0.05, 0.04]], [[100.0, 80.0]]]])
    common = BranchIndependentFPQuantizer(
        "fp6_e3m2", torch.tensor([100.0, 100.0]),
        branch_channels=(1, 1), channel_dim=1)
    independent = BranchIndependentFPQuantizer(
        "fp6_e3m2", torch.tensor([0.05, 100.0]),
        branch_channels=(1, 1), channel_dim=1)
    common.quantize_with_codes(tensor)
    independent.quantize_with_codes(tensor)
    common_p99 = common.diagnostics()["branch_0_reference_p99"]
    independent_p99 = independent.diagnostics()["branch_0_reference_p99"]
    assert common_p99 == pytest.approx(independent_p99)


def test_branch_diagnostics_accept_repeated_identical_evaluation_batch():
    tensor = torch.tensor([[[[0.05, 0.04]], [[100.0, 80.0]]]])
    quantizer = BranchIndependentFPQuantizer(
        "fp6_e3m2", torch.tensor([0.05, 100.0]),
        branch_channels=(1, 1), channel_dim=1)
    quantizer.quantize_with_codes(tensor)
    quantizer.quantize_with_codes(tensor)
    diagnostics = quantizer.diagnostics()
    assert diagnostics["branch_0_histogram_count"] == 4
    assert diagnostics["branch_1_histogram_count"] == 4


def test_attribution_rows_separate_weight_activation_interaction_and_branch_scale():
    rows = [
        {"configuration": "BOTH_W8A8", "pooled_rmse": 0.10},
        {"configuration": "ID_DEC0_W6A6", "pooled_rmse": 0.17},
        {"configuration": "ID_DEC0_W6A8", "pooled_rmse": 0.12},
        {"configuration": "ID_DEC0_W8A6", "pooled_rmse": 0.13},
        {"configuration": "ID_DEC0_W8A6_BRANCH", "pooled_rmse": 0.11},
        {"configuration": "ID_DEC1_W6A6", "pooled_rmse": 0.18},
        {"configuration": "ID_DEC1_W6A8", "pooled_rmse": 0.14},
        {"configuration": "ID_DEC1_W8A6", "pooled_rmse": 0.15},
        {"configuration": "ID_DEC1_W8A6_BRANCH", "pooled_rmse": 0.12},
        {"configuration": "JOINT_W6A6", "pooled_rmse": 0.20},
        {"configuration": "JOINT_W6A8", "pooled_rmse": 0.15},
        {"configuration": "JOINT_W8A6", "pooled_rmse": 0.16},
        {"configuration": "JOINT_W8A6_BRANCH", "pooled_rmse": 0.13},
    ]
    attribution = dict((row["scope"], row)
                       for row in _attribution_rows(rows))
    assert attribution["id_dec0"]["weight_penalty"] == pytest.approx(0.02)
    assert attribution["id_dec0"]["activation_penalty"] == pytest.approx(0.03)
    assert attribution["id_dec0"]["interaction_penalty"] == pytest.approx(0.02)
    assert attribution["id_dec0"]["branch_scale_recovery"] == \
        pytest.approx(0.02)
    assert attribution["joint"]["branch_scale_recovery"] == \
        pytest.approx(0.03)


def test_pareto_rows_remove_configs_worse_at_same_or_higher_bit_cost():
    rows = [
        {"configuration": "A", "pooled_rmse": 0.10,
         "average_weight_bits": 7.0, "average_activation_bits": 7.0},
        {"configuration": "B", "pooled_rmse": 0.11,
         "average_weight_bits": 6.0, "average_activation_bits": 6.0},
        {"configuration": "C", "pooled_rmse": 0.12,
         "average_weight_bits": 7.0, "average_activation_bits": 7.0},
        {"configuration": "D", "pooled_rmse": 0.09,
         "average_weight_bits": 8.0, "average_activation_bits": 8.0},
    ]
    assert tuple(row["configuration"] for row in _pareto_rows(rows)) == (
        "B", "A", "D")
