from types import SimpleNamespace
import math

import pytest
import torch
import torch.nn as nn

from scripts.hardware_aligned_quantization import HardwareAlignedInstrumentor
from scripts.run_nyu_nlspn_selective_channel_smoothing_fp6 import (
    BRANCH_LAYOUTS,
    DECODER_ATTRIBUTION_ARTIFACTS,
    DECODER_ATTRIBUTION_GROUPS,
    EARLY_MODULES,
    INITIAL_DEPTH_MODULES,
    _activation_protection_candidates,
    _activation_protection_base_assignment,
    _activation_protection_budget,
    _aggregate_depth_rows,
    _branch_maxima,
    _capture_nlspn_output,
    _configure_corrected_fp_candidate,
    _depth_error_row,
    _decoder_attribution_candidates,
    _decoder_cumulative_attribution_rows,
    _decoder_isolated_attribution_rows,
    _early_fp16_candidates,
    _early_fp16_attribution_rows,
    _experiment_runners,
    _generic_fp_activation_configuration,
    _signal_error_row,
    _require_equal_capture,
    _select_activation_protection_result,
    _validate_branch_layouts,
)
from spn_quant.fp_formats import make_quantizer
from spn_quant.fp_mixed_precision import FPFormatAssignment
from spn_quant.selective_channel_smoothing import (
    BranchIndependentFPQuantizer,
    TrackedFPQuantizer,
)


class _Site(object):
    def __init__(self, site, role, owner_kind):
        self.site = site
        self.role = role
        self.owner_kind = owner_kind


class _ConcatAdapter(object):
    def __init__(self):
        self.calls = []

    def disable(self):
        self.calls.append("disable")


class _Instrumentor(object):
    def __init__(self):
        self.external_ownership = None
        self.weight_formats = None
        self.activation_formats = None
        self.enabled_groups = None
        self.external_output_ownership = None
        self.runtime_statistics = None
        self.relu_quantizers = {"stale": object()}

    def set_external_ownership(self, inputs, outputs):
        self.external_ownership = (set(inputs), set(outputs))

    def configure_floating_point(
            self, weight_formats, activation_formats, enabled_groups,
            external_output_ownership):
        self.weight_formats = dict(weight_formats)
        self.activation_formats = dict(activation_formats)
        self.enabled_groups = set(enabled_groups)
        self.external_output_ownership = bool(external_output_ownership)

    def set_runtime_statistics(self, enabled):
        self.runtime_statistics = bool(enabled)


class _PropagationAdapter(object):
    def __init__(self):
        self.calls = []

    def configure_fp16(self):
        self.calls.append("fp16")


def _candidate():
    assignment = FPFormatAssignment(
        {
            "id_dec0.0": "fp6_e3m2",
            "id_dec1.0": "fp6_e3m2",
        },
        {
            ("activation::id_dec0.0::input", "module_input"):
            "fp8_e4m3fn",
            ("activation::id_dec1.0::input", "module_input"):
            "fp8_e4m3fn",
        },
    )
    return SimpleNamespace(assignment=assignment)


def _full_fp6_assignment():
    modules = EARLY_MODULES + INITIAL_DEPTH_MODULES + (
        "dec5.0", "dec4.0", "dec3.0", "dec2.0", "gd_dec1.0")
    return FPFormatAssignment(
        dict((name, "fp6_e3m2") for name in modules),
        dict((("activation::%s::input" % name, "module_input"),
              "fp6_e3m2") for name in modules),
    )


def _evaluator():
    sites = {
        "activation::id_dec0.0::input": _Site(
            "activation::id_dec0.0::input", "module_input", "module_input"),
        "activation::id_dec1.0::input": _Site(
            "activation::id_dec1.0::input", "module_input", "module_input"),
    }
    return SimpleNamespace(
        sites=sites,
        concat_adapter=_ConcatAdapter(),
        instrumentor=_Instrumentor(),
        propagation_adapter=_PropagationAdapter(),
        registry=SimpleNamespace(blocks=("initial_depth",)),
    )


def test_generic_activation_configuration_requires_module_boundaries():
    evaluator = _evaluator()
    formats = _generic_fp_activation_configuration(
        evaluator, _candidate().assignment)
    assert formats == {
        ("id_dec0.0", "input"): "fp8_e4m3fn",
        ("id_dec1.0", "input"): "fp8_e4m3fn",
    }

    evaluator.sites["activation::id_dec0.0::input"].owner_kind =         "concat_input"
    with pytest.raises(ValueError, match="module sites"):
        _generic_fp_activation_configuration(
            evaluator, _candidate().assignment)


def test_corrected_fp_configuration_has_one_generic_owner():
    evaluator = _evaluator()
    _configure_corrected_fp_candidate(evaluator, _candidate())

    assert evaluator.concat_adapter.calls == ["disable"]
    assert evaluator.instrumentor.external_ownership == (set(), set())
    assert evaluator.instrumentor.weight_formats["id_dec0.0"] ==         "fp6_e3m2"
    assert evaluator.instrumentor.weight_formats["id_dec1.0"] ==         "fp6_e3m2"
    assert evaluator.instrumentor.activation_formats[
        ("id_dec0.0", "input")] == "fp8_e4m3fn"
    assert evaluator.instrumentor.external_output_ownership is True
    assert evaluator.instrumentor.runtime_statistics is False
    assert evaluator.instrumentor.relu_quantizers == {}
    assert evaluator.propagation_adapter.calls == ["fp16"]


def test_cleared_external_ownership_applies_fp6_weight_and_one_input_qdq():
    model = nn.Sequential(nn.Conv2d(2, 2, 1, bias=False)).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([
            [[[0.13]], [[0.91]]],
            [[[-0.37]], [[1.73]]],
        ]))
    instrumentor = HardwareAlignedInstrumentor(
        model, lambda name, module: "block",
        externally_owned_inputs={"0"}, externally_owned_outputs={"0"})
    sample = torch.tensor([[
        [[0.2, 0.7], [1.1, -0.4]],
        [[0.5, -0.3], [0.9, 1.4]],
    ]])
    instrumentor.observe()
    with torch.no_grad():
        model(sample)
    instrumentor.freeze()

    original = model[0].weight.detach().clone()
    instrumentor.set_external_ownership(set(), set())
    instrumentor.configure_floating_point(
        {"0": "fp6_e3m2"},
        {("0", "input"): "fp8_e4m3fn"},
        {"block"},
        external_output_ownership=True)

    flattened = original.reshape(original.shape[0], -1)
    maximum = flattened.abs().amax(dim=1)
    direct = make_quantizer(
        "fp6_e3m2", maximum, broadcast_shape=(2, 1, 1, 1))
    expected, _ = direct.quantize_with_codes(original)
    torch.testing.assert_close(model[0].weight, expected, rtol=0.0, atol=0.0)
    assert bool((model[0].weight != original).any().item())

    quantizer = instrumentor.quantizers[("0", "input")]
    with torch.no_grad():
        model(sample)
    assert quantizer.numel == sample.numel()
    instrumentor.close()


def test_tracked_quantizer_separates_native_and_new_zero_codes():
    quantizer = TrackedFPQuantizer(
        make_quantizer("fp6_e3m2", torch.tensor(28.0)))
    quantizer.quantize_with_codes(torch.tensor([0.0, 0.01, 1.0, 40.0]))
    row = quantizer.diagnostics()
    assert row["calls"] == 1
    assert row["total_count"] == 4
    assert row["native_zero_count"] == 1
    assert row["new_zero_count"] == 1
    assert row["reference_nonzero_count"] == 3
    assert row["saturation_count"] == 1
    assert row["native_zero_ratio"] == pytest.approx(0.25)
    assert row["new_zero_ratio"] == pytest.approx(1.0 / 3.0)
    assert math.isfinite(row["nonzero_sqnr_db"])


def test_branch_quantizer_reports_sparse_damage_per_branch():
    quantizer = BranchIndependentFPQuantizer(
        "fp6_e3m2", torch.tensor([28.0, 28.0]),
        branch_channels=(1, 1), channel_dim=1)
    tensor = torch.tensor([[[[0.0, 0.01]], [[1.0, 40.0]]]])
    quantizer.quantize_with_codes(tensor)
    row = quantizer.diagnostics()
    assert row["calls"] == 1
    assert row["branch_0_native_zero_count"] == 1
    assert row["branch_0_new_zero_count"] == 1
    assert row["branch_0_reference_nonzero_count"] == 1
    assert row["branch_1_saturation_count"] == 1
    assert row["branch_1_new_zero_count"] == 0
    assert math.isfinite(row["branch_1_nonzero_sqnr_db"])


def test_branch_quantizer_supports_explicit_mixed_formats():
    quantizer = BranchIndependentFPQuantizer(
        ("fp6_e3m2", "fp8_e4m3fn"), torch.tensor([28.0, 448.0]),
        branch_channels=(1, 1), channel_dim=1)
    tensor = torch.tensor([[[[1.0]], [[1.0]]]])
    quantizer.quantize_with_codes(tensor)
    row = quantizer.diagnostics()
    assert row["branch_0_format"] == "fp6_e3m2"
    assert row["branch_1_format"] == "fp8_e4m3fn"


def test_activation_protection_candidate_matrix_is_exact_and_immutable():
    source = _full_fp6_assignment()
    base = _activation_protection_base_assignment(source)
    candidates = _activation_protection_candidates(base)
    assert tuple(candidate.name for candidate in candidates) == (
        "BASE",
        "DEPTH_A8",
        "RGB_A8",
        "STEM_A8",
        "EARLY_A8",
        "STEM_EARLY_A8",
        "STEM_EARLY_ID_BRANCH",
        "FULL_BRANCH_AWARE",
    )
    for candidate in candidates:
        assert set(candidate.assignment.weight_formats.values()) == {
            "fp6_e3m2"}
        for name in INITIAL_DEPTH_MODULES:
            owner = ("activation::%s::input" % name, "module_input")
            assert candidate.assignment.activation_formats[owner] == \
                "fp8_e4m3fn"
    selected = dict((candidate.name, candidate) for candidate in candidates)
    stem_formats = dict(selected["DEPTH_A8"].branch_formats)
    assert stem_formats["conv2.0.conv1"] == (
        "fp6_e3m2", "fp8_e4m3fn")
    assert dict(selected["RGB_A8"].branch_formats)["conv2.0.conv1"] == (
        "fp8_e4m3fn", "fp6_e3m2")
    assert tuple(dict(selected["STEM_EARLY_ID_BRANCH"].branch_formats)) == (
        "conv2.0.conv1", "id_dec1.0", "id_dec0.0")
    assert tuple(dict(selected["FULL_BRANCH_AWARE"].branch_formats)) == tuple(
        BRANCH_LAYOUTS)
    assert source.activation_formats[
        ("activation::id_dec0.0::input", "module_input")] == "fp6_e3m2"


def test_activation_protection_branch_layout_matches_official_nlspn():
    assert BRANCH_LAYOUTS == {
        "conv2.0.conv1": (48, 16),
        "dec4.0": (256, 512),
        "dec3.0": (128, 256),
        "dec2.0": (64, 128),
        "id_dec1.0": (64, 64),
        "id_dec0.0": (64, 64),
    }


def test_branch_maxima_follow_declared_channel_boundaries():
    maxima = torch.arange(1, 7, dtype=torch.float32)
    result = _branch_maxima(maxima, (2, 4))
    torch.testing.assert_close(result, torch.tensor([2.0, 6.0]))
    with pytest.raises(ValueError, match="channel count"):
        _branch_maxima(maxima, (2, 3))


def test_branch_layout_validation_requires_exact_consumer_shapes():
    modules = dict(
        (name, nn.Conv2d(sum(layout), 1, 1))
        for name, layout in BRANCH_LAYOUTS.items())
    evaluator = SimpleNamespace(
        instrumentor=SimpleNamespace(modules=modules))
    _validate_branch_layouts(evaluator)

    modules["dec4.0"] = nn.Conv2d(767, 1, 1)
    with pytest.raises(ValueError, match="input channels"):
        _validate_branch_layouts(evaluator)


def test_depth_error_row_reports_all_pooled_sums():
    prediction = torch.tensor([[[1.0, 4.0, 8.0]]])
    target = torch.tensor([[[2.0, 3.0, 0.0]]])
    row = _depth_error_row(prediction, target, sample_index=7)
    assert row["sample_index"] == 7
    assert row["valid_pixels"] == 2
    assert row["squared_error_sum"] == pytest.approx(2.0)
    assert row["absolute_error_sum"] == pytest.approx(2.0)
    assert row["absolute_relative_error_sum"] == pytest.approx(
        1.0 / 2.0 + 1.0 / 3.0)
    assert row["inverse_squared_error_sum"] == pytest.approx(
        (1.0 - 0.5) ** 2 + (0.25 - 1.0 / 3.0) ** 2)
    assert row["RMSE"] == pytest.approx(1.0)
    assert row["prediction_finite"] is True
    assert row["prediction_positive"] is True


def test_signal_error_row_reports_mse_sqnr_and_iteration():
    reference = torch.tensor([1.0, 2.0])
    quantized = torch.tensor([1.5, 1.0])
    row = _signal_error_row(
        "affinity", 0, reference, quantized)
    assert row["signal"] == "affinity"
    assert row["iteration"] == 0
    assert row["numel"] == 2
    assert row["mse"] == pytest.approx(0.625)
    assert row["mae"] == pytest.approx(0.75)
    assert row["max_abs_error"] == pytest.approx(1.0)
    assert row["sqnr_db"] == pytest.approx(
        10.0 * math.log10(5.0 / 1.25))


def test_aggregate_depth_rows_reports_pooled_depth_metrics():
    rows = (
        {
            "valid_pixels": 2,
            "squared_error_sum": 2.0,
            "absolute_error_sum": 2.0,
            "absolute_relative_error_sum": 1.0,
            "inverse_squared_error_sum": 0.5,
            "RMSE": 1.0,
        },
        {
            "valid_pixels": 3,
            "squared_error_sum": 3.0,
            "absolute_error_sum": 1.5,
            "absolute_relative_error_sum": 0.6,
            "inverse_squared_error_sum": 0.25,
            "RMSE": 1.0,
        },
    )
    metrics = _aggregate_depth_rows(rows)
    assert metrics["pooled_rmse"] == pytest.approx(1.0)
    assert metrics["mean_sample_rmse"] == pytest.approx(1.0)
    assert metrics["pooled_mae"] == pytest.approx(3.5 / 5.0)
    assert metrics["pooled_absrel"] == pytest.approx(1.6 / 5.0)
    assert metrics["pooled_irmse"] == pytest.approx(math.sqrt(0.75 / 5.0))
    assert metrics["valid_pixels"] == 5


def test_selection_uses_rmse_tolerance_then_activation_budget():
    rows = (
        {"configuration": "LOWEST", "pooled_rmse": 0.16000,
         "average_activation_bits": 6.5, "valid": True},
        {"configuration": "CHEAPER_TIE", "pooled_rmse": 0.16005,
         "average_activation_bits": 6.1, "valid": True},
        {"configuration": "OUTSIDE_TIE", "pooled_rmse": 0.16011,
         "average_activation_bits": 6.0, "valid": True},
    )
    selected = _select_activation_protection_result(rows)
    assert selected["configuration"] == "CHEAPER_TIE"


def test_stem_branch_budget_uses_rgb_and_depth_channel_fractions():
    base = _activation_protection_base_assignment(_full_fp6_assignment())
    candidates = dict(
        (candidate.name, candidate)
        for candidate in _activation_protection_candidates(base))
    costs = SimpleNamespace(
        weight_macs=tuple((name, 1) for name in base.weight_formats),
        activation_elements=tuple(
            (owner, 1) for owner in base.activation_formats))
    base_bits = _activation_protection_budget(
        candidates["BASE"], costs)[1]
    depth_bits = _activation_protection_budget(
        candidates["DEPTH_A8"], costs)[1]
    rgb_bits = _activation_protection_budget(
        candidates["RGB_A8"], costs)[1]
    assert base_bits < depth_bits < rgb_bits


def test_nlspn_output_capture_requires_all_signals_and_iterations():
    output = {
        "pred": torch.ones(1, 1, 2, 2),
        "pred_init": torch.full((1, 1, 2, 2), 2.0),
        "pred_inter": [torch.full((1, 1, 2, 2), float(index))
                       for index in range(1, 4)],
        "guidance": torch.full((1, 8, 2, 2), 3.0),
        "offset": torch.full((1, 18, 2, 2), 4.0),
        "aff": torch.full((1, 9, 2, 2), 5.0),
        "confidence": torch.full((1, 1, 2, 2), 0.5),
    }
    capture = _capture_nlspn_output(output, expected_iterations=3)
    assert tuple(capture) == (
        "prediction", "initial_depth", "guidance", "offset", "affinity",
        "confidence", "states")
    assert len(capture["states"]) == 3
    _require_equal_capture(capture, _capture_nlspn_output(output, 3))

    with pytest.raises(RuntimeError, match="iteration count"):
        _capture_nlspn_output(output, expected_iterations=2)


def test_early_fp16_candidate_matrix_is_exact():
    base = _activation_protection_base_assignment(_full_fp6_assignment())
    candidates = _early_fp16_candidates(base)
    assert tuple(candidate.name for candidate in candidates) == (
        "EARLY_W6A8",
        "EARLY_W6A16_CONV2_0_CONV1",
        "EARLY_W6A16_CONV2_0_CONV2",
        "EARLY_W6A16_CONV3_0_DOWNSAMPLE",
        "EARLY_W6A16",
        "EARLY_W16A8",
        "EARLY_W16A16",
    )
    selected = dict((candidate.name, candidate) for candidate in candidates)
    for name in EARLY_MODULES:
        owner = ("activation::%s::input" % name, "module_input")
        assert selected["EARLY_W6A8"].assignment.activation_formats[owner] == \
            "fp8_e4m3fn"
        assert selected["EARLY_W6A16"].assignment.activation_formats[owner] == \
            "fp16_ieee"
        assert selected["EARLY_W16A8"].assignment.weight_formats[name] == \
            "fp16_ieee"
        assert selected["EARLY_W16A16"].assignment.weight_formats[name] == \
            "fp16_ieee"
    owner = ("activation::conv2.0.conv1::input", "module_input")
    assert selected[
        "EARLY_W6A16_CONV2_0_CONV1"].assignment.activation_formats[owner] == \
        "fp16_ieee"
    assert set(selected["EARLY_W6A8"].assignment.weight_formats.values()) == {
        "fp6_e3m2"}


def test_decoder_attribution_candidate_matrix_is_exact_and_immutable():
    source = _activation_protection_base_assignment(_full_fp6_assignment())
    candidates = _decoder_attribution_candidates(source)
    assert DECODER_ATTRIBUTION_GROUPS == (
        ("dec5", ("dec5.0",)),
        ("dec4", ("dec4.0",)),
        ("dec3", ("dec3.0",)),
        ("dec2", ("dec2.0",)),
        ("guidance", ("gd_dec1.0",)),
        ("initial_depth", ("id_dec1.0", "id_dec0.0")),
    )
    assert tuple(candidate.name for candidate in candidates) == (
        "EARLY_W16A16_BASE",
        "ISO_DEC5_W16A16",
        "ISO_DEC4_W16A16",
        "ISO_DEC3_W16A16",
        "ISO_DEC2_W16A16",
        "ISO_GUIDANCE_W16A16",
        "ISO_INITIAL_DEPTH_W16A16",
        "PREFIX_DEC5_DEC4_W16A16",
        "PREFIX_DEC5_DEC4_DEC3_W16A16",
        "PREFIX_SHARED_DECODER_W16A16",
        "PREFIX_SHARED_GUIDANCE_W16A16",
        "PREFIX_SHARED_GUIDANCE_INITIAL_W16A16",
    )
    selected = dict((candidate.name, candidate) for candidate in candidates)
    base = selected["EARLY_W16A16_BASE"].assignment
    for name in EARLY_MODULES:
        owner = ("activation::%s::input" % name, "module_input")
        assert base.weight_formats[name] == "fp16_ieee"
        assert base.activation_formats[owner] == "fp16_ieee"
    for name in ("dec5.0", "dec4.0", "dec3.0", "dec2.0",
                 "gd_dec1.0"):
        owner = ("activation::%s::input" % name, "module_input")
        assert base.weight_formats[name] == "fp6_e3m2"
        assert base.activation_formats[owner] == "fp6_e3m2"
    for name in INITIAL_DEPTH_MODULES:
        owner = ("activation::%s::input" % name, "module_input")
        assert base.weight_formats[name] == "fp6_e3m2"
        assert base.activation_formats[owner] == "fp8_e4m3fn"
    final = selected[
        "PREFIX_SHARED_GUIDANCE_INITIAL_W16A16"].assignment
    for _, modules in DECODER_ATTRIBUTION_GROUPS:
        for name in modules:
            owner = ("activation::%s::input" % name, "module_input")
            assert final.weight_formats[name] == "fp16_ieee"
            assert final.activation_formats[owner] == "fp16_ieee"
    assert source.weight_formats["dec4.0"] == "fp6_e3m2"


def test_early_fp16_attribution_separates_activation_weight_and_interaction():
    rows = (
        {"configuration": "EARLY_W6A8", "pooled_rmse": 0.20},
        {"configuration": "EARLY_W6A16_CONV2_0_CONV1",
         "pooled_rmse": 0.19},
        {"configuration": "EARLY_W6A16_CONV2_0_CONV2",
         "pooled_rmse": 0.18},
        {"configuration": "EARLY_W6A16_CONV3_0_DOWNSAMPLE",
         "pooled_rmse": 0.17},
        {"configuration": "EARLY_W6A16", "pooled_rmse": 0.16},
        {"configuration": "EARLY_W16A8", "pooled_rmse": 0.15},
        {"configuration": "EARLY_W16A16", "pooled_rmse": 0.13},
    )
    values = dict(
        (row["term"], row["rmse_delta"])
        for row in _early_fp16_attribution_rows(rows, 0.10))
    assert values["early_activation_recovery"] == pytest.approx(0.04)
    assert values["early_weight_recovery"] == pytest.approx(0.05)
    assert values["weight_activation_interaction"] == pytest.approx(0.02)
    assert values["fully_protected_residual"] == pytest.approx(0.03)
    assert values["conv2.0.conv1_activation_recovery"] == pytest.approx(0.01)
    assert values["conv2.0.conv2_activation_recovery"] == pytest.approx(0.02)
    assert values[
        "conv3.0.downsample.0_activation_recovery"] == pytest.approx(0.03)


def _decoder_attribution_result_rows():
    values = (
        ("EARLY_W16A16_BASE", 0.200),
        ("ISO_DEC5_W16A16", 0.190),
        ("ISO_DEC4_W16A16", 0.180),
        ("ISO_DEC3_W16A16", 0.170),
        ("ISO_DEC2_W16A16", 0.160),
        ("ISO_GUIDANCE_W16A16", 0.150),
        ("ISO_INITIAL_DEPTH_W16A16", 0.140),
        ("PREFIX_DEC5_DEC4_W16A16", 0.175),
        ("PREFIX_DEC5_DEC4_DEC3_W16A16", 0.160),
        ("PREFIX_SHARED_DECODER_W16A16", 0.155),
        ("PREFIX_SHARED_GUIDANCE_W16A16", 0.145),
        ("PREFIX_SHARED_GUIDANCE_INITIAL_W16A16", 0.130),
    )
    return tuple(
        {"configuration": name, "pooled_rmse": rmse}
        for name, rmse in values)


def test_decoder_isolated_attribution_reports_group_and_interaction_recovery():
    rows = _decoder_isolated_attribution_rows(
        _decoder_attribution_result_rows())
    values = dict((row["term"], row["rmse_recovery"]) for row in rows)
    assert values["dec5_isolated"] == pytest.approx(0.010)
    assert values["dec4_isolated"] == pytest.approx(0.020)
    assert values["dec3_isolated"] == pytest.approx(0.030)
    assert values["dec2_isolated"] == pytest.approx(0.040)
    assert values["guidance_isolated"] == pytest.approx(0.050)
    assert values["initial_depth_isolated"] == pytest.approx(0.060)
    assert values["shared_decoder_interaction"] == pytest.approx(-0.055)
    assert values["downstream_interaction"] == pytest.approx(-0.140)


def test_decoder_cumulative_attribution_reports_ordered_marginal_recovery():
    rows = _decoder_cumulative_attribution_rows(
        _decoder_attribution_result_rows())
    assert tuple(row["group"] for row in rows) == (
        "dec5", "dec4", "dec3", "dec2", "guidance", "initial_depth",
        "total")
    values = dict((row["group"], row["rmse_recovery"]) for row in rows)
    assert values["dec5"] == pytest.approx(0.010)
    assert values["dec4"] == pytest.approx(0.015)
    assert values["dec3"] == pytest.approx(0.015)
    assert values["dec2"] == pytest.approx(0.005)
    assert values["guidance"] == pytest.approx(0.010)
    assert values["initial_depth"] == pytest.approx(0.015)
    assert values["total"] == pytest.approx(0.070)


def test_decoder_attribution_experiment_has_strict_artifact_schema():
    assert "decoder-attribution" in _experiment_runners()
    assert DECODER_ATTRIBUTION_ARTIFACTS == (
        "summary.csv",
        "sample_metrics.csv",
        "module_diagnostics.csv",
        "effective_weight_metrics.csv",
        "propagation_signal_metrics.csv",
        "propagation_state_metrics.csv",
        "isolated_attribution.csv",
        "cumulative_attribution.csv",
    )
