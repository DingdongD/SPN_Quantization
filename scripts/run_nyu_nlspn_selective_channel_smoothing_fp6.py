#!/usr/bin/env python3
"""Evaluate selective NLSPN channel smoothing under FP6 and FP16 propagation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.nyu_model_runtime import NYUModelRuntime
from scripts.run_nyu_four_model_fp4_fp8 import (
    FPFormatEvaluator,
    _format_assignment,
    _format_budget,
)
from scripts.run_nyu_model_p3t3_search import _site_boundary
from scripts.run_nyu_unified_fp16_task_aware_allocation import (
    _aggregate,
    _build_contract,
    _costs,
    _read_score_tables,
    _reference_metrics,
    _runtime_args,
    _settings,
    json_safe,
    load_protocol,
    ordinary_score_tables,
)
from spn_quant import mixed_precision
from spn_quant.fp_formats import FORMAT_SPECS, make_quantizer
from spn_quant.fp_mixed_precision import FPFormatAssignment
from spn_quant.model_contracts import propagation_owned_modules
from spn_quant.selective_channel_smoothing import (
    BranchIndependentFPQuantizer,
    ChannelSmoothedFPQuantizer,
    TrackedFPQuantizer,
    channel_smoothing_scales,
    smooth_conv_weight,
)


FORMAT_FP6 = "fp6_e3m2"
FORMAT_FP8 = "fp8_e4m3fn"
FORMAT_FP16 = "fp16_ieee"
ANALYSIS_MODULES = (
    "conv2.0.conv1",
    "conv2.0.conv2",
    "conv3.0.conv1",
    "conv6.0",
)
PROTECTED_MODULES = (
    "id_dec0.0",
    "id_dec1.0",
)
BOUNDARY_BRANCH_NAMES = {
    "id_dec0.0": ("id_fd1", "fe1"),
    "id_dec1.0": ("fd2", "fe2"),
}
EARLY_MODULES = (
    "conv2.0.conv1",
    "conv2.0.conv2",
    "conv3.0.downsample.0",
)
INITIAL_DEPTH_MODULES = (
    "id_dec1.0",
    "id_dec0.0",
)
DECODER_ATTRIBUTION_GROUPS = (
    ("dec5", ("dec5.0",)),
    ("dec4", ("dec4.0",)),
    ("dec3", ("dec3.0",)),
    ("dec2", ("dec2.0",)),
    ("guidance", ("gd_dec1.0",)),
    ("initial_depth", INITIAL_DEPTH_MODULES),
)
DECODER_ATTRIBUTION_ARTIFACTS = (
    "summary.csv",
    "sample_metrics.csv",
    "module_diagnostics.csv",
    "effective_weight_metrics.csv",
    "propagation_signal_metrics.csv",
    "propagation_state_metrics.csv",
    "isolated_attribution.csv",
    "cumulative_attribution.csv",
)
BRANCH_LAYOUTS = {
    "conv2.0.conv1": (48, 16),
    "dec4.0": (256, 512),
    "dec3.0": (128, 256),
    "dec2.0": (64, 128),
    "id_dec1.0": (64, 64),
    "id_dec0.0": (64, 64),
}
BRANCH_NAMES = {
    "conv2.0.conv1": ("fe1_rgb", "fe1_dep"),
    "dec4.0": ("fd5", "fe5"),
    "dec3.0": ("fd4", "fe4"),
    "dec2.0": ("fd3", "fe3"),
    "id_dec1.0": ("fd2", "fe2"),
    "id_dec0.0": ("id_fd1", "fe1"),
}
EPSILON = 1e-8


def _generic_fp_activation_configuration(evaluator, assignment):
    generic = {}
    for owner, format_name in assignment.activation_formats.items():
        site_name, role = owner
        site = evaluator.sites[site_name]
        if site.role != role:
            raise ValueError("FP activation role differs from contract")
        if site.owner_kind not in ("module_input", "module_output"):
            raise ValueError(
                "NLSPN activation protection requires module sites")
        boundary = _site_boundary(site)
        if boundary in generic and generic[boundary] != format_name:
            raise ValueError("one FP boundary has multiple formats")
        generic[boundary] = format_name
    return generic


def _configure_corrected_fp_candidate(evaluator, candidate):
    evaluator.concat_adapter.disable()
    evaluator.instrumentor.set_external_ownership(set(), set())
    activation_formats = _generic_fp_activation_configuration(
        evaluator, candidate.assignment)
    evaluator.instrumentor.configure_floating_point(
        dict(candidate.assignment.weight_formats), activation_formats,
        set(evaluator.registry.blocks), external_output_ownership=True)
    evaluator.instrumentor.set_runtime_statistics(False)
    evaluator.instrumentor.relu_quantizers = {}
    evaluator.propagation_adapter.configure_fp16()


@dataclass(frozen=True)
class SelectiveSmoothingCandidate:
    name: str
    assignment: FPFormatAssignment
    smooth_modules: tuple
    alpha: float
    scope: str
    weight_bits: int
    activation_bits: int
    branch_independent_modules: tuple


@dataclass(frozen=True)
class ActivationProtectionCandidate:
    name: str
    assignment: FPFormatAssignment
    branch_formats: tuple


def _activation_owner(module_name):
    return "activation::%s::input" % module_name, "module_input"


def _activation_protection_base_assignment(assignment):
    weights = dict(assignment.weight_formats)
    activations = dict(assignment.activation_formats)
    for name in INITIAL_DEPTH_MODULES:
        weights[name]
        owner = _activation_owner(name)
        activations[owner]
        activations[owner] = FORMAT_FP8
    return FPFormatAssignment(weights, activations)


def _promote_activation_modules(assignment, modules):
    return _set_activation_modules(assignment, modules, FORMAT_FP8)


def _set_activation_modules(assignment, modules, format_name):
    weights = dict(assignment.weight_formats)
    activations = dict(assignment.activation_formats)
    for name in modules:
        weights[name]
        owner = _activation_owner(name)
        activations[owner]
        activations[owner] = str(format_name)
    return FPFormatAssignment(weights, activations)


def _set_weight_modules(assignment, modules, format_name):
    weights = dict(assignment.weight_formats)
    activations = dict(assignment.activation_formats)
    for name in modules:
        weights[name]
        weights[name] = str(format_name)
    return FPFormatAssignment(weights, activations)


def _early_fp16_candidates(base_assignment):
    early_a8 = _set_activation_modules(
        base_assignment, EARLY_MODULES, FORMAT_FP8)
    early_a16 = _set_activation_modules(
        early_a8, EARLY_MODULES, FORMAT_FP16)
    early_w16a8 = _set_weight_modules(
        early_a8, EARLY_MODULES, FORMAT_FP16)
    return (
        ActivationProtectionCandidate("EARLY_W6A8", early_a8, ()),
        ActivationProtectionCandidate(
            "EARLY_W6A16_CONV2_0_CONV1",
            _set_activation_modules(
                early_a8, ("conv2.0.conv1",), FORMAT_FP16), ()),
        ActivationProtectionCandidate(
            "EARLY_W6A16_CONV2_0_CONV2",
            _set_activation_modules(
                early_a8, ("conv2.0.conv2",), FORMAT_FP16), ()),
        ActivationProtectionCandidate(
            "EARLY_W6A16_CONV3_0_DOWNSAMPLE",
            _set_activation_modules(
                early_a8, ("conv3.0.downsample.0",), FORMAT_FP16), ()),
        ActivationProtectionCandidate("EARLY_W6A16", early_a16, ()),
        ActivationProtectionCandidate("EARLY_W16A8", early_w16a8, ()),
        ActivationProtectionCandidate(
            "EARLY_W16A16",
            _set_activation_modules(
                early_w16a8, EARLY_MODULES, FORMAT_FP16), ()),
    )


def _protect_fp16_modules(assignment, modules):
    protected = _set_weight_modules(assignment, modules, FORMAT_FP16)
    return _set_activation_modules(protected, modules, FORMAT_FP16)


def _decoder_attribution_candidates(base_assignment):
    baseline = _protect_fp16_modules(
        base_assignment, EARLY_MODULES)
    groups = dict(DECODER_ATTRIBUTION_GROUPS)
    isolated = (
        ("ISO_DEC5_W16A16", groups["dec5"]),
        ("ISO_DEC4_W16A16", groups["dec4"]),
        ("ISO_DEC3_W16A16", groups["dec3"]),
        ("ISO_DEC2_W16A16", groups["dec2"]),
        ("ISO_GUIDANCE_W16A16", groups["guidance"]),
        ("ISO_INITIAL_DEPTH_W16A16", groups["initial_depth"]),
    )
    candidates = [ActivationProtectionCandidate(
        "EARLY_W16A16_BASE", baseline, ())]
    candidates.extend(
        ActivationProtectionCandidate(
            name, _protect_fp16_modules(baseline, modules), ())
        for name, modules in isolated)
    prefix = groups["dec5"] + groups["dec4"]
    cumulative = [
        ("PREFIX_DEC5_DEC4_W16A16", prefix),
    ]
    prefix += groups["dec3"]
    cumulative.append(("PREFIX_DEC5_DEC4_DEC3_W16A16", prefix))
    prefix += groups["dec2"]
    cumulative.append(("PREFIX_SHARED_DECODER_W16A16", prefix))
    prefix += groups["guidance"]
    cumulative.append(("PREFIX_SHARED_GUIDANCE_W16A16", prefix))
    prefix += groups["initial_depth"]
    cumulative.append((
        "PREFIX_SHARED_GUIDANCE_INITIAL_W16A16", prefix))
    candidates.extend(
        ActivationProtectionCandidate(
            name, _protect_fp16_modules(baseline, modules), ())
        for name, modules in cumulative)
    return tuple(candidates)


def _activation_protection_candidates(base_assignment):
    stem_boundary = "conv2.0.conv1"
    later_early = tuple(
        name for name in EARLY_MODULES if name != stem_boundary)
    stem_early = _promote_activation_modules(base_assignment, later_early)
    stem_fp8 = ((stem_boundary, (FORMAT_FP8, FORMAT_FP8)),)
    initial_depth_fp8 = tuple(
        (name, (FORMAT_FP8, FORMAT_FP8))
        for name in INITIAL_DEPTH_MODULES)
    decoder_fp6 = tuple(
        (name, (FORMAT_FP6, FORMAT_FP6))
        for name in ("dec4.0", "dec3.0", "dec2.0"))
    return (
        ActivationProtectionCandidate("BASE", base_assignment, ()),
        ActivationProtectionCandidate(
            "DEPTH_A8", base_assignment,
            ((stem_boundary, (FORMAT_FP6, FORMAT_FP8)),)),
        ActivationProtectionCandidate(
            "RGB_A8", base_assignment,
            ((stem_boundary, (FORMAT_FP8, FORMAT_FP6)),)),
        ActivationProtectionCandidate(
            "STEM_A8", base_assignment, stem_fp8),
        ActivationProtectionCandidate(
            "EARLY_A8",
            _promote_activation_modules(base_assignment, EARLY_MODULES), ()),
        ActivationProtectionCandidate(
            "STEM_EARLY_A8", stem_early, stem_fp8),
        ActivationProtectionCandidate(
            "STEM_EARLY_ID_BRANCH", stem_early,
            stem_fp8 + initial_depth_fp8),
        ActivationProtectionCandidate(
            "FULL_BRANCH_AWARE", stem_early,
            stem_fp8 + decoder_fp6 + initial_depth_fp8),
    )


def _branch_maxima(channel_maximum, branch_channels):
    maximum = torch.as_tensor(channel_maximum, dtype=torch.float32).reshape(-1)
    channels = tuple(int(value) for value in branch_channels)
    if maximum.numel() != sum(channels):
        raise ValueError("branch channel count differs from calibration")
    return torch.stack(tuple(
        values.max() for values in torch.split(maximum, channels)))


def _validate_branch_layouts(evaluator):
    modules = evaluator.instrumentor.modules
    for name, branch_channels in BRANCH_LAYOUTS.items():
        module = modules[name]
        if not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            raise TypeError("branch consumer must be Conv2d or ConvTranspose2d")
        if int(module.in_channels) != sum(branch_channels):
            raise ValueError("branch consumer input channels differ: %s" % name)


def _depth_error_row(prediction, target, sample_index):
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    valid = torch.isfinite(target) & (target > 1e-4)
    valid_pixels = int(valid.sum().item())
    if valid_pixels <= 0:
        raise ValueError("depth sample has no valid pixels")
    values = prediction[valid].double()
    reference = target[valid].double()
    if not bool(torch.isfinite(values).all().item()):
        raise RuntimeError("depth prediction is non-finite")
    if bool((values <= 1e-4).any().item()):
        raise RuntimeError("depth prediction is non-positive")
    difference = values - reference
    absolute = difference.abs()
    inverse = values.reciprocal() - reference.reciprocal()
    squared_error_sum = float(difference.square().sum().item())
    return {
        "sample_index": int(sample_index),
        "valid_pixels": valid_pixels,
        "squared_error_sum": squared_error_sum,
        "absolute_error_sum": float(absolute.sum().item()),
        "absolute_relative_error_sum":
        float((absolute / reference).sum().item()),
        "inverse_squared_error_sum":
        float(inverse.square().sum().item()),
        "RMSE": math.sqrt(squared_error_sum / float(valid_pixels)),
        "prediction_finite": True,
        "prediction_positive": True,
    }


def _signal_error_row(signal, iteration, reference, quantized):
    if reference.shape != quantized.shape or reference.numel() <= 0:
        raise ValueError("signal tensors differ")
    if not bool(torch.isfinite(reference).all().item()) or \
            not bool(torch.isfinite(quantized).all().item()):
        raise RuntimeError("signal tensor is non-finite")
    source = reference.double()
    difference = source - quantized.double()
    signal_energy = float(source.square().sum().item())
    error_energy = float(difference.square().sum().item())
    if signal_energy <= 0.0:
        raise RuntimeError("signal reference energy is zero")
    return {
        "signal": str(signal),
        "iteration": int(iteration),
        "numel": int(reference.numel()),
        "mse": error_energy / float(reference.numel()),
        "mae": float(difference.abs().mean().item()),
        "max_abs_error": float(difference.abs().max().item()),
        "sqnr_db": float("inf") if error_energy == 0.0 else
        10.0 * math.log10(signal_energy / error_energy),
    }


def _aggregate_depth_rows(rows):
    if not rows:
        raise ValueError("depth metric rows are empty")
    valid_pixels = sum(int(row["valid_pixels"]) for row in rows)
    if valid_pixels <= 0:
        raise ValueError("depth metric rows contain no valid pixels")
    squared_error = sum(float(row["squared_error_sum"]) for row in rows)
    absolute_error = sum(float(row["absolute_error_sum"]) for row in rows)
    absolute_relative_error = sum(
        float(row["absolute_relative_error_sum"]) for row in rows)
    inverse_squared_error = sum(
        float(row["inverse_squared_error_sum"]) for row in rows)
    return {
        "pooled_rmse": math.sqrt(squared_error / float(valid_pixels)),
        "mean_sample_rmse": sum(float(row["RMSE"]) for row in rows) /
        float(len(rows)),
        "pooled_mae": absolute_error / float(valid_pixels),
        "pooled_absrel": absolute_relative_error / float(valid_pixels),
        "pooled_irmse": math.sqrt(
            inverse_squared_error / float(valid_pixels)),
        "valid_pixels": valid_pixels,
    }


def _select_activation_protection_result(rows):
    valid = tuple(row for row in rows if bool(row["valid"]))
    if not valid:
        raise RuntimeError("activation protection produced no valid result")
    best_rmse = min(float(row["pooled_rmse"]) for row in valid)
    tied = tuple(
        row for row in valid
        if float(row["pooled_rmse"]) - best_rmse < 0.0001)
    return min(
        tied,
        key=lambda row: (
            float(row["average_activation_bits"]),
            float(row["pooled_rmse"]), str(row["configuration"])))


def _early_fp16_attribution_rows(result_rows, fp32_rmse):
    values = dict(
        (row["configuration"], float(row["pooled_rmse"]))
        for row in result_rows)
    required = (
        "EARLY_W6A8",
        "EARLY_W6A16_CONV2_0_CONV1",
        "EARLY_W6A16_CONV2_0_CONV2",
        "EARLY_W6A16_CONV3_0_DOWNSAMPLE",
        "EARLY_W6A16",
        "EARLY_W16A8",
        "EARLY_W16A16",
    )
    if set(values) != set(required):
        raise ValueError("early FP16 attribution result coverage differs")
    baseline = values["EARLY_W6A8"]
    rows = [
        {"term": "early_activation_recovery", "rmse_delta":
         baseline - values["EARLY_W6A16"]},
        {"term": "early_weight_recovery", "rmse_delta":
         baseline - values["EARLY_W16A8"]},
        {"term": "weight_activation_interaction", "rmse_delta":
         values["EARLY_W16A16"] - values["EARLY_W6A16"] -
         values["EARLY_W16A8"] + baseline},
        {"term": "fully_protected_residual", "rmse_delta":
         values["EARLY_W16A16"] - float(fp32_rmse)},
    ]
    sites = (
        ("conv2.0.conv1", "EARLY_W6A16_CONV2_0_CONV1"),
        ("conv2.0.conv2", "EARLY_W6A16_CONV2_0_CONV2"),
        ("conv3.0.downsample.0",
         "EARLY_W6A16_CONV3_0_DOWNSAMPLE"),
    )
    for site, configuration in sites:
        rows.append({
            "term": "%s_activation_recovery" % site,
            "rmse_delta": baseline - values[configuration],
        })
    return tuple(rows)


def _decoder_attribution_values(result_rows):
    values = dict(
        (row["configuration"], float(row["pooled_rmse"]))
        for row in result_rows)
    required = (
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
    if len(result_rows) != len(required) or set(values) != set(required):
        raise ValueError("decoder attribution result coverage differs")
    return values


def _decoder_isolated_attribution_rows(result_rows):
    values = _decoder_attribution_values(result_rows)
    baseline = values["EARLY_W16A16_BASE"]
    isolated = (
        ("dec5", "ISO_DEC5_W16A16"),
        ("dec4", "ISO_DEC4_W16A16"),
        ("dec3", "ISO_DEC3_W16A16"),
        ("dec2", "ISO_DEC2_W16A16"),
        ("guidance", "ISO_GUIDANCE_W16A16"),
        ("initial_depth", "ISO_INITIAL_DEPTH_W16A16"),
    )
    rows = []
    recoveries = {}
    for group, configuration in isolated:
        recovery = baseline - values[configuration]
        recoveries[group] = recovery
        rows.append({
            "term": "%s_isolated" % group,
            "configuration": configuration,
            "baseline_rmse": baseline,
            "pooled_rmse": values[configuration],
            "rmse_recovery": recovery,
        })
    shared_recovery = baseline - values["PREFIX_SHARED_DECODER_W16A16"]
    rows.append({
        "term": "shared_decoder_interaction",
        "configuration": "PREFIX_SHARED_DECODER_W16A16",
        "baseline_rmse": baseline,
        "pooled_rmse": values["PREFIX_SHARED_DECODER_W16A16"],
        "rmse_recovery": shared_recovery - sum(
            recoveries[name] for name in ("dec5", "dec4", "dec3", "dec2")),
    })
    total_recovery = baseline - values[
        "PREFIX_SHARED_GUIDANCE_INITIAL_W16A16"]
    rows.append({
        "term": "downstream_interaction",
        "configuration": "PREFIX_SHARED_GUIDANCE_INITIAL_W16A16",
        "baseline_rmse": baseline,
        "pooled_rmse": values[
            "PREFIX_SHARED_GUIDANCE_INITIAL_W16A16"],
        "rmse_recovery": total_recovery - sum(recoveries.values()),
    })
    return tuple(rows)


def _decoder_cumulative_attribution_rows(result_rows):
    values = _decoder_attribution_values(result_rows)
    steps = (
        ("dec5", "ISO_DEC5_W16A16"),
        ("dec4", "PREFIX_DEC5_DEC4_W16A16"),
        ("dec3", "PREFIX_DEC5_DEC4_DEC3_W16A16"),
        ("dec2", "PREFIX_SHARED_DECODER_W16A16"),
        ("guidance", "PREFIX_SHARED_GUIDANCE_W16A16"),
        ("initial_depth", "PREFIX_SHARED_GUIDANCE_INITIAL_W16A16"),
    )
    baseline_name = "EARLY_W16A16_BASE"
    previous = baseline_name
    rows = []
    for group, configuration in steps:
        rows.append({
            "group": group,
            "previous_configuration": previous,
            "configuration": configuration,
            "previous_rmse": values[previous],
            "pooled_rmse": values[configuration],
            "rmse_recovery": values[previous] - values[configuration],
        })
        previous = configuration
    final = steps[-1][1]
    rows.append({
        "group": "total",
        "previous_configuration": baseline_name,
        "configuration": final,
        "previous_rmse": values[baseline_name],
        "pooled_rmse": values[final],
        "rmse_recovery": values[baseline_name] - values[final],
    })
    return tuple(rows)


def _capture_nlspn_output(output, expected_iterations):
    if not isinstance(output, dict):
        raise TypeError("official NLSPN output must be a dictionary")
    keys = ("pred", "pred_init", "pred_inter", "guidance", "offset",
            "aff", "confidence")
    for key in keys:
        output[key]
    states = output["pred_inter"]
    if not isinstance(states, (tuple, list)):
        raise TypeError("NLSPN propagation states must be a sequence")
    if len(states) != int(expected_iterations):
        raise RuntimeError("NLSPN propagation iteration count differs")
    tensors = {
        "prediction": output["pred"],
        "initial_depth": output["pred_init"],
        "guidance": output["guidance"],
        "offset": output["offset"],
        "affinity": output["aff"],
        "confidence": output["confidence"],
    }
    for name, tensor in tensors.items():
        if not torch.is_tensor(tensor) or tensor.numel() <= 0:
            raise TypeError("NLSPN %s must be a nonempty tensor" % name)
        if not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError("NLSPN %s is non-finite" % name)
    for state in states:
        if not torch.is_tensor(state) or state.numel() <= 0:
            raise TypeError("NLSPN propagation state must be nonempty")
        if not bool(torch.isfinite(state).all().item()):
            raise RuntimeError("NLSPN propagation state is non-finite")
    return {
        "prediction": tensors["prediction"].detach().cpu().clone(),
        "initial_depth": tensors["initial_depth"].detach().cpu().clone(),
        "guidance": tensors["guidance"].detach().cpu().clone(),
        "offset": tensors["offset"].detach().cpu().clone(),
        "affinity": tensors["affinity"].detach().cpu().clone(),
        "confidence": tensors["confidence"].detach().cpu().clone(),
        "states": tuple(state.detach().cpu().clone() for state in states),
    }


def _require_equal_capture(first, second):
    names = ("prediction", "initial_depth", "guidance", "offset",
             "affinity", "confidence")
    for name in names:
        if not torch.equal(first[name], second[name]):
            raise RuntimeError("paired NLSPN %s differs" % name)
    if len(first["states"]) != len(second["states"]):
        raise RuntimeError("paired NLSPN propagation iteration count differs")
    for iteration, (left, right) in enumerate(
            zip(first["states"], second["states"]), 1):
        if not torch.equal(left, right):
            raise RuntimeError(
                "paired NLSPN state differs at iteration %d" % iteration)


def _forward_nlspn_capture(evaluator, batch):
    model_args, ground_truth = evaluator.runtime.model_input(
        batch, evaluator.device)
    output = evaluator.model(*model_args)
    capture = _capture_nlspn_output(
        output, evaluator.model.prop_layer.prop_time)
    return capture, ground_truth.detach().cpu().clone()


def _input_channel_dim(module):
    if isinstance(module, nn.ConvTranspose2d):
        return 0
    if isinstance(module, nn.Conv2d):
        return 1
    raise TypeError("selective channel smoothing requires Conv modules")


def _channel_maximum(observer):
    if not observer.observed:
        raise RuntimeError("channel smoothing observer is empty")
    return torch.maximum(observer.minimum.abs(), observer.maximum.abs())


def _imbalance(values):
    values = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    positive = values[values > 0.0]
    if positive.numel() == 0:
        raise ValueError("channel imbalance requires positive values")
    return float(positive.max().item() / positive.median().item())


def _weight_quantize(source, module, format_name):
    output_dim = 1 if isinstance(module, nn.ConvTranspose2d) else 0
    flattened = source.movedim(output_dim, 0).reshape(
        source.shape[output_dim], -1)
    maximum = flattened.abs().amax(dim=1)
    shape = [1] * source.ndim
    shape[output_dim] = source.shape[output_dim]
    quantizer = make_quantizer(
        format_name, maximum, broadcast_shape=tuple(shape))
    quantized, _ = quantizer.quantize_with_codes(source)
    difference = source.double() - quantized.double()
    signal = float(source.double().square().sum().item())
    error = float(difference.square().sum().item())
    sqnr = float("inf") if error == 0.0 else 10.0 * math.log10(signal / error)
    return quantized, quantizer, sqnr


def _initial_depth_branch_channels(channels):
    if int(channels) != 128:
        raise ValueError("official initial-depth concat requires 128 channels")
    return 64, 64


class NLSPNSelectiveSmoothingEvaluator(FPFormatEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        missing = set(ANALYSIS_MODULES + PROTECTED_MODULES) - \
            set(self.instrumentor.modules)
        if missing:
            raise KeyError("NLSPN selective modules are missing: %s" %
                           sorted(missing))
        self.smoothing_diagnostics = {}
        self.boundary_quantizers = {}

    def _configure_candidate(self, candidate):
        super()._configure_candidate(candidate)
        self.smoothing_diagnostics = {}
        self.boundary_quantizers = {}
        smooth = set(candidate.smooth_modules)
        if not smooth <= set(ANALYSIS_MODULES):
            raise ValueError("unknown selective smoothing modules")
        for name in ANALYSIS_MODULES:
            module = self.instrumentor.modules[name]
            key = (name, "input")
            owner = ("activation::%s::input" % name, "module_input")
            activation_format = candidate.assignment.activation_formats[owner]
            weight_format = candidate.assignment.weight_formats[name]
            activation_maximum = _channel_maximum(
                self.instrumentor.channel_observers[key])
            original_weight = self.instrumentor.original_weights[name].to(
                device=module.weight.device, dtype=module.weight.dtype)
            input_dim = _input_channel_dim(module)
            scales = torch.ones_like(activation_maximum)
            if name in smooth:
                scales = channel_smoothing_scales(
                    activation_maximum, original_weight.cpu(), input_dim,
                    candidate.alpha, EPSILON)
            transformed_weight = smooth_conv_weight(
                original_weight, scales, input_dim)
            quantized_weight, weight_quantizer, weight_sqnr = _weight_quantize(
                transformed_weight, module, weight_format)
            with torch.no_grad():
                module.weight.copy_(quantized_weight)
            self.instrumentor.weight_scales[name] = weight_quantizer.scale
            transformed_activation_maximum = activation_maximum / scales
            activation_quantizer = ChannelSmoothedFPQuantizer(
                activation_format,
                float(transformed_activation_maximum.max().item()),
                scales,
                channel_dim=1)
            self.instrumentor.quantizers[key] = activation_quantizer
            self.smoothing_diagnostics[name] = {
                "smoothed": name in smooth,
                "alpha": candidate.alpha,
                "activation_channel_imbalance_before":
                _imbalance(activation_maximum),
                "activation_channel_imbalance_after":
                _imbalance(transformed_activation_maximum),
                "activation_maximum_before":
                float(activation_maximum.max().item()),
                "activation_maximum_after":
                float(transformed_activation_maximum.max().item()),
                "weight_sqnr_db": weight_sqnr,
                "channel_scale_minimum": float(scales.min().item()),
                "channel_scale_maximum": float(scales.max().item()),
                "activation_quantizer": activation_quantizer,
            }
        unknown = set(candidate.branch_independent_modules) - \
            set(PROTECTED_MODULES)
        if unknown:
            raise ValueError("unknown branch-independent modules: %s" %
                             sorted(unknown))
        for name in PROTECTED_MODULES:
            module = self.instrumentor.modules[name]
            key = (name, "input")
            owner = ("activation::%s::input" % name, "module_input")
            format_name = candidate.assignment.activation_formats[owner]
            channel_maximum = _channel_maximum(
                self.instrumentor.channel_observers[key])
            channels = int(module.in_channels)
            branch_channels = _initial_depth_branch_channels(channels)
            if channel_maximum.numel() != channels:
                raise ValueError(
                    "initial-depth channel calibration differs from model")
            first, second = torch.split(channel_maximum, branch_channels)
            branch_maxima = torch.stack((first.max(), second.max()))
            if name not in candidate.branch_independent_modules:
                branch_maxima = branch_maxima.max().repeat(2)
            quantizer = BranchIndependentFPQuantizer(
                format_name, branch_maxima, branch_channels, channel_dim=1)
            self.instrumentor.quantizers[key] = quantizer
            self.boundary_quantizers[name] = quantizer
        self.instrumentor.set_runtime_statistics(False)

    def diagnostic_rows(self, candidate_name):
        rows = []
        for name in ANALYSIS_MODULES:
            values = self.smoothing_diagnostics[name]
            quantizer = values["activation_quantizer"]
            runtime = quantizer.diagnostics()
            rows.append({
                "configuration": candidate_name,
                "module": name,
                "smoothed": values["smoothed"],
                "alpha": values["alpha"],
                "activation_channel_imbalance_before":
                values["activation_channel_imbalance_before"],
                "activation_channel_imbalance_after":
                values["activation_channel_imbalance_after"],
                "activation_maximum_before":
                values["activation_maximum_before"],
                "activation_maximum_after":
                values["activation_maximum_after"],
                "activation_sqnr_db": runtime["sqnr_db"],
                "activation_zero_code_ratio": runtime["zero_code_ratio"],
                "activation_saturation_ratio": runtime["saturation_ratio"],
                "weight_sqnr_db": values["weight_sqnr_db"],
                "channel_scale_minimum": values["channel_scale_minimum"],
                "channel_scale_maximum": values["channel_scale_maximum"],
            })
        return rows


class NLSPNActivationProtectionEvaluator(FPFormatEvaluator):
    """Evaluate one corrected FP owner and collect activation damage."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _validate_branch_layouts(self)
        self.tracked_quantizers = {}
        self.branch_quantizers = {}

    def _configure_candidate(self, candidate):
        _configure_corrected_fp_candidate(self, candidate)
        self.tracked_quantizers = {}
        self.branch_quantizers = {}
        branch_formats = dict(candidate.branch_formats)
        branch_modules = set(branch_formats)
        unknown = branch_modules - set(BRANCH_LAYOUTS)
        if unknown:
            raise ValueError("unknown branch-aware modules: %s" %
                             sorted(unknown))
        formats = candidate.assignment.activation_formats
        for name in self.instrumentor.modules:
            owner = _activation_owner(name)
            if owner not in formats:
                continue
            format_name = formats[owner]
            if format_name == FORMAT_FP6 and name not in branch_modules:
                continue
            key = (name, "input")
            self.instrumentor.quantizers[key]
            if name in branch_modules:
                channel_maximum = _channel_maximum(
                    self.instrumentor.channel_observers[key])
                branch_channels = BRANCH_LAYOUTS[name]
                quantizer = BranchIndependentFPQuantizer(
                    branch_formats[name],
                    _branch_maxima(channel_maximum, branch_channels),
                    branch_channels, channel_dim=1)
                self.branch_quantizers[name] = quantizer
            else:
                quantizer = TrackedFPQuantizer(
                    self.instrumentor.quantizers[key])
            self.instrumentor.quantizers[key] = quantizer
            self.tracked_quantizers[name] = quantizer

    def module_diagnostic_rows(self, configuration):
        rows = []
        for name, quantizer in self.tracked_quantizers.items():
            if name in self.branch_quantizers:
                continue
            values = quantizer.diagnostics()
            if values["calls"] != 2:
                raise RuntimeError("module input QDQ call count differs")
            rows.append({
                "configuration": configuration,
                "module": name,
                "format": quantizer.format,
                "input_qdq_calls": values["calls"],
                "total_count": values["total_count"],
                "native_zero_count": values["native_zero_count"],
                "reference_nonzero_count":
                values["reference_nonzero_count"],
                "new_zero_count": values["new_zero_count"],
                "saturation_count": values["saturation_count"],
                "native_zero_ratio": values["native_zero_ratio"],
                "new_zero_ratio": values["new_zero_ratio"],
                "saturation_ratio": values["saturation_ratio"],
                "nonzero_sqnr_db": values["nonzero_sqnr_db"],
            })
        return rows

    def branch_diagnostic_rows(self, configuration):
        rows = []
        for name, quantizer in self.branch_quantizers.items():
            values = quantizer.diagnostics()
            if values["calls"] != 2:
                raise RuntimeError("branch input QDQ call count differs")
            for index, branch_name in enumerate(BRANCH_NAMES[name]):
                prefix = "branch_%d_" % index
                rows.append({
                    "configuration": configuration,
                    "module": name,
                    "branch": branch_name,
                    "format": values[prefix + "format"],
                    "input_qdq_calls": values["calls"],
                    "calibration_maximum":
                    values[prefix + "calibration_maximum"],
                    "reference_maximum":
                    values[prefix + "reference_maximum"],
                    "reference_p99": values[prefix + "reference_p99"],
                    "reference_rms": values[prefix + "reference_rms"],
                    "total_count": values[prefix + "total_count"],
                    "native_zero_count":
                    values[prefix + "native_zero_count"],
                    "reference_nonzero_count":
                    values[prefix + "reference_nonzero_count"],
                    "new_zero_count": values[prefix + "new_zero_count"],
                    "saturation_count":
                    values[prefix + "saturation_count"],
                    "native_zero_ratio":
                    values[prefix + "native_zero_ratio"],
                    "new_zero_ratio": values[prefix + "new_zero_ratio"],
                    "saturation_ratio":
                    values[prefix + "saturation_ratio"],
                    "nonzero_sqnr_db":
                    values[prefix + "nonzero_sqnr_db"],
                })
        return rows

    def effective_weight_rows(self, candidate):
        rows = []
        for name, module in self.instrumentor.modules.items():
            reference = self.instrumentor.original_weights[name].double()
            effective = module.weight.detach().cpu().double()
            difference = reference - effective
            signal = float(reference.square().sum().item())
            error = float(difference.square().sum().item())
            if signal <= 0.0:
                raise RuntimeError("effective weight reference is empty")
            rows.append({
                "configuration": candidate.name,
                "module": name,
                "format": candidate.assignment.weight_formats[name],
                "weight_elements": int(reference.numel()),
                "changed_weight_elements":
                int((reference != effective).sum().item()),
                "weight_mse": error / float(reference.numel()),
                "weight_sqnr_db": float("inf") if error == 0.0 else
                10.0 * math.log10(signal / error),
            })
        return rows

    def boundary_diagnostic_rows(self, candidate):
        rows = []
        for module_name in PROTECTED_MODULES:
            quantizer = self.boundary_quantizers[module_name]
            diagnostics = quantizer.diagnostics()
            weight_bits, activation_bits = _boundary_module_bits(
                candidate, module_name)
            for branch, branch_name in enumerate(
                    BOUNDARY_BRANCH_NAMES[module_name]):
                prefix = "branch_%d_" % branch
                rows.append({
                    "configuration": candidate.name,
                    "scope": candidate.scope,
                    "module": module_name,
                    "branch": branch_name,
                    "weight_bits": weight_bits,
                    "activation_bits": activation_bits,
                    "branch_independent": module_name in
                    candidate.branch_independent_modules,
                    "calibration_maximum":
                    diagnostics[prefix + "calibration_maximum"],
                    "reference_maximum":
                    diagnostics[prefix + "reference_maximum"],
                    "reference_p99": diagnostics[prefix + "reference_p99"],
                    "reference_rms": diagnostics[prefix + "reference_rms"],
                    "sqnr_db": diagnostics[prefix + "sqnr_db"],
                    "zero_code_ratio":
                    diagnostics[prefix + "zero_code_ratio"],
                    "saturation_ratio":
                    diagnostics[prefix + "saturation_ratio"],
                })
        return rows


def _disable_activation_protection_quantization(evaluator):
    evaluator.instrumentor.disable()
    evaluator.concat_adapter.disable()
    if evaluator.propagation_projection_instrumentor is not None:
        evaluator.propagation_projection_instrumentor.disable()
    evaluator.propagation_adapter.disable()
    if evaluator.joint_adapter is not None:
        evaluator.joint_adapter.unbind_qdrop_sites()


def _sample_depth_rows(configuration, capture, ground_truth,
                       evaluation_batches):
    prediction = capture["prediction"]
    if prediction.shape[0] != len(evaluation_batches) or \
            ground_truth.shape[0] != len(evaluation_batches):
        raise RuntimeError("NLSPN evaluation batch output count differs")
    rows = []
    for position, (sample_index, batch) in enumerate(evaluation_batches):
        del batch
        row = _depth_error_row(
            prediction[position], ground_truth[position], sample_index)
        row["configuration"] = str(configuration)
        rows.append(row)
    return rows


def _propagation_error_rows(configuration, reference, quantized):
    rows = []
    for signal in ("initial_depth", "guidance", "offset", "affinity",
                   "confidence", "prediction"):
        iteration = len(reference["states"]) if signal == "prediction" else 0
        row = _signal_error_row(
            signal, iteration, reference[signal], quantized[signal])
        row["configuration"] = str(configuration)
        rows.append(row)
    state_rows = []
    if len(reference["states"]) != len(quantized["states"]):
        raise RuntimeError("NLSPN propagation state count differs")
    for iteration, (source, result) in enumerate(
            zip(reference["states"], quantized["states"]), 1):
        row = _signal_error_row("state", iteration, source, result)
        row["configuration"] = str(configuration)
        state_rows.append(row)
    return rows, state_rows


def _activation_protection_budget(candidate, costs):
    weight_bits, _, weight_fractions, _ = _format_budget(
        candidate.assignment, costs)
    activation_costs = dict(costs.activation_elements)
    branch_formats = dict(candidate.branch_formats)
    format_costs = dict((name, 0.0) for name in FORMAT_SPECS)
    for owner, format_name in candidate.assignment.activation_formats.items():
        cost = float(activation_costs[owner])
        module_name = owner[0].split("::")[1]
        if module_name not in branch_formats:
            format_costs[format_name] += cost
            continue
        channels = BRANCH_LAYOUTS[module_name]
        formats = branch_formats[module_name]
        if len(formats) != len(channels):
            raise ValueError("branch format and channel counts differ")
        total_channels = float(sum(channels))
        for branch_channels, branch_format in zip(channels, formats):
            format_costs[branch_format] += \
                cost * float(branch_channels) / total_channels
    total_cost = sum(format_costs.values())
    if total_cost <= 0.0:
        raise RuntimeError("activation protection budget is empty")
    activation_fractions = dict(
        (name, cost / total_cost) for name, cost in format_costs.items()
        if cost > 0.0)
    activation_bits = sum(
        float(FORMAT_SPECS[name].bits) * fraction
        for name, fraction in activation_fractions.items())
    return weight_bits, activation_bits, weight_fractions, \
        activation_fractions


def _assignment_payload(candidate, costs):
    weight_bits, activation_bits, weight_fractions, activation_fractions = \
        _activation_protection_budget(candidate, costs)
    return {
        "average_weight_bits": weight_bits,
        "average_activation_bits": activation_bits,
        "weight_formats": dict(candidate.assignment.weight_formats),
        "activation_formats": {
            "%s::%s" % owner: format_name
            for owner, format_name in
            candidate.assignment.activation_formats.items()
        },
        "weight_format_fractions": weight_fractions,
        "activation_format_fractions": activation_fractions,
        "branch_formats": dict(
            (name, list(formats)) for name, formats in
            candidate.branch_formats),
    }


def _reference_activation_protection(evaluator):
    _disable_activation_protection_quantization(evaluator)
    with torch.no_grad():
        capture, ground_truth = _forward_nlspn_capture(
            evaluator, evaluator.evaluation_batch)
    rows = _sample_depth_rows(
        "FP32", capture, ground_truth, evaluator.evaluation_batches)
    return capture, ground_truth, rows, _aggregate_depth_rows(rows)


def _evaluate_activation_protection_candidate(
        evaluator, candidate, reference_capture, reference_ground_truth):
    evaluator._configure_candidate(candidate)
    effective_weights = evaluator.effective_weight_rows(candidate)
    with torch.no_grad():
        first, ground_truth = _forward_nlspn_capture(
            evaluator, evaluator.evaluation_batch)
        second, second_ground_truth = _forward_nlspn_capture(
            evaluator, evaluator.evaluation_batch)
    if not torch.equal(reference_ground_truth, ground_truth) or \
            not torch.equal(ground_truth, second_ground_truth):
        raise RuntimeError("paired NLSPN ground truth changed")
    _require_equal_capture(first, second)
    depth_rows = _sample_depth_rows(
        candidate.name, first, ground_truth, evaluator.evaluation_batches)
    metrics = _aggregate_depth_rows(depth_rows)
    signal_rows, state_rows = _propagation_error_rows(
        candidate.name, reference_capture, first)
    module_rows = evaluator.module_diagnostic_rows(candidate.name)
    branch_rows = evaluator.branch_diagnostic_rows(candidate.name)
    return {
        "metrics": metrics,
        "depth_rows": depth_rows,
        "signal_rows": signal_rows,
        "state_rows": state_rows,
        "module_rows": module_rows,
        "branch_rows": branch_rows,
        "effective_weight_rows": effective_weights,
    }

def _protected_fp6_assignment(evaluator, activation_scores, activation_costs):
    assignment = _format_assignment(
        evaluator, FORMAT_FP6, FORMAT_FP6,
        activation_scores, activation_costs, 0.0)
    weights = dict(assignment.weight_formats)
    activations = dict(assignment.activation_formats)
    for name in PROTECTED_MODULES:
        weights[name] = FORMAT_FP8
        owner = ("activation::%s::input" % name, "module_input")
        activations[owner] = FORMAT_FP8
    return FPFormatAssignment(weights, activations)


def _candidates(assignment):
    return (
        SelectiveSmoothingCandidate(
            "PROTECTED_FP6_BASELINE", assignment, (), 0.0,
            "smooth", 8, 8, ()),
        SelectiveSmoothingCandidate(
            "SMOOTH_CONV2_ALPHA_050", assignment,
            ("conv2.0.conv1", "conv2.0.conv2"), 0.5,
            "smooth", 8, 8, ()),
        SelectiveSmoothingCandidate(
            "SMOOTH_CONV3_ALPHA_050", assignment,
            ("conv3.0.conv1",), 0.5, "smooth", 8, 8, ()),
        SelectiveSmoothingCandidate(
            "SMOOTH_CONV6_ALPHA_050", assignment,
            ("conv6.0",), 0.5, "smooth", 8, 8, ()),
        SelectiveSmoothingCandidate(
            "SMOOTH_ALL_ALPHA_025", assignment, ANALYSIS_MODULES, 0.25,
            "smooth", 8, 8, ()),
        SelectiveSmoothingCandidate(
            "SMOOTH_ALL_ALPHA_050", assignment, ANALYSIS_MODULES, 0.5,
            "smooth", 8, 8, ()),
        SelectiveSmoothingCandidate(
            "SMOOTH_ALL_ALPHA_075", assignment, ANALYSIS_MODULES, 0.75,
            "smooth", 8, 8, ()),
    )


def _boundary_assignment(assignment, modules, weight_bits, activation_bits):
    formats = {6: FORMAT_FP6, 8: FORMAT_FP8}
    weights = dict(assignment.weight_formats)
    activations = dict(assignment.activation_formats)
    for name in PROTECTED_MODULES:
        weights[name] = FORMAT_FP8
        owner = ("activation::%s::input" % name, "module_input")
        activations[owner] = FORMAT_FP8
    for name in modules:
        weights[name] = formats[int(weight_bits)]
        owner = ("activation::%s::input" % name, "module_input")
        activations[owner] = formats[int(activation_bits)]
    return FPFormatAssignment(weights, activations)


def _boundary_module_bits(candidate, module_name):
    owner = ("activation::%s::input" % module_name, "module_input")
    weight_format = candidate.assignment.weight_formats[module_name]
    activation_format = candidate.assignment.activation_formats[owner]
    return FORMAT_SPECS[weight_format].bits, FORMAT_SPECS[activation_format].bits


def _boundary_candidate(name, assignment, scope, modules, weight_bits,
                        activation_bits, branch_independent_modules):
    return SelectiveSmoothingCandidate(
        name,
        _boundary_assignment(
            assignment, modules, weight_bits, activation_bits),
        (), 0.0, scope, int(weight_bits), int(activation_bits),
        tuple(branch_independent_modules))


def _boundary_candidates(assignment):
    output = [_boundary_candidate(
        "BOTH_W8A8", assignment, "both", PROTECTED_MODULES,
        8, 8, ())]
    scopes = (
        ("ID_DEC0", "id_dec0", ("id_dec0.0",)),
        ("ID_DEC1", "id_dec1", ("id_dec1.0",)),
        ("JOINT", "joint", PROTECTED_MODULES),
    )
    for prefix, scope, modules in scopes:
        for weight_bits, activation_bits in ((6, 6), (6, 8), (8, 6)):
            output.append(_boundary_candidate(
                "%s_W%dA%d" % (prefix, weight_bits, activation_bits),
                assignment, scope, modules, weight_bits, activation_bits, ()))
        output.append(_boundary_candidate(
            "%s_W8A6_BRANCH" % prefix, assignment, scope, modules,
            8, 6, modules))
    return tuple(output)


def _attribution_rows(result_rows):
    values = dict(
        (row["configuration"], float(row["pooled_rmse"]))
        for row in result_rows)
    baseline = values["BOTH_W8A8"]
    rows = []
    for prefix, scope in (
            ("ID_DEC0", "id_dec0"),
            ("ID_DEC1", "id_dec1"),
            ("JOINT", "joint")):
        w6a6 = values["%s_W6A6" % prefix]
        w6a8 = values["%s_W6A8" % prefix]
        w8a6 = values["%s_W8A6" % prefix]
        branch = values["%s_W8A6_BRANCH" % prefix]
        rows.append({
            "scope": scope,
            "w8a8_rmse": baseline,
            "weight_penalty": w6a8 - baseline,
            "activation_penalty": w8a6 - baseline,
            "interaction_penalty": w6a6 - w6a8 - w8a6 + baseline,
            "branch_scale_recovery": w8a6 - branch,
        })
    return rows


def _pareto_rows(result_rows):
    selected = []
    for row in result_rows:
        rmse = float(row["pooled_rmse"])
        weight_bits = float(row["average_weight_bits"])
        activation_bits = float(row["average_activation_bits"])
        dominated = False
        for other in result_rows:
            if other is row:
                continue
            other_rmse = float(other["pooled_rmse"])
            other_weight = float(other["average_weight_bits"])
            other_activation = float(other["average_activation_bits"])
            no_worse = other_rmse <= rmse and \
                other_weight <= weight_bits and \
                other_activation <= activation_bits
            strictly_better = other_rmse < rmse or \
                other_weight < weight_bits or \
                other_activation < activation_bits
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            selected.append(dict(row))
    return tuple(sorted(
        selected,
        key=lambda row: (
            float(row["average_weight_bits"]) +
            float(row["average_activation_bits"]),
            float(row["pooled_rmse"]), str(row["configuration"]))))


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write empty selective smoothing results")
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise ValueError("selective smoothing row schema differs")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(config, output_root):
    payload = load_protocol(Path(config).resolve())
    spec = payload["models"]["nlspn"]
    hard = payload["hard_deployment"]
    device = str(spec["device"])
    runtime = NYUModelRuntime.from_args(_runtime_args(spec, device))
    model = runtime.build_model(runtime.device)
    contract = _build_contract("nlspn", model)
    costs = _costs("nlspn", spec, model, runtime)
    evaluator = NLSPNSelectiveSmoothingEvaluator(
        runtime, model, contract,
        mixed_precision.build_registry(contract, costs),
        _settings(spec, hard, device))
    reference = _reference_metrics(evaluator)
    weight_scores, activation_scores = _read_score_tables(spec)
    _, activation_scores = ordinary_score_tables(
        contract, costs, weight_scores, activation_scores)
    scalar_scores = dict(
        (owner, float(scores[4]) - float(scores[8]))
        for owner, scores in activation_scores.items())
    assignment = _protected_fp6_assignment(
        evaluator, scalar_scores, dict(costs.activation_elements))

    output = Path(output_root)
    if output.exists():
        raise FileExistsError("selective smoothing output exists: %s" % output)
    output.mkdir(parents=True)
    result_rows = []
    diagnostic_rows = []
    for candidate in _candidates(assignment):
        metrics = _aggregate(evaluator._evaluate_candidate(candidate))
        result_rows.append({
            "configuration": candidate.name,
            "pooled_rmse": metrics["pooled_rmse"],
            "mean_sample_rmse": metrics["mean_sample_rmse"],
            "delta_vs_fp32": metrics["pooled_rmse"] -
            reference["pooled_rmse"],
            "relative_fp_loss": metrics["pooled_rmse"] /
            reference["pooled_rmse"] - 1.0,
            "valid": metrics["valid"],
            "alpha": candidate.alpha,
            "smooth_modules": ";".join(candidate.smooth_modules),
        })
        diagnostic_rows.extend(evaluator.diagnostic_rows(candidate.name))
    _write_csv(output / "summary.csv", result_rows)
    _write_csv(output / "module_diagnostics.csv", diagnostic_rows)
    artifact = {
        "model": "nlspn",
        "propagation_dtype": "fp16",
        "formats": [FORMAT_FP6, FORMAT_FP8],
        "calibration_count": int(spec["calibration_count"]),
        "evaluation_indices": list(spec["evaluation_indices"]),
        "reference": reference,
        "protected_modules": list(PROTECTED_MODULES),
        "analysis_modules": list(ANALYSIS_MODULES),
        "propagation_owned_modules": list(propagation_owned_modules(contract)),
        "results": result_rows,
        "module_diagnostics": diagnostic_rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(json_safe(artifact), indent=2, sort_keys=True,
                   allow_nan=False) + "\n", encoding="utf-8")
    evaluator.close()
    runtime.close()
    return output


def run_boundary_ablation(config, output_root):
    payload = load_protocol(Path(config).resolve())
    spec = payload["models"]["nlspn"]
    hard = payload["hard_deployment"]
    device = str(spec["device"])
    runtime = NYUModelRuntime.from_args(_runtime_args(spec, device))
    model = runtime.build_model(runtime.device)
    contract = _build_contract("nlspn", model)
    costs = _costs("nlspn", spec, model, runtime)
    evaluator = NLSPNSelectiveSmoothingEvaluator(
        runtime, model, contract,
        mixed_precision.build_registry(contract, costs),
        _settings(spec, hard, device))
    reference = _reference_metrics(evaluator)
    weight_scores, activation_scores = _read_score_tables(spec)
    _, activation_scores = ordinary_score_tables(
        contract, costs, weight_scores, activation_scores)
    scalar_scores = dict(
        (owner, float(scores[4]) - float(scores[8]))
        for owner, scores in activation_scores.items())
    assignment = _format_assignment(
        evaluator, FORMAT_FP6, FORMAT_FP6, scalar_scores,
        dict(costs.activation_elements), 0.0)

    output = Path(output_root)
    if output.exists():
        raise FileExistsError("initial-depth ablation output exists: %s" % output)
    output.mkdir(parents=True)
    result_rows = []
    boundary_rows = []
    assignments = {}
    for candidate in _boundary_candidates(assignment):
        metrics = _aggregate(evaluator._evaluate_candidate(candidate))
        weight_bits, activation_bits, weight_fractions, activation_fractions = \
            _format_budget(candidate.assignment, costs)
        result_rows.append({
            "configuration": candidate.name,
            "scope": candidate.scope,
            "pooled_rmse": metrics["pooled_rmse"],
            "mean_sample_rmse": metrics["mean_sample_rmse"],
            "delta_vs_fp32": metrics["pooled_rmse"] -
            reference["pooled_rmse"],
            "relative_fp_loss": metrics["pooled_rmse"] /
            reference["pooled_rmse"] - 1.0,
            "valid": metrics["valid"],
            "boundary_weight_bits": candidate.weight_bits,
            "boundary_activation_bits": candidate.activation_bits,
            "branch_independent_modules":
            ";".join(candidate.branch_independent_modules),
            "average_weight_bits": weight_bits,
            "average_activation_bits": activation_bits,
        })
        boundary_rows.extend(evaluator.boundary_diagnostic_rows(candidate))
        assignments[candidate.name] = {
            "weight_formats": dict(candidate.assignment.weight_formats),
            "activation_formats": {
                "%s::%s" % owner: format_name
                for owner, format_name in
                candidate.assignment.activation_formats.items()
            },
            "weight_format_fractions": weight_fractions,
            "activation_format_fractions": activation_fractions,
        }
    attribution = _attribution_rows(result_rows)
    pareto = _pareto_rows(result_rows)
    _write_csv(output / "summary.csv", result_rows)
    _write_csv(output / "attribution.csv", attribution)
    _write_csv(output / "boundary_diagnostics.csv", boundary_rows)
    _write_csv(output / "pareto.csv", pareto)
    artifact = {
        "model": "nlspn",
        "experiment": "initial_depth_boundary_factorial",
        "propagation_dtype": "fp16",
        "ordinary_format": FORMAT_FP6,
        "boundary_formats": [FORMAT_FP6, FORMAT_FP8],
        "calibration_count": int(spec["calibration_count"]),
        "evaluation_indices": list(spec["evaluation_indices"]),
        "reference": reference,
        "protected_modules": list(PROTECTED_MODULES),
        "propagation_owned_modules": list(propagation_owned_modules(contract)),
        "results": result_rows,
        "attribution": attribution,
        "boundary_diagnostics": boundary_rows,
        "pareto": pareto,
        "assignments": assignments,
    }
    (output / "manifest.json").write_text(
        json.dumps(json_safe(artifact), indent=2, sort_keys=True,
                   allow_nan=False) + "\n", encoding="utf-8")
    evaluator.close()
    runtime.close()
    return output


def run_activation_protection(config, output_root):
    payload = load_protocol(Path(config).resolve())
    spec = payload["models"]["nlspn"]
    hard = payload["hard_deployment"]
    device = str(spec["device"])
    runtime = NYUModelRuntime.from_args(_runtime_args(spec, device))
    model = runtime.build_model(runtime.device)
    contract = _build_contract("nlspn", model)
    costs = _costs("nlspn", spec, model, runtime)
    evaluator = NLSPNActivationProtectionEvaluator(
        runtime, model, contract,
        mixed_precision.build_registry(contract, costs),
        _settings(spec, hard, device))

    output = Path(output_root)
    if output.exists():
        raise FileExistsError(
            "activation protection output exists: %s" % output)
    output.mkdir(parents=True)

    weight_scores, activation_scores = _read_score_tables(spec)
    _, activation_scores = ordinary_score_tables(
        contract, costs, weight_scores, activation_scores)
    scalar_scores = dict(
        (owner, float(scores[4]) - float(scores[8]))
        for owner, scores in activation_scores.items())
    assignment = _format_assignment(
        evaluator, FORMAT_FP6, FORMAT_FP6, scalar_scores,
        dict(costs.activation_elements), 0.0)
    base_assignment = _activation_protection_base_assignment(assignment)
    candidates = _activation_protection_candidates(base_assignment)

    reference_capture, reference_ground_truth, reference_rows, reference = \
        _reference_activation_protection(evaluator)
    sample_rows = list(reference_rows)
    signal_rows = []
    state_rows = []
    module_rows = []
    branch_rows = []
    effective_weight_rows = []
    assignments = {}
    measurements = []
    for candidate in candidates:
        assignment_values = _assignment_payload(candidate, costs)
        if abs(float(assignment_values["average_weight_bits"]) - 6.0) > 1e-12:
            raise RuntimeError("activation protection weight budget differs")
        result = _evaluate_activation_protection_candidate(
            evaluator, candidate, reference_capture, reference_ground_truth)
        measurements.append((candidate, assignment_values, result["metrics"]))
        sample_rows.extend(result["depth_rows"])
        signal_rows.extend(result["signal_rows"])
        state_rows.extend(result["state_rows"])
        module_rows.extend(result["module_rows"])
        branch_rows.extend(result["branch_rows"])
        effective_weight_rows.extend(result["effective_weight_rows"])
        assignments[candidate.name] = assignment_values

    for candidate in candidates:
        for name in INITIAL_DEPTH_MODULES:
            rows = tuple(
                row for row in effective_weight_rows
                if row["configuration"] == candidate.name and
                row["module"] == name)
            if len(rows) != 1 or rows[0]["format"] != FORMAT_FP6 or \
                    int(rows[0]["changed_weight_elements"]) <= 0:
                raise RuntimeError(
                    "initial-depth effective FP6 weight check failed")

    base = measurements[0][2]
    result_rows = []
    for candidate, assignment_values, metrics in measurements:
        result_rows.append({
            "model": "nlspn",
            "configuration": candidate.name,
            "pooled_rmse": metrics["pooled_rmse"],
            "mean_sample_rmse": metrics["mean_sample_rmse"],
            "pooled_mae": metrics["pooled_mae"],
            "pooled_absrel": metrics["pooled_absrel"],
            "pooled_irmse": metrics["pooled_irmse"],
            "delta_vs_fp32": metrics["pooled_rmse"] -
            reference["pooled_rmse"],
            "relative_fp_loss": metrics["pooled_rmse"] /
            reference["pooled_rmse"] - 1.0,
            "delta_vs_base": metrics["pooled_rmse"] -
            base["pooled_rmse"],
            "relative_base_loss": metrics["pooled_rmse"] /
            base["pooled_rmse"] - 1.0,
            "valid": True,
            "reproducible": True,
            "propagation_dtype": "fp16",
            "average_weight_bits":
            assignment_values["average_weight_bits"],
            "average_activation_bits":
            assignment_values["average_activation_bits"],
        })
    selected = _select_activation_protection_result(result_rows)
    pareto = _pareto_rows(result_rows)

    _write_csv(output / "summary.csv", result_rows)
    _write_csv(output / "sample_metrics.csv", sample_rows)
    _write_csv(output / "module_diagnostics.csv", module_rows)
    _write_csv(output / "branch_diagnostics.csv", branch_rows)
    _write_csv(
        output / "effective_weight_metrics.csv", effective_weight_rows)
    _write_csv(output / "propagation_signal_metrics.csv", signal_rows)
    _write_csv(output / "propagation_state_metrics.csv", state_rows)
    _write_csv(output / "pareto.csv", pareto)

    calibration_metadata = json.loads(
        Path(spec["calibration_metadata"]).read_text(encoding="utf-8"))
    if Path(calibration_metadata["checkpoint"]).resolve() != \
            Path(spec["checkpoint"]).resolve():
        raise RuntimeError("calibration checkpoint identity differs")
    artifact = {
        "model": "nlspn",
        "experiment": "fp6_activation_protection",
        "device": device,
        "checkpoint": str(Path(spec["checkpoint"]).resolve()),
        "checkpoint_sha256": calibration_metadata["checkpoint_sha256"],
        "checkpoint_architecture": spec["checkpoint_architecture"],
        "expected_architecture_class": spec["expected_architecture_class"],
        "calibration_count": int(spec["calibration_count"]),
        "calibration_indices": list(evaluator.calibration_indices),
        "evaluation_indices": list(spec["evaluation_indices"]),
        "propagation_dtype": "fp16",
        "propagation_iterations": int(spec["propagation_iterations"]),
        "propagation_owned_modules": list(
            propagation_owned_modules(contract)),
        "ordinary_weight_format": FORMAT_FP6,
        "ordinary_activation_format": FORMAT_FP6,
        "initial_depth_activation_format": FORMAT_FP8,
        "formats": dict(
            (name, {"bits": FORMAT_SPECS[name].bits,
                    "maximum": FORMAT_SPECS[name].maximum})
            for name in (FORMAT_FP6, FORMAT_FP8)),
        "branch_layouts": dict(
            (name, list(channels)) for name, channels in
            BRANCH_LAYOUTS.items()),
        "single_qdq_ownership": {
            "concat_adapter": "disabled",
            "ordinary_instrumentor": "weight_and_input",
            "direct_initial_depth_output_qdq": False,
            "paired_forward_calls": 2,
        },
        "reference": reference,
        "results": result_rows,
        "selected_configuration": selected["configuration"],
        "selected_improves_base":
        float(selected["pooled_rmse"]) < float(base["pooled_rmse"]),
        "pareto": pareto,
        "assignments": assignments,
        "artifacts": (
            "summary.csv", "sample_metrics.csv",
            "module_diagnostics.csv", "branch_diagnostics.csv",
            "effective_weight_metrics.csv",
            "propagation_signal_metrics.csv",
            "propagation_state_metrics.csv", "pareto.csv"),
    }
    (output / "manifest.json").write_text(
        json.dumps(json_safe(artifact), indent=2, sort_keys=True,
                   allow_nan=False) + "\n", encoding="utf-8")
    evaluator.close()
    runtime.close()
    return output


def run_early_fp16_attribution(config, output_root):
    payload = load_protocol(Path(config).resolve())
    spec = payload["models"]["nlspn"]
    hard = payload["hard_deployment"]
    device = str(spec["device"])
    runtime = NYUModelRuntime.from_args(_runtime_args(spec, device))
    model = runtime.build_model(runtime.device)
    contract = _build_contract("nlspn", model)
    costs = _costs("nlspn", spec, model, runtime)
    evaluator = NLSPNActivationProtectionEvaluator(
        runtime, model, contract,
        mixed_precision.build_registry(contract, costs),
        _settings(spec, hard, device))

    output = Path(output_root)
    if output.exists():
        raise FileExistsError("early FP16 output exists: %s" % output)
    output.mkdir(parents=True)

    weight_scores, activation_scores = _read_score_tables(spec)
    _, activation_scores = ordinary_score_tables(
        contract, costs, weight_scores, activation_scores)
    scalar_scores = dict(
        (owner, float(scores[4]) - float(scores[8]))
        for owner, scores in activation_scores.items())
    assignment = _format_assignment(
        evaluator, FORMAT_FP6, FORMAT_FP6, scalar_scores,
        dict(costs.activation_elements), 0.0)
    candidates = _early_fp16_candidates(
        _activation_protection_base_assignment(assignment))

    reference_capture, reference_ground_truth, reference_rows, reference = \
        _reference_activation_protection(evaluator)
    sample_rows = list(reference_rows)
    signal_rows = []
    state_rows = []
    module_rows = []
    effective_weight_rows = []
    assignments = {}
    measurements = []
    for candidate in candidates:
        assignment_values = _assignment_payload(candidate, costs)
        result = _evaluate_activation_protection_candidate(
            evaluator, candidate, reference_capture, reference_ground_truth)
        if result["branch_rows"]:
            raise RuntimeError("early FP16 attribution enabled branch QDQ")
        measurements.append((candidate, assignment_values, result["metrics"]))
        sample_rows.extend(result["depth_rows"])
        signal_rows.extend(result["signal_rows"])
        state_rows.extend(result["state_rows"])
        module_rows.extend(result["module_rows"])
        effective_weight_rows.extend(result["effective_weight_rows"])
        assignments[candidate.name] = assignment_values

    for candidate in candidates:
        for name in EARLY_MODULES:
            rows = tuple(
                row for row in effective_weight_rows
                if row["configuration"] == candidate.name and
                row["module"] == name)
            expected = candidate.assignment.weight_formats[name]
            if len(rows) != 1 or rows[0]["format"] != expected or \
                    int(rows[0]["changed_weight_elements"]) <= 0:
                raise RuntimeError("early effective weight check failed")

    baseline = measurements[0][2]
    result_rows = []
    for candidate, assignment_values, metrics in measurements:
        result_rows.append({
            "model": "nlspn",
            "configuration": candidate.name,
            "pooled_rmse": metrics["pooled_rmse"],
            "mean_sample_rmse": metrics["mean_sample_rmse"],
            "pooled_mae": metrics["pooled_mae"],
            "pooled_absrel": metrics["pooled_absrel"],
            "pooled_irmse": metrics["pooled_irmse"],
            "delta_vs_fp32": metrics["pooled_rmse"] -
            reference["pooled_rmse"],
            "relative_fp_loss": metrics["pooled_rmse"] /
            reference["pooled_rmse"] - 1.0,
            "delta_vs_early_w6a8": metrics["pooled_rmse"] -
            baseline["pooled_rmse"],
            "relative_early_w6a8": metrics["pooled_rmse"] /
            baseline["pooled_rmse"] - 1.0,
            "valid": True,
            "reproducible": True,
            "propagation_dtype": "fp16",
            "average_weight_bits":
            assignment_values["average_weight_bits"],
            "average_activation_bits":
            assignment_values["average_activation_bits"],
        })
    attribution = _early_fp16_attribution_rows(
        result_rows, reference["pooled_rmse"])

    _write_csv(output / "summary.csv", result_rows)
    _write_csv(output / "sample_metrics.csv", sample_rows)
    _write_csv(output / "module_diagnostics.csv", module_rows)
    _write_csv(
        output / "effective_weight_metrics.csv", effective_weight_rows)
    _write_csv(output / "propagation_signal_metrics.csv", signal_rows)
    _write_csv(output / "propagation_state_metrics.csv", state_rows)
    _write_csv(output / "attribution.csv", attribution)

    calibration_metadata = json.loads(
        Path(spec["calibration_metadata"]).read_text(encoding="utf-8"))
    if Path(calibration_metadata["checkpoint"]).resolve() != \
            Path(spec["checkpoint"]).resolve():
        raise RuntimeError("calibration checkpoint identity differs")
    artifact = {
        "model": "nlspn",
        "experiment": "early_fp16_error_attribution",
        "device": device,
        "checkpoint": str(Path(spec["checkpoint"]).resolve()),
        "checkpoint_sha256": calibration_metadata["checkpoint_sha256"],
        "checkpoint_architecture": spec["checkpoint_architecture"],
        "calibration_count": int(spec["calibration_count"]),
        "calibration_indices": list(evaluator.calibration_indices),
        "evaluation_indices": list(spec["evaluation_indices"]),
        "propagation_dtype": "fp16",
        "propagation_iterations": int(spec["propagation_iterations"]),
        "propagation_owned_modules": list(
            propagation_owned_modules(contract)),
        "early_modules": list(EARLY_MODULES),
        "formats": dict(
            (name, {"bits": FORMAT_SPECS[name].bits,
                    "maximum": FORMAT_SPECS[name].maximum})
            for name in (FORMAT_FP6, FORMAT_FP8, FORMAT_FP16)),
        "fp16_qdq": "FP32_to_IEEE_FP16_to_FP32",
        "single_qdq_ownership": {
            "concat_adapter": "disabled",
            "ordinary_instrumentor": "weight_and_input",
            "direct_initial_depth_output_qdq": False,
            "paired_forward_calls": 2,
        },
        "reference": reference,
        "results": result_rows,
        "attribution": attribution,
        "assignments": assignments,
        "artifacts": (
            "summary.csv", "sample_metrics.csv",
            "module_diagnostics.csv", "effective_weight_metrics.csv",
            "propagation_signal_metrics.csv",
            "propagation_state_metrics.csv", "attribution.csv"),
    }
    (output / "manifest.json").write_text(
        json.dumps(json_safe(artifact), indent=2, sort_keys=True,
                   allow_nan=False) + "\n", encoding="utf-8")
    evaluator.close()
    runtime.close()
    return output


def run_decoder_attribution(config, output_root):
    payload = load_protocol(Path(config).resolve())
    spec = payload["models"]["nlspn"]
    hard = payload["hard_deployment"]
    device = str(spec["device"])
    runtime = NYUModelRuntime.from_args(_runtime_args(spec, device))
    model = runtime.build_model(runtime.device)
    contract = _build_contract("nlspn", model)
    costs = _costs("nlspn", spec, model, runtime)
    evaluator = NLSPNActivationProtectionEvaluator(
        runtime, model, contract,
        mixed_precision.build_registry(contract, costs),
        _settings(spec, hard, device))

    output = Path(output_root)
    if output.exists():
        raise FileExistsError("decoder attribution output exists: %s" % output)
    output.mkdir(parents=True)

    weight_scores, activation_scores = _read_score_tables(spec)
    _, activation_scores = ordinary_score_tables(
        contract, costs, weight_scores, activation_scores)
    scalar_scores = dict(
        (owner, float(scores[4]) - float(scores[8]))
        for owner, scores in activation_scores.items())
    assignment = _format_assignment(
        evaluator, FORMAT_FP6, FORMAT_FP6, scalar_scores,
        dict(costs.activation_elements), 0.0)
    candidates = _decoder_attribution_candidates(
        _activation_protection_base_assignment(assignment))

    reference_capture, reference_ground_truth, reference_rows, reference = \
        _reference_activation_protection(evaluator)
    sample_rows = list(reference_rows)
    signal_rows = []
    state_rows = []
    module_rows = []
    effective_weight_rows = []
    assignments = {}
    measurements = []
    for candidate in candidates:
        assignment_values = _assignment_payload(candidate, costs)
        result = _evaluate_activation_protection_candidate(
            evaluator, candidate, reference_capture, reference_ground_truth)
        if result["branch_rows"]:
            raise RuntimeError("decoder attribution enabled branch QDQ")
        measurements.append((candidate, assignment_values, result["metrics"]))
        sample_rows.extend(result["depth_rows"])
        signal_rows.extend(result["signal_rows"])
        state_rows.extend(result["state_rows"])
        module_rows.extend(result["module_rows"])
        effective_weight_rows.extend(result["effective_weight_rows"])
        assignments[candidate.name] = assignment_values
        print("%s pooled RMSE: %.9f" % (
            candidate.name, result["metrics"]["pooled_rmse"]), flush=True)

    target_modules = EARLY_MODULES + tuple(
        module
        for _, modules in DECODER_ATTRIBUTION_GROUPS
        for module in modules)
    for candidate in candidates:
        for name in target_modules:
            rows = tuple(
                row for row in effective_weight_rows
                if row["configuration"] == candidate.name and
                row["module"] == name)
            expected = candidate.assignment.weight_formats[name]
            if len(rows) != 1 or rows[0]["format"] != expected or \
                    int(rows[0]["changed_weight_elements"]) <= 0:
                raise RuntimeError(
                    "decoder attribution effective weight check failed")

    baseline = measurements[0][2]
    result_rows = []
    for candidate, assignment_values, metrics in measurements:
        result_rows.append({
            "model": "nlspn",
            "configuration": candidate.name,
            "pooled_rmse": metrics["pooled_rmse"],
            "mean_sample_rmse": metrics["mean_sample_rmse"],
            "pooled_mae": metrics["pooled_mae"],
            "pooled_absrel": metrics["pooled_absrel"],
            "pooled_irmse": metrics["pooled_irmse"],
            "delta_vs_fp32": metrics["pooled_rmse"] -
            reference["pooled_rmse"],
            "relative_fp_loss": metrics["pooled_rmse"] /
            reference["pooled_rmse"] - 1.0,
            "delta_vs_early_w16a16": metrics["pooled_rmse"] -
            baseline["pooled_rmse"],
            "relative_early_w16a16": metrics["pooled_rmse"] /
            baseline["pooled_rmse"] - 1.0,
            "valid": True,
            "reproducible": True,
            "propagation_dtype": "fp16",
            "average_weight_bits":
            assignment_values["average_weight_bits"],
            "average_activation_bits":
            assignment_values["average_activation_bits"],
        })
    isolated = _decoder_isolated_attribution_rows(result_rows)
    cumulative = _decoder_cumulative_attribution_rows(result_rows)

    _write_csv(output / "summary.csv", result_rows)
    _write_csv(output / "sample_metrics.csv", sample_rows)
    _write_csv(output / "module_diagnostics.csv", module_rows)
    _write_csv(
        output / "effective_weight_metrics.csv", effective_weight_rows)
    _write_csv(output / "propagation_signal_metrics.csv", signal_rows)
    _write_csv(output / "propagation_state_metrics.csv", state_rows)
    _write_csv(output / "isolated_attribution.csv", isolated)
    _write_csv(output / "cumulative_attribution.csv", cumulative)

    calibration_metadata = json.loads(
        Path(spec["calibration_metadata"]).read_text(encoding="utf-8"))
    if Path(calibration_metadata["checkpoint"]).resolve() != \
            Path(spec["checkpoint"]).resolve():
        raise RuntimeError("calibration checkpoint identity differs")
    artifact = {
        "model": "nlspn",
        "experiment": "decoder_guidance_initial_depth_fp16_attribution",
        "device": device,
        "checkpoint": str(Path(spec["checkpoint"]).resolve()),
        "checkpoint_sha256": calibration_metadata["checkpoint_sha256"],
        "checkpoint_architecture": spec["checkpoint_architecture"],
        "calibration_count": int(spec["calibration_count"]),
        "calibration_indices": list(evaluator.calibration_indices),
        "evaluation_indices": list(spec["evaluation_indices"]),
        "propagation_dtype": "fp16",
        "propagation_iterations": int(spec["propagation_iterations"]),
        "propagation_owned_modules": list(
            propagation_owned_modules(contract)),
        "fixed_early_modules": list(EARLY_MODULES),
        "attribution_groups": dict(
            (name, list(modules)) for name, modules in
            DECODER_ATTRIBUTION_GROUPS),
        "formats": dict(
            (name, {"bits": FORMAT_SPECS[name].bits,
                    "maximum": FORMAT_SPECS[name].maximum})
            for name in (FORMAT_FP6, FORMAT_FP8, FORMAT_FP16)),
        "fp16_qdq": "FP32_to_IEEE_FP16_to_FP32",
        "single_qdq_ownership": {
            "concat_adapter": "disabled",
            "ordinary_instrumentor": "weight_and_input",
            "direct_initial_depth_output_qdq": False,
            "paired_forward_calls": 2,
        },
        "reference": reference,
        "results": result_rows,
        "isolated_attribution": isolated,
        "cumulative_attribution": cumulative,
        "assignments": assignments,
        "artifacts": DECODER_ATTRIBUTION_ARTIFACTS,
    }
    (output / "manifest.json").write_text(
        json.dumps(json_safe(artifact), indent=2, sort_keys=True,
                   allow_nan=False) + "\n", encoding="utf-8")
    evaluator.close()
    runtime.close()
    return output


def _experiment_runners():
    return {
        "smoothing": run,
        "initial-depth-boundary": run_boundary_ablation,
        "activation-protection": run_activation_protection,
        "early-fp16-attribution": run_early_fp16_attribution,
        "decoder-attribution": run_decoder_attribution,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--experiment", required=True,
        choices=("smoothing", "initial-depth-boundary",
                 "activation-protection", "early-fp16-attribution",
                 "decoder-attribution"))
    args = parser.parse_args(tuple(sys.argv[1:]))
    print(_experiment_runners()[args.experiment](
        args.config, args.output_root))


if __name__ == "__main__":
    main()
