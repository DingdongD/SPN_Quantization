#!/usr/bin/env python3
"""Diagnose and optimize activation resolution for strict CSPN W4A4."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train_nyu_iteration_sweep as sweep  # noqa: E402
from scripts.export_nyu_predictions import (  # noqa: E402
    load_model_state,
    load_run_args,
    prepare_args,
)
from scripts.hardware_aligned_quantization import (  # noqa: E402
    HardwareAlignedInstrumentor,
    prepare_hardware_model,
)
from scripts.nyu_quantization_analysis import (  # noqa: E402
    classify_module,
    regional_depth_metrics,
)
from scripts.run_nyu_rtn_quantization import (  # noqa: E402
    aggregate_region_rows,
    batch_from_sample,
    calibration_dataset,
    evaluation_dataset,
    load_sample_indices,
    prediction_payload,
    prepare_prediction_dir,
    seeded_sample,
    write_csv,
    write_json,
    write_prediction_payload,
)
from spn_quant.activation_resolution import (  # noqa: E402
    ActivationResolutionRecorder,
)
from spn_quant.adapters import install_model_semantic_adapter  # noqa: E402
from spn_quant.adapters.cspn import CSPNStructuralMergeAdapter  # noqa: E402
from spn_quant.propagation import (  # noqa: E402
    PropagationQuantConfig,
    install_propagation_adapter,
)
from spn_quant.specs import QuantSpec  # noqa: E402
from spn_quant.runtime import EdgeQDQRuntime  # noqa: E402
from spn_quant.rotation import CSPNRotationController  # noqa: E402


PROPAGATION_A8_Q13 = {
    "affinity_bits": 8,
    "confidence_bits": 8,
    "offset_bits": 8,
    "state_bits": 8,
    "coefficient_fraction_bits": 13,
}

CALIBRATION_SAMPLES = 128
EVALUATION_SAMPLES = 64
ORDINARY_GROUPS = frozenset(("encoder", "decoder", "depth_head"))
GROUP_SIZES = (128, 64, 32, 16, 8, 1)
SCALE_FACTORS = (1.0, 0.95, 0.9, 0.85, 0.75, 0.625, 0.5)
EXPECTED_EVALUATION_CONFIGS = (
    "FP32", "PA_ONLY", "W4_ONLY", "A4_ONLY", "W4A4_RTN",
    "W4A4_HYBRID_GROUP128", "W4A4_HYBRID_GROUP64",
    "W4A4_HYBRID_GROUP32", "W4A4_HYBRID_GROUP16",
    "W4A4_HYBRID_GROUP8", "W4A4_CHANNEL",
    "W4A4_SELECTIVE_W4A4_CHANNEL", "W4A4_MERGE_SHARED",
    "W4A4_RESIDUAL", "W4A4_CALIBRATED_SCALE",
)
EXPECTED_PREDICTION_CONFIGS = (
    "FP32", "W4A4_RTN", "W4A4_CHANNEL", "W4A4_CALIBRATED_SCALE",
)

STRICT_ACTIVATION_OWNERS = frozenset((
    ("conv1_1", "input"), ("conv2", "input"),
    ("gud_up_proj_layer1.conv2", "input"),
    ("gud_up_proj_layer1.relu#0", "relu_output"),
    ("gud_up_proj_layer1.relu#1", "relu_output"),
    ("gud_up_proj_layer1.sc_conv1", "output"),
    ("gud_up_proj_layer2.conv1", "input"),
    ("gud_up_proj_layer2.conv1_1", "input"),
    ("gud_up_proj_layer2.conv2", "input"),
    ("gud_up_proj_layer2.relu#0", "relu_output"),
    ("gud_up_proj_layer2.relu#1", "relu_output"),
    ("gud_up_proj_layer2.relu#2", "relu_output"),
    ("gud_up_proj_layer2.sc_conv1", "input"),
    ("gud_up_proj_layer2.sc_conv1", "output"),
    ("gud_up_proj_layer3.conv1", "input"),
    ("gud_up_proj_layer3.conv1_1", "input"),
    ("gud_up_proj_layer3.conv2", "input"),
    ("gud_up_proj_layer3.relu#0", "relu_output"),
    ("gud_up_proj_layer3.relu#1", "relu_output"),
    ("gud_up_proj_layer3.relu#2", "relu_output"),
    ("gud_up_proj_layer3.sc_conv1", "input"),
    ("gud_up_proj_layer3.sc_conv1", "output"),
    ("gud_up_proj_layer4.conv1", "input"),
    ("gud_up_proj_layer4.conv2", "input"),
    ("gud_up_proj_layer4.relu#0", "relu_output"),
    ("gud_up_proj_layer4.relu#1", "relu_output"),
    ("gud_up_proj_layer4.relu#2", "relu_output"),
    ("gud_up_proj_layer4.sc_conv1", "input"),
    ("gud_up_proj_layer4.sc_conv1", "output"),
    ("gud_up_proj_layer5.conv1", "input"),
    ("layer1.0.conv1", "input"), ("layer1.0.conv2", "input"),
    ("layer1.0.relu#0", "relu_output"),
    ("layer1.0.relu#1", "relu_output"),
    ("layer1.1.conv1", "input"), ("layer1.1.conv2", "input"),
    ("layer1.1.relu#0", "relu_output"),
    ("layer1.1.relu#1", "relu_output"),
    ("layer2.0.conv1", "input"), ("layer2.0.conv2", "input"),
    ("layer2.0.downsample.0", "input"),
    ("layer2.0.downsample.0", "output"),
    ("layer2.0.relu#0", "relu_output"),
    ("layer2.0.relu#1", "relu_output"),
    ("layer2.1.conv1", "input"), ("layer2.1.conv2", "input"),
    ("layer2.1.relu#0", "relu_output"),
    ("layer2.1.relu#1", "relu_output"),
    ("layer3.0.conv1", "input"), ("layer3.0.conv2", "input"),
    ("layer3.0.downsample.0", "input"),
    ("layer3.0.downsample.0", "output"),
    ("layer3.0.relu#0", "relu_output"),
    ("layer3.0.relu#1", "relu_output"),
    ("layer3.1.conv1", "input"), ("layer3.1.conv2", "input"),
    ("layer3.1.relu#0", "relu_output"),
    ("layer3.1.relu#1", "relu_output"),
    ("layer4.0.conv1", "input"), ("layer4.0.conv2", "input"),
    ("layer4.0.downsample.0", "input"),
    ("layer4.0.downsample.0", "output"),
    ("layer4.0.relu#0", "relu_output"),
    ("layer4.0.relu#1", "relu_output"),
    ("layer4.1.conv1", "input"), ("layer4.1.conv2", "input"),
    ("layer4.1.relu#0", "relu_output"),
    ("layer4.1.relu#1", "relu_output"),
    ("relu#0", "relu_output"),
))
SAMPLE_FIELDS = (
    "model", "config", "sample_index", "RMSE", "MAE", "ABS_REL",
    "IRMSE", "flat_RMSE", "boundary_RMSE", "nonfinite_ratio",
)


@dataclass(frozen=True)
class BlockSite:
    name: str
    module: str


CSPN_BLOCK_SITES = (
    BlockSite("encoder_stem", "conv1_1"),
    BlockSite("encoder_layer1", "layer1"),
    BlockSite("encoder_layer2", "layer2"),
    BlockSite("encoder_layer3", "layer3"),
    BlockSite("encoder_layer4", "layer4"),
    BlockSite("decoder_layer1", "gud_up_proj_layer1"),
    BlockSite("decoder_layer2", "gud_up_proj_layer2"),
    BlockSite("decoder_layer3", "gud_up_proj_layer3"),
    BlockSite("decoder_layer4", "gud_up_proj_layer4"),
    BlockSite("initial_depth", "gud_up_proj_layer5"),
    BlockSite("propagation", "post_process_layer"),
)


def _configuration(name: str, weight_groups, activation_groups,
                   propagation, granularity: str = "tensor",
                   group_size: Optional[int] = None,
                   promoted_owners=(), selected_owners=(),
                   scale_factors=(), merge_policy: str = "none",
                   smooth_groups=(), smooth_alpha=None,
                   activation_range_overrides=(),
                   rotation_range_overrides=(),
                   activation_permutations=(),
                   activation_isolations=(), weight_bit_overrides=(),
                   activation_bit_overrides=(),
                   dynamic: bool = False
                   ) -> Dict[str, object]:
    if merge_policy not in ("none", "shared", "residual"):
        raise ValueError("unknown CSPN merge policy: %s" % merge_policy)
    smooth_groups = set(smooth_groups)
    if bool(smooth_groups) != (smooth_alpha is not None):
        raise ValueError(
            "SmoothQuant groups and alpha must be declared together")
    return {
        "name": name,
        "w_bits": 4,
        "a_bits": 4,
        "weight_groups": set(weight_groups),
        "activation_groups": set(activation_groups),
        "propagation": None if propagation is None else dict(propagation),
        "granularity": granularity,
        "group_size": group_size,
        "promoted_owners": tuple(promoted_owners),
        "selected_owners": tuple(selected_owners),
        "scale_factors": tuple(scale_factors),
        "merge_policy": merge_policy,
        "smooth_groups": smooth_groups,
        "smooth_alpha": smooth_alpha,
        "activation_range_overrides": tuple(activation_range_overrides),
        "rotation_range_overrides": tuple(rotation_range_overrides),
        "activation_permutations": tuple(activation_permutations),
        "activation_isolations": tuple(activation_isolations),
        "weight_bit_overrides": tuple(weight_bit_overrides),
        "activation_bit_overrides": tuple(activation_bit_overrides),
        "dynamic": bool(dynamic),
        "quantize_bias": False,
    }


def build_attribution_configurations() -> Tuple[Dict[str, object], ...]:
    return (
        _configuration("FP32", set(), set(), None),
        _configuration("PA_ONLY", set(), set(), PROPAGATION_A8_Q13),
        _configuration(
            "W4_ONLY", ORDINARY_GROUPS, set(), PROPAGATION_A8_Q13),
        _configuration(
            "A4_ONLY", set(), ORDINARY_GROUPS, PROPAGATION_A8_Q13),
        _configuration(
            "W4A4_RTN", ORDINARY_GROUPS, ORDINARY_GROUPS,
            PROPAGATION_A8_Q13),
    )


def build_group_configurations() -> Tuple[Dict[str, object], ...]:
    rows = []
    for group_size in GROUP_SIZES:
        if group_size == 1:
            name = "W4A4_CHANNEL"
            granularity = "channel"
        else:
            name = "W4A4_HYBRID_GROUP%d" % group_size
            granularity = "hybrid_group_tensor"
        rows.append(_configuration(
            name, ORDINARY_GROUPS, ORDINARY_GROUPS,
            PROPAGATION_A8_Q13, granularity, group_size))
    return tuple(rows)


def attribution_interaction(values: Dict[str, float]) -> float:
    return (
        float(values["W4A4_RTN"])
        - float(values["W4_ONLY"])
        - float(values["A4_ONLY"])
        + float(values["PA_ONLY"])
    )


def attribution_components(values: Dict[str, float]) -> Dict[str, float]:
    return {
        "propagation": float(values["PA_ONLY"]) - float(values["FP32"]),
        "weight": float(values["W4_ONLY"]) - float(values["PA_ONLY"]),
        "activation": float(values["A4_ONLY"]) - float(values["PA_ONLY"]),
        "interaction": attribution_interaction(values),
    }


def validate_sample_coverage(rows: Sequence[Dict[str, object]],
                             configurations: Sequence[str],
                             indices: Sequence[int]) -> None:
    expected = tuple(int(index) for index in indices)
    if len(expected) != len(set(expected)):
        raise ValueError("evaluation indices contain duplicates")
    expected_set = set(expected)
    for config in configurations:
        selected = [row for row in rows if row["config"] == config]
        actual = [int(row["sample_index"]) for row in selected]
        if len(actual) != len(expected) or set(actual) != expected_set:
            raise ValueError("sample coverage mismatch: %s" % config)


def validate_evaluation_contract(configs, prediction_configs) -> None:
    names = tuple(str(config["name"]) for config in configs)
    if names != EXPECTED_EVALUATION_CONFIGS:
        raise ValueError("evaluation configuration contract mismatch")
    if set(prediction_configs) != set(EXPECTED_PREDICTION_CONFIGS):
        raise ValueError("prediction configuration contract mismatch")


def validate_prediction_coverage(model_output: Path,
                                 indices: Sequence[int]) -> None:
    expected = {
        "sample_%05d.npz" % int(index) for index in indices
    }
    prediction_root = model_output / "predictions"
    directories = {
        path.name for path in prediction_root.iterdir() if path.is_dir()
    }
    if directories != set(EXPECTED_PREDICTION_CONFIGS):
        raise ValueError("prediction directories do not match contract")
    for config in EXPECTED_PREDICTION_CONFIGS:
        directory = prediction_root / config
        actual = {path.name for path in directory.glob("sample_*.npz")}
        if actual != expected:
            raise ValueError("prediction coverage mismatch: %s" % config)


def select_candidate_sites(rows: Sequence[Dict[str, object]],
                           limit: int) -> Tuple[str, ...]:
    limit = int(limit)
    if limit <= 0:
        raise ValueError("candidate site limit must be positive")
    calibration = [row for row in rows if row["split"] == "calibration"]
    ranked = sorted(
        calibration,
        key=lambda row: (
            -float(row["zero_collapse_error_energy"]),
            str(row["site"]),
        ))
    return tuple(str(row["site"]) for row in ranked[:limit])


def select_calibration_configuration(
        rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    calibration = [row for row in rows if row["split"] == "calibration"]
    if not calibration:
        raise ValueError("configuration selection requires calibration rows")
    return min(
        calibration,
        key=lambda row: (
            float(row["block_output_mse"]),
            -float(row["block_output_sqnr"]),
            str(row["config"]),
        ))


def select_activation_scale(
        rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    eligible = [
        row for row in rows
        if row["split"] == "calibration"
        and float(row["clipping_error_ratio"]) <= 0.5
    ]
    if not eligible:
        raise ValueError("scale selection requires eligible calibration rows")
    return min(
        eligible,
        key=lambda row: (
            float(row["block_output_mse"]),
            -float(row["factor"]),
        ))


def select_transferred_configuration(rows, base_name, candidate_name):
    by_name = dict((str(row["config"]), row) for row in rows)
    base = by_name[base_name]
    candidate = by_name[candidate_name]
    if float(candidate["RMSE"]) < float(base["RMSE"]):
        return candidate
    return base


def prediction_configuration_names(
        global_name, final_name, scale_base_name, calibrated_scale_names):
    return {
        "FP32", "W4A4_RTN", str(global_name), str(final_name),
        str(scale_base_name),
    } | set(str(name) for name in calibrated_scale_names)


def decoder_merge_sites(owners) -> Tuple[str, ...]:
    decoder_blocks = {
        "gud_up_proj_layer1",
        "gud_up_proj_layer2",
        "gud_up_proj_layer3",
        "gud_up_proj_layer4",
    }
    sites = []
    for module, _kind in owners:
        block = str(module).split(".", 1)[0]
        site = "%s::add#0" % block
        if block in decoder_blocks and site not in sites:
            sites.append(site)
    return tuple(sites)


def owner_block(owner):
    module = str(owner[0])
    if module.startswith("rotation.layer4_signed_skip"):
        return "decoder_layer4"
    if module.startswith("rotation.decoder_entry"):
        return "decoder_layer1"
    if module.startswith("gud_up_proj_layer"):
        index = module[len("gud_up_proj_layer")]
        return "initial_depth" if index == "5" else "decoder_layer%s" % index
    if module.startswith("layer"):
        return "encoder_layer%s" % module[len("layer")]
    if module == "conv1_1":
        return "encoder_stem"
    if module == "relu#0":
        return "encoder_layer1"
    if module == "conv2":
        return "decoder_layer1"
    raise ValueError("activation owner has no CSPN block: %s" % (owner,))


def build_merge_configurations(base):
    return (
        _derived_configuration(
            "W4A4_MERGE_SHARED", base, scale_factors=(),
            merge_policy="shared"),
        _derived_configuration(
            "W4A4_RESIDUAL", base, scale_factors=(),
            merge_policy="residual"),
    )


def build_merge_adapters(model, merge_sites):
    adapters = {}
    for policy in ("shared", "residual"):
        adapters[policy] = CSPNStructuralMergeAdapter(
            model, "shared", None, EdgeQDQRuntime(),
            site_policies=dict((site, policy) for site in merge_sites))
    return adapters


def site_granularity(channels: int, group_size: Optional[int]) -> str:
    channels = int(channels)
    if channels <= 0:
        raise ValueError("activation channels must be positive")
    if group_size is None:
        return "tensor"
    group_size = int(group_size)
    if group_size <= 0:
        raise ValueError("activation group size must be positive")
    if group_size == 1:
        return "channel"
    return "group" if channels % group_size == 0 else "tensor"


def strict_owned_inputs():
    return {
        "gud_up_proj_layer1.conv1",
        "gud_up_proj_layer1.sc_conv1",
        "gud_up_proj_layer4.conv1_1",
    }


def strict_owned_outputs():
    return {"conv1_1", "conv2", "gud_up_proj_layer5.conv1"}


def validate_strict_site_contract(instrumentor, rotation) -> None:
    ordinary_sites = instrumentor.activation_site_keys(ORDINARY_GROUPS)
    ordinary_owners = set(activation_owner(key) for key in ordinary_sites)
    if ordinary_owners != STRICT_ACTIVATION_OWNERS:
        raise RuntimeError(
            "official CSPN strict activation sites changed: missing=%s extra=%s" %
            (sorted(STRICT_ACTIVATION_OWNERS - ordinary_owners),
             sorted(ordinary_owners - STRICT_ACTIVATION_OWNERS)))
    expected_rotation_channels = {
        "decoder_entry": 512,
        "layer4_signed_skip": 64,
    }
    if rotation.channels != expected_rotation_channels:
        raise RuntimeError(
            "official CSPN strict rotation boundary contract changed")


def build_rotation_group_sizes(rotation, group_size: Optional[int]):
    sizes = {}
    for name in rotation.channels:
        channels = int(rotation.channels[name])
        granularity = site_granularity(channels, group_size)
        sizes[name] = None if granularity == "tensor" else int(group_size)
    return sizes


def build_rotation_activation_specs(
        rotation, bits: int, group_size: Optional[int],
        selected_owners=(), promoted_owners=()):
    selected = set(tuple(owner) for owner in selected_owners)
    promoted = set(tuple(owner) for owner in promoted_owners)
    group_sizes = build_rotation_group_sizes(rotation, group_size)
    specs = {}
    for name in rotation.channels:
        channels = int(rotation.channels[name])
        owner = ("rotation.%s" % name, "boundary")
        apply_group = not selected or owner in selected
        current = group_sizes[name] if apply_group else None
        base = QuantSpec.signed_tensor(int(bits))
        spec = _granular_spec(
            base, 1, channels, current)
        specs[owner] = spec.with_bits(8) if owner in promoted else spec
    return specs


def activation_owner(key) -> Tuple[str, str]:
    if isinstance(key, str):
        return key, "relu_output"
    return str(key[0]), str(key[1])


def apply_activation_bit_assignment(
        specs: Dict[object, QuantSpec], rotation_specs,
        assignment):
    declared = tuple(
        (tuple(owner), int(bits)) for owner, bits in assignment)
    owners = tuple(owner for owner, bits in declared)
    if len(owners) != len(set(owners)):
        raise ValueError("activation bit assignment contains duplicates")
    allowed = (2, 4, 6, 8)
    for owner, bits in declared:
        if bits not in allowed:
            raise ValueError(
                "activation bits must be one of %s: %s=%d" %
                (allowed, owner, bits))
    if not declared:
        return dict(specs), dict(rotation_specs)
    expected = {
        activation_owner(key) for key in specs
    } | set(tuple(owner) for owner in rotation_specs)
    if set(owners) != expected:
        raise ValueError(
            "activation bit assignment coverage mismatch: missing=%s "
            "extra=%s" % (
                sorted(expected - set(owners), key=str),
                sorted(set(owners) - expected, key=str)))
    bits_by_owner = dict(declared)
    ordinary = dict(
        (key, specs[key].with_bits(bits_by_owner[activation_owner(key)]))
        for key in specs)
    boundaries = dict(
        (owner, rotation_specs[owner].with_bits(
            bits_by_owner[tuple(owner)]))
        for owner in rotation_specs)
    return ordinary, boundaries


def _granular_spec(base: QuantSpec, channel_dim: int, channels: int,
                   group_size: Optional[int]) -> QuantSpec:
    granularity = site_granularity(channels, group_size)
    if granularity == "tensor":
        return base
    if granularity == "channel":
        return QuantSpec(
            bits=base.bits, scheme=base.scheme,
            granularity="channel", axis=channel_dim,
            observer=base.observer, transform=base.transform,
            signed=base.signed, preserve_zero=base.preserve_zero,
            dynamic=base.dynamic)
    return QuantSpec(
        bits=base.bits, scheme=base.scheme,
        granularity="group", axis=channel_dim,
        group_size=int(group_size), observer=base.observer,
        transform=base.transform, signed=base.signed,
        preserve_zero=base.preserve_zero, dynamic=base.dynamic)


def build_activation_specs(
        instrumentor: HardwareAlignedInstrumentor,
        groups, bits: int, group_size: Optional[int],
        selected_owners=(), promoted_owners=(),
        dynamic: bool = False) -> Dict[object, QuantSpec]:
    groups = set(groups)
    base_specs = instrumentor.tensor_activation_specs(bits, groups)
    selected = set(tuple(owner) for owner in selected_owners)
    promoted = set(tuple(owner) for owner in promoted_owners)
    specs = {}
    for key in instrumentor.activation_site_keys(groups):
        base = base_specs[key]
        owner = activation_owner(key)
        observer = instrumentor.channel_observers[key] \
            if not isinstance(key, str) else \
            instrumentor.relu_channel_observers[key]
        channels = int(observer.minimum.numel())
        apply_group = not selected or owner in selected
        spec = _granular_spec(
            base, observer.channel_dim, channels,
            group_size if apply_group else None)
        spec = spec.with_bits(8) if owner in promoted else spec
        specs[key] = spec.with_dynamic(dynamic)
    return specs


def build_activation_maxima(
        instrumentor: HardwareAlignedInstrumentor,
        specs: Dict[object, QuantSpec], scale_factors) -> Dict[object, object]:
    factors = dict((tuple(owner), float(factor))
                   for owner, factor in scale_factors)
    for owner in factors:
        if not math.isfinite(factors[owner]) or factors[owner] <= 0.0:
            raise ValueError("activation scale factor must be finite and positive")
    maxima = {}
    observed_owners = set()
    for key in specs:
        owner = activation_owner(key)
        if owner not in factors:
            continue
        observed_owners.add(owner)
        observer = instrumentor.channel_observers[key] \
            if not isinstance(key, str) else \
            instrumentor.relu_channel_observers[key]
        spec = specs[key]
        extent = observer.maximum if not spec.signed else torch.maximum(
            observer.minimum.abs(), observer.maximum.abs())
        if spec.granularity == "tensor":
            maximum = float(extent.max().item())
        elif spec.granularity == "channel":
            maximum = extent.clone()
        else:
            groups = int(extent.numel()) // int(spec.group_size)
            maximum = extent.reshape(
                groups, int(spec.group_size)).amax(dim=1)
        maxima[key] = maximum * factors[owner]
    missing = set(factors) - observed_owners
    if missing:
        raise ValueError("activation scale owners were not observed: %s" %
                         sorted(missing))
    return maxima


def build_smooth_channel_maxima(
        instrumentor: HardwareAlignedInstrumentor,
        groups) -> Dict[str, torch.Tensor]:
    maxima = {}
    for key in instrumentor.activation_site_keys(groups):
        if isinstance(key, str) or key[1] != "input":
            continue
        name = str(key[0])
        observer = instrumentor.channel_observers[key]
        maxima[name] = torch.maximum(
            observer.minimum.abs(), observer.maximum.abs())
    return maxima


class ModuleOutputCapture(object):
    def __init__(self, model: nn.Module,
                 sites: Sequence[BlockSite]) -> None:
        modules = dict(model.named_modules())
        self.current = {}
        self.handles = []
        for site in sites:
            if site.module not in modules:
                raise ValueError("missing CSPN block module: %s" % site.module)
            self.handles.append(modules[site.module].register_forward_hook(
                self._hook(site.name)))

    def _hook(self, name: str):
        def capture(module, inputs, output):
            del module, inputs
            if not torch.is_tensor(output):
                raise TypeError("CSPN block output must be a tensor: %s" % name)
            self.current[name] = output.detach()
        return capture

    def reset(self) -> None:
        self.current = {}

    def values(self) -> Dict[str, torch.Tensor]:
        return dict(self.current)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


class BlockErrorAccumulator(object):
    def __init__(self, sites: Sequence[BlockSite]) -> None:
        self.totals = dict((site.name, {
            "signal_energy": 0.0,
            "error_energy": 0.0,
            "elements": 0,
        }) for site in sites)

    def update(self, reference: Dict[str, torch.Tensor],
               candidate: Dict[str, torch.Tensor]) -> None:
        if set(reference) != set(self.totals) or set(candidate) != set(self.totals):
            raise ValueError("CSPN block capture coverage mismatch")
        for name in self.totals:
            left = reference[name].to(torch.float64)
            right = candidate[name].to(torch.float64)
            if left.shape != right.shape:
                raise ValueError("CSPN block shape mismatch: %s" % name)
            difference = right - left
            self.totals[name]["signal_energy"] += float(
                left.square().sum().item())
            self.totals[name]["error_energy"] += float(
                difference.square().sum().item())
            self.totals[name]["elements"] += int(left.numel())

    @staticmethod
    def _summary(total: Dict[str, object]) -> Dict[str, float]:
        signal = float(total["signal_energy"])
        error = float(total["error_energy"])
        elements = int(total["elements"])
        sqnr = float("inf") if error == 0.0 else \
            float("-inf") if signal == 0.0 else \
            10.0 * math.log10(signal / error)
        return {
            "elements": elements,
            "signal_energy": signal,
            "error_energy": error,
            "block_output_mse": error / float(elements),
            "block_output_sqnr": sqnr,
        }

    def rows(self) -> List[Dict[str, object]]:
        rows = []
        for name in self.totals:
            row = {"block": name}
            row.update(self._summary(self.totals[name]))
            rows.append(row)
        return rows

    def aggregate(self) -> Dict[str, float]:
        total = {
            "signal_energy": sum(
                float(row["signal_energy"]) for row in self.totals.values()),
            "error_energy": sum(
                float(row["error_energy"]) for row in self.totals.values()),
            "elements": sum(
                int(row["elements"]) for row in self.totals.values()),
        }
        return self._summary(total)


def cspn_quant_group(name: str, module: nn.Module) -> Optional[str]:
    if name.startswith("gud_up_proj_layer6"):
        return None
    return classify_module("cspn", name, module)


def depth_sample_metrics(gt, pred, sparse):
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    sparse = np.asarray(sparse)
    regions = regional_depth_metrics(gt, pred, sparse)
    by_region = dict((row["region"], row) for row in regions)
    valid = np.isfinite(gt) & (gt > 1e-4)
    inverse_error = (
        1.0 / np.maximum(pred[valid], 1e-6)
        - 1.0 / np.maximum(gt[valid], 1e-6))
    nonfinite = valid & ~np.isfinite(pred)
    return {
        "RMSE": by_region["all"]["RMSE"],
        "MAE": by_region["all"]["MAE"],
        "ABS_REL": by_region["all"]["ABS_REL"],
        "IRMSE": float(np.sqrt(np.mean(inverse_error ** 2))),
        "flat_RMSE": by_region["smooth"]["RMSE"],
        "boundary_RMSE": by_region["boundary"]["RMSE"],
        "nonfinite_ratio": float(np.count_nonzero(nonfinite)) /
        float(np.count_nonzero(valid)),
    }, regions


def _load_cspn(saved_args, checkpoint: Path, device: torch.device):
    if saved_args.model != "cspn":
        raise ValueError("activation resolution runner requires model=cspn")
    model, architecture = sweep.BUILDERS["cspn"](saved_args, device)
    state = torch.load(
        str(checkpoint), map_location="cpu", weights_only=False)
    load_report = load_model_state(model, state["net"], "cspn")
    model.eval()
    return model, architecture, load_report


def _model_args(saved_args, sample, device):
    batch = batch_from_sample(sample)
    model_args, _ = sweep.batch_to_model_input("cspn", batch, device)
    return model_args


def _forward(model, saved_args, sample, device,
             capture: ModuleOutputCapture):
    capture.reset()
    output = model(*_model_args(saved_args, sample, device))
    prediction = sweep.extract_pred(output).detach().cpu()[0, 0]
    blocks = capture.values()
    if len(blocks) != len(CSPN_BLOCK_SITES):
        raise RuntimeError("CSPN block capture is incomplete")
    return prediction, blocks


def configure_merge_adapters(config, merge_adapters):
    for adapter in merge_adapters.values():
        adapter.disable()
    policy = config["merge_policy"]
    if policy == "none":
        return None
    adapter = merge_adapters[policy]
    adapter.reset_statistics()
    adapter.quantize()
    return adapter


def _configure_quantized(
        config: Dict[str, object],
        instrumentor: HardwareAlignedInstrumentor,
        rotation, propagation, merge_adapters):
    rotation.disable()
    active_merge_adapter = configure_merge_adapters(config, merge_adapters)
    if config["name"] == "FP32":
        instrumentor.disable()
        propagation.disable()
        return {}, {}, active_merge_adapter
    specs = build_activation_specs(
        instrumentor,
        config["activation_groups"],
        int(config["a_bits"]),
        config["group_size"],
        selected_owners=config["selected_owners"],
        promoted_owners=config["promoted_owners"],
        dynamic=config["dynamic"])
    rotation_specs = build_rotation_activation_specs(
        rotation, int(config["a_bits"]), config["group_size"],
        selected_owners=config["selected_owners"],
        promoted_owners=config["promoted_owners"]) \
        if config["activation_groups"] else {}
    specs, rotation_specs = apply_activation_bit_assignment(
        specs, rotation_specs, config["activation_bit_overrides"])
    generic_owners = set(activation_owner(key) for key in specs)
    rotation_owners = set(rotation_specs)
    declared_factors = dict(
        (tuple(owner), float(factor))
        for owner, factor in config["scale_factors"])
    unknown_factors = set(declared_factors) - generic_owners - rotation_owners
    if unknown_factors:
        raise ValueError("activation scale owners were not observed: %s" %
                         sorted(unknown_factors))
    generic_factors = tuple(
        (owner, declared_factors[owner])
        for owner in declared_factors if owner in generic_owners)
    activation_maxima = build_activation_maxima(
        instrumentor, specs, generic_factors)
    activation_range_overrides = dict(
        config["activation_range_overrides"])
    rotation_range_overrides = dict(config["rotation_range_overrides"])
    if bool(activation_range_overrides) != bool(rotation_range_overrides):
        raise ValueError(
            "ordinary and rotation ranges must be declared together")
    if activation_range_overrides:
        if set(activation_range_overrides) != set(specs):
            raise ValueError(
                "ordinary range overrides must cover every activation spec")
        if set(rotation_range_overrides) != set(rotation.channels):
            raise ValueError(
                "rotation range overrides must cover every boundary")
        activation_maxima = activation_range_overrides
    smooth_channel_maxima = build_smooth_channel_maxima(
        instrumentor, config["smooth_groups"])
    instrumentor.configure_components_with_ranges(
        int(config["w_bits"]), int(config["a_bits"]),
        config["weight_groups"], config["activation_groups"],
        specs, bool(config["quantize_bias"]), activation_maxima,
        smooth_channel_maxima=smooth_channel_maxima,
        smooth_alpha=config["smooth_alpha"],
        activation_permutations=dict(
            config["activation_permutations"]),
        activation_isolations=dict(
            config["activation_isolations"]),
        weight_bit_overrides=dict(config["weight_bit_overrides"]))
    if config["activation_groups"]:
        rotation_group_sizes = {}
        rotation_bits = {}
        rotation_factors = {}
        for name in rotation.channels:
            owner = ("rotation.%s" % name, "boundary")
            spec = rotation_specs[owner]
            rotation_bits[name] = int(spec.bits)
            rotation_group_sizes[name] = spec.group_size \
                if spec.granularity == "group" else \
                1 if spec.granularity == "channel" else None
            rotation_factors[name] = declared_factors[owner] \
                if owner in declared_factors else 1.0
        methods = {
            "decoder_entry": "identity",
            "layer4_signed_skip": "identity",
        }
        if rotation_range_overrides:
            rotation.configure_specs_with_ranges(
                methods, rotation_bits, rotation_group_sizes,
                rotation_factors, rotation_range_overrides,
                quantize=True, absorb_weights=False)
        else:
            rotation.configure_specs(
                methods, rotation_bits, rotation_group_sizes,
                rotation_factors, quantize=True, absorb_weights=False)
    propagation.configure(PropagationQuantConfig(**config["propagation"]))
    return specs, rotation_specs, active_merge_adapter


def _calibrate(model, saved_args, dataset, indices, device,
               seed, instrumentor, rotation, propagation) -> None:
    instrumentor.observe()
    rotation.observe()
    propagation.observe()
    with torch.no_grad():
        for rank, index in enumerate(indices, 1):
            sample = seeded_sample(dataset, index, seed)
            model(*_model_args(saved_args, sample, device))
            if rank % 16 == 0 or rank == len(indices):
                print("CSPN calibration %d/%d" %
                      (rank, len(indices)), flush=True)
    instrumentor.freeze()
    rotation.freeze()
    propagation.freeze()


def _calibrate_merge(model, saved_args, dataset, indices, device,
                     seed, instrumentor, rotation, propagation,
                     merge_adapters, policy) -> None:
    instrumentor.disable()
    rotation.disable()
    propagation.disable()
    for adapter in merge_adapters.values():
        adapter.disable()
    merge_adapter = merge_adapters[policy]
    merge_adapter.observe()
    with torch.no_grad():
        for rank, index in enumerate(indices, 1):
            sample = seeded_sample(dataset, index, seed)
            model(*_model_args(saved_args, sample, device))
            if rank % 16 == 0 or rank == len(indices):
                print("CSPN merge calibration %d/%d" %
                      (rank, len(indices)), flush=True)
    merge_adapter.freeze(4)


def _derived_configuration(name, base, scale_factors,
                           merge_policy=None):
    policy = base["merge_policy"] if merge_policy is None else merge_policy
    return _configuration(
        name,
        base["weight_groups"],
        base["activation_groups"],
        base["propagation"],
        base["granularity"],
        base["group_size"],
        promoted_owners=base["promoted_owners"],
        selected_owners=base["selected_owners"],
        scale_factors=scale_factors,
        merge_policy=policy,
        smooth_groups=base["smooth_groups"],
        smooth_alpha=base["smooth_alpha"],
        activation_range_overrides=base["activation_range_overrides"],
        rotation_range_overrides=base["rotation_range_overrides"],
        activation_permutations=base["activation_permutations"],
        activation_isolations=base["activation_isolations"],
        weight_bit_overrides=base["weight_bit_overrides"],
        activation_bit_overrides=base["activation_bit_overrides"],
        dynamic=base["dynamic"])


def _owner_clipping_ratio(tensor_rows, owner) -> float:
    selected = [
        row for row in tensor_rows
        if (str(row["module"]), str(row["kind"])) == tuple(owner)
    ]
    if not selected:
        raise ValueError("missing activation rows for scale owner: %s" %
                         (owner,))
    clipping = sum(
        float(row["clipping_error_energy"]) for row in selected)
    total = sum(float(row["total_error_energy"]) for row in selected)
    return 0.0 if total == 0.0 else clipping / total


def _spec_by_owner(specs: Dict[object, QuantSpec]):
    rows = {}
    for key in specs:
        owner = activation_owner(key)
        if owner in rows and rows[owner] != specs[key]:
            raise ValueError("activation owner has inconsistent specs: %s" %
                             (owner,))
        rows[owner] = specs[key]
    return rows


def _annotate_activation_rows(rows, config, specs, rotation_specs):
    by_owner = _spec_by_owner(specs)
    by_owner.update(_spec_by_owner(rotation_specs))
    output = []
    for source in rows:
        owner = (str(source["module"]), str(source["kind"]))
        if owner not in by_owner:
            raise ValueError("missing activation spec for recorded owner: %s" %
                             (owner,))
        spec = by_owner[owner]
        row = dict(source)
        row["model"] = "cspn"
        row["config"] = config["name"]
        row["bits"] = spec.bits
        row["granularity"] = spec.granularity
        row["group_size"] = "" if spec.group_size is None else spec.group_size
        output.append(row)
    return output


def run_configuration(
        reference_model, quantized_model, saved_args, dataset, indices,
        split, device, seed, config, instrumentor, rotation, propagation,
        merge_adapters, reference_capture, quantized_capture, sample_capacity,
        prediction_root=None):
    specs, rotation_specs, active_merge_adapter = _configure_quantized(
        config, instrumentor, rotation, propagation, merge_adapters)
    recorder = None
    if config["activation_groups"]:
        recorder = ActivationResolutionRecorder(split, sample_capacity)
        instrumentor.set_activation_recorder(recorder)
        rotation.set_activation_recorder(recorder)
    else:
        instrumentor.clear_activation_recorder()
        rotation.clear_activation_recorder()

    sample_rows = []
    region_rows = []
    propagation_rows = []
    block_error = BlockErrorAccumulator(CSPN_BLOCK_SITES)
    prediction_dir = None
    if prediction_root is not None:
        prediction_dir = prepare_prediction_dir(
            prediction_root, str(config["name"]))

    with torch.no_grad():
        for rank, index in enumerate(indices, 1):
            sample = seeded_sample(dataset, index, seed)
            reference_prediction, reference_blocks = _forward(
                reference_model, saved_args, sample, device,
                reference_capture)
            if config["name"] == "FP32":
                prediction = reference_prediction
                blocks = reference_blocks
            else:
                prediction, blocks = _forward(
                    quantized_model, saved_args, sample, device,
                    quantized_capture)
            block_error.update(reference_blocks, blocks)

            if split == "evaluation":
                pred = prediction.numpy()
                if not np.isfinite(pred).all():
                    raise RuntimeError(
                        "non-finite prediction: config=%s sample=%d" %
                        (config["name"], index))
                gt = sample["depth"][0].numpy()
                sparse = sample["rgbd"][3].numpy()
                metrics, regions = depth_sample_metrics(gt, pred, sparse)
                metrics.update({
                    "model": "cspn",
                    "config": config["name"],
                    "sample_index": int(index),
                })
                sample_rows.append(metrics)
                for region in regions:
                    row = dict(region)
                    row.update({
                        "model": "cspn",
                        "config": config["name"],
                        "sample_index": int(index),
                    })
                    region_rows.append(row)
                if prediction_dir is not None:
                    payload = prediction_payload(
                        gt, reference_prediction.numpy(), pred,
                        int(index), "cspn", str(config["name"]),
                        sparse=sparse,
                        rgb=sample["rgbd"][:3].permute(1, 2, 0).numpy())
                    write_prediction_payload(prediction_dir, payload)

            if config["propagation"] is not None:
                for source in propagation.statistics():
                    row = dict(source)
                    row.update({
                        "model": "cspn",
                        "config": config["name"],
                        "split": split,
                        "sample_index": int(index),
                    })
                    propagation_rows.append(row)
            if rank % 16 == 0 or rank == len(indices):
                print("%s %s %d/%d" % (
                    config["name"], split, rank, len(indices)), flush=True)

    instrumentor.clear_activation_recorder()
    rotation.clear_activation_recorder()
    block_rows = []
    for source in block_error.rows():
        row = dict(source)
        row.update({
            "model": "cspn",
            "config": config["name"],
            "split": split,
        })
        block_rows.append(row)
    aggregate = block_error.aggregate()
    aggregate.update({
        "model": "cspn",
        "config": config["name"],
        "split": split,
        "block": "__all__",
    })
    block_rows.append(aggregate)

    tensor_rows = [] if recorder is None else \
        _annotate_activation_rows(
            recorder.tensor_rows(), config, specs, rotation_specs)
    channel_rows = [] if recorder is None else \
        _annotate_activation_rows(
            recorder.channel_rows(), config, specs, rotation_specs)
    layer_rows = [] if config["name"] == "FP32" else \
        instrumentor.statistics()
    for row in layer_rows:
        row.update({"model": "cspn", "config": config["name"]})
    merge_rows = []
    if active_merge_adapter is not None:
        selected_sites = set(active_merge_adapter.site_policies)
        for source in active_merge_adapter.manifest():
            if source["merge"] not in selected_sites:
                continue
            row = dict(source)
            row.update({
                "model": "cspn",
                "config": config["name"],
                "split": split,
            })
            merge_rows.append(row)
    return {
        "sample_rows": sample_rows,
        "region_rows": region_rows,
        "propagation_rows": propagation_rows,
        "block_rows": block_rows,
        "tensor_rows": tensor_rows,
        "channel_rows": channel_rows,
        "layer_rows": layer_rows,
        "merge_rows": merge_rows,
    }


def activation_granularity_summary(rows, config_name):
    selected = [
        row for row in rows
        if row["config"] == config_name and row["split"] == "calibration"
    ]
    element_counts = {"tensor": 0, "group": 0, "channel": 0}
    site_counts = {"tensor": 0, "group": 0, "channel": 0}
    for row in selected:
        granularity = str(row["granularity"])
        element_counts[granularity] += int(row["elements"])
        site_counts[granularity] += 1
    total_elements = sum(element_counts.values())
    total_sites = sum(site_counts.values())
    return {
        "activation_elements": total_elements,
        "tensor_element_fraction": element_counts["tensor"] /
        float(total_elements),
        "group_element_fraction": element_counts["group"] /
        float(total_elements),
        "channel_element_fraction": element_counts["channel"] /
        float(total_elements),
        "tensor_site_fraction": site_counts["tensor"] / float(total_sites),
        "group_site_fraction": site_counts["group"] / float(total_sites),
        "channel_site_fraction": site_counts["channel"] / float(total_sites),
    }


def _configuration_manifest(
        configurations, instrumentor, rotation, activation_rows):
    rows = []
    for config in configurations:
        specs = build_activation_specs(
            instrumentor, config["activation_groups"],
            int(config["a_bits"]), config["group_size"],
            selected_owners=config["selected_owners"],
            promoted_owners=config["promoted_owners"],
            dynamic=config["dynamic"])
        rotation_specs = build_rotation_activation_specs(
            rotation, int(config["a_bits"]), config["group_size"],
            selected_owners=config["selected_owners"],
            promoted_owners=config["promoted_owners"]) \
            if config["activation_groups"] else {}
        specs, rotation_specs = apply_activation_bit_assignment(
            specs, rotation_specs, config["activation_bit_overrides"])
        scale_count = 0
        granularity_counts = {"tensor": 0, "group": 0, "channel": 0}
        for key in specs:
            spec = specs[key]
            observer = instrumentor.channel_observers[key] \
                if not isinstance(key, str) else \
                instrumentor.relu_channel_observers[key]
            channels = int(observer.minimum.numel())
            if spec.granularity == "tensor":
                scale_count += 1
            elif spec.granularity == "channel":
                scale_count += channels
            else:
                scale_count += channels // int(spec.group_size)
            granularity_counts[spec.granularity] += 1
        for owner in rotation_specs:
            spec = rotation_specs[owner]
            name = owner[0].split(".", 1)[1]
            channels = int(rotation.channels[name])
            if spec.granularity == "tensor":
                scale_count += 1
            elif spec.granularity == "channel":
                scale_count += channels
            else:
                scale_count += channels // int(spec.group_size)
            granularity_counts[spec.granularity] += 1
        granularity_summary = {
            "activation_elements": 0,
            "tensor_element_fraction": 0.0,
            "group_element_fraction": 0.0,
            "channel_element_fraction": 0.0,
            "tensor_site_fraction": 0.0,
            "group_site_fraction": 0.0,
            "channel_site_fraction": 0.0,
        } if not config["activation_groups"] else \
            activation_granularity_summary(
                activation_rows, str(config["name"]))
        row = {
            "config": config["name"],
            "weight_bits": "" if config["name"] in ("FP32", "PA_ONLY", "A4_ONLY")
            else config["w_bits"],
            "activation_bits": "" if not config["activation_groups"] else
            config["a_bits"],
            "weight_groups": ";".join(sorted(config["weight_groups"])),
            "activation_groups": ";".join(
                sorted(config["activation_groups"])),
            "granularity": config["granularity"],
            "group_size": "" if config["group_size"] is None else
            config["group_size"],
            "dynamic": int(config["dynamic"]),
            "activation_sites": len(specs) + len(rotation_specs),
            "activation_scales": scale_count,
            "tensor_sites": granularity_counts["tensor"],
            "group_sites": granularity_counts["group"],
            "channel_sites": granularity_counts["channel"],
            "promoted_sites": len(config["promoted_owners"]),
            "selected_sites": len(config["selected_owners"]),
            "calibrated_scale_sites": len(config["scale_factors"]),
            "merge_policy": config["merge_policy"],
            "bias_format": "fp32",
            "guidance_head": "fp32",
            "propagation": "fp32" if config["propagation"] is None else
            "a8_int16_q13_int32",
        }
        row.update(granularity_summary)
        rows.append(row)
    return rows


def _mean_metric_rows(sample_rows, configurations):
    output = []
    for config in configurations:
        selected = [
            row for row in sample_rows if row["config"] == config["name"]]
        if not selected:
            raise ValueError("missing evaluation metrics: %s" % config["name"])
        row = {"config": config["name"], "samples": len(selected)}
        for field in ("RMSE", "MAE", "ABS_REL", "IRMSE",
                      "flat_RMSE", "boundary_RMSE"):
            row[field] = float(np.mean(np.asarray(
                [float(source[field]) for source in selected],
                dtype=np.float64)))
        output.append(row)
    return output


def _attribution_rows(mean_rows, block_rows):
    means = dict((row["config"], row) for row in mean_rows)
    output = []
    for metric in ("RMSE", "MAE", "ABS_REL", "IRMSE",
                   "flat_RMSE", "boundary_RMSE"):
        values = dict((name, float(means[name][metric])) for name in (
            "FP32", "PA_ONLY", "W4_ONLY", "A4_ONLY", "W4A4_RTN"))
        for source, value in attribution_components(values).items():
            output.append({
                "scope": "end_to_end",
                "metric": metric,
                "source": source,
                "delta": value,
            })
    calibration = [
        row for row in block_rows
        if row["split"] == "calibration" and row["block"] == "__all__"
        and row["config"] in (
            "FP32", "PA_ONLY", "W4_ONLY", "A4_ONLY", "W4A4_RTN")]
    values = dict((row["config"], float(row["block_output_mse"]))
                  for row in calibration)
    for source, value in attribution_components(values).items():
        output.append({
            "scope": "calibration_blocks",
            "metric": "block_output_mse",
            "source": source,
            "delta": value,
        })
    return output


def _candidate_owner_sites(tensor_rows, candidate_sites):
    by_site = dict((str(row["site"]), row) for row in tensor_rows)
    owner_sites = []
    observed = set()
    for site in candidate_sites:
        row = by_site[site]
        owner = (str(row["module"]), str(row["kind"]))
        if owner not in observed:
            observed.add(owner)
            owner_sites.append((site, owner))
    return tuple(owner_sites)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-metrics", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--calibration-samples", type=int,
        choices=(CALIBRATION_SAMPLES,), required=True)
    parser.add_argument("--sample-capacity", type=int, required=True)
    parser.add_argument("--candidate-sites", type=int, required=True)
    parser.add_argument("--sensitive-sites", type=int, required=True)
    parser.add_argument("--fold-max-error", type=float, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.device.startswith("cuda"):
        raise ValueError("CSPN activation resolution requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.sample_capacity <= 0 or args.candidate_sites <= 0 or \
            args.sensitive_sites <= 0:
        raise ValueError("sampling and candidate counts must be positive")
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False

    run_dir = Path(args.run_dir)
    checkpoint = Path(args.checkpoint)
    saved_args = prepare_args(load_run_args(run_dir), args)
    saved_args.data_root = args.data_root
    reference_model, architecture, reference_load = _load_cspn(
        saved_args, checkpoint, device)
    quantized_model, quantized_architecture, quantized_load = _load_cspn(
        saved_args, checkpoint, device)
    if architecture != quantized_architecture or reference_load != quantized_load:
        raise RuntimeError("paired CSPN model construction is inconsistent")

    trainset = calibration_dataset(saved_args)
    if len(trainset) < CALIBRATION_SAMPLES:
        raise ValueError("NYU training split has fewer than 128 samples")
    calibration_indices = np.random.RandomState(args.seed).choice(
        len(trainset), CALIBRATION_SAMPLES, replace=False).tolist()
    preparation_sample = seeded_sample(
        trainset, calibration_indices[0], args.seed)
    preparation_args = _model_args(
        saved_args, preparation_sample, device)
    reference_preparation = prepare_hardware_model(
        reference_model, preparation_args,
        excluded_pairs=(("conv1_1", "bn1"),))
    quantized_preparation = prepare_hardware_model(
        quantized_model, preparation_args,
        excluded_pairs=(("conv1_1", "bn1"),))
    for preparation in (reference_preparation, quantized_preparation):
        if float(preparation["primary_max_abs_error"]) > args.fold_max_error:
            raise RuntimeError("Conv-BN fold exceeds declared error threshold")
    if reference_preparation["folded_pairs"] != \
            quantized_preparation["folded_pairs"]:
        raise RuntimeError("paired CSPN fold manifests differ")

    semantic = install_model_semantic_adapter(
        quantized_model, "cspn", strict=True)
    boundaries = semantic.rotation_boundaries()
    semantic.close()
    instrumentor = HardwareAlignedInstrumentor(
        quantized_model, cspn_quant_group,
        quantized_preparation["fused_relu_producers"],
        externally_owned_outputs=strict_owned_outputs(),
        externally_owned_inputs=strict_owned_inputs())
    rotation = CSPNRotationController(
        quantized_model, boundaries, seed=args.seed)
    propagation = install_propagation_adapter("cspn", quantized_model)
    reference_capture = ModuleOutputCapture(
        reference_model, CSPN_BLOCK_SITES)
    quantized_capture = ModuleOutputCapture(
        quantized_model, CSPN_BLOCK_SITES)
    merge_adapters = {}

    started = time.time()
    _calibrate(
        quantized_model, saved_args, trainset, calibration_indices,
        device, args.seed, instrumentor, rotation, propagation)
    validate_strict_site_contract(instrumentor, rotation)

    attribution_configs = build_attribution_configurations()
    group_configs = build_group_configurations()
    calibration_results = {}
    calibration_order = list(attribution_configs) + list(group_configs)
    for config in calibration_order:
        calibration_results[config["name"]] = run_configuration(
            reference_model, quantized_model, saved_args,
            trainset, calibration_indices, "calibration",
            device, args.seed, config, instrumentor, rotation, propagation,
            merge_adapters, reference_capture, quantized_capture,
            args.sample_capacity)

    base_tensor_rows = calibration_results["W4A4_RTN"]["tensor_rows"]
    candidate_sites = select_candidate_sites(
        base_tensor_rows, args.candidate_sites)
    candidate_owner_sites = _candidate_owner_sites(
        base_tensor_rows, candidate_sites)
    intervention_rows = []
    intervention_configs = []
    for index, (site, owner) in enumerate(candidate_owner_sites, 1):
        config = _configuration(
            "A8_SITE_%02d" % index,
            ORDINARY_GROUPS, ORDINARY_GROUPS,
            PROPAGATION_A8_Q13, promoted_owners=(owner,))
        result = run_configuration(
            reference_model, quantized_model, saved_args,
            trainset, calibration_indices, "calibration",
            device, args.seed, config, instrumentor, rotation, propagation,
            merge_adapters, reference_capture, quantized_capture,
            args.sample_capacity)
        calibration_results[config["name"]] = result
        intervention_configs.append(config)
        block_name = owner_block(owner)
        baseline_block = next(
            row for row in calibration_results["W4A4_RTN"]["block_rows"]
            if row["block"] == block_name)
        promoted_block = next(
            row for row in result["block_rows"]
            if row["block"] == block_name)
        intervention_rows.append({
            "config": config["name"],
            "site": site,
            "module": owner[0],
            "kind": owner[1],
            "block": block_name,
            "baseline_block_mse": baseline_block["block_output_mse"],
            "promoted_block_mse": promoted_block["block_output_mse"],
            "block_mse_delta": float(
                baseline_block["block_output_mse"])
            - float(promoted_block["block_output_mse"]),
            "block_sqnr_delta": float(promoted_block["block_output_sqnr"])
            - float(baseline_block["block_output_sqnr"]),
            "selected": 0,
        })
    ranked_interventions = sorted(
        intervention_rows,
        key=lambda row: (-float(row["block_mse_delta"]), str(row["site"])))
    selected_interventions = [
        row for row in ranked_interventions
        if float(row["block_mse_delta"]) > 0.0][:args.sensitive_sites]
    for row in selected_interventions:
        row["selected"] = 1
    sensitive_owners = tuple(
        (str(row["module"]), str(row["kind"]))
        for row in selected_interventions)

    group_selection_rows = []
    for config in (attribution_configs[-1],) + group_configs:
        aggregate = next(
            row for row in calibration_results[config["name"]]["block_rows"]
            if row["block"] == "__all__")
        row = dict(aggregate)
        row["config"] = config["name"]
        group_selection_rows.append(row)
    selected_global = select_calibration_configuration(group_selection_rows)
    config_by_name = dict(
        (config["name"], config)
        for config in calibration_order + intervention_configs)
    selected_global_config = config_by_name[selected_global["config"]]

    selective_configs = []
    if selected_global_config["group_size"] is not None and sensitive_owners:
        selective = _configuration(
            "W4A4_SELECTIVE_%s" % selected_global_config["name"],
            ORDINARY_GROUPS, ORDINARY_GROUPS, PROPAGATION_A8_Q13,
            selected_global_config["granularity"],
            selected_global_config["group_size"],
            selected_owners=sensitive_owners)
        calibration_results[selective["name"]] = run_configuration(
            reference_model, quantized_model, saved_args,
            trainset, calibration_indices, "calibration",
            device, args.seed, selective, instrumentor, rotation, propagation,
            merge_adapters, reference_capture, quantized_capture,
            args.sample_capacity)
        selective_configs.append(selective)
        aggregate = next(
            row for row in calibration_results[selective["name"]]["block_rows"]
            if row["block"] == "__all__")
        group_selection_rows.append(dict(aggregate, config=selective["name"]))

    extension_configs = []
    merge_sites = decoder_merge_sites(sensitive_owners)
    selected_before_residual = select_calibration_configuration(
        group_selection_rows)
    candidate_configs = calibration_order + selective_configs
    candidate_by_name = dict(
        (config["name"], config) for config in candidate_configs)
    residual_base = candidate_by_name[selected_before_residual["config"]]
    if merge_sites:
        merge_adapters.update(build_merge_adapters(
            quantized_model, merge_sites))
        for policy in ("shared", "residual"):
            _calibrate_merge(
                quantized_model, saved_args, trainset, calibration_indices,
                device, args.seed, instrumentor, rotation, propagation,
                merge_adapters, policy)
        extension_configs.extend(build_merge_configurations(residual_base))
        for config in extension_configs:
            calibration_results[config["name"]] = run_configuration(
                reference_model, quantized_model, saved_args,
                trainset, calibration_indices, "calibration",
                device, args.seed, config, instrumentor, rotation,
                propagation, merge_adapters, reference_capture,
                quantized_capture, args.sample_capacity)
            aggregate = next(
                row for row in calibration_results[config["name"]]["block_rows"]
                if row["block"] == "__all__")
            group_selection_rows.append(dict(
                aggregate, config=config["name"]))

    scale_search_rows = []
    calibrated_scale_configs = []
    selected_before_scale = select_calibration_configuration(
        group_selection_rows)
    candidate_by_name.update(dict(
        (config["name"], config) for config in extension_configs))
    scale_base = candidate_by_name[selected_before_scale["config"]]
    selected_factors = {}
    for owner_index, owner in enumerate(sensitive_owners, 1):
        block_name = owner_block(owner)
        owner_rows = []
        for factor in SCALE_FACTORS:
            factors = dict(selected_factors)
            factors[owner] = factor
            scale_factors = tuple(
                (current, factors[current]) for current in sensitive_owners
                if current in factors)
            candidate = _derived_configuration(
                "SCALE_%02d_%04d" %
                (owner_index, int(round(factor * 1000.0))),
                scale_base, scale_factors=scale_factors)
            result = run_configuration(
                reference_model, quantized_model, saved_args,
                trainset, calibration_indices, "calibration",
                device, args.seed, candidate, instrumentor, rotation,
                propagation,
                merge_adapters, reference_capture, quantized_capture,
                args.sample_capacity)
            block = next(
                row for row in result["block_rows"]
                if row["block"] == block_name)
            owner_tensor_rows = [
                row for row in result["tensor_rows"]
                if (row["module"], row["kind"]) == owner]
            owner_nonzero = sum(
                float(row["nonzero_elements"])
                for row in owner_tensor_rows)
            owner_new_zeros = sum(
                float(row["new_zero_elements"]) for row in owner_tensor_rows)
            owner_signal = sum(
                float(row["signal_energy"]) for row in owner_tensor_rows)
            owner_error = sum(
                float(row["total_error_energy"]) for row in owner_tensor_rows)
            row = {
                "split": "calibration",
                "owner_index": owner_index,
                "module": owner[0],
                "kind": owner[1],
                "block": block_name,
                "factor": factor,
                "block_output_mse": block["block_output_mse"],
                "block_output_sqnr": block["block_output_sqnr"],
                "activation_new_zero_rate": owner_new_zeros /
                owner_nonzero,
                "activation_sqnr": 10.0 * math.log10(
                    owner_signal / owner_error),
                "clipping_error_ratio": _owner_clipping_ratio(
                    result["tensor_rows"], owner),
                "selected": 0,
            }
            owner_rows.append(row)
            scale_search_rows.append(row)
        selected_scale = select_activation_scale(owner_rows)
        selected_scale["selected"] = 1
        selected_factors[owner] = float(selected_scale["factor"])

    if selected_factors:
        calibrated_scale = _derived_configuration(
            "W4A4_CALIBRATED_SCALE", scale_base,
            scale_factors=tuple(
                (owner, selected_factors[owner])
                for owner in sensitive_owners if owner in selected_factors))
        calibration_results[calibrated_scale["name"]] = run_configuration(
            reference_model, quantized_model, saved_args,
            trainset, calibration_indices, "calibration",
            device, args.seed, calibrated_scale, instrumentor, rotation,
            propagation, merge_adapters, reference_capture, quantized_capture,
            args.sample_capacity)
        calibrated_scale_configs.append(calibrated_scale)
        aggregate = next(
            row for row in calibration_results[
                calibrated_scale["name"]]["block_rows"]
            if row["block"] == "__all__")
        group_selection_rows.append(dict(
            aggregate, config=calibrated_scale["name"]))

    selected_final = select_calibration_configuration(group_selection_rows)
    evaluation_configs = list(attribution_configs) + list(group_configs) + \
        selective_configs + extension_configs + calibrated_scale_configs
    evalset = evaluation_dataset(saved_args)
    evaluation_indices = load_sample_indices(args.sample_metrics)
    if len(evaluation_indices) != EVALUATION_SAMPLES:
        raise ValueError("CSPN evaluation requires exactly 64 sample indices")
    if max(evaluation_indices) >= len(evalset):
        raise ValueError("evaluation sample index exceeds NYU validation split")

    model_output = Path(args.out_dir) / "cspn"
    analysis_output = Path(args.out_dir) / "analysis"
    model_output.mkdir(parents=True, exist_ok=True)
    analysis_output.mkdir(parents=True, exist_ok=True)
    prediction_configs = prediction_configuration_names(
        selected_global["config"], selected_final["config"],
        scale_base["name"],
        tuple(config["name"] for config in calibrated_scale_configs))
    validate_evaluation_contract(evaluation_configs, prediction_configs)
    evaluation_results = {}
    for config in evaluation_configs:
        prediction_root = model_output \
            if config["name"] in prediction_configs else None
        evaluation_results[config["name"]] = run_configuration(
            reference_model, quantized_model, saved_args,
            evalset, evaluation_indices, "evaluation",
            device, args.seed, config, instrumentor, rotation, propagation,
            merge_adapters, reference_capture, quantized_capture,
            args.sample_capacity,
            prediction_root=prediction_root)

    sample_rows = []
    region_rows = []
    propagation_rows = []
    block_rows = []
    tensor_rows = []
    channel_rows = []
    layer_rows = []
    merge_rows = []
    for config in (calibration_order + intervention_configs +
                   selective_configs + extension_configs +
                   calibrated_scale_configs):
        result = calibration_results[config["name"]]
        block_rows.extend(result["block_rows"])
        tensor_rows.extend(result["tensor_rows"])
        channel_rows.extend(result["channel_rows"])
        propagation_rows.extend(result["propagation_rows"])
        merge_rows.extend(result["merge_rows"])
    for config in evaluation_configs:
        result = evaluation_results[config["name"]]
        sample_rows.extend(result["sample_rows"])
        region_rows.extend(result["region_rows"])
        block_rows.extend(result["block_rows"])
        tensor_rows.extend(result["tensor_rows"])
        channel_rows.extend(result["channel_rows"])
        propagation_rows.extend(result["propagation_rows"])
        layer_rows.extend(result["layer_rows"])
        merge_rows.extend(result["merge_rows"])

    validate_sample_coverage(
        sample_rows,
        tuple(str(config["name"]) for config in evaluation_configs),
        evaluation_indices)
    validate_prediction_coverage(model_output, evaluation_indices)
    mean_rows = _mean_metric_rows(sample_rows, evaluation_configs)
    accepted_final = selected_final
    calibrated_scale_transferred = False
    if calibrated_scale_configs and selected_final["config"] == \
            calibrated_scale_configs[0]["name"]:
        accepted_final = select_transferred_configuration(
            mean_rows, str(scale_base["name"]),
            str(calibrated_scale_configs[0]["name"]))
        calibrated_scale_transferred = \
            accepted_final["config"] == calibrated_scale_configs[0]["name"]
    regional_rows = []
    for config in evaluation_configs:
        selected = [
            row for row in region_rows if row["config"] == config["name"]]
        for source in aggregate_region_rows(selected):
            row = dict(source)
            row.update({"model": "cspn", "config": config["name"]})
            regional_rows.append(row)

    write_csv(
        model_output / "config_manifest.csv",
        _configuration_manifest(
            evaluation_configs, instrumentor, rotation, tensor_rows),
        ("config", "weight_bits", "activation_bits", "weight_groups",
         "activation_groups", "granularity", "group_size",
         "activation_sites", "activation_scales", "activation_elements",
         "tensor_element_fraction", "group_element_fraction",
         "channel_element_fraction", "bias_format", "guidance_head",
         "propagation"))
    write_csv(model_output / "sample_metrics.csv", sample_rows, SAMPLE_FIELDS)
    write_csv(
        model_output / "aggregate_metrics.csv", mean_rows,
        ("config", "samples", "RMSE", "MAE", "ABS_REL", "IRMSE",
         "flat_RMSE", "boundary_RMSE"))
    write_csv(
        model_output / "regional_metrics.csv", regional_rows,
        ("model", "config", "region", "RMSE", "MAE", "ABS_REL"))
    write_csv(
        model_output / "activation_resolution_metrics.csv", tensor_rows,
        ("model", "config", "split", "site", "module", "group",
         "kind", "bits", "granularity", "group_size"))
    write_csv(
        model_output / "activation_channel_metrics.csv", channel_rows,
        ("model", "config", "split", "site", "module", "group",
         "kind", "channel", "bits", "granularity", "group_size"))
    write_csv(
        model_output / "block_attribution_metrics.csv", block_rows,
        ("model", "config", "split", "block", "block_output_mse",
         "block_output_sqnr"))
    write_csv(
        model_output / "layer_quantization_metrics.csv", layer_rows,
        ("model", "config", "module", "group", "kind"))
    write_csv(
        model_output / "propagation_metrics.csv", propagation_rows,
        ("model", "config", "split", "sample_index", "signal",
         "iteration"))
    write_csv(
        analysis_output / "error_source_summary.csv",
        _attribution_rows(mean_rows, block_rows),
        ("scope", "metric", "source", "delta"))
    write_csv(
        analysis_output / "sensitive_activation_sites.csv",
        intervention_rows,
        ("config", "site", "module", "kind", "block",
         "baseline_block_mse",
         "promoted_block_mse", "block_mse_delta", "block_sqnr_delta",
         "selected"))
    write_csv(
        analysis_output / "group_size_search.csv", group_selection_rows,
        ("config", "split", "block_output_mse", "block_output_sqnr"))
    write_csv(
        analysis_output / "activation_scale_search.csv", scale_search_rows,
        ("split", "owner_index", "module", "kind", "block", "factor",
         "block_output_mse", "block_output_sqnr",
         "activation_new_zero_rate", "activation_sqnr",
         "clipping_error_ratio", "selected"))
    write_csv(
        model_output / "merge_branch_metrics.csv", merge_rows,
        ("model", "config", "split", "merge", "operation", "policy",
         "branch_bits", "scales", "output_bits", "output_scale"))
    write_json(model_output / "metadata.json", {
        "model": "cspn",
        "architecture": architecture,
        "model_class": type(reference_model).__name__,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_load": reference_load,
        "data_root": str(Path(args.data_root).resolve()),
        "seed": args.seed,
        "calibration_samples": CALIBRATION_SAMPLES,
        "calibration_indices": calibration_indices,
        "evaluation_samples": EVALUATION_SAMPLES,
        "evaluation_indices": evaluation_indices,
        "selected_global_config": selected_global["config"],
        "selected_final_calibration_config": selected_final["config"],
        "accepted_final_config": accepted_final["config"],
        "selected_final_config": accepted_final["config"],
        "calibrated_scale_base_config": scale_base["name"],
        "calibrated_scale_transferred": calibrated_scale_transferred,
        "sensitive_owners": [list(owner) for owner in sensitive_owners],
        "shared_merge_sites": list(merge_sites),
        "residual_merge_sites": list(merge_sites),
        "selected_scale_factors": [
            {"module": owner[0], "kind": owner[1],
             "factor": selected_factors[owner]}
            for owner in sensitive_owners if owner in selected_factors],
        "ordinary_activation_sites": len(
            instrumentor.activation_site_keys(ORDINARY_GROUPS)),
        "rotation_activation_sites": [
            {"name": name, "channels": int(rotation.channels[name])}
            for name in rotation.channels],
        "activation_resolution_coverage": {
            "ordinary_qdq": {
                "artifact": "activation_resolution_metrics.csv",
                "sites": len(instrumentor.activation_site_keys(
                    ORDINARY_GROUPS)),
            },
            "rotation_qdq": {
                "artifact": "activation_resolution_metrics.csv",
                "sites": len(rotation.channels),
            },
            "structural_merges": {
                "artifact": "merge_branch_metrics.csv",
                "sites": list(merge_sites),
                "policies": ["shared", "residual"],
            },
            "initial_depth": {
                "artifact": "block_attribution_metrics.csv",
                "block": "initial_depth",
            },
            "propagation": {
                "artifact": "propagation_metrics.csv",
                "guidance": "fp32",
            },
        },
        "externally_owned_inputs": instrumentor.externally_owned_inputs(),
        "externally_owned_outputs": instrumentor.externally_owned_outputs(),
        "guidance_head": "fp32",
        "bias_format": "fp32",
        "propagation": dict(PROPAGATION_A8_Q13),
        "coefficient_format": "signed_int16_q13",
        "propagation_accumulator": "int32",
        "fold_max_abs_error": max(
            float(reference_preparation["primary_max_abs_error"]),
            float(quantized_preparation["primary_max_abs_error"])),
        "folded_pairs": reference_preparation["folded_pairs"],
        "elapsed_seconds": time.time() - started,
    })

    reference_capture.close()
    quantized_capture.close()
    for policy in reversed(tuple(merge_adapters)):
        merge_adapters[policy].close()
    propagation.close()
    rotation.close()
    instrumentor.close()


if __name__ == "__main__":
    main()
