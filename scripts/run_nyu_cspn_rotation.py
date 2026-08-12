#!/usr/bin/env python3
"""Evaluate selective CSPN channel rotation under W4A4 quantization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.nyu_quantization_analysis import classify_module  # noqa: E402


PROPAGATION_A8_Q13 = {
    "affinity_bits": 8,
    "confidence_bits": 8,
    "offset_bits": 8,
    "state_bits": 8,
    "coefficient_fraction_bits": 13,
}

END_TO_END_FIELDS = (
    "model", "config", "sample_index", "RMSE", "MAE", "ABS_REL",
    "IRMSE", "flat_RMSE", "boundary_RMSE", "nonfinite_ratio",
)

BOUNDARY_FIELDS = (
    "model", "config", "boundary", "method", "bits", "group_size",
    "minimum", "maximum", "p75", "p99", "p99_9", "p99_99",
    "kurtosis", "channel_imbalance", "sqnr", "zero_code_ratio",
    "saturation_ratio", "block_output_mse", "block_output_sqnr",
)


def _quantized_configuration(
        name: str, decoder_entry: str, layer4_signed_skip: str,
        group_size: Optional[int]) -> Dict[str, object]:
    return {
        "name": name,
        "w_bits": 4,
        "a_bits": 4,
        "enabled_groups": {"encoder", "decoder", "depth_head"},
        "rotation_methods": {
            "decoder_entry": decoder_entry,
            "layer4_signed_skip": layer4_signed_skip,
        },
        "group_size": group_size,
        "quantize_bias": False,
        "propagation": dict(PROPAGATION_A8_Q13),
    }


def build_configurations(group_size: int) -> Sequence[Dict[str, object]]:
    group_size = int(group_size)
    return (
        {
            "name": "FP32",
            "w_bits": None,
            "a_bits": None,
            "enabled_groups": set(),
            "rotation_methods": {
                "decoder_entry": "identity",
                "layer4_signed_skip": "identity",
            },
            "group_size": None,
            "quantize_bias": False,
            "propagation": None,
        },
        _quantized_configuration(
            "RTN_W4A4", "identity", "identity", None),
        _quantized_configuration(
            "GROUP_W4A4", "identity", "identity", group_size),
        _quantized_configuration(
            "RANDOM_decoder_entry", "random", "identity", None),
        _quantized_configuration(
            "RANDOM_layer4_signed_skip", "identity", "random", None),
        _quantized_configuration(
            "RANDOM_both", "random", "random", None),
        _quantized_configuration(
            "HADAMARD_decoder_entry", "hadamard", "identity", None),
        _quantized_configuration(
            "HADAMARD_layer4_signed_skip", "identity", "hadamard", None),
        _quantized_configuration(
            "HADAMARD_both", "hadamard", "hadamard", None),
        _quantized_configuration(
            "HADAMARD_GROUP_both", "hadamard", "hadamard", group_size),
    )


def cspn_quant_group(name: str, module: nn.Module) -> Optional[str]:
    if name.startswith("gud_up_proj_layer6"):
        return None
    return classify_module("cspn", name, module)


def rotation_owned_inputs():
    return {
        "gud_up_proj_layer1.conv1",
        "gud_up_proj_layer1.sc_conv1",
        "gud_up_proj_layer4.conv1_1",
    }


def rotation_owned_outputs():
    return {"conv1_1", "conv2", "gud_up_proj_layer5.conv1"}


def validate_fp_equivalence(reference: torch.Tensor,
                            candidate: torch.Tensor, site: str) -> None:
    if reference.shape != candidate.shape or not torch.allclose(
            reference, candidate, rtol=1e-4, atol=1e-5):
        maximum = float((candidate - reference).abs().max().item()) \
            if reference.shape == candidate.shape else float("inf")
        raise RuntimeError(
            "FP equivalence failed at %s: max_abs_error=%.8f" %
            (site, maximum))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--sample-metrics", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--out-dir",
        default="profile_logs/nyu_cspn_rotation_w4a4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--group-size", type=int, choices=(16, 32, 64),
                        default=32)
    return parser.parse_args(argv)


def main(argv=None):
    parse_args(argv)
    raise RuntimeError("CSPN rotation evaluation is not implemented")


if __name__ == "__main__":
    main()
