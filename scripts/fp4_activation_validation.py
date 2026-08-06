#!/usr/bin/env python3
"""Controlled E2M1 activation-validation configuration and contracts."""

from __future__ import division

import re

import numpy as np


FP4_CONFIG_NAMES = (
    "FP32",
    "FP4V_W8A4",
    "FP4V_W8E2M1",
    "FP4V_W8A8",
    "FP4V_W4A4",
    "FP4V_W4E2M1",
    "FP4V_W4A8",
)


SEMANTIC_A8_RULES = {
    "cspn": (
        ("sparse_depth_input", "input", r"^conv1_1$"),
        ("initial_depth", "output", r"^gud_up_proj_layer5\.conv1$"),
        ("guidance", "output", r"^gud_up_proj_layer6\.conv1$"),
    ),
    "dyspn": (
        ("sparse_depth_input", "input", r"^base\.conv1_dep\.0$"),
        ("combined_prop_inputs", "output",
         r"^base\.gd_dec0_dyspn_\d+_\d+\.0$"),
    ),
    "nlspn": (
        ("sparse_depth_input", "input", r"^conv1_dep\.0$"),
        ("initial_depth", "output", r"^id_dec0\.0$"),
        ("guidance", "output", r"^gd_dec0\.0$"),
        ("confidence", "output", r"^cf_dec0\.0$"),
    ),
    "completionformer": (
        ("sparse_depth_input", "input", r"^backbone\.conv1_dep\.0$"),
        ("initial_depth", "output", r"^backbone\.dep_dec0\.0$"),
        ("guidance", "output", r"^backbone\.gd_dec0\.0$"),
        ("confidence", "output", r"^backbone\.cf_dec0\.0$"),
    ),
}


def _propagation_a8():
    return {
        "affinity_bits": 8,
        "confidence_bits": 8,
        "offset_bits": 8,
        "state_bits": 8,
        "coefficient_fraction_bits": 13,
    }


def _validation_config(name, groups, weight_bits, activation_bits,
                       activation_mode):
    return {
        "name": name,
        "w_bits": int(weight_bits),
        "a_bits": int(activation_bits),
        "groups": set(groups),
        "state_bits": None,
        "propagation": _propagation_a8(),
        "external_output_ownership": True,
        "activation_mode": activation_mode,
        "quantize_bias": False,
    }


def build_fp4_validation_configurations(groups):
    all_groups = set(groups)
    return [
        {
            "name": "FP32",
            "w_bits": None,
            "a_bits": None,
            "groups": set(),
            "state_bits": None,
            "propagation": None,
            "external_output_ownership": True,
            "activation_mode": "uniform",
            "quantize_bias": False,
        },
        _validation_config("FP4V_W8A4", all_groups, 8, 4, "uniform"),
        _validation_config("FP4V_W8E2M1", all_groups, 8, 4, "e2m1"),
        _validation_config("FP4V_W8A8", all_groups, 8, 8, "uniform"),
        _validation_config("FP4V_W4A4", all_groups, 4, 4, "uniform"),
        _validation_config("FP4V_W4E2M1", all_groups, 4, 4, "e2m1"),
        _validation_config("FP4V_W4A8", all_groups, 4, 8, "uniform"),
    ]


def resolve_semantic_a8_overrides(model_name, module_names):
    if model_name not in SEMANTIC_A8_RULES:
        raise ValueError("unknown FP4 validation model: %s" % model_name)
    names = sorted(set(module_names))
    bit_overrides = {}
    format_overrides = {}
    manifest = []
    for role, kind, pattern in SEMANTIC_A8_RULES[model_name]:
        expression = re.compile(pattern)
        matches = [name for name in names if expression.search(name)]
        if len(matches) != 1:
            raise RuntimeError(
                "%s semantic boundary %s matched %d modules: %s" %
                (model_name, role, len(matches), matches))
        module = matches[0]
        key = (module, kind)
        bit_overrides[key] = 8
        format_overrides[key] = "uniform"
        manifest.append({
            "model": model_name,
            "role": role,
            "module": module,
            "kind": kind,
            "bits": 8,
            "format": "uniform",
        })
    return bit_overrides, format_overrides, manifest


def paired_bootstrap_rmse_difference(int4_rmse, e2m1_rmse, resamples, seed):
    int4 = np.asarray(int4_rmse, dtype=np.float64)
    e2m1 = np.asarray(e2m1_rmse, dtype=np.float64)
    if int4.shape != e2m1.shape:
        raise ValueError("paired RMSE arrays must have identical shape")
    if int4.ndim != 1 or int4.size == 0:
        raise ValueError("paired RMSE arrays must be non-empty vectors")
    if not np.isfinite(int4).all() or not np.isfinite(e2m1).all():
        raise ValueError("paired RMSE arrays must be finite")
    if int(resamples) <= 0:
        raise ValueError("bootstrap resamples must be positive")

    differences = e2m1 - int4
    generator = np.random.RandomState(int(seed))
    indices = generator.randint(
        0, differences.size, size=(int(resamples), differences.size))
    bootstrap_means = differences[indices].mean(axis=1)
    return {
        "mean_difference": float(differences.mean()),
        "ci_lower": float(np.percentile(bootstrap_means, 2.5)),
        "ci_upper": float(np.percentile(bootstrap_means, 97.5)),
        "samples": int(differences.size),
    }


def a4_to_a8_recovery(int4_rmse, e2m1_rmse, a8_rmse):
    values = np.asarray(
        [int4_rmse, e2m1_rmse, a8_rmse], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("recovery RMSE values must be finite")
    denominator = float(int4_rmse) - float(a8_rmse)
    if denominator <= 0.0:
        return None
    return ((float(int4_rmse) - float(e2m1_rmse)) / denominator)
