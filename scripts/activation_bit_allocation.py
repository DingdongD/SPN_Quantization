#!/usr/bin/env python3
"""Sensitivity selection and accounting for mixed activation precision."""

from __future__ import division

import math


GROUP_ORDER = ("encoder", "attention", "decoder", "depth_head",
               "propagation_head")


def select_sensitive_modules(rows, limit=4,
                             config="HW_W4A4_MinMax"):
    lowest = {}
    groups = {}
    for row in rows:
        if row.get("config") != config or row.get("kind") != "input":
            continue
        sqnr = float(row["sqnr_db"])
        if not math.isfinite(sqnr):
            continue
        module = row["module"]
        lowest[module] = min(sqnr, lowest.get(module, float("inf")))
        groups[module] = row["group"]
    ordered = sorted(lowest, key=lambda module: (lowest[module], module))
    return [{
        "module": module,
        "group": groups[module],
        "sqnr_db": lowest[module],
    } for module in ordered[:int(limit)]]


def _configuration(name, all_groups, w_bits=4, a_bits=4, state_bits=None,
                   overrides=None, selection="baseline"):
    overrides = dict(overrides or {})
    return {
        "name": name,
        "w_bits": int(w_bits),
        "a_bits": int(a_bits),
        "groups": set(all_groups),
        "state_bits": state_bits,
        "activation_bit_overrides": overrides,
        "selection": selection,
        "selected_modules": sorted(overrides),
    }


def build_mixed_configurations(module_groups, candidates):
    module_groups = dict(module_groups)
    all_groups = set(module_groups.values())
    configs = [{
        "name": "FP32", "w_bits": None, "a_bits": None,
        "groups": set(), "state_bits": None,
        "activation_bit_overrides": {}, "selection": "fp32",
        "selected_modules": [],
    }]
    configs.append(_configuration("MP_W4A4_base", all_groups))
    configs.append(_configuration(
        "MP_W4A8_full", all_groups, a_bits=8, selection="full_a8"))
    configs.append(_configuration(
        "MP_W8A8_full", all_groups, w_bits=8, a_bits=8,
        selection="full_w8a8"))

    for index, candidate in enumerate(candidates, 1):
        configs.append(_configuration(
            "MP_site%02d_A8" % index, all_groups,
            overrides={candidate["module"]: 8}, selection="single_site"))

    for group in GROUP_ORDER:
        modules = sorted(name for name, current in module_groups.items()
                         if current == group)
        if modules:
            configs.append(_configuration(
                "MP_%s_A8" % group, all_groups,
                overrides=dict((name, 8) for name in modules),
                selection="group"))

    heads = sorted(name for name, group in module_groups.items()
                   if group in ("depth_head", "propagation_head"))
    if heads:
        configs.append(_configuration(
            "MP_heads_A8", all_groups,
            overrides=dict((name, 8) for name in heads),
            selection="heads"))
    if candidates:
        configs.append(_configuration(
            "MP_top%d_A8" % len(candidates), all_groups,
            overrides=dict((row["module"], 8) for row in candidates),
            selection="topk"))

    configs.append(_configuration(
        "MP_W4A4_stateA16", all_groups, state_bits=16,
        selection="state"))
    configs.append(_configuration(
        "MP_W4A4_stateA8", all_groups, state_bits=8,
        selection="state"))
    return configs


def config_manifest_rows(configs):
    rows = []
    for config in configs:
        modules = config.get("selected_modules", []) or [""]
        for module in modules:
            rows.append({
                "config": config["name"],
                "selection": config.get("selection", ""),
                "module": module,
                "activation_bits": config.get(
                    "activation_bit_overrides", {}).get(
                        module, config.get("a_bits")),
                "default_activation_bits": config.get("a_bits"),
                "weight_bits": config.get("w_bits"),
                "state_bits": config.get("state_bits"),
            })
    return rows


def _nonfinite_rates(rows):
    totals = {}
    for row in rows:
        key = (row["model"], row["config"])
        bad, finite = totals.setdefault(key, [0, 0])
        totals[key][0] = bad + int(float(row.get("nonfinite_pixels", 0)))
        totals[key][1] = finite + int(float(row.get("num_pixels", 0)))
    return dict((key, bad / float(bad + finite) if bad + finite else 0.0)
                for key, (bad, finite) in totals.items())


def _optional_int(value):
    if value in (None, ""):
        return None
    return int(float(value))


def allocation_summary_rows(regional_rows, sample_rows, manifest_rows,
                            layer_rows):
    rates = _nonfinite_rates(sample_rows)
    all_regions = dict(((row["model"], row["config"]), row)
                       for row in regional_rows if row.get("region") == "all")
    models = sorted(set(row["model"] for row in regional_rows))
    output = []
    for model in models:
        base_key = (model, "MP_W4A4_base")
        if base_key not in all_regions:
            continue
        base_rmse = float(all_regions[base_key]["RMSE"])
        base_invalid = rates.get(base_key, 0.0)
        module_numel = {}
        for row in layer_rows:
            if row.get("model") != model \
                    or row.get("config") != "MP_W4A4_base" \
                    or row.get("kind") not in ("input", "output"):
                continue
            module = row["module"]
            module_numel[module] = module_numel.get(module, 0.0) + \
                float(row["numel"])
        total_numel = sum(module_numel.values())
        specs = {}
        for row in manifest_rows:
            if row.get("model") not in (None, "", model):
                continue
            config = row["config"]
            spec = specs.setdefault(config, {
                "selection": row.get("selection", ""),
                "default_bits": _optional_int(
                    row.get("default_activation_bits")),
                "state_bits": _optional_int(row.get("state_bits")),
                "overrides": {},
            })
            module = row.get("module", "")
            if module:
                spec["overrides"][module] = _optional_int(
                    row.get("activation_bits"))
        for (current_model, config), metrics in sorted(all_regions.items()):
            if current_model != model or config not in specs:
                continue
            spec = specs[config]
            default_bits = spec["default_bits"]
            if default_bits is None:
                traffic = float("nan")
                extra = float("nan")
            else:
                traffic = total_numel * default_bits
                for module, bits in spec["overrides"].items():
                    traffic += module_numel.get(module, 0.0) * \
                        (bits - default_bits)
                extra = traffic - total_numel * 4.0
            rmse = float(metrics["RMSE"])
            invalid = rates.get((model, config), 0.0)
            cost_gbit = extra / 1e9 if math.isfinite(extra) else float("nan")
            benefit = base_invalid - invalid if base_invalid > 0.0 \
                else base_rmse - rmse
            output.append({
                "model": model,
                "config": config,
                "selection": spec["selection"],
                "selected_modules": len(spec["overrides"]),
                "default_activation_bits": default_bits,
                "state_bits": spec["state_bits"],
                "RMSE": rmse,
                "MAE": float(metrics["MAE"]),
                "ABS_REL": float(metrics["ABS_REL"]),
                "nonfinite_rate": invalid,
                "rmse_gain": base_rmse - rmse,
                "invalid_rate_reduction": base_invalid - invalid,
                "activation_bit_traffic": traffic,
                "extra_activation_bits": extra,
                "extra_activation_ratio": (
                    extra / (total_numel * 4.0) if total_numel else float("nan")),
                "benefit_per_extra_gbit": (
                    benefit / cost_gbit if math.isfinite(cost_gbit)
                    and cost_gbit > 0.0 else float("nan")),
            })
    return output


def rank_allocation_rows(rows):
    output = []
    models = sorted(set(row["model"] for row in rows))
    for model in models:
        points = [dict(row) for row in rows if row["model"] == model]
        points.sort(key=lambda row: (
            float(row["nonfinite_rate"]), float(row["RMSE"]),
            float(row.get("extra_activation_bits", float("inf")))))
        for rank, row in enumerate(points, 1):
            row["rank"] = rank
            output.append(row)
    return output
