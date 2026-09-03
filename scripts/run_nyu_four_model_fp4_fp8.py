#!/usr/bin/env python3
"""Evaluate FP4/FP8 mixed precision under the unified FP16 propagation protocol."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from spn_quant.fp_formats import FORMAT_SPECS, make_quantizer
from spn_quant.fp_mixed_precision import (
    FPFormatAssignment,
    GROUP_NAMES,
    build_format_assignment,
    build_grouped_three_level_format_assignment,
    build_grouped_format_assignment,
    weighted_format_fractions,
)
from scripts.run_nyu_unified_fp16_task_aware_allocation import (
    MODEL_NAMES,
    _aggregate,
    _build_contract,
    _costs,
    _metric_row,
    _read_score_tables,
    _reference_metrics,
    _runtime_args,
    _settings,
    load_protocol,
    ordinary_score_tables,
    validate_protocol_payload,
    json_safe,
)
from scripts.run_nyu_model_p3t3_search import (
    HardDeploymentP3T3Evaluator,
    _site_boundary,
)
from scripts.nyu_model_runtime import NYUModelRuntime
from spn_quant import mixed_precision
from spn_quant.model_contracts import propagation_owned_modules


FORMAT_FP4 = "fp4_e2m1"
FORMAT_FP6 = "fp6_e3m2"
FORMAT_FP8 = "fp8_e4m3fn"
MIXED_FRACTIONS = (0.25, 0.50, 0.75)


@dataclass(frozen=True)
class FPFormatCandidate:
    name: str
    assignment: FPFormatAssignment


def _site_format_configuration(evaluator, assignment):
    generic = {}
    joint_attention = {}
    joint_concat = {}
    for owner, format_name in assignment.activation_formats.items():
        site_name, role = owner
        site = evaluator.sites[site_name]
        if site.role != role:
            raise ValueError("FP format activation role differs from contract")
        if site.owner_kind in ("module_input", "module_output"):
            boundary = _site_boundary(site)
            if evaluator.concat_adapter is not None and \
                    boundary[0] in evaluator.concat_adapter.consumer_modules:
                continue
            if boundary in generic and generic[boundary] != format_name:
                raise ValueError("one FP format boundary has multiple formats")
            generic[boundary] = format_name
        elif site.owner_kind == "attention_qkv":
            family, module_name, attention_role = site_name.split("::")
            if family != "attention" or "attention_%s" % attention_role != role:
                raise ValueError("invalid Attention FP format site")
            joint_attention.setdefault(module_name, {})[attention_role] = \
                format_name
        elif site.owner_kind == "concat_input":
            family, module_name, concat_role = site_name.split("::")
            if family != "concat" or "concat_%s" % concat_role != role:
                raise ValueError("invalid concat FP format site")
            branch = concat_role[:-len("_input")]
            joint_concat.setdefault(module_name, {})[branch] = format_name
        else:
            raise ValueError("unsupported FP format activation owner")
    return generic, joint_attention, joint_concat


class FPFormatEvaluator(HardDeploymentP3T3Evaluator):
    """Use calibrated FP4/FP8 fake quantization with FP16 propagation."""

    def _configure_candidate(self, candidate):
        assignment = candidate.assignment
        generic, attention_formats, joint_concat_formats = \
            _site_format_configuration(self, assignment)
        self.instrumentor.configure_floating_point(
            dict(assignment.weight_formats), generic,
            set(self.registry.blocks), external_output_ownership=True)
        self.instrumentor.set_runtime_statistics(False)
        self.instrumentor.relu_quantizers = {}

        if self.concat_adapter is not None:
            external_quantizers = {}
            for name in self.concat_adapter.consumer_modules:
                owner = ("activation::%s::input" % name, "module_input")
                format_name = assignment.activation_formats[owner]
                controller = self.concat_adapter.controllers[name]
                external_quantizers[name] = {
                    "transformer": make_quantizer(
                        format_name, controller.transformer_maximum),
                    "cnn": make_quantizer(
                        format_name, controller.cnn_maximum),
                    "output": make_quantizer(
                        format_name, controller.output_maximum),
                }
            self.concat_adapter.configure_floating_point(external_quantizers)

        if self.joint_adapter is not None:
            attention_quantizers = {}
            for name in self.joint_adapter.attention_names():
                controller = self.joint_adapter.attention_controllers[name]
                roles = attention_formats[name]
                attention_quantizers[name] = {}
                for role in ("q", "k", "v"):
                    attention_quantizers[name][role] = make_quantizer(
                        roles[role], controller.maxima[role],
                        broadcast_shape=(1, int(controller.num_heads), 1, 1))
            concat_quantizers = {}
            for name in self.joint_adapter.concat_names():
                controller = self.joint_adapter.concat_controllers[name]
                roles = joint_concat_formats[name]
                concat_quantizers[name] = {
                    "transformer": make_quantizer(
                        roles["transformer"], controller.transformer_maximum),
                    "cnn": make_quantizer(
                        roles["cnn"], controller.cnn_maximum),
                    "output": make_quantizer(
                        roles["transformer"], controller.output_maximum),
                }
            self.joint_adapter.configure_fp_formats(
                attention_quantizers, concat_quantizers)
        self.propagation_adapter.configure_fp16()


def _format_assignment(evaluator, weight_format, activation_format,
                        activation_scores, activation_costs,
                        promotion_fraction):
    return build_format_assignment(
        evaluator.contract, weight_format, activation_format,
        activation_scores, promotion_fraction, activation_costs)


def _format_score_gain(score_tables, names):
    if set(score_tables) != set(names):
        raise ValueError("FP format sensitivity coverage differs")
    return dict(
        (name, float(score_tables[name][4]) - float(score_tables[name][8]))
        for name in names)


def _group_budgets(payload):
    root = payload["fp_format_group_budgets"]
    if set(root) != {"weight_average_bits", "activation_average_bits",
                     "activation_minimum_fp8_fraction"}:
        raise KeyError("FP format group budget fields differ")
    if set(root["weight_average_bits"]) != set(GROUP_NAMES) or \
            set(root["activation_average_bits"]) != set(GROUP_NAMES) or \
            set(root["activation_minimum_fp8_fraction"]) != set(GROUP_NAMES):
        raise KeyError("FP format group budget coverage differs")
    budgets = {}
    for group in GROUP_NAMES:
        weight_bits = float(root["weight_average_bits"][group])
        activation_bits = float(root["activation_average_bits"][group])
        minimum_fp8_fraction = float(
            root["activation_minimum_fp8_fraction"][group])
        if not 4.0 <= weight_bits <= 8.0 or \
                not 4.0 <= activation_bits <= 8.0:
            raise ValueError("FP format group budget is outside [4, 8]")
        if not 0.0 <= minimum_fp8_fraction <= 1.0 or \
                4.0 + 4.0 * minimum_fp8_fraction > activation_bits:
            raise ValueError("FP format activation floor is infeasible")
        budgets[group] = {
            "weight_average_bits": weight_bits,
            "activation_average_bits": activation_bits,
            "activation_minimum_fp8_fraction": minimum_fp8_fraction,
        }
    return budgets


def _global_budgets(payload):
    budgets = payload["fp_format_global_budgets"]
    if set(budgets) != {"weight_average_bits", "activation_average_bits"}:
        raise KeyError("FP format global budget fields differ")
    result = dict((key, float(budgets[key])) for key in budgets)
    if any(not 4.0 <= value <= 8.0 for value in result.values()):
        raise ValueError("FP format global budget is outside [4, 8]")
    return result


def _format_budget(assignment, costs):
    weight_costs = dict(costs.weight_macs)
    activation_costs = dict(costs.activation_elements)
    weight_fractions = weighted_format_fractions(
        assignment.weight_formats, weight_costs)
    activation_fractions = weighted_format_fractions(
        assignment.activation_formats, activation_costs)
    weight_bits = sum(
        float(fraction) * float(FORMAT_SPECS[name].bits)
        for name, fraction in weight_fractions.items())
    activation_bits = sum(
        float(fraction) * float(FORMAT_SPECS[name].bits)
        for name, fraction in activation_fractions.items())
    return weight_bits, activation_bits, weight_fractions, activation_fractions


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RuntimeError("cannot write empty result CSV: %s" % path)
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise RuntimeError("result row schema differs: %s" % path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_model(payload, model_name, output_root):
    spec = payload["models"][model_name]
    hard = payload["hard_deployment"]
    device = str(spec["device"])
    runtime = NYUModelRuntime.from_args(_runtime_args(spec, device))
    model = runtime.build_model(runtime.device)
    contract = _build_contract(model_name, model)
    costs = _costs(model_name, spec, model, runtime)
    settings = _settings(spec, hard, device)
    evaluator = FPFormatEvaluator(
        runtime, model, contract,
        mixed_precision.build_registry(contract, costs), settings)
    reference = _reference_metrics(evaluator)
    weight_scores, activation_scores = _read_score_tables(spec)
    weight_scores, activation_scores = ordinary_score_tables(
        contract, costs, weight_scores, activation_scores)
    weight_score_tables = weight_scores
    activation_score_tables = activation_scores
    activation_scores = dict(
        (owner, float(scores[4]) - float(scores[8]))
        for owner, scores in activation_scores.items())
    activation_costs = dict(costs.activation_elements)
    group_budgets = _group_budgets(payload)
    global_budgets = _global_budgets(payload)
    grouped_assignment, grouped_audit = build_grouped_format_assignment(
        contract,
        _format_score_gain(weight_scores, dict(costs.weight_macs)),
        activation_scores,
        dict(costs.weight_macs),
        activation_costs,
        group_budgets,
        global_budgets)
    three_level_assignment, three_level_audit = \
        build_grouped_three_level_format_assignment(
            contract,
            weight_score_tables,
            activation_score_tables,
            dict(costs.weight_macs),
            activation_costs,
            group_budgets,
            global_budgets)
    model_output = output_root / model_name
    model_output.mkdir(parents=True)

    candidates = []
    for name, weight_format, activation_format, fraction in (
            ("UNIFORM_FP8W_FP8A", FORMAT_FP8, FORMAT_FP8, 0.0),
            ("UNIFORM_FP6W_FP6A", FORMAT_FP6, FORMAT_FP6, 0.0),
            ("UNIFORM_FP4W_FP4A", FORMAT_FP4, FORMAT_FP4, 0.0),
            ("UNIFORM_FP4W_FP6A", FORMAT_FP4, FORMAT_FP6, 0.0),
            ("UNIFORM_FP6W_FP4A", FORMAT_FP6, FORMAT_FP4, 0.0),
            ("UNIFORM_FP4W_FP8A", FORMAT_FP4, FORMAT_FP8, 0.0),
            ("UNIFORM_FP8W_FP4A", FORMAT_FP8, FORMAT_FP4, 0.0)):
        candidates.append(FPFormatCandidate(
            name, _format_assignment(
                evaluator, weight_format, activation_format,
                activation_scores, activation_costs, fraction)))
    for fraction in MIXED_FRACTIONS:
        name = "MIXED_FP4W_FP4A_FP8A_%02d" % int(fraction * 100.0)
        candidates.append(FPFormatCandidate(
            name, _format_assignment(
                evaluator, FORMAT_FP4, FORMAT_FP4, activation_scores,
                activation_costs, fraction)))
    candidates.append(FPFormatCandidate(
        "GROUPED_FP4_FP8_BUDGETED", grouped_assignment))
    candidates.append(FPFormatCandidate(
        "GROUPED_FP4_FP6_FP8_BUDGETED", three_level_assignment))

    rows = []
    pareto = []
    assignment_payloads = {}
    for candidate in candidates:
        metrics = _aggregate(evaluator._evaluate_candidate(candidate))
        weight_bits, activation_bits, weight_fractions, activation_fractions = \
            _format_budget(candidate.assignment, costs)
        row = {
            "model": model_name,
            "configuration": candidate.name,
            "pooled_rmse": metrics["pooled_rmse"],
            "mean_sample_rmse": metrics["mean_sample_rmse"],
            "delta_vs_fp32": metrics["pooled_rmse"] - reference["pooled_rmse"],
            "relative_fp_loss": metrics["pooled_rmse"] /
            reference["pooled_rmse"] - 1.0,
            "valid": metrics["valid"],
            "average_weight_bits": weight_bits,
            "average_activation_bits": activation_bits,
        }
        rows.append(row)
        pareto.append(dict(row))
        assignment_payloads[candidate.name] = {
            "weight_formats": dict(candidate.assignment.weight_formats),
            "activation_formats": {
                "%s::%s" % owner: format_name
                for owner, format_name in
                candidate.assignment.activation_formats.items()
            },
            "weight_format_fractions": weight_fractions,
            "activation_format_fractions": activation_fractions,
        }
        if candidate.name == "GROUPED_FP4_FP8_BUDGETED":
            assignment_payloads[candidate.name]["group_audit"] = grouped_audit
        if candidate.name == "GROUPED_FP4_FP6_FP8_BUDGETED":
            assignment_payloads[candidate.name]["group_audit"] = \
                three_level_audit
    _write_csv(model_output / "summary.csv", rows)
    _write_csv(model_output / "pareto.csv", pareto)
    artifact = {
        "model": model_name,
        "device": device,
        "propagation_dtype": "fp16",
        "propagation_owned_modules": list(propagation_owned_modules(contract)),
        "calibration_count": int(spec["calibration_count"]),
        "evaluation_indices": list(spec["evaluation_indices"]),
        "formats": [FORMAT_FP4, FORMAT_FP6, FORMAT_FP8],
        "global_budgets": global_budgets,
        "reference": reference,
        "results": rows,
        "assignments": assignment_payloads,
        "group_budgets": group_budgets,
    }
    (model_output / "manifest.json").write_text(
        json.dumps(json_safe(artifact), indent=2, sort_keys=True,
                   allow_nan=False) + "\n", encoding="utf-8")
    evaluator.close()
    runtime.close()
    return model_output


def _payload_for_output(config: Path, output_root: Path):
    payload = dict(load_protocol(config))
    payload["output_root"] = str(output_root)
    return payload


def run_model(config: Path, model_name: str, output_root: Path) -> Path:
    payload = _payload_for_output(config, output_root)
    if model_name not in MODEL_NAMES:
        raise ValueError("unknown model: %s" % model_name)
    if not output_root.is_dir():
        raise FileNotFoundError("FP4/FP8 output root is missing: %s" % output_root)
    return _run_model(payload, model_name, output_root)


def run(config: Path, output_root: Path) -> Path:
    config_path = Path(config).resolve()
    payload = _payload_for_output(config_path, output_root)
    if output_root.exists():
        raise FileExistsError("FP4/FP8 output already exists: %s" % output_root)
    output_root.mkdir(parents=True)
    for model_name in MODEL_NAMES:
        spec = payload["models"][model_name]
        environment = dict(os.environ)
        environment["SPN_EXTERNAL_ROOT"] = payload["external_root"]
        environment["COMPLETIONFORMER_ROOT"] = payload["completionformer_root"]
        environment["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + \
            environment["PYTHONPATH"]
        subprocess.run(
            (str(spec["python_executable"]), str(Path(__file__)),
             "--config", str(config_path), "--output-root", str(output_root),
             "--model", model_name),
            cwd=str(spec["data_root"]), env=environment, check=True)
    return output_root


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_NAMES)
    return parser


def main():
    args = build_parser().parse_args(tuple(sys.argv[1:]))
    print(run_model(args.config, args.model, args.output_root)
          if args.model is not None else run(args.config, args.output_root))


if __name__ == "__main__":
    main()
