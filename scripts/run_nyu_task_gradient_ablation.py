#!/usr/bin/env python3
"""Run ranked single-module W4A4 ablations and task-aware mixed allocation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.nyu_model_runtime import NYUModelRuntime  # noqa: E402
from scripts.run_nyu_four_model_propagation_dtype_ablation import (  # noqa: E402
    _build_cspn_uniform_contract,
    _cspn_activation_cost_rows,
    _cspn_weight_cost_rows,
    _runtime_args,
)
from scripts.run_nyu_model_p3t3_search import (  # noqa: E402
    HardDeploymentP3T3Evaluator,
    HardDeploymentSettings,
    _read_activation_cost_rows,
    _read_weight_cost_rows,
)
from spn_quant import mixed_precision  # noqa: E402
from spn_quant.model_contracts import build_model_quantization_contract  # noqa: E402
from spn_quant.qdrop_targets import resolve_qdrop_targets  # noqa: E402
from spn_quant.task_aware_allocation import (  # noqa: E402
    boundary_budget_allocation,
)


MODEL_NAMES = ("cspn", "dyspn", "nlspn", "completionformer")
BIT_LEVELS = (4, 6, 8)


class FloatPropagationEvaluator(HardDeploymentP3T3Evaluator):
    """Keep the propagation loop in FP32 while quantizing its projection W8A8."""

    def _configure_candidate(self, candidate):
        super()._configure_candidate(candidate)
        self.propagation_adapter.configure_float("fp32")


def _load_rows(path: Path, key: str):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        raise ValueError("sensitivity CSV is empty: %s" % path)
    if any(key not in row for row in rows):
        raise ValueError("sensitivity CSV lacks %s: %s" % (key, path))
    return rows


def _score_table(rows, key: str):
    table = {}
    for row in rows:
        name = str(row[key])
        bits = int(row["bits"])
        if bits not in BIT_LEVELS:
            raise ValueError("unsupported sensitivity bit level")
        if name in table and bits in table[name]:
            raise ValueError("duplicate sensitivity row: %s/%d" % (name, bits))
        table.setdefault(name, {})[bits] = float(
            row["normalized_gradient_weighted_error"])
    for name, scores in table.items():
        if set(scores) != set(BIT_LEVELS):
            raise ValueError("incomplete sensitivity scores: %s" % name)
    return table


def _ranked_modules(weight_scores, activation_scores, count: int):
    if int(count) <= 0:
        raise ValueError("top module count must be positive")
    names = tuple(sorted(set(weight_scores) | set(activation_scores)))
    weight_rank = dict((name, rank) for rank, name in enumerate(sorted(
        weight_scores, key=lambda name: (-weight_scores[name][4], name)), 1))
    activation_rank = dict((name, rank) for rank, name in enumerate(sorted(
        activation_scores, key=lambda name: (-activation_scores[name][4], name)), 1))
    ranked = tuple(sorted(
        names,
        key=lambda name: (
            weight_rank.get(name, len(names) + 1) +
            activation_rank.get(name, len(names) + 1),
            min(weight_rank.get(name, len(names) + 1),
                activation_rank.get(name, len(names) + 1)),
            name,
        )))
    return ranked[:int(count)]


def _module_from_owner(owner):
    site, role = owner
    del role
    parts = str(site).split("::")
    if len(parts) == 3 and parts[0] == "activation" and \
            parts[2] in ("input", "output"):
        return parts[1]
    return None


def _assignment(contract, weight_bits, activation_bits):
    return mixed_precision.BitAssignment(
        weight_bits=tuple((name, bits) for name, bits in weight_bits.items()),
        activation_bits=tuple((owner, bits)
                              for owner, bits in activation_bits.items()),
        model_name=contract.model_name,
    )


def _candidate(contract, weight_bits, activation_bits, name, stage):
    assignment = _assignment(contract, weight_bits, activation_bits)
    return mixed_precision.P3T3Candidate(
        name=name,
        stage=stage,
        prefix=(),
        tail=(),
        promoted_blocks=(),
        assignment=assignment,
    )


def _aggregate(rows):
    pixels = sum(int(row["valid_pixels"]) for row in rows)
    if pixels <= 0:
        raise ValueError("evaluation rows contain no valid pixels")
    squared = sum(float(row["squared_error_sum"]) for row in rows)
    finite = all(bool(row["prediction_finite"]) and
                 bool(row["propagation_valid"]) and
                 bool(row["reproducible"]) and
                 math.isfinite(float(row["squared_error_sum"]))
                 for row in rows)
    pooled = math.sqrt(squared / float(pixels)) if finite else float("inf")
    mean = sum(float(row["RMSE"]) for row in rows) / len(rows) \
        if finite else float("inf")
    return {
        "pooled_rmse": pooled,
        "mean_sample_rmse": mean,
        "valid_pixels": pixels,
        "valid": finite,
        "prediction_finite": all(bool(row["prediction_finite"]) for row in rows),
        "propagation_valid": all(bool(row["propagation_valid"]) for row in rows),
        "reproducible": all(bool(row["reproducible"]) for row in rows),
    }


def _json_metric(value):
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _reference_metrics(evaluator):
    evaluator.instrumentor.disable()
    if evaluator.propagation_projection_instrumentor is not None:
        evaluator.propagation_projection_instrumentor.disable()
    evaluator.propagation_adapter.disable()
    if evaluator.joint_adapter is not None:
        evaluator.joint_adapter.unbind_qdrop_sites()
    with torch.no_grad():
        prediction, ground_truth = evaluator._forward(evaluator.evaluation_batch)
    rows = []
    for position, (sample_index, batch) in enumerate(evaluator.evaluation_batches):
        del batch
        sample_prediction = prediction[position]
        sample_ground_truth = ground_truth[position]
        valid = torch.isfinite(sample_ground_truth) & (sample_ground_truth > 1e-4)
        pixels = int(valid.sum().item())
        if pixels <= 0:
            raise ValueError("reference sample has no valid depth pixels")
        values = sample_prediction[valid]
        if not bool(torch.isfinite(values).all().item()):
            raise RuntimeError("FP32 reference prediction is non-finite")
        if bool((values <= 1e-4).any().item()):
            raise RuntimeError("FP32 reference prediction is non-positive")
        difference = values.double() - sample_ground_truth[valid].double()
        squared = float(difference.square().sum().item())
        rows.append({
            "sample_index": int(sample_index),
            "squared_error_sum": squared,
            "valid_pixels": pixels,
            "RMSE": math.sqrt(squared / float(pixels)),
            "prediction_finite": True,
            "propagation_valid": True,
            "reproducible": True,
        })
    return _aggregate(rows)


def _settings(spec, hard, device):
    return HardDeploymentSettings(
        device=device,
        calibration_metadata=Path(spec["calibration_metadata"]),
        calibration_count=int(spec["calibration_count"]),
        evaluation_indices=tuple(int(index) for index in spec["evaluation_indices"]),
        base_weight_bits=8,
        base_activation_bits=8,
        promotion_weight_bits=8,
        promotion_activation_bits=8,
        fold_conv_bn=bool(hard["fold_conv_bn"]),
        fold_max_error=float(hard["fold_max_error"]),
        joint_clip_factors=tuple(float(value) for value in hard["joint_clip_factors"]),
        joint_search_rounds=int(hard["joint_search_rounds"]),
        joint_cache_sample_limit=int(hard["joint_cache_sample_limit"]),
        joint_cache_byte_limit=int(hard["joint_cache_byte_limit"]),
    )


def _costs(model_name, spec, model, runtime):
    if model_name == "cspn":
        plan = resolve_qdrop_targets("cspn", model)
        weight_rows = _cspn_weight_cost_rows(
            plan, model, runtime, Path(spec["calibration_metadata"]),
            int(spec["calibration_count"]),
            tuple(int(index) for index in spec["evaluation_indices"]))
        activation_rows = _cspn_activation_cost_rows(
            plan, model, runtime, Path(spec["calibration_metadata"]),
            int(spec["calibration_count"]),
            tuple(int(index) for index in spec["evaluation_indices"]))
    else:
        weight_rows = _read_weight_cost_rows(Path(spec["weight_cost_rows"]))
        activation_rows = _read_activation_cost_rows(Path(spec["activation_cost_rows"]))
    return mixed_precision.CostBasis(
        weight_macs=weight_rows, activation_elements=activation_rows)


def _build_contract(model_name, model):
    if model_name == "cspn":
        return _build_cspn_uniform_contract(model)
    return build_model_quantization_contract(model_name, model)


def _allocation(contract, costs, weight_scores, activation_scores,
                measured, weight_budget, activation_budget, max_delta,
                boundary_bits):
    weight_costs = dict(costs.weight_macs)
    activation_costs = dict(costs.activation_elements)
    measured_by_module = dict((row["module"], row) for row in measured)
    fixed_weight = {}
    fixed_activation = {}
    for module, row in measured_by_module.items():
        if not bool(row["valid"]) or float(row["delta_vs_w8a8_pooled_rmse"]) > max_delta:
            fixed_weight[module] = 8
            owners = tuple(owner for owner in activation_costs
                           if _module_from_owner(owner) == module)
            for owner in owners:
                fixed_activation[owner] = 8

    activation_unit_scores = {}
    fixed_unmapped = []
    for owner in activation_costs:
        module = _module_from_owner(owner)
        if module is None:
            fixed_activation[owner] = 8
            fixed_unmapped.append(owner)
        elif module not in activation_scores:
            fixed_activation[owner] = 8
            fixed_unmapped.append(owner)
        else:
            activation_unit_scores[owner] = activation_scores[module]
    all_activation_scores = dict(activation_unit_scores)
    all_activation_scores.update((owner, {4: 0.0, 6: 0.0, 8: 0.0})
                                 for owner in fixed_activation)
    weight_assignment = dict(boundary_budget_allocation(
        weight_costs, weight_scores, weight_budget, fixed_weight,
        boundary_bits))
    activation_assignment = dict(boundary_budget_allocation(
        activation_costs, all_activation_scores, activation_budget,
        fixed_activation, boundary_bits))
    assignment = _assignment(contract, weight_assignment, activation_assignment)
    return assignment, tuple(fixed_weight), tuple(fixed_activation), tuple(fixed_unmapped)


def run(config_path: Path, sensitivity_root: Path, model_name: str,
        device: str, output: Path, top_modules: int, weight_budget: float,
        activation_budget: float, max_ablation_delta: float,
        boundary_bits: int, ablation_bits: int):
    if model_name not in MODEL_NAMES:
        raise ValueError("unknown model: %s" % model_name)
    if int(ablation_bits) not in (4, 6):
        raise ValueError("ablation bits must be 4 or 6")
    if output.exists():
        raise FileExistsError("ablation output already exists: %s" % output)
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    spec = payload["models"][model_name]
    hard = payload["hard_deployment"]
    runtime = NYUModelRuntime.from_args(_runtime_args(spec, device))
    model = runtime.build_model(runtime.device)
    contract = _build_contract(model_name, model)
    costs = _costs(model_name, spec, model, runtime)
    weight_scores = _score_table(
        _load_rows(Path(sensitivity_root) / model_name / "weight_sensitivity.csv",
                   "module"),
        "module")
    activation_scores = _score_table(
        _load_rows(Path(sensitivity_root) / model_name / "module_sensitivity.csv",
                   "module"),
        "module")
    weight_cost_names = set(name for name, value in costs.weight_macs)
    weight_scores = dict((name, scores) for name, scores in weight_scores.items()
                         if name in weight_cost_names)
    activation_scores = dict(
        (name, scores) for name, scores in activation_scores.items()
        if name in set(contract.weight_modules))
    ranked = _ranked_modules(weight_scores, activation_scores, top_modules)
    registry = mixed_precision.build_registry(contract, costs)
    evaluator = FloatPropagationEvaluator(
        runtime, model, contract, registry, _settings(spec, hard, device),
        propagation_projection_precision=(
            int(hard["propagation_projection_weight_bits"]),
            int(hard["propagation_projection_activation_bits"])))
    output.mkdir(parents=True)
    try:
        reference_metrics = _reference_metrics(evaluator)
        reference_pooled = reference_metrics["pooled_rmse"]
        base_weights = dict((name, 8) for name in contract.weight_modules)
        base_activations = dict(((owner, role), 8)
                                for block in contract.blocks
                                for owner, role in block.activation_owners)
        baseline = _candidate(contract, base_weights, base_activations,
                              "UNIFORM_W8A8_FP32_PROP", "baseline")
        baseline_rows = evaluator._evaluate_candidate(baseline)
        baseline_metrics = _aggregate(baseline_rows)
        if not baseline_metrics["valid"]:
            raise RuntimeError("W8A8 FP32 propagation baseline is invalid")
        w6_weights = dict((name, 6) for name in contract.weight_modules)
        w6_activations = dict(((owner, role), 6)
                              for block in contract.blocks
                              for owner, role in block.activation_owners)
        w6_candidate = _candidate(
            contract, w6_weights, w6_activations,
            "UNIFORM_W6A6_FP32_PROP", "boundary")
        w6_rows = evaluator._evaluate_candidate(w6_candidate)
        w6_metrics = _aggregate(w6_rows)
        if not w6_metrics["valid"]:
            raise RuntimeError("W6A6 FP32 propagation baseline is invalid")
        ablation_rows = []
        for module in ranked:
            weights = dict(base_weights)
            activations = dict(base_activations)
            if module not in weights:
                raise ValueError("ranked module is outside contract: %s" % module)
            weights[module] = int(ablation_bits)
            changed_activation = []
            for owner in activations:
                if _module_from_owner(owner) == module:
                    activations[owner] = int(ablation_bits)
                    changed_activation.append(owner)
            candidate = _candidate(
                contract, weights, activations,
                "MODULE_%s_W%sA%s" % (module, ablation_bits, ablation_bits),
                "single_module")
            rows = evaluator._evaluate_candidate(candidate)
            metrics = _aggregate(rows)
            ablation_rows.append({
                "model": model_name,
                "module": module,
                "rank": ranked.index(module) + 1,
                "changed_activation_owner_count": len(changed_activation),
                "changed_activation_owners": json.dumps(changed_activation),
                "pooled_rmse": metrics["pooled_rmse"],
                "mean_sample_rmse": metrics["mean_sample_rmse"],
                "delta_vs_w8a8_pooled_rmse": metrics["pooled_rmse"] - baseline_metrics["pooled_rmse"],
                "relative_delta_vs_w8a8": metrics["pooled_rmse"] / baseline_metrics["pooled_rmse"] - 1.0,
                "delta_vs_fp32_pooled_rmse": metrics["pooled_rmse"] - reference_pooled,
                "valid": metrics["valid"],
                "prediction_finite": metrics["prediction_finite"],
                "propagation_valid": metrics["propagation_valid"],
                "reproducible": metrics["reproducible"],
            })
        with (output / "single_module_ablation.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(ablation_rows[0]))
            writer.writeheader()
            writer.writerows(ablation_rows)
        assignment, fixed_weight, fixed_activation, fixed_unmapped = _allocation(
            contract, costs, weight_scores, activation_scores, ablation_rows,
            weight_budget, activation_budget, max_ablation_delta,
            boundary_bits)
        final_candidate = mixed_precision.P3T3Candidate(
            name="TASK_AWARE_W%s_A%s" % (weight_budget, activation_budget),
            stage="task_aware_mixed",
            prefix=(), tail=(), promoted_blocks=(), assignment=assignment)
        final_rows = evaluator._evaluate_candidate(final_candidate)
        final_metrics = _aggregate(final_rows)
        weight_audit = mixed_precision.audit_budget(assignment, costs)
        payload_out = {
            "model": model_name,
            "device": device,
            "calibration_count": int(spec["calibration_count"]),
            "evaluation_indices": list(spec["evaluation_indices"]),
            "ordinary_precision_levels": list(BIT_LEVELS),
            "ablation_bits": int(ablation_bits),
            "propagation_projection_precision": [8, 8],
            "propagation_loop_precision": "fp32",
            "budget_definition": "weight MAC weighted average and activation element weighted average over ordinary contract owners",
            "requested_weight_budget": float(weight_budget),
            "requested_activation_budget": float(activation_budget),
            "boundary_bits": int(boundary_bits),
            "actual_weight_budget": weight_audit.average_weight_bits,
            "actual_activation_budget": weight_audit.average_activation_bits,
            "budget_feasible": weight_audit.average_weight_bits <= weight_budget and
                weight_audit.average_activation_bits <= activation_budget,
            "fixed_weight_modules": list(fixed_weight),
            "fixed_activation_owners": [list(owner) for owner in fixed_activation],
            "unmapped_activation_owners": [list(owner) for owner in fixed_unmapped],
            "ranked_modules": list(ranked),
            "max_ablation_delta": float(max_ablation_delta),
            "fp32_reference_pooled_rmse": _json_metric(reference_pooled),
            "w8a8_pooled_rmse": _json_metric(baseline_metrics["pooled_rmse"]),
            "w6a6_pooled_rmse": _json_metric(w6_metrics["pooled_rmse"]),
            "final_pooled_rmse": _json_metric(final_metrics["pooled_rmse"]),
            "final_mean_sample_rmse": _json_metric(final_metrics["mean_sample_rmse"]),
            "final_valid": final_metrics["valid"],
            "assignment": {
                "weight_bits": [[name, bits] for name, bits in assignment.weight_bits],
                "activation_bits": [[[owner[0], owner[1]], bits]
                                    for owner, bits in assignment.activation_bits],
            },
            "cost_audit": {
                "weight_fractions": [list(row) for row in weight_audit.weight_mac_fractions],
                "activation_fractions": [list(row) for row in weight_audit.activation_element_fractions],
            },
        }
        (output / "mixed_assignment.json").write_text(
            json.dumps(payload_out, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                "model", "configuration", "pooled_rmse", "mean_sample_rmse",
                "delta_vs_fp32", "delta_vs_w8a8", "valid", "actual_weight_bits",
                "actual_activation_bits", "requested_weight_bits",
                "requested_activation_bits"))
            writer.writeheader()
            writer.writerow({
                "model": model_name, "configuration": "FP32_REFERENCE",
                "pooled_rmse": reference_pooled, "mean_sample_rmse":
                    reference_metrics["mean_sample_rmse"],
                "delta_vs_fp32": 0.0, "delta_vs_w8a8": None, "valid": True,
                "actual_weight_bits": 32, "actual_activation_bits": 32,
                "requested_weight_bits": None, "requested_activation_bits": None})
            writer.writerow({
                "model": model_name, "configuration": "W8A8_FP32_PROP",
                "pooled_rmse": baseline_metrics["pooled_rmse"],
                "mean_sample_rmse": baseline_metrics["mean_sample_rmse"],
                "delta_vs_fp32": baseline_metrics["pooled_rmse"] - reference_pooled,
                "delta_vs_w8a8": 0.0, "valid": baseline_metrics["valid"],
                "actual_weight_bits": 8, "actual_activation_bits": 8,
                "requested_weight_bits": 8, "requested_activation_bits": 8})
            writer.writerow({
                "model": model_name, "configuration": "W6A6_FP32_PROP",
                "pooled_rmse": w6_metrics["pooled_rmse"],
                "mean_sample_rmse": w6_metrics["mean_sample_rmse"],
                "delta_vs_fp32": w6_metrics["pooled_rmse"] - reference_pooled,
                "delta_vs_w8a8": w6_metrics["pooled_rmse"] - baseline_metrics["pooled_rmse"],
                "valid": w6_metrics["valid"],
                "actual_weight_bits": 6, "actual_activation_bits": 6,
                "requested_weight_bits": 6, "requested_activation_bits": 6})
            writer.writerow({
                "model": model_name, "configuration": "TASK_AWARE_MIXED",
                "pooled_rmse": final_metrics["pooled_rmse"],
                "mean_sample_rmse": final_metrics["mean_sample_rmse"],
                "delta_vs_fp32": final_metrics["pooled_rmse"] - reference_pooled,
                "delta_vs_w8a8": final_metrics["pooled_rmse"] - baseline_metrics["pooled_rmse"],
                "valid": final_metrics["valid"],
                "actual_weight_bits": weight_audit.average_weight_bits,
                "actual_activation_bits": weight_audit.average_activation_bits,
                "requested_weight_bits": weight_budget,
                "requested_activation_bits": activation_budget})
    finally:
        evaluator.close()
        runtime.close()
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sensitivity-root", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-modules", type=int, required=True)
    parser.add_argument("--weight-budget", type=float, required=True)
    parser.add_argument("--activation-budget", type=float, required=True)
    parser.add_argument("--max-ablation-delta", type=float, required=True)
    parser.add_argument("--boundary-bits", type=int, choices=BIT_LEVELS,
                        required=True)
    parser.add_argument("--ablation-bits", type=int, choices=(4, 6),
                        required=True)
    return parser


def main():
    args = build_parser().parse_args(tuple(sys.argv[1:]))
    print(run(
        args.config, args.sensitivity_root, args.model, args.device, args.output,
        args.top_modules, args.weight_budget, args.activation_budget,
        args.max_ablation_delta, args.boundary_bits, args.ablation_bits))


if __name__ == "__main__":
    main()
