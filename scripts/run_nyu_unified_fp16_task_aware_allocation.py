#!/usr/bin/env python3
"""Evaluate four official SPN models with FP16 propagation and task-aware W/A bits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import subprocess
from typing import Mapping, Sequence, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.nyu_model_runtime import NYUModelRuntime
from scripts.run_nyu_task_gradient_ablation import (
    _aggregate,
    _build_contract,
    _candidate,
    _costs,
    _load_rows,
    _module_from_owner,
    _score_table,
    _settings,
)
from scripts.run_nyu_model_p3t3_search import (
    HardDeploymentP3T3Evaluator,
    _propagation_valid,
)
from spn_quant import mixed_precision
from spn_quant.model_contracts import (
    propagation_owned_modules,
    validate_propagation_ownership,
)
from spn_quant.task_aware_allocation import greedy_promotion_allocation


MODEL_NAMES = ("cspn", "dyspn", "nlspn", "completionformer")
BIT_LEVELS = (4, 6, 8)
EXPECTED_CALIBRATION_COUNT = 128
EXPECTED_EVALUATION_COUNT = 64


def validate_protocol_payload(payload: Mapping[str, object]) -> None:
    protocol = payload["protocol"]
    models = tuple(str(name) for name in protocol["models"])
    if models != MODEL_NAMES:
        raise ValueError("protocol model order must include all four models")
    if str(protocol["propagation_dtype"]).lower() != "fp16":
        raise ValueError("unified propagation precision must be FP16")
    if tuple(int(bits) for bits in protocol["bit_levels"]) != BIT_LEVELS:
        raise ValueError("protocol bit levels must be exactly 4, 6, and 8")
    if int(protocol["calibration_count"]) != EXPECTED_CALIBRATION_COUNT:
        raise ValueError("protocol calibration count must equal 128")
    if int(protocol["evaluation_count"]) != EXPECTED_EVALUATION_COUNT:
        raise ValueError("protocol evaluation count must equal 64")
    pairs = tuple(tuple(float(value) for value in pair)
                  for pair in protocol["budget_pairs"])
    if not pairs:
        raise ValueError("protocol budget pairs must be nonempty")
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("protocol budget pairs require weight and activation")
        if any(not math.isfinite(value) or value < 4.0 or value > 8.0
               for value in pair):
            raise ValueError("protocol budgets must be in [4, 8]")


def json_safe(value):
    if isinstance(value, Mapping):
        return dict((str(key), json_safe(item))
                    for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError("unsupported JSON artifact value: %s" % type(value).__name__)


def load_protocol(path: Path) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_protocol_payload(payload)
    for model_name in MODEL_NAMES:
        spec = payload["models"][model_name]
        for key in (
                "model", "run_dir", "checkpoint", "python_executable", "device",
                "propagation_iterations", "data_root",
                "calibration_metadata", "calibration_count",
                "evaluation_indices", "expected_architecture_class",
                "checkpoint_architecture", "required_cuda_extension",
                "native_cuda_operator", "sensitivity_dir"):
            spec[key]
        if str(spec["model"]) != model_name:
            raise ValueError("model specification name differs from key")
        if int(spec["calibration_count"]) != EXPECTED_CALIBRATION_COUNT:
            raise ValueError("model calibration count must equal 128")
        if len(tuple(spec["evaluation_indices"])) != EXPECTED_EVALUATION_COUNT:
            raise ValueError("model evaluation indices must contain 64 samples")
    return payload


class FP16PropagationEvaluator(HardDeploymentP3T3Evaluator):
    """Use ordinary hardware QDQ with an entirely FP16 propagation adapter."""

    def _configure_candidate(self, candidate):
        mixed_precision.validate_assignment_ownership(
            self.contract, candidate.assignment)
        generic_bits, joint_bits = self._activation_configuration(candidate)
        self.instrumentor.configure(
            self.settings.base_weight_bits,
            self.settings.base_activation_bits,
            set(self.registry.blocks),
            weight_bit_overrides=dict(candidate.assignment.weight_bits),
            activation_bit_overrides=generic_bits,
            external_output_ownership=True,
            quantize_bias=False,
        )
        self.instrumentor.set_runtime_statistics(False)
        missing = set(generic_bits) - set(self.instrumentor.quantizers)
        if missing:
            raise RuntimeError(
                "contract activation boundaries were not calibrated: %s" %
                sorted(missing))
        self.instrumentor.quantizers = dict(
            (key, self.instrumentor.quantizers[key]) for key in generic_bits)
        self.instrumentor.relu_quantizers = {}
        if self.concat_adapter is not None:
            assignment_weights = dict(candidate.assignment.weight_bits)
            assignment_activations = dict(candidate.assignment.activation_bits)
            concat_weights = {}
            concat_activations = {}
            for name in self.concat_adapter.consumer_modules:
                concat_weights[name] = assignment_weights[name]
                owner = ("activation::%s::input" % name, "module_input")
                if owner not in assignment_activations:
                    raise ValueError(
                        "concat consumer activation assignment is missing: %s" %
                        (owner,))
                concat_activations[name] = assignment_activations[owner]
            self.concat_adapter.configure(
                concat_weights, concat_activations, concat_activations)
        self.propagation_adapter.configure_fp16()
        self._configure_joint(joint_bits)

    def _cache_evaluation_batches(self):
        self.evaluation_batches = tuple(
            (int(index), self._sample_batch(self.valset, index))
            for index in self.settings.evaluation_indices)
        if not self.evaluation_batches:
            raise ValueError("evaluation identities must be nonempty")

    def _evaluate_candidate(self, candidate):
        self._configure_candidate(candidate)
        rows = []
        with torch.no_grad():
            for sample_index, batch in self.evaluation_batches:
                first, ground_truth = self._forward(batch)
                first_propagation = tuple(self.propagation_adapter.statistics())
                second, second_ground_truth = self._forward(batch)
                second_propagation = tuple(
                    self.propagation_adapter.statistics())
                if not torch.equal(ground_truth, second_ground_truth):
                    raise RuntimeError("paired ground truth changed between forwards")
                finite = bool(torch.isfinite(first).all().item()) and \
                    bool(torch.isfinite(second).all().item())
                reproducible = finite and torch.equal(first, second)
                propagation_valid = _propagation_valid(
                    self.runtime.model_name, self.preserve_input,
                    first_propagation) and _propagation_valid(
                    self.runtime.model_name, self.preserve_input,
                    second_propagation)
                prediction = first[0]
                target = ground_truth[0]
                valid = torch.isfinite(target) & (target > 1e-4)
                valid_pixels = int(valid.sum().item())
                if valid_pixels <= 0:
                    raise ValueError("evaluation sample has no valid depth pixels")
                values = prediction[valid]
                prediction_finite = finite and bool(
                    torch.isfinite(values).all().item())
                positive = prediction_finite and bool((values > 1e-4).all().item())
                difference = values.double() - target[valid].double()
                squared_error_sum = float(difference.square().sum().item())
                rows.append({
                    "config": candidate.name,
                    "sample_index": int(sample_index),
                    "squared_error_sum": squared_error_sum,
                    "valid_pixels": valid_pixels,
                    "RMSE": math.sqrt(squared_error_sum / float(valid_pixels)),
                    "prediction_finite": prediction_finite,
                    "propagation_valid": propagation_valid and positive,
                    "reproducible": reproducible,
                })
        return rows


def _reference_metrics(evaluator):
    evaluator.instrumentor.disable()
    if evaluator.concat_adapter is not None:
        evaluator.concat_adapter.disable()
    evaluator.propagation_adapter.disable()
    if evaluator.joint_adapter is not None:
        evaluator.joint_adapter.unbind_qdrop_sites()
    rows = []
    with torch.no_grad():
        for sample_index, batch in evaluator.evaluation_batches:
            prediction, ground_truth = evaluator._forward(batch)
            values = prediction[0]
            target = ground_truth[0]
            valid = torch.isfinite(target) & (target > 1e-4)
            valid_pixels = int(valid.sum().item())
            if valid_pixels <= 0:
                raise ValueError("reference sample has no valid depth pixels")
            prediction_values = values[valid]
            if not bool(torch.isfinite(prediction_values).all().item()):
                raise RuntimeError("FP32 reference prediction is non-finite")
            if bool((prediction_values <= 1e-4).any().item()):
                raise RuntimeError("FP32 reference prediction is non-positive")
            difference = prediction_values.double() - target[valid].double()
            squared_error_sum = float(difference.square().sum().item())
            rows.append({
                "sample_index": int(sample_index),
                "squared_error_sum": squared_error_sum,
                "valid_pixels": valid_pixels,
                "RMSE": math.sqrt(squared_error_sum / float(valid_pixels)),
                "prediction_finite": True,
                "propagation_valid": True,
                "reproducible": True,
            })
    return _aggregate(rows)


def _runtime_args(spec: Mapping[str, object], device: str):
    return type("RuntimeArgs", (), {
        "model": spec["model"],
        "run_dir": Path(spec["run_dir"]),
        "checkpoint": Path(spec["checkpoint"]),
        "expected_architecture_class": spec["expected_architecture_class"],
        "required_cuda_extension": spec["required_cuda_extension"],
        "propagation_iterations": int(spec["propagation_iterations"]),
        "data_root": Path(spec["data_root"]),
        "device": device,
        "checkpoint_architecture": spec["checkpoint_architecture"],
        "native_cuda_operator": spec["native_cuda_operator"],
    })()


def _read_score_tables(spec: Mapping[str, object]):
    root = Path(spec["sensitivity_dir"])
    weight_scores = _score_table(
        _load_rows(root / "weight_sensitivity.csv", "module"), "module")
    activation_scores = _score_table(
        _load_rows(root / "module_sensitivity.csv", "module"), "module")
    return weight_scores, activation_scores


def sensitivity_module_from_owner(owner):
    site, role = owner
    del role
    parts = str(site).split("::")
    if len(parts) == 3 and parts[0] == "activation" and \
            parts[2] in ("input", "output"):
        return parts[1]
    if len(parts) == 3 and parts[0] == "attention":
        if parts[2] == "q":
            return parts[1] + ".q"
        if parts[2] in ("k", "v"):
            return parts[1] + ".kv"
        raise ValueError("unsupported attention owner: %s" % (owner,))
    if len(parts) == 3 and parts[0] == "concat":
        if parts[2] not in ("cnn_input", "transformer_input"):
            raise ValueError("unsupported concat owner: %s" % (owner,))
        return parts[1]
    raise ValueError("unsupported activation owner: %s" % (owner,))


def ordinary_score_tables(contract, costs, weight_scores, activation_scores):
    weight_names = set(name for name, value in costs.weight_macs)
    activation_modules = {}
    for owner, value in costs.activation_elements:
        module = sensitivity_module_from_owner(owner)
        activation_modules[owner] = module
    ordinary_names = weight_names | set(activation_modules.values())
    protected_names = set(propagation_owned_modules(contract))
    unknown_weight = sorted(set(weight_scores) - ordinary_names - protected_names)
    if unknown_weight:
        raise ValueError("unknown weight sensitivity modules: %s" %
                         unknown_weight)
    unknown_activation = sorted(
        set(activation_scores) - ordinary_names - protected_names)
    if unknown_activation:
        raise ValueError("unknown activation sensitivity modules: %s" %
                         unknown_activation)
    missing_weight = sorted(weight_names - set(weight_scores))
    if missing_weight:
        raise ValueError("missing weight sensitivity modules: %s" %
                         missing_weight)
    owner_scores = {}
    for owner, module in activation_modules.items():
        if module not in activation_scores:
            raise ValueError("missing activation sensitivity module: %s" %
                             module)
        owner_scores[owner] = activation_scores[module]
    return (
        dict((name, weight_scores[name]) for name in sorted(weight_names)),
        owner_scores,
    )


def _assignment_for_budget(contract, costs, weight_scores, activation_scores,
                           weight_budget: float, activation_budget: float):
    weight_costs = dict(costs.weight_macs)
    activation_costs = dict(costs.activation_elements)
    if set(weight_scores) != set(weight_costs):
        raise ValueError("weight sensitivity and cost coverage differ")
    if set(activation_scores) != set(activation_costs):
        raise ValueError("activation sensitivity and cost coverage differ")
    weight_bits = greedy_promotion_allocation(
        weight_costs, weight_scores, weight_budget, fixed_bits={})
    activation_bits = greedy_promotion_allocation(
        activation_costs, activation_scores, activation_budget,
        fixed_bits={})
    assignment = mixed_precision.BitAssignment(
        weight_bits=tuple(weight_bits),
        activation_bits=tuple(activation_bits),
        model_name=contract.model_name,
    )
    mixed_precision.validate_assignment_ownership(contract, assignment)
    audit = mixed_precision.audit_budget(assignment, costs)
    if audit.average_weight_bits > float(weight_budget) or \
            audit.average_activation_bits > float(activation_budget):
        raise RuntimeError("selected assignment exceeds requested budgets")
    return assignment, audit


def _assignment_payload(assignment):
    return {
        "model_name": assignment.model_name,
        "weight_bits": [[name, bits] for name, bits in assignment.weight_bits],
        "activation_bits": [
            [[owner[0], owner[1]], bits]
            for owner, bits in assignment.activation_bits
        ],
    }


def _metric_row(model_name, configuration, metrics, audit, reference):
    return {
        "model": model_name,
        "configuration": configuration,
        "pooled_rmse": metrics["pooled_rmse"],
        "mean_sample_rmse": metrics["mean_sample_rmse"],
        "delta_vs_fp32": metrics["pooled_rmse"] - reference["pooled_rmse"],
        "relative_fp_loss": metrics["pooled_rmse"] /
        reference["pooled_rmse"] - 1.0,
        "valid": metrics["valid"],
        "average_weight_bits": None if audit is None
        else audit.average_weight_bits,
        "average_activation_bits": None if audit is None
        else audit.average_activation_bits,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RuntimeError("cannot write empty result CSV: %s" % path)
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise RuntimeError("result row schema differs: %s" % path)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_manifest_payload(payload: Mapping[str, object],
                         model_outputs: Sequence[str]):
    output_root = str(payload["output_root"])
    expected = tuple(output_root + "/" + model_name
                     for model_name in MODEL_NAMES)
    if tuple(model_outputs) != expected:
        raise ValueError("unified model outputs do not cover all models")
    return json_safe({
        "protocol": payload["protocol"],
        "models": list(MODEL_NAMES),
        "model_outputs": list(model_outputs),
    })


def finalize_run(config: Path) -> Path:
    payload = load_protocol(config)
    output_root = Path(payload["output_root"])
    if not output_root.is_dir():
        raise FileNotFoundError("unified output root is missing: %s" %
                                output_root)
    model_outputs = tuple(str(output_root / model_name)
                          for model_name in MODEL_NAMES)
    for model_output in model_outputs:
        model_root = Path(model_output)
        if not model_root.is_dir():
            raise FileNotFoundError("model output is missing: %s" % model_root)
        for filename in ("summary.csv", "pareto.csv", "manifest.json"):
            if not (model_root / filename).is_file():
                raise FileNotFoundError(
                    "model result artifact is missing: %s" %
                    (model_root / filename))
    manifest_path = output_root / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("unified run manifest already exists: %s" %
                              manifest_path)
    manifest_path.write_text(
        json.dumps(run_manifest_payload(payload, model_outputs), indent=2,
                   sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return manifest_path


def _run_model(payload: Mapping[str, object], model_name: str,
               output_root: Path) -> Path:
    spec = payload["models"][model_name]
    hard = payload["hard_deployment"]
    device = str(spec["device"])
    runtime = NYUModelRuntime.from_args(_runtime_args(spec, device))
    model = runtime.build_model(runtime.device)
    contract = _build_contract(model_name, model)
    validate_propagation_ownership(contract, model)
    costs = _costs(model_name, spec, model, runtime)
    registry = mixed_precision.build_registry(contract, costs)
    weight_scores, activation_scores = _read_score_tables(spec)
    weight_scores, activation_scores = ordinary_score_tables(
        contract, costs, weight_scores, activation_scores)
    settings = _settings(spec, hard, device)
    evaluator = FP16PropagationEvaluator(
        runtime, model, contract, registry, settings)
    reference = _reference_metrics(evaluator)
    all_weight_bits = dict((name, 8) for name in contract.weight_modules)
    all_activation_bits = dict(
        ((owner, role), 8)
        for block in contract.blocks
        for owner, role in block.activation_owners)
    baseline = _candidate(
        contract, all_weight_bits, all_activation_bits,
        "UNIFORM_W8A8_FP16_PROP", "baseline")
    baseline_metrics = _aggregate(evaluator._evaluate_candidate(baseline))
    if not baseline_metrics["valid"]:
        raise RuntimeError("%s W8A8 FP16 propagation baseline is invalid" %
                           model_name)

    model_output = output_root / model_name
    model_output.mkdir(parents=True)
    metric_rows = [
        _metric_row(model_name, "FP32_REFERENCE", reference, None, reference),
        _metric_row(model_name, "UNIFORM_W8A8_FP16_PROP",
                    baseline_metrics, mixed_precision.audit_budget(
                        baseline.assignment, costs), reference),
    ]
    pareto_rows = []
    assignment_payloads = {}
    for weight_budget, activation_budget in (
            tuple(tuple(float(value) for value in pair)
                  for pair in payload["protocol"]["budget_pairs"])):
        assignment, audit = _assignment_for_budget(
            contract, costs, weight_scores, activation_scores,
            weight_budget, activation_budget)
        candidate = _candidate(
            contract,
            dict(assignment.weight_bits),
            dict(assignment.activation_bits),
            "TASK_AWARE_W%.2f_A%.2f" %
            (weight_budget, activation_budget),
            "task_aware_mixed",
        )
        metrics = _aggregate(evaluator._evaluate_candidate(candidate))
        configuration = "TASK_AWARE_W%.2f_A%.2f" % (
            weight_budget, activation_budget)
        metric_rows.append(
            _metric_row(model_name, configuration, metrics, audit, reference))
        pareto_rows.append({
            "model": model_name,
            "requested_weight_budget": weight_budget,
            "requested_activation_budget": activation_budget,
            "actual_weight_bits": audit.average_weight_bits,
            "actual_activation_bits": audit.average_activation_bits,
            "pooled_rmse": metrics["pooled_rmse"],
            "relative_fp_loss": metrics["pooled_rmse"] /
            reference["pooled_rmse"] - 1.0,
            "valid": metrics["valid"],
        })
        assignment_payloads[configuration] = {
            "assignment": _assignment_payload(assignment),
            "budget": {
                "requested_weight_bits": weight_budget,
                "requested_activation_bits": activation_budget,
                "actual_weight_bits": audit.average_weight_bits,
                "actual_activation_bits": audit.average_activation_bits,
            },
        }
    _write_csv(model_output / "summary.csv", metric_rows)
    _write_csv(model_output / "pareto.csv", pareto_rows)
    artifact = {
        "model": model_name,
        "device": device,
        "propagation_dtype": "fp16",
        "propagation_owned_modules": list(
            propagation_owned_modules(contract)),
        "ordinary_blocks": list(contract.block_names),
        "calibration_count": int(spec["calibration_count"]),
        "evaluation_indices": list(spec["evaluation_indices"]),
        "assignments": assignment_payloads,
        "summary": metric_rows,
        "baseline": baseline_metrics,
        "reference": reference,
    }
    (model_output / "manifest.json").write_text(
        json.dumps(json_safe(artifact), indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
        encoding="utf-8")
    evaluator.close()
    runtime.close()
    return model_output


def run(config: Path) -> Path:
    payload = load_protocol(config)
    output_root = Path(payload["output_root"])
    if output_root.exists():
        raise FileExistsError("unified output already exists: %s" % output_root)
    output_root.mkdir(parents=True)
    model_outputs = []
    for model_name in MODEL_NAMES:
        spec = payload["models"][model_name]
        environment = dict(os.environ)
        environment["SPN_EXTERNAL_ROOT"] = payload["external_root"]
        environment["COMPLETIONFORMER_ROOT"] = payload[
            "completionformer_root"]
        environment["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + \
            os.environ["PYTHONPATH"]
        subprocess.run(
            (str(spec["python_executable"]), str(Path(__file__)),
             "--config", str(Path(config)), "--model", model_name),
             cwd=str(spec["data_root"]), env=environment, check=True)
        model_outputs.append(str(output_root / model_name))
    finalize_run(config)
    return output_root


def run_model(config: Path, model_name: str) -> Path:
    payload = load_protocol(config)
    if model_name not in MODEL_NAMES:
        raise ValueError("unknown model: %s" % model_name)
    output_root = Path(payload["output_root"])
    if not output_root.exists():
        raise FileNotFoundError("unified output root is missing: %s" %
                                output_root)
    return _run_model(payload, model_name, output_root)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--finalize", action="store_true")
    return parser


def main():
    args = build_parser().parse_args(tuple(sys.argv[1:]))
    if args.finalize:
        if args.model is not None:
            raise ValueError("--finalize cannot be combined with --model")
        print(finalize_run(args.config))
    else:
        print(run_model(args.config, args.model) if args.model is not None else
              run(args.config))


if __name__ == "__main__":
    main()
