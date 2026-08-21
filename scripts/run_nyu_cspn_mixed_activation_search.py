#!/usr/bin/env python3
"""Search P3/T3-derived mixed activations for official CSPN."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Mapping, Sequence, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts import run_nyu_cspn_stem_precision as stem_runner
from scripts import run_nyu_cspn_task_sensitive_bits as task_runner
from scripts.run_nyu_rtn_quantization import (
    calibration_dataset,
    evaluation_dataset,
    write_csv,
    write_json,
)
from spn_quant import cspn_task_sensitive_bits as allocation


@dataclass(frozen=True)
class ActivationSearchResult:
    candidates: Tuple[task_runner.RuntimeCandidate, ...]
    rows: Tuple[Mapping[str, object], ...]
    selected_assignment: allocation.BitAssignment
    audit: allocation.ActivationBudgetAudit


def parse_devices(value: str) -> Tuple[str, ...]:
    devices = tuple(
        token.strip() for token in str(value).split(",") if token.strip())
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("CUDA devices must be nonempty and unique")
    if any(not device.startswith("cuda:") for device in devices):
        raise ValueError("mixed activation search requires explicit CUDA devices")
    return devices


def load_precision_config(path: Path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload["model"] != "cspn":
        raise ValueError("mixed task-aware configuration requires model=cspn")
    search = payload["search"]
    if tuple(int(bits) for bits in search["activation_bits"]) != \
            allocation.MIXED_ACTIVATION_BITS:
        raise ValueError("mixed activation bits must be [4, 6, 8]")
    if float(search["activation_budget_bits"]) != 6.0:
        raise ValueError("mixed activation budget must be 6.0 bits")
    if int(search["group_size"]) != 8:
        raise ValueError("mixed activation search requires Group-8")
    if not math.isfinite(float(search["boundary_threshold_m"])) or \
            float(search["boundary_threshold_m"]) <= 0.0:
        raise ValueError("boundary threshold must be finite and positive")

    loss = payload["loss"]
    loss_values = (
        float(loss["depth"]),
        float(loss["boundary"]),
        float(loss["teacher"]),
        float(loss["propagation"]),
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in loss_values):
        raise ValueError("task loss weights must be finite and nonnegative")
    training = payload["training"]
    training_values = (
        int(training["epochs"]),
        int(training["patience"]),
        float(training["min_relative_improvement"]),
        int(training["batch_size"]),
        int(training["val_batch_size"]),
        int(training["workers"]),
        float(training["learning_rate"]),
        float(training["momentum"]),
        float(training["weight_decay"]),
        float(training["max_gradient_norm"]),
        int(training["seed"]),
        int(training["max_train_samples"]),
        int(training["max_val_samples"]),
        float(training["fold_max_error"]),
        int(training["log_interval"]),
    )
    if not all(math.isfinite(float(value)) for value in training_values):
        raise ValueError("training configuration must be finite")
    acceptance = payload["acceptance"]
    acceptance_values = (
        float(acceptance["rmse_m"]),
        float(acceptance["average_activation_bits"]),
        float(acceptance["nonfinite_ratio"]),
        float(acceptance["nonpositive_ratio"]),
        float(acceptance["anchor_max_error"]),
        float(acceptance["coefficient_sum_max_error"]),
        float(acceptance["contraction_violation_ratio"]),
    )
    if not all(math.isfinite(value) for value in acceptance_values):
        raise ValueError("acceptance configuration must be finite")
    return payload


def validate_identity_isolation(
        calibration_identities: Sequence[Tuple[object, object]],
        evaluation_identities: Sequence[Tuple[object, object]]) -> None:
    if set(calibration_identities) & set(evaluation_identities):
        raise ValueError("calibration and evaluation identities overlap")


def validate_index_protocol(calibration, evaluation):
    protocol = stem_runner.index_protocol(calibration, evaluation)
    if protocol.selection != "32_tail_96_kmedoids":
        raise ValueError("mixed activation search requires stratified calibration")
    validate_identity_isolation(
        protocol.calibration_identities, protocol.evaluation_identities)
    return protocol


def _candidate_name(
        assignment: allocation.BitAssignment,
        registry: allocation.AllocationRegistry) -> str:
    activation_bits = dict(assignment.activation_bits)
    values = []
    for block in allocation.P3_T3_PROTECTED_BLOCKS:
        block_values = {
            activation_bits[owner]
            for owner in registry.activations_by_block[block]}
        if len(block_values) != 1:
            raise ValueError("mixed activation candidate is not block-uniform")
        values.append(str(next(iter(block_values))))
    return "MIXED_A" + "_".join(values)


def _runtime_candidates(
        registry: allocation.AllocationRegistry,
        basis: allocation.CostBasis,
        maximum_bits: float):
    assignments = allocation.build_p3_t3_activation_candidates(
        registry, basis, maximum_bits)
    return tuple(
        task_runner.RuntimeCandidate(
            _candidate_name(assignment, registry), "joint", assignment)
        for assignment in assignments)


def run_activation_search(
        evaluator,
        registry: allocation.AllocationRegistry,
        basis: allocation.CostBasis,
        maximum_bits: float) -> ActivationSearchResult:
    candidates = _runtime_candidates(registry, basis, maximum_bits)
    rows = tuple(evaluator.calibration("mixed_activation", candidates))
    if tuple(str(row["config"]) for row in rows) != tuple(
            candidate.name for candidate in candidates):
        raise ValueError("mixed activation result order differs from candidates")
    if any(row["assignment"] != candidate.assignment
           for row, candidate in zip(rows, candidates)):
        raise ValueError("mixed activation result assignment differs")
    selected = allocation.select_measured_activation_candidate(
        tuple(candidate.assignment for candidate in candidates),
        rows,
        basis,
        maximum_bits,
    )
    return ActivationSearchResult(
        candidates=candidates,
        rows=rows,
        selected_assignment=selected,
        audit=allocation.audit_activation_budget(
            selected, basis, maximum_bits),
    )


def _basis_payload(basis: allocation.CostBasis):
    return {
        "weight_macs": [
            {"module": module, "macs": macs}
            for module, macs in basis.weight_macs],
        "activation_elements": [
            {"module": owner[0], "kind": owner[1], "elements": elements}
            for owner, elements in basis.activation_elements],
    }


def _metric_rows(result: ActivationSearchResult, basis: allocation.CostBasis):
    output = []
    for row in result.rows:
        assignment = row["assignment"]
        audit = allocation.audit_activation_budget(
            assignment, basis, result.audit.maximum_activation_bits)
        persisted = dict(
            (key, row[key]) for key in row if key != "assignment")
        persisted["assignment"] = json.dumps(
            task_runner.assignment_payload(assignment), sort_keys=True)
        persisted["average_activation_bits"] = audit.average_activation_bits
        output.append(persisted)
    return tuple(output)


def publish_search_result(
        staging: Path,
        output: Path,
        result: ActivationSearchResult,
        basis: allocation.CostBasis,
        source_manifest: Mapping[str, object]) -> None:
    root = Path(staging)
    final = Path(output)
    if not root.is_dir():
        raise FileNotFoundError("mixed activation staging directory is missing")
    if final.exists():
        raise FileExistsError("mixed activation output already exists")
    phase_cache = root / "phase_cache"
    if phase_cache.is_dir():
        shutil.rmtree(phase_cache)

    write_json(
        root / "selected_assignment.json",
        task_runner.assignment_payload(result.selected_assignment))
    write_json(root / "cost_basis.json", _basis_payload(basis))
    write_csv(
        root / "candidate_metrics.csv",
        _metric_rows(result, basis),
        ("config", "calibration_RMSE", "boundary_RMSE",
         "propagation_MSE", "average_activation_bits",
         "nonfinite_ratio", "nonpositive_ratio"),
    )
    write_json(root / "search_manifest.json", {
        "candidate_count": len(result.candidates),
        "selected_config": next(
            candidate.name for candidate in result.candidates
            if candidate.assignment == result.selected_assignment),
        "budget": {
            "activation_numerator": result.audit.activation_numerator,
            "activation_denominator": result.audit.activation_denominator,
            "average_activation_bits": result.audit.average_activation_bits,
            "maximum_activation_bits": result.audit.maximum_activation_bits,
            "feasible": result.audit.feasible,
        },
        "source": dict(source_manifest),
    })
    artifact_names = (
        "candidate_metrics.csv",
        "cost_basis.json",
        "search_manifest.json",
        "selected_assignment.json",
    )
    hashes = dict(
        (name, stem_runner._sha256(root / name)) for name in artifact_names)
    write_json(root / "artifact_sha256.json", hashes)
    if dict((name, stem_runner._sha256(root / name))
            for name in artifact_names) != hashes:
        raise RuntimeError("mixed activation artifacts changed before publication")
    root.rename(final)


def _bootstrap_assignment(registry: allocation.AllocationRegistry):
    seed = allocation.p3_t3_assignment(registry)
    return allocation.BitAssignment(
        weight_bits=seed.weight_bits,
        activation_bits=tuple(
            (owner, 4) for owner, bits in seed.activation_bits),
    )


def _prepare_staging(output: Path) -> Path:
    final = Path(output)
    staging = Path(str(final) + ".incomplete")
    if final.exists() or staging.exists():
        raise FileExistsError("mixed activation output path already exists")
    staging.mkdir(parents=True)
    (staging / "phase_cache").mkdir()
    return staging


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--calibration-indices", required=True)
    parser.add_argument("--calibration-metadata", required=True)
    parser.add_argument("--evaluation-protocol", required=True)
    parser.add_argument("--precision-config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold-max-error", type=float, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    devices = parse_devices(args.devices)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    precision_config_path = Path(args.precision_config)
    precision_config = load_precision_config(precision_config_path)
    if int(args.seed) != int(precision_config["training"]["seed"]):
        raise ValueError("runner seed differs from precision configuration")
    if float(args.fold_max_error) != float(
            precision_config["training"]["fold_max_error"]):
        raise ValueError("fold threshold differs from precision configuration")

    checkpoint = Path(args.checkpoint)
    calibration_indices_path = Path(args.calibration_indices)
    calibration_metadata_path = Path(args.calibration_metadata)
    evaluation_protocol_path = Path(args.evaluation_protocol)
    calibration_payload = json.loads(
        calibration_indices_path.read_text(encoding="utf-8"))
    calibration_metadata = json.loads(
        calibration_metadata_path.read_text(encoding="utf-8"))
    evaluation_metadata = json.loads(
        evaluation_protocol_path.read_text(encoding="utf-8"))
    protocol = validate_index_protocol(
        calibration_payload, evaluation_metadata)
    if int(args.seed) != protocol.seed:
        raise ValueError("runner seed differs from evaluation protocol")
    checkpoint_sha256 = stem_runner._sha256(checkpoint)
    stem_runner._validate_source_metadata(
        args, calibration_metadata, evaluation_metadata, checkpoint_sha256)

    args.device = task_runner.coordinator_device(devices)
    saved_args = stem_runner._saved_args(args)
    trainset = calibration_dataset(saved_args)
    valset = evaluation_dataset(saved_args)
    if max(protocol.calibration_indices) >= len(trainset):
        raise ValueError("calibration index exceeds the train split")
    trainset = task_runner.materialize_samples(
        trainset, protocol.calibration_indices, protocol.seed)
    output = Path(args.out_dir)
    staging = _prepare_staging(output)
    workers = tuple(task_runner.CSPNEvaluator(
        saved_args,
        checkpoint,
        trainset,
        valset,
        protocol.calibration_indices,
        protocol.evaluation_indices,
        protocol.seed,
        torch.device(device),
        args.fold_max_error,
        None,
    ) for device in devices)
    evaluator = task_runner.ParallelEvaluator(
        workers, staging / "phase_cache")
    registry = task_runner.expected_registry()
    bootstrap_assignment = _bootstrap_assignment(registry)
    bootstrap = task_runner.RuntimeCandidate(
        _candidate_name(bootstrap_assignment, registry),
        "joint",
        bootstrap_assignment,
    )
    evaluator.calibration("mixed_activation", (bootstrap,))
    basis = evaluator.cost_basis()
    result = run_activation_search(
        evaluator,
        registry,
        basis,
        float(precision_config["search"]["activation_budget_bits"]),
    )
    evaluator.close()
    publish_search_result(staging, output, result, basis, {
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_indices_sha256": stem_runner._sha256(
            calibration_indices_path),
        "calibration_metadata_sha256": stem_runner._sha256(
            calibration_metadata_path),
        "evaluation_protocol_sha256": stem_runner._sha256(
            evaluation_protocol_path),
        "config_sha256": stem_runner._sha256(precision_config_path),
        "seed": protocol.seed,
        "devices": list(devices),
    })


if __name__ == "__main__":
    main()
