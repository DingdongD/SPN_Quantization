#!/usr/bin/env python3
"""Train selected official SPN models with strict task-aware QAT."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from spn_quant.experiment_config import MODEL_ORDER  # noqa: E402
from spn_quant.mixed_precision import (  # noqa: E402
    BitAssignment,
    CostBasis,
    audit_activation_budget,
)
from spn_quant.model_contracts import QuantizationModelContract  # noqa: E402
from spn_quant.qdrop_targets import QDropTargetPlan  # noqa: E402


SELECTED_QAT_METHODS = (
    "lsqplus_w4a4",
    "lsqplus_w6a6",
    "hawq_mixed_le6",
    "mixed_task_aware",
)

CHECKPOINT_FIELDS = frozenset((
    "format_version",
    "model_name",
    "method",
    "model_state",
    "method_state",
    "optimizer_state",
    "scheduler_state",
    "epoch",
    "assignment",
    "contract_manifest",
    "training_config",
    "calibration_indices",
    "validation_indices",
    "convergence",
    "history",
    "train_generator_state",
    "torch_rng_state",
    "numpy_rng_state",
    "python_rng_state",
    "cuda_rng_state",
    "deterministic_algorithms",
    "hard_deployment_validation",
    "run_state",
))

HARD_DEPLOYMENT_VALIDATION_FIELDS = frozenset((
    "validated",
    "epoch",
    "method",
    "materialized_weight_count",
    "activation_owner_count",
    "canonical_master_weights",
    "protected_scale_roles_excluded",
    "hard_model_state_sha256",
    "method_state_sha256",
    "qparams",
    "qparams_sha256",
    "evaluation_samples",
    "evaluation_rmse",
    "deployment_fingerprint",
))


class ModelActivationRangeCollector(object):
    """Collect exact calibration ranges for contract-owned QAT edges."""

    def __init__(self, model: nn.Module,
                 contract: QuantizationModelContract,
                 target_plan: QDropTargetPlan) -> None:
        if target_plan.model != contract.model_name:
            raise ValueError("activation range target model differs")
        expected = _contract_owners(contract)
        sites = dict(
            ((site.site, site.role), site)
            for site in target_plan.activation_sites)
        if set(sites) != set(expected) or len(sites) != len(expected):
            raise ValueError(
                "activation range sites differ from model contract")
        self.contract = contract
        self.sites = sites
        self.ranges = {}
        self.handles = []
        modules = dict(model.named_modules())
        for owner in expected:
            site = sites[owner]
            if site.owner_kind not in ("module_input", "module_output"):
                continue
            parts = site.site.split("::")
            if len(parts) != 3 or parts[0] != "activation" or \
                    parts[2] not in ("input", "output"):
                raise ValueError(
                    "activation range module site is invalid: %s" %
                    site.site)
            module_name = parts[1]
            if module_name not in modules:
                raise KeyError(
                    "activation range module is missing: %s" % module_name)
            if parts[2] == "input":
                def pre_hook(current, inputs, target=owner):
                    del current
                    if not inputs or not torch.is_tensor(inputs[0]):
                        raise TypeError(
                            "activation range input must start with tensor")
                    self._record(target, inputs[0])
                handle = modules[module_name].register_forward_pre_hook(
                    pre_hook)
            else:
                def post_hook(current, inputs, output, target=owner):
                    del current, inputs
                    if not torch.is_tensor(output):
                        raise TypeError(
                            "activation range output must be tensor")
                    self._record(target, output)
                handle = modules[module_name].register_forward_hook(post_hook)
            self.handles.append(handle)

    def _record(self, owner, tensor: torch.Tensor) -> None:
        if not bool(torch.isfinite(tensor).all().item()):
            raise FloatingPointError(
                "activation calibration contains non-finite values: %s" %
                (owner,))
        minimum = tensor.detach().amin().to(torch.float32)
        maximum = tensor.detach().amax().to(torch.float32)
        if owner in self.ranges:
            previous_minimum, previous_maximum = self.ranges[owner]
            minimum = torch.minimum(previous_minimum, minimum)
            maximum = torch.maximum(previous_maximum, maximum)
        self.ranges[owner] = minimum, maximum

    def initialization_rows(self, joint_adapter):
        rows = []
        for owner in _contract_owners(self.contract):
            site = self.sites[owner]
            if site.owner_kind in ("module_input", "module_output"):
                if owner not in self.ranges:
                    raise RuntimeError(
                        "contract activation was not calibrated: %s" %
                        (owner,))
                minimum, maximum = self.ranges[owner]
                tensor = torch.stack((minimum, maximum))
            elif site.owner_kind in ("attention_qkv", "concat_input"):
                if joint_adapter is None:
                    raise RuntimeError(
                        "joint activation calibration adapter is missing")
                tensor = joint_adapter.qdrop_initialization_tensor(site)
            else:
                raise ValueError(
                    "unsupported activation calibration owner kind: %s" %
                    site.owner_kind)
            rows.append((owner, tensor))
        return tuple(rows)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _sha256_json(payload) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_state_sha256(state) -> str:
    if not isinstance(state, dict) or not state:
        raise ValueError("fingerprinted tensor state must be nonempty")
    rows = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise TypeError("fingerprinted state must contain named tensors")
        tensor = value.detach().cpu().contiguous()
        header = json.dumps({
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(
            header + b"\0" + tensor.numpy().tobytes()).hexdigest()
        rows.append((name, digest))
    return _sha256_json(rows)


def _deployment_fingerprint_payload(record):
    return {
        "epoch": int(record["epoch"]),
        "method": str(record["method"]),
        "hard_model_state_sha256": record["hard_model_state_sha256"],
        "method_state_sha256": record["method_state_sha256"],
        "qparams_sha256": record["qparams_sha256"],
        "evaluation_samples": int(record["evaluation_samples"]),
        "evaluation_rmse": float(record["evaluation_rmse"]),
    }


def validate_hard_deployment_record(record, epoch: int, method_state) -> None:
    if set(record) != HARD_DEPLOYMENT_VALIDATION_FIELDS:
        raise ValueError("hard deployment validation fields changed")
    integer_fields = (
        "validated", "epoch", "materialized_weight_count",
        "activation_owner_count", "canonical_master_weights",
        "protected_scale_roles_excluded", "evaluation_samples",
    )
    if any(isinstance(record[name], bool) or not isinstance(record[name], int)
           for name in integer_fields):
        raise TypeError("hard deployment integer evidence is invalid")
    if record["validated"] != 1 or record["epoch"] != int(epoch) or \
            record["materialized_weight_count"] <= 0 or \
            record["activation_owner_count"] < 0 or \
            record["canonical_master_weights"] != 1 or \
            record["protected_scale_roles_excluded"] != 1 or \
            record["evaluation_samples"] <= 0:
        raise ValueError(
            "hard deployment validation must match every evaluation epoch")
    if record["method"] not in ("lsqplus", "hawq", "mixed_task_aware"):
        raise ValueError("hard deployment method identity is invalid")
    rmse = record["evaluation_rmse"]
    if isinstance(rmse, bool) or not isinstance(rmse, float) or not \
            math.isfinite(rmse):
        raise ValueError("hard deployment evaluation RMSE is invalid")
    for name in (
            "hard_model_state_sha256", "method_state_sha256",
            "qparams_sha256", "deployment_fingerprint"):
        value = record[name]
        if not isinstance(value, str) or len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value):
            raise ValueError("hard deployment fingerprint is invalid: %s" % name)
    expected_method_sha256 = tensor_state_sha256(method_state)
    if record["method_state_sha256"] != expected_method_sha256:
        raise ValueError("hard deployment method state fingerprint differs")
    if record["qparams_sha256"] != _sha256_json(record["qparams"]):
        raise ValueError("hard deployment qparam fingerprint differs")
    expected_deployment = _sha256_json(
        _deployment_fingerprint_payload(record))
    if record["deployment_fingerprint"] != expected_deployment:
        raise ValueError("hard deployment fingerprint differs")


def validate_hard_deployment_against_controller(record, controller) -> None:
    hard_state = controller.hard_model_state_dict()
    if record["hard_model_state_sha256"] != tensor_state_sha256(hard_state):
        raise ValueError("hard deployment materialized weight fingerprint differs")
    qparams = controller.deployment_qparams()
    if record["qparams"] != qparams or record["qparams_sha256"] != \
            _sha256_json(qparams):
        raise ValueError("hard deployment qparams differ from method state")


def hard_deployment_evaluation_record(
        epoch: int, manifest, evaluation, hard_model_state,
        method_state, qparams):
    required_manifest = {
        "validated", "method", "materialized_weight_count",
        "activation_owner_count", "canonical_master_weights",
        "protected_scale_roles_excluded",
    }
    if set(manifest) < required_manifest or int(manifest["validated"]) != 1:
        raise ValueError("hard deployment validation failed")
    samples = int(evaluation["samples"])
    rmse = float(evaluation["RMSE"])
    if samples <= 0 or not math.isfinite(rmse):
        raise ValueError("hard deployment evaluation metrics are invalid")
    record = {
        "validated": 1,
        "epoch": int(epoch),
        "method": str(manifest["method"]),
        "materialized_weight_count":
            int(manifest["materialized_weight_count"]),
        "activation_owner_count": int(manifest["activation_owner_count"]),
        "canonical_master_weights":
            int(manifest["canonical_master_weights"]),
        "protected_scale_roles_excluded":
            int(manifest["protected_scale_roles_excluded"]),
        "hard_model_state_sha256": tensor_state_sha256(hard_model_state),
        "method_state_sha256": tensor_state_sha256(method_state),
        "qparams": qparams,
        "qparams_sha256": _sha256_json(qparams),
        "evaluation_samples": samples,
        "evaluation_rmse": rmse,
        "deployment_fingerprint": "",
    }
    record["deployment_fingerprint"] = _sha256_json(
        _deployment_fingerprint_payload(record))
    validate_hard_deployment_record(record, epoch, method_state)
    return record


def validate_hard_deployment_stability(before, after) -> None:
    if set(before) != {"model", "method"} or set(after) != set(before):
        raise ValueError("hard deployment stability state fields changed")
    for family in ("model", "method"):
        if set(before[family]) != set(after[family]):
            raise RuntimeError(
                "hard deployment %s state mutated during evaluation" % family)
        if any(not torch.equal(before[family][key], after[family][key])
               for key in before[family]):
            raise RuntimeError(
                "hard deployment %s state mutated during evaluation" % family)


def selected_qat_methods():
    return SELECTED_QAT_METHODS


def validate_method_assignment_paths(
        method: str, hawq_assignment, p3_t3_assignment,
        hawq_trace_artifact) -> None:
    if method not in SELECTED_QAT_METHODS:
        raise ValueError("unsupported selected QAT method: %s" % method)
    if method == "hawq_mixed_le6":
        if hawq_assignment is None:
            raise ValueError("HAWQ QAT requires its mixed assignment")
        if hawq_trace_artifact is None:
            raise ValueError("HAWQ QAT requires its trace artifact")
        if p3_t3_assignment is not None:
            raise ValueError("HAWQ QAT cannot use a P3/T3 assignment")
    elif method == "mixed_task_aware":
        if p3_t3_assignment is None:
            raise ValueError("mixed task-aware QAT requires P3/T3 assignment")
        if hawq_assignment is not None:
            raise ValueError("mixed task-aware QAT cannot use HAWQ assignment")
        if hawq_trace_artifact is not None:
            raise ValueError(
                "mixed task-aware QAT cannot use a HAWQ trace artifact")
    elif hawq_assignment is not None or p3_t3_assignment is not None:
        raise ValueError("uniform LSQ++ QAT cannot use assignment paths")
    elif hawq_trace_artifact is not None:
        raise ValueError("uniform LSQ++ QAT cannot use a HAWQ trace artifact")


def _contract_owners(contract: QuantizationModelContract):
    return tuple(
        owner for block in contract.blocks for owner in block.activation_owners)


def _validate_assignment_coverage(
        assignment: BitAssignment,
        contract: QuantizationModelContract) -> None:
    if assignment.model_name != contract.model_name:
        raise ValueError("QAT assignment model differs from contract")
    weights = tuple(name for name, bits in assignment.weight_bits)
    owners = tuple(owner for owner, bits in assignment.activation_bits)
    if set(weights) != set(contract.weight_modules) or \
            len(weights) != len(contract.weight_modules):
        raise ValueError("QAT weight assignment coverage differs from contract")
    expected_owners = _contract_owners(contract)
    if set(owners) != set(expected_owners) or \
            len(owners) != len(expected_owners):
        raise ValueError(
            "QAT activation assignment coverage differs from contract")
    if set(weights).intersection(contract.protected_modules):
        raise ValueError("QAT assignment contains protected modules")
    protected = set(contract.protected_roles)
    if any(owner[1] in protected for owner in owners):
        raise ValueError("QAT assignment contains protected activation roles")


def uniform_qat_assignment(
        contract: QuantizationModelContract, bits: int) -> BitAssignment:
    bits = int(bits)
    if bits not in (4, 6):
        raise ValueError("uniform LSQ++ precision must be 4 or 6 bits")
    assignment = BitAssignment(
        weight_bits=tuple(
            (name, bits) for name in contract.weight_modules),
        activation_bits=tuple(
            (owner, bits) for owner in _contract_owners(contract)),
        model_name=contract.model_name,
    )
    _validate_assignment_coverage(assignment, contract)
    return assignment


def mixed_task_aware_assignment(
        contract: QuantizationModelContract,
        p3_t3_assignment: BitAssignment,
        activation_bits: Sequence,
        costs: CostBasis,
        maximum_activation_bits: float):
    _validate_assignment_coverage(p3_t3_assignment, contract)
    if not set(bits for name, bits in p3_t3_assignment.weight_bits) <= {4, 8}:
        raise ValueError("P3/T3 mixed QAT weights must use W4 or W8")
    assignment = BitAssignment(
        weight_bits=p3_t3_assignment.weight_bits,
        activation_bits=tuple(activation_bits),
        model_name=contract.model_name,
    )
    _validate_assignment_coverage(assignment, contract)
    if not set(bits for owner, bits in assignment.activation_bits) <= \
            {4, 6, 8}:
        raise ValueError("mixed task-aware activations must use A4, A6, or A8")
    maximum = float(maximum_activation_bits)
    if not math.isfinite(maximum) or maximum <= 0.0 or maximum > 6.0:
        raise ValueError(
            "mixed task-aware activation budget must lie in (0, 6]")
    audit = audit_activation_budget(assignment, costs, maximum)
    if not audit.feasible:
        raise ValueError("mixed task-aware activation assignment exceeds budget")
    return assignment, audit


def _hawq_assignment(payload) -> BitAssignment:
    required = {
        "model_name",
        "weight_block_bits",
        "activation_block_bits",
        "weight_bits",
        "activation_bits",
    }
    if set(payload) != required:
        raise ValueError("HAWQ assignment fields changed")
    schemas = (
        (payload["weight_block_bits"], {"block", "bits"}),
        (payload["activation_block_bits"], {"block", "bits"}),
        (payload["weight_bits"], {"module", "bits"}),
        (payload["activation_bits"], {"site", "role", "bits"}),
    )
    if any(set(row) != fields for rows, fields in schemas for row in rows):
        raise ValueError("HAWQ assignment row fields changed")
    return BitAssignment(
        weight_bits=tuple(
            (str(row["module"]), int(row["bits"]))
            for row in payload["weight_bits"]),
        activation_bits=tuple(
            ((str(row["site"]), str(row["role"])), int(row["bits"]))
            for row in payload["activation_bits"]),
        model_name=str(payload["model_name"]),
    )


def _validated_hawq_calibration(payload):
    calibration = payload["calibration"]
    if set(calibration) != {"count", "indices", "identity_sha256"}:
        raise ValueError("HAWQ calibration provenance fields changed")
    raw_indices = tuple(calibration["indices"])
    if any(isinstance(index, bool) or not isinstance(index, int)
           for index in raw_indices):
        raise TypeError("HAWQ calibration identities must be integers")
    indices = tuple(int(index) for index in raw_indices)
    if isinstance(calibration["count"], bool) or not isinstance(
            calibration["count"], int) or calibration["count"] != 128 or \
            len(indices) != 128 or len(indices) != len(set(indices)) or any(
                index < 0 for index in indices):
        raise ValueError("HAWQ calibration provenance is invalid")
    from scripts.run_nyu_qdrop_reconstruction import (
        ordered_sample_identity_sha256,
    )
    identity = ordered_sample_identity_sha256("train", indices)
    if calibration["identity_sha256"] != identity:
        raise ValueError("HAWQ calibration identity differs from indices")
    return indices, identity


def load_hawq_qat_assignment(
        path: Path,
        trace_artifact: Path,
        contract: QuantizationModelContract,
        expected_checkpoint: Path,
        expected_trace_settings,
        expected_maximum_weight_bits: float,
        expected_maximum_activation_bits: float) -> BitAssignment:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "model_name",
        "provenance",
        "calibration",
        "contract",
        "average_weight_bits",
        "average_weight_mac_bits",
        "average_activation_bits",
        "assignment",
        "objective",
        "constraints",
        "cost_basis",
        "solver_success",
        "solver_status",
    }
    if set(payload) != required:
        raise ValueError("HAWQ QAT artifact fields changed")
    if str(payload["model_name"]) != contract.model_name:
        raise ValueError("HAWQ assignment model differs from contract")
    from scripts.run_nyu_model_hawq_trace import (
        HAWQTraceSettings,
        _file_sha256,
        capture_checkpoint_identity,
        load_trace_artifact,
    )
    checkpoint = capture_checkpoint_identity(expected_checkpoint)
    expected_checkpoint_payload = {
        "path": str(checkpoint.path),
        "size_bytes": checkpoint.size_bytes,
        "sha256": checkpoint.sha256,
    }
    provenance = payload["provenance"]
    if set(provenance) != {
            "checkpoint", "trace_settings", "trace_artifact_sha256"}:
        raise ValueError("HAWQ provenance fields changed")
    if provenance["checkpoint"] != expected_checkpoint_payload:
        raise ValueError("HAWQ checkpoint identity differs")
    if isinstance(expected_trace_settings, HAWQTraceSettings):
        trace_settings = expected_trace_settings
        expected_settings = dict(
            (field, getattr(expected_trace_settings, field))
            for field in (
                "batch_size", "probes_per_batch", "seed",
                "depth_mse_weight", "boundary_mse_weight",
                "boundary_threshold_m"))
    elif isinstance(expected_trace_settings, dict):
        expected_settings = dict(expected_trace_settings)
        trace_settings = HAWQTraceSettings(**expected_settings)
    else:
        raise TypeError("expected HAWQ trace settings are invalid")
    if provenance["trace_settings"] != expected_settings:
        raise ValueError("HAWQ trace settings differ from selected config")
    trace_sha256 = provenance["trace_artifact_sha256"]
    if not isinstance(trace_sha256, str) or len(trace_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in trace_sha256):
        raise ValueError("HAWQ trace artifact fingerprint is invalid")
    trace_path = Path(trace_artifact)
    actual_trace_sha256 = _file_sha256(trace_path)
    if actual_trace_sha256 != trace_sha256:
        raise ValueError("HAWQ trace artifact fingerprint differs")
    if not isinstance(payload["solver_success"], bool) or not \
            payload["solver_success"]:
        raise ValueError("HAWQ solver success evidence is missing")
    if payload["solver_status"] not in frozenset(("optimal",)):
        raise ValueError("HAWQ solver success evidence is invalid")
    indices, calibration_identity = _validated_hawq_calibration(payload)
    trace_contract, trace_problem, trace_indices, trace_identity = \
        load_trace_artifact(
            trace_path,
            expected_model_name=contract.model_name,
            expected_checkpoint=expected_checkpoint,
            expected_calibration_indices=indices,
            expected_calibration_identity=calibration_identity,
            expected_trace_settings=trace_settings,
            bits=(4, 6, 8),
        )
    if _file_sha256(trace_path) != actual_trace_sha256:
        raise ValueError("HAWQ trace artifact changed during validation")
    if trace_contract != contract or trace_indices != indices or \
            trace_identity != calibration_identity:
        raise ValueError("HAWQ trace artifact contract identity differs")
    averages = (
        float(payload["average_weight_bits"]),
        float(payload["average_activation_bits"]),
    )
    if any(not math.isfinite(value) or value <= 0.0 or value > 6.0
           for value in averages):
        raise ValueError("HAWQ assignment exceeds the six-bit budget")
    assignment = _hawq_assignment(payload["assignment"])
    _validate_assignment_coverage(assignment, contract)
    if not set(bits for name, bits in assignment.weight_bits) <= {4, 6, 8} \
            or not set(bits for owner, bits in assignment.activation_bits) <= \
            {4, 6, 8}:
        raise ValueError("HAWQ QAT assignment contains unsupported bits")
    contract_payload = payload["contract"]
    expected_contract = {
        "blocks": list(contract.block_names),
        "protected_roles": list(contract.protected_roles),
        "protected_modules": list(contract.protected_modules),
        "attention_edges": list(contract.attention_edges),
        "concat_edges": list(contract.concat_edges),
    }
    if contract_payload != expected_contract:
        raise ValueError("HAWQ QAT artifact contract changed")
    weight_block_rows = tuple(
        (str(row["block"]), int(row["bits"]))
        for row in payload["assignment"]["weight_block_bits"])
    activation_block_rows = tuple(
        (str(row["block"]), int(row["bits"]))
        for row in payload["assignment"]["activation_block_bits"])
    expected_blocks = tuple(contract.block_names)
    if tuple(name for name, bits in weight_block_rows) != expected_blocks or \
            tuple(name for name, bits in activation_block_rows) != \
            expected_blocks:
        raise ValueError("HAWQ block assignment coverage changed")
    block_weights = dict(weight_block_rows)
    block_activations = dict(activation_block_rows)
    expected_weights = dict(
        (name, block_weights[block.name])
        for block in contract.blocks for name in block.weight_modules)
    expected_activations = dict(
        (owner, block_activations[block.name])
        for block in contract.blocks for owner in block.activation_owners)
    if dict(assignment.weight_bits) != expected_weights or \
            dict(assignment.activation_bits) != expected_activations:
        raise ValueError("HAWQ block and owner assignments differ")
    basis = payload["cost_basis"]
    if set(basis) != {
            "weight_parameters", "weight_macs", "activation_traffic"}:
        raise ValueError("HAWQ QAT cost basis fields changed")
    parameter_rows = tuple(basis["weight_parameters"])
    mac_rows = tuple(basis["weight_macs"])
    traffic_rows = tuple(basis["activation_traffic"])
    if any(set(row) != {"block", "parameters"}
           for row in parameter_rows) or any(
               set(row) != {"module", "macs"} for row in mac_rows) or any(
                   set(row) != {"site", "role", "elements"}
                   for row in traffic_rows):
        raise ValueError("HAWQ cost basis assignment fields changed")
    numeric_cost_rows = tuple(
        (row, field) for rows, field in (
            (parameter_rows, "parameters"),
            (mac_rows, "macs"),
            (traffic_rows, "elements"),
        ) for row in rows)
    if any(isinstance(row[field], bool) or not isinstance(row[field], int)
           for row, field in numeric_cost_rows):
        raise TypeError("HAWQ cost basis values must be integers")
    weight_parameters = dict(
        (str(row["block"]), int(row["parameters"]))
        for row in parameter_rows)
    weight_macs = dict(
        (str(row["module"]), int(row["macs"]))
        for row in mac_rows)
    activation_traffic = dict(
        ((str(row["site"]), str(row["role"])), int(row["elements"]))
        for row in traffic_rows)
    if set(weight_parameters) != set(contract.block_names) or len(
            parameter_rows) != len(contract.block_names) or \
            set(weight_macs) != set(contract.weight_modules) or len(
                mac_rows) != len(contract.weight_modules) or \
            set(activation_traffic) != set(_contract_owners(contract)) or len(
                traffic_rows) != len(_contract_owners(contract)):
        raise ValueError("HAWQ cost basis assignment coverage changed")
    if any(value <= 0 for value in weight_parameters.values()) or \
            any(value <= 0 for value in weight_macs.values()) or \
            any(value <= 0 for value in activation_traffic.values()):
        raise ValueError("HAWQ QAT cost basis must be positive")
    trace_parameter_rows = tuple(
        {"block": row.name, "parameters": row.weight_parameters}
        for row in trace_problem.blocks)
    trace_mac_rows = tuple(
        {"module": name, "macs": value}
        for name, value in trace_problem.weight_macs)
    trace_traffic_rows = tuple(
        {"site": owner[0], "role": owner[1], "elements": value}
        for owner, value in trace_problem.activation_traffic)
    if parameter_rows != trace_parameter_rows or mac_rows != trace_mac_rows \
            or traffic_rows != trace_traffic_rows:
        raise ValueError("HAWQ cost basis differs from trace artifact")

    parameter_average = sum(
        block_weights[name] * weight_parameters[name]
        for name in weight_parameters) / float(sum(weight_parameters.values()))
    mac_average = sum(
        dict(assignment.weight_bits)[name] * weight_macs[name]
        for name in weight_macs) / float(sum(weight_macs.values()))
    activation_average = sum(
        dict(assignment.activation_bits)[owner] * activation_traffic[owner]
        for owner in activation_traffic) / float(
            sum(activation_traffic.values()))
    reported = (
        float(payload["average_weight_bits"]),
        float(payload["average_weight_mac_bits"]),
        float(payload["average_activation_bits"]),
    )
    recomputed = (parameter_average, mac_average, activation_average)
    if any(not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
           for actual, expected in zip(reported, recomputed)):
        raise ValueError("HAWQ recomputed average bits differ from report")
    if any(value > 6.0 for value in recomputed):
        raise ValueError("HAWQ recomputed assignment exceeds six-bit budget")
    maximum_weight = float(expected_maximum_weight_bits)
    maximum_activation = float(expected_maximum_activation_bits)
    if not math.isfinite(maximum_weight) or not \
            0.0 < maximum_weight <= 6.0 or not \
            math.isfinite(maximum_activation) or not \
            0.0 < maximum_activation <= 6.0:
        raise ValueError("expected HAWQ budgets must lie in (0, 6]")
    if recomputed[0] > maximum_weight or recomputed[1] > maximum_weight or \
            recomputed[2] > maximum_activation:
        raise ValueError("HAWQ constraint assignment is infeasible")
    objective = payload["objective"]
    if set(objective) != {
            "kind", "activation_sensitivity", "total", "components",
            "selected_components"} or objective["kind"] != \
            "weight_hessian_times_squared_quantization_error" or \
            objective["activation_sensitivity"] != "not_estimated":
        raise ValueError("HAWQ objective identity changed")
    expected_component_keys = tuple(
        (block, bits) for block in contract.block_names
        for bits in (4, 6, 8))
    component_rows = tuple(objective["components"])
    if tuple((str(row["block"]), int(row["bits"]))
             for row in component_rows) != expected_component_keys:
        raise ValueError("HAWQ objective component coverage changed")
    components = {}
    for row in component_rows:
        if set(row) != {
                "block", "bits", "normalized_trace",
                "quantization_error", "cost"}:
            raise ValueError("HAWQ objective component fields changed")
        values = tuple(float(row[name]) for name in (
            "normalized_trace", "quantization_error", "cost"))
        if any(not math.isfinite(value) or value < 0.0 for value in values) or \
                not math.isclose(
                    values[2], values[0] * values[1],
                    rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("HAWQ objective component arithmetic differs")
        components[(str(row["block"]), int(row["bits"]))] = row
    if component_rows != tuple(
            vars(row) for row in trace_problem.objective_components):
        raise ValueError(
            "HAWQ objective components differ from trace artifact")
    expected_selected = tuple(
        components[(block, block_weights[block])]
        for block in contract.block_names)
    if tuple(objective["selected_components"]) != expected_selected:
        raise ValueError("HAWQ selected objective components differ")
    selected_total = sum(float(row["cost"]) for row in expected_selected)
    if not math.isfinite(float(objective["total"])) or not math.isclose(
            float(objective["total"]), selected_total,
            rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("HAWQ objective total arithmetic differs")
    constraints = payload["constraints"]
    constraint_fields = {
        "maximum_average_weight_bits",
        "maximum_average_activation_bits",
        "average_weight_parameter_bits",
        "average_weight_mac_bits",
        "average_activation_traffic_bits",
        "weight_parameter_residual",
        "weight_mac_residual",
        "activation_traffic_residual",
    }
    if set(constraints) != constraint_fields:
        raise ValueError("HAWQ constraint fields changed")
    expected_constraints = {
        "maximum_average_weight_bits": maximum_weight,
        "maximum_average_activation_bits": maximum_activation,
        "average_weight_parameter_bits": recomputed[0],
        "average_weight_mac_bits": recomputed[1],
        "average_activation_traffic_bits": recomputed[2],
        "weight_parameter_residual": maximum_weight - recomputed[0],
        "weight_mac_residual": maximum_weight - recomputed[1],
        "activation_traffic_residual": maximum_activation - recomputed[2],
    }
    if any(not math.isfinite(float(constraints[name])) or not math.isclose(
            float(constraints[name]), expected,
            rel_tol=1e-12, abs_tol=1e-12)
           for name, expected in expected_constraints.items()):
        raise ValueError("HAWQ constraint residual arithmetic differs")
    return assignment


def validate_checkpoint_payload(payload) -> None:
    if set(payload) != CHECKPOINT_FIELDS:
        raise ValueError("selected QAT checkpoint field contract mismatch")
    if int(payload["format_version"]) != 2:
        raise ValueError("selected QAT checkpoint format changed")
    if payload["method"] not in SELECTED_QAT_METHODS:
        raise ValueError("selected QAT checkpoint method is invalid")
    if payload["model_name"] not in MODEL_ORDER:
        raise ValueError("selected QAT checkpoint model is invalid")
    calibration_indices = tuple(
        int(index) for index in payload["calibration_indices"])
    validation_indices = tuple(
        int(index) for index in payload["validation_indices"])
    if len(calibration_indices) != 128 or len(calibration_indices) != len(
            set(calibration_indices)) or any(
                index < 0 for index in calibration_indices):
        raise ValueError(
            "selected QAT checkpoint calibration identities are invalid")
    if not validation_indices or len(validation_indices) != len(
            set(validation_indices)) or any(
                index < 0 for index in validation_indices):
        raise ValueError(
            "selected QAT checkpoint validation identities are invalid")
    model_state = payload["model_state"]
    if not isinstance(model_state, dict) or not model_state or any(
            "parametrizations" in name for name in model_state):
        raise ValueError(
            "selected QAT checkpoint requires canonical FP32 master weights")
    for name, value in model_state.items():
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise TypeError("selected QAT model state must contain tensors")
        is_weight = name == "weight" or name.endswith(".weight")
        if is_weight and value.dtype != torch.float32:
            raise ValueError(
                "selected QAT checkpoint weights must be finite FP32 tensors")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(
                "selected QAT checkpoint state must contain finite FP32 data")
    method_state = payload["method_state"]
    if not isinstance(method_state, dict) or not method_state:
        raise ValueError("selected QAT method state must be nonempty")
    for name, value in method_state.items():
        if not isinstance(name, str) or not torch.is_tensor(value) or \
                value.numel() == 0:
            raise TypeError("selected QAT method state must contain tensors")
        if value.is_floating_point() and not bool(
                torch.isfinite(value).all().item()):
            raise ValueError("selected QAT method state must be finite")
    epoch = int(payload["epoch"])
    validation = payload["hard_deployment_validation"]
    validate_hard_deployment_record(validation, epoch, method_state)
    _validate_run_state(payload["run_state"], payload["convergence"])
    if not isinstance(payload["deterministic_algorithms"], bool):
        raise TypeError("deterministic algorithm metadata must be boolean")


def validate_resume_contract(saved, expected) -> None:
    if saved != expected:
        raise ValueError("selected QAT resume contract changed")


def _validate_run_state(run_state, convergence=None) -> None:
    if set(run_state) != {"terminal", "completed", "reason"}:
        raise ValueError("selected QAT run state fields changed")
    if not isinstance(run_state["terminal"], bool) or not isinstance(
            run_state["completed"], bool):
        raise TypeError("selected QAT terminal state must be boolean")
    if run_state["terminal"] != run_state["completed"]:
        raise ValueError("selected QAT terminal and completed state differ")
    reason = str(run_state["reason"])
    if run_state["terminal"]:
        if reason not in ("validation_plateau", "max_epochs"):
            raise ValueError("selected QAT terminal reason is invalid")
    elif reason != "running":
        raise ValueError("selected QAT running reason is invalid")
    if convergence is not None and str(convergence["reason"]) != reason:
        raise ValueError("selected QAT run and convergence reasons differ")


def checkpoint_resume_epoch(payload):
    _validate_run_state(payload["run_state"])
    if payload["run_state"]["terminal"]:
        return None
    epoch = int(payload["epoch"])
    if epoch < 0:
        raise ValueError("selected QAT checkpoint epoch is invalid")
    return epoch + 1


def validate_training_state_subcontracts(
        payload, optimizer, scheduler, tracker) -> None:
    optimizer_state = payload["optimizer_state"]
    current_optimizer = optimizer.state_dict()
    if set(optimizer_state) != {"state", "param_groups"} or \
            set(current_optimizer) != set(optimizer_state):
        raise ValueError("selected QAT optimizer state fields changed")
    saved_groups = tuple(optimizer_state["param_groups"])
    current_groups = tuple(current_optimizer["param_groups"])
    if len(saved_groups) != len(current_groups):
        raise ValueError("selected QAT optimizer group count changed")
    serialized_to_parameter = {}
    for saved, current, live in zip(
            saved_groups, current_groups, optimizer.param_groups):
        if set(saved) != set(current):
            raise ValueError("selected QAT optimizer group fields changed")
        if list(saved["params"]) != list(current["params"]) or len(
                saved["params"]) != len(live["params"]):
            raise ValueError("selected QAT optimizer parameter topology changed")
        mutable = {"lr", "params"}
        if any(saved[name] != current[name]
               for name in set(current) - mutable):
            raise ValueError("selected QAT optimizer hyperparameters changed")
        learning_rate = saved["lr"]
        if isinstance(learning_rate, bool) or not isinstance(
                learning_rate, (int, float)) or not \
                math.isfinite(float(learning_rate)) or \
                float(learning_rate) <= 0.0:
            raise ValueError("selected QAT optimizer learning rate is invalid")
        serialized_to_parameter.update(dict(zip(
            saved["params"], live["params"])))
    if set(optimizer_state["state"]) - set(serialized_to_parameter):
        raise ValueError("selected QAT optimizer state has unknown parameters")
    for parameter_id, state in optimizer_state["state"].items():
        if set(state) != {"momentum_buffer"}:
            raise ValueError("selected QAT SGD state fields changed")
        momentum = state["momentum_buffer"]
        parameter = serialized_to_parameter[parameter_id]
        if not torch.is_tensor(momentum) or momentum.shape != parameter.shape \
                or momentum.dtype != parameter.dtype or not bool(
                    torch.isfinite(momentum).all().item()):
            raise ValueError("selected QAT optimizer momentum state is invalid")

    scheduler_state = payload["scheduler_state"]
    current_scheduler = scheduler.state_dict()
    if set(scheduler_state) != set(current_scheduler):
        raise ValueError("selected QAT scheduler state fields changed")
    mutable_scheduler = {
        "best", "num_bad_epochs", "cooldown_counter", "last_epoch",
        "_last_lr",
    }
    if any(scheduler_state[name] != current_scheduler[name]
           for name in set(current_scheduler) - mutable_scheduler):
        raise ValueError("selected QAT scheduler hyperparameters changed")
    if "_last_lr" not in scheduler_state:
        raise ValueError("selected QAT scheduler LR state is missing")
    scheduler_lrs = scheduler_state["_last_lr"]
    if len(scheduler_lrs) != len(saved_groups) or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value)) or float(value) <= 0.0 or
            not math.isclose(
                float(value), float(group["lr"]), rel_tol=0.0, abs_tol=0.0)
            for value, group in zip(scheduler_lrs, saved_groups)):
        raise ValueError("selected QAT scheduler and optimizer LR differ")
    scheduler_best = scheduler_state["best"]
    if isinstance(scheduler_best, bool) or not isinstance(
            scheduler_best, (int, float)) or math.isnan(
                float(scheduler_best)) or float(scheduler_best) < 0.0:
        raise ValueError("selected QAT scheduler best state is invalid")
    for name in ("num_bad_epochs", "cooldown_counter", "last_epoch"):
        value = scheduler_state[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("selected QAT scheduler counters are invalid")

    convergence = payload["convergence"]
    current_convergence = tracker.state_dict()
    if set(convergence) != set(current_convergence):
        raise ValueError("selected QAT convergence state fields changed")
    immutable_convergence = {
        "max_epochs", "patience", "min_relative_improvement",
    }
    if any(convergence[name] != current_convergence[name]
           for name in immutable_convergence):
        raise ValueError("selected QAT convergence hyperparameters changed")
    best_epoch = convergence["best_epoch"]
    no_improvement = convergence["no_improvement_epochs"]
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or \
            best_epoch < 0 or isinstance(no_improvement, bool) or not \
            isinstance(no_improvement, int) or no_improvement < 0:
        raise ValueError("selected QAT convergence state is invalid")
    best_rmse = float(convergence["best_rmse"])
    significant_rmse = float(convergence["significant_best_rmse"])
    if math.isnan(best_rmse) or best_rmse < 0.0 or \
            math.isnan(significant_rmse) or significant_rmse < 0.0:
        raise ValueError("selected QAT convergence state is invalid")
    if best_epoch == 0 and (
            not math.isinf(best_rmse) or not math.isinf(significant_rmse) or
            no_improvement != 0):
        raise ValueError("selected QAT initial convergence state is invalid")
    if best_epoch > 0 and (
            not math.isfinite(best_rmse) or not math.isfinite(significant_rmse)):
        raise ValueError("selected QAT convergence state is invalid")
    if "epoch" in payload:
        epoch = int(payload["epoch"])
        if best_epoch > epoch or scheduler_state["last_epoch"] != epoch:
            raise ValueError("selected QAT epoch state contracts differ")
        if payload["run_state"]["reason"] == "running" and (
                epoch >= int(convergence["max_epochs"]) or
                no_improvement >= int(convergence["patience"])):
            raise ValueError("selected QAT running convergence state is terminal")
        if payload["run_state"]["reason"] == "max_epochs" and \
                epoch < int(convergence["max_epochs"]):
            raise ValueError("selected QAT max-epoch state is inconsistent")
        if payload["run_state"]["reason"] == "validation_plateau" and \
                no_improvement < int(convergence["patience"]):
            raise ValueError("selected QAT plateau state is inconsistent")
    _validate_run_state(payload["run_state"], convergence)


def capture_rng_state(generator: torch.Generator, device: torch.device):
    device = torch.device(device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("selected QAT RNG capture requires explicit CUDA")
    return {
        "train_generator_state": generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "cuda_rng_state": torch.cuda.get_rng_state(device),
        "deterministic_algorithms":
            bool(torch.are_deterministic_algorithms_enabled()),
    }


def restore_rng_state(payload, generator: torch.Generator,
                      device: torch.device) -> None:
    required = {
        "train_generator_state",
        "torch_rng_state",
        "numpy_rng_state",
        "python_rng_state",
        "cuda_rng_state",
        "deterministic_algorithms",
    }
    if set(payload) != required:
        raise ValueError("selected QAT RNG state contract changed")
    device = torch.device(device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("selected QAT RNG restore requires explicit CUDA")
    generator.set_state(payload["train_generator_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    torch.cuda.set_rng_state(payload["cuda_rng_state"], device)
    torch.use_deterministic_algorithms(
        bool(payload["deterministic_algorithms"]))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train one official selected SPN QAT method")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_ORDER, required=True)
    parser.add_argument(
        "--method", choices=SELECTED_QAT_METHODS, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hawq-assignment", type=Path)
    parser.add_argument("--hawq-trace-artifact", type=Path)
    parser.add_argument("--p3-t3-assignment", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--validation-batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--momentum", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--scheduler-factor", type=float, required=True)
    parser.add_argument("--scheduler-patience", type=int, required=True)
    parser.add_argument("--scheduler-threshold", type=float, required=True)
    parser.add_argument("--scheduler-min-lr", type=float, required=True)
    parser.add_argument("--max-gradient-norm", type=float, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument(
        "--min-relative-improvement", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--hawq-range-momentum", type=float, required=True)
    parser.add_argument("--depth-loss-weight", type=float, required=True)
    parser.add_argument("--boundary-loss-weight", type=float, required=True)
    parser.add_argument("--teacher-loss-weight", type=float, required=True)
    parser.add_argument(
        "--initial-depth-loss-weight", type=float, required=True)
    parser.add_argument(
        "--propagation-loss-weight", type=float, required=True)
    parser.add_argument("--boundary-threshold-m", type=float, required=True)
    parser.add_argument("--fold-max-error", type=float, required=True)
    parser.add_argument(
        "--joint-clip-factors", type=float, nargs="+", required=True)
    parser.add_argument("--joint-search-rounds", type=int, required=True)
    parser.add_argument(
        "--joint-cache-sample-limit", type=int, required=True)
    parser.add_argument("--joint-cache-byte-limit", type=int, required=True)
    parser.add_argument("--log-interval", type=int, required=True)
    return parser


def _training_config(args):
    positive_integers = (
        args.epochs,
        args.batch_size,
        args.validation_batch_size,
        args.patience,
        args.scheduler_patience,
        args.joint_search_rounds,
        args.joint_cache_sample_limit,
        args.joint_cache_byte_limit,
        args.log_interval,
    )
    if any(int(value) <= 0 for value in positive_integers):
        raise ValueError("selected QAT integer settings must be positive")
    if int(args.workers) < 0 or int(args.seed) < 0:
        raise ValueError("selected QAT workers and seed must be nonnegative")
    positive_floats = (
        args.learning_rate,
        args.max_gradient_norm,
        args.scheduler_factor,
        args.scheduler_threshold,
        args.scheduler_min_lr,
        args.boundary_threshold_m,
    ) + tuple(args.joint_clip_factors)
    if any(not math.isfinite(float(value)) or float(value) <= 0.0
           for value in positive_floats):
        raise ValueError("selected QAT positive settings are invalid")
    if not 0.0 < float(args.scheduler_factor) < 1.0:
        raise ValueError("selected QAT scheduler factor must lie in (0, 1)")
    if not math.isfinite(float(args.momentum)) or not \
            0.0 <= float(args.momentum) < 1.0:
        raise ValueError("selected QAT momentum must lie in [0, 1)")
    if not math.isfinite(float(args.weight_decay)) or \
            float(args.weight_decay) < 0.0:
        raise ValueError("selected QAT weight decay must be nonnegative")
    if not math.isfinite(float(args.min_relative_improvement)) or \
            float(args.min_relative_improvement) < 0.0:
        raise ValueError(
            "selected QAT minimum improvement must be nonnegative")
    if not math.isfinite(float(args.hawq_range_momentum)) or not \
            0.0 <= float(args.hawq_range_momentum) < 1.0:
        raise ValueError("selected QAT HAWQ momentum must lie in [0, 1)")
    loss_weights = (
        args.depth_loss_weight,
        args.boundary_loss_weight,
        args.teacher_loss_weight,
        args.initial_depth_loss_weight,
        args.propagation_loss_weight,
    )
    if any(not math.isfinite(float(value)) or float(value) < 0.0
           for value in loss_weights):
        raise ValueError("selected QAT loss weights must be nonnegative")
    if not math.isfinite(float(args.fold_max_error)) or \
            float(args.fold_max_error) < 0.0:
        raise ValueError("selected QAT fold error must be nonnegative")
    return {
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "validation_batch_size": int(args.validation_batch_size),
        "workers": int(args.workers),
        "learning_rate": float(args.learning_rate),
        "momentum": float(args.momentum),
        "weight_decay": float(args.weight_decay),
        "scheduler_factor": float(args.scheduler_factor),
        "scheduler_patience": int(args.scheduler_patience),
        "scheduler_threshold": float(args.scheduler_threshold),
        "scheduler_min_lr": float(args.scheduler_min_lr),
        "max_gradient_norm": float(args.max_gradient_norm),
        "patience": int(args.patience),
        "min_relative_improvement":
            float(args.min_relative_improvement),
        "seed": int(args.seed),
        "hawq_range_momentum": float(args.hawq_range_momentum),
        "depth_loss_weight": float(args.depth_loss_weight),
        "boundary_loss_weight": float(args.boundary_loss_weight),
        "teacher_loss_weight": float(args.teacher_loss_weight),
        "initial_depth_loss_weight":
            float(args.initial_depth_loss_weight),
        "propagation_loss_weight": float(args.propagation_loss_weight),
        "boundary_threshold_m": float(args.boundary_threshold_m),
        "fold_max_error": float(args.fold_max_error),
        "joint_clip_factors": tuple(
            float(value) for value in args.joint_clip_factors),
        "joint_search_rounds": int(args.joint_search_rounds),
        "joint_cache_sample_limit": int(args.joint_cache_sample_limit),
        "joint_cache_byte_limit": int(args.joint_cache_byte_limit),
        "log_interval": int(args.log_interval),
    }


def _assignment_payload(assignment: BitAssignment):
    return {
        "model_name": assignment.model_name,
        "weight_bits": [list(row) for row in assignment.weight_bits],
        "activation_bits": [
            [list(owner), bits]
            for owner, bits in assignment.activation_bits],
    }


def _contract_manifest(contract: QuantizationModelContract):
    return {
        "model_name": contract.model_name,
        "blocks": [{
            "name": block.name,
            "weight_modules": list(block.weight_modules),
            "activation_owners": [
                list(owner) for owner in block.activation_owners],
        } for block in contract.blocks],
        "protected_roles": list(contract.protected_roles),
        "protected_modules": list(contract.protected_modules),
        "attention_edges": list(contract.attention_edges),
        "concat_edges": list(contract.concat_edges),
    }


def _p3_t3_tuple_assignment(payload) -> BitAssignment:
    if set(payload) != {"model_name", "weight_bits", "activation_bits"}:
        raise ValueError("P3/T3 tuple assignment fields changed")
    weight_rows = tuple(payload["weight_bits"])
    activation_rows = tuple(payload["activation_bits"])
    if any(not isinstance(row, (list, tuple)) or len(row) != 2 or
           isinstance(row[1], bool) or not isinstance(row[1], int)
           for row in weight_rows) or any(
               not isinstance(row, (list, tuple)) or len(row) != 2 or
               not isinstance(row[0], (list, tuple)) or len(row[0]) != 2 or
               isinstance(row[1], bool) or not isinstance(row[1], int)
               for row in activation_rows):
        raise ValueError("P3/T3 tuple assignment rows changed")
    return BitAssignment(
        weight_bits=tuple(
            (str(row[0]), int(row[1])) for row in weight_rows),
        activation_bits=tuple(
            ((str(row[0][0]), str(row[0][1])), int(row[1]))
            for row in activation_rows),
        model_name=str(payload["model_name"]),
    )


def _normalized_p3_t3_costs(assignment, costs, base_weight, base_activation):
    weight_bits = dict(assignment.weight_bits)
    activation_bits = dict(assignment.activation_bits)
    weight_costs = dict(costs.weight_macs)
    activation_costs = dict(costs.activation_elements)
    if set(weight_bits) != set(weight_costs) or \
            set(activation_bits) != set(activation_costs):
        raise ValueError("P3/T3 assignment and cost coverage differ")
    weight_denominator = int(base_weight) * sum(weight_costs.values())
    activation_denominator = int(base_activation) * \
        sum(activation_costs.values())
    if weight_denominator <= 0 or activation_denominator <= 0:
        raise ValueError("P3/T3 cost denominator must be positive")
    weight = sum(
        weight_bits[name] * weight_costs[name] for name in weight_costs
    ) / float(weight_denominator)
    activation = sum(
        activation_bits[owner] * activation_costs[owner]
        for owner in activation_costs
    ) / float(activation_denominator)
    return weight, activation, weight_denominator, activation_denominator


def load_p3_t3_qat_assignment(
        path: Path,
        contract: QuantizationModelContract,
        precision,
        expected_checkpoint: Path,
        expected_evaluation_indices: Sequence[int]):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "format_version", "artifact_kind",
        "model_name", "source_checkpoint", "prefix", "tail",
        "selected_candidate", "precision", "budgets", "expected_samples",
        "evaluation", "cost_definition", "cost_basis", "assignment",
        "candidates",
    }
    if set(payload) != required:
        raise ValueError("P3/T3 QAT artifact fields changed")
    if isinstance(payload["format_version"], bool) or not isinstance(
            payload["format_version"], int) or \
            payload["format_version"] != 2:
        raise ValueError("P3/T3 QAT artifact version is unsupported")
    if payload["artifact_kind"] != "nyu_model_p3_t3_assignment":
        raise ValueError("P3/T3 QAT artifact kind differs")
    if str(payload["model_name"]) != contract.model_name:
        raise ValueError("P3/T3 artifact model differs from contract")
    from scripts.run_nyu_model_hawq_trace import capture_checkpoint_identity
    checkpoint = capture_checkpoint_identity(expected_checkpoint)
    expected_checkpoint_payload = {
        "path": str(checkpoint.path),
        "size_bytes": checkpoint.size_bytes,
        "sha256": checkpoint.sha256,
    }
    if payload["source_checkpoint"] != expected_checkpoint_payload:
        raise ValueError("P3/T3 checkpoint identity differs")
    precision_fields = {
        "base_weight_bits", "base_activation_bits",
        "promotion_weight_bits", "promotion_activation_bits",
    }
    expected_precision = dict(
        (name, int(precision[name])) for name in precision_fields)
    if set(payload["precision"]) != precision_fields or dict(
            (name, int(payload["precision"][name]))
            for name in precision_fields) != expected_precision:
        raise ValueError("P3/T3 precision differs from selected config")
    basis = payload["cost_basis"]
    if set(basis) != {"activation_elements", "weight_macs"}:
        raise ValueError("P3/T3 cost basis fields changed")
    weight_cost_rows = tuple(basis["weight_macs"])
    activation_cost_rows = tuple(basis["activation_elements"])
    if any(not isinstance(row, (list, tuple)) or len(row) != 2 or
           isinstance(row[1], bool) or not isinstance(row[1], int)
           for row in weight_cost_rows) or any(
               not isinstance(row, (list, tuple)) or len(row) != 2 or
               not isinstance(row[0], (list, tuple)) or len(row[0]) != 2 or
               isinstance(row[1], bool) or not isinstance(row[1], int)
               for row in activation_cost_rows):
        raise ValueError("P3/T3 cost basis rows changed")
    costs = CostBasis(
        weight_macs=tuple(
            (str(row[0]), int(row[1])) for row in weight_cost_rows),
        activation_elements=tuple(
            ((str(row[0][0]), str(row[0][1])), int(row[1]))
            for row in activation_cost_rows),
    )
    weight_cost_names = tuple(name for name, cost in costs.weight_macs)
    activation_cost_owners = tuple(
        owner for owner, cost in costs.activation_elements)
    if set(weight_cost_names) != set(contract.weight_modules) or \
            len(weight_cost_names) != len(contract.weight_modules) or \
            set(activation_cost_owners) != set(_contract_owners(contract)) or \
            len(activation_cost_owners) != len(_contract_owners(contract)):
        raise ValueError("P3/T3 cost basis contract coverage differs")
    base_weight = expected_precision["base_weight_bits"]
    base_activation = expected_precision["base_activation_bits"]
    root_assignment = _p3_t3_tuple_assignment(payload["assignment"])
    _validate_assignment_coverage(root_assignment, contract)
    from scripts.run_nyu_model_p3t3_search import (
        P3T3CandidateResult,
        P3T3SampleEvidence,
        _prefix_knee,
        build_p3_t3_candidates,
    )
    from spn_quant.mixed_precision import build_registry
    expected_candidates = build_p3_t3_candidates(
        contract,
        build_registry(contract, costs),
        base_weight,
        base_activation,
        expected_precision["promotion_weight_bits"],
        expected_precision["promotion_activation_bits"],
    )
    candidate_rows = tuple(payload["candidates"])
    if len(candidate_rows) != len(expected_candidates):
        raise ValueError("P3/T3 candidate evidence coverage differs")
    candidate_fields = {
        "name", "stage", "prefix", "tail", "pooled_rmse",
        "mean_sample_rmse", "normalized_weight_cost",
        "normalized_activation_cost", "valid", "metrics_finite",
        "sample_rmse", "sample_evidence", "paired_sample_differences",
        "assignment",
    }
    raw_evaluation_indices = tuple(expected_evaluation_indices)
    if any(isinstance(index, bool) or not isinstance(index, int)
           for index in raw_evaluation_indices):
        raise TypeError("P3/T3 evaluation identities must be integers")
    evaluation_indices = tuple(int(index)
                               for index in raw_evaluation_indices)
    if not evaluation_indices or len(evaluation_indices) != len(
            set(evaluation_indices)):
        raise ValueError("P3/T3 expected evaluation identities are invalid")
    if isinstance(payload["expected_samples"], bool) or not isinstance(
            payload["expected_samples"], int) or \
            payload["expected_samples"] != len(evaluation_indices):
        raise ValueError("P3/T3 expected sample count differs")
    evaluation = payload["evaluation"]
    if not isinstance(evaluation, dict) or set(evaluation) != {
            "split", "count", "indices", "identity_sha256"}:
        raise ValueError("P3/T3 evaluation identity fields changed")
    encoded_identity = json.dumps(
        [["val", index] for index in evaluation_indices],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    expected_evaluation = {
        "split": "val",
        "count": len(evaluation_indices),
        "indices": list(evaluation_indices),
        "identity_sha256": hashlib.sha256(encoded_identity).hexdigest(),
    }
    if evaluation != expected_evaluation:
        raise ValueError("P3/T3 evaluation identity differs")
    parsed = []
    baseline_samples = None
    for expected, row in zip(expected_candidates, candidate_rows):
        if set(row) != candidate_fields or str(row["name"]) != expected.name \
                or str(row["stage"]) != expected.stage or \
                tuple(row["prefix"]) != expected.prefix or \
                tuple(row["tail"]) != expected.tail:
            raise ValueError("P3/T3 candidate identity evidence differs")
        candidate_assignment = _p3_t3_tuple_assignment(row["assignment"])
        if candidate_assignment != expected.assignment:
            raise ValueError("P3/T3 candidate assignment evidence differs")
        weight_cost, activation_cost, weight_denominator, \
            activation_denominator = _normalized_p3_t3_costs(
                candidate_assignment, costs, base_weight, base_activation)
        if not math.isclose(
                float(row["normalized_weight_cost"]), weight_cost,
                rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("P3/T3 weight cost audit differs")
        if not math.isclose(
                float(row["normalized_activation_cost"]), activation_cost,
                rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("P3/T3 activation cost audit differs")
        sample_rows = tuple(row["sample_rmse"])
        if any(not isinstance(sample, (list, tuple)) or len(sample) != 2 or
               isinstance(sample[0], bool) or not isinstance(sample[0], int)
               for sample in sample_rows):
            raise ValueError("P3/T3 candidate sample row fields changed")
        if tuple(int(sample[0]) for sample in sample_rows) != \
                evaluation_indices:
            raise ValueError("P3/T3 candidate sample evidence differs")
        evidence_rows = tuple(row["sample_evidence"])
        evidence_fields = {
            "sample_index", "squared_error_sum", "valid_pixels",
            "prediction_finite", "propagation_valid", "reproducible",
        }
        if len(evidence_rows) != len(evaluation_indices) or any(
                not isinstance(sample, dict) or
                set(sample) != evidence_fields
                for sample in evidence_rows):
            raise ValueError("P3/T3 raw sample evidence fields changed")
        if tuple(sample["sample_index"] for sample in evidence_rows) != \
                evaluation_indices or any(
                    isinstance(sample["sample_index"], bool) or
                    not isinstance(sample["sample_index"], int)
                    for sample in evidence_rows):
            raise ValueError("P3/T3 raw sample identities differ")
        valid = row["valid"]
        metrics_finite = row["metrics_finite"]
        if not isinstance(valid, bool) or not isinstance(metrics_finite, bool):
            raise TypeError("P3/T3 candidate validity must be boolean")
        if len(row["paired_sample_differences"]) != len(evaluation_indices):
            raise ValueError("P3/T3 paired candidate evidence differs")
        evidence = []
        sample_values = []
        raw_finite = True
        raw_valid = True
        squared_total = 0.0
        pixel_total = 0
        for raw, reported in zip(evidence_rows, sample_rows):
            pixels = raw["valid_pixels"]
            flags = (
                raw["prediction_finite"],
                raw["propagation_valid"],
                raw["reproducible"],
            )
            if isinstance(pixels, bool) or not isinstance(pixels, int) or \
                    pixels <= 0:
                raise ValueError(
                    "P3/T3 raw valid pixels must be positive integers")
            if not all(isinstance(value, bool) for value in flags):
                raise TypeError("P3/T3 raw validity flags must be booleans")
            squared_value = raw["squared_error_sum"]
            if squared_value is None:
                squared = float("inf")
                finite = False
            else:
                if isinstance(squared_value, bool) or not isinstance(
                        squared_value, (int, float)):
                    raise TypeError(
                        "P3/T3 raw squared error must be numeric or null")
                squared = float(squared_value)
                if not math.isfinite(squared) or squared < 0.0:
                    raise ValueError(
                        "P3/T3 raw squared error must be nonnegative")
                finite = True
            derived_sample = math.sqrt(squared / float(pixels)) \
                if finite else float("inf")
            reported_value = reported[1]
            if finite:
                if isinstance(reported_value, bool) or not isinstance(
                        reported_value, (int, float)) or not math.isfinite(
                            float(reported_value)) or float(
                                reported_value) < 0.0 or not math.isclose(
                                    float(reported_value), derived_sample,
                                    rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(
                        "P3/T3 sample RMSE evidence differs from raw sums")
            elif reported_value is not None:
                raise ValueError(
                    "P3/T3 sample RMSE evidence differs from raw sums")
            evidence.append(P3T3SampleEvidence(
                sample_index=int(raw["sample_index"]),
                squared_error_sum=squared,
                valid_pixels=pixels,
                prediction_finite=flags[0],
                propagation_valid=flags[1],
                reproducible=flags[2],
            ))
            sample_values.append(derived_sample)
            raw_finite = raw_finite and finite
            raw_valid = raw_valid and finite and all(flags)
            if finite:
                squared_total += squared
            pixel_total += pixels
        pooled_rmse = math.sqrt(squared_total / float(pixel_total)) \
            if raw_finite else float("inf")
        mean_sample_rmse = sum(sample_values) / float(len(sample_values)) \
            if raw_finite else float("inf")
        for name, reported_value, derived_value in (
                ("pooled RMSE", row["pooled_rmse"], pooled_rmse),
                ("mean sample RMSE", row["mean_sample_rmse"],
                 mean_sample_rmse)):
            if math.isfinite(derived_value):
                if isinstance(reported_value, bool) or not isinstance(
                        reported_value, (int, float)) or not math.isfinite(
                            float(reported_value)) or float(
                                reported_value) < 0.0 or not math.isclose(
                                    float(reported_value), derived_value,
                                    rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(
                        "P3/T3 %s evidence differs from raw sums" % name)
            elif reported_value is not None:
                raise ValueError(
                    "P3/T3 %s evidence differs from raw sums" % name)
        if valid != raw_valid:
            raise ValueError("P3/T3 candidate validity evidence differs")
        sample_values = tuple(sample_values)
        if baseline_samples is None:
            baseline_samples = sample_values
        expected_differences = tuple(
            value - baseline
            if math.isfinite(value) and math.isfinite(baseline)
            else float("inf")
            for value, baseline in zip(sample_values, baseline_samples))
        for actual, expected_difference in zip(
                row["paired_sample_differences"], expected_differences):
            if math.isfinite(expected_difference):
                if isinstance(actual, bool) or not isinstance(
                        actual, (int, float)) or not math.isfinite(
                            float(actual)) or not math.isclose(
                                float(actual), expected_difference,
                                rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(
                        "P3/T3 paired sample evidence differs")
            elif actual is not None:
                raise ValueError("P3/T3 paired sample evidence differs")
        actual_finite = raw_finite and all(
            math.isfinite(value) for value in expected_differences)
        if metrics_finite != actual_finite:
            raise ValueError("P3/T3 candidate stability evidence differs")
        parsed.append(P3T3CandidateResult(
            name=expected.name,
            stage=expected.stage,
            prefix=expected.prefix,
            tail=expected.tail,
            assignment=candidate_assignment,
            pooled_rmse=pooled_rmse,
            mean_sample_rmse=mean_sample_rmse,
            normalized_weight_cost=weight_cost,
            normalized_activation_cost=activation_cost,
            valid=valid,
            sample_rmse=tuple(zip(evaluation_indices, sample_values)),
            sample_evidence=tuple(evidence),
            paired_sample_differences=expected_differences,
        ))
    if not parsed[0].valid or not math.isfinite(parsed[0].pooled_rmse):
        raise ValueError("P3/T3 evidence lacks a stable finite baseline")
    budgets = payload["budgets"]
    if set(budgets) != {
            "maximum_normalized_weight_cost",
            "maximum_normalized_activation_cost"}:
        raise ValueError("P3/T3 budget fields changed")
    maximum_weight = float(budgets["maximum_normalized_weight_cost"])
    maximum_activation = float(
        budgets["maximum_normalized_activation_cost"])
    if not math.isfinite(maximum_weight) or maximum_weight <= 0.0 or \
            not math.isfinite(maximum_activation) or maximum_activation <= 0.0:
        raise ValueError("P3/T3 budgets must be finite and positive")
    selected_name = str(payload["selected_candidate"])
    selected_rows = tuple(row for row in parsed if row.name == selected_name)
    if len(selected_rows) != 1:
        raise ValueError("P3/T3 selected candidate evidence is not unique")
    selected = selected_rows[0]
    if root_assignment != selected.assignment:
        raise ValueError("P3/T3 root and selected candidate assignment differ")
    if tuple(payload["prefix"]) != selected.prefix or \
            tuple(payload["tail"]) != selected.tail:
        raise ValueError("P3/T3 root selection labels differ from evidence")
    if not selected.valid or not math.isfinite(selected.pooled_rmse) or \
            not math.isfinite(selected.mean_sample_rmse):
        raise ValueError("P3/T3 selected candidate must be stable and finite")
    if selected.normalized_weight_cost > maximum_weight:
        raise ValueError("P3/T3 selected candidate exceeds weight budget")
    if selected.normalized_activation_cost > maximum_activation:
        raise ValueError("P3/T3 selected candidate exceeds activation budget")
    prefix = _prefix_knee(tuple(parsed))
    eligible = tuple(
        row for row in parsed
        if row.stage == "interaction" and row.prefix == prefix.prefix and
        row.valid and row.normalized_weight_cost <= maximum_weight and
        row.normalized_activation_cost <= maximum_activation)
    if not eligible:
        raise ValueError("P3/T3 candidate evidence has no eligible selection")
    measured_selection = min(eligible, key=lambda row: (
        row.pooled_rmse,
        row.mean_sample_rmse,
        row.normalized_weight_cost,
        row.normalized_activation_cost,
        row.tail,
        row.name,
    ))
    if measured_selection.name != selected_name:
        raise ValueError("P3/T3 selected candidate differs from evidence")
    definition = payload["cost_definition"]
    expected_definition = {
        "activation_denominator": activation_denominator,
        "activation_formula":
            "sum(activation_bits*elements)/activation_denominator",
        "weight_denominator": weight_denominator,
        "weight_formula": "sum(weight_bits*macs)/weight_denominator",
    }
    if definition != expected_definition:
        raise ValueError("P3/T3 cost definition arithmetic differs")
    return root_assignment, costs, {
        "normalized_weight_cost": selected.normalized_weight_cost,
        "maximum_normalized_weight_cost": maximum_weight,
        "weight_feasible": 1,
        "normalized_activation_cost": selected.normalized_activation_cost,
        "maximum_normalized_activation_cost": maximum_activation,
        "activation_feasible": 1,
    }


def _selected_assignment(args, selected, model_config, contract):
    method = args.method
    method_config = selected.method_hyperparameters[method]
    if method == "lsqplus_w4a4":
        if (int(method_config["weight_bits"]),
                int(method_config["activation_bits"]),
                int(method_config["initialization_count"])) != (4, 4, 128):
            raise ValueError("selected LSQ++ W4A4 configuration changed")
        return uniform_qat_assignment(contract, 4), None
    if method == "lsqplus_w6a6":
        if (int(method_config["weight_bits"]),
                int(method_config["activation_bits"]),
                int(method_config["initialization_count"])) != (6, 6, 128):
            raise ValueError("selected LSQ++ W6A6 configuration changed")
        return uniform_qat_assignment(contract, 6), None
    if method == "hawq_mixed_le6":
        if int(method_config["calibration_count"]) != 128:
            raise ValueError("selected HAWQ calibration count changed")
        if tuple(int(value) for value in method_config["bits"]) != (4, 6, 8):
            raise ValueError("selected HAWQ candidate bits changed")
        assignment = load_hawq_qat_assignment(
            args.hawq_assignment,
            args.hawq_trace_artifact,
            contract,
            model_config.checkpoint,
            dict(method_config["trace"]),
            float(method_config["maximum_average_weight_bits"]),
            float(method_config["maximum_average_activation_bits"]),
        )
        payload = json.loads(
            Path(args.hawq_assignment).read_text(encoding="utf-8"))
        maxima = (
            float(method_config["maximum_average_weight_bits"]),
            float(method_config["maximum_average_activation_bits"]),
        )
        actual = (
            float(payload["average_weight_bits"]),
            float(payload["average_activation_bits"]),
        )
        if actual[0] > maxima[0] or actual[1] > maxima[1]:
            raise ValueError("HAWQ assignment exceeds configured budget")
        return assignment, None
    mixed_config = selected.method_hyperparameters["mixed_task_aware"]
    if tuple(int(value) for value in mixed_config["weight_bits"]) != (4, 8) \
            or tuple(int(value) for value in
                     mixed_config["activation_bits"]) != (4, 6, 8):
        raise ValueError("mixed task-aware precision choices changed")
    p3_t3, costs, p3_t3_audit = load_p3_t3_qat_assignment(
        args.p3_t3_assignment,
        contract,
        selected.method_hyperparameters["p3_t3_mixed_ptq"],
        model_config.checkpoint,
        model_config.evaluation_indices,
    )
    assignment, activation_audit = mixed_task_aware_assignment(
        contract,
        p3_t3,
        p3_t3.activation_bits,
        costs,
        float(mixed_config["maximum_average_activation_bits"]),
    )
    audit = dict(p3_t3_audit)
    audit.update({
        "average_activation_bits": activation_audit.average_activation_bits,
        "maximum_activation_bits":
            activation_audit.maximum_activation_bits,
        "activation_numerator": activation_audit.activation_numerator,
        "activation_denominator": activation_audit.activation_denominator,
        "average_activation_feasible": int(activation_audit.feasible),
    })
    return assignment, audit


def _propagation_config():
    from spn_quant.propagation import PropagationQuantConfig
    return PropagationQuantConfig(
        affinity_bits=8,
        confidence_bits=8,
        offset_bits=8,
        state_bits=8,
        coefficient_fraction_bits=13,
    )


def _calibration_indices(model_config, dataset_size: int):
    payload = json.loads(
        Path(model_config.calibration_metadata).read_text(encoding="utf-8"))
    indices = tuple(int(index) for index in payload["calibration_indices"])
    evaluation = tuple(
        int(index) for index in payload["evaluation_indices"])
    if len(indices) != int(model_config.calibration_count) or \
            len(indices) != 128 or len(indices) != len(set(indices)):
        raise ValueError(
            "selected QAT requires 128 unique calibration identities")
    if any(index < 0 or index >= int(dataset_size) for index in indices):
        raise ValueError("selected QAT calibration identity is out of range")
    if evaluation != tuple(model_config.evaluation_indices):
        raise ValueError("selected QAT evaluation identities changed")
    if payload["calibration_source"]["selection"] != \
            "32_tail_96_kmedoids":
        raise ValueError("selected QAT requires stratified calibration")
    return indices


def _selected_target_plan(model_name, model, contract):
    from spn_quant.qdrop_targets import resolve_qdrop_targets
    resolved = resolve_qdrop_targets(model_name, model)
    owners = set(_contract_owners(contract))
    sites = tuple(
        site for site in resolved.activation_sites
        if (site.site, site.role) in owners)
    return QDropTargetPlan(
        model=model_name,
        blocks=tuple(contract.block_names),
        activation_sites=sites,
        excluded_sites=resolved.excluded_sites,
    )


def _sample_batch(dataset, index: int, seed: int):
    from scripts.run_nyu_rtn_quantization import (
        batch_from_sample,
        seeded_sample,
    )
    return batch_from_sample(seeded_sample(dataset, index, seed))


def _joint_adapter(model, contract, training):
    if not contract.attention_edges and not contract.concat_edges:
        return None
    if contract.model_name != "completionformer":
        raise ValueError("joint QAT owners require CompletionFormer")
    if len(contract.attention_edges) % 3 or len(contract.concat_edges) % 2:
        raise ValueError("CompletionFormer joint owner cardinality changed")
    from spn_quant.adapters.completionformer_joint import (
        CompletionFormerJointAdapter,
    )
    return CompletionFormerJointAdapter(
        model=model,
        expected_attention_modules=len(contract.attention_edges) // 3,
        expected_concat_modules=len(contract.concat_edges) // 2,
        weight_bits=4,
        qkv_bits=4,
        probability_bits=8,
        concat_bits=4,
        output_bits=4,
        clip_factors=training["joint_clip_factors"],
        search_rounds=training["joint_search_rounds"],
        cache_sample_limit=training["joint_cache_sample_limit"],
        cache_byte_limit=training["joint_cache_byte_limit"],
    )


class PreparedSelectedQAT(object):
    def __init__(self, student_runtime, teacher_runtime, model, teacher,
                 contract, assignment, budget_audit, controller,
                 propagation, joint, student_semantic, teacher_semantic,
                 trainset, valset, calibration_indices,
                 graph_preparation) -> None:
        self.student_runtime = student_runtime
        self.teacher_runtime = teacher_runtime
        self.model = model
        self.teacher = teacher
        self.contract = contract
        self.assignment = assignment
        self.budget_audit = budget_audit
        self.controller = controller
        self.propagation = propagation
        self.joint = joint
        self.student_semantic = student_semantic
        self.teacher_semantic = teacher_semantic
        self.trainset = trainset
        self.valset = valset
        self.calibration_indices = tuple(calibration_indices)
        self.graph_preparation = graph_preparation
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        self.teacher_semantic.close()
        self.student_semantic.close()
        self.controller.remove()
        self.propagation.close()
        if self.joint is not None:
            self.joint.close()
        self.teacher_runtime.close()
        self.student_runtime.close()
        self.closed = True


class MaterializedDeploymentContext(object):
    def __init__(self, resources, runtime, model, controller,
                 propagation) -> None:
        self.resources = resources
        self.student_runtime = runtime
        self.model = model
        self.controller = controller
        self.propagation = propagation
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        self.resources.close()
        self.closed = True


def build_materialized_deployment_context(
        prepared, model_config, training, hard_state, method_state,
        deployment_qparams, device):
    from scripts.hardware_aligned_quantization import prepare_hardware_model
    from scripts.nyu_model_runtime import NYUModelRuntime
    from spn_quant.model_contracts import build_model_quantization_contract
    from spn_quant.propagation import install_propagation_adapter
    from spn_quant.qat.model_methods import ModelHardDeploymentController

    if tensor_state_sha256(method_state) != tensor_state_sha256(
            prepared.controller.method_state_dict()):
        raise ValueError("hard deployment method state differs from training")
    if deployment_qparams != prepared.controller.deployment_qparams():
        raise ValueError(
            "hard deployment qparams differ from exact method state")
    if tensor_state_sha256(hard_state) != tensor_state_sha256(
            prepared.controller.hard_model_state_dict()):
        raise ValueError("hard deployment weights differ from training")

    with ExitStack() as stack:
        runtime = NYUModelRuntime.from_config(model_config)
        stack.callback(runtime.close)
        model = runtime.build_model(device)
        contract = build_model_quantization_contract(model_config.model, model)
        if contract != prepared.contract:
            raise ValueError("hard deployment contract differs from training")
        trainset = runtime.build_dataset("train")
        first = _sample_batch(
            trainset, prepared.calibration_indices[0], training["seed"])
        model_args, target = runtime.model_input(first, device)
        del target
        preparation = prepare_hardware_model(model, model_args, fold=True)
        if float(preparation["primary_max_abs_error"]) > \
                training["fold_max_error"]:
            raise RuntimeError(
                "hard deployment Conv-BN fold exceeds threshold")
        expected_graph = {
            "folded_pairs": prepared.graph_preparation["folded_pairs"],
            "unfolded_fanout_pairs":
                prepared.graph_preparation["unfolded_fanout_pairs"],
            "unfolded_conv_bn_pairs":
                prepared.graph_preparation["unfolded_conv_bn_pairs"],
        }
        actual_graph = dict(
            (name, list(preparation[name])) for name in expected_graph)
        if actual_graph != expected_graph:
            raise ValueError("hard deployment graph preparation differs")
        model.load_state_dict(hard_state, strict=True)
        target_plan = _selected_target_plan(model_config.model, model, contract)
        joint = _joint_adapter(model, contract, training)
        if joint is not None:
            stack.callback(joint.close)
        propagation = install_propagation_adapter(model_config.model, model)
        stack.callback(propagation.close)
        controller = ModelHardDeploymentController(
            model,
            contract,
            target_plan,
            prepared.controller.config,
            deployment_qparams,
            joint_adapter=joint,
            propagation_adapter=propagation,
        )
        controller.install()
        stack.callback(controller.remove)
        model.eval()
        controller.activation_modules.eval()
        resources = stack.pop_all()
    return MaterializedDeploymentContext(
        resources, runtime, model, controller, propagation)


def prepare_selected_qat(args, selected, model_config, training):
    from scripts.hardware_aligned_quantization import prepare_hardware_model
    from scripts.nyu_model_runtime import NYUModelRuntime
    from spn_quant.adapters import install_model_semantic_adapter
    from spn_quant.model_contracts import build_model_quantization_contract
    from spn_quant.propagation import install_propagation_adapter
    from spn_quant.qat.model_methods import (
        ModelMethodQATConfig,
        ModelMethodQATController,
    )

    device = torch.device(model_config.device)
    student_runtime = NYUModelRuntime.from_config(model_config)
    teacher_runtime = NYUModelRuntime.from_config(model_config)
    model = student_runtime.build_model(device)
    teacher = teacher_runtime.build_model(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    contract = build_model_quantization_contract(model_config.model, model)
    target_plan = _selected_target_plan(model_config.model, model, contract)
    assignment, budget_audit = _selected_assignment(
        args, selected, model_config, contract)
    trainset = student_runtime.build_dataset("train")
    valset = student_runtime.build_dataset("val")
    indices = _calibration_indices(model_config, len(trainset))
    if args.method == "hawq_mixed_le6":
        hawq_payload = json.loads(
            Path(args.hawq_assignment).read_text(encoding="utf-8"))
        hawq_indices = tuple(
            int(index) for index in hawq_payload["calibration"]["indices"])
        if hawq_indices != indices:
            raise ValueError(
                "HAWQ and QAT calibration identities differ")
    first = _sample_batch(trainset, indices[0], training["seed"])
    model_args, target = student_runtime.model_input(first, device)
    del target
    preparation = prepare_hardware_model(model, model_args, fold=True)
    if float(preparation["primary_max_abs_error"]) > \
            training["fold_max_error"]:
        raise RuntimeError("selected QAT Conv-BN fold exceeds threshold")
    graph_preparation = {
        "folded_pairs": list(preparation["folded_pairs"]),
        "unfolded_fanout_pairs": list(
            preparation["unfolded_fanout_pairs"]),
        "unfolded_conv_bn_pairs": list(
            preparation["unfolded_conv_bn_pairs"]),
        "primary_max_abs_error":
            float(preparation["primary_max_abs_error"]),
    }
    joint = _joint_adapter(model, contract, training)
    propagation = install_propagation_adapter(model_config.model, model)
    collector = ModelActivationRangeCollector(model, contract, target_plan)
    propagation.observe()
    if joint is not None:
        joint.observe_qdrop_ranges()
    model.eval()
    with torch.no_grad():
        for index in indices:
            batch = _sample_batch(trainset, index, training["seed"])
            calibration_args, calibration_target = \
                student_runtime.model_input(batch, device)
            del calibration_target
            model(*calibration_args)
    propagation.freeze()
    propagation.configure(_propagation_config())
    if joint is not None:
        joint.freeze_qdrop_ranges()
    initialization_rows = collector.initialization_rows(joint)
    collector.close()
    internal_method = {
        "lsqplus_w4a4": "lsqplus",
        "lsqplus_w6a6": "lsqplus",
        "hawq_mixed_le6": "hawq",
        "mixed_task_aware": "mixed_task_aware",
    }[args.method]
    controller = ModelMethodQATController(
        model,
        contract,
        target_plan,
        ModelMethodQATConfig(
            method=internal_method,
            weight_bits=assignment.weight_bits,
            activation_bits=assignment.activation_bits,
            propagation=_propagation_config(),
            hawq_range_momentum=training["hawq_range_momentum"],
        ),
        joint_adapter=joint,
        propagation_adapter=propagation,
    )
    controller.initialize_activations(initialization_rows)
    controller.install()
    student_semantic = install_model_semantic_adapter(
        model, model_config.model, strict=True)
    student_semantic.delegate_quantization()
    teacher_semantic = install_model_semantic_adapter(
        teacher, model_config.model, strict=True)
    teacher_semantic.delegate_quantization()
    return PreparedSelectedQAT(
        student_runtime,
        teacher_runtime,
        model,
        teacher,
        contract,
        assignment,
        budget_audit,
        controller,
        propagation,
        joint,
        student_semantic,
        teacher_semantic,
        trainset,
        valset,
        indices,
        graph_preparation,
    )


def _build_loaders(prepared, model_config, training, generator):
    from torch.utils.data import DataLoader, Subset
    fixed = set(model_config.evaluation_indices)
    validation_indices = tuple(
        index for index in range(len(prepared.valset)) if index not in fixed)
    if not validation_indices:
        raise RuntimeError("selected QAT validation split is empty")
    trainloader = DataLoader(
        prepared.trainset,
        batch_size=training["batch_size"],
        shuffle=True,
        num_workers=training["workers"],
        pin_memory=True,
        drop_last=True,
        generator=generator,
    )
    valloader = DataLoader(
        Subset(prepared.valset, validation_indices),
        batch_size=training["validation_batch_size"],
        shuffle=False,
        num_workers=training["workers"],
        pin_memory=True,
        drop_last=False,
    )
    if len(trainloader) == 0 or len(valloader) == 0:
        raise RuntimeError("selected QAT train or validation loader is empty")
    return trainloader, valloader, validation_indices


def _task_forward(prepared, model_input, target, loss_weights,
                  boundary_threshold_m):
    from spn_quant.qat.task_loss import model_task_aware_loss
    prepared.teacher.eval()
    prepared.teacher_semantic.begin_task_capture()
    with torch.no_grad():
        teacher_output = prepared.teacher(*model_input)
        teacher_capture = prepared.teacher_semantic.task_capture()
    if not torch.equal(
            prepared.teacher_runtime.prediction(teacher_output),
            teacher_capture.prediction):
        raise RuntimeError("teacher semantic prediction capture changed")
    prepared.student_semantic.begin_task_capture()
    output = prepared.model(*model_input)
    student_capture = prepared.student_semantic.task_capture()
    prediction = prepared.student_runtime.prediction(output)
    if not torch.equal(prediction, student_capture.prediction):
        raise RuntimeError("student semantic prediction capture changed")
    loss = model_task_aware_loss(
        student_capture,
        teacher_capture,
        target,
        target > 0.0,
        loss_weights,
        boundary_threshold_m,
    )
    if any(not bool(torch.isfinite(value).all().item())
           for value in loss.as_dict().values()):
        raise FloatingPointError("selected task-aware QAT loss is non-finite")
    return prediction, loss


def _metric_accumulator():
    from scripts import train_nyu_iteration_sweep as sweep
    return dict((key, 0.0) for key in sweep.METRIC_KEYS)


def _finish_metrics(total, samples: int):
    from scripts import train_nyu_iteration_sweep as sweep
    if int(samples) <= 0:
        raise RuntimeError("selected QAT epoch has no samples")
    return dict(
        (key, total[key] / float(samples)) for key in sweep.METRIC_KEYS)


def _train_epoch(prepared, loader, optimizer, device, epoch,
                 training, loss_weights):
    from scripts import train_nyu_cspn_group_a4_qat as qat_base
    from scripts import train_nyu_iteration_sweep as sweep
    qat_base.set_qat_train_mode(prepared.model)
    prepared.controller.activation_modules.train()
    prepared.teacher.eval()
    total = _metric_accumulator()
    samples = 0
    loss_sum = 0.0
    gradient_sum = 0.0
    started = time.time()
    for step, sample in enumerate(loader, 1):
        model_input, target = prepared.student_runtime.model_input(
            sample, device)
        optimizer.zero_grad(set_to_none=True)
        prediction, loss = _task_forward(
            prepared,
            model_input,
            target,
            loss_weights,
            training["boundary_threshold_m"],
        )
        loss.total.backward()
        gradient_norm = prepared.controller.assert_finite_gradients()
        qat_base.clip_gradients(
            prepared.controller,
            gradient_norm,
            training["max_gradient_norm"],
        )
        optimizer.step()
        qat_base.assert_finite_parameters(prepared.controller)
        batch = int(target.shape[0])
        samples += batch
        loss_sum += float(loss.total.detach().item()) * batch
        gradient_sum += gradient_norm * batch
        metrics = sweep.evaluate_error(target.detach(), prediction.detach())
        for key in sweep.METRIC_KEYS:
            total[key] += float(metrics[key]) * batch
        if step % training["log_interval"] == 0:
            print("epoch=%d step=%d/%d loss=%.6f" % (
                epoch, step, len(loader), loss_sum / float(samples)),
                flush=True)
    result = _finish_metrics(total, samples)
    result["samples"] = samples
    result["loss"] = loss_sum / float(samples)
    result["grad_norm"] = gradient_sum / float(samples)
    result["seconds"] = time.time() - started
    result["lr"] = float(optimizer.param_groups[0]["lr"])
    return result


def _evaluate_epoch(prepared, loader, device):
    from scripts import train_nyu_iteration_sweep as sweep
    from scripts.run_nyu_model_p3t3_search import _propagation_valid
    prepared.model.eval()
    prepared.controller.activation_modules.eval()
    total = _metric_accumulator()
    samples = 0
    loss_sum = 0.0
    started = time.time()
    with torch.no_grad():
        for sample in loader:
            model_input, target = prepared.student_runtime.model_input(
                sample, device)
            prediction = prepared.student_runtime.prediction(
                prepared.model(*model_input))
            if not _propagation_valid(prepared.propagation.statistics()):
                raise RuntimeError(
                    "hard deployment propagation invariants failed")
            loss = sweep.masked_l1(prediction, target)
            sweep.validate_batch_numerics(prediction, target, loss)
            batch = int(target.shape[0])
            samples += batch
            loss_sum += float(loss.item()) * batch
            metrics = sweep.evaluate_error(target, prediction)
            for key in sweep.METRIC_KEYS:
                total[key] += float(metrics[key]) * batch
    result = _finish_metrics(total, samples)
    result["samples"] = samples
    result["loss"] = loss_sum / float(samples)
    result["seconds"] = time.time() - started
    return result


def _hard_deployment_state(controller):
    return {
        "model": controller.hard_model_state_dict(),
        "method": controller.method_state_dict(),
    }


def _evaluate_hard_deployment_epoch(
        prepared, loader, device, epoch, model_config, training,
        context_factory=build_materialized_deployment_context):
    before = _hard_deployment_state(prepared.controller)
    hard_state = before["model"]
    method_state = before["method"]
    qparams = prepared.controller.deployment_qparams()
    manifest = prepared.controller.hard_deployment_manifest()
    context = context_factory(
        prepared,
        model_config,
        training,
        hard_state,
        method_state,
        qparams,
        device,
    )
    try:
        deployed_state = dict(
            (name, value.detach().cpu().clone())
            for name, value in context.model.state_dict().items())
        if tensor_state_sha256(deployed_state) != \
                tensor_state_sha256(hard_state):
            raise RuntimeError(
                "materialized hard model state differs after strict load")
        if context.controller.deployment_qparams() != qparams:
            raise RuntimeError("materialized hard qparams differ")
        evaluation = _evaluate_epoch(context, loader, device)
    finally:
        context.close()
    after = _hard_deployment_state(prepared.controller)
    validate_hard_deployment_stability(before, after)
    record = hard_deployment_evaluation_record(
        epoch,
        manifest,
        evaluation,
        hard_state,
        method_state,
        qparams,
    )
    return evaluation, record


def _freeze_and_revalidate_terminal_hawq(
        prepared, loader, device, epoch, model_config, training,
        initial_evaluation, evaluator=_evaluate_hard_deployment_epoch):
    prepared.controller.freeze_activation_ranges()
    evaluation, record = evaluator(
        prepared, loader, device, epoch, model_config, training)
    if int(evaluation["samples"]) != int(initial_evaluation["samples"]) or \
            float(evaluation["RMSE"]) != \
            float(initial_evaluation["RMSE"]):
        raise RuntimeError(
            "terminal HAWQ hard revalidation changed evaluation metrics")
    return evaluation, record


def _resume_contract(prepared, method, training, validation_indices):
    return {
        "model_name": prepared.contract.model_name,
        "method": method,
        "assignment": _assignment_payload(prepared.assignment),
        "contract_manifest": _contract_manifest(prepared.contract),
        "training_config": dict(training),
        "validation_indices": list(validation_indices),
    }


def _checkpoint_payload(epoch, prepared, method, optimizer, scheduler,
                        tracker, training, history, generator,
                        hard_validation, validation_indices, device):
    rng = capture_rng_state(generator, device)
    method_state = prepared.controller.method_state_dict()
    terminal = tracker.reason != "running"
    payload = {
        "format_version": 2,
        "model_name": prepared.contract.model_name,
        "method": method,
        "model_state": prepared.controller.canonical_model_state_dict(),
        "method_state": method_state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": int(epoch),
        "assignment": _assignment_payload(prepared.assignment),
        "contract_manifest": _contract_manifest(prepared.contract),
        "training_config": dict(training),
        "calibration_indices": list(prepared.calibration_indices),
        "validation_indices": list(validation_indices),
        "convergence": tracker.state_dict(),
        "history": list(history),
        "hard_deployment_validation": dict(hard_validation),
        "run_state": {
            "terminal": terminal,
            "completed": terminal,
            "reason": str(tracker.reason),
        },
    }
    payload.update(rng)
    validate_checkpoint_payload(payload)
    validate_training_state_subcontracts(
        payload, optimizer, scheduler, tracker)
    validate_hard_deployment_against_controller(
        hard_validation, prepared.controller)
    return payload


def _restore_checkpoint(path, prepared, method, optimizer, scheduler,
                        tracker, training, validation_indices,
                        generator, device):
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    validate_checkpoint_payload(payload)
    expected = _resume_contract(
        prepared, method, training, validation_indices)
    saved = {
        "model_name": payload["model_name"],
        "method": payload["method"],
        "assignment": payload["assignment"],
        "contract_manifest": payload["contract_manifest"],
        "training_config": payload["training_config"],
        "validation_indices": payload["validation_indices"],
    }
    if tuple(payload["calibration_indices"]) != \
            tuple(prepared.calibration_indices):
        raise ValueError("selected QAT resume calibration identities changed")
    validate_resume_contract(saved, expected)
    validate_training_state_subcontracts(
        payload, optimizer, scheduler, tracker)
    prepared.controller.load_canonical_model_state_dict(
        payload["model_state"])
    prepared.controller.load_method_state_dict(payload["method_state"])
    validate_hard_deployment_against_controller(
        payload["hard_deployment_validation"], prepared.controller)
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    tracker.load_state_dict(payload["convergence"])
    restore_rng_state(dict(
        (key, payload[key]) for key in (
            "train_generator_state",
            "torch_rng_state",
            "numpy_rng_state",
            "python_rng_state",
            "cuda_rng_state",
            "deterministic_algorithms",
        )), generator, device)
    return checkpoint_resume_epoch(payload), list(payload["history"]), payload


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_cli(argv):
    from scripts import train_nyu_cspn_group_a4_qat as qat_base
    from scripts import train_nyu_iteration_sweep as sweep
    from spn_quant.experiment_config import load_selected_quantization_config
    from spn_quant.qat.task_loss import ModelTaskLossWeights

    args = build_parser().parse_args(tuple(argv))
    validate_method_assignment_paths(
        args.method, args.hawq_assignment, args.p3_t3_assignment,
        args.hawq_trace_artifact)
    selected = load_selected_quantization_config(args.config)
    models = tuple(
        model for model in selected.models if model.model == args.model)
    if len(models) != 1:
        raise ValueError("selected QAT model entry is not unique")
    model_config = models[0]
    if str(args.device) != model_config.device:
        raise ValueError("selected QAT device differs from model config")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("selected QAT requires an explicit CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for selected QAT")
    if args.output.exists():
        raise FileExistsError("selected QAT output already exists: %s" %
                              args.output)
    if not args.output.parent.is_dir():
        raise FileNotFoundError(
            "selected QAT output parent is missing: %s" %
            args.output.parent)
    for assignment_path in (
            args.hawq_assignment, args.hawq_trace_artifact,
            args.p3_t3_assignment, args.resume):
        if assignment_path is not None and not assignment_path.is_file():
            raise FileNotFoundError(
                "selected QAT input is missing: %s" % assignment_path)
    training = _training_config(args)
    sweep.seed_all(training["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=False, exist_ok=False)
    prepared = prepare_selected_qat(
        args, selected, model_config, training)
    try:
        torch.set_num_threads(int(prepared.student_runtime.saved_args.torch_threads))
        generator = torch.Generator().manual_seed(training["seed"])
        trainloader, valloader, validation_indices = _build_loaders(
            prepared, model_config, training, generator)
        optimizer = torch.optim.SGD(
            prepared.controller.parameters(),
            lr=training["learning_rate"],
            momentum=training["momentum"],
            weight_decay=training["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=training["scheduler_factor"],
            patience=training["scheduler_patience"],
            threshold=training["scheduler_threshold"],
            min_lr=training["scheduler_min_lr"],
        )
        tracker = qat_base.QATConvergenceTracker(
            training["epochs"],
            training["patience"],
            training["min_relative_improvement"],
        )
        loss_weights = ModelTaskLossWeights(
            depth=training["depth_loss_weight"],
            boundary=training["boundary_loss_weight"],
            teacher=training["teacher_loss_weight"],
            initial_depth=training["initial_depth_loss_weight"],
            propagation=training["propagation_loss_weight"],
        )
        manifest = {
            "format_version": 1,
            "model": model_config.model,
            "method": args.method,
            "assignment": _assignment_payload(prepared.assignment),
            "contract": _contract_manifest(prepared.contract),
            "controller": prepared.controller.manifest(),
            "graph_preparation": prepared.graph_preparation,
            "training_config": training,
            "calibration_metadata":
                str(model_config.calibration_metadata.resolve()),
            "validation_indices": list(validation_indices),
        }
        if prepared.budget_audit is not None:
            manifest["mixed_precision_budget_audit"] = dict(
                prepared.budget_audit)
        _write_json(args.output / "manifest.json", manifest)
        start_epoch = 1
        history = []
        restored_payload = None
        if args.resume is not None:
            start_epoch, history, restored_payload = _restore_checkpoint(
                args.resume,
                prepared,
                args.method,
                optimizer,
                scheduler,
                tracker,
                training,
                validation_indices,
                generator,
                device,
            )
        if start_epoch is None:
            torch.save(restored_payload, args.output / "final.pt")
            _write_json(args.output / "convergence.json", tracker.state_dict())
            return args.output / "final.pt"
        if start_epoch > training["epochs"]:
            raise ValueError("selected QAT resume is past configured epochs")
        final_payload = None
        final_validation = None
        for epoch in range(start_epoch, training["epochs"] + 1):
            train_values = _train_epoch(
                prepared,
                trainloader,
                optimizer,
                device,
                epoch,
                training,
                loss_weights,
            )
            validation, hard_validation = \
                _evaluate_hard_deployment_epoch(
                    prepared, valloader, device, epoch,
                    model_config, training)
            previous_best = tracker.best_rmse
            stop = tracker.update(epoch, validation["RMSE"])
            is_best = float(validation["RMSE"]) < previous_best
            if stop and args.method == "hawq_mixed_le6":
                validation, hard_validation = \
                    _freeze_and_revalidate_terminal_hawq(
                        prepared,
                        valloader,
                        device,
                        epoch,
                        model_config,
                        training,
                        validation,
                    )
            validation["hard_deployment_validated"] = 1
            scheduler.step(validation["RMSE"])
            history.extend((
                {"epoch": epoch, "split": "train", **train_values},
                {"epoch": epoch, "split": "validation", **validation},
            ))
            payload = _checkpoint_payload(
                epoch,
                prepared,
                args.method,
                optimizer,
                scheduler,
                tracker,
                training,
                history,
                generator,
                hard_validation,
                validation_indices,
                device,
            )
            torch.save(payload, args.output / "last.pt")
            if is_best:
                torch.save(payload, args.output / "best.pt")
            final_payload = payload
            final_validation = validation
            print(
                "model=%s method=%s epoch=%d train_RMSE=%.6f val_RMSE=%.6f" % (
                    args.model,
                    args.method,
                    epoch,
                    train_values["RMSE"],
                    validation["RMSE"],
                ),
                flush=True,
            )
            if stop:
                break
        if final_payload is None or final_validation is None:
            raise RuntimeError("selected QAT completed no evaluation epoch")
        validate_checkpoint_payload(final_payload)
        torch.save(final_payload, args.output / "final.pt")
        _write_json(args.output / "convergence.json", tracker.state_dict())
        return args.output / "final.pt"
    finally:
        prepared.close()


def main(argv=None):
    path = run_cli(sys.argv[1:] if argv is None else argv)
    print("selected QAT checkpoint: %s" % path)


if __name__ == "__main__":
    main()
