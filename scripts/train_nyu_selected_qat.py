#!/usr/bin/env python3
"""Train selected official SPN models with strict task-aware QAT."""

from __future__ import annotations

import argparse
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


def hard_deployment_evaluation_record(epoch: int, manifest, evaluation):
    if int(manifest["validated"]) != 1:
        raise ValueError("hard deployment validation failed")
    samples = int(evaluation["samples"])
    rmse = float(evaluation["RMSE"])
    if samples <= 0 or not math.isfinite(rmse):
        raise ValueError("hard deployment evaluation metrics are invalid")
    record = dict(manifest)
    record["epoch"] = int(epoch)
    record["evaluation_samples"] = samples
    record["evaluation_rmse"] = rmse
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
        method: str, hawq_assignment, p3_t3_assignment) -> None:
    if method not in SELECTED_QAT_METHODS:
        raise ValueError("unsupported selected QAT method: %s" % method)
    if method == "hawq_mixed_le6":
        if hawq_assignment is None:
            raise ValueError("HAWQ QAT requires its mixed assignment")
        if p3_t3_assignment is not None:
            raise ValueError("HAWQ QAT cannot use a P3/T3 assignment")
    elif method == "mixed_task_aware":
        if p3_t3_assignment is None:
            raise ValueError("mixed task-aware QAT requires P3/T3 assignment")
        if hawq_assignment is not None:
            raise ValueError("mixed task-aware QAT cannot use HAWQ assignment")
    elif hawq_assignment is not None or p3_t3_assignment is not None:
        raise ValueError("uniform LSQ++ QAT cannot use assignment paths")


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
    return BitAssignment(
        weight_bits=tuple(
            (str(row["module"]), int(row["bits"]))
            for row in payload["weight_bits"]),
        activation_bits=tuple(
            ((str(row["site"]), str(row["role"])), int(row["bits"]))
            for row in payload["activation_bits"]),
        model_name=str(payload["model_name"]),
    )


def load_hawq_qat_assignment(
        path: Path,
        contract: QuantizationModelContract) -> BitAssignment:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if str(payload["model_name"]) != contract.model_name:
        raise ValueError("HAWQ assignment model differs from contract")
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
    required = {
        "model_name",
        "calibration",
        "contract",
        "average_weight_bits",
        "average_weight_mac_bits",
        "average_activation_bits",
        "assignment",
        "objective",
        "constraints",
        "cost_basis",
        "solver_status",
    }
    if set(payload) != required:
        raise ValueError("HAWQ QAT artifact fields changed")
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
    block_weights = dict(
        (str(row["block"]), int(row["bits"]))
        for row in payload["assignment"]["weight_block_bits"])
    block_activations = dict(
        (str(row["block"]), int(row["bits"]))
        for row in payload["assignment"]["activation_block_bits"])
    if set(block_weights) != set(contract.block_names) or \
            set(block_activations) != set(contract.block_names):
        raise ValueError("HAWQ block assignment coverage changed")
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
    weight_parameters = dict(
        (str(row["block"]), int(row["parameters"]))
        for row in basis["weight_parameters"])
    weight_macs = dict(
        (str(row["module"]), int(row["macs"]))
        for row in basis["weight_macs"])
    activation_traffic = dict(
        ((str(row["site"]), str(row["role"])), int(row["elements"]))
        for row in basis["activation_traffic"])
    if set(weight_parameters) != set(contract.block_names) or \
            set(weight_macs) != set(contract.weight_modules) or \
            set(activation_traffic) != set(_contract_owners(contract)):
        raise ValueError("HAWQ QAT cost basis coverage changed")
    if any(value <= 0 for value in weight_parameters.values()) or \
            any(value <= 0 for value in weight_macs.values()) or \
            any(value <= 0 for value in activation_traffic.values()):
        raise ValueError("HAWQ QAT cost basis must be positive")

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
    calibration = payload["calibration"]
    if set(calibration) != {"count", "indices", "identity_sha256"}:
        raise ValueError("HAWQ calibration provenance fields changed")
    indices = tuple(int(index) for index in calibration["indices"])
    if int(calibration["count"]) != 128 or len(indices) != 128 or \
            len(indices) != len(set(indices)) or any(
                index < 0 for index in indices):
        raise ValueError("HAWQ calibration provenance is invalid")
    from scripts.run_nyu_qdrop_reconstruction import (
        ordered_sample_identity_sha256,
    )
    if str(calibration["identity_sha256"]) != \
            ordered_sample_identity_sha256("train", indices):
        raise ValueError("HAWQ calibration identity differs from indices")
    return assignment


def validate_checkpoint_payload(payload) -> None:
    if set(payload) != CHECKPOINT_FIELDS:
        raise ValueError("selected QAT checkpoint field contract mismatch")
    if int(payload["format_version"]) != 1:
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
    epoch = int(payload["epoch"])
    validation = payload["hard_deployment_validation"]
    if set(validation) < {"epoch", "validated"} or \
            int(validation["epoch"]) != epoch or \
            int(validation["validated"]) != 1:
        raise ValueError(
            "hard deployment validation must match every evaluation epoch")
    if not isinstance(payload["deterministic_algorithms"], bool):
        raise TypeError("deterministic algorithm metadata must be boolean")


def validate_resume_contract(saved, expected) -> None:
    if saved != expected:
        raise ValueError("selected QAT resume contract changed")


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


def _cost_basis_from_p3_t3(path: Path) -> CostBasis:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    basis = payload["cost_basis"]
    if set(basis) != {"activation_elements", "weight_macs"}:
        raise ValueError("P3/T3 cost basis fields changed")
    return CostBasis(
        weight_macs=tuple(
            (str(row[0]), int(row[1])) for row in basis["weight_macs"]),
        activation_elements=tuple(
            ((str(row[0][0]), str(row[0][1])), int(row[1]))
            for row in basis["activation_elements"]),
    )


def _selected_assignment(args, selected, contract):
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
            args.hawq_assignment, contract)
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
    from scripts.run_nyu_selected_ptq import load_p3_t3_assignment
    p3_t3 = load_p3_t3_assignment(
        args.p3_t3_assignment,
        contract,
        selected.method_hyperparameters["p3_t3_mixed_ptq"],
    )
    costs = _cost_basis_from_p3_t3(args.p3_t3_assignment)
    return mixed_task_aware_assignment(
        contract,
        p3_t3,
        p3_t3.activation_bits,
        costs,
        float(mixed_config["maximum_average_activation_bits"]),
    )


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
        args, selected, contract)
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


def _evaluate_hard_deployment_epoch(prepared, loader, device, epoch):
    prepared.model.eval()
    prepared.controller.activation_modules.eval()
    before = _hard_deployment_state(prepared.controller)
    manifest = prepared.controller.hard_deployment_manifest()
    evaluation = _evaluate_epoch(prepared, loader, device)
    after = _hard_deployment_state(prepared.controller)
    validate_hard_deployment_stability(before, after)
    record = hard_deployment_evaluation_record(
        epoch, manifest, evaluation)
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
    payload = {
        "format_version": 1,
        "model_name": prepared.contract.model_name,
        "method": method,
        "model_state": prepared.controller.canonical_model_state_dict(),
        "method_state": prepared.controller.method_state_dict(),
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
    }
    payload.update(rng)
    validate_checkpoint_payload(payload)
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
    prepared.controller.load_canonical_model_state_dict(
        payload["model_state"])
    prepared.controller.load_method_state_dict(payload["method_state"])
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
    return int(payload["epoch"]) + 1, list(payload["history"])


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
        args.method, args.hawq_assignment, args.p3_t3_assignment)
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
            args.hawq_assignment, args.p3_t3_assignment, args.resume):
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
            manifest["mixed_activation_budget"] = {
                "average_activation_bits":
                    prepared.budget_audit.average_activation_bits,
                "maximum_activation_bits":
                    prepared.budget_audit.maximum_activation_bits,
                "activation_numerator":
                    prepared.budget_audit.activation_numerator,
                "activation_denominator":
                    prepared.budget_audit.activation_denominator,
            }
        _write_json(args.output / "manifest.json", manifest)
        start_epoch = 1
        history = []
        if args.resume is not None:
            start_epoch, history = _restore_checkpoint(
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
                    prepared, valloader, device, epoch)
            validation["hard_deployment_validated"] = 1
            scheduler.step(validation["RMSE"])
            history.extend((
                {"epoch": epoch, "split": "train", **train_values},
                {"epoch": epoch, "split": "validation", **validation},
            ))
            previous_best = tracker.best_rmse
            stop = tracker.update(epoch, validation["RMSE"])
            is_best = float(validation["RMSE"]) < previous_best
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
        if args.method == "hawq_mixed_le6":
            prepared.controller.freeze_activation_ranges()
            final_payload["method_state"] = \
                prepared.controller.method_state_dict()
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
