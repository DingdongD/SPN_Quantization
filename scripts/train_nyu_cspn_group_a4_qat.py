#!/usr/bin/env python3
"""Train strict Static-G8 or Dynamic-G8 W4A4 QAT on official CSPN."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
from argparse import Namespace
from typing import Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_nyu_cspn_activation_resolution as base  # noqa: E402
from scripts import run_nyu_cspn_mixed_activation_search as mixed_search  # noqa: E402
from scripts import run_nyu_cspn_stem_precision as stem_runner  # noqa: E402
from scripts import run_nyu_cspn_task_sensitive_bits as task_runner  # noqa: E402
from scripts import train_nyu_iteration_sweep as sweep  # noqa: E402
from scripts.hardware_aligned_quantization import (  # noqa: E402
    HardwareAlignedInstrumentor,
    prepare_hardware_model,
)
from scripts.run_nyu_rtn_quantization import (  # noqa: E402
    calibration_dataset,
    seeded_sample,
)
from spn_quant.adapters import install_model_semantic_adapter  # noqa: E402
from spn_quant.propagation import (  # noqa: E402
    PropagationQuantConfig,
    install_propagation_adapter,
)
from spn_quant.qat import (  # noqa: E402
    CSPNQATConfig,
    CSPNQATController,
    CSPNTaskLossWeights,
    cspn_task_aware_loss,
    cspn_hard_activation_bits,
)
from spn_quant.activation_boundaries import (  # noqa: E402
    CSPNActivationBoundaryController,
)
from spn_quant import cspn_task_sensitive_bits as allocation  # noqa: E402


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    patience: int
    min_relative_improvement: float
    batch_size: int
    val_batch_size: int
    workers: int
    learning_rate: float
    momentum: float
    weight_decay: float
    max_gradient_norm: float
    seed: int

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.epochs > 30:
            raise ValueError("QAT epochs must be between 1 and 30")
        if self.patience <= 0:
            raise ValueError("QAT patience must be positive")
        if not 0.0 < self.min_relative_improvement < 1.0:
            raise ValueError("relative improvement must be between zero and one")
        if self.batch_size <= 0 or self.val_batch_size <= 0:
            raise ValueError("QAT batch sizes must be positive")
        if self.workers < 0:
            raise ValueError("QAT workers must be nonnegative")
        if self.learning_rate <= 0.0:
            raise ValueError("QAT learning rate must be positive")
        if self.momentum < 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer values must be nonnegative")
        if self.max_gradient_norm <= 0.0:
            raise ValueError("gradient norm limit must be positive")


@dataclass(frozen=True)
class CalibrationMetadata:
    calibration_indices: Tuple[int, ...]
    evaluation_indices: Tuple[int, ...]


@dataclass(frozen=True)
class MixedPrecisionInputs:
    precision_config: Mapping[str, object]
    assignment: allocation.BitAssignment
    cost_basis: allocation.CostBasis
    budget: allocation.ActivationBudgetAudit
    loss_weights: CSPNTaskLossWeights
    boundary_threshold_m: float
    precision_config_sha256: str
    assignment_sha256: str
    cost_basis_sha256: str


class QATConvergenceTracker:
    def __init__(self, max_epochs: int, patience: int,
                 min_relative_improvement: float) -> None:
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.min_relative_improvement = float(min_relative_improvement)
        self.best_rmse = float("inf")
        self.best_epoch = 0
        self.significant_best_rmse = float("inf")
        self.no_improvement_epochs = 0
        self.reason = "running"

    def update(self, epoch: int, rmse: float) -> bool:
        epoch = int(epoch)
        rmse = float(rmse)
        if not math.isfinite(rmse):
            raise FloatingPointError("validation RMSE is non-finite")
        if rmse < self.best_rmse:
            self.best_rmse = rmse
            self.best_epoch = epoch
        threshold = 1.0 - self.min_relative_improvement
        if not math.isfinite(self.significant_best_rmse) or \
                rmse < self.significant_best_rmse * threshold:
            self.significant_best_rmse = rmse
            self.no_improvement_epochs = 0
        else:
            self.no_improvement_epochs += 1
        if self.no_improvement_epochs >= self.patience:
            self.reason = "validation_plateau"
            return True
        if epoch >= self.max_epochs:
            self.reason = "max_epochs"
            return True
        return False

    def state_dict(self):
        return {
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "min_relative_improvement": self.min_relative_improvement,
            "best_rmse": self.best_rmse,
            "best_epoch": self.best_epoch,
            "significant_best_rmse": self.significant_best_rmse,
            "no_improvement_epochs": self.no_improvement_epochs,
            "reason": self.reason,
        }

    def load_state_dict(self, state) -> None:
        self.max_epochs = int(state["max_epochs"])
        self.patience = int(state["patience"])
        self.min_relative_improvement = float(
            state["min_relative_improvement"])
        self.best_rmse = float(state["best_rmse"])
        self.best_epoch = int(state["best_epoch"])
        self.significant_best_rmse = float(
            state["significant_best_rmse"])
        self.no_improvement_epochs = int(state["no_improvement_epochs"])
        self.reason = str(state["reason"])


def load_calibration_metadata(path: Path) -> CalibrationMetadata:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    calibration_indices = tuple(
        int(index) for index in payload["calibration_indices"])
    evaluation_indices = tuple(
        int(index) for index in payload["evaluation_indices"])
    if len(calibration_indices) != 128:
        raise ValueError("CSPN QAT calibration requires 128 indices")
    if len(evaluation_indices) != 64:
        raise ValueError("CSPN QAT evaluation requires 64 indices")
    if len(calibration_indices) != len(set(calibration_indices)):
        raise ValueError("CSPN QAT calibration indices must be unique")
    if len(evaluation_indices) != len(set(evaluation_indices)):
        raise ValueError("CSPN QAT evaluation indices must be unique")
    if payload["calibration_source"]["selection"] != \
            "32_tail_96_kmedoids":
        raise ValueError("CSPN QAT requires stratified calibration indices")
    return CalibrationMetadata(
        calibration_indices=calibration_indices,
        evaluation_indices=evaluation_indices,
    )


def load_assignment(path: Path) -> allocation.BitAssignment:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assignment = allocation.BitAssignment(
        weight_bits=tuple(
            (str(row["module"]), int(row["bits"]))
            for row in payload["weight_bits"]),
        activation_bits=tuple(
            ((str(row["module"]), str(row["kind"])), int(row["bits"]))
            for row in payload["activation_bits"]),
    )
    task_runner.validate_assignment_contract(assignment)
    return assignment


def load_cost_basis(path: Path) -> allocation.CostBasis:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return allocation.CostBasis(
        weight_macs=tuple(
            (str(row["module"]), int(row["macs"]))
            for row in payload["weight_macs"]),
        activation_elements=tuple(
            ((str(row["module"]), str(row["kind"])), int(row["elements"]))
            for row in payload["activation_elements"]),
    )


def load_mixed_precision_inputs(args) -> MixedPrecisionInputs:
    precision_config_path = Path(args.precision_config)
    assignment_path = Path(args.assignment)
    cost_basis_path = Path(args.cost_basis)
    precision_config = mixed_search.load_precision_config(
        precision_config_path)
    assignment = load_assignment(assignment_path)
    cost_basis = load_cost_basis(cost_basis_path)
    search = precision_config["search"]
    maximum_bits = float(search["activation_budget_bits"])
    budget = allocation.audit_activation_budget(
        assignment, cost_basis, maximum_bits)
    if not budget.feasible:
        raise ValueError("mixed activation assignment exceeds its budget")
    registry = task_runner.expected_registry()
    if assignment.weight_bits != allocation.p3_t3_assignment(
            registry).weight_bits:
        raise ValueError("mixed QAT weight assignment differs from P3/T3")
    acceptance = precision_config["acceptance"]
    if float(acceptance["average_activation_bits"]) != maximum_bits:
        raise ValueError("search and acceptance activation budgets differ")
    loss = precision_config["loss"]
    return MixedPrecisionInputs(
        precision_config=precision_config,
        assignment=assignment,
        cost_basis=cost_basis,
        budget=budget,
        loss_weights=CSPNTaskLossWeights(
            depth=float(loss["depth"]),
            boundary=float(loss["boundary"]),
            teacher=float(loss["teacher"]),
            propagation=float(loss["propagation"]),
        ),
        boundary_threshold_m=float(search["boundary_threshold_m"]),
        precision_config_sha256=stem_runner._sha256(precision_config_path),
        assignment_sha256=stem_runner._sha256(assignment_path),
        cost_basis_sha256=stem_runner._sha256(cost_basis_path),
    )


def checkpoint_contract(mode: str, calibration_indices: Tuple[int, ...],
                        owner_manifest: Tuple[str, ...]):
    if mode not in ("static", "dynamic"):
        raise ValueError("quantization mode must be static or dynamic")
    return {
        "mode": mode,
        "calibration_indices": list(calibration_indices),
        "owner_manifest": list(owner_manifest),
    }


def validate_resume_contract(contract, mode: str,
                             calibration_indices: Tuple[int, ...],
                             owner_manifest: Tuple[str, ...]) -> None:
    if contract["mode"] != mode:
        raise ValueError("resume quantization mode changed")
    if tuple(contract["calibration_indices"]) != tuple(calibration_indices):
        raise ValueError("resume calibration indices changed")
    if tuple(contract["owner_manifest"]) != tuple(owner_manifest):
        raise ValueError("resume owner manifest changed")


def mixed_checkpoint_contract(
        metadata: CalibrationMetadata,
        owners: Tuple[str, ...],
        mixed: MixedPrecisionInputs,
        calibration_metadata_sha256: str,
        early_stopping_indices: Tuple[int, ...],
        propagation: PropagationQuantConfig):
    return {
        "mode": "mixed_static",
        "precision_config_sha256": mixed.precision_config_sha256,
        "assignment_sha256": mixed.assignment_sha256,
        "cost_basis_sha256": mixed.cost_basis_sha256,
        "calibration_metadata_sha256": str(calibration_metadata_sha256),
        "calibration_indices": list(metadata.calibration_indices),
        "early_stopping_identities": [
            "validation:%d" % index for index in early_stopping_indices],
        "fixed64_identities": [
            "validation:%d" % index for index in metadata.evaluation_indices],
        "owner_manifest": list(owners),
        "assignment": task_runner.assignment_payload(mixed.assignment),
        "loss_weights": asdict(mixed.loss_weights),
        "boundary_threshold_m": mixed.boundary_threshold_m,
        "average_activation_bits": mixed.budget.average_activation_bits,
        "propagation": {
            "affinity_bits": propagation.affinity_bits,
            "confidence_bits": propagation.confidence_bits,
            "offset_bits": propagation.offset_bits,
            "state_bits": propagation.state_bits,
            "coefficient_fraction_bits":
                propagation.coefficient_fraction_bits,
        },
    }


def validate_mixed_resume_contract(saved, expected) -> None:
    if saved != expected:
        raise ValueError("mixed QAT resume contract changed")


def set_qat_train_mode(model: nn.Module) -> None:
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def assert_finite_parameters(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all().item()):
            raise FloatingPointError(
                "QAT parameter contains non-finite values: %s" % name)


def clip_gradients(model: nn.Module, gradient_norm: float,
                   maximum: float) -> None:
    gradient_norm = float(gradient_norm)
    maximum = float(maximum)
    if maximum <= 0.0:
        raise ValueError("gradient norm limit must be positive")
    if not math.isfinite(gradient_norm):
        raise FloatingPointError("global gradient norm is non-finite")
    if gradient_norm <= maximum:
        return
    factor = maximum / gradient_norm
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(factor)


def owner_manifest_tuple(manifest) -> Tuple[str, ...]:
    return tuple(
        ["weight:%s" % name for name in manifest["weight_modules"]]
        + ["activation:%s" % owner
           for owner in manifest["ordinary_owners"]]
        + ["structural:%s" % owner
           for owner in manifest["structural_owners"]]
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("static", "dynamic", "mixed_static"),
        required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--calibration-metadata", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument(
        "--min-relative-improvement", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--val-batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--momentum", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--max-gradient-norm", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-train-samples", type=int, required=True)
    parser.add_argument("--max-val-samples", type=int, required=True)
    parser.add_argument("--fold-max-error", type=float, required=True)
    parser.add_argument("--log-interval", type=int, required=True)
    parser.add_argument("--precision-config")
    parser.add_argument("--assignment")
    parser.add_argument("--cost-basis")
    parser.add_argument("--resume")
    return parser.parse_args(argv)


def validate_mode_paths(args) -> None:
    paths = (args.precision_config, args.assignment, args.cost_basis)
    if args.mode == "mixed_static":
        if any(path is None for path in paths):
            raise ValueError(
                "mixed_static requires precision config, assignment, and cost basis")
        return
    if any(path is not None for path in paths):
        raise ValueError("mixed precision paths are only valid in mixed_static")


def validate_mixed_cli_config(args, precision_config) -> None:
    training = precision_config["training"]
    observed = {
        "epochs": args.epochs,
        "patience": args.patience,
        "min_relative_improvement": args.min_relative_improvement,
        "batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "workers": args.workers,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "max_gradient_norm": args.max_gradient_norm,
        "seed": args.seed,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "fold_max_error": args.fold_max_error,
        "log_interval": args.log_interval,
    }
    expected = {
        "epochs": int(training["epochs"]),
        "patience": int(training["patience"]),
        "min_relative_improvement": float(
            training["min_relative_improvement"]),
        "batch_size": int(training["batch_size"]),
        "val_batch_size": int(training["val_batch_size"]),
        "workers": int(training["workers"]),
        "learning_rate": float(training["learning_rate"]),
        "momentum": float(training["momentum"]),
        "weight_decay": float(training["weight_decay"]),
        "max_gradient_norm": float(training["max_gradient_norm"]),
        "seed": int(training["seed"]),
        "max_train_samples": int(training["max_train_samples"]),
        "max_val_samples": int(training["max_val_samples"]),
        "fold_max_error": float(training["fold_max_error"]),
        "log_interval": int(training["log_interval"]),
    }
    if observed != expected:
        raise ValueError("mixed QAT CLI differs from precision configuration")


def saved_checkpoint_args(checkpoint_path: Path, args) -> Namespace:
    payload = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False)
    saved_args = Namespace(**payload["args"])
    if saved_args.model != "cspn":
        raise ValueError("QAT source checkpoint must contain official CSPN")
    saved_args.from_scratch = True
    saved_args.data_root = args.data_root
    saved_args.device = args.device
    saved_args.seed = args.seed
    saved_args.batch_size = args.batch_size
    saved_args.val_batch_size = args.val_batch_size
    saved_args.workers = args.workers
    saved_args.max_train_samples = args.max_train_samples
    saved_args.max_val_samples = args.max_val_samples
    return saved_args


def _limited_dataset(dataset, count: int):
    if count < 0:
        raise ValueError("sample limits must be nonnegative")
    if count == 0:
        return dataset
    if count > len(dataset):
        raise ValueError("sample limit exceeds dataset size")
    return Subset(dataset, tuple(range(count)))


def validation_indices(
        sample_count: int,
        excluded_indices: Tuple[int, ...]) -> Tuple[int, ...]:
    count = int(sample_count)
    excluded = tuple(int(index) for index in excluded_indices)
    if count <= 0:
        raise ValueError("validation sample count must be positive")
    if len(excluded) != len(set(excluded)):
        raise ValueError("excluded validation indices must be unique")
    if any(index < 0 or index >= count for index in excluded):
        raise ValueError("excluded validation index exceeds dataset")
    excluded_set = set(excluded)
    return tuple(index for index in range(count) if index not in excluded_set)


def build_loaders(saved_args, config: TrainingConfig,
                  train_generator: torch.Generator,
                  excluded_validation_indices: Tuple[int, ...]):
    trainset = sweep.CspnOfficialDataset(
        csv_file=saved_args.train_list,
        root_dir=str(saved_args.data_root),
        split="train",
        n_sample=saved_args.n_sample,
        seed=config.seed,
    )
    valset = sweep.CspnOfficialDataset(
        csv_file=saved_args.eval_list,
        root_dir=str(saved_args.data_root),
        split="val",
        n_sample=saved_args.n_sample,
        seed=config.seed,
    )
    trainset = _limited_dataset(trainset, saved_args.max_train_samples)
    selected_validation_indices = validation_indices(
        len(valset), excluded_validation_indices) \
        if excluded_validation_indices else tuple(range(len(valset)))
    if saved_args.max_val_samples < 0:
        raise ValueError("sample limits must be nonnegative")
    if saved_args.max_val_samples > len(selected_validation_indices):
        raise ValueError("sample limit exceeds dataset size")
    if saved_args.max_val_samples > 0:
        selected_validation_indices = selected_validation_indices[
            :saved_args.max_val_samples]
    valset = Subset(valset, selected_validation_indices)
    trainloader = DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.workers, pin_memory=True, drop_last=True,
        generator=train_generator)
    valloader = DataLoader(
        valset, batch_size=config.val_batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True, drop_last=False)
    if len(trainloader) == 0 or len(valloader) == 0:
        raise RuntimeError("empty CSPN QAT train or validation loader")
    return trainloader, valloader, selected_validation_indices


def _propagation_config() -> PropagationQuantConfig:
    return PropagationQuantConfig(
        affinity_bits=8,
        confidence_bits=8,
        offset_bits=8,
        state_bits=8,
        coefficient_fraction_bits=13,
    )


def prepare_qat_model(saved_args, checkpoint_path: Path,
                      metadata: CalibrationMetadata, mode: str,
                      fold_max_error: float, device: torch.device,
                      seed: int):
    model, architecture, load_report = base._load_cspn(
        saved_args, checkpoint_path, device)
    trainset = calibration_dataset(saved_args)
    if max(metadata.calibration_indices) >= len(trainset):
        raise ValueError("calibration index exceeds NYU training split")
    sample = seeded_sample(
        trainset, metadata.calibration_indices[0], seed)
    model_args = base._model_args(saved_args, sample, device)
    preparation = prepare_hardware_model(
        model, model_args, excluded_pairs=(("conv1_1", "bn1"),))
    if float(preparation["primary_max_abs_error"]) > fold_max_error:
        raise RuntimeError("Conv-BN fold exceeds declared error threshold")

    semantic = install_model_semantic_adapter(
        model, "cspn", strict=True)
    boundaries = semantic.activation_boundaries()
    semantic.close()
    instrumentor = HardwareAlignedInstrumentor(
        model, base.cspn_quant_group,
        preparation["fused_relu_producers"],
        externally_owned_outputs=base.strict_owned_outputs(),
        externally_owned_inputs=base.strict_owned_inputs())
    boundary_controller = CSPNActivationBoundaryController(model, boundaries)
    hard_propagation = install_propagation_adapter("cspn", model)

    base._calibrate(
        model, saved_args, trainset, metadata.calibration_indices,
        device, seed, instrumentor, boundary_controller, hard_propagation)
    base.validate_strict_site_contract(instrumentor, boundary_controller)

    activation_specs = base.build_activation_specs(
        instrumentor, base.ORDINARY_GROUPS, 4, 8,
        dynamic=mode == "dynamic")
    instrumentor.configure_components(
        4, 4, set(), base.ORDINARY_GROUPS,
        activation_specs, quantize_bias=False)

    boundary_specs = base.build_boundary_activation_specs(
        boundary_controller, 4, 8)
    bit_widths = {}
    group_sizes = {}
    scale_factors = {}
    for name in boundary_controller.channels:
        spec = boundary_specs[("boundary_controller.%s" % name, "boundary")]
        bit_widths[name] = int(spec.bits)
        group_sizes[name] = int(spec.group_size)
        scale_factors[name] = 1.0
    boundary_controller.configure_specs(
        bit_widths, group_sizes, scale_factors, quantize=True)

    propagation_config = _propagation_config()
    hard_propagation.configure(propagation_config)
    weight_modules = tuple(sorted(
        name for name in instrumentor.modules
        if instrumentor.groups[name] in base.ORDINARY_GROUPS))
    qat_config = CSPNQATConfig(
        mode=mode,
        weight_bits=tuple((name, 4) for name in weight_modules),
        activation_bits=cspn_hard_activation_bits(
            instrumentor, boundary_controller),
        group_size=8, propagation=propagation_config)
    controller = CSPNQATController(
        model, instrumentor, boundary_controller, hard_propagation,
        qat_config)
    controller.install()
    controller.set_runtime_statistics(False)
    manifest = controller.manifest()
    if len(manifest["ordinary_owners"]) != len(
            base.STRICT_ACTIVATION_OWNERS):
        raise RuntimeError("official CSPN activation owner count changed")
    if tuple(manifest["structural_owners"]) != (
            "decoder_entry", "layer4_signed_skip"):
        raise RuntimeError("official CSPN structural owner contract changed")
    if any(name.startswith("gud_up_proj_layer6")
           for name in manifest["weight_modules"]):
        raise RuntimeError("guidance weight entered the QAT owner set")
    return (
        model, controller, instrumentor, boundary_controller, hard_propagation,
        architecture, load_report, preparation, manifest,
    )


def prepare_mixed_qat_model(
        saved_args,
        checkpoint_path: Path,
        metadata: CalibrationMetadata,
        mixed: MixedPrecisionInputs,
        fold_max_error: float,
        device: torch.device,
        seed: int):
    trainset = calibration_dataset(saved_args)
    if max(metadata.calibration_indices) >= len(trainset):
        raise ValueError("calibration index exceeds NYU training split")
    sample = seeded_sample(trainset, metadata.calibration_indices[0], seed)
    model_args = base._model_args(saved_args, sample, device)
    model, architecture, load_report, preparation = \
        stem_runner._prepare_model(
            saved_args, checkpoint_path, device, model_args, fold_max_error)
    instrumentor, boundary_controller, hard_propagation, stem = \
        stem_runner._build_quantization_context(model, preparation, seed)
    stem_runner._calibrate(
        model,
        saved_args,
        trainset,
        metadata.calibration_indices,
        device,
        seed,
        instrumentor,
        boundary_controller,
        hard_propagation,
        stem,
        "MIXED_TASK_AWARE_QAT",
    )
    stem_runner._validate_site_contract(instrumentor, boundary_controller)
    candidate = task_runner.RuntimeCandidate(
        "MIXED_TASK_AWARE_QAT", "joint", mixed.assignment)
    _, specs, boundary_specs = \
        task_runner.configure_runtime_context(
            candidate,
            instrumentor,
            boundary_controller,
            hard_propagation,
            stem,
        )
    activation_bits = cspn_hard_activation_bits(
        instrumentor, boundary_controller)
    expected_generic_activations = tuple(
        (owner, bits) for owner, bits in mixed.assignment.activation_bits
        if owner != task_runner.STEM_INPUT_OWNER)
    if activation_bits != tuple(sorted(expected_generic_activations)):
        raise RuntimeError("mixed QAT hard activation assignment changed")
    qat_config = CSPNQATConfig(
        mode="mixed_static",
        weight_bits=mixed.assignment.weight_bits,
        activation_bits=activation_bits,
        group_size=8,
        propagation=_propagation_config(),
    )
    controller = CSPNQATController(
        model,
        instrumentor,
        boundary_controller,
        hard_propagation,
        qat_config,
    )
    controller.install()
    controller.set_runtime_statistics(False)
    manifest = controller.manifest()
    manifest["ordinary_owners"] = sorted(
        manifest["ordinary_owners"] + [str(task_runner.STEM_INPUT_OWNER)])
    manifest["assignment"] = task_runner.assignment_payload(mixed.assignment)
    manifest["external_activation_bits"] = [{
        "module": task_runner.STEM_INPUT_OWNER[0],
        "kind": task_runner.STEM_INPUT_OWNER[1],
        "bits": dict(mixed.assignment.activation_bits)[
            task_runner.STEM_INPUT_OWNER],
    }]
    manifest["activation_budget"] = {
        "numerator": mixed.budget.activation_numerator,
        "denominator": mixed.budget.activation_denominator,
        "average_bits": mixed.budget.average_activation_bits,
        "maximum_bits": mixed.budget.maximum_activation_bits,
    }
    manifest["site_counts"] = {
        "ordinary": len(specs),
        "boundary": len(boundary_specs),
        "stem": 1,
    }
    if any(name.startswith("gud_up_proj_layer6")
           for name in manifest["weight_modules"]):
        raise RuntimeError("guidance weight entered the QAT owner set")
    return (
        model,
        controller,
        instrumentor,
        boundary_controller,
        hard_propagation,
        stem,
        architecture,
        load_report,
        preparation,
        manifest,
    )


def prepare_teacher(saved_args, checkpoint_path: Path, device: torch.device):
    teacher, architecture, load_report = base._load_cspn(
        saved_args, checkpoint_path, device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    propagation = install_propagation_adapter("cspn", teacher)
    propagation.capture_training_states()
    propagation.set_runtime_statistics(False)
    return teacher, propagation, architecture, load_report


def _metric_accumulator():
    return dict((key, 0.0) for key in sweep.METRIC_KEYS)


def _update_metrics(total, metric, batch_size: int) -> None:
    for key in sweep.METRIC_KEYS:
        total[key] += float(metric[key]) * batch_size


def _finish_metrics(total, samples: int):
    if samples <= 0:
        raise RuntimeError("CSPN QAT epoch has no samples")
    return dict((key, total[key] / samples) for key in sweep.METRIC_KEYS)


def task_aware_forward(
        model: nn.Module,
        controller: CSPNQATController,
        teacher: nn.Module,
        teacher_propagation,
        model_input,
        target: torch.Tensor,
        loss_weights: CSPNTaskLossWeights,
        boundary_threshold_m: float):
    teacher.eval()
    with torch.no_grad():
        teacher_prediction = sweep.extract_pred(teacher(*model_input))
        teacher_states = tuple(teacher_propagation.last_states())
    prediction = sweep.extract_pred(model(*model_input))
    student_states = controller.propagation.proxy_states()
    if len(student_states) != 24 or len(teacher_states) != 24:
        raise RuntimeError("official CSPN task loss requires 24 propagation states")
    sweep.validate_batch_numerics(teacher_prediction, target)
    sweep.validate_batch_numerics(prediction, target)
    losses = cspn_task_aware_loss(
        prediction,
        target,
        target > 0.0,
        teacher_prediction,
        student_states,
        teacher_states,
        loss_weights,
        boundary_threshold_m,
    )
    for name in ("total", "depth", "boundary", "teacher", "propagation"):
        if not bool(torch.isfinite(losses[name]).all().item()):
            raise FloatingPointError("mixed QAT %s loss is non-finite" % name)
    return prediction, losses


def train_epoch(model: nn.Module, controller: CSPNQATController,
                loader, optimizer, device: torch.device,
                epoch: int, log_interval: int,
                max_gradient_norm: float):
    set_qat_train_mode(model)
    metric_sum = _metric_accumulator()
    loss_sum = 0.0
    sample_count = 0
    gradient_sum = 0.0
    started = time.time()
    for step, sample in enumerate(loader, 1):
        model_input, target = sweep.batch_to_model_input(
            "cspn", sample, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = sweep.extract_pred(model(*model_input))
        loss = sweep.masked_l1(prediction, target)
        sweep.validate_batch_numerics(prediction, target, loss)
        loss.backward()
        gradient_norm = controller.assert_finite_gradients()
        clip_gradients(model, gradient_norm, max_gradient_norm)
        optimizer.step()
        assert_finite_parameters(model)

        batch_size = int(target.shape[0])
        sample_count += batch_size
        loss_sum += float(loss.detach().item()) * batch_size
        gradient_sum += gradient_norm * batch_size
        metric = sweep.evaluate_error(
            target.detach(), prediction.detach())
        _update_metrics(metric_sum, metric, batch_size)
        if log_interval > 0 and step % log_interval == 0:
            print(
                "epoch=%d step=%d/%d loss=%.6f rmse=%.6f" % (
                    epoch, step, len(loader),
                    loss_sum / sample_count,
                    _finish_metrics(metric_sum, sample_count)["RMSE"]),
                flush=True)
    result = _finish_metrics(metric_sum, sample_count)
    result["loss"] = loss_sum / sample_count
    result["grad_norm"] = gradient_sum / sample_count
    result["seconds"] = time.time() - started
    result["lr"] = float(optimizer.param_groups[0]["lr"])
    return result


def train_task_aware_epoch(
        model: nn.Module,
        controller: CSPNQATController,
        teacher: nn.Module,
        teacher_propagation,
        loader,
        optimizer,
        device: torch.device,
        epoch: int,
        log_interval: int,
        max_gradient_norm: float,
        loss_weights: CSPNTaskLossWeights,
        boundary_threshold_m: float):
    set_qat_train_mode(model)
    teacher.eval()
    metric_sum = _metric_accumulator()
    loss_sum = 0.0
    sample_count = 0
    gradient_sum = 0.0
    started = time.time()
    for step, sample in enumerate(loader, 1):
        model_input, target = sweep.batch_to_model_input(
            "cspn", sample, device)
        optimizer.zero_grad(set_to_none=True)
        prediction, losses = task_aware_forward(
            model,
            controller,
            teacher,
            teacher_propagation,
            model_input,
            target,
            loss_weights,
            boundary_threshold_m,
        )
        losses["total"].backward()
        gradient_norm = controller.assert_finite_gradients()
        clip_gradients(model, gradient_norm, max_gradient_norm)
        optimizer.step()
        assert_finite_parameters(model)

        batch_size = int(target.shape[0])
        sample_count += batch_size
        loss_sum += float(losses["total"].detach().item()) * batch_size
        gradient_sum += gradient_norm * batch_size
        metric = sweep.evaluate_error(
            target.detach(), prediction.detach())
        _update_metrics(metric_sum, metric, batch_size)
        if log_interval > 0 and step % log_interval == 0:
            print(
                "epoch=%d step=%d/%d loss=%.6f rmse=%.6f" % (
                    epoch, step, len(loader),
                    loss_sum / sample_count,
                    _finish_metrics(metric_sum, sample_count)["RMSE"]),
                flush=True)
    result = _finish_metrics(metric_sum, sample_count)
    result["loss"] = loss_sum / sample_count
    result["grad_norm"] = gradient_sum / sample_count
    result["seconds"] = time.time() - started
    result["lr"] = float(optimizer.param_groups[0]["lr"])
    return result


def evaluate_epoch(model: nn.Module, loader, device: torch.device):
    model.eval()
    metric_sum = _metric_accumulator()
    loss_sum = 0.0
    sample_count = 0
    started = time.time()
    with torch.no_grad():
        for sample in loader:
            model_input, target = sweep.batch_to_model_input(
                "cspn", sample, device)
            prediction = sweep.extract_pred(model(*model_input))
            loss = sweep.masked_l1(prediction, target)
            sweep.validate_batch_numerics(prediction, target, loss)
            batch_size = int(target.shape[0])
            sample_count += batch_size
            loss_sum += float(loss.item()) * batch_size
            metric = sweep.evaluate_error(target, prediction)
            _update_metrics(metric_sum, metric, batch_size)
    result = _finish_metrics(metric_sum, sample_count)
    result["loss"] = loss_sum / sample_count
    result["grad_norm"] = ""
    result["seconds"] = time.time() - started
    result["lr"] = ""
    return result


def _checkpoint_payload(
        epoch: int, model: nn.Module, controller: CSPNQATController,
        optimizer, scheduler, tracker: QATConvergenceTracker,
        contract, config: TrainingConfig, history,
        train_generator: torch.Generator, validation):
    return {
        "epoch": int(epoch),
        "net": controller.qat_state_dict(),
        "canonical_net": controller.canonical_state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "tracker": tracker.state_dict(),
        "contract": contract,
        "training_config": asdict(config),
        "history": list(history),
        "train_generator_state": train_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(model.parameters().__next__().device),
        "val": dict(validation),
    }


def _restore_checkpoint(
        path: Path, model: nn.Module, optimizer, scheduler,
        tracker: QATConvergenceTracker, contract,
        config: TrainingConfig, train_generator: torch.Generator):
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if contract["mode"] == "mixed_static":
        validate_mixed_resume_contract(payload["contract"], contract)
    else:
        validate_resume_contract(
            payload["contract"], contract["mode"],
            tuple(contract["calibration_indices"]),
            tuple(contract["owner_manifest"]))
    if payload["training_config"] != asdict(config):
        raise ValueError("resume training configuration changed")
    model.load_state_dict(payload["net"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    tracker.load_state_dict(payload["tracker"])
    train_generator.set_state(payload["train_generator_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.cuda.set_rng_state(
        payload["cuda_rng_state"], model.parameters().__next__().device)
    return int(payload["epoch"]) + 1, list(payload["history"])


def _write_metrics(path: Path, history) -> None:
    fields = (
        "epoch", "split", "loss", "MSE", "RMSE", "MAE", "ABS_REL",
        "DELTA1.02", "DELTA1.05", "DELTA1.10", "DELTA1.25",
        "DELTA1.25^2", "DELTA1.25^3", "seconds", "lr", "grad_norm",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow(dict((field, row[field]) for field in fields))


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _epoch_row(epoch: int, split: str, values):
    row = {"epoch": epoch, "split": split}
    for key in (
            "loss", "MSE", "RMSE", "MAE", "ABS_REL", "DELTA1.02",
            "DELTA1.05", "DELTA1.10", "DELTA1.25", "DELTA1.25^2",
            "DELTA1.25^3", "seconds", "lr", "grad_norm"):
        row[key] = values[key]
    return row


def main(argv=None) -> None:
    args = parse_args(argv)
    validate_mode_paths(args)
    if not args.device.startswith("cuda"):
        raise ValueError("CSPN QAT requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.fold_max_error < 0.0 or args.log_interval < 0:
        raise ValueError("fold error and log interval must be nonnegative")
    config = TrainingConfig(
        epochs=args.epochs,
        patience=args.patience,
        min_relative_improvement=args.min_relative_improvement,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        workers=args.workers,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        max_gradient_norm=args.max_gradient_norm,
        seed=args.seed,
    )
    checkpoint_path = Path(args.checkpoint)
    calibration_metadata_path = Path(args.calibration_metadata)
    metadata = load_calibration_metadata(calibration_metadata_path)
    mixed = None
    if args.mode == "mixed_static":
        mixed = load_mixed_precision_inputs(args)
        validate_mixed_cli_config(args, mixed.precision_config)
    saved_args = saved_checkpoint_args(checkpoint_path, args)
    torch.set_num_threads(int(saved_args.torch_threads))
    sweep.seed_all(config.seed)
    torch.backends.cudnn.benchmark = bool(saved_args.cudnn_benchmark)
    torch.backends.cuda.matmul.allow_tf32 = bool(saved_args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(saved_args.allow_tf32)
    device = torch.device(args.device)

    output_dir = Path(args.output_root) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "mixed_static":
        prepared = prepare_mixed_qat_model(
            saved_args,
            checkpoint_path,
            metadata,
            mixed,
            args.fold_max_error,
            device,
            config.seed,
        )
        model, controller = prepared[0], prepared[1]
        architecture, load_report = prepared[6], prepared[7]
        preparation, manifest = prepared[8], prepared[9]
        teacher, teacher_propagation, teacher_architecture, teacher_load = \
            prepare_teacher(saved_args, checkpoint_path, device)
        if teacher_architecture != architecture or teacher_load != load_report:
            raise RuntimeError("student and teacher official checkpoint loads differ")
    else:
        prepared = prepare_qat_model(
            saved_args, checkpoint_path, metadata, args.mode,
            args.fold_max_error, device, config.seed)
        model, controller = prepared[0], prepared[1]
        architecture, load_report = prepared[5], prepared[6]
        preparation, manifest = prepared[7], prepared[8]
    owners = owner_manifest_tuple(manifest)

    sweep.seed_all(config.seed)
    train_generator = torch.Generator().manual_seed(config.seed)
    excluded = metadata.evaluation_indices \
        if args.mode == "mixed_static" else ()
    trainloader, valloader, early_stopping_indices = build_loaders(
        saved_args, config, train_generator, excluded)
    if args.mode == "mixed_static":
        contract = mixed_checkpoint_contract(
            metadata,
            owners,
            mixed,
            stem_runner._sha256(calibration_metadata_path),
            early_stopping_indices,
            _propagation_config(),
        )
    else:
        contract = checkpoint_contract(
            args.mode, metadata.calibration_indices, owners)
    _write_json(output_dir / "manifest.json", {
        "architecture": architecture,
        "load_report": load_report,
        "preparation": preparation,
        "quantization": manifest,
        "calibration_indices": list(metadata.calibration_indices),
        "early_stopping_indices": list(early_stopping_indices),
        "evaluation_indices": list(metadata.evaluation_indices),
        "training": asdict(config),
        "contract": contract,
    })
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.learning_rate,
        momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3,
        threshold=1e-4, min_lr=1e-6)
    tracker = QATConvergenceTracker(
        config.epochs, config.patience,
        config.min_relative_improvement)
    start_epoch = 1
    history = []
    if args.resume is not None:
        start_epoch, history = _restore_checkpoint(
            Path(args.resume), model, optimizer, scheduler,
            tracker, contract, config, train_generator)

    final_epoch = start_epoch - 1
    for epoch in range(start_epoch, config.epochs + 1):
        if args.mode == "mixed_static":
            training = train_task_aware_epoch(
                model,
                controller,
                teacher,
                teacher_propagation,
                trainloader,
                optimizer,
                device,
                epoch,
                args.log_interval,
                config.max_gradient_norm,
                mixed.loss_weights,
                mixed.boundary_threshold_m,
            )
        else:
            training = train_epoch(
                model, controller, trainloader, optimizer,
                device, epoch, args.log_interval,
                config.max_gradient_norm)
        validation = evaluate_epoch(model, valloader, device)
        validation["lr"] = float(optimizer.param_groups[0]["lr"])
        history.extend((
            _epoch_row(epoch, "train", training),
            _epoch_row(epoch, "validation", validation),
        ))
        _write_metrics(output_dir / "metrics.csv", history)

        improved = float(validation["RMSE"]) < tracker.best_rmse
        should_stop = tracker.update(epoch, validation["RMSE"])
        scheduler.step(validation["RMSE"])
        payload = _checkpoint_payload(
            epoch, model, controller, optimizer, scheduler,
            tracker, contract, config, history,
            train_generator, validation)
        torch.save(payload, output_dir / "last.pt")
        if improved:
            torch.save(payload, output_dir / "best_qat.pt")
            torch.save({
                "net": controller.canonical_state_dict(),
                "epoch": epoch,
                "val": dict(validation),
                "contract": contract,
                "quantization": manifest,
                "preparation": preparation,
                "architecture": architecture,
            }, output_dir / "best.pt")
        final_epoch = epoch
        print(
            "mode=%s epoch=%d train_rmse=%.6f val_rmse=%.6f "
            "best_rmse=%.6f lr=%.6g" % (
                args.mode, epoch, training["RMSE"], validation["RMSE"],
                tracker.best_rmse, optimizer.param_groups[0]["lr"]),
            flush=True)
        if should_stop:
            break

    if final_epoch < start_epoch:
        raise RuntimeError("resume checkpoint already reached max epochs")
    _write_json(output_dir / "run_summary.json", {
        "mode": args.mode,
        "final_epoch": final_epoch,
        "best_epoch": tracker.best_epoch,
        "best_rmse": tracker.best_rmse,
        "stop_reason": tracker.reason,
        "train_samples": len(trainloader.dataset),
        "validation_samples": len(valloader.dataset),
    })


if __name__ == "__main__":
    main()
