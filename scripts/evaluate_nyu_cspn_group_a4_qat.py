#!/usr/bin/env python3
"""Evaluate CSPN Static/Dynamic Group-8 W4A4 on the fresh hard path."""

from __future__ import annotations

import argparse
from argparse import Namespace
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_nyu_cspn_activation_resolution as base  # noqa: E402
from scripts import run_nyu_cspn_stem_precision as stem_runner  # noqa: E402
from scripts import run_nyu_cspn_task_sensitive_bits as task_runner  # noqa: E402
from scripts import train_nyu_cspn_group_a4_qat as qat_runner  # noqa: E402
from scripts import train_nyu_iteration_sweep as sweep  # noqa: E402
from scripts.hardware_aligned_quantization import (  # noqa: E402
    HardwareAlignedInstrumentor,
    prepare_hardware_model,
)
from scripts.run_nyu_rtn_quantization import (  # noqa: E402
    calibration_dataset,
    evaluation_dataset,
    seeded_sample,
    write_csv,
    write_json,
)
from spn_quant.adapters import install_model_semantic_adapter  # noqa: E402
from spn_quant.propagation import install_propagation_adapter  # noqa: E402
from spn_quant.activation_boundaries import (  # noqa: E402
    CSPNActivationBoundaryController,
)
from spn_quant import cspn_task_sensitive_bits as allocation  # noqa: E402


CONFIGURATIONS = (
    "FP32",
    "PTQ_STATIC_G8_W4A4",
    "PTQ_DYNAMIC_G8_W4A4",
    "QAT_STATIC_G8_W4A4",
    "QAT_DYNAMIC_G8_W4A4",
)
EXPECTED_CONFIGS = (
    "FP32",
    "UNIFORM_W6A6",
    "P3_T3",
    "MIXED_TASK_AWARE_QAT",
)
RAW_PREDICTION_FIELDS = {
    "gt", "fp32", "pred", "abs_err", "valid_gt", "nonfinite",
    "sample_index", "model", "config", "sparse", "rgb",
}
VISUAL_PREDICTION_FIELDS = RAW_PREDICTION_FIELDS | {"model_rgb"}


def configuration_mode(name: str):
    return {
        "FP32": None,
        "PTQ_STATIC_G8_W4A4": "static",
        "PTQ_DYNAMIC_G8_W4A4": "dynamic",
        "QAT_STATIC_G8_W4A4": "static",
        "QAT_DYNAMIC_G8_W4A4": "dynamic",
    }[name]


def mixed_checkpoint_for(name: str, args):
    if name not in EXPECTED_CONFIGS:
        raise ValueError("unknown mixed QAT evaluation configuration")
    if name == "MIXED_TASK_AWARE_QAT":
        return Path(args.mixed_checkpoint)
    return None


def validate_mixed_assignment_contracts(
        p3_assignment: allocation.BitAssignment,
        mixed_assignment: allocation.BitAssignment,
        basis: allocation.CostBasis,
        maximum_bits: float):
    if p3_assignment.weight_bits != mixed_assignment.weight_bits:
        raise ValueError("P3/T3 and mixed QAT weight assignments differ")
    audit = allocation.audit_activation_budget(
        mixed_assignment, basis, maximum_bits)
    if not audit.feasible:
        raise ValueError("mixed QAT activation budget is infeasible")
    return audit


def validate_canonical_state(state) -> None:
    offending = [
        key for key in state
        if ".parametrizations.weight." in str(key)
    ]
    if offending:
        raise ValueError("canonical checkpoint contains parametrization keys")


def validate_prediction_coverage(root: Path, indices) -> None:
    validate_prediction_coverage_for(root, indices, CONFIGURATIONS)


def validate_prediction_coverage_for(root: Path, indices, configurations) -> None:
    expected = set(int(index) for index in indices)
    if len(expected) != 64:
        raise ValueError("prediction coverage requires 64 unique indices")
    for config in configurations:
        directory = Path(root) / "predictions" / config
        observed = set(
            int(path.stem.split("_")[1])
            for path in directory.glob("sample_*.npz"))
        if observed != expected:
            raise RuntimeError(
                "prediction coverage mismatch for %s" % config)


def visualization_dataset(saved_args):
    return sweep.NyuHdf5Dataset(
        csv_file=saved_args.eval_list,
        root_dir=str(saved_args.data_root),
        split="val",
        n_sample=saved_args.n_sample,
        seed=saved_args.seed,
    )


def upgrade_prediction_visuals(root: Path, indices, dataset) -> None:
    _upgrade_prediction_visuals(root, indices, dataset, CONFIGURATIONS)


def upgrade_mixed_prediction_visuals(root: Path, indices, dataset) -> None:
    _upgrade_prediction_visuals(root, indices, dataset, EXPECTED_CONFIGS)


def _upgrade_prediction_visuals(
        root: Path, indices, dataset, configurations) -> None:
    root = Path(root)
    declared_indices = tuple(int(index) for index in indices)
    if len(set(declared_indices)) != len(declared_indices):
        raise ValueError("visualization sample indices must be unique")
    for index in declared_indices:
        sample = dataset[index]
        rgbd = sample["rgbd"]
        if not torch.is_tensor(rgbd) or rgbd.ndim != 3 or \
                int(rgbd.shape[0]) < 3:
            raise ValueError("visualization dataset must provide CHW RGBD")
        natural_rgb = rgbd[:3].permute(1, 2, 0).numpy()
        if not bool(np.isfinite(natural_rgb).all()):
            raise ValueError("visualization RGB must be finite")
        if float(natural_rgb.min()) < 0.0 or \
                float(natural_rgb.max()) > 1.0:
            raise ValueError("visualization RGB must lie in [0, 1]")
        natural_rgb = natural_rgb.astype(np.float32, copy=False)
        for config in configurations:
            path = root / "predictions" / config / \
                ("sample_%05d.npz" % index)
            with np.load(path, allow_pickle=False) as source:
                payload = dict(
                    (key, source[key]) for key in source.files)
            fields = set(payload)
            if fields != RAW_PREDICTION_FIELDS and \
                    fields != VISUAL_PREDICTION_FIELDS:
                raise ValueError(
                    "prediction payload fields changed: %s" % config)
            if int(payload["sample_index"].item()) != index:
                raise ValueError("prediction visualization index changed")
            if str(payload["model"].item()) != "cspn" or \
                    str(payload["config"].item()) != config:
                raise ValueError("prediction visualization identity changed")
            model_rgb = payload["rgb"] if \
                fields == RAW_PREDICTION_FIELDS else payload["model_rgb"]
            if model_rgb.shape != natural_rgb.shape or \
                    model_rgb.shape[:2] != payload["gt"].shape:
                raise ValueError("prediction RGB shapes changed")
            if not bool(np.isfinite(model_rgb).all()):
                raise ValueError("model RGB must be finite")
            if fields == VISUAL_PREDICTION_FIELDS:
                if not np.array_equal(payload["rgb"], natural_rgb):
                    raise ValueError(
                        "natural visualization RGB changed: %s" % config)
                continue
            payload["model_rgb"] = model_rgb
            payload["rgb"] = natural_rgb
            pending = path.with_name(path.name + ".pending")
            with pending.open("wb") as handle:
                np.savez_compressed(handle, **payload)
            pending.replace(path)
    write_json(root / "prediction_visualization_manifest.json", {
        "model": "cspn",
        "samples": len(declared_indices),
        "configurations": list(configurations),
        "rgb": "natural_nyu_hdf5_display_rgb",
        "model_rgb": "exact_official_cspn_model_input",
        "sparse": "exact_official_cspn_500_point_input",
    })


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--fp32-checkpoint", required=True)
    parser.add_argument("--static-checkpoint")
    parser.add_argument("--dynamic-checkpoint")
    parser.add_argument("--mixed-checkpoint")
    parser.add_argument("--precision-config")
    parser.add_argument("--assignment")
    parser.add_argument("--cost-basis")
    parser.add_argument("--mixed-protocol", action="store_true")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--calibration-metadata", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold-max-error", type=float, required=True)
    parser.add_argument("--sample-capacity", type=int, required=True)
    return parser.parse_args(argv)


def validate_protocol_paths(args) -> None:
    legacy_paths = (args.static_checkpoint, args.dynamic_checkpoint)
    mixed_paths = (
        args.mixed_checkpoint,
        args.precision_config,
        args.assignment,
        args.cost_basis,
    )
    if args.mixed_protocol:
        if any(path is None for path in mixed_paths):
            raise ValueError("mixed evaluation requires all mixed protocol paths")
        if any(path is not None for path in legacy_paths):
            raise ValueError("legacy QAT checkpoints are invalid in mixed protocol")
        return
    if any(path is None for path in legacy_paths):
        raise ValueError("legacy evaluation requires static and dynamic checkpoints")
    if any(path is not None for path in mixed_paths):
        raise ValueError("mixed protocol paths require --mixed-protocol")


@dataclass
class EvaluationContext:
    reference_model: torch.nn.Module
    quantized_model: torch.nn.Module
    saved_args: Namespace
    instrumentor: HardwareAlignedInstrumentor
    boundary_controller: CSPNActivationBoundaryController
    propagation: object
    reference_capture: base.ModuleOutputCapture
    quantized_capture: base.ModuleOutputCapture
    config: dict
    trainset: object
    preparation: dict


@dataclass
class MixedEvaluationContext:
    reference_model: torch.nn.Module
    quantized_model: torch.nn.Module
    saved_args: Namespace
    instrumentor: object
    boundary_controller: object
    propagation: object
    stem: object
    reference_capture: object
    quantized_capture: object
    config: dict
    trainset: object
    preparation: dict
    assignment: allocation.BitAssignment
    budget: allocation.ActivationBudgetAudit
    qat_source: object


def _source_args(path: Path, args) -> Namespace:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    saved_args = Namespace(**payload["args"])
    if saved_args.model != "cspn":
        raise ValueError("evaluation source checkpoint must contain CSPN")
    saved_args.from_scratch = True
    saved_args.data_root = args.data_root
    saved_args.device = args.device
    saved_args.seed = args.seed
    return saved_args


def _configuration(name: str):
    mode = configuration_mode(name)
    if mode is None:
        return base._configuration("FP32", set(), set(), None)
    return base._configuration(
        name, base.ORDINARY_GROUPS, base.ORDINARY_GROUPS,
        base.PROPAGATION_A8_Q13,
        granularity="hybrid_group_tensor", group_size=8,
        dynamic=mode == "dynamic")


def _load_canonical_checkpoint(
        model: torch.nn.Module, path: Path, mode: str,
        calibration_indices) -> dict:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    state = payload["net"]
    validate_canonical_state(state)
    if payload["contract"]["mode"] != mode:
        raise ValueError("QAT checkpoint quantization mode changed")
    if tuple(payload["contract"]["calibration_indices"]) != tuple(
            calibration_indices):
        raise ValueError("QAT checkpoint calibration indices changed")
    model.load_state_dict(state, strict=True)
    return {
        "epoch": int(payload["epoch"]),
        "validation": payload["val"],
    }


def _load_mixed_canonical_checkpoint(
        model: torch.nn.Module,
        path: Path,
        mixed: qat_runner.MixedPrecisionInputs):
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    state = payload["net"]
    validate_canonical_state(state)
    contract = payload["contract"]
    if contract["mode"] != "mixed_static":
        raise ValueError("mixed QAT checkpoint mode changed")
    if contract["precision_config_sha256"] != \
            mixed.precision_config_sha256:
        raise ValueError("mixed QAT precision config changed")
    if contract["assignment_sha256"] != mixed.assignment_sha256:
        raise ValueError("mixed QAT assignment changed")
    if contract["cost_basis_sha256"] != mixed.cost_basis_sha256:
        raise ValueError("mixed QAT cost basis changed")
    if contract["assignment"] != task_runner.assignment_payload(
            mixed.assignment):
        raise ValueError("mixed QAT checkpoint assignment payload changed")
    model.load_state_dict(state, strict=True)
    return {
        "epoch": int(payload["epoch"]),
        "validation": payload["val"],
    }


def mixed_assignment_for(
        name: str,
        registry: allocation.AllocationRegistry,
        mixed: qat_runner.MixedPrecisionInputs):
    if name == "UNIFORM_W6A6":
        return allocation.uniform_assignment(registry, 6, 6)
    if name == "P3_T3":
        return allocation.p3_t3_assignment(registry)
    if name == "MIXED_TASK_AWARE_QAT":
        return mixed.assignment
    raise ValueError("mixed assignment requested for non-quantized config")


def build_mixed_context(
        name: str,
        args,
        metadata,
        mixed: qat_runner.MixedPrecisionInputs,
        device: torch.device):
    if name not in EXPECTED_CONFIGS[1:]:
        raise ValueError("mixed quantized context requires a quantized config")
    source_path = Path(args.fp32_checkpoint)
    saved_args = _source_args(source_path, args)
    reference_model, reference_architecture, reference_load = \
        base._load_cspn(saved_args, source_path, device)
    trainset = calibration_dataset(saved_args)
    sample = seeded_sample(
        trainset, metadata["calibration_indices"][0], args.seed)
    model_args = base._model_args(saved_args, sample, device)
    quantized_model, architecture, load_report, preparation = \
        stem_runner._prepare_model(
            saved_args,
            source_path,
            device,
            model_args,
            args.fold_max_error,
        )
    if architecture != reference_architecture or load_report != reference_load:
        raise RuntimeError("fresh mixed CSPN source loads differ")
    instrumentor, boundary_controller, propagation, stem = \
        stem_runner._build_quantization_context(
            quantized_model, preparation, args.seed)
    stem_runner._calibrate(
        quantized_model,
        saved_args,
        trainset,
        metadata["calibration_indices"],
        device,
        args.seed,
        instrumentor,
        boundary_controller,
        propagation,
        stem,
        name,
    )
    stem_runner._validate_site_contract(instrumentor, boundary_controller)
    qat_source = None
    checkpoint = mixed_checkpoint_for(name, args)
    if checkpoint is not None:
        qat_source = _load_mixed_canonical_checkpoint(
            quantized_model, checkpoint, mixed)
        instrumentor.refresh_parameter_sources()
        if not torch.equal(
                quantized_model.conv1_1.weight.detach().cpu(),
                stem.original_weight):
            raise RuntimeError(
                "mixed QAT changed stem weights without a stem refresh contract")
    registry = task_runner.expected_registry()
    assignment = mixed_assignment_for(name, registry, mixed)
    p3_assignment = allocation.p3_t3_assignment(registry)
    if name == "MIXED_TASK_AWARE_QAT":
        budget = validate_mixed_assignment_contracts(
            p3_assignment,
            assignment,
            mixed.cost_basis,
            float(mixed.precision_config["search"]["activation_budget_bits"]),
        )
    else:
        budget = allocation.audit_activation_budget(
            assignment,
            mixed.cost_basis,
            8.0,
        )
    candidate = task_runner.RuntimeCandidate(name, "final", assignment)
    config, _, _ = task_runner.configure_runtime_context(
        candidate,
        instrumentor,
        boundary_controller,
        propagation,
        stem,
    )
    config["qat_source"] = qat_source
    return MixedEvaluationContext(
        reference_model=reference_model,
        quantized_model=quantized_model,
        saved_args=saved_args,
        instrumentor=instrumentor,
        boundary_controller=boundary_controller,
        propagation=propagation,
        stem=stem,
        reference_capture=base.ModuleOutputCapture(
            reference_model, base.CSPN_BLOCK_SITES),
        quantized_capture=base.ModuleOutputCapture(
            quantized_model, base.CSPN_BLOCK_SITES),
        config=config,
        trainset=trainset,
        preparation=preparation,
        assignment=assignment,
        budget=budget,
        qat_source=qat_source,
    )


def prepare_deployment_state(
        name: str, model: torch.nn.Module, saved_args: Namespace,
        trainset, calibration_indices, device: torch.device, seed: int,
        instrumentor, boundary_controller, propagation, qat_path):
    mode = configuration_mode(name)
    if mode is None:
        raise ValueError("deployment preparation requires quantization")
    base._calibrate(
        model, saved_args, trainset, calibration_indices,
        device, seed, instrumentor, boundary_controller, propagation)
    if name.startswith("QAT_"):
        if qat_path is None:
            raise ValueError("QAT deployment requires a checkpoint")
        source = _load_canonical_checkpoint(
            model, Path(qat_path), mode, calibration_indices)
        instrumentor.refresh_parameter_sources()
        return source
    if qat_path is not None:
        raise ValueError("PTQ deployment cannot load a QAT checkpoint")
    return None


def build_context(name: str, args, metadata, device: torch.device):
    source_path = Path(args.fp32_checkpoint)
    saved_args = _source_args(source_path, args)
    reference_model, reference_architecture, reference_load = \
        base._load_cspn(saved_args, source_path, device)
    quantized_model, quantized_architecture, quantized_load = \
        base._load_cspn(saved_args, source_path, device)
    if reference_architecture != quantized_architecture:
        raise RuntimeError("fresh CSPN architectures differ")
    if reference_load != quantized_load:
        raise RuntimeError("fresh CSPN checkpoint loads differ")

    trainset = calibration_dataset(saved_args)
    sample = seeded_sample(
        trainset, metadata["calibration_indices"][0], args.seed)
    model_args = base._model_args(saved_args, sample, device)
    preparation = prepare_hardware_model(
        quantized_model, model_args,
        excluded_pairs=(("conv1_1", "bn1"),))
    if float(preparation["primary_max_abs_error"]) > args.fold_max_error:
        raise RuntimeError("Conv-BN fold exceeds declared error threshold")

    semantic = install_model_semantic_adapter(
        quantized_model, "cspn", strict=True)
    boundaries = semantic.activation_boundaries()
    semantic.close()
    instrumentor = HardwareAlignedInstrumentor(
        quantized_model, base.cspn_quant_group,
        preparation["fused_relu_producers"],
        externally_owned_outputs=base.strict_owned_outputs(),
        externally_owned_inputs=base.strict_owned_inputs())
    boundary_controller = CSPNActivationBoundaryController(
        quantized_model, boundaries)
    propagation = install_propagation_adapter("cspn", quantized_model)
    mode = configuration_mode(name)
    qat_source = None
    if mode is not None:
        qat_path = None
        if name == "QAT_STATIC_G8_W4A4":
            qat_path = Path(args.static_checkpoint)
        elif name == "QAT_DYNAMIC_G8_W4A4":
            qat_path = Path(args.dynamic_checkpoint)
        qat_source = prepare_deployment_state(
            name,
            quantized_model, saved_args, trainset,
            metadata["calibration_indices"], device, args.seed,
            instrumentor, boundary_controller, propagation, qat_path)
        base.validate_strict_site_contract(instrumentor, boundary_controller)

    config = _configuration(name)
    config["qat_source"] = qat_source
    return EvaluationContext(
        reference_model=reference_model,
        quantized_model=quantized_model,
        saved_args=saved_args,
        instrumentor=instrumentor,
        boundary_controller=boundary_controller,
        propagation=propagation,
        reference_capture=base.ModuleOutputCapture(
            reference_model, base.CSPN_BLOCK_SITES),
        quantized_capture=base.ModuleOutputCapture(
            quantized_model, base.CSPN_BLOCK_SITES),
        config=config,
        trainset=trainset,
        preparation=preparation,
    )


def _full_validation(context: EvaluationContext, args):
    base._configure_quantized(
        context.config, context.instrumentor, context.boundary_controller,
        context.propagation, {})
    context.instrumentor.clear_activation_recorder()
    context.boundary_controller.clear_activation_recorder()
    dataset = evaluation_dataset(context.saved_args)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    model = context.reference_model \
        if context.config["name"] == "FP32" else context.quantized_model
    model.eval()
    rows = []
    with torch.no_grad():
        offset = 0
        for sample in loader:
            model_input, target = sweep.batch_to_model_input(
                "cspn", sample, torch.device(args.device))
            prediction = sweep.extract_pred(model(*model_input))
            sweep.validate_batch_numerics(prediction, target)
            for batch_index in range(int(target.shape[0])):
                gt = target[batch_index, 0].detach().cpu().numpy()
                pred = prediction[
                    batch_index, 0].detach().cpu().numpy()
                sparse = sample["rgbd"][batch_index, 3].numpy()
                metrics, _ = base.depth_sample_metrics(gt, pred, sparse)
                metrics.update({
                    "model": "cspn",
                    "config": context.config["name"],
                    "sample_index": offset + batch_index,
                })
                rows.append(metrics)
            offset += int(target.shape[0])
    if len(rows) != len(dataset):
        raise RuntimeError("full validation sample coverage is incomplete")
    return rows


def _aggregate(rows, config: str):
    output = {"model": "cspn", "config": config, "samples": len(rows)}
    for key in (
            "RMSE", "MAE", "ABS_REL", "IRMSE",
            "flat_RMSE", "boundary_RMSE", "nonfinite_ratio",
            "nonpositive_ratio"):
        output[key] = float(np.mean(np.asarray(
            [float(row[key]) for row in rows], dtype=np.float64)))
    return output


def precision_summary(
        model: torch.nn.Module,
        assignment: allocation.BitAssignment,
        basis: allocation.CostBasis,
        activation_budget: allocation.ActivationBudgetAudit):
    modules = dict(model.named_modules())
    weight_elements = dict(
        (name, int(modules[name].weight.numel()))
        for name, bits in assignment.weight_bits)
    total_weight_elements = sum(weight_elements.values())
    weight_macs = dict(basis.weight_macs)
    total_weight_macs = sum(weight_macs.values())
    weight_bits = dict(assignment.weight_bits)
    return {
        "average_weight_bits": sum(
            weight_bits[name] * weight_elements[name]
            for name in weight_elements) / float(total_weight_elements),
        "average_activation_bits": activation_budget.average_activation_bits,
        "activation_element_fractions": dict(
            activation_budget.activation_element_fractions),
        "w8_weight_element_fraction": sum(
            weight_elements[name] for name in weight_elements
            if weight_bits[name] == 8) / float(total_weight_elements),
        "w8_weight_mac_fraction": sum(
            weight_macs[name] for name in weight_macs
            if weight_bits[name] == 8) / float(total_weight_macs),
    }


def mixed_acceptance_report(
        sample_rows,
        propagation_rows,
        mixed: qat_runner.MixedPrecisionInputs):
    rows = tuple(
        row for row in sample_rows
        if str(row["config"]) == "MIXED_TASK_AWARE_QAT")
    if len(rows) != 64:
        raise ValueError("mixed QAT acceptance requires 64 sample rows")
    propagation = tuple(
        row for row in propagation_rows
        if str(row["config"]) == "MIXED_TASK_AWARE_QAT" and
        str(row["split"]) == "evaluation")
    anchors = tuple(
        float(row["anchor_max_error"]) for row in propagation
        if str(row["signal"]) == "anchor")
    constraints = tuple(
        row for row in propagation
        if str(row["signal"]) == "affinity_constraints")
    if not anchors or not constraints:
        raise ValueError("mixed QAT propagation acceptance rows are incomplete")
    metrics = {
        "rmse_m": float(np.mean(np.asarray(
            [float(row["RMSE"]) for row in rows], dtype=np.float64))),
        "average_activation_bits": mixed.budget.average_activation_bits,
        "nonfinite_ratio": float(np.mean(np.asarray(
            [float(row["nonfinite_ratio"]) for row in rows],
            dtype=np.float64))),
        "nonpositive_ratio": float(np.mean(np.asarray(
            [float(row["nonpositive_ratio"]) for row in rows],
            dtype=np.float64))),
        "anchor_max_error": max(anchors),
        "coefficient_sum_max_error": max(
            float(row["coefficient_sum_max_error"])
            for row in constraints),
        "contraction_violation_ratio": max(
            float(row["contraction_violation_rate"])
            for row in constraints),
    }
    thresholds = mixed.precision_config["acceptance"]
    gates = {
        "rmse_m": metrics["rmse_m"] <= float(thresholds["rmse_m"]),
        "average_activation_bits": metrics["average_activation_bits"] <=
            float(thresholds["average_activation_bits"]),
        "nonfinite_ratio": metrics["nonfinite_ratio"] ==
            float(thresholds["nonfinite_ratio"]),
        "nonpositive_ratio": metrics["nonpositive_ratio"] ==
            float(thresholds["nonpositive_ratio"]),
        "anchor_max_error": metrics["anchor_max_error"] ==
            float(thresholds["anchor_max_error"]),
        "coefficient_sum_max_error":
            metrics["coefficient_sum_max_error"] ==
            float(thresholds["coefficient_sum_max_error"]),
        "contraction_violation_ratio":
            metrics["contraction_violation_ratio"] ==
            float(thresholds["contraction_violation_ratio"]),
    }
    return {
        "accepted": all(gates.values()),
        "metrics": metrics,
        "thresholds": dict(thresholds),
        "gates": gates,
        "failed_gates": [name for name in gates if not gates[name]],
    }


def _append_result(target, result) -> None:
    for key in target:
        target[key].extend(result[key])


def _close_context(context: EvaluationContext) -> None:
    context.reference_capture.close()
    context.quantized_capture.close()
    context.instrumentor.close()
    context.boundary_controller.close()
    context.propagation.close()


def _close_mixed_context(context: MixedEvaluationContext) -> None:
    context.reference_capture.close()
    context.quantized_capture.close()
    context.stem.close()
    context.instrumentor.close()
    context.boundary_controller.close()
    context.propagation.close()


def _run_mixed_evaluation(args, metadata, device: torch.device) -> None:
    mixed = qat_runner.load_mixed_precision_inputs(args)
    registry = task_runner.expected_registry()
    validate_mixed_assignment_contracts(
        allocation.p3_t3_assignment(registry),
        mixed.assignment,
        mixed.cost_basis,
        float(mixed.precision_config["search"]["activation_budget_bits"]),
    )
    calibration_indices = metadata["calibration_indices"]
    evaluation_indices = metadata["evaluation_indices"]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = {
        "sample_rows": [],
        "region_rows": [],
        "propagation_rows": [],
        "block_rows": [],
        "tensor_rows": [],
        "channel_rows": [],
        "layer_rows": [],
        "merge_rows": [],
    }
    full_rows = []
    aggregate_rows = []
    manifests = []
    precision_rows = []
    stem_rows = []

    for name in EXPECTED_CONFIGS:
        print("fresh hard mixed evaluation config=%s" % name, flush=True)
        if name == "FP32":
            context = build_context(name, args, metadata, device)
            assignment = None
        else:
            context = build_mixed_context(name, args, metadata, mixed, device)
            assignment = context.assignment
            calibration_result = base.run_configuration(
                context.reference_model,
                context.quantized_model,
                context.saved_args,
                context.trainset,
                calibration_indices,
                "calibration",
                device,
                args.seed,
                context.config,
                context.instrumentor,
                context.boundary_controller,
                context.propagation,
                {},
                context.reference_capture,
                context.quantized_capture,
                args.sample_capacity,
            )
            _append_result(rows, calibration_result)
        current_full = _full_validation(context, args)
        full_rows.extend(current_full)
        aggregate_rows.append(_aggregate(current_full, name))
        fixed_dataset = evaluation_dataset(context.saved_args)
        fixed_result = base.run_configuration(
            context.reference_model,
            context.quantized_model,
            context.saved_args,
            fixed_dataset,
            evaluation_indices,
            "evaluation",
            device,
            args.seed,
            context.config,
            context.instrumentor,
            context.boundary_controller,
            context.propagation,
            {},
            context.reference_capture,
            context.quantized_capture,
            args.sample_capacity,
            prediction_root=output_root,
        )
        _append_result(rows, fixed_result)
        if assignment is not None:
            precision = precision_summary(
                context.quantized_model,
                assignment,
                mixed.cost_basis,
                context.budget,
            )
            precision["config"] = name
            precision_rows.append(precision)
            for row in context.stem.statistics():
                current = dict(row)
                current["config"] = name
                stem_rows.append(current)
            manifests.append({
                "config": name,
                "assignment": task_runner.assignment_payload(assignment),
                "preparation": context.preparation,
                "qat_source": context.qat_source,
            })
            _close_mixed_context(context)
        else:
            manifests.append({
                "config": name,
                "assignment": "FP32",
                "preparation": context.preparation,
                "qat_source": None,
            })
            _close_context(context)
        torch.cuda.empty_cache()

    visual_saved_args = _source_args(Path(args.fp32_checkpoint), args)
    upgrade_mixed_prediction_visuals(
        output_root,
        evaluation_indices,
        visualization_dataset(visual_saved_args),
    )
    validate_prediction_coverage_for(
        output_root, evaluation_indices, EXPECTED_CONFIGS)
    acceptance = mixed_acceptance_report(
        rows["sample_rows"], rows["propagation_rows"], mixed)
    write_csv(output_root / "aggregate_metrics.csv", aggregate_rows)
    write_csv(output_root / "sample_metrics_full.csv", full_rows)
    write_csv(output_root / "sample_metrics_64.csv", rows["sample_rows"])
    write_csv(output_root / "region_metrics_64.csv", rows["region_rows"])
    write_csv(
        output_root / "propagation_metrics.csv", rows["propagation_rows"])
    write_csv(output_root / "block_metrics.csv", rows["block_rows"])
    write_csv(
        output_root / "activation_tensor_metrics.csv", rows["tensor_rows"])
    write_csv(
        output_root / "activation_channel_metrics.csv", rows["channel_rows"])
    write_csv(
        output_root / "activation_layer_metrics.csv", rows["layer_rows"])
    write_csv(output_root / "precision_summary.csv", precision_rows)
    write_csv(output_root / "stem_metrics.csv", stem_rows)
    write_json(output_root / "acceptance_report.json", acceptance)
    write_json(output_root / "evaluation_manifest.json", {
        "configurations": list(EXPECTED_CONFIGS),
        "calibration_indices": list(calibration_indices),
        "evaluation_indices": list(evaluation_indices),
        "runs": manifests,
        "acceptance": acceptance,
    })
    print("four fresh hard mixed configurations evaluated", flush=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    validate_protocol_paths(args)
    if not args.device.startswith("cuda"):
        raise ValueError("CSPN hard evaluation requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.batch_size <= 0 or args.workers < 0 or \
            args.fold_max_error < 0.0 or args.sample_capacity <= 0:
        raise ValueError("evaluation numeric arguments are invalid")
    metadata = json.loads(
        Path(args.calibration_metadata).read_text(encoding="utf-8"))
    calibration_indices = tuple(
        int(index) for index in metadata["calibration_indices"])
    evaluation_indices = tuple(
        int(index) for index in metadata["evaluation_indices"])
    if len(calibration_indices) != 128 or len(evaluation_indices) != 64:
        raise ValueError("evaluation requires stratified 128 and paired 64")
    if metadata["calibration_source"]["selection"] != \
            "32_tail_96_kmedoids":
        raise ValueError("evaluation requires stratified calibration")
    metadata["calibration_indices"] = calibration_indices
    metadata["evaluation_indices"] = evaluation_indices

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sweep.seed_all(args.seed)
    if args.mixed_protocol:
        _run_mixed_evaluation(args, metadata, device)
        return
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = {
        "sample_rows": [],
        "region_rows": [],
        "propagation_rows": [],
        "block_rows": [],
        "tensor_rows": [],
        "channel_rows": [],
        "layer_rows": [],
        "merge_rows": [],
    }
    full_rows = []
    aggregate_rows = []
    manifests = []

    for name in CONFIGURATIONS:
        print("fresh hard evaluation config=%s" % name, flush=True)
        context = build_context(name, args, metadata, device)
        if configuration_mode(name) is not None:
            calibration_result = base.run_configuration(
                context.reference_model, context.quantized_model,
                context.saved_args, context.trainset,
                calibration_indices, "calibration", device, args.seed,
                context.config, context.instrumentor, context.boundary_controller,
                context.propagation, {}, context.reference_capture,
                context.quantized_capture, args.sample_capacity)
            _append_result(rows, calibration_result)
        current_full = _full_validation(context, args)
        full_rows.extend(current_full)
        aggregate_rows.append(_aggregate(current_full, name))

        fixed_dataset = evaluation_dataset(context.saved_args)
        fixed_result = base.run_configuration(
            context.reference_model, context.quantized_model,
            context.saved_args, fixed_dataset, evaluation_indices,
            "evaluation", device, args.seed, context.config,
            context.instrumentor, context.boundary_controller, context.propagation,
            {}, context.reference_capture, context.quantized_capture,
            args.sample_capacity, prediction_root=output_root)
        _append_result(rows, fixed_result)
        manifests.append({
            "config": name,
            "mode": configuration_mode(name),
            "preparation": context.preparation,
            "qat_source": context.config["qat_source"],
        })
        _close_context(context)
        torch.cuda.empty_cache()

    visual_saved_args = _source_args(Path(args.fp32_checkpoint), args)
    upgrade_prediction_visuals(
        output_root, evaluation_indices,
        visualization_dataset(visual_saved_args))
    validate_prediction_coverage(output_root, evaluation_indices)
    write_csv(output_root / "aggregate_metrics.csv", aggregate_rows)
    write_csv(output_root / "sample_metrics_full.csv", full_rows)
    write_csv(output_root / "sample_metrics_64.csv", rows["sample_rows"])
    write_csv(output_root / "region_metrics_64.csv", rows["region_rows"])
    write_csv(output_root / "propagation_metrics.csv",
              rows["propagation_rows"])
    write_csv(output_root / "block_metrics.csv", rows["block_rows"])
    write_csv(output_root / "activation_tensor_metrics.csv",
              rows["tensor_rows"])
    write_csv(output_root / "activation_channel_metrics.csv",
              rows["channel_rows"])
    write_csv(output_root / "activation_layer_metrics.csv",
              rows["layer_rows"])
    write_json(output_root / "evaluation_manifest.json", {
        "configurations": list(CONFIGURATIONS),
        "calibration_indices": list(calibration_indices),
        "evaluation_indices": list(evaluation_indices),
        "prediction_visualization": {
            "rgb": "natural_nyu_hdf5_display_rgb",
            "model_rgb": "exact_official_cspn_model_input",
            "sparse": "exact_official_cspn_500_point_input",
        },
        "runs": manifests,
    })
    print("five fresh hard configurations evaluated", flush=True)


if __name__ == "__main__":
    main()
