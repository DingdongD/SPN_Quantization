#!/usr/bin/env python3
"""Train official CSPN with LSQ+ or HAWQ quantization-aware training."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts import run_nyu_cspn_activation_resolution as base  # noqa: E402
from scripts import run_nyu_cspn_task_sensitive_bits as task_bits  # noqa: E402
from scripts import train_nyu_cspn_group_a4_qat as qat_base  # noqa: E402
from scripts import train_nyu_iteration_sweep as sweep  # noqa: E402
from scripts.hardware_aligned_quantization import (  # noqa: E402
    HardwareAlignedInstrumentor,
    prepare_hardware_model,
)
from scripts.run_nyu_rtn_quantization import (  # noqa: E402
    calibration_dataset,
    seeded_sample,
)
from spn_quant.activation_boundaries import (  # noqa: E402
    CSPNActivationBoundaryController,
)
from spn_quant.adapters import install_model_semantic_adapter  # noqa: E402
from spn_quant.cspn_task_sensitive_bits import (  # noqa: E402
    AllocationRegistry,
    BitAssignment,
)
from spn_quant.propagation import (  # noqa: E402
    PropagationQuantConfig,
    install_propagation_adapter,
)
from spn_quant.qat.cspn import cspn_hard_activation_bits  # noqa: E402
from spn_quant.qat.cspn_methods import (  # noqa: E402
    CSPNMethodQATConfig,
    CSPNMethodQATController,
)
from spn_quant.qat.cspn_task_loss import (  # noqa: E402
    CSPNTaskLossWeights,
)
from spn_quant.qat.method_config import (  # noqa: E402
    CSPNMethodExperimentConfig,
    load_method_config,
)


METHODS = (
    "lsqplus_w4a4",
    "lsqplus_w6a6",
    "hawq_mixed_le6",
)
CHECKPOINT_FIELDS = frozenset((
    "model_state",
    "method_state",
    "optimizer",
    "scheduler",
    "epoch",
    "convergence",
    "assignment",
    "method_config",
    "owner_manifest",
    "history",
    "train_generator_state",
    "torch_rng_state",
    "numpy_rng_state",
    "cuda_rng_state",
))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--calibration-metadata", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--assignment")
    parser.add_argument("--resume")
    return parser.parse_args(argv)


def validate_method_paths(args) -> None:
    if args.method == "hawq_mixed_le6":
        if args.assignment is None:
            raise ValueError("HAWQ training requires an assignment path")
    elif args.assignment is not None:
        raise ValueError("assignment paths are valid only for HAWQ")


def expected_registry() -> AllocationRegistry:
    return task_bits.expected_registry()


def uniform_method_assignment(
        registry: AllocationRegistry, bits: int) -> BitAssignment:
    bits = int(bits)
    if bits not in (4, 6):
        raise ValueError("LSQ+ uniform precision must be 4 or 6 bits")
    return task_bits.allocation.uniform_assignment(registry, bits, bits)


def _assignment_payload(assignment: BitAssignment):
    return {
        "weight_bits": [
            {"module": name, "bits": bits}
            for name, bits in assignment.weight_bits],
        "activation_bits": [
            {"module": owner[0], "kind": owner[1], "bits": bits}
            for owner, bits in assignment.activation_bits],
    }


def load_hawq_assignment(
        path: Path, registry: AllocationRegistry) -> BitAssignment:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if float(payload["average_weight_bits"]) > 6.0 or \
            float(payload["average_activation_bits"]) > 6.0:
        raise ValueError("HAWQ assignment exceeds the six-bit budget")
    assignment = BitAssignment(
        weight_bits=tuple(
            (str(row["module"]), int(row["bits"]))
            for row in payload["weight_bits"]),
        activation_bits=tuple(
            ((str(row["module"]), str(row["kind"])), int(row["bits"]))
            for row in payload["activation_bits"]),
    )
    task_bits.validate_assignment_contract(assignment)
    stem_weights = registry.weights_by_block["stem"]
    stem_activations = registry.activations_by_block["stem"]
    depth_weights = registry.weights_by_block["initial_depth"]
    depth_activations = registry.activations_by_block["initial_depth"]
    weight_bits = dict(assignment.weight_bits)
    activation_bits = dict(assignment.activation_bits)
    if any(weight_bits[name] != 8 for name in stem_weights + depth_weights) or \
            any(activation_bits[owner] != 8
                for owner in stem_activations + depth_activations):
        raise ValueError("HAWQ fixed eight-bit blocks changed")
    return assignment


def validate_checkpoint_payload(payload) -> None:
    if set(payload) != CHECKPOINT_FIELDS:
        raise ValueError("training checkpoint field contract mismatch")


def validate_resume_contract(saved, expected) -> None:
    if saved != expected:
        raise ValueError("resume method or assignment contract changed")


def update_convergence(tracker, epoch: int, rmse: float):
    previous_best = tracker.best_rmse
    stop = tracker.update(epoch, rmse)
    return float(rmse) < previous_best, stop


def _propagation_config() -> PropagationQuantConfig:
    return PropagationQuantConfig(
        affinity_bits=8,
        confidence_bits=8,
        offset_bits=8,
        state_bits=8,
        coefficient_fraction_bits=13,
    )


def _saved_args(checkpoint: Path, args, config):
    payload = torch.load(
        str(checkpoint), map_location="cpu", weights_only=False)
    saved = SimpleNamespace(**payload["args"])
    if saved.model != "cspn":
        raise ValueError("QAT checkpoint must contain official CSPN")
    saved.from_scratch = True
    saved.data_root = args.data_root
    saved.device = args.device
    saved.seed = config.training.seed
    saved.batch_size = config.training.batch_size
    saved.val_batch_size = config.training.val_batch_size
    saved.workers = config.training.workers
    return saved


def _configure_hard_activations(
        instrumentor, boundary_controller, hard_propagation,
        assignment: BitAssignment) -> None:
    activation_specs = base.build_activation_specs(
        instrumentor, base.ORDINARY_GROUPS, 4, 8)
    boundary_specs = base.build_boundary_activation_specs(
        boundary_controller, 4, 8)
    activation_specs, boundary_specs = base.apply_activation_bit_assignment(
        activation_specs, boundary_specs, assignment.activation_bits)
    instrumentor.configure_components_with_ranges(
        4,
        4,
        base.ORDINARY_GROUPS,
        base.ORDINARY_GROUPS,
        activation_specs,
        False,
        activation_maxima={},
        weight_bit_overrides=dict(assignment.weight_bits),
        weight_modules=tuple(
            name for name, bits in assignment.weight_bits),
    )
    with torch.no_grad():
        for name, bits in assignment.weight_bits:
            module = instrumentor.modules[name]
            module.weight.copy_(instrumentor.original_weights[name].to(
                device=module.weight.device, dtype=module.weight.dtype))
    bit_widths = {}
    group_sizes = {}
    scale_factors = {}
    for name in boundary_controller.channels:
        owner = "boundary_controller.%s" % name, "boundary"
        spec = boundary_specs[owner]
        bit_widths[name] = int(spec.bits)
        group_sizes[name] = int(spec.group_size)
        scale_factors[name] = 1.0
    boundary_controller.configure_specs(
        bit_widths, group_sizes, scale_factors, quantize=True)
    hard_propagation.configure(_propagation_config())


def _activation_initialization_rows(
        instrumentor, boundary_controller, device: torch.device):
    rows = []
    observed = set()
    for key in instrumentor.activation_site_keys(base.ORDINARY_GROUPS):
        owner = base.activation_owner(key)
        observer = instrumentor.channel_observers[key] \
            if not isinstance(key, str) else \
            instrumentor.relu_channel_observers[key]
        minimum = float(observer.minimum.min().item())
        maximum = float(observer.maximum.max().item())
        if owner in observed:
            raise RuntimeError("activation initialization owner is duplicated")
        rows.append((owner, torch.tensor(
            [minimum, maximum], device=device, dtype=torch.float32)))
        observed.add(owner)
    for name in boundary_controller.channels:
        owner = "boundary_controller.%s" % name, "boundary"
        observer = boundary_controller.observers[name]
        if owner in observed:
            raise RuntimeError("boundary initialization owner is duplicated")
        rows.append((owner, torch.tensor(
            [observer.minimum, observer.maximum],
            device=device, dtype=torch.float32)))
        observed.add(owner)
    return tuple(rows)


def prepare_method_model(
        saved, checkpoint: Path, metadata, method: str,
        assignment: BitAssignment, config: CSPNMethodExperimentConfig,
        device: torch.device):
    model, architecture, load_report = base._load_cspn(
        saved, checkpoint, device)
    dataset = calibration_dataset(saved)
    if max(metadata.calibration_indices) >= len(dataset):
        raise ValueError("calibration index exceeds NYU train split")
    sample = seeded_sample(
        dataset, metadata.calibration_indices[0], config.training.seed)
    model_args = base._model_args(saved, sample, device)
    preparation = prepare_hardware_model(
        model, model_args, excluded_pairs=(("conv1_1", "bn1"),))
    if preparation["primary_max_abs_error"] > config.training.fold_max_error:
        raise RuntimeError("Conv-BN fold exceeds declared error threshold")
    semantic = install_model_semantic_adapter(model, "cspn", strict=True)
    boundaries = semantic.activation_boundaries()
    semantic.close()
    instrumentor = HardwareAlignedInstrumentor(
        model,
        base.cspn_quant_group,
        preparation["fused_relu_producers"],
        externally_owned_outputs=base.strict_owned_outputs(),
        externally_owned_inputs=base.strict_owned_inputs(),
    )
    boundary_controller = CSPNActivationBoundaryController(model, boundaries)
    hard_propagation = install_propagation_adapter("cspn", model)
    base._calibrate(
        model, saved, dataset, metadata.calibration_indices,
        device, config.training.seed, instrumentor,
        boundary_controller, hard_propagation)
    base.validate_strict_site_contract(instrumentor, boundary_controller)
    _configure_hard_activations(
        instrumentor, boundary_controller, hard_propagation, assignment)
    method_name = "lsqplus" if method.startswith("lsqplus") else "hawq"
    controller = CSPNMethodQATController(
        model,
        instrumentor,
        boundary_controller,
        hard_propagation,
        CSPNMethodQATConfig(
            method=method_name,
            weight_bits=assignment.weight_bits,
            activation_bits=cspn_hard_activation_bits(
                instrumentor, boundary_controller),
            propagation=_propagation_config(),
            hawq_range_momentum=config.hawq.activation_range_momentum,
        ),
    )
    controller.initialize_activations(_activation_initialization_rows(
        instrumentor, boundary_controller, device))
    controller.install()
    controller.set_runtime_statistics(False)
    manifest = controller.manifest()
    if manifest["weight_bits"] != assignment.weight_bits or \
            manifest["activation_bits"] != assignment.activation_bits:
        raise RuntimeError("method controller assignment changed")
    return (
        model, controller, instrumentor, boundary_controller,
        hard_propagation, architecture, load_report, preparation, manifest)


def _build_loaders(saved, metadata, config, generator):
    trainset = sweep.CspnOfficialDataset(
        csv_file=saved.train_list,
        root_dir=str(saved.data_root),
        split="train",
        n_sample=saved.n_sample,
        seed=config.training.seed,
    )
    valset = sweep.CspnOfficialDataset(
        csv_file=saved.eval_list,
        root_dir=str(saved.data_root),
        split="val",
        n_sample=saved.n_sample,
        seed=config.training.seed,
    )
    fixed = set(metadata.evaluation_indices)
    validation_indices = tuple(
        index for index in range(len(valset)) if index not in fixed)
    trainloader = DataLoader(
        trainset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.workers,
        pin_memory=True,
        drop_last=True,
        generator=generator,
    )
    valloader = DataLoader(
        Subset(valset, validation_indices),
        batch_size=config.training.val_batch_size,
        shuffle=False,
        num_workers=config.training.workers,
        pin_memory=True,
        drop_last=False,
    )
    if len(trainloader) == 0 or len(valloader) == 0:
        raise RuntimeError("empty CSPN method train or validation loader")
    return trainloader, valloader, validation_indices


def _metric_accumulator():
    return dict((key, 0.0) for key in sweep.METRIC_KEYS)


def _finish_metrics(total, samples: int):
    if samples <= 0:
        raise RuntimeError("CSPN method epoch has no samples")
    return dict((key, total[key] / samples) for key in sweep.METRIC_KEYS)


def _train_epoch(model, controller, teacher, teacher_propagation,
                 loader, optimizer, device, epoch, config, loss_weights):
    qat_base.set_qat_train_mode(model)
    controller.activation_modules.train()
    teacher.eval()
    total = _metric_accumulator()
    samples = 0
    loss_sum = 0.0
    gradient_sum = 0.0
    started = time.time()
    for step, sample in enumerate(loader, 1):
        model_input, target = sweep.batch_to_model_input(
            "cspn", sample, device)
        optimizer.zero_grad(set_to_none=True)
        prediction, losses = qat_base.task_aware_forward(
            model, controller, teacher, teacher_propagation,
            model_input, target, loss_weights,
            config.hawq.trace.boundary_threshold_m)
        losses["total"].backward()
        gradient_norm = controller.assert_finite_gradients()
        qat_base.clip_gradients(
            controller, gradient_norm, config.training.max_gradient_norm)
        optimizer.step()
        qat_base.assert_finite_parameters(controller)
        batch = int(target.shape[0])
        samples += batch
        loss_sum += float(losses["total"].detach().item()) * batch
        gradient_sum += gradient_norm * batch
        metrics = sweep.evaluate_error(target.detach(), prediction.detach())
        for key in sweep.METRIC_KEYS:
            total[key] += float(metrics[key]) * batch
        if step % config.training.log_interval == 0:
            print("epoch=%d step=%d/%d loss=%.6f" % (
                epoch, step, len(loader), loss_sum / samples), flush=True)
    result = _finish_metrics(total, samples)
    result["loss"] = loss_sum / samples
    result["grad_norm"] = gradient_sum / samples
    result["seconds"] = time.time() - started
    result["lr"] = float(optimizer.param_groups[0]["lr"])
    return result


def _evaluate_epoch(model, controller, loader, device):
    model.eval()
    controller.activation_modules.eval()
    total = _metric_accumulator()
    samples = 0
    loss_sum = 0.0
    started = time.time()
    with torch.no_grad():
        for sample in loader:
            model_input, target = sweep.batch_to_model_input(
                "cspn", sample, device)
            prediction = sweep.extract_pred(model(*model_input))
            loss = sweep.masked_l1(prediction, target)
            sweep.validate_batch_numerics(prediction, target, loss)
            batch = int(target.shape[0])
            samples += batch
            loss_sum += float(loss.item()) * batch
            metrics = sweep.evaluate_error(target, prediction)
            for key in sweep.METRIC_KEYS:
                total[key] += float(metrics[key]) * batch
    result = _finish_metrics(total, samples)
    result["loss"] = loss_sum / samples
    result["grad_norm"] = ""
    result["seconds"] = time.time() - started
    result["lr"] = ""
    return result


def _contract(method, assignment, metadata, manifest):
    return {
        "method": method,
        "assignment": _assignment_payload(assignment),
        "calibration_indices": list(metadata.calibration_indices),
        "owner_manifest": list(manifest),
    }


def _checkpoint_payload(
        epoch, controller, optimizer, scheduler, tracker,
        assignment, method_config, owner_manifest, history, generator):
    return {
        "model_state": controller.canonical_model_state_dict(),
        "method_state": controller.method_state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "convergence": tracker.state_dict(),
        "assignment": _assignment_payload(assignment),
        "method_config": method_config,
        "owner_manifest": list(owner_manifest),
        "history": list(history),
        "train_generator_state": generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(
            controller.model.parameters().__next__().device),
    }


def _restore_checkpoint(
        path, controller, optimizer, scheduler, tracker,
        expected_contract, generator):
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    validate_checkpoint_payload(payload)
    saved_contract = {
        "method": payload["method_config"]["method"],
        "assignment": payload["assignment"],
        "calibration_indices": payload["method_config"][
            "calibration_indices"],
        "owner_manifest": payload["owner_manifest"],
    }
    validate_resume_contract(saved_contract, expected_contract)
    controller.load_canonical_model_state_dict(payload["model_state"])
    controller.load_method_state_dict(payload["method_state"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    tracker.load_state_dict(payload["convergence"])
    generator.set_state(payload["train_generator_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.cuda.set_rng_state(
        payload["cuda_rng_state"],
        controller.model.parameters().__next__().device)
    return int(payload["epoch"]) + 1, list(payload["history"])


def _owner_manifest(assignment: BitAssignment):
    return tuple(
        ["weight:%s" % name for name, bits in assignment.weight_bits] +
        ["activation:%s" % (owner,)
         for owner, bits in assignment.activation_bits] +
        ["guidance:fp32", "propagation:A8_Q13_INT32"])


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv=None) -> None:
    args = parse_args(argv)
    validate_method_paths(args)
    if not args.device.startswith("cuda"):
        raise ValueError("CSPN method QAT requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    config = load_method_config(Path(args.config))
    metadata = qat_base.load_calibration_metadata(
        Path(args.calibration_metadata))
    registry = expected_registry()
    if args.method == "lsqplus_w4a4":
        assignment = uniform_method_assignment(registry, 4)
    elif args.method == "lsqplus_w6a6":
        assignment = uniform_method_assignment(registry, 6)
    elif args.method == "hawq_mixed_le6":
        assignment = load_hawq_assignment(Path(args.assignment), registry)
    else:
        raise ValueError("unsupported CSPN method")
    device = torch.device(args.device)
    saved = _saved_args(Path(args.checkpoint), args, config)
    torch.set_num_threads(int(saved.torch_threads))
    sweep.seed_all(config.training.seed)
    torch.backends.cudnn.benchmark = bool(saved.cudnn_benchmark)
    torch.backends.cuda.matmul.allow_tf32 = bool(saved.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(saved.allow_tf32)
    prepared = prepare_method_model(
        saved, Path(args.checkpoint), metadata, args.method,
        assignment, config, device)
    model, controller = prepared[0], prepared[1]
    teacher, teacher_propagation, teacher_architecture, teacher_load = \
        qat_base.prepare_teacher(saved, Path(args.checkpoint), device)
    if teacher_architecture != prepared[5] or teacher_load != prepared[6]:
        raise RuntimeError("student and teacher official checkpoint loads differ")
    generator = torch.Generator().manual_seed(config.training.seed)
    trainloader, valloader, validation_indices = _build_loaders(
        saved, metadata, config, generator)
    owner_manifest = _owner_manifest(assignment)
    contract = _contract(args.method, assignment, metadata, owner_manifest)
    method_config = {
        "method": args.method,
        "experiment": asdict(config),
        "calibration_indices": list(metadata.calibration_indices),
        "validation_indices": list(validation_indices),
    }
    output = Path(args.output_root) / args.method
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "manifest.json", {
        "architecture": prepared[5],
        "load_report": prepared[6],
        "preparation": prepared[7],
        "quantization": prepared[8],
        "assignment": _assignment_payload(assignment),
        "method_config": method_config,
        "owner_manifest": list(owner_manifest),
    })
    optimizer = torch.optim.SGD(
        controller.parameters(),
        lr=config.training.learning_rate,
        momentum=config.training.momentum,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3,
        threshold=1e-4, min_lr=1e-6)
    tracker = qat_base.QATConvergenceTracker(
        config.training.epochs,
        config.training.patience,
        config.training.min_relative_improvement,
    )
    start_epoch = 1
    history = []
    if args.resume is not None:
        start_epoch, history = _restore_checkpoint(
            Path(args.resume), controller, optimizer, scheduler,
            tracker, contract, generator)
    loss_weights = CSPNTaskLossWeights(
        depth=config.loss.depth,
        boundary=config.loss.boundary,
        teacher=config.loss.teacher,
        propagation=config.loss.propagation,
    )
    for epoch in range(start_epoch, config.training.epochs + 1):
        train_values = _train_epoch(
            model, controller, teacher, teacher_propagation,
            trainloader, optimizer, device, epoch, config, loss_weights)
        validation = _evaluate_epoch(model, controller, valloader, device)
        scheduler.step(validation["RMSE"])
        history.extend((
            {"epoch": epoch, "split": "train", **train_values},
            {"epoch": epoch, "split": "validation", **validation},
        ))
        is_best, stop = update_convergence(
            tracker, epoch, validation["RMSE"])
        payload = _checkpoint_payload(
            epoch, controller, optimizer, scheduler, tracker,
            assignment, method_config, owner_manifest, history, generator)
        validate_checkpoint_payload(payload)
        torch.save(payload, output / "last.pt")
        if is_best:
            torch.save(payload, output / "best.pt")
        print(
            "method=%s epoch=%d train_RMSE=%.6f val_RMSE=%.6f" % (
                args.method, epoch, train_values["RMSE"],
                validation["RMSE"]),
            flush=True,
        )
        if stop:
            break
    if args.method == "hawq_mixed_le6":
        controller.freeze_activation_ranges()
    final_payload = _checkpoint_payload(
        int(history[-1]["epoch"]), controller, optimizer, scheduler,
        tracker, assignment, method_config, owner_manifest, history, generator)
    torch.save(final_payload, output / "final.pt")
    _write_json(output / "convergence.json", tracker.state_dict())
    teacher_propagation.close()
    controller.remove()
    prepared[2].close()
    prepared[3].close()
    prepared[4].close()


if __name__ == "__main__":
    main()
