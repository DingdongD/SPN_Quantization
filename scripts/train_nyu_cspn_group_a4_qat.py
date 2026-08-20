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
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_nyu_cspn_activation_resolution as base  # noqa: E402
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
from spn_quant.qat import CSPNQATConfig, CSPNQATController  # noqa: E402
from spn_quant.activation_boundaries import (  # noqa: E402
    CSPNActivationBoundaryController,
)


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
    if payload["calibration_source"]["selection"] != \
            "32_tail_96_kmedoids":
        raise ValueError("CSPN QAT requires stratified calibration indices")
    return CalibrationMetadata(
        calibration_indices=calibration_indices,
        evaluation_indices=evaluation_indices,
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
        "--mode", choices=("static", "dynamic"), required=True)
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
    parser.add_argument("--resume")
    return parser.parse_args(argv)


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


def build_loaders(saved_args, config: TrainingConfig,
                  train_generator: torch.Generator):
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
    valset = _limited_dataset(valset, saved_args.max_val_samples)
    trainloader = DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.workers, pin_memory=True, drop_last=True,
        generator=train_generator)
    valloader = DataLoader(
        valset, batch_size=config.val_batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True, drop_last=False)
    if len(trainloader) == 0 or len(valloader) == 0:
        raise RuntimeError("empty CSPN QAT train or validation loader")
    return trainloader, valloader


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
        mode=mode, weight_bits=4, activation_bits=4,
        group_size=8, propagation=propagation_config)
    controller = CSPNQATController(
        model, instrumentor, boundary_controller, hard_propagation,
        weight_modules, qat_config)
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


def _metric_accumulator():
    return dict((key, 0.0) for key in sweep.METRIC_KEYS)


def _update_metrics(total, metric, batch_size: int) -> None:
    for key in sweep.METRIC_KEYS:
        total[key] += float(metric[key]) * batch_size


def _finish_metrics(total, samples: int):
    if samples <= 0:
        raise RuntimeError("CSPN QAT epoch has no samples")
    return dict((key, total[key] / samples) for key in sweep.METRIC_KEYS)


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
    metadata = load_calibration_metadata(Path(args.calibration_metadata))
    saved_args = saved_checkpoint_args(checkpoint_path, args)
    torch.set_num_threads(int(saved_args.torch_threads))
    sweep.seed_all(config.seed)
    torch.backends.cudnn.benchmark = bool(saved_args.cudnn_benchmark)
    torch.backends.cuda.matmul.allow_tf32 = bool(saved_args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(saved_args.allow_tf32)
    device = torch.device(args.device)

    output_dir = Path(args.output_root) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_qat_model(
        saved_args, checkpoint_path, metadata, args.mode,
        args.fold_max_error, device, config.seed)
    model, controller = prepared[0], prepared[1]
    architecture, load_report = prepared[5], prepared[6]
    preparation, manifest = prepared[7], prepared[8]
    owners = owner_manifest_tuple(manifest)
    contract = checkpoint_contract(
        args.mode, metadata.calibration_indices, owners)
    _write_json(output_dir / "manifest.json", {
        "architecture": architecture,
        "load_report": load_report,
        "preparation": preparation,
        "quantization": manifest,
        "calibration_indices": list(metadata.calibration_indices),
        "evaluation_indices": list(metadata.evaluation_indices),
        "training": asdict(config),
    })

    sweep.seed_all(config.seed)
    train_generator = torch.Generator().manual_seed(config.seed)
    trainloader, valloader = build_loaders(
        saved_args, config, train_generator)
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
