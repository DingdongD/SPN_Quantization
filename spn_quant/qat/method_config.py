"""Configuration contracts for CSPN LSQ+ and HAWQ experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Tuple


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("%s must be finite and positive" % name)
    return value


def _nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("%s must be finite and nonnegative" % name)
    return value


def _positive_integer(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError("%s must be positive" % name)
    return value


def _bits(name: str, values, expected) -> Tuple[int, ...]:
    bits = tuple(int(value) for value in values)
    if bits != tuple(expected):
        raise ValueError("%s must equal %s" % (name, tuple(expected)))
    return bits


@dataclass(frozen=True)
class LSQPlusMethodConfig:
    bits: Tuple[int, ...]
    initialization_batch_size: int
    initialization_batches: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bits", _bits(
            "LSQ+ bits", self.bits, (4, 6)))
        object.__setattr__(self, "initialization_batch_size",
                           _positive_integer(
                               "LSQ+ initialization batch size",
                               self.initialization_batch_size))
        object.__setattr__(self, "initialization_batches",
                           _positive_integer(
                               "LSQ+ initialization batches",
                               self.initialization_batches))
        if self.initialization_batch_size * self.initialization_batches != 128:
            raise ValueError("LSQ+ initialization must cover 128 samples")


@dataclass(frozen=True)
class HAWQTraceConfig:
    batch_size: int
    probes_per_batch: int
    seed: int
    depth_mse_weight: float
    boundary_mse_weight: float
    boundary_threshold_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "batch_size", _positive_integer(
            "HAWQ trace batch size", self.batch_size))
        object.__setattr__(self, "probes_per_batch", _positive_integer(
            "HAWQ probes per batch", self.probes_per_batch))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "depth_mse_weight", _positive(
            "HAWQ depth MSE weight", self.depth_mse_weight))
        object.__setattr__(self, "boundary_mse_weight", _nonnegative(
            "HAWQ boundary MSE weight", self.boundary_mse_weight))
        object.__setattr__(self, "boundary_threshold_m", _positive(
            "HAWQ boundary threshold", self.boundary_threshold_m))
        if 128 % self.batch_size != 0:
            raise ValueError("HAWQ trace batch size must divide 128")


@dataclass(frozen=True)
class HAWQMethodConfig:
    bits: Tuple[int, ...]
    maximum_average_weight_bits: float
    maximum_average_activation_bits: float
    fixed_blocks: Tuple[str, ...]
    activation_range_momentum: float
    trace: HAWQTraceConfig

    def __post_init__(self) -> None:
        object.__setattr__(self, "bits", _bits(
            "HAWQ bits", self.bits, (4, 6, 8)))
        object.__setattr__(self, "maximum_average_weight_bits", _positive(
            "HAWQ maximum average weight bits",
            self.maximum_average_weight_bits))
        object.__setattr__(self, "maximum_average_activation_bits", _positive(
            "HAWQ maximum average activation bits",
            self.maximum_average_activation_bits))
        if self.maximum_average_weight_bits != 6.0 or \
                self.maximum_average_activation_bits != 6.0:
            raise ValueError("HAWQ formal budgets must equal 6 bits")
        blocks = tuple(str(block) for block in self.fixed_blocks)
        if blocks != ("encoder_stem", "initial_depth"):
            raise ValueError("HAWQ fixed block contract changed")
        object.__setattr__(self, "fixed_blocks", blocks)
        momentum = float(self.activation_range_momentum)
        if not 0.0 <= momentum < 1.0:
            raise ValueError("HAWQ activation momentum must lie in [0, 1)")
        object.__setattr__(self, "activation_range_momentum", momentum)
        if not isinstance(self.trace, HAWQTraceConfig):
            raise TypeError("HAWQ trace configuration is invalid")


@dataclass(frozen=True)
class MethodLossConfig:
    depth: float
    boundary: float
    teacher: float
    propagation: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "depth", _positive(
            "depth loss weight", self.depth))
        object.__setattr__(self, "boundary", _nonnegative(
            "boundary loss weight", self.boundary))
        object.__setattr__(self, "teacher", _nonnegative(
            "teacher loss weight", self.teacher))
        object.__setattr__(self, "propagation", _nonnegative(
            "propagation loss weight", self.propagation))


@dataclass(frozen=True)
class TrainingMethodConfig:
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
    fold_max_error: float
    log_interval: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "epochs", _positive_integer(
            "epochs", self.epochs))
        object.__setattr__(self, "patience", _positive_integer(
            "patience", self.patience))
        object.__setattr__(self, "batch_size", _positive_integer(
            "batch size", self.batch_size))
        object.__setattr__(self, "val_batch_size", _positive_integer(
            "validation batch size", self.val_batch_size))
        workers = int(self.workers)
        if workers < 0:
            raise ValueError("workers must be nonnegative")
        object.__setattr__(self, "workers", workers)
        object.__setattr__(self, "min_relative_improvement", _positive(
            "minimum relative improvement", self.min_relative_improvement))
        object.__setattr__(self, "learning_rate", _positive(
            "learning rate", self.learning_rate))
        object.__setattr__(self, "momentum", _nonnegative(
            "momentum", self.momentum))
        object.__setattr__(self, "weight_decay", _nonnegative(
            "weight decay", self.weight_decay))
        object.__setattr__(self, "max_gradient_norm", _positive(
            "maximum gradient norm", self.max_gradient_norm))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "fold_max_error", _positive(
            "fold maximum error", self.fold_max_error))
        object.__setattr__(self, "log_interval", _positive_integer(
            "log interval", self.log_interval))


@dataclass(frozen=True)
class EvaluationMethodConfig:
    samples: int
    batch_size: int
    workers: int
    depth_min_m: float
    depth_max_m: float
    detail_samples: int
    font_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", _positive_integer(
            "evaluation samples", self.samples))
        if self.samples != 64:
            raise ValueError("formal evaluation requires 64 samples")
        object.__setattr__(self, "batch_size", _positive_integer(
            "evaluation batch size", self.batch_size))
        workers = int(self.workers)
        if workers < 0:
            raise ValueError("evaluation workers must be nonnegative")
        object.__setattr__(self, "workers", workers)
        minimum = _nonnegative("minimum depth", self.depth_min_m)
        maximum = _positive("maximum depth", self.depth_max_m)
        if maximum <= minimum:
            raise ValueError("maximum depth must exceed minimum depth")
        object.__setattr__(self, "depth_min_m", minimum)
        object.__setattr__(self, "depth_max_m", maximum)
        object.__setattr__(self, "detail_samples", _positive_integer(
            "detail samples", self.detail_samples))
        object.__setattr__(self, "font_size", _positive_integer(
            "font size", self.font_size))


@dataclass(frozen=True)
class CSPNMethodExperimentConfig:
    model: str
    source_revisions: Tuple[Tuple[str, str], ...]
    lsqplus: LSQPlusMethodConfig
    hawq: HAWQMethodConfig
    loss: MethodLossConfig
    training: TrainingMethodConfig
    gpus: Tuple[Tuple[str, int], ...]
    evaluation: EvaluationMethodConfig

    def __post_init__(self) -> None:
        if self.model != "cspn":
            raise ValueError("LSQ+/HAWQ experiment requires CSPN")
        names = tuple(name for name, revision in self.source_revisions)
        if names != ("lsqplus", "hawq"):
            raise ValueError("source revision contract changed")
        if any(len(revision) != 40 for name, revision in self.source_revisions):
            raise ValueError("source revisions must be full Git commits")
        expected_gpus = (
            "baselines", "lsqplus_w4a4", "lsqplus_w6a6",
            "hawq_mixed_le6")
        if tuple(name for name, gpu in self.gpus) != expected_gpus:
            raise ValueError("GPU owner contract changed")
        indices = tuple(gpu for name, gpu in self.gpus)
        if len(indices) != len(set(indices)) or any(gpu < 0 for gpu in indices):
            raise ValueError("formal jobs require distinct nonnegative GPUs")


def load_method_config(path: Path) -> CSPNMethodExperimentConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    revisions = payload["source_revisions"]
    lsqplus = payload["lsqplus"]
    hawq = payload["hawq"]
    trace = hawq["trace"]
    loss = payload["loss"]
    training = payload["training"]
    gpus = payload["gpus"]
    evaluation = payload["evaluation"]
    return CSPNMethodExperimentConfig(
        model=str(payload["model"]),
        source_revisions=(
            ("lsqplus", str(revisions["lsqplus"])),
            ("hawq", str(revisions["hawq"])),
        ),
        lsqplus=LSQPlusMethodConfig(
            bits=tuple(lsqplus["bits"]),
            initialization_batch_size=int(
                lsqplus["initialization_batch_size"]),
            initialization_batches=int(lsqplus["initialization_batches"]),
        ),
        hawq=HAWQMethodConfig(
            bits=tuple(hawq["bits"]),
            maximum_average_weight_bits=float(
                hawq["maximum_average_weight_bits"]),
            maximum_average_activation_bits=float(
                hawq["maximum_average_activation_bits"]),
            fixed_blocks=tuple(hawq["fixed_blocks"]),
            activation_range_momentum=float(
                hawq["activation_range_momentum"]),
            trace=HAWQTraceConfig(
                batch_size=int(trace["batch_size"]),
                probes_per_batch=int(trace["probes_per_batch"]),
                seed=int(trace["seed"]),
                depth_mse_weight=float(trace["depth_mse_weight"]),
                boundary_mse_weight=float(trace["boundary_mse_weight"]),
                boundary_threshold_m=float(trace["boundary_threshold_m"]),
            ),
        ),
        loss=MethodLossConfig(
            depth=float(loss["depth"]),
            boundary=float(loss["boundary"]),
            teacher=float(loss["teacher"]),
            propagation=float(loss["propagation"]),
        ),
        training=TrainingMethodConfig(
            epochs=int(training["epochs"]),
            patience=int(training["patience"]),
            min_relative_improvement=float(
                training["min_relative_improvement"]),
            batch_size=int(training["batch_size"]),
            val_batch_size=int(training["val_batch_size"]),
            workers=int(training["workers"]),
            learning_rate=float(training["learning_rate"]),
            momentum=float(training["momentum"]),
            weight_decay=float(training["weight_decay"]),
            max_gradient_norm=float(training["max_gradient_norm"]),
            seed=int(training["seed"]),
            fold_max_error=float(training["fold_max_error"]),
            log_interval=int(training["log_interval"]),
        ),
        gpus=(
            ("baselines", int(gpus["baselines"])),
            ("lsqplus_w4a4", int(gpus["lsqplus_w4a4"])),
            ("lsqplus_w6a6", int(gpus["lsqplus_w6a6"])),
            ("hawq_mixed_le6", int(gpus["hawq_mixed_le6"])),
        ),
        evaluation=EvaluationMethodConfig(
            samples=int(evaluation["samples"]),
            batch_size=int(evaluation["batch_size"]),
            workers=int(evaluation["workers"]),
            depth_min_m=float(evaluation["depth_min_m"]),
            depth_max_m=float(evaluation["depth_max_m"]),
            detail_samples=int(evaluation["detail_samples"]),
            font_size=int(evaluation["font_size"]),
        ),
    )
