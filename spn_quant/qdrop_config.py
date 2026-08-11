"""Strict configuration for official-aligned QDrop W4A4 evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path


OFFICIAL_QDROP_COMMIT = "4a9ca007ce91b66620b911de97df36d5109ecae0"


@dataclass(frozen=True)
class QDropReferenceConfig:
    repository: str
    branch: str
    commit: str


@dataclass(frozen=True)
class QDropQuantizationConfig:
    weight_bits: int
    activation_bits: int
    weight_clip_ratio: float
    activation_scale_minimum: float


@dataclass(frozen=True)
class QDropSearchConfig:
    calibration_samples: int
    reconstruction_samples: int
    validation_samples: int
    steps: int
    quant_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class QDropReconstructionConfig:
    batch_size: int
    steps: int
    weight_learning_rate: float
    activation_learning_rate: float
    round_loss_weight: float
    warmup_fraction: float
    beta_start: float
    beta_end: float
    loss_power: float


@dataclass(frozen=True)
class QDropFormalConfig:
    seeds: tuple[int, ...]
    evaluation_samples: int
    evaluation_seed: int


@dataclass(frozen=True)
class QDropConfig:
    reference: QDropReferenceConfig
    quantization: QDropQuantizationConfig
    search: QDropSearchConfig
    reconstruction: QDropReconstructionConfig
    formal: QDropFormalConfig


def _require_keys(name, payload, fields):
    actual = set(payload)
    expected = set(fields)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise KeyError("%s missing required fields: %s" % (name, missing))
    if extra:
        raise KeyError("%s contains unknown fields: %s" % (name, extra))


def _positive(name, value):
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("%s must be positive and finite" % name)
    return value


def _validate(config):
    if config.reference.commit != OFFICIAL_QDROP_COMMIT:
        raise ValueError("official QDrop commit mismatch")
    if config.reference.branch != "qdrop":
        raise ValueError("official QDrop branch must be qdrop")
    if config.quantization.weight_bits != 4 or \
            config.quantization.activation_bits != 4:
        raise ValueError("QDrop protocol requires exactly W4A4")
    if config.quantization.weight_clip_ratio != 1.0:
        raise ValueError("QDrop exact weight contract requires clip ratio 1.0")
    _positive(
        "activation_scale_minimum",
        config.quantization.activation_scale_minimum)
    if config.search.calibration_samples != 1024:
        raise ValueError("QDrop search requires 1024 calibration samples")
    if config.search.reconstruction_samples != 896 or \
            config.search.validation_samples != 128:
        raise ValueError("QDrop search split must be 896+128")
    if config.search.reconstruction_samples + \
            config.search.validation_samples != \
            config.search.calibration_samples:
        raise ValueError("QDrop search split does not cover calibration data")
    if config.search.steps <= 0:
        raise ValueError("QDrop search steps must be positive")
    if 0.5 not in config.search.quant_probabilities:
        raise ValueError("QDrop search must include official 0.5 probability")
    if len(config.search.quant_probabilities) != \
            len(set(config.search.quant_probabilities)):
        raise ValueError("QDrop search probabilities must be unique")
    if any(value < 0.0 or value > 1.0
           for value in config.search.quant_probabilities):
        raise ValueError("QDrop probabilities must be in [0, 1]")
    if config.reconstruction.steps != 20000:
        raise ValueError("formal QDrop reconstruction requires 20000 steps")
    if config.reconstruction.batch_size <= 0:
        raise ValueError("QDrop batch size must be positive")
    _positive(
        "weight_learning_rate",
        config.reconstruction.weight_learning_rate)
    _positive(
        "activation_learning_rate",
        config.reconstruction.activation_learning_rate)
    _positive("loss_power", config.reconstruction.loss_power)
    if config.reconstruction.round_loss_weight < 0.0:
        raise ValueError("round_loss_weight cannot be negative")
    if not 0.0 <= config.reconstruction.warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if config.reconstruction.beta_start <= \
            config.reconstruction.beta_end or \
            config.reconstruction.beta_end <= 0.0:
        raise ValueError("QDrop beta range must decrease and remain positive")
    if len(config.formal.seeds) != 3 or \
            len(set(config.formal.seeds)) != 3:
        raise ValueError("formal QDrop seeds must contain three unique values")
    if config.formal.evaluation_samples != 64:
        raise ValueError("formal QDrop evaluation requires 64 samples")


def load_qdrop_config(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require_keys(
        "root", payload,
        ("reference", "quantization", "search", "reconstruction", "formal"))

    reference = payload["reference"]
    _require_keys("reference", reference, ("repository", "branch", "commit"))
    quantization = payload["quantization"]
    _require_keys(
        "quantization", quantization,
        ("weight_bits", "activation_bits", "weight_clip_ratio",
         "activation_scale_minimum"))
    search = payload["search"]
    _require_keys(
        "search", search,
        ("calibration_samples", "reconstruction_samples",
         "validation_samples", "steps", "quant_probabilities"))
    reconstruction = payload["reconstruction"]
    _require_keys(
        "reconstruction", reconstruction,
        ("batch_size", "steps", "weight_learning_rate",
         "activation_learning_rate", "round_loss_weight",
         "warmup_fraction", "beta_start", "beta_end", "loss_power"))
    formal = payload["formal"]
    _require_keys(
        "formal", formal,
        ("seeds", "evaluation_samples", "evaluation_seed"))

    config = QDropConfig(
        reference=QDropReferenceConfig(
            repository=str(reference["repository"]),
            branch=str(reference["branch"]),
            commit=str(reference["commit"])),
        quantization=QDropQuantizationConfig(
            weight_bits=int(quantization["weight_bits"]),
            activation_bits=int(quantization["activation_bits"]),
            weight_clip_ratio=float(quantization["weight_clip_ratio"]),
            activation_scale_minimum=float(
                quantization["activation_scale_minimum"])),
        search=QDropSearchConfig(
            calibration_samples=int(search["calibration_samples"]),
            reconstruction_samples=int(search["reconstruction_samples"]),
            validation_samples=int(search["validation_samples"]),
            steps=int(search["steps"]),
            quant_probabilities=tuple(
                float(value) for value in search["quant_probabilities"])),
        reconstruction=QDropReconstructionConfig(
            batch_size=int(reconstruction["batch_size"]),
            steps=int(reconstruction["steps"]),
            weight_learning_rate=float(
                reconstruction["weight_learning_rate"]),
            activation_learning_rate=float(
                reconstruction["activation_learning_rate"]),
            round_loss_weight=float(
                reconstruction["round_loss_weight"]),
            warmup_fraction=float(reconstruction["warmup_fraction"]),
            beta_start=float(reconstruction["beta_start"]),
            beta_end=float(reconstruction["beta_end"]),
            loss_power=float(reconstruction["loss_power"])),
        formal=QDropFormalConfig(
            seeds=tuple(int(value) for value in formal["seeds"]),
            evaluation_samples=int(formal["evaluation_samples"]),
            evaluation_seed=int(formal["evaluation_seed"])))
    _validate(config)
    return config
