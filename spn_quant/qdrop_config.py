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
class QDropPrecisionConfig:
    name: str
    weight_bits: int
    activation_bits: int
    official: int


@dataclass(frozen=True)
class QDropQuantizationConfig:
    variants: tuple[QDropPrecisionConfig, ...]
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
    capture_batch_size: int
    cache_cuda_byte_limit: int
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

    def precision(self, name):
        matches = tuple(
            variant for variant in self.quantization.variants
            if variant.name == str(name))
        if len(matches) != 1:
            raise KeyError("undeclared QDrop precision: %s" % name)
        return matches[0]


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
    names = tuple(
        variant.name for variant in config.quantization.variants)
    if len(names) != len(set(names)):
        raise ValueError("QDrop precision names must be unique")
    if names != ("W4A4", "W6A6"):
        raise ValueError(
            "QDrop requires matched W4A4 or W6A6 precision variants")
    for variant in config.quantization.variants:
        bits = (variant.weight_bits, variant.activation_bits)
        expected = "W%dA%d" % bits
        if bits not in ((4, 4), (6, 6)) or variant.name != expected:
            raise ValueError(
                "QDrop requires matched W4A4 or W6A6 precision variants")
        expected_official = 1 if bits == (4, 4) else 0
        if variant.official != expected_official:
            if bits == (6, 6):
                raise ValueError("QDrop W6A6 must be marked as an extension")
            raise ValueError("QDrop W4A4 must be marked as official")
    if config.quantization.weight_clip_ratio != 1.0:
        raise ValueError("QDrop exact weight contract requires clip ratio 1.0")
    _positive(
        "activation_scale_minimum",
        config.quantization.activation_scale_minimum)
    if config.search.calibration_samples != 128:
        raise ValueError("QDrop search requires 128 calibration samples")
    if config.search.reconstruction_samples != 112 or \
            config.search.validation_samples != 16:
        raise ValueError("QDrop search split must be 112+16")
    if config.search.reconstruction_samples + \
            config.search.validation_samples != \
            config.search.calibration_samples:
        raise ValueError("QDrop search split does not cover calibration data")
    if config.search.steps <= 0:
        raise ValueError("QDrop search steps must be positive")
    if config.search.quant_probabilities != (0.5,):
        raise ValueError("QDrop requires official fixed 0.5 probability")
    if config.reconstruction.steps != 20000:
        raise ValueError("formal QDrop reconstruction requires 20000 steps")
    if config.reconstruction.batch_size <= 0:
        raise ValueError("QDrop batch size must be positive")
    if config.reconstruction.capture_batch_size <= 0 or \
            config.reconstruction.capture_batch_size > \
            config.reconstruction.batch_size:
        raise ValueError(
            "QDrop capture batch size must be positive and not exceed batch size")
    if config.reconstruction.cache_cuda_byte_limit <= 0:
        raise ValueError("QDrop CUDA cache byte limit must be positive")
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
        ("variants", "weight_clip_ratio",
         "activation_scale_minimum"))
    variants = quantization["variants"]
    if not isinstance(variants, list) or not variants:
        raise ValueError("quantization variants must be a nonempty list")
    for index, variant in enumerate(variants):
        _require_keys(
            "quantization variant %d" % index, variant,
            ("name", "weight_bits", "activation_bits", "official"))
    search = payload["search"]
    _require_keys(
        "search", search,
        ("calibration_samples", "reconstruction_samples",
         "validation_samples", "steps", "quant_probabilities"))
    reconstruction = payload["reconstruction"]
    _require_keys(
        "reconstruction", reconstruction,
        ("batch_size", "capture_batch_size", "cache_cuda_byte_limit", "steps",
         "weight_learning_rate", "activation_learning_rate", "round_loss_weight",
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
            variants=tuple(QDropPrecisionConfig(
                name=str(variant["name"]),
                weight_bits=int(variant["weight_bits"]),
                activation_bits=int(variant["activation_bits"]),
                official=int(variant["official"]),
            ) for variant in variants),
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
            capture_batch_size=int(reconstruction["capture_batch_size"]),
            cache_cuda_byte_limit=int(
                reconstruction["cache_cuda_byte_limit"]),
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
