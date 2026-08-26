"""Strict persisted-input contracts for the three official NYU models."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Tuple


DATASET_IDENTITY = "nyu_depth_v2"
SELECTION_IDENTITY = "32_tail_96_kmedoids"
CALIBRATION_COUNT = 128
EVALUATION_COUNT = 64

RAW_DESCRIPTOR_FIELDS = (
    "depth_mean",
    "depth_p50",
    "depth_p95",
    "depth_max",
    "depth_valid_ratio",
    "rgb_luminance_mean",
    "rgb_luminance_std",
    "rgb_contrast",
    "rgb_edge_density",
    "sparse_valid_count",
    "sparse_quadrant_0",
    "sparse_quadrant_1",
    "sparse_quadrant_2",
    "sparse_quadrant_3",
    "sparse_grid_occupancy",
    "sparse_centroid_spread",
)


def _activation_fields(model: str, groups) -> Tuple[str, ...]:
    return tuple(
        "%s_%s_%s" % (model, group, statistic)
        for group in groups
        for statistic in ("p99", "maximum", "channel_imbalance"))


ACTIVATION_DESCRIPTOR_FIELDS = {
    "dyspn": _activation_fields(
        "dyspn", ("encoder_stem", "decoder_fusion", "initial_depth",
                  "signed_guidance")),
    "nlspn": _activation_fields(
        "nlspn", ("encoder_stem", "decoder_fusion", "initial_depth",
                  "signed_guidance")),
    "completionformer": _activation_fields(
        "completionformer",
        ("encoder_stem", "decoder_fusion", "initial_depth",
         "signed_guidance", "attention_projection")),
}

CALIBRATION_INDEX_FIELDS = frozenset((
    "format_version", "model", "dataset", "split", "selection", "count",
    "indices", "checkpoint", "checkpoint_sha256", "data_root",
    "train_list", "train_list_sha256", "descriptor_schema_sha256",
    "ordered_identity_sha256",
))
EVALUATION_PROTOCOL_FIELDS = frozenset((
    "format_version", "model", "dataset", "split", "evaluation_samples",
    "evaluation_indices", "seed", "checkpoint", "checkpoint_sha256",
    "data_root", "evaluation_list", "evaluation_list_sha256",
    "ordered_identity_sha256",
))
CALIBRATION_METADATA_FIELDS = frozenset((
    "format_version", "model", "dataset", "checkpoint",
    "checkpoint_sha256", "data_root", "train_list", "train_list_sha256",
    "evaluation_list", "evaluation_list_sha256", "calibration_indices",
    "evaluation_indices", "calibration_source", "evaluation_source",
    "descriptor_schema", "cost_coverage", "artifact_sha256",
))
CALIBRATION_SOURCE_FIELDS = frozenset((
    "split", "selection", "count", "ordered_identity_sha256",
))
EVALUATION_SOURCE_FIELDS = frozenset((
    "split", "count", "seed", "ordered_identity_sha256",
))
DESCRIPTOR_SCHEMA_FIELDS = frozenset(("raw", "activation", "sha256"))
COST_COVERAGE_FIELDS = frozenset((
    "weight_modules", "activation_owners",
))
ARTIFACT_SHA256_FIELDS = frozenset((
    "calibration_indices", "evaluation_protocol", "weight_cost_rows",
    "activation_cost_rows",
))


@dataclass(frozen=True)
class StaticInputPaths:
    calibration_metadata: Path
    calibration_indices: Path
    evaluation_protocol: Path
    weight_cost_rows: Path
    activation_cost_rows: Path


@dataclass(frozen=True)
class ValidatedStaticInputs:
    model: str
    calibration_indices: Tuple[int, ...]
    evaluation_indices: Tuple[int, ...]
    evaluation_seed: int
    weight_costs: Tuple[Tuple[str, int], ...]
    activation_costs: Tuple[Tuple[Tuple[str, str], int], ...]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ordered_split_identity_sha256(split: str, indices) -> str:
    payload = [[str(split), int(index)] for index in indices]
    encoded = json.dumps(
        payload, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def descriptor_schema_sha256(model: str) -> str:
    payload = {
        "raw": list(RAW_DESCRIPTOR_FIELDS),
        "activation": list(ACTIVATION_DESCRIPTOR_FIELDS[model]),
    }
    encoded = json.dumps(
        payload, separators=(",", ":"), sort_keys=True,
        ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _load_exact_json(path: Path, fields, family: str):
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError("%s is missing: %s" % (family, source))
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != fields:
        raise KeyError("%s schema changed" % family)
    if isinstance(payload["format_version"], bool) or \
            int(payload["format_version"]) != 1:
        raise ValueError("%s format version changed" % family)
    return payload


def _ordered_indices(values, count: int, family: str) -> Tuple[int, ...]:
    if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values):
        raise TypeError("%s must contain integer identities" % family)
    indices = tuple(int(value) for value in values)
    if len(indices) != int(count):
        raise ValueError("%s must contain exactly %d identities" %
                         (family, int(count)))
    if len(indices) != len(set(indices)) or any(index < 0 for index in indices):
        raise ValueError("%s must be unique and nonnegative" % family)
    return indices


def _dataset_identities(path: Path, family: str) -> Tuple[str, ...]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError("%s is missing: %s" % (family, source))
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "Name" not in reader.fieldnames:
            raise ValueError("%s requires a Name column" % family)
        names = tuple(str(row["Name"]) for row in reader)
    if not names or any(not name for name in names) or \
            len(names) != len(set(names)):
        raise ValueError("%s identities must be nonempty and unique" % family)
    return names


def _weight_cost_rows(path: Path) -> Tuple[Tuple[str, int], ...]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ("module", "macs"):
            raise ValueError("weight costs require module,macs columns")
        rows = tuple((str(row["module"]), int(row["macs"])) for row in reader)
    if not rows or len(rows) != len(set(name for name, value in rows)) or \
            any(not name or value <= 0 for name, value in rows):
        raise ValueError("weight costs must be unique and positive")
    return rows


def _activation_cost_rows(
        path: Path) -> Tuple[Tuple[Tuple[str, str], int], ...]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ("site", "role", "elements"):
            raise ValueError(
                "activation costs require site,role,elements columns")
        rows = tuple(
            ((str(row["site"]), str(row["role"])), int(row["elements"]))
            for row in reader)
    owners = tuple(owner for owner, value in rows)
    if not rows or len(rows) != len(set(owners)) or any(
            not owner[0] or not owner[1] or value <= 0
            for owner, value in rows):
        raise ValueError("activation costs must be unique and positive")
    return rows


def _same_path(value, expected: Path, family: str) -> None:
    if Path(value).resolve() != Path(expected).resolve():
        raise ValueError("%s identity changed" % family)


def _validate_common_identity(payload, model_config, checkpoint_sha256) -> None:
    if str(payload["model"]) != str(model_config.model):
        raise ValueError("static input model identity changed")
    if str(payload["dataset"]) != DATASET_IDENTITY:
        raise ValueError("static input dataset identity changed")
    _same_path(payload["checkpoint"], model_config.checkpoint, "checkpoint")
    if str(payload["checkpoint_sha256"]) != checkpoint_sha256:
        raise ValueError("static input checkpoint SHA256 changed")
    _same_path(payload["data_root"], model_config.data_root, "data root")


def validate_static_input_bundle(
        model_config, paths: StaticInputPaths) -> ValidatedStaticInputs:
    metadata = _load_exact_json(
        paths.calibration_metadata, CALIBRATION_METADATA_FIELDS,
        "calibration metadata")
    calibration = _load_exact_json(
        paths.calibration_indices, CALIBRATION_INDEX_FIELDS,
        "calibration indices")
    evaluation = _load_exact_json(
        paths.evaluation_protocol, EVALUATION_PROTOCOL_FIELDS,
        "evaluation protocol")
    checkpoint_sha256 = file_sha256(model_config.checkpoint)
    for payload in (metadata, calibration, evaluation):
        _validate_common_identity(payload, model_config, checkpoint_sha256)

    if set(metadata["calibration_source"]) != CALIBRATION_SOURCE_FIELDS or \
            set(metadata["evaluation_source"]) != EVALUATION_SOURCE_FIELDS or \
            set(metadata["descriptor_schema"]) != DESCRIPTOR_SCHEMA_FIELDS or \
            set(metadata["cost_coverage"]) != COST_COVERAGE_FIELDS or \
            set(metadata["artifact_sha256"]) != ARTIFACT_SHA256_FIELDS:
        raise KeyError("calibration metadata nested schema changed")

    calibration_indices = _ordered_indices(
        calibration["indices"], CALIBRATION_COUNT,
        "calibration identities")
    metadata_calibration = _ordered_indices(
        metadata["calibration_indices"], CALIBRATION_COUNT,
        "metadata calibration identities")
    evaluation_indices = _ordered_indices(
        evaluation["evaluation_indices"], EVALUATION_COUNT,
        "evaluation identities")
    metadata_evaluation = _ordered_indices(
        metadata["evaluation_indices"], EVALUATION_COUNT,
        "metadata evaluation identities")
    if calibration_indices != metadata_calibration:
        raise ValueError("calibration identities differ across static inputs")
    if evaluation_indices != metadata_evaluation or \
            evaluation_indices != tuple(model_config.evaluation_indices):
        raise ValueError("evaluation identities differ across static inputs")
    if int(model_config.calibration_count) != CALIBRATION_COUNT:
        raise ValueError("configured calibration count changed")

    calibration_source = metadata["calibration_source"]
    evaluation_source = metadata["evaluation_source"]
    calibration_identity = ordered_split_identity_sha256(
        "train", calibration_indices)
    evaluation_identity = ordered_split_identity_sha256(
        "validation", evaluation_indices)
    if str(calibration["split"]) != "train" or \
            str(calibration["selection"]) != SELECTION_IDENTITY or \
            int(calibration["count"]) != CALIBRATION_COUNT or \
            str(calibration["ordered_identity_sha256"]) != \
            calibration_identity or \
            calibration_source != {
                "split": "train",
                "selection": SELECTION_IDENTITY,
                "count": CALIBRATION_COUNT,
                "ordered_identity_sha256": calibration_identity,
            }:
        raise ValueError("calibration selection identity changed")
    evaluation_seed = int(evaluation["seed"])
    if str(evaluation["split"]) != "validation" or \
            int(evaluation["evaluation_samples"]) != EVALUATION_COUNT or \
            str(evaluation["ordered_identity_sha256"]) != \
            evaluation_identity or \
            evaluation_source != {
                "split": "validation",
                "count": EVALUATION_COUNT,
                "seed": evaluation_seed,
                "ordered_identity_sha256": evaluation_identity,
            }:
        raise ValueError("evaluation protocol identity changed")

    train_list = Path(metadata["train_list"]).resolve()
    evaluation_list = Path(metadata["evaluation_list"]).resolve()
    _same_path(calibration["train_list"], train_list, "train list")
    _same_path(evaluation["evaluation_list"], evaluation_list,
               "evaluation list")
    train_sha256 = file_sha256(train_list)
    evaluation_sha256 = file_sha256(evaluation_list)
    if str(metadata["train_list_sha256"]) != train_sha256 or \
            str(calibration["train_list_sha256"]) != train_sha256:
        raise ValueError("train-list SHA256 changed")
    if str(metadata["evaluation_list_sha256"]) != evaluation_sha256 or \
            str(evaluation["evaluation_list_sha256"]) != evaluation_sha256:
        raise ValueError("evaluation-list SHA256 changed")
    train_names = _dataset_identities(train_list, "train list")
    evaluation_names = _dataset_identities(evaluation_list, "evaluation list")
    if max(calibration_indices) >= len(train_names):
        raise ValueError("calibration identity exceeds train split")
    if max(evaluation_indices) >= len(evaluation_names):
        raise ValueError("evaluation identity exceeds validation split")

    schema = metadata["descriptor_schema"]
    expected_schema_sha256 = descriptor_schema_sha256(model_config.model)
    if tuple(schema["raw"]) != RAW_DESCRIPTOR_FIELDS or \
            tuple(schema["activation"]) != \
            ACTIVATION_DESCRIPTOR_FIELDS[model_config.model] or \
            str(schema["sha256"]) != expected_schema_sha256 or \
            str(calibration["descriptor_schema_sha256"]) != \
            expected_schema_sha256:
        raise ValueError("descriptor schema identity changed")

    weight_costs = _weight_cost_rows(paths.weight_cost_rows)
    activation_costs = _activation_cost_rows(paths.activation_cost_rows)
    coverage = metadata["cost_coverage"]
    expected_weight_modules = tuple(str(value)
                                    for value in coverage["weight_modules"])
    expected_activation_owners = tuple(
        (str(value[0]), str(value[1]))
        for value in coverage["activation_owners"])
    if tuple(name for name, value in weight_costs) != \
            expected_weight_modules or \
            tuple(owner for owner, value in activation_costs) != \
            expected_activation_owners:
        raise ValueError("static input cost coverage changed")

    expected_hashes = {
        "calibration_indices": file_sha256(paths.calibration_indices),
        "evaluation_protocol": file_sha256(paths.evaluation_protocol),
        "weight_cost_rows": file_sha256(paths.weight_cost_rows),
        "activation_cost_rows": file_sha256(paths.activation_cost_rows),
    }
    if metadata["artifact_sha256"] != expected_hashes:
        raise ValueError("static input artifact SHA256 changed")
    return ValidatedStaticInputs(
        model=str(model_config.model),
        calibration_indices=calibration_indices,
        evaluation_indices=evaluation_indices,
        evaluation_seed=evaluation_seed,
        weight_costs=weight_costs,
        activation_costs=activation_costs,
    )
