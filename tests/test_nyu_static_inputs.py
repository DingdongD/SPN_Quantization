import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from spn_quant.nyu_static_inputs import (
    ACTIVATION_DESCRIPTOR_FIELDS,
    DATASET_IDENTITY,
    RAW_DESCRIPTOR_FIELDS,
    StaticInputPaths,
    ordered_split_identity_sha256,
    validate_static_input_bundle,
)


def _sha256(path):
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bundle(tmp_path, model="nlspn"):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"official checkpoint\n")
    train_list = tmp_path / "train.csv"
    evaluation_list = tmp_path / "val.csv"
    _write_csv(
        train_list, ("Name",),
        ({"Name": "train_%03d.h5" % index} for index in range(160)))
    _write_csv(
        evaluation_list, ("Name",),
        ({"Name": "val_%03d.h5" % index} for index in range(80)))
    calibration = tuple(range(128))
    evaluation = tuple(range(64))
    calibration_path = tmp_path / "calibration_indices.json"
    evaluation_path = tmp_path / "evaluation_protocol.json"
    weight_path = tmp_path / "weight_cost_rows.csv"
    activation_path = tmp_path / "activation_cost_rows.csv"
    metadata_path = tmp_path / "calibration_metadata.json"
    descriptor_schema = {
        "raw": list(RAW_DESCRIPTOR_FIELDS),
        "activation": list(ACTIVATION_DESCRIPTOR_FIELDS[model]),
    }
    descriptor_sha256 = hashlib.sha256(json.dumps(
        descriptor_schema, separators=(",", ":"), sort_keys=True,
    ).encode("ascii")).hexdigest()
    checkpoint_sha256 = _sha256(checkpoint)
    common = {
        "format_version": 1,
        "model": model,
        "dataset": DATASET_IDENTITY,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "data_root": str(tmp_path.resolve()),
    }
    calibration_path.write_text(json.dumps({
        **common,
        "split": "train",
        "selection": "32_tail_96_kmedoids",
        "count": 128,
        "indices": list(calibration),
        "train_list": str(train_list.resolve()),
        "train_list_sha256": _sha256(train_list),
        "descriptor_schema_sha256": descriptor_sha256,
        "ordered_identity_sha256": ordered_split_identity_sha256(
            "train", calibration),
    }), encoding="utf-8")
    evaluation_path.write_text(json.dumps({
        **common,
        "split": "validation",
        "evaluation_samples": 64,
        "evaluation_indices": list(evaluation),
        "seed": 20260812,
        "evaluation_list": str(evaluation_list.resolve()),
        "evaluation_list_sha256": _sha256(evaluation_list),
        "ordered_identity_sha256": ordered_split_identity_sha256(
            "validation", evaluation),
    }), encoding="utf-8")
    _write_csv(weight_path, ("module", "macs"), ({
        "module": "encoder.weight", "macs": 17,
    },))
    _write_csv(activation_path, ("site", "role", "elements"), ({
        "site": "activation::encoder.weight::input",
        "role": "module_input",
        "elements": 23,
    },))
    metadata_path.write_text(json.dumps({
        **common,
        "train_list": str(train_list.resolve()),
        "train_list_sha256": _sha256(train_list),
        "evaluation_list": str(evaluation_list.resolve()),
        "evaluation_list_sha256": _sha256(evaluation_list),
        "calibration_indices": list(calibration),
        "evaluation_indices": list(evaluation),
        "calibration_source": {
            "split": "train",
            "selection": "32_tail_96_kmedoids",
            "count": 128,
            "ordered_identity_sha256": ordered_split_identity_sha256(
                "train", calibration),
        },
        "evaluation_source": {
            "split": "validation",
            "count": 64,
            "seed": 20260812,
            "ordered_identity_sha256": ordered_split_identity_sha256(
                "validation", evaluation),
        },
        "descriptor_schema": {
            **descriptor_schema,
            "sha256": descriptor_sha256,
        },
        "cost_coverage": {
            "weight_modules": ["encoder.weight"],
            "activation_owners": [[
                "activation::encoder.weight::input", "module_input"]],
        },
        "artifact_sha256": {
            "calibration_indices": _sha256(calibration_path),
            "evaluation_protocol": _sha256(evaluation_path),
            "weight_cost_rows": _sha256(weight_path),
            "activation_cost_rows": _sha256(activation_path),
        },
    }), encoding="utf-8")
    model_config = SimpleNamespace(
        model=model,
        checkpoint=checkpoint,
        data_root=tmp_path,
        calibration_metadata=metadata_path,
        calibration_count=128,
        evaluation_indices=evaluation,
    )
    paths = StaticInputPaths(
        calibration_metadata=metadata_path,
        calibration_indices=calibration_path,
        evaluation_protocol=evaluation_path,
        weight_cost_rows=weight_path,
        activation_cost_rows=activation_path,
    )
    return model_config, paths


def test_semantic_static_input_bundle_validates_exact_order_and_costs(tmp_path):
    model_config, paths = _bundle(tmp_path)

    validated = validate_static_input_bundle(model_config, paths)

    assert validated.model == "nlspn"
    assert validated.calibration_indices == tuple(range(128))
    assert validated.evaluation_indices == tuple(range(64))


@pytest.mark.parametrize(
    "artifact, field, message",
    (
        ("calibration_metadata", "calibration_indices",
         "calibration identities"),
        ("calibration_metadata", "evaluation_indices",
         "evaluation identities"),
        ("calibration_metadata", "checkpoint_sha256", "checkpoint SHA256"),
        ("calibration_metadata", "cost_coverage", "cost coverage"),
    ),
)
def test_semantic_static_input_bundle_rejects_cross_file_changes(
        tmp_path, artifact, field, message):
    model_config, paths = _bundle(tmp_path)
    path = getattr(paths, artifact)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if field in ("calibration_indices", "evaluation_indices"):
        payload[field][0], payload[field][1] = payload[field][1], payload[field][0]
    elif field == "checkpoint_sha256":
        payload[field] = "0" * 64
    else:
        payload[field]["weight_modules"] = ["other.weight"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_static_input_bundle(model_config, paths)
