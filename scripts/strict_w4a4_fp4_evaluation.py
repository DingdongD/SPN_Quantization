#!/usr/bin/env python3
"""Contracts and statistics for strict W4A4 and FP4 evaluation."""

from __future__ import division, print_function

import csv
import json
from pathlib import Path

import numpy as np


MODEL_ORDER = ("cspn", "dyspn", "nlspn", "completionformer")
METHOD_ORDER = ("rtn", "adaround", "brecq")
PRIMARY_CONFIGS = (
    "FP32", "FP4V_W4A4", "FP4V_W4E2M1", "FP4V_W4A8")
STRESS_CONFIGS = ("FP32", "HW_W4A4_full")
STRICT_METHODS = {
    "adaround": "adaround_strict",
    "brecq": "brecq_strict",
}
MAX_RELATIVE_RMSE_DEGRADATION = 0.10


def performance_decision(fp32_rmse, quant_rmse, rtn_rmse,
                         nonfinite_samples, nonfinite_pixels):
    values = np.asarray(
        [fp32_rmse, quant_rmse, rtn_rmse], dtype=np.float64)
    if not np.isfinite(values).all() or float(fp32_rmse) <= 0.0:
        raise ValueError("finite positive RMSE values are required")

    relative = (
        (float(quant_rmse) - float(fp32_rmse)) / float(fp32_rmse))
    if int(nonfinite_samples) != 0 or int(nonfinite_pixels) != 0:
        status = "rejected_nonfinite"
    elif relative > MAX_RELATIVE_RMSE_DEGRADATION:
        status = "rejected_fp32_degradation"
    elif float(quant_rmse) > float(rtn_rmse):
        status = "rejected_rtn_regression"
    else:
        status = "preserved"
    return {
        "relative_rmse_degradation": relative,
        "delta_vs_rtn": float(quant_rmse) - float(rtn_rmse),
        "status": status,
    }


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_reconstruction(metadata, method, model):
    if method == "rtn":
        if "reconstruction" in metadata:
            raise ValueError("RTN result carries a reconstruction contract")
        return

    reconstruction = metadata["reconstruction"]
    if int(reconstruction["exact_weight_contract"]) != 1:
        raise ValueError("non-exact strict weight contract")
    if reconstruction["method"] != STRICT_METHODS[method]:
        raise ValueError("strict reconstruction method mismatch")
    if int(reconstruction["weight_bits"]) != 4:
        raise ValueError("strict reconstruction weight-bit mismatch")
    if int(reconstruction["activation_bits"]) != 0:
        raise ValueError("strict reconstruction activation-bit mismatch")
    if reconstruction["activation_policy"] != "evaluation_backend_owned":
        raise ValueError("strict activation ownership mismatch")

    manifest_path = Path(reconstruction["manifest"])
    contract_path = Path(reconstruction["strict_deployment_contract"])
    if not manifest_path.is_file():
        raise ValueError("missing strict reconstruction manifest: %s" % manifest_path)
    if not contract_path.is_file():
        raise ValueError("missing strict deployment contract: %s" % contract_path)
    manifest = read_json(manifest_path)
    if int(manifest["strict"]) != 1:
        raise ValueError("non-strict reconstruction manifest")
    if manifest["method"] != STRICT_METHODS[method]:
        raise ValueError("strict manifest method mismatch")
    if manifest["model"] != model:
        raise ValueError("strict manifest model mismatch")
    if int(manifest["weight_bits"]) != 4:
        raise ValueError("strict manifest weight-bit mismatch")
    if int(manifest["activation_bits"]) != 0 or \
            manifest["activation_manifest"]:
        raise ValueError("strict manifest contains activation reconstruction")
    if list(manifest["targets"]) != list(reconstruction["targets"]):
        raise ValueError("strict manifest target mismatch")
    if Path(manifest["deployment_contract"]).resolve() != \
            contract_path.resolve():
        raise ValueError("strict deployment contract path mismatch")


def _validate_metadata(metadata, method, model, expected_samples,
                       configs, quant_backend, execution, bias_contract):
    if metadata["model"] != model:
        raise ValueError("metadata model mismatch")
    if tuple(metadata["configs"]) != tuple(configs):
        raise ValueError("configuration order mismatch")
    if metadata["quant_backend"] != quant_backend:
        raise ValueError("quantization backend mismatch")
    if metadata["quantization_execution"] != execution:
        raise ValueError("quantization execution mismatch")
    if metadata["hardware_alignment"]["bias_contract"] != bias_contract:
        raise ValueError("bias contract mismatch")
    if int(metadata["evaluation_samples"]) != int(expected_samples):
        raise ValueError("evaluation sample count mismatch")
    if len(metadata["evaluation_indices"]) != int(expected_samples):
        raise ValueError("evaluation index count mismatch")
    if len(metadata["calibration_indices"]) != int(expected_samples):
        raise ValueError("calibration index count mismatch")
    _validate_reconstruction(metadata, method, model)


def _validate_sample_rows(model_root, model, configs, indices):
    rows = read_csv(model_root / "sample_metrics.csv")
    expected = {
        (config, int(sample_index))
        for config in configs for sample_index in indices}
    observed = []
    for row in rows:
        if row["model"] != model:
            raise ValueError("sample metric model mismatch")
        metrics = np.asarray([
            float(row["RMSE"]), float(row["MAE"]),
            float(row["ABS_REL"]),
        ], dtype=np.float64)
        if not np.isfinite(metrics).all():
            raise ValueError("sample metrics contain nonfinite values")
        observed.append((row["config"], int(row["sample_index"])))
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("sample metrics do not match required samples")
    return rows


def _validate_prediction_payloads(model_root, model, configs, indices):
    expected = set(int(index) for index in indices)
    for config in configs:
        paths = sorted(
            (model_root / "predictions" / config).glob("sample_*.npz"))
        observed = []
        for path in paths:
            with np.load(str(path), allow_pickle=False) as payload:
                sample_index = int(payload["sample_index"])
                if str(payload["model"]) != model:
                    raise ValueError("prediction payload model mismatch")
                if str(payload["config"]) != config:
                    raise ValueError("prediction payload config mismatch")
                observed.append(sample_index)
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise ValueError(
                "prediction payloads do not match required samples: %s/%s" %
                (model, config))


def _canonical_semantic_rows(rows):
    fields = ("model", "role", "module", "kind", "bits", "format")
    return tuple(sorted(
        tuple(str(row[field]) for field in fields) for row in rows))


def _validate_fp4_contract(model_root, metadata, model):
    contract = metadata["fp4_validation"]
    if contract["activation_format"] != "scaled_e2m1_rne":
        raise ValueError("E2M1 activation format mismatch")
    if contract["bias_contract"] != "fp32_isolation":
        raise ValueError("FP4 bias contract mismatch")
    if contract["propagation_signals"] != "a8":
        raise ValueError("FP4 propagation contract mismatch")

    semantic_rows = read_csv(model_root / "semantic_a8_boundaries.csv")
    if not semantic_rows:
        raise ValueError("missing semantic A8 boundaries")
    for row in semantic_rows:
        if row["model"] != model or int(row["bits"]) != 8 or \
                row["format"] != "uniform":
            raise ValueError("semantic A8 boundary mismatch")

    manifest_rows = read_csv(model_root / "fp4_manifest.csv")
    expected = {
        "FP4V_W4A4": ("uniform", 4),
        "FP4V_W4E2M1": ("e2m1", 4),
        "FP4V_W4A8": ("uniform", 8),
    }
    ownership = None
    for config in PRIMARY_CONFIGS[1:]:
        current = [row for row in manifest_rows if row["config"] == config]
        if not current:
            raise ValueError("missing FP4 manifest rows for %s" % config)
        expected_format, expected_bits = expected[config]
        if any(row["format"] != expected_format or
               int(row["bits"]) != expected_bits for row in current):
            raise ValueError("FP4 manifest format mismatch for %s" % config)
        current_ownership = {
            (row["module"], row["kind"]) for row in current}
        if ownership is None:
            ownership = current_ownership
        elif current_ownership != ownership:
            raise ValueError("FP4 activation ownership mismatch")
    return _canonical_semantic_rows(semantic_rows)


def _validate_result(model_root, method, model, expected_samples,
                     configs, quant_backend, execution, bias_contract):
    metadata = read_json(model_root / "metadata.json")
    _validate_metadata(
        metadata, method, model, expected_samples,
        configs, quant_backend, execution, bias_contract)
    indices = tuple(int(index) for index in metadata["evaluation_indices"])
    _validate_sample_rows(model_root, model, configs, indices)
    _validate_prediction_payloads(model_root, model, configs, indices)
    return metadata


def validate_result_root(root, expected_samples):
    root = Path(root)
    references = {}
    global_calibration = None
    global_evaluation = None
    provenance_fields = (
        "model_class", "model_module", "source_sha256",
        "checkpoint_sha256", "source_git_commit",
    )
    for model in MODEL_ORDER:
        reference_identity = None
        reference_calibration = None
        reference_evaluation = None
        reference_semantics = None
        for method in METHOD_ORDER:
            primary_root = root / "primary" / method / model
            stress_root = root / "stress" / method / model
            primary = _validate_result(
                primary_root, method, model, expected_samples,
                PRIMARY_CONFIGS, "fp4",
                "float_e2m1_qdq_integer_normalization_reference",
                "fp32_isolation")
            stress = _validate_result(
                stress_root, method, model, expected_samples,
                STRESS_CONFIGS, "hardware", "hardware_aligned_qdq",
                "int32 scale=sx*sw[o]")
            identity = {
                field: primary["model_provenance"][field]
                for field in provenance_fields}
            stress_identity = {
                field: stress["model_provenance"][field]
                for field in provenance_fields}
            if stress_identity != identity:
                raise ValueError("model provenance mismatch for stress %s" % model)
            if stress["calibration_indices"] != primary["calibration_indices"]:
                raise ValueError("calibration indices differ for stress %s" % model)
            if stress["evaluation_indices"] != primary["evaluation_indices"]:
                raise ValueError("evaluation indices differ for stress %s" % model)
            semantics = _validate_fp4_contract(primary_root, primary, model)
            if reference_identity is None:
                reference_identity = identity
                reference_calibration = primary["calibration_indices"]
                reference_evaluation = primary["evaluation_indices"]
                reference_semantics = semantics
            else:
                if identity != reference_identity:
                    raise ValueError("model provenance mismatch for %s" % model)
                if primary["calibration_indices"] != reference_calibration:
                    raise ValueError("calibration indices differ for %s" % model)
                if primary["evaluation_indices"] != reference_evaluation:
                    raise ValueError("evaluation indices differ for %s" % model)
                if semantics != reference_semantics:
                    raise ValueError("semantic A8 boundary mismatch for %s" % model)
        if global_calibration is None:
            global_calibration = reference_calibration
            global_evaluation = reference_evaluation
        else:
            if reference_calibration != global_calibration:
                raise ValueError("global calibration indices differ")
            if reference_evaluation != global_evaluation:
                raise ValueError("global evaluation indices differ")
        references[model] = {
            "model_provenance": reference_identity,
            "calibration_indices": reference_calibration,
            "evaluation_indices": reference_evaluation,
            "semantic_a8_boundaries": reference_semantics,
        }
    return references
