#!/usr/bin/env python3
"""Strict formal evaluation for the selected official NYU SPN models."""

from __future__ import annotations

import argparse
from argparse import Namespace
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SELECTED_METHODS = (
    "fp32",
    "rtn_w8a8",
    "rtn_w4a4",
    "qdrop_w6a6",
    "brecq_w6a6",
    "hawq_mixed_le6",
    "lsqplus_w6a6",
    "lsqplus_w4a4",
    "mixed_task_aware",
    "p3_t3_mixed_ptq",
)

SELECTED_CONFIGURATION_LABELS = (
    "FP32",
    "RTN W8A8",
    "RTN W4A4",
    "QDrop W6A6",
    "BRECQ W6A6",
    "HAWQ mixed<=6",
    "LSQ++ W6A6",
    "LSQ++ W4A4",
    "mixed task-aware QAT",
    "P3/T3 mixed PTQ",
)

METHOD_LABELS = dict(zip(SELECTED_METHODS, SELECTED_CONFIGURATION_LABELS))
PTQ_METHODS = (
    "rtn_w8a8",
    "rtn_w4a4",
    "qdrop_w6a6",
    "brecq_w6a6",
    "p3_t3_mixed_ptq",
)
QAT_METHODS = (
    "hawq_mixed_le6",
    "lsqplus_w6a6",
    "lsqplus_w4a4",
    "mixed_task_aware",
)
EXPECTED_EVALUATION_SAMPLES = 64
NYU_SPARSE_DEPTH_MAX_M = 10.0
PREDICTION_FIELDS = frozenset((
    "format_version",
    "model",
    "method",
    "sample_index",
    "evaluation_identity",
    "artifact_index_sha256",
    "sparse_depth_max_m",
    "rgb",
    "sparse",
    "gt",
    "pred",
    "valid_gt",
    "abs_error",
))
FORMAL_RUN_FIELDS = frozenset((
    "format_version",
    "complete",
    "model",
    "method",
    "artifact_kind",
    "artifact",
    "artifact_sha256",
    "supporting_artifacts",
    "artifact_index",
    "artifact_index_sha256",
    "evaluation_indices",
    "evaluation_identity",
    "prediction_directory",
    "metrics",
    "cost",
    "assignment",
    "cost_basis",
    "diagnostics",
))
ARTIFACT_INDEX_FIELDS = frozenset((
    "format_version",
    "model",
    "evaluation_indices",
    "evaluation_identity",
    "ptq_matrix",
    "methods",
    "preparation",
))
ARTIFACT_REFERENCE_FIELDS = frozenset(("path", "sha256"))
METHOD_ARTIFACT_FIELDS = frozenset((
    "method",
    "artifact_kind",
    "artifact",
    "artifact_sha256",
    "supporting_artifacts",
))
SUPPORTING_ARTIFACT_FIELDS = frozenset(("name", "path", "sha256"))
PREPARATION_FIELDS = frozenset((
    "fold_conv_bn",
    "fold_max_error",
    "joint_clip_factors",
    "joint_search_rounds",
    "joint_cache_sample_limit",
    "joint_cache_byte_limit",
))
PTQ_MATRIX_FIELDS = frozenset((
    "format_version",
    "model",
    "methods",
    "calibration_identity",
    "evaluation_identity",
    "hard_deployment_manifests",
))


@dataclass(frozen=True)
class PredictionAggregation:
    squared_error_sum: float
    valid_pixel_count: int
    pooled_rmse: float
    mean_sample_rmse: float
    pooled_mae: float
    pooled_abs_rel: float
    pooled_irmse: float
    nonpositive_pixel_count: int
    nonpositive_ratio: float
    sample_metrics: Tuple[dict, ...]


@dataclass(frozen=True)
class FormalAggregation:
    aggregate_metrics: Tuple[dict, ...]
    sample_metrics: Tuple[dict, ...]
    relative_fp_loss: Tuple[dict, ...]


@dataclass(frozen=True)
class MethodArtifact:
    method: str
    artifact_kind: str
    artifact: Path
    artifact_sha256: str
    supporting_artifacts: Tuple[Tuple[str, Path, str], ...]


@dataclass(frozen=True)
class FormalArtifactIndex:
    source: Path
    fingerprint: str
    model: str
    evaluation_indices: Tuple[int, ...]
    evaluation_identity: str
    ptq_matrix: Path
    methods: Mapping[str, MethodArtifact]
    preparation: Mapping[str, object]


@dataclass(frozen=True)
class IndexedCostContracts:
    artifact_index_sha256: str
    methods: Mapping[str, dict]


class FormalDeployment(object):
    """One fresh FP32 or materialized hard-deployment evaluation context."""

    def __init__(self, *, runtime, model, contract, artifact,
                 artifact_kind, assignment, cost_basis,
                 diagnostics_sources, propagation, closer) -> None:
        self.runtime = runtime
        self.model = model
        self.contract = contract
        self.artifact = Path(artifact)
        self.artifact_kind = str(artifact_kind)
        self.assignment = assignment
        self.cost_basis = cost_basis
        self.diagnostics_sources = tuple(diagnostics_sources)
        self.propagation = propagation
        self._closer = closer
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        self._closer()
        self.closed = True


def _output_tensors(value) -> Tuple[torch.Tensor, ...]:
    if torch.is_tensor(value):
        return (value,)
    if isinstance(value, Mapping):
        tensors = []
        for key in sorted(value):
            tensors.extend(_output_tensors(value[key]))
        return tuple(tensors)
    if isinstance(value, (tuple, list)):
        tensors = []
        for current in value:
            tensors.extend(_output_tensors(current))
        return tuple(tensors)
    return ()


class ContractBlockOutputCapture(object):
    """Capture generic contract-block outputs without model-specific rules."""

    def __init__(self, model, contract) -> None:
        modules = dict(model.named_modules())
        self._owners = tuple(block.name for block in contract.blocks)
        missing = tuple(owner for owner in self._owners
                        if owner not in modules)
        if missing:
            raise ValueError(
                "contract block output modules are missing: %s" %
                (missing,))
        self._values = {}
        self._handles = tuple(
            modules[owner].register_forward_hook(self._hook(owner))
            for owner in self._owners)

    def _hook(self, owner):
        def capture(module, inputs, output):
            del module, inputs
            tensors = _output_tensors(output)
            if not tensors:
                raise TypeError(
                    "contract block output contains no tensors: %s" % owner)
            if owner in self._values:
                raise RuntimeError(
                    "contract block executed more than once: %s" % owner)
            if any(not bool(torch.isfinite(tensor).all().item())
                   for tensor in tensors if tensor.is_floating_point()):
                raise FloatingPointError(
                    "contract block output is non-finite: %s" % owner)
            self._values[owner] = tuple(
                tensor.detach().clone() for tensor in tensors)
        return capture

    def begin(self) -> None:
        self._values = {}

    def values(self):
        if set(self._values) != set(self._owners) or \
                len(self._values) != len(self._owners):
            raise RuntimeError("contract block capture coverage differs")
        return dict((owner, self._values[owner]) for owner in self._owners)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = ()


def paired_block_diagnostic_rows(
        reference, candidate, model: str, method: str,
        sample_index: int) -> Tuple[dict, ...]:
    if tuple(reference) != tuple(candidate):
        raise ValueError("paired contract block coverage differs")
    rows = []
    for owner in reference:
        left = tuple(reference[owner])
        right = tuple(candidate[owner])
        if not left or len(left) != len(right):
            raise ValueError(
                "paired contract block tensor coverage differs: %s" % owner)
        flattened_left = torch.cat(tuple(
            tensor.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
            for tensor in left))
        flattened_right = torch.cat(tuple(
            tensor.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
            for tensor in right))
        rows.append(paired_tensor_diagnostic_row(
            reference=flattened_left,
            candidate=flattened_right,
            model=model,
            method=method,
            sample_index=sample_index,
            iteration=-1,
            owner=owner,
            owner_kind="quantization_block",
        ))
    return tuple(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _validated_sha256(value, family: str) -> str:
    fingerprint = str(value)
    if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef"
            for character in fingerprint):
        raise ValueError("%s fingerprint is invalid" % family)
    return fingerprint


def ordered_evaluation_identity(indices: Sequence[int]) -> str:
    return _ordered_sample_identity("val", indices)


def _ordered_sample_identity(split: str, indices: Sequence[int]) -> str:
    values = tuple(int(index) for index in indices)
    if not values or len(values) != len(set(values)) or any(
            index < 0 for index in values):
        raise ValueError("evaluation indices must be unique and nonnegative")
    encoded = json.dumps(
        [[str(split), index] for index in values],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_artifact_reference(payload, family):
    if set(payload) != ARTIFACT_REFERENCE_FIELDS:
        raise KeyError("%s artifact reference fields changed" % family)
    path = Path(payload["path"])
    if not path.is_absolute():
        raise ValueError("%s artifact path must be absolute" % family)
    if not path.is_file():
        raise FileNotFoundError("%s artifact is missing: %s" % (family, path))
    expected = _validated_sha256(payload["sha256"], family)
    if file_sha256(path) != expected:
        raise RuntimeError("%s artifact fingerprint changed" % family)
    return path, expected


def load_formal_artifact_index(
        path: Path, expected_model: str,
        expected_indices: Sequence[int]) -> FormalArtifactIndex:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError("formal artifact index is missing: %s" % source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if set(payload) != ARTIFACT_INDEX_FIELDS:
        raise KeyError("formal artifact index fields changed")
    if isinstance(payload["format_version"], bool) or not isinstance(
            payload["format_version"], int) or \
            int(payload["format_version"]) != 1:
        raise ValueError("formal artifact index version changed")
    model = str(payload["model"])
    if model != str(expected_model):
        raise ValueError("formal artifact index model changed")
    indices = tuple(int(index) for index in payload["evaluation_indices"])
    expected = tuple(int(index) for index in expected_indices)
    if indices != expected or len(indices) != EXPECTED_EVALUATION_SAMPLES or \
            len(indices) != len(set(indices)):
        raise ValueError("formal artifact evaluation identities changed")
    identity = ordered_evaluation_identity(indices)
    if str(payload["evaluation_identity"]) != identity:
        raise ValueError("formal artifact evaluation identity changed")
    ptq_matrix, ptq_matrix_sha256 = _validated_artifact_reference(
        payload["ptq_matrix"], "selected PTQ matrix")
    del ptq_matrix_sha256
    ptq_payload = json.loads(ptq_matrix.read_text(encoding="utf-8"))
    if set(ptq_payload) != PTQ_MATRIX_FIELDS:
        raise KeyError("selected PTQ matrix fields changed")
    if int(ptq_payload["format_version"]) != 1 or \
            str(ptq_payload["model"]) != model:
        raise ValueError("selected PTQ matrix identity changed")
    if tuple(str(method) for method in ptq_payload["methods"]) != PTQ_METHODS:
        raise ValueError("selected PTQ method order changed")
    if str(ptq_payload["evaluation_identity"]) != identity:
        raise ValueError("selected PTQ evaluation identity changed")
    calibration_identity = str(ptq_payload["calibration_identity"])
    if len(calibration_identity) != 64 or any(
            character not in "0123456789abcdef"
            for character in calibration_identity):
        raise ValueError("selected PTQ calibration identity is invalid")
    manifest_paths = ptq_payload["hard_deployment_manifests"]
    if set(manifest_paths) != set(PTQ_METHODS) or \
            len(manifest_paths) != len(PTQ_METHODS):
        raise ValueError("selected PTQ manifest method set changed")
    rows = tuple(payload["methods"])
    actual_methods = tuple(str(row["method"]) for row in rows)
    if actual_methods != SELECTED_METHODS:
        raise ValueError("formal artifact method order changed")
    expected_kinds = dict((method, "strict_ptq_manifest")
                          for method in PTQ_METHODS)
    expected_kinds.update(dict((method, "terminal_qat_checkpoint")
                               for method in QAT_METHODS))
    expected_kinds["fp32"] = "official_checkpoint"
    expected_support = {
        "fp32": (),
        "rtn_w8a8": (),
        "rtn_w4a4": (),
        "qdrop_w6a6": (),
        "brecq_w6a6": (),
        "hawq_mixed_le6": (
            "hawq_assignment", "hawq_trace_artifact"),
        "lsqplus_w6a6": (),
        "lsqplus_w4a4": (),
        "mixed_task_aware": ("p3_t3_assignment",),
        "p3_t3_mixed_ptq": ("p3_t3_assignment",),
    }
    methods = {}
    for row in rows:
        if set(row) != METHOD_ARTIFACT_FIELDS:
            raise KeyError("formal method artifact fields changed")
        method = str(row["method"])
        kind = str(row["artifact_kind"])
        if kind != expected_kinds[method]:
            raise ValueError(
                "formal artifact kind changed for %s" % method)
        artifact, artifact_sha256 = _validated_artifact_reference({
            "path": row["artifact"],
            "sha256": row["artifact_sha256"],
        }, "%s primary" % method)
        supporting = []
        names = []
        for support in row["supporting_artifacts"]:
            if set(support) != SUPPORTING_ARTIFACT_FIELDS:
                raise KeyError(
                    "formal supporting artifact fields changed")
            name = str(support["name"])
            support_path, support_sha256 = _validated_artifact_reference({
                "path": support["path"],
                "sha256": support["sha256"],
            }, "%s %s" % (method, name))
            names.append(name)
            supporting.append((name, support_path, support_sha256))
        if len(names) != len(set(names)):
            raise ValueError(
                "formal supporting artifacts contain duplicate names")
        if tuple(names) != expected_support[method]:
            raise ValueError(
                "formal supporting artifact set changed for %s" % method)
        methods[method] = MethodArtifact(
            method=method,
            artifact_kind=kind,
            artifact=artifact,
            artifact_sha256=artifact_sha256,
            supporting_artifacts=tuple(supporting),
        )
    for method in PTQ_METHODS:
        if Path(manifest_paths[method]).resolve() != \
                methods[method].artifact.resolve():
            raise ValueError(
                "selected PTQ manifest path differs for %s" % method)
    mixed_p3 = methods["mixed_task_aware"].supporting_artifacts[0][1]
    ptq_p3 = methods["p3_t3_mixed_ptq"].supporting_artifacts[0][1]
    if mixed_p3.resolve() != ptq_p3.resolve():
        raise ValueError(
            "mixed QAT and P3/T3 PTQ assignments differ")
    preparation = payload["preparation"]
    if set(preparation) != PREPARATION_FIELDS:
        raise KeyError("formal graph preparation fields changed")
    fold = preparation["fold_conv_bn"]
    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in (0, 1):
        raise TypeError("formal fold setting must be integer zero or one")
    fold_error = float(preparation["fold_max_error"])
    clip_factors = tuple(float(value)
                         for value in preparation["joint_clip_factors"])
    positive_integers = (
        int(preparation["joint_search_rounds"]),
        int(preparation["joint_cache_sample_limit"]),
        int(preparation["joint_cache_byte_limit"]),
    )
    if not math.isfinite(fold_error) or fold_error < 0.0 or \
            not clip_factors or any(
                not math.isfinite(value) or value <= 0.0
                for value in clip_factors) or any(
                    value <= 0 for value in positive_integers):
        raise ValueError("formal graph preparation settings are invalid")
    normalized_preparation = {
        "fold_conv_bn": fold,
        "fold_max_error": fold_error,
        "joint_clip_factors": clip_factors,
        "joint_search_rounds": positive_integers[0],
        "joint_cache_sample_limit": positive_integers[1],
        "joint_cache_byte_limit": positive_integers[2],
    }
    return FormalArtifactIndex(
        source=source.resolve(),
        fingerprint=file_sha256(source),
        model=model,
        evaluation_indices=indices,
        evaluation_identity=identity,
        ptq_matrix=ptq_matrix,
        methods=methods,
        preparation=normalized_preparation,
    )


def _model_config(selected, model_name):
    rows = tuple(model for model in selected.models
                 if model.model == str(model_name))
    if len(rows) != 1:
        raise ValueError("selected formal model entry is not unique")
    return rows[0]


def _supporting_path(entry: MethodArtifact, name: str) -> Path:
    matches = tuple(path for current, path, sha256
                    in entry.supporting_artifacts if current == name)
    if len(matches) != 1:
        raise ValueError(
            "%s requires one %s supporting artifact" %
            (entry.method, name))
    return matches[0]


def _load_tensor_payload(path: Path):
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _validate_hard_weight_payload(
        payload, method: str, model_name: str, module_names) -> dict:
    required = {
        "format_version", "strict", "method", "model",
        "module_names", "state_dict",
    }
    optional = {"weight_bits", "activation_bits"}
    if set(payload) not in (required, required | optional):
        raise KeyError("hard weight checkpoint fields changed")
    if int(payload["format_version"]) != 1 or int(payload["strict"]) != 1:
        raise ValueError("hard weight checkpoint is not strict version 1")
    if str(payload["method"]) != method or \
            str(payload["model"]) != model_name:
        raise ValueError("hard weight checkpoint identity changed")
    if tuple(payload["module_names"]) != tuple(module_names):
        raise ValueError("hard weight checkpoint ownership changed")
    state = payload["state_dict"]
    if not isinstance(state, dict) or not state or any(
            not isinstance(name, str) or not torch.is_tensor(value)
            for name, value in state.items()):
        raise TypeError("hard weight checkpoint must contain named tensors")
    if any("parametrizations.weight" in name for name in state):
        raise ValueError("hard weight checkpoint contains soft weights")
    if any(value.is_floating_point() and not bool(
            torch.isfinite(value).all().item()) for value in state.values()):
        raise FloatingPointError("hard weight checkpoint is non-finite")
    return state


def _state_equal(left, right) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[name].detach().cpu(), right[name].detach().cpu())
        for name in left)


def _assignment_mapping(assignment):
    return {
        "weight_bits": tuple(assignment.weight_bits),
        "activation_bits": tuple(assignment.activation_bits),
    }


def _validated_p3_inputs(
        index, selected, model_config, contract, trusted_p3_t3):
    from scripts import train_nyu_selected_qat as qat_runner

    path = _supporting_path(
        index.methods["p3_t3_mixed_ptq"], "p3_t3_assignment")
    assignment, costs, evidence = qat_runner.load_p3_t3_qat_assignment(
        path,
        contract,
        selected.method_hyperparameters["p3_t3_mixed_ptq"],
        model_config.checkpoint,
        model_config.evaluation_indices,
        trusted_p3_t3[0],
        trusted_p3_t3[1],
        trusted_p3_t3[2],
    )
    del evidence
    return path, assignment, costs


def _calibration_indices(model_config):
    payload = json.loads(
        Path(model_config.calibration_metadata).read_text(encoding="utf-8"))
    calibration = tuple(
        int(value) for value in payload["calibration_indices"])
    evaluation = tuple(
        int(value) for value in payload["evaluation_indices"])
    if len(calibration) != 128 or len(calibration) != len(set(calibration)) or \
            any(value < 0 for value in calibration):
        raise ValueError("formal calibration requires 128 unique identities")
    if evaluation != tuple(model_config.evaluation_indices):
        raise ValueError("formal evaluation identities changed in metadata")
    return calibration


def _validate_ptq_manifest(
        index, method, contract, model_config):
    from scripts.run_nyu_selected_ptq import (
        validate_hard_deployment_manifest,
    )

    matrix = json.loads(index.ptq_matrix.read_text(encoding="utf-8"))
    expected_calibration_identity = _ordered_sample_identity(
        "train", _calibration_indices(model_config))
    if str(matrix["calibration_identity"]) != \
            expected_calibration_identity:
        raise ValueError("selected PTQ calibration identity changed")
    return validate_hard_deployment_manifest(
        index.methods[method].artifact,
        method,
        contract,
        expected_calibration_identity,
        index.evaluation_identity,
    )


def _hard_deployment_settings(index, model_config, selected):
    from scripts.run_nyu_model_p3t3_search import HardDeploymentSettings

    precision = selected.method_hyperparameters["p3_t3_mixed_ptq"]
    preparation = index.preparation
    return HardDeploymentSettings(
        device=model_config.device,
        calibration_metadata=model_config.calibration_metadata,
        calibration_count=model_config.calibration_count,
        evaluation_indices=model_config.evaluation_indices,
        base_weight_bits=int(precision["base_weight_bits"]),
        base_activation_bits=int(precision["base_activation_bits"]),
        promotion_weight_bits=int(precision["promotion_weight_bits"]),
        promotion_activation_bits=int(precision[
            "promotion_activation_bits"]),
        fold_conv_bn=bool(preparation["fold_conv_bn"]),
        fold_max_error=float(preparation["fold_max_error"]),
        joint_clip_factors=tuple(preparation["joint_clip_factors"]),
        joint_search_rounds=int(preparation["joint_search_rounds"]),
        joint_cache_sample_limit=int(
            preparation["joint_cache_sample_limit"]),
        joint_cache_byte_limit=int(
            preparation["joint_cache_byte_limit"]),
    )


def _prepare_fp32_deployment(
        index, selected, model_config, runtime, model, contract,
        trusted_p3_t3):
    entry = index.methods["fp32"]
    if entry.artifact.resolve() != model_config.checkpoint.resolve():
        raise ValueError("FP32 artifact differs from configured checkpoint")
    p3_path, p3_assignment, costs = _validated_p3_inputs(
        index, selected, model_config, contract, trusted_p3_t3)
    del p3_path, p3_assignment
    assignment = {
        "weight_bits": tuple(
            (name, 32) for name in contract.weight_modules),
        "activation_bits": tuple(
            (owner, 32) for block in contract.blocks
            for owner in block.activation_owners),
    }

    def close():
        runtime.close()

    return FormalDeployment(
        runtime=runtime,
        model=model,
        contract=contract,
        artifact=entry.artifact,
        artifact_kind=entry.artifact_kind,
        assignment=assignment,
        cost_basis={
            "weight_macs": costs.weight_macs,
            "activation_elements": costs.activation_elements,
        },
        diagnostics_sources=(),
        propagation=None,
        closer=close,
    )


def _prepare_rtn_deployment(
        method, index, selected, model_config, runtime, model, contract,
        trusted_p3_t3):
    from scripts.run_nyu_model_p3t3_search import (
        HardDeploymentP3T3Evaluator,
    )
    from spn_quant.mixed_precision import AllocationRegistry, P3T3Candidate
    from spn_quant.mixed_precision import BitAssignment

    manifest = _validate_ptq_manifest(
        index, method, contract, model_config)
    deployment = _load_tensor_payload(Path(manifest["deployment_contract"]))
    required = {
        "format_version", "strict", "method", "model", "weight_bits",
        "activation_bits", "activation_manifest", "joint_manifest",
        "graph_contract", "protected_modules",
    }
    if set(deployment) != required or int(deployment["format_version"]) != 1 \
            or int(deployment["strict"]) != 1:
        raise ValueError("RTN deployment contract fields changed")
    if str(deployment["method"]) != method or \
            str(deployment["model"]) != model_config.model:
        raise ValueError("RTN deployment contract identity changed")
    if tuple(deployment["protected_modules"]) != contract.protected_modules:
        raise ValueError("RTN deployment protected modules changed")
    assignment = BitAssignment(
        weight_bits=tuple(
            (str(row[0]), int(row[1]))
            for row in deployment["weight_bits"]),
        activation_bits=tuple(
            ((str(row[0][0]), str(row[0][1])), int(row[1]))
            for row in deployment["activation_bits"]),
        model_name=model_config.model,
    )
    registry = AllocationRegistry(
        weights_by_block=dict(
            (block.name, block.weight_modules) for block in contract.blocks),
        activations_by_block=dict(
            (block.name, block.activation_owners) for block in contract.blocks),
        blocks=contract.block_names,
        model_name=contract.model_name,
    )
    candidate = P3T3Candidate(
        name=method,
        stage="formal_hard_deployment",
        prefix=(),
        tail=(),
        promoted_blocks=(),
        assignment=assignment,
    )
    evaluator = HardDeploymentP3T3Evaluator(
        runtime,
        model,
        contract,
        registry,
        _hard_deployment_settings(index, model_config, selected),
    )
    ready = False
    try:
        evaluator.configure_hard_candidate(candidate)
        if evaluator.graph_preparation != deployment["graph_contract"]:
            raise ValueError("RTN formal graph preparation changed")
        if tuple(evaluator.instrumentor.manifest()) != tuple(
                deployment["activation_manifest"]):
            raise ValueError("RTN formal activation contract changed")
        hard_payload = _load_tensor_payload(Path(manifest["hard_weights"]))
        hard_state = _validate_hard_weight_payload(
            hard_payload, method, model_config.model,
            contract.weight_modules)
        current = dict((name, value.detach().cpu())
                       for name, value in model.state_dict().items())
        if not _state_equal(current, hard_state):
            raise RuntimeError("RTN rematerialized hard state differs")
        model.load_state_dict(hard_state, strict=True)
        p3_path, p3_assignment, costs = _validated_p3_inputs(
            index, selected, model_config, contract, trusted_p3_t3)
        del p3_path, p3_assignment
        ready = True
    finally:
        if not ready:
            evaluator.close()
            runtime.close()

    def close():
        evaluator.close()
        runtime.close()

    diagnostics_sources = (evaluator.instrumentor,)
    if evaluator.joint_adapter is not None:
        diagnostics_sources += \
            evaluator.joint_adapter.qdrop_diagnostic_sources()
    return FormalDeployment(
        runtime=runtime,
        model=model,
        contract=contract,
        artifact=index.methods[method].artifact,
        artifact_kind=index.methods[method].artifact_kind,
        assignment=_assignment_mapping(assignment),
        cost_basis={
            "weight_macs": costs.weight_macs,
            "activation_elements": costs.activation_elements,
        },
        diagnostics_sources=diagnostics_sources,
        propagation=evaluator.propagation_adapter,
        closer=close,
    )


def _prepare_qdrop_deployment(
        method, index, selected, model_config, runtime, model, contract,
        trusted_p3_t3):
    from scripts.hardware_aligned_quantization import prepare_hardware_model
    from scripts import run_nyu_qdrop_reconstruction as qdrop_runner
    from scripts.run_nyu_rtn_quantization import (
        batch_from_sample,
        seeded_sample,
    )
    from spn_quant.deployment_contract import (
        file_sha256 as deployment_file_sha256,
        validate_graph_preparation,
    )
    from spn_quant.propagation import install_propagation_adapter
    from spn_quant.qdrop_contract import (
        QDropContractInstrumentor,
        load_qdrop_contract,
    )
    from scripts.nyu_quantization_analysis import classify_module

    manifest = _validate_ptq_manifest(
        index, method, contract, model_config)
    qdrop_contract = load_qdrop_contract(Path(manifest["deployment_contract"]))
    if deployment_file_sha256(model_config.checkpoint) != str(
            qdrop_contract["source_checkpoint_sha256"]):
        raise RuntimeError("QDrop source checkpoint fingerprint changed")
    trainset = runtime.build_dataset("train")
    calibration_indices = _calibration_indices(model_config)
    seed = int(runtime.saved_args.seed)
    first = batch_from_sample(seeded_sample(
        trainset, calibration_indices[0], seed))
    model_args, target = runtime.model_input(first, runtime.device)
    del target
    preparation = prepare_hardware_model(
        model,
        model_args,
        fold=bool(qdrop_contract["graph_contract"]["fold"]),
    )
    if bool(qdrop_contract["graph_contract"]["fold"]) != bool(
            index.preparation["fold_conv_bn"]):
        raise ValueError("QDrop formal fold mode differs from artifact index")
    if float(preparation["primary_max_abs_error"]) > float(
            index.preparation["fold_max_error"]):
        raise RuntimeError("QDrop formal Conv-BN fold exceeds threshold")
    validate_graph_preparation(
        preparation, qdrop_contract["graph_contract"])
    precision = Namespace(
        weight_bits=int(qdrop_contract["weight_bits"]),
        activation_bits=int(qdrop_contract["activation_bits"]),
    )
    joint = qdrop_runner._joint_adapter(
        model_config.model, model, precision)
    base_instrumentor = qdrop_runner._instrumentor(
        runtime.saved_args, model, preparation, joint)
    group_fn = lambda name, module: classify_module(
        model_config.model, name, module)
    instrumentor = QDropContractInstrumentor(
        base_instrumentor, qdrop_contract, group_fn=group_fn)
    if joint is not None:
        instrumentor.bind_joint_adapter(joint)
    propagation = install_propagation_adapter(model_config.model, model)
    ready = False
    try:
        instrumentor.observe()
        propagation.observe()
        model.eval()
        with torch.no_grad():
            for sample_index in calibration_indices:
                sample = batch_from_sample(seeded_sample(
                    trainset, sample_index, seed))
                inputs, target = runtime.model_input(sample, runtime.device)
                del target
                model(*inputs)
        instrumentor.freeze()
        propagation.freeze()
        groups = set(base_instrumentor.groups.values())
        instrumentor.configure(
            w_bits=precision.weight_bits,
            a_bits=precision.activation_bits,
            enabled_groups=groups,
            activation_overrides={},
            weight_bit_overrides={},
            activation_bit_overrides={},
            external_output_ownership=True,
            quantize_bias=False,
        )
        from scripts.train_nyu_selected_qat import _propagation_config
        propagation.configure(_propagation_config())
        hard_payload = _load_tensor_payload(Path(manifest["hard_weights"]))
        hard_state = _validate_hard_weight_payload(
            hard_payload, method, model_config.model,
            contract.weight_modules)
        current = dict((name, value.detach().cpu())
                       for name, value in model.state_dict().items())
        if not _state_equal(current, hard_state):
            raise RuntimeError("QDrop rematerialized hard state differs")
        model.load_state_dict(hard_state, strict=True)
        p3_path, p3_assignment, costs = _validated_p3_inputs(
            index, selected, model_config, contract, trusted_p3_t3)
        del p3_path, p3_assignment
        ready = True
    finally:
        if not ready:
            propagation.close()
            instrumentor.close()
            runtime.close()

    assignment = {
        "weight_bits": tuple(
            (name, precision.weight_bits) for name in contract.weight_modules),
        "activation_bits": tuple(
            (owner, precision.activation_bits) for block in contract.blocks
            for owner in block.activation_owners),
    }

    def close():
        propagation.close()
        instrumentor.close()
        runtime.close()

    return FormalDeployment(
        runtime=runtime,
        model=model,
        contract=contract,
        artifact=index.methods[method].artifact,
        artifact_kind=index.methods[method].artifact_kind,
        assignment=assignment,
        cost_basis={
            "weight_macs": costs.weight_macs,
            "activation_elements": costs.activation_elements,
        },
        diagnostics_sources=(instrumentor,),
        propagation=propagation,
        closer=close,
    )


def _qat_args(method, entry, config, launch_spec):
    support = dict((name, path)
                   for name, path, sha256 in entry.supporting_artifacts)
    return Namespace(
        config=Path(config),
        launch_spec=Path(launch_spec),
        method=method,
        hawq_assignment=support["hawq_assignment"]
            if method == "hawq_mixed_le6" else None,
        hawq_trace_artifact=support["hawq_trace_artifact"]
            if method == "hawq_mixed_le6" else None,
        p3_t3_assignment=support["p3_t3_assignment"]
            if method == "mixed_task_aware" else None,
    )


def _prepare_qat_deployment(
        method, index, selected, model_config, config, launch_spec,
        trusted_p3_t3):
    from scripts import train_nyu_selected_qat as qat_runner

    entry = index.methods[method]
    payload = _load_tensor_payload(entry.artifact)
    qat_runner.validate_checkpoint_payload(payload)
    if str(payload["model_name"]) != model_config.model or \
            str(payload["method"]) != method:
        raise ValueError("terminal QAT checkpoint identity changed")
    if payload["run_state"] != {
            "terminal": True,
            "completed": True,
            "reason": payload["convergence"]["reason"]}:
        raise ValueError("formal QAT checkpoint is not completed")
    training = dict(payload["training_config"])
    prepared = qat_runner.prepare_selected_qat(
        _qat_args(method, entry, config, launch_spec),
        selected, model_config, training)
    materialized = None
    ready = False
    try:
        if payload["contract_manifest"] != qat_runner._contract_manifest(
                prepared.contract):
            raise ValueError("terminal QAT model contract changed")
        if payload["assignment"] != qat_runner._assignment_payload(
                prepared.assignment):
            raise ValueError("terminal QAT assignment changed")
        if tuple(payload["calibration_indices"]) != \
                prepared.calibration_indices:
            raise ValueError("terminal QAT calibration identities changed")
        prepared.controller.load_canonical_model_state_dict(
            payload["model_state"])
        prepared.controller.load_method_state_dict(payload["method_state"])
        qat_runner.validate_hard_deployment_against_controller(
            payload["hard_deployment_validation"], prepared.controller)
        hard_state = prepared.controller.hard_model_state_dict()
        weight_diagnostic_qparams = \
            prepared.controller.deployment_weight_diagnostic_qparams()
        qparams = prepared.controller.deployment_qparams()
        materialized = qat_runner.build_materialized_deployment_context(
            prepared,
            model_config,
            training,
            hard_state,
            payload["method_state"],
            qparams,
            torch.device(model_config.device),
        )
        materialized.controller.configure_weight_code_statistics(
            weight_diagnostic_qparams)
        _, _, costs = _validated_p3_inputs(
            index, selected, model_config, prepared.contract,
            trusted_p3_t3)
        assignment = _assignment_mapping(prepared.assignment)
        materialized_contract = prepared.contract
        ready = True
    finally:
        prepared.close()
        if not ready and materialized is not None:
            materialized.close()
    if materialized is None:
        raise RuntimeError("terminal QAT deployment was not materialized")

    def close():
        materialized.close()

    return FormalDeployment(
        runtime=materialized.student_runtime,
        model=materialized.model,
        contract=materialized_contract,
        artifact=entry.artifact,
        artifact_kind=entry.artifact_kind,
        assignment=assignment,
        cost_basis={
            "weight_macs": costs.weight_macs,
            "activation_elements": costs.activation_elements,
        },
        diagnostics_sources=(materialized.controller,),
        propagation=materialized.propagation,
        closer=close,
    )


def prepare_formal_deployment(
        method: str, index: FormalArtifactIndex,
        selected, model_config, config: Path,
        launch_spec: Path) -> FormalDeployment:
    """Prepare only a fresh FP32 model or a strict hard deployment."""
    method = str(method)
    if method not in SELECTED_METHODS:
        raise ValueError("unsupported selected formal method: %s" % method)
    from scripts.nyu_model_runtime import NYUModelRuntime
    from scripts import train_nyu_selected_qat as qat_runner
    from spn_quant.model_contracts import build_model_quantization_contract

    trusted_p3_t3 = qat_runner.load_trusted_p3_t3_contract(
        config, launch_spec, model_config.model)

    if method in QAT_METHODS:
        return _prepare_qat_deployment(
            method, index, selected, model_config, config, launch_spec,
            trusted_p3_t3)
    runtime = NYUModelRuntime.from_config(model_config)
    ready = False
    try:
        model = runtime.build_model(runtime.device)
        contract = build_model_quantization_contract(
            model_config.model, model)
        if method == "fp32":
            deployment = _prepare_fp32_deployment(
                index, selected, model_config, runtime, model, contract,
                trusted_p3_t3)
        elif method in ("rtn_w8a8", "rtn_w4a4", "p3_t3_mixed_ptq"):
            deployment = _prepare_rtn_deployment(
                method, index, selected, model_config,
                runtime, model, contract, trusted_p3_t3)
        else:
            deployment = _prepare_qdrop_deployment(
                method, index, selected, model_config,
                runtime, model, contract, trusted_p3_t3)
        ready = True
        return deployment
    finally:
        if not ready and not runtime.closed:
            runtime.close()


def _validated_arrays(record):
    sample_index = int(record["sample_index"])
    if sample_index < 0:
        raise ValueError("prediction sample index must be nonnegative")
    gt = np.asarray(record["gt"], dtype=np.float32)
    pred = np.asarray(record["pred"], dtype=np.float32)
    if gt.ndim != 2 or pred.shape != gt.shape:
        raise ValueError("prediction and GT must be matching depth maps")
    if not bool(np.isfinite(gt).all()):
        raise FloatingPointError("ground truth contains non-finite values")
    if not bool(np.isfinite(pred).all()):
        raise FloatingPointError("prediction contains non-finite values")
    valid = gt > np.float32(1e-4)
    if not bool(np.any(valid)):
        raise RuntimeError("evaluation sample has no valid GT")
    return sample_index, gt, pred, valid


def prediction_sample_metrics(record) -> dict:
    sample_index, gt, pred, valid = _validated_arrays(record)
    target = gt[valid].astype(np.float64)
    values = pred[valid].astype(np.float64)
    difference = values - target
    absolute = np.abs(difference)
    inverse = 1.0 / np.maximum(values, 1e-6) - 1.0 / target
    pixels = int(valid.sum())
    nonpositive = int(np.count_nonzero(values <= 0.0))
    squared_error_sum = float(np.square(difference).sum())
    absolute_error_sum = float(absolute.sum())
    abs_rel_sum = float((absolute / target).sum())
    inverse_squared_error_sum = float(np.square(inverse).sum())
    return {
        "sample_index": sample_index,
        "valid_pixel_count": pixels,
        "squared_error_sum": squared_error_sum,
        "absolute_error_sum": absolute_error_sum,
        "abs_rel_sum": abs_rel_sum,
        "inverse_squared_error_sum": inverse_squared_error_sum,
        "nonpositive_pixel_count": nonpositive,
        "sample_rmse": math.sqrt(squared_error_sum / float(pixels)),
        "sample_mae": absolute_error_sum / float(pixels),
        "sample_abs_rel": abs_rel_sum / float(pixels),
        "sample_irmse": math.sqrt(
            inverse_squared_error_sum / float(pixels)),
        "sample_nonpositive_ratio": nonpositive / float(pixels),
    }


def aggregate_predictions(records) -> PredictionAggregation:
    records = tuple(records)
    if not records:
        raise ValueError("prediction aggregation requires records")
    rows = tuple(prediction_sample_metrics(record) for record in records)
    indices = tuple(int(row["sample_index"]) for row in rows)
    if len(indices) != len(set(indices)):
        raise ValueError("prediction aggregation contains duplicate identities")
    pixels = sum(int(row["valid_pixel_count"]) for row in rows)
    squared_error = math.fsum(
        float(row["squared_error_sum"]) for row in rows)
    absolute_error = math.fsum(
        float(row["absolute_error_sum"]) for row in rows)
    abs_rel = math.fsum(float(row["abs_rel_sum"]) for row in rows)
    inverse_squared_error = math.fsum(
        float(row["inverse_squared_error_sum"]) for row in rows)
    nonpositive = sum(
        int(row["nonpositive_pixel_count"]) for row in rows)
    return PredictionAggregation(
        squared_error_sum=squared_error,
        valid_pixel_count=pixels,
        pooled_rmse=math.sqrt(squared_error / float(pixels)),
        mean_sample_rmse=math.fsum(
            float(row["sample_rmse"]) for row in rows) / float(len(rows)),
        pooled_mae=absolute_error / float(pixels),
        pooled_abs_rel=abs_rel / float(pixels),
        pooled_irmse=math.sqrt(inverse_squared_error / float(pixels)),
        nonpositive_pixel_count=nonpositive,
        nonpositive_ratio=nonpositive / float(pixels),
        sample_metrics=rows,
    )


def prediction_path(root: Path, method: str, sample_index: int) -> Path:
    if method not in SELECTED_METHODS:
        raise ValueError("unsupported selected evaluation method: %s" % method)
    return Path(root) / "predictions" / method / (
        "sample_%05d.npz" % int(sample_index))


def _validated_display_inputs(rgb, sparse, target_shape,
                              sparse_depth_max_m: float):
    rgb_array = np.asarray(rgb, dtype=np.float32)
    sparse_array = np.asarray(sparse, dtype=np.float32)
    if rgb_array.shape != tuple(target_shape) + (3,) or \
            sparse_array.shape != tuple(target_shape):
        raise ValueError("RGB, sparse depth, and dense depth shapes differ")
    if not bool(np.isfinite(rgb_array).all()):
        raise FloatingPointError("RGB must be finite")
    if not bool(np.isfinite(sparse_array).all()):
        raise FloatingPointError("sparse depth must be finite")
    if float(rgb_array.min()) < 0.0 or float(rgb_array.max()) > 1.0:
        raise ValueError("display RGB must lie in [0, 1]")
    maximum = float(sparse_depth_max_m)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("sparse depth maximum must be finite and positive")
    if float(sparse_array.min()) < 0.0 or \
            float(sparse_array.max()) > maximum:
        raise ValueError(
            "sparse depth must lie in the declared meter domain [0, %.6g]" %
            maximum)
    return rgb_array, sparse_array, maximum


def write_prediction_export(
        *, root: Path, model: str, method: str, sample_index: int,
        evaluation_identity: str, artifact_index_sha256: str,
        sparse_depth_max_m: float, rgb, sparse, gt, pred) -> Path:
    record = {
        "sample_index": int(sample_index),
        "gt": np.asarray(gt, dtype=np.float32),
        "pred": np.asarray(pred, dtype=np.float32),
    }
    _, target, prediction, valid = _validated_arrays(record)
    rgb_array, sparse_array, sparse_maximum = _validated_display_inputs(
        rgb, sparse, target.shape, sparse_depth_max_m)
    if not isinstance(evaluation_identity, str) or \
            len(evaluation_identity) != 64:
        raise ValueError("evaluation identity must be a SHA256 string")
    artifact_fingerprint = _validated_sha256(
        artifact_index_sha256, "artifact index")
    output = prediction_path(root, method, sample_index)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("prediction export already exists: %s" % output)
    absolute_error = np.zeros_like(target, dtype=np.float32)
    absolute_error[valid] = np.abs(
        prediction[valid] - target[valid]).astype(np.float32)
    np.savez_compressed(
        output,
        format_version=np.int64(1),
        model=np.asarray(str(model)),
        method=np.asarray(method),
        sample_index=np.int64(sample_index),
        evaluation_identity=np.asarray(evaluation_identity),
        artifact_index_sha256=np.asarray(artifact_fingerprint),
        sparse_depth_max_m=np.float64(sparse_maximum),
        rgb=rgb_array,
        sparse=sparse_array,
        gt=target,
        pred=prediction,
        valid_gt=valid,
        abs_error=absolute_error,
    )
    return output


def load_prediction_export(
        path: Path, expected_model: str, expected_method: str,
        expected_index: int, expected_identity: str,
        expected_artifact_index_sha256: str) -> dict:
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError("prediction export is missing: %s" % source_path)
    with np.load(source_path, allow_pickle=False) as source:
        payload = dict((name, source[name]) for name in source.files)
    if set(payload) != PREDICTION_FIELDS:
        raise KeyError("prediction export fields changed: %s" % source_path)
    if int(payload["format_version"].item()) != 1:
        raise ValueError("prediction export version changed")
    if str(payload["model"].item()) != str(expected_model) or \
            str(payload["method"].item()) != str(expected_method) or \
            int(payload["sample_index"].item()) != int(expected_index):
        raise ValueError("prediction export identity changed")
    if str(payload["evaluation_identity"].item()) != str(expected_identity):
        raise ValueError("prediction evaluation identity changed")
    expected_fingerprint = _validated_sha256(
        expected_artifact_index_sha256, "artifact index")
    if str(payload["artifact_index_sha256"].item()) != expected_fingerprint:
        raise ValueError("prediction artifact index fingerprint changed")
    persisted_sparse_maximum = float(
        payload["sparse_depth_max_m"].item())
    if not math.isfinite(persisted_sparse_maximum) or \
            persisted_sparse_maximum != NYU_SPARSE_DEPTH_MAX_M:
        raise ValueError(
            "prediction fixed NYU sparse-depth maximum changed")
    _, gt, pred, valid = _validated_arrays(payload)
    rgb, sparse, sparse_maximum = _validated_display_inputs(
        payload["rgb"], payload["sparse"], gt.shape,
        NYU_SPARSE_DEPTH_MAX_M)
    payload["rgb"] = rgb
    payload["sparse"] = sparse
    payload["sparse_depth_max_m"] = np.asarray(
        sparse_maximum, dtype=np.float64)
    if not np.array_equal(payload["valid_gt"], valid):
        raise ValueError("prediction valid-GT mask changed")
    expected_error = np.zeros_like(gt, dtype=np.float32)
    expected_error[valid] = np.abs(pred[valid] - gt[valid])
    if not np.array_equal(payload["abs_error"], expected_error):
        raise ValueError("prediction absolute error changed")
    return payload


def _validate_method_order(methods) -> Tuple[str, ...]:
    values = tuple(str(method) for method in methods)
    if values != SELECTED_METHODS:
        raise ValueError("selected evaluation method order changed")
    return values


def build_relative_loss_rows(aggregate_rows) -> Tuple[dict, ...]:
    rows = tuple(aggregate_rows)
    if not rows or str(rows[0]["method"]) != "fp32":
        raise ValueError("relative loss requires FP32 as the first row")
    fp32 = rows[0]
    metrics = (
        ("pooled_rmse", "m"),
        ("mean_sample_rmse", "m"),
        ("pooled_mae", "m"),
        ("pooled_abs_rel", ""),
        ("pooled_irmse", "inverse_m"),
    )
    output = []
    for row in rows:
        current = {
            "model": str(row["model"]),
            "method": str(row["method"]),
            "configuration": str(row["configuration"]),
        }
        for metric, unit in metrics:
            baseline = float(fp32[metric])
            value = float(row[metric])
            suffix = "_delta_%s" % unit if unit else "_delta"
            current[metric + suffix] = value - baseline
            current[metric + "_relative_percent"] = \
                0.0 if baseline == 0.0 else \
                (value / baseline - 1.0) * 100.0
        output.append(current)
    return tuple(output)


def aggregate_prediction_exports(
        root: Path, artifact_index: FormalArtifactIndex,
        methods=SELECTED_METHODS) -> FormalAggregation:
    methods = _validate_method_order(methods)
    if not isinstance(artifact_index, FormalArtifactIndex):
        raise TypeError("formal aggregation requires its artifact index")
    model = artifact_index.model
    indices = artifact_index.evaluation_indices
    if len(indices) != EXPECTED_EVALUATION_SAMPLES or \
            len(indices) != len(set(indices)):
        raise ValueError("formal evaluation requires 64 unique identities")
    identity = ordered_evaluation_identity(indices)
    aggregate_rows = []
    sample_rows = []
    reference_inputs = {}
    for method in methods:
        records = []
        directory = Path(root) / "predictions" / method
        actual_paths = tuple(sorted(directory.glob("sample_*.npz"))) \
            if directory.is_dir() else ()
        expected_paths = tuple(
            prediction_path(root, method, index) for index in indices)
        if actual_paths != tuple(sorted(expected_paths)):
            raise RuntimeError("prediction coverage mismatch for %s" % method)
        for sample_index, path in zip(indices, expected_paths):
            payload = load_prediction_export(
                path, model, method, sample_index, identity,
                expected_artifact_index_sha256=artifact_index.fingerprint)
            aligned = (
                payload["rgb"], payload["sparse"], payload["gt"],
                payload["valid_gt"], payload["sparse_depth_max_m"],
            )
            if method == "fp32":
                reference_inputs[sample_index] = tuple(
                    value.copy() for value in aligned)
            elif any(not np.array_equal(value, reference)
                     for value, reference in zip(
                         aligned, reference_inputs[sample_index])):
                raise ValueError(
                    "prediction aligned input differs for %s sample %d" %
                    (method, sample_index))
            records.append(payload)
        result = aggregate_predictions(records)
        aggregate_rows.append({
            "model": str(model),
            "method": method,
            "configuration": METHOD_LABELS[method],
            "samples": len(records),
            "valid_pixel_count": result.valid_pixel_count,
            "squared_error_sum": result.squared_error_sum,
            "pooled_rmse": result.pooled_rmse,
            "mean_sample_rmse": result.mean_sample_rmse,
            "pooled_mae": result.pooled_mae,
            "pooled_abs_rel": result.pooled_abs_rel,
            "pooled_irmse": result.pooled_irmse,
            "nonpositive_pixel_count": result.nonpositive_pixel_count,
            "nonpositive_ratio": result.nonpositive_ratio,
        })
        sample_rows.extend(dict(
            row,
            model=str(model),
            method=method,
            configuration=METHOD_LABELS[method],
        ) for row in result.sample_metrics)
    relative_rows = build_relative_loss_rows(aggregate_rows)
    return FormalAggregation(
        aggregate_metrics=tuple(aggregate_rows),
        sample_metrics=tuple(sample_rows),
        relative_fp_loss=relative_rows,
    )


def _write_csv(path: Path, rows) -> None:
    rows = tuple(rows)
    if not rows:
        raise ValueError("cannot write an empty formal evaluation table")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_aggregation_tables(root: Path, result: FormalAggregation) -> None:
    output = Path(root)
    _write_csv(output / "aggregate_metrics.csv", result.aggregate_metrics)
    _write_csv(output / "sample_metrics.csv", result.sample_metrics)
    _write_csv(output / "relative_fp_loss.csv", result.relative_fp_loss)


def assignment_cost_row(method: str, assignment, basis) -> dict:
    weights = tuple(
        (str(name), int(bits)) for name, bits in assignment["weight_bits"])
    activations = tuple(
        ((str(owner[0]), str(owner[1])), int(bits))
        for owner, bits in assignment["activation_bits"])
    weight_costs = tuple(
        (str(name), int(cost)) for name, cost in basis["weight_macs"])
    activation_costs = tuple(
        ((str(owner[0]), str(owner[1])), int(cost))
        for owner, cost in basis["activation_elements"])
    if any(cost <= 0 for name, cost in weight_costs) or any(
            cost <= 0 for owner, cost in activation_costs):
        raise ValueError("formal cost basis values must be positive")
    weight_bits = dict(weights)
    activation_bits = dict(activations)
    weight_cost = dict(weight_costs)
    activation_cost = dict(activation_costs)
    if len(weight_bits) != len(weights) or set(weight_bits) != set(weight_cost):
        raise ValueError("weight assignment and cost coverage differ")
    if len(activation_bits) != len(activations) or \
            set(activation_bits) != set(activation_cost):
        raise ValueError("activation assignment and cost coverage differ")
    supported = {4, 6, 8, 32}
    if set(weight_bits.values()) - supported or \
            set(activation_bits.values()) - supported:
        raise ValueError("formal cost assignment contains unsupported bits")
    weight_denominator = sum(weight_cost.values())
    activation_denominator = sum(activation_cost.values())

    def share(values, costs, bits, denominator):
        return sum(costs[name] for name in costs
                   if values[name] == bits) / float(denominator)

    return {
        "method": str(method),
        "configuration": METHOD_LABELS[str(method)],
        "average_weight_bits": sum(
            weight_bits[name] * weight_cost[name]
            for name in weight_cost) / float(weight_denominator),
        "average_activation_bits": sum(
            activation_bits[name] * activation_cost[name]
            for name in activation_cost) / float(activation_denominator),
        "weight_w4_share": share(
            weight_bits, weight_cost, 4, weight_denominator),
        "weight_w6_share": share(
            weight_bits, weight_cost, 6, weight_denominator),
        "weight_w8_share": share(
            weight_bits, weight_cost, 8, weight_denominator),
        "weight_fp32_share": share(
            weight_bits, weight_cost, 32, weight_denominator),
        "activation_a4_share": share(
            activation_bits, activation_cost, 4, activation_denominator),
        "activation_a6_share": share(
            activation_bits, activation_cost, 6, activation_denominator),
        "activation_a8_share": share(
            activation_bits, activation_cost, 8, activation_denominator),
        "activation_fp32_share": share(
            activation_bits, activation_cost, 32, activation_denominator),
    }


def _uniform_contract_assignment(contract, bits: int) -> dict:
    precision = int(bits)
    if precision not in (4, 6, 8, 32):
        raise ValueError("uniform formal precision is unsupported")
    return {
        "weight_bits": tuple(
            (name, precision) for name in contract.weight_modules),
        "activation_bits": tuple(
            (owner, precision) for block in contract.blocks
            for owner in block.activation_owners),
    }


def _cost_basis_mapping(costs) -> dict:
    return {
        "weight_macs": tuple(
            (str(name), int(value)) for name, value in costs.weight_macs),
        "activation_elements": tuple(
            ((str(owner[0]), str(owner[1])), int(value))
            for owner, value in costs.activation_elements),
    }


def load_indexed_cost_contracts(
        index: FormalArtifactIndex, selected, model_config,
        config: Path, launch_spec: Path) -> IndexedCostContracts:
    """Reconstruct publication costs from indexed artifacts and method rules."""
    if not isinstance(index, FormalArtifactIndex):
        raise TypeError("indexed costs require a formal artifact index")
    if str(model_config.model) != index.model or tuple(
            model_config.evaluation_indices) != index.evaluation_indices:
        raise ValueError("indexed cost model identity changed")
    from scripts.nyu_model_runtime import NYUModelRuntime
    from scripts import train_nyu_selected_qat as qat_runner
    from scripts.run_nyu_selected_ptq import load_p3_t3_assignment
    from spn_quant.model_contracts import build_model_quantization_contract

    trusted_p3_t3 = qat_runner.load_trusted_p3_t3_contract(
        config, launch_spec, model_config.model)

    runtime = NYUModelRuntime.from_config(model_config)
    try:
        model = runtime.build_model(runtime.device)
        contract = build_model_quantization_contract(index.model, model)
        p3_path, p3_assignment, costs = _validated_p3_inputs(
            index, selected, model_config, contract, trusted_p3_t3)
    finally:
        runtime.close()
    basis = _cost_basis_mapping(costs)
    if index.methods["fp32"].artifact.resolve() != \
            model_config.checkpoint.resolve():
        raise ValueError("indexed FP32 cost checkpoint changed")
    p3_ptq_assignment = load_p3_t3_assignment(
        p3_path,
        contract,
        selected.method_hyperparameters["p3_t3_mixed_ptq"],
    )
    if p3_ptq_assignment != p3_assignment:
        raise ValueError("indexed P3/T3 strict loaders disagree")

    ptq_assignments = {}
    for method in PTQ_METHODS:
        manifest = _validate_ptq_manifest(
            index, method, contract, model_config)
        if method == "p3_t3_mixed_ptq":
            ptq_assignments[method] = _assignment_mapping(p3_assignment)
            continue
        ptq_assignments[method] = {
            "weight_bits": tuple(
                (str(name), int(manifest["weight_bits"]))
                for name in manifest["module_names"]),
            "activation_bits": tuple(
                ((str(owner[0]), str(owner[1])),
                 int(manifest["activation_bits"]))
                for owner in manifest["activation_owners"]),
        }

    hawq_entry = index.methods["hawq_mixed_le6"]
    hawq_config = selected.method_hyperparameters["hawq_mixed_le6"]
    hawq_assignment = qat_runner.load_hawq_qat_assignment(
        _supporting_path(hawq_entry, "hawq_assignment"),
        _supporting_path(hawq_entry, "hawq_trace_artifact"),
        contract,
        model_config.checkpoint,
        dict(hawq_config["trace"]),
        float(hawq_config["maximum_average_weight_bits"]),
        float(hawq_config["maximum_average_activation_bits"]),
    )
    hawq_payload = json.loads(_supporting_path(
        hawq_entry, "hawq_assignment").read_text(encoding="utf-8"))
    hawq_basis = {
        "weight_macs": tuple(
            (str(row["module"]), int(row["macs"]))
            for row in hawq_payload["cost_basis"]["weight_macs"]),
        "activation_elements": tuple(
            ((str(row["site"]), str(row["role"])), int(row["elements"]))
            for row in hawq_payload["cost_basis"]["activation_traffic"]),
    }
    _require_exact_finite_equal(
        hawq_basis, basis, "indexed HAWQ cost basis")

    mixed_config = selected.method_hyperparameters["mixed_task_aware"]
    mixed_assignment, mixed_audit = qat_runner.mixed_task_aware_assignment(
        contract,
        p3_assignment,
        p3_assignment.activation_bits,
        costs,
        float(mixed_config["maximum_average_activation_bits"]),
    )
    del mixed_audit
    assignments = {
        "fp32": _uniform_contract_assignment(contract, 32),
        "rtn_w8a8": ptq_assignments["rtn_w8a8"],
        "rtn_w4a4": ptq_assignments["rtn_w4a4"],
        "qdrop_w6a6": ptq_assignments["qdrop_w6a6"],
        "brecq_w6a6": ptq_assignments["brecq_w6a6"],
        "hawq_mixed_le6": _assignment_mapping(hawq_assignment),
        "lsqplus_w6a6": _uniform_contract_assignment(contract, 6),
        "lsqplus_w4a4": _uniform_contract_assignment(contract, 4),
        "mixed_task_aware": _assignment_mapping(mixed_assignment),
        "p3_t3_mixed_ptq": ptq_assignments["p3_t3_mixed_ptq"],
    }
    contract_manifest = qat_runner._contract_manifest(contract)
    for method in QAT_METHODS:
        payload = _load_tensor_payload(index.methods[method].artifact)
        qat_runner.validate_checkpoint_payload(payload)
        if str(payload["model_name"]) != index.model or \
                str(payload["method"]) != method or \
                payload["contract_manifest"] != contract_manifest:
            raise ValueError("indexed QAT cost contract identity changed")
        if payload["run_state"] != {
                "terminal": True,
                "completed": True,
                "reason": payload["convergence"]["reason"]}:
            raise ValueError("indexed QAT cost contract is incomplete")
        _require_exact_finite_equal(
            payload["assignment"], assignments[method],
            "%s indexed QAT assignment" % method)
    return IndexedCostContracts(
        artifact_index_sha256=index.fingerprint,
        methods=dict((method, {
            "assignment": assignments[method],
            "cost_basis": basis,
        }) for method in SELECTED_METHODS),
    )


def _validate_indexed_cost_contracts(
        cost_contracts, artifact_index: FormalArtifactIndex) -> Mapping[str, dict]:
    if not isinstance(cost_contracts, IndexedCostContracts):
        raise TypeError("formal summary requires indexed cost contracts")
    if cost_contracts.artifact_index_sha256 != artifact_index.fingerprint:
        raise ValueError("indexed cost contract artifact index changed")
    methods = cost_contracts.methods
    if tuple(methods) != SELECTED_METHODS:
        raise ValueError("indexed cost contract method order changed")
    normalized = {}
    for method in SELECTED_METHODS:
        row = methods[method]
        if set(row) != {"assignment", "cost_basis"}:
            raise KeyError("indexed cost contract fields changed")
        cost = assignment_cost_row(
            method, row["assignment"], row["cost_basis"])
        normalized[method] = {
            "assignment": row["assignment"],
            "cost_basis": row["cost_basis"],
            "cost": cost,
        }
    return normalized


def _json_ready(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return dict((str(key), _json_ready(current))
                    for key, current in value.items())
    if isinstance(value, (list, tuple)):
        return [_json_ready(current) for current in value]
    return value


def _write_json(path: Path, payload) -> None:
    Path(path).write_text(
        json.dumps(
            _json_ready(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def paired_tensor_diagnostic_row(
        *, reference, candidate, model: str, method: str,
        sample_index: int, iteration: int, owner: str,
        owner_kind: str) -> dict:
    if not torch.is_tensor(reference) or not torch.is_tensor(candidate):
        raise TypeError("paired diagnostics require tensors")
    if tuple(reference.shape) != tuple(candidate.shape) or \
            reference.numel() <= 0:
        raise ValueError("paired diagnostic tensor shapes differ")
    left = reference.detach().to(dtype=torch.float64, device="cpu")
    right = candidate.detach().to(dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(left).all().item()) or not bool(
            torch.isfinite(right).all().item()):
        raise FloatingPointError("paired diagnostic tensors must be finite")
    difference = right - left
    signal_energy = float(left.square().sum().item())
    error_energy = float(difference.square().sum().item())
    elements = int(left.numel())
    if error_energy == 0.0:
        sqnr = 300.0
    elif signal_energy == 0.0:
        sqnr = -300.0
    else:
        sqnr = 10.0 * math.log10(signal_energy / error_energy)
        sqnr = min(max(sqnr, -300.0), 300.0)
    row = {
        "model": str(model),
        "method": str(method),
        "sample_index": int(sample_index),
        "iteration": int(iteration),
        "owner": str(owner),
        "owner_kind": str(owner_kind),
        "elements": elements,
        "signal_energy": signal_energy,
        "error_energy": error_energy,
        "mse": error_energy / float(elements),
        "sqnr_db": sqnr,
    }
    if any(not math.isfinite(float(row[field])) for field in (
            "signal_energy", "error_energy", "mse", "sqnr_db")):
        raise FloatingPointError("paired diagnostic metrics must be finite")
    return row


def paired_task_diagnostic_rows(
        reference, candidate, model: str, method: str,
        sample_index: int) -> Tuple[dict, ...]:
    reference_states = tuple(reference.propagation_states)
    candidate_states = tuple(candidate.propagation_states)
    if not reference_states or len(reference_states) != len(candidate_states):
        raise ValueError("paired propagation iteration coverage differs")
    rows = [paired_tensor_diagnostic_row(
        reference=reference.initial_depth,
        candidate=candidate.initial_depth,
        model=model,
        method=method,
        sample_index=sample_index,
        iteration=-1,
        owner="initial_depth",
        owner_kind="semantic_state",
    )]
    rows.extend(paired_tensor_diagnostic_row(
        reference=left,
        candidate=right,
        model=model,
        method=method,
        sample_index=sample_index,
        iteration=iteration,
        owner="propagation_state",
        owner_kind="semantic_state",
    ) for iteration, (left, right) in enumerate(zip(
        reference_states, candidate_states)))
    return tuple(rows)


def _runtime_code_row(row) -> bool:
    return str(row.get("owner_kind", row.get("kind", "quantizer"))) not in \
        ("weight", "bias")


def _diagnostic_counter_snapshot(deployment) -> Tuple[dict, ...]:
    activation_bits = dict(deployment.assignment["activation_bits"])
    rows = []
    for source_index, source in enumerate(deployment.diagnostics_sources):
        if not hasattr(source, "counter_snapshot") or not callable(
                source.counter_snapshot):
            raise TypeError(
                "hard-code diagnostic source requires counter_snapshot")
        source_rows = source.counter_snapshot()
        if not isinstance(source_rows, (tuple, list)):
            raise TypeError(
                "hard-code counter_snapshot must return ordered rows")
        for row in source_rows:
            current = dict(row)
            activation_owner = (str(current["module"]), str(current["role"])) \
                if "role" in current else None
            if activation_owner in activation_bits:
                current["bits"] = activation_bits[activation_owner]
            source_owner = current["owner"] if "owner" in current else \
                current["owner_name"] if "owner_name" in current else \
                current["module"]
            if isinstance(source_owner, (tuple, list)):
                owner = "::".join(str(value) for value in source_owner)
            else:
                owner = str(source_owner)
            owner_kind = str(current["owner_kind"]) \
                if "owner_kind" in current else str(current["kind"]) \
                if "kind" in current else "quantizer"
            current.update({
                "owner": owner,
                "owner_kind": owner_kind,
            })
            current["source_index"] = source_index
            required = {
                "numel", "zero_code_count", "saturation_count",
                "source_index", "owner", "owner_kind",
            }
            if not required <= set(current):
                raise KeyError(
                    "hard-code counter snapshot fields changed")
            counters = tuple(current[field] for field in (
                "numel", "zero_code_count", "saturation_count"))
            if any(isinstance(value, bool) or not isinstance(
                    value, (int, np.integer)) for value in counters):
                raise TypeError("hard-code counters must be integers")
            numel, zero_codes, saturated = tuple(
                int(value) for value in counters)
            if numel < 0 or zero_codes < 0 or saturated < 0 or \
                    zero_codes > numel or saturated > numel:
                raise ValueError("hard-code counters are invalid")
            if _runtime_code_row(current):
                rows.append(current)
    output = tuple(rows)
    if not output:
        raise RuntimeError(
            "hard deployment produced no runtime code counters")
    return output


def hard_code_owner_contract(rows) -> Tuple[dict, ...]:
    contract = tuple({
        "source_index": int(row["source_index"]),
        "owner": str(row["owner"]),
        "owner_kind": str(row["owner_kind"]),
    } for row in rows)
    identities = tuple(
        (row["source_index"], row["owner"], row["owner_kind"])
        for row in contract)
    if not contract or len(identities) != len(set(identities)):
        raise ValueError("hard-code owner contract is empty or ambiguous")
    return contract


def hard_code_counter_deltas(
        before, after, owner_contract, model: str, method: str,
        sample_index: int) -> Tuple[dict, ...]:
    before = tuple(before)
    after = tuple(after)
    contract = tuple(owner_contract)

    def identities(rows):
        return tuple({
            "source_index": int(row["source_index"]),
            "owner": str(row["owner"]),
            "owner_kind": str(row["owner_kind"]),
        } for row in rows)

    if identities(before) != contract or identities(after) != contract:
        raise ValueError("hard-code owner coverage differs from deployment contract")
    rows = []
    metadata_fields = (
        "module", "role", "kind", "group", "owner_name", "bits")
    for left, right in zip(before, after):
        deltas = {}
        for field in (
                "numel", "zero_code_count", "saturation_count"):
            delta = int(right[field]) - int(left[field])
            if delta < 0:
                raise ValueError("hard-code cumulative counter decreased")
            deltas[field] = delta
        if deltas["numel"] <= 0 or \
                deltas["zero_code_count"] > deltas["numel"] or \
                deltas["saturation_count"] > deltas["numel"]:
            raise ValueError("hard-code sample counter delta is invalid")
        current = dict(
            (field, right[field]) for field in metadata_fields
            if field in right)
        if "calls" in left or "calls" in right:
            if "calls" not in left or "calls" not in right:
                raise ValueError("hard-code call counter coverage changed")
            calls = int(right["calls"]) - int(left["calls"])
            if calls <= 0:
                raise ValueError("hard-code call counter did not advance")
            current["calls"] = calls
        current.update({
            "model": str(model),
            "method": str(method),
            "sample_index": int(sample_index),
            "iteration": -1,
            "source_index": int(right["source_index"]),
            "owner": str(right["owner"]),
            "owner_kind": str(right["owner_kind"]),
            "numel": deltas["numel"],
            "zero_code_count": deltas["zero_code_count"],
            "saturation_count": deltas["saturation_count"],
            "zero_code_rate": deltas["zero_code_count"] /
                float(deltas["numel"]),
            "saturation_rate": deltas["saturation_count"] /
                float(deltas["numel"]),
        })
        rows.append(current)
    return tuple(rows)


def _weighted_diagnostic_ratio(rows, field):
    selected = tuple(row for row in rows
                     if field in row and "numel" in row)
    if not selected:
        return None
    denominator = sum(int(row["numel"]) for row in selected)
    if denominator <= 0:
        raise ValueError("quantization diagnostic element count is invalid")
    numerator = math.fsum(
        float(row[field]) * int(row["numel"]) for row in selected)
    value = numerator / float(denominator)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("quantization diagnostic ratio is invalid")
    return value


def _clone_runtime_value(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return dict((key, _clone_runtime_value(current))
                    for key, current in value.items())
    if isinstance(value, tuple):
        return tuple(_clone_runtime_value(current) for current in value)
    if isinstance(value, list):
        return [_clone_runtime_value(current) for current in value]
    return value


def evaluate_formal_deployment(
        method: str, deployment: FormalDeployment,
        reference_deployment: FormalDeployment,
        index: FormalArtifactIndex, root: Path) -> dict:
    """Evaluate a prepared hard context and publish its completion last."""
    from scripts.run_nyu_model_p3t3_search import (
        _preserve_input_policy,
        _propagation_valid,
    )
    from scripts.run_nyu_rtn_quantization import (
        batch_from_sample,
        seeded_sample,
    )
    from spn_quant.adapters import install_model_semantic_adapter

    method = str(method)
    if method != index.methods[method].method:
        raise ValueError("formal deployment method identity changed")
    if reference_deployment.artifact.resolve() != \
            index.methods["fp32"].artifact.resolve():
        raise ValueError("formal FP32 diagnostic reference changed")
    method_root = Path(root) / "methods" / method
    prediction_root = Path(root) / "predictions" / method
    for output in (method_root, prediction_root):
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                "formal method output must be empty: %s" % output)
        output.mkdir(parents=True, exist_ok=True)
    dataset = deployment.runtime.build_dataset("val")
    if any(sample_index >= len(dataset)
           for sample_index in index.evaluation_indices):
        raise ValueError("formal evaluation identity is outside val dataset")
    seed = int(deployment.runtime.saved_args.seed)
    records = []
    propagation_rows = []
    quantization_rows = []
    quantization_owner_contract = () if method == "fp32" else \
        hard_code_owner_contract(
            _diagnostic_counter_snapshot(deployment))
    block_rows = []
    semantic_rows = []
    deployment.model.eval()
    reference_deployment.model.eval()
    candidate_semantic = install_model_semantic_adapter(
        deployment.model, index.model, strict=True)
    candidate_semantic.delegate_quantization()
    candidate_blocks = ContractBlockOutputCapture(
        deployment.model, deployment.contract)
    shared_reference = reference_deployment is deployment
    if shared_reference:
        reference_semantic = candidate_semantic
        reference_blocks = candidate_blocks
    else:
        reference_semantic = install_model_semantic_adapter(
            reference_deployment.model, index.model, strict=True)
        reference_semantic.delegate_quantization()
        reference_blocks = ContractBlockOutputCapture(
            reference_deployment.model, reference_deployment.contract)
    try:
        with torch.no_grad():
            for sample_index in index.evaluation_indices:
                sample = seeded_sample(dataset, sample_index, seed)
                batch = batch_from_sample(sample)
                model_input, target = deployment.runtime.model_input(
                    batch, deployment.runtime.device)
                counters_before = () if method == "fp32" else \
                    _diagnostic_counter_snapshot(deployment)
                if shared_reference:
                    candidate_semantic.begin_task_capture()
                    candidate_blocks.begin()
                    candidate_output = deployment.model(
                        *_clone_runtime_value(model_input))
                    prediction = deployment.runtime.prediction(candidate_output)
                    candidate_task = candidate_semantic.task_capture()
                    candidate_block_values = candidate_blocks.values()
                    reference_task = candidate_task
                    reference_block_values = candidate_block_values
                else:
                    reference_semantic.begin_task_capture()
                    reference_blocks.begin()
                    reference_output = reference_deployment.model(
                        *_clone_runtime_value(model_input))
                    reference_prediction = reference_deployment.runtime.prediction(
                        reference_output)
                    if tuple(reference_prediction.shape) != tuple(target.shape) or \
                            not bool(torch.isfinite(
                                reference_prediction).all().item()):
                        raise ValueError(
                            "formal FP32 diagnostic prediction changed")
                    reference_task = reference_semantic.task_capture()
                    reference_block_values = reference_blocks.values()
                    candidate_semantic.begin_task_capture()
                    candidate_blocks.begin()
                    candidate_output = deployment.model(
                        *_clone_runtime_value(model_input))
                    prediction = deployment.runtime.prediction(candidate_output)
                    candidate_task = candidate_semantic.task_capture()
                    candidate_block_values = candidate_blocks.values()
                semantic_rows.extend(paired_task_diagnostic_rows(
                    reference_task, candidate_task, index.model, method,
                    sample_index))
                block_rows.extend(paired_block_diagnostic_rows(
                    reference_block_values, candidate_block_values,
                    index.model, method, sample_index))
                if tuple(prediction.shape) != tuple(target.shape) or \
                        int(prediction.shape[0]) != 1 or \
                        int(prediction.shape[1]) != 1:
                    raise ValueError("formal prediction tensor shape changed")
                if not bool(torch.isfinite(prediction).all().item()):
                    raise FloatingPointError(
                        "formal prediction contains non-finite values")
                if deployment.propagation is not None:
                    current_propagation = tuple(
                        deployment.propagation.statistics())
                    preserve_input = _preserve_input_policy(
                        index.model, deployment.propagation)
                    if not _propagation_valid(
                            index.model, preserve_input,
                            current_propagation):
                        raise RuntimeError(
                            "formal propagation invariants failed: %s sample %d" %
                            (method, sample_index))
                    propagation_rows.extend(dict(
                        row,
                        model=index.model,
                        method=method,
                        sample_index=int(sample_index),
                        iteration=int(row["iteration"])
                        if "iteration" in row else -1,
                        owner=str(row["signal"])
                        if "signal" in row else "propagation_invariant",
                        owner_kind="propagation_invariant",
                    ) for row in current_propagation)
                if method != "fp32":
                    counters_after = _diagnostic_counter_snapshot(deployment)
                    quantization_rows.extend(hard_code_counter_deltas(
                        counters_before,
                        counters_after,
                        quantization_owner_contract,
                        index.model,
                        method,
                        sample_index,
                    ))
                gt = target[0, 0].detach().cpu().numpy().astype(np.float32)
                pred = prediction[0, 0].detach().cpu().numpy().astype(np.float32)
                rgbd = sample["rgbd"]
                if not torch.is_tensor(rgbd) or rgbd.ndim != 3 or \
                        int(rgbd.shape[0]) != 4:
                    raise ValueError("formal NYU sample must provide CHW RGBD")
                rgb = rgbd[:3].permute(1, 2, 0).numpy().astype(np.float32)
                sparse = rgbd[3].numpy().astype(np.float32)
                write_prediction_export(
                    root=root,
                    model=index.model,
                    method=method,
                    sample_index=sample_index,
                    evaluation_identity=index.evaluation_identity,
                    artifact_index_sha256=index.fingerprint,
                    sparse_depth_max_m=NYU_SPARSE_DEPTH_MAX_M,
                    rgb=rgb,
                    sparse=sparse,
                    gt=gt,
                    pred=pred,
                )
                records.append({
                    "sample_index": sample_index,
                    "gt": gt,
                    "pred": pred,
                })
    finally:
        candidate_blocks.close()
        candidate_semantic.close()
        if not shared_reference:
            reference_blocks.close()
            reference_semantic.close()
    aggregation = aggregate_predictions(records)
    aggregate_row = {
        "model": index.model,
        "method": method,
        "configuration": METHOD_LABELS[method],
        "samples": len(records),
        "valid_pixel_count": aggregation.valid_pixel_count,
        "squared_error_sum": aggregation.squared_error_sum,
        "pooled_rmse": aggregation.pooled_rmse,
        "mean_sample_rmse": aggregation.mean_sample_rmse,
        "pooled_mae": aggregation.pooled_mae,
        "pooled_abs_rel": aggregation.pooled_abs_rel,
        "pooled_irmse": aggregation.pooled_irmse,
        "nonpositive_pixel_count": aggregation.nonpositive_pixel_count,
        "nonpositive_ratio": aggregation.nonpositive_ratio,
    }
    sample_rows = tuple(dict(
        row,
        model=index.model,
        method=method,
        configuration=METHOD_LABELS[method],
    ) for row in aggregation.sample_metrics)
    cost = assignment_cost_row(
        method, deployment.assignment, deployment.cost_basis)
    diagnostics_path = method_root / "diagnostics.json"
    if method != "fp32" and not quantization_rows:
        raise RuntimeError("formal hard-code diagnostics are empty")
    if not block_rows or not semantic_rows:
        raise RuntimeError("formal paired diagnostics are empty")
    _write_json(diagnostics_path, {
        "format_version": 1,
        "model": index.model,
        "method": method,
        "hard_deployment": int(method != "fp32"),
        "samples": len(records),
        "prediction_finite_ratio": 1.0,
        "prediction_nonpositive_ratio": aggregation.nonpositive_ratio,
        "weighted_saturation_ratio": _weighted_diagnostic_ratio(
            quantization_rows, "saturation_rate"),
        "weighted_zero_code_ratio": _weighted_diagnostic_ratio(
            quantization_rows, "zero_code_rate"),
        "quantization_owner_contract": quantization_owner_contract,
        "quantization_statistics": quantization_rows,
        "propagation_statistics": propagation_rows,
        "block_output_statistics": block_rows,
        "semantic_state_statistics": semantic_rows,
    })
    _write_csv(method_root / "aggregate_metrics.csv", (aggregate_row,))
    _write_csv(method_root / "sample_metrics.csv", sample_rows)
    _write_csv(method_root / "cost.csv", (cost,))
    formal_run = {
        "format_version": 1,
        "complete": 1,
        "model": index.model,
        "method": method,
        "artifact_kind": deployment.artifact_kind,
        "artifact": str(deployment.artifact.resolve()),
        "artifact_sha256": file_sha256(deployment.artifact),
        "supporting_artifacts": [
            {"name": name, "path": str(path.resolve()), "sha256": sha256}
            for name, path, sha256 in
            index.methods[method].supporting_artifacts],
        "artifact_index": str(index.source),
        "artifact_index_sha256": index.fingerprint,
        "evaluation_indices": list(index.evaluation_indices),
        "evaluation_identity": index.evaluation_identity,
        "prediction_directory": str(prediction_root.resolve()),
        "metrics": aggregate_row,
        "cost": cost,
        "assignment": deployment.assignment,
        "cost_basis": deployment.cost_basis,
        "diagnostics": str(diagnostics_path.resolve()),
    }
    _write_json(method_root / "formal_run.json", formal_run)
    return formal_run


def run_formal_method(
        *, config: Path, launch_spec: Path, model: str, method: str,
        artifact_index: Path, output_root: Path) -> dict:
    from spn_quant.experiment_config import load_selected_quantization_config

    selected = load_selected_quantization_config(config)
    model_config = _model_config(selected, model)
    index = load_formal_artifact_index(
        artifact_index, model, model_config.evaluation_indices)
    deployment = prepare_formal_deployment(
        method, index, selected, model_config, config, launch_spec)
    reference = None
    try:
        reference = deployment if method == "fp32" else \
            prepare_formal_deployment(
                "fp32", index, selected, model_config, config, launch_spec)
        return evaluate_formal_deployment(
            method, deployment, reference, index, output_root)
    finally:
        deployment.close()
        if reference is not None and reference is not deployment:
            reference.close()


def _require_exact_finite_equal(actual, expected, family: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError("%s fields differ" % family)
        for key in expected:
            _require_exact_finite_equal(
                actual[key], expected[key], "%s.%s" % (family, key))
        return
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or \
                len(actual) != len(expected):
            raise ValueError("%s sequence differs" % family)
        for position, (left, right) in enumerate(zip(actual, expected)):
            _require_exact_finite_equal(
                left, right, "%s[%d]" % (family, position))
        return
    if isinstance(expected, bool):
        if not isinstance(actual, bool) or actual is not expected:
            raise ValueError("%s differs" % family)
        return
    if isinstance(expected, (int, float, np.integer, np.floating)):
        if isinstance(actual, bool) or not isinstance(
                actual, (int, float, np.integer, np.floating)):
            raise TypeError("%s must be numeric" % family)
        left = float(actual)
        right = float(expected)
        if not math.isfinite(left) or not math.isfinite(right) or left != right:
            raise ValueError("%s differs" % family)
        return
    if actual != expected:
        raise ValueError("%s differs" % family)


def _validate_formal_diagnostics(path, index, method) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "format_version", "model", "method", "hard_deployment", "samples",
        "prediction_finite_ratio", "prediction_nonpositive_ratio",
        "weighted_saturation_ratio", "weighted_zero_code_ratio",
        "quantization_owner_contract",
        "quantization_statistics", "propagation_statistics",
        "block_output_statistics", "semantic_state_statistics",
    }
    if set(payload) != required:
        raise KeyError("formal diagnostics fields changed for %s" % method)
    if int(payload["format_version"]) != 1 or \
            str(payload["model"]) != index.model or \
            str(payload["method"]) != method or \
            int(payload["samples"]) != EXPECTED_EVALUATION_SAMPLES:
        raise ValueError("formal diagnostics identity changed for %s" % method)
    top_level_ratios = (
        float(payload["prediction_finite_ratio"]),
        float(payload["prediction_nonpositive_ratio"]),
    )
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0
           for value in top_level_ratios):
        raise ValueError("formal diagnostic ratios are invalid for %s" % method)
    block_rows = tuple(payload["block_output_statistics"])
    semantic_rows = tuple(payload["semantic_state_statistics"])
    quantization_rows = tuple(payload["quantization_statistics"])
    owner_contract = tuple(payload["quantization_owner_contract"])
    if not block_rows or not semantic_rows or \
            (method != "fp32" and not quantization_rows):
        raise RuntimeError("formal diagnostics are empty for %s" % method)
    identity_fields = {
        "model", "method", "sample_index", "iteration", "owner",
        "owner_kind",
    }
    for family, rows in (
            ("block", block_rows),
            ("semantic", semantic_rows),
            ("quantization", quantization_rows)):
        for row in rows:
            if not identity_fields <= set(row):
                raise KeyError(
                    "%s diagnostic identity fields changed for %s" %
                    (family, method))
            if str(row["model"]) != index.model or \
                    str(row["method"]) != method:
                raise ValueError(
                    "%s diagnostic identity changed for %s" %
                    (family, method))
            if family in ("block", "semantic"):
                metric_fields = {
                    "elements", "signal_energy", "error_energy", "mse",
                    "sqnr_db",
                }
                if not metric_fields <= set(row):
                    raise KeyError(
                        "%s diagnostic metric fields changed for %s" %
                        (family, method))
                values = tuple(float(row[field]) for field in (
                    "signal_energy", "error_energy", "mse", "sqnr_db"))
                if int(row["elements"]) <= 0 or any(
                        not math.isfinite(value) for value in values) or any(
                            float(row[field]) < 0.0 for field in (
                                "signal_energy", "error_energy", "mse")):
                    raise ValueError(
                        "%s diagnostic metrics are invalid for %s" %
                        (family, method))
    expected_samples = set(index.evaluation_indices)
    block_samples = set(int(row["sample_index"]) for row in block_rows)
    semantic_samples = set(int(row["sample_index"]) for row in semantic_rows)
    if block_samples != expected_samples or semantic_samples != expected_samples:
        raise ValueError("paired diagnostic sample coverage changed for %s" % method)
    block_owners = None
    propagation_iterations = None
    for sample_index in index.evaluation_indices:
        current_blocks = tuple(
            str(row["owner"]) for row in block_rows
            if int(row["sample_index"]) == sample_index)
        current_initial = tuple(
            row for row in semantic_rows
            if int(row["sample_index"]) == sample_index and
            str(row["owner"]) == "initial_depth" and
            int(row["iteration"]) == -1)
        current_iterations = tuple(
            int(row["iteration"]) for row in semantic_rows
            if int(row["sample_index"]) == sample_index and
            str(row["owner"]) == "propagation_state")
        if not current_blocks or len(current_blocks) != len(
                set(current_blocks)) or len(current_initial) != 1 or \
                current_iterations != tuple(range(len(current_iterations))) or \
                not current_iterations:
            raise ValueError(
                "paired diagnostic owner coverage changed for %s" % method)
        if block_owners is None:
            block_owners = current_blocks
            propagation_iterations = current_iterations
        elif current_blocks != block_owners or \
                current_iterations != propagation_iterations:
            raise ValueError(
                "paired diagnostic iteration coverage changed for %s" % method)
    if method != "fp32":
        contract_fields = {"source_index", "owner", "owner_kind"}
        if not owner_contract or any(
                not isinstance(row, dict) or set(row) != contract_fields
                for row in owner_contract):
            raise ValueError(
                "formal hard-code owner contract changed for %s" % method)
        contract_identities = tuple(
            (int(row["source_index"]), str(row["owner"]),
             str(row["owner_kind"])) for row in owner_contract)
        if len(contract_identities) != len(set(contract_identities)) or any(
                source_index < 0 or not owner or not owner_kind
                for source_index, owner, owner_kind in contract_identities):
            raise ValueError(
                "formal hard-code owner contract is invalid for %s" % method)
        quantization_samples = set(
            int(row["sample_index"]) for row in quantization_rows)
        if quantization_samples != expected_samples:
            raise ValueError(
                "formal hard-code sample coverage changed for %s" % method)
        code_rows = tuple(row for row in quantization_rows
                          if "zero_code_rate" in row and
                          "saturation_rate" in row and "numel" in row and
                          "zero_code_count" in row and
                          "saturation_count" in row and
                          "source_index" in row)
        if len(code_rows) != len(quantization_rows):
            raise ValueError(
                "formal hard-code counter fields changed for %s" % method)
        if not code_rows:
            raise RuntimeError(
                "formal hard-code diagnostics are empty for %s" % method)
        if payload["weighted_zero_code_ratio"] is None or \
                payload["weighted_saturation_ratio"] is None:
            raise RuntimeError(
                "formal weighted hard-code diagnostics are empty for %s" %
                method)
        weighted = (
            float(payload["weighted_zero_code_ratio"]),
            float(payload["weighted_saturation_ratio"]),
        )
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0
               for value in weighted):
            raise ValueError(
                "formal weighted hard-code diagnostics are invalid for %s" %
                method)
        for row in code_rows:
            values = (
                float(row["zero_code_rate"]),
                float(row["saturation_rate"]),
            )
            numel = int(row["numel"])
            zero_codes = int(row["zero_code_count"])
            saturated = int(row["saturation_count"])
            if numel <= 0 or zero_codes < 0 or saturated < 0 or \
                    zero_codes > numel or saturated > numel or any(
                    not math.isfinite(value) or value < 0.0 or value > 1.0
                    for value in values) or values != (
                        zero_codes / float(numel),
                        saturated / float(numel)):
                raise ValueError(
                    "formal hard-code diagnostics are invalid for %s" % method)
        for sample_index in index.evaluation_indices:
            current = tuple(
                (int(row["source_index"]), str(row["owner"]),
                 str(row["owner_kind"])) for row in code_rows
                if int(row["sample_index"]) == sample_index)
            if current != contract_identities:
                raise ValueError(
                    "formal hard-code owner coverage changed for %s" % method)
    elif owner_contract:
        raise ValueError("FP32 hard-code owner contract must be empty")


def build_method_summary(
        root: Path, methods, artifact_index: FormalArtifactIndex,
        aggregation: FormalAggregation, cost_contracts) -> Tuple[dict, ...]:
    methods = _validate_method_order(methods)
    if not isinstance(artifact_index, FormalArtifactIndex):
        raise TypeError("formal summary requires its artifact index")
    if not isinstance(aggregation, FormalAggregation):
        raise TypeError("formal summary requires prediction aggregation")
    indexed_costs = _validate_indexed_cost_contracts(
        cost_contracts, artifact_index)
    aggregate_rows = tuple(aggregation.aggregate_metrics)
    if tuple(str(row["method"]) for row in aggregate_rows) != methods:
        raise ValueError("prediction aggregate method order changed")
    expected_metrics = dict(
        (str(row["method"]), dict(row)) for row in aggregate_rows)
    rows = []
    expected_kinds = dict((method, "strict_ptq_manifest")
                          for method in PTQ_METHODS)
    expected_kinds.update(dict((method, "terminal_qat_checkpoint")
                               for method in QAT_METHODS))
    expected_kinds["fp32"] = "official_checkpoint"
    for method in methods:
        path = Path(root) / "methods" / method / "formal_run.json"
        if not path.is_file():
            raise FileNotFoundError(
                "completed formal run is missing for %s: %s" %
                (method, path))
        payload = json.loads(path.read_text(encoding="utf-8"))
        if set(payload) != FORMAL_RUN_FIELDS:
            raise KeyError("formal run fields changed for %s" % method)
        if int(payload["format_version"]) != 1 or \
                int(payload["complete"]) != 1:
            raise ValueError("formal run is incomplete for %s" % method)
        if str(payload["method"]) != method:
            raise ValueError("formal run method identity changed")
        if str(payload["artifact_kind"]) != expected_kinds[method]:
            raise ValueError("formal run artifact kind changed")
        model = str(payload["model"])
        indices = tuple(int(index) for index in payload["evaluation_indices"])
        if model != artifact_index.model:
            raise ValueError("formal run model identity changed")
        if indices != artifact_index.evaluation_indices:
            raise ValueError("formal run evaluation indices changed")
        if str(payload["evaluation_identity"]) != \
                artifact_index.evaluation_identity:
            raise ValueError("formal run evaluation identity changed")
        if Path(payload["artifact_index"]) != \
                artifact_index.source or str(
                    payload["artifact_index_sha256"]) != \
                artifact_index.fingerprint:
            raise ValueError("formal run artifact index changed")
        entry = artifact_index.methods[method]
        artifact = Path(payload["artifact"])
        diagnostics = Path(payload["diagnostics"])
        predictions = Path(payload["prediction_directory"])
        if artifact != entry.artifact.resolve() or str(
                payload["artifact_sha256"]) != entry.artifact_sha256:
            raise ValueError("formal run primary artifact changed")
        expected_support = tuple({
            "name": name,
            "path": str(support_path.resolve()),
            "sha256": sha256,
        } for name, support_path, sha256 in entry.supporting_artifacts)
        _require_exact_finite_equal(
            payload["supporting_artifacts"], expected_support,
            "%s supporting artifacts" % method)
        expected_diagnostics = Path(root) / "methods" / method / \
            "diagnostics.json"
        expected_predictions = Path(root) / "predictions" / method
        if diagnostics != expected_diagnostics.resolve() or \
                not diagnostics.is_file():
            raise FileNotFoundError(
                "formal diagnostics are missing: %s" % diagnostics)
        if predictions != expected_predictions.resolve() or \
                not predictions.is_dir():
            raise FileNotFoundError(
                "formal predictions are missing: %s" % predictions)
        _validate_formal_diagnostics(diagnostics, artifact_index, method)
        _require_exact_finite_equal(
            payload["metrics"], expected_metrics[method],
            "%s manifest metrics" % method)
        indexed = indexed_costs[method]
        _require_exact_finite_equal(
            payload["assignment"], indexed["assignment"],
            "%s indexed assignment" % method)
        _require_exact_finite_equal(
            payload["cost_basis"], indexed["cost_basis"],
            "%s indexed cost basis" % method)
        _require_exact_finite_equal(
            payload["cost"], indexed["cost"],
            "%s manifest cost" % method)
        authoritative = dict(payload)
        authoritative["metrics"] = dict(expected_metrics[method])
        authoritative["assignment"] = indexed["assignment"]
        authoritative["cost_basis"] = indexed["cost_basis"]
        authoritative["cost"] = indexed["cost"]
        rows.append(authoritative)
    return tuple(rows)


def write_method_summary(
        root: Path, artifact_index: FormalArtifactIndex,
        aggregation: FormalAggregation, rows) -> None:
    rows = tuple(rows)
    if tuple(str(row["method"]) for row in rows) != SELECTED_METHODS:
        raise ValueError("formal method summary order changed")
    cost_rows = tuple(dict(row["cost"]) for row in rows)
    diagnostic_rows = tuple({
        "model": str(row["model"]),
        "method": str(row["method"]),
        "configuration": METHOD_LABELS[str(row["method"])],
        "diagnostics": str(row["diagnostics"]),
    } for row in rows)
    _write_csv(Path(root) / "cost_table.csv", cost_rows)
    _write_csv(Path(root) / "diagnostics_index.csv", diagnostic_rows)
    _write_json(Path(root) / "selected_method_summary.json", {
        "format_version": 1,
        "model": artifact_index.model,
        "artifact_index": str(artifact_index.source),
        "artifact_index_sha256": artifact_index.fingerprint,
        "methods": list(SELECTED_METHODS),
        "configurations": list(SELECTED_CONFIGURATION_LABELS),
        "evaluation_indices": list(artifact_index.evaluation_indices),
        "evaluation_identity": artifact_index.evaluation_identity,
        "primary_metric": "pooled_rmse",
        "diagnostic_metric": "mean_sample_rmse",
        "aggregate_metrics": aggregation.aggregate_metrics,
        "relative_fp_loss": aggregation.relative_fp_loss,
        "runs": list(rows),
    })


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate strict selected NYU quantization artifacts")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    worker = subparsers.add_parser("evaluate")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--launch-spec", type=Path, required=True)
    worker.add_argument("--model", required=True)
    worker.add_argument("--method", choices=SELECTED_METHODS, required=True)
    worker.add_argument("--artifact-index", type=Path, required=True)
    worker.add_argument("--output-root", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--config", type=Path, required=True)
    aggregate.add_argument("--launch-spec", type=Path, required=True)
    aggregate.add_argument("--model", required=True)
    aggregate.add_argument("--artifact-index", type=Path, required=True)
    aggregate.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.operation == "evaluate":
        run_formal_method(
            config=args.config,
            launch_spec=args.launch_spec,
            model=args.model,
            method=args.method,
            artifact_index=args.artifact_index,
            output_root=args.output_root,
        )
        return
    from spn_quant.experiment_config import load_selected_quantization_config
    selected = load_selected_quantization_config(args.config)
    model_config = _model_config(selected, args.model)
    index = load_formal_artifact_index(
        args.artifact_index, args.model, model_config.evaluation_indices)
    result = aggregate_prediction_exports(
        args.output_root, index)
    cost_contracts = load_indexed_cost_contracts(
        index, selected, model_config, args.config, args.launch_spec)
    rows = build_method_summary(
        args.output_root,
        SELECTED_METHODS,
        index,
        result,
        cost_contracts,
    )
    write_aggregation_tables(args.output_root, result)
    write_method_summary(args.output_root, index, result, rows)


if __name__ == "__main__":
    main()
