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
PREDICTION_FIELDS = frozenset((
    "format_version",
    "model",
    "method",
    "sample_index",
    "evaluation_identity",
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
    "evaluation_indices",
    "evaluation_identity",
    "prediction_directory",
    "metrics",
    "cost",
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
    model: str
    evaluation_indices: Tuple[int, ...]
    evaluation_identity: str
    ptq_matrix: Path
    methods: Mapping[str, MethodArtifact]
    preparation: Mapping[str, object]


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


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
    expected = str(payload["sha256"])
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


def _validated_p3_inputs(index, selected, model_config, contract):
    from scripts import train_nyu_selected_qat as qat_runner

    path = _supporting_path(
        index.methods["p3_t3_mixed_ptq"], "p3_t3_assignment")
    assignment, costs, evidence = qat_runner.load_p3_t3_qat_assignment(
        path,
        contract,
        selected.method_hyperparameters["p3_t3_mixed_ptq"],
        model_config.checkpoint,
        model_config.evaluation_indices,
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
        index, selected, model_config, runtime, model, contract):
    entry = index.methods["fp32"]
    if entry.artifact.resolve() != model_config.checkpoint.resolve():
        raise ValueError("FP32 artifact differs from configured checkpoint")
    p3_path, p3_assignment, costs = _validated_p3_inputs(
        index, selected, model_config, contract)
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
        method, index, selected, model_config, runtime, model, contract):
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
            index, selected, model_config, contract)
        del p3_path, p3_assignment
        ready = True
    finally:
        if not ready:
            evaluator.close()
            runtime.close()

    def close():
        evaluator.close()
        runtime.close()

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
        diagnostics_sources=(evaluator.instrumentor,),
        propagation=evaluator.propagation_adapter,
        closer=close,
    )


def _prepare_qdrop_deployment(
        method, index, selected, model_config, runtime, model, contract):
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
            index, selected, model_config, contract)
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


def _qat_args(method, entry):
    support = dict((name, path)
                   for name, path, sha256 in entry.supporting_artifacts)
    return Namespace(
        method=method,
        hawq_assignment=support["hawq_assignment"]
            if method == "hawq_mixed_le6" else None,
        hawq_trace_artifact=support["hawq_trace_artifact"]
            if method == "hawq_mixed_le6" else None,
        p3_t3_assignment=support["p3_t3_assignment"]
            if method == "mixed_task_aware" else None,
    )


def _prepare_qat_deployment(
        method, index, selected, model_config):
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
        _qat_args(method, entry), selected, model_config, training)
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
        _, _, costs = _validated_p3_inputs(
            index, selected, model_config, prepared.contract)
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
        diagnostics_sources=(),
        propagation=materialized.propagation,
        closer=close,
    )


def prepare_formal_deployment(
        method: str, index: FormalArtifactIndex,
        selected, model_config) -> FormalDeployment:
    """Prepare only a fresh FP32 model or a strict hard deployment."""
    method = str(method)
    if method not in SELECTED_METHODS:
        raise ValueError("unsupported selected formal method: %s" % method)
    from scripts.nyu_model_runtime import NYUModelRuntime
    from spn_quant.model_contracts import build_model_quantization_contract

    if method in QAT_METHODS:
        return _prepare_qat_deployment(
            method, index, selected, model_config)
    runtime = NYUModelRuntime.from_config(model_config)
    ready = False
    try:
        model = runtime.build_model(runtime.device)
        contract = build_model_quantization_contract(
            model_config.model, model)
        if method == "fp32":
            deployment = _prepare_fp32_deployment(
                index, selected, model_config, runtime, model, contract)
        elif method in ("rtn_w8a8", "rtn_w4a4", "p3_t3_mixed_ptq"):
            deployment = _prepare_rtn_deployment(
                method, index, selected, model_config,
                runtime, model, contract)
        else:
            deployment = _prepare_qdrop_deployment(
                method, index, selected, model_config,
                runtime, model, contract)
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


def write_prediction_export(
        *, root: Path, model: str, method: str, sample_index: int,
        evaluation_identity: str, rgb, sparse, gt, pred) -> Path:
    record = {
        "sample_index": int(sample_index),
        "gt": np.asarray(gt, dtype=np.float32),
        "pred": np.asarray(pred, dtype=np.float32),
    }
    _, target, prediction, valid = _validated_arrays(record)
    rgb_array = np.asarray(rgb, dtype=np.float32)
    sparse_array = np.asarray(sparse, dtype=np.float32)
    if rgb_array.shape != target.shape + (3,) or \
            sparse_array.shape != target.shape:
        raise ValueError("RGB, sparse depth, and dense depth shapes differ")
    if not bool(np.isfinite(rgb_array).all()) or not bool(
            np.isfinite(sparse_array).all()):
        raise FloatingPointError("RGB and sparse depth must be finite")
    if float(rgb_array.min()) < 0.0 or float(rgb_array.max()) > 1.0:
        raise ValueError("display RGB must lie in [0, 1]")
    if not isinstance(evaluation_identity, str) or \
            len(evaluation_identity) != 64:
        raise ValueError("evaluation identity must be a SHA256 string")
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
        expected_index: int, expected_identity: str) -> dict:
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
    _, gt, pred, valid = _validated_arrays(payload)
    if payload["rgb"].shape != gt.shape + (3,) or \
            payload["sparse"].shape != gt.shape:
        raise ValueError("prediction export input shapes changed")
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
        ("pooled_irmse", ""),
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
        root: Path, model: str, indices: Sequence[int],
        methods=SELECTED_METHODS) -> FormalAggregation:
    methods = _validate_method_order(methods)
    indices = tuple(int(index) for index in indices)
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
        for index, path in zip(indices, expected_paths):
            payload = load_prediction_export(
                path, model, method, index, identity)
            aligned = (
                payload["rgb"], payload["sparse"], payload["gt"],
                payload["valid_gt"],
            )
            if method == "fp32":
                reference_inputs[index] = tuple(
                    value.copy() for value in aligned)
            elif any(not np.array_equal(value, reference, equal_nan=True)
                     for value, reference in zip(
                         aligned, reference_inputs[index])):
                raise ValueError(
                    "prediction aligned input differs for %s sample %d" %
                    (method, index))
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


def _diagnostic_statistics(deployment):
    activation_bits = dict(deployment.assignment["activation_bits"])
    rows = []
    for source_index, source in enumerate(deployment.diagnostics_sources):
        for row in source.statistics():
            current = dict(row)
            owner = (str(current["module"]), str(current["role"])) \
                if "role" in current else None
            if owner in activation_bits:
                current["bits"] = activation_bits[owner]
            current["source_index"] = source_index
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


def evaluate_formal_deployment(
        method: str, deployment: FormalDeployment,
        index: FormalArtifactIndex, root: Path) -> dict:
    """Evaluate a prepared hard context and publish its completion last."""
    from scripts.run_nyu_model_p3t3_search import _propagation_valid
    from scripts.run_nyu_rtn_quantization import (
        batch_from_sample,
        seeded_sample,
    )

    method = str(method)
    if method != index.methods[method].method:
        raise ValueError("formal deployment method identity changed")
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
    deployment.model.eval()
    with torch.no_grad():
        for sample_index in index.evaluation_indices:
            sample = seeded_sample(dataset, sample_index, seed)
            batch = batch_from_sample(sample)
            model_input, target = deployment.runtime.model_input(
                batch, deployment.runtime.device)
            prediction = deployment.runtime.prediction(
                deployment.model(*model_input))
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
                if not _propagation_valid(current_propagation):
                    raise RuntimeError(
                        "formal propagation invariants failed: %s sample %d" %
                        (method, sample_index))
                propagation_rows.extend(dict(
                    row,
                    sample_index=int(sample_index),
                ) for row in current_propagation)
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
    quantization_statistics = _diagnostic_statistics(deployment)
    _write_json(diagnostics_path, {
        "format_version": 1,
        "model": index.model,
        "method": method,
        "hard_deployment": int(method != "fp32"),
        "samples": len(records),
        "prediction_finite_ratio": 1.0,
        "prediction_nonpositive_ratio": aggregation.nonpositive_ratio,
        "weighted_saturation_ratio": _weighted_diagnostic_ratio(
            quantization_statistics, "saturation_rate"),
        "weighted_zero_code_ratio": _weighted_diagnostic_ratio(
            quantization_statistics, "zero_code_rate"),
        "quantization_statistics": quantization_statistics,
        "propagation_statistics": propagation_rows,
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
        "evaluation_indices": list(index.evaluation_indices),
        "evaluation_identity": index.evaluation_identity,
        "prediction_directory": str(prediction_root.resolve()),
        "metrics": aggregate_row,
        "cost": cost,
        "diagnostics": str(diagnostics_path.resolve()),
    }
    _write_json(method_root / "formal_run.json", formal_run)
    return formal_run


def run_formal_method(
        *, config: Path, model: str, method: str,
        artifact_index: Path, output_root: Path) -> dict:
    from spn_quant.experiment_config import load_selected_quantization_config

    selected = load_selected_quantization_config(config)
    model_config = _model_config(selected, model)
    index = load_formal_artifact_index(
        artifact_index, model, model_config.evaluation_indices)
    deployment = prepare_formal_deployment(
        method, index, selected, model_config)
    try:
        return evaluate_formal_deployment(
            method, deployment, index, output_root)
    finally:
        deployment.close()


def build_method_summary(
        root: Path, methods, expected_model=None,
        expected_indices=None) -> Tuple[dict, ...]:
    methods = _validate_method_order(methods)
    expected = None if expected_indices is None else tuple(
        int(index) for index in expected_indices)
    expected_identity = None if expected is None else \
        ordered_evaluation_identity(expected)
    rows = []
    observed_model = None
    observed_indices = None
    observed_identity = None
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
        if len(indices) != EXPECTED_EVALUATION_SAMPLES or \
                len(indices) != len(set(indices)):
            raise ValueError("formal run requires 64 unique identities")
        identity = ordered_evaluation_identity(indices)
        if str(payload["evaluation_identity"]) != identity:
            raise ValueError("formal run evaluation identity changed")
        if expected_model is not None and model != str(expected_model):
            raise ValueError("formal run model identity changed")
        if expected is not None and indices != expected:
            raise ValueError("formal run evaluation indices changed")
        if expected_identity is not None and identity != expected_identity:
            raise ValueError("formal run evaluation identity changed")
        if observed_model is None:
            observed_model = model
            observed_indices = indices
            observed_identity = identity
        elif (model, indices, identity) != (
                observed_model, observed_indices, observed_identity):
            raise ValueError("formal run identities differ across methods")
        artifact = Path(payload["artifact"])
        diagnostics = Path(payload["diagnostics"])
        predictions = Path(payload["prediction_directory"])
        if not artifact.is_file():
            raise FileNotFoundError("source artifact is missing: %s" % artifact)
        if file_sha256(artifact) != str(payload["artifact_sha256"]):
            raise RuntimeError("source artifact fingerprint changed")
        if not diagnostics.is_file():
            raise FileNotFoundError(
                "formal diagnostics are missing: %s" % diagnostics)
        if not predictions.is_dir():
            raise FileNotFoundError(
                "formal predictions are missing: %s" % predictions)
        rows.append(payload)
    return tuple(rows)


def write_method_summary(root: Path, rows) -> None:
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
        "model": str(rows[0]["model"]),
        "methods": list(SELECTED_METHODS),
        "configurations": list(SELECTED_CONFIGURATION_LABELS),
        "evaluation_indices": list(rows[0]["evaluation_indices"]),
        "evaluation_identity": str(rows[0]["evaluation_identity"]),
        "primary_metric": "pooled_rmse",
        "diagnostic_metric": "mean_sample_rmse",
        "runs": list(rows),
    })


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate strict selected NYU quantization artifacts")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    worker = subparsers.add_parser("evaluate")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--model", required=True)
    worker.add_argument("--method", choices=SELECTED_METHODS, required=True)
    worker.add_argument("--artifact-index", type=Path, required=True)
    worker.add_argument("--output-root", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--config", type=Path, required=True)
    aggregate.add_argument("--model", required=True)
    aggregate.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.operation == "evaluate":
        run_formal_method(
            config=args.config,
            model=args.model,
            method=args.method,
            artifact_index=args.artifact_index,
            output_root=args.output_root,
        )
        return
    from spn_quant.experiment_config import load_selected_quantization_config
    selected = load_selected_quantization_config(args.config)
    model_config = _model_config(selected, args.model)
    result = aggregate_prediction_exports(
        args.output_root, args.model, model_config.evaluation_indices)
    write_aggregation_tables(args.output_root, result)
    rows = build_method_summary(
        args.output_root,
        SELECTED_METHODS,
        expected_model=args.model,
        expected_indices=model_config.evaluation_indices,
    )
    write_method_summary(args.output_root, rows)


if __name__ == "__main__":
    main()
