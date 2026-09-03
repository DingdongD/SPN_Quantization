#!/usr/bin/env python3
"""Build and execute the strict three-model selected-quantization DAG."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.evaluate_nyu_selected_quantization import (  # noqa: E402
    PTQ_METHODS,
    QAT_METHODS,
    SELECTED_METHODS,
    file_sha256 as evaluation_file_sha256,
    load_formal_artifact_index,
    ordered_evaluation_identity,
)
from spn_quant.experiment_config import (  # noqa: E402
    MODEL_ORDER,
    SelectedQuantizationConfig,
    load_selected_quantization_config,
)
from spn_quant.nyu_static_inputs import (  # noqa: E402
    StaticInputPaths,
    file_sha256 as static_file_sha256,
    validate_static_input_bundle,
)


LAUNCH_SPEC_FIELDS = frozenset((
    "format_version",
    "orchestrator_python",
    "orchestrator_device",
    "orchestrator_environment",
    "model_environments",
    "qdrop_config",
    "static_inputs",
    "model_inputs",
    "p3_t3_budgets",
    "p3_t3_policy",
    "hard_deployment",
    "qat",
))
STATIC_INPUT_SETTING_FIELDS = frozenset((
    "calibration_seed",
    "evaluation_seed",
    "candidate_samples",
    "tail_samples",
))
MODEL_INPUT_FIELDS = frozenset((
    "calibration_indices",
    "evaluation_protocol",
    "weight_cost_rows",
    "activation_cost_rows",
))
P3_T3_BUDGET_FIELDS = frozenset((
    "maximum_normalized_weight_cost",
    "maximum_normalized_activation_cost",
))
P3_T3_POLICY_FIELDS = frozenset((
    "metric_aggregation",
    "maximum_relative_rmse_loss",
))
HARD_DEPLOYMENT_FIELDS = frozenset((
    "fold_conv_bn",
    "fold_max_error",
    "joint_clip_factors",
    "joint_search_rounds",
    "joint_cache_sample_limit",
    "joint_cache_byte_limit",
))
QAT_SETTING_FIELDS = frozenset((
    "epochs",
    "batch_size",
    "validation_batch_size",
    "workers",
    "learning_rate",
    "momentum",
    "weight_decay",
    "scheduler_factor",
    "scheduler_patience",
    "scheduler_threshold",
    "scheduler_min_lr",
    "max_gradient_norm",
    "patience",
    "min_relative_improvement",
    "seed",
    "hawq_range_momentum",
    "depth_loss_weight",
    "boundary_loss_weight",
    "teacher_loss_weight",
    "initial_depth_loss_weight",
    "propagation_loss_weight",
    "boundary_threshold_m",
    "log_interval",
))
LAUNCH_QAT_ORDER = (
    "lsqplus_w4a4",
    "lsqplus_w6a6",
    "hawq_mixed_le6",
    "mixed_task_aware",
)
QAT_ENVIRONMENT = (("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),)
QAT_CLI_FIELDS = (
    ("epochs", "--epochs"),
    ("batch_size", "--batch-size"),
    ("validation_batch_size", "--validation-batch-size"),
    ("workers", "--workers"),
    ("learning_rate", "--learning-rate"),
    ("momentum", "--momentum"),
    ("weight_decay", "--weight-decay"),
    ("scheduler_factor", "--scheduler-factor"),
    ("scheduler_patience", "--scheduler-patience"),
    ("scheduler_threshold", "--scheduler-threshold"),
    ("scheduler_min_lr", "--scheduler-min-lr"),
    ("max_gradient_norm", "--max-gradient-norm"),
    ("patience", "--patience"),
    ("min_relative_improvement", "--min-relative-improvement"),
    ("seed", "--seed"),
    ("hawq_range_momentum", "--hawq-range-momentum"),
    ("depth_loss_weight", "--depth-loss-weight"),
    ("boundary_loss_weight", "--boundary-loss-weight"),
    ("teacher_loss_weight", "--teacher-loss-weight"),
    ("initial_depth_loss_weight", "--initial-depth-loss-weight"),
    ("propagation_loss_weight", "--propagation-loss-weight"),
    ("boundary_threshold_m", "--boundary-threshold-m"),
    ("log_interval", "--log-interval"),
)
OUTPUT_POLICIES = frozenset((
    "existing_empty_parent",
    "absent_parent",
    "command_file",
    "command_tree",
))
JOB_MANIFEST_FIELDS = frozenset((
    "format_version",
    "job_id",
    "model",
    "name",
    "kind",
    "method",
    "device",
    "command",
    "environment",
    "inputs",
    "input_revisions",
    "output_path",
    "produced_outputs",
    "dependencies",
    "log_path",
    "state",
    "start_time_utc",
    "end_time_utc",
    "exit_status",
))


@dataclass(frozen=True)
class ModelLaunchInputs:
    calibration_indices: Path
    evaluation_protocol: Path
    weight_cost_rows: Path
    activation_cost_rows: Path


@dataclass(frozen=True)
class LaunchSpec:
    source: Path
    orchestrator_python: Path
    orchestrator_device: str
    orchestrator_environment: Mapping[str, str]
    model_environments: Mapping[str, Mapping[str, str]]
    qdrop_config: Path
    static_inputs: Mapping[str, int]
    model_inputs: Mapping[str, ModelLaunchInputs]
    p3_t3_budgets: Mapping[str, Mapping[str, float]]
    p3_t3_policy: Mapping[str, object]
    hard_deployment: Mapping[str, object]
    qat: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class LaunchConfiguration:
    config_path: Path
    experiment: SelectedQuantizationConfig
    spec: LaunchSpec


@dataclass(frozen=True)
class LaunchJob:
    job_id: str
    model: Optional[str]
    name: str
    kind: str
    method: Optional[str]
    device: str
    command: Tuple[str, ...]
    environment: Mapping[str, str]
    inputs: Tuple[Path, ...]
    output: Path
    dependencies: Tuple[str, ...]
    output_policy: str
    produced_outputs: Tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if not self.job_id or not self.name or not self.kind:
            raise ValueError("launch job identity fields must be nonempty")
        _indexed_cuda_device(self.device, "launch job")
        if not self.command or not Path(self.command[0]).is_absolute():
            raise ValueError("launch job Python command must be absolute")
        if "CUDA_VISIBLE_DEVICES" in self.environment:
            raise ValueError("CUDA_VISIBLE_DEVICES remapping is forbidden")
        if not self.output.is_absolute():
            raise ValueError("launch job output must be absolute")
        if self.output_policy not in OUTPUT_POLICIES:
            raise ValueError("unsupported launch output policy")
        if len(self.dependencies) != len(set(self.dependencies)):
            raise ValueError("launch job dependencies contain duplicates")
        if any(not path.is_absolute() for path in self.inputs):
            raise ValueError("launch job inputs must be absolute")


class JobExecutionError(RuntimeError):
    pass


class LaunchGraph(object):
    def __init__(self, jobs: Sequence[LaunchJob]) -> None:
        self.jobs = tuple(jobs)
        self._by_id = dict((job.job_id, job) for job in self.jobs)
        if len(self._by_id) != len(self.jobs):
            raise ValueError("launch graph job identifiers are not unique")
        for job in self.jobs:
            missing = tuple(
                dependency for dependency in job.dependencies
                if dependency not in self._by_id)
            if missing:
                raise KeyError("launch graph dependencies are missing: %s" %
                               (missing,))
            if job.job_id in job.dependencies:
                raise ValueError("launch graph contains a self dependency")
        output_owners = {}
        for job in self.jobs:
            for path in _declared_outputs(job):
                resolved = path.resolve()
                if resolved in output_owners:
                    raise ValueError(
                        "launch output has multiple producers: %s" % resolved)
                output_owners[resolved] = job.job_id
        self._topological_order = self._validate_acyclic()
        self._validate_generated_inputs(output_owners)

    @property
    def models(self) -> Tuple[str, ...]:
        return tuple(
            model for model in MODEL_ORDER
            if any(job.model == model for job in self.jobs))

    def _validate_acyclic(self) -> Tuple[str, ...]:
        pending = dict(
            (job.job_id, set(job.dependencies)) for job in self.jobs)
        order = []
        while pending:
            ready = tuple(
                job.job_id for job in self.jobs
                if job.job_id in pending and not pending[job.job_id])
            if not ready:
                raise ValueError("launch graph contains a dependency cycle")
            for job_id in ready:
                del pending[job_id]
                order.append(job_id)
            for dependencies in pending.values():
                dependencies.difference_update(ready)
        return tuple(order)

    def topological_jobs(self) -> Tuple[LaunchJob, ...]:
        return tuple(self._by_id[job_id]
                     for job_id in self._topological_order)

    def _validate_generated_inputs(self, output_owners) -> None:
        ancestors = {}
        for job_id in self._topological_order:
            dependencies = set(self._by_id[job_id].dependencies)
            for dependency in tuple(dependencies):
                dependencies.update(ancestors[dependency])
            ancestors[job_id] = dependencies
        for job in self.jobs:
            for source in job.inputs:
                resolved = source.resolve()
                owner = output_owners[resolved] \
                    if resolved in output_owners else None
                if owner is not None and owner not in ancestors[job.job_id]:
                    raise ValueError(
                        "generated input lacks dependency: %s <- %s" %
                        (job.job_id, owner))

    def jobs_for_model(self, model: str) -> Tuple[LaunchJob, ...]:
        return tuple(job for job in self.topological_jobs()
                     if job.model == str(model))

    def predecessors(
            self, name: str, model: Optional[str] = None):
        if model is not None:
            job_id = "%s:%s" % (model, name)
            if job_id not in self._by_id:
                raise KeyError("launch graph job is missing: %s" % job_id)
            return set(_local_job_name(dependency)
                       for dependency in self._by_id[job_id].dependencies)
        matches = tuple(job for job in self.jobs if job.name == str(name))
        if not matches:
            raise KeyError("launch graph job name is missing: %s" % name)
        if len(matches) == 1 and matches[0].model is None:
            return set(matches[0].dependencies)
        predecessor_sets = tuple(
            frozenset(_local_job_name(dependency)
                      for dependency in job.dependencies)
            for job in matches)
        if len(set(predecessor_sets)) != 1:
            raise ValueError("model launch dependencies differ: %s" % name)
        return set(predecessor_sets[0])


def _local_job_name(job_id: str) -> str:
    separator = job_id.find(":")
    return job_id[separator + 1:] if separator >= 0 else job_id


def _freeze_string_mapping(payload, family: str):
    if not isinstance(payload, dict) or not payload:
        raise ValueError("%s environment must be a nonempty object" % family)
    if "CUDA_VISIBLE_DEVICES" in payload:
        raise ValueError("CUDA_VISIBLE_DEVICES remapping is forbidden")
    values = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key or not isinstance(value, str):
            raise TypeError("%s environment must contain strings" % family)
        values[key] = value
    return MappingProxyType(values)


def _absolute_path(value, family: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("%s path must be absolute" % family)
    return path


def _indexed_cuda_device(value, family: str) -> str:
    device = str(value)
    if not device.startswith("cuda:") or not device[5:].isdigit():
        raise ValueError("%s device must be indexed CUDA" % family)
    return device


def _executable_path(value, family: str) -> Path:
    path = _absolute_path(value, family)
    if not path.is_file() or not os.access(str(path), os.X_OK):
        raise FileNotFoundError("%s is not executable: %s" % (family, path))
    return path


def _finite_positive(value, family: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("%s must be finite and positive" % family)
    return result


def _load_launch_spec(
        path: Path, experiment: SelectedQuantizationConfig) -> LaunchSpec:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if set(payload) != LAUNCH_SPEC_FIELDS:
        raise KeyError("three-model launch specification fields changed")
    version = payload["format_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError("three-model launch specification version changed")
    orchestrator_python = _executable_path(
        payload["orchestrator_python"], "orchestrator Python")
    orchestrator_device = _indexed_cuda_device(
        payload["orchestrator_device"], "orchestrator")
    orchestrator_environment = _freeze_string_mapping(
        payload["orchestrator_environment"], "orchestrator")
    if tuple(payload["model_environments"]) != MODEL_ORDER:
        raise ValueError("model environment order changed")
    model_config = dict((row.model, row) for row in experiment.models)
    evaluation_identities = tuple(
        tuple(model_config[model].evaluation_indices)
        for model in MODEL_ORDER)
    if any(len(indices) != 64 or len(indices) != len(set(indices))
           for indices in evaluation_identities):
        raise ValueError("three-model evaluation identities must be 64 unique samples")
    if len(set(evaluation_identities)) != 1:
        raise ValueError("three-model evaluation identities must be shared")
    environments = {}
    required_environment = frozenset((
        "PYTHONHASHSEED",
        "PYTHONPATH",
        "SPN_DATA_ROOT",
        "SPN_EXTERNAL_ROOT",
        "COMPLETIONFORMER_ROOT",
    ))
    for model in MODEL_ORDER:
        environment = _freeze_string_mapping(
            payload["model_environments"][model], model)
        if not required_environment.issubset(environment):
            raise KeyError("%s model environment is incomplete" % model)
        for name in (
                "PYTHONPATH", "SPN_DATA_ROOT", "SPN_EXTERNAL_ROOT",
                "COMPLETIONFORMER_ROOT"):
            path_value = _absolute_path(
                environment[name], "%s %s" % (model, name))
            if name != "PYTHONPATH" and not path_value.is_dir():
                raise FileNotFoundError(
                    "%s %s directory is missing: %s" %
                    (model, name, path_value))
        if Path(environment["SPN_DATA_ROOT"]) != model_config[model].data_root:
            raise ValueError("%s SPN_DATA_ROOT differs from config" % model)
        environments[model] = environment
    if tuple(payload["model_inputs"]) != MODEL_ORDER:
        raise ValueError("model launch input order changed")
    model_inputs = {}
    for model in MODEL_ORDER:
        row = payload["model_inputs"][model]
        if set(row) != MODEL_INPUT_FIELDS:
            raise KeyError("%s launch input fields changed" % model)
        model_inputs[model] = ModelLaunchInputs(
            calibration_indices=_absolute_path(
                row["calibration_indices"], "%s calibration indices" % model),
            evaluation_protocol=_absolute_path(
                row["evaluation_protocol"], "%s evaluation protocol" % model),
            weight_cost_rows=_absolute_path(
                row["weight_cost_rows"], "%s weight costs" % model),
            activation_cost_rows=_absolute_path(
                row["activation_cost_rows"],
                "%s activation costs" % model),
        )
    if tuple(payload["p3_t3_budgets"]) != MODEL_ORDER:
        raise ValueError("P3/T3 model budget order changed")
    budgets = {}
    for model in MODEL_ORDER:
        row = payload["p3_t3_budgets"][model]
        if set(row) != P3_T3_BUDGET_FIELDS:
            raise KeyError("%s P3/T3 budget fields changed" % model)
        budgets[model] = MappingProxyType(dict(
            (name, _finite_positive(row[name], "%s %s" % (model, name)))
            for name in P3_T3_BUDGET_FIELDS))
    p3_policy = payload["p3_t3_policy"]
    if set(p3_policy) != P3_T3_POLICY_FIELDS:
        raise KeyError("P3/T3 selection policy fields changed")
    if p3_policy["metric_aggregation"] != "mean_of_per_sample_rmse":
        raise ValueError("P3/T3 metric aggregation policy changed")
    maximum_relative_rmse_loss = float(
        p3_policy["maximum_relative_rmse_loss"])
    if not math.isfinite(maximum_relative_rmse_loss) or \
            maximum_relative_rmse_loss < 0.0:
        raise ValueError("P3/T3 relative RMSE loss policy is invalid")
    hard = payload["hard_deployment"]
    if set(hard) != HARD_DEPLOYMENT_FIELDS:
        raise KeyError("hard-deployment launch fields changed")
    if not isinstance(hard["fold_conv_bn"], bool):
        raise TypeError("fold_conv_bn must be boolean")
    normalized_hard = {
        "fold_conv_bn": hard["fold_conv_bn"],
        "fold_max_error": float(hard["fold_max_error"]),
        "joint_clip_factors": tuple(
            float(value) for value in hard["joint_clip_factors"]),
        "joint_search_rounds": int(hard["joint_search_rounds"]),
        "joint_cache_sample_limit": int(hard["joint_cache_sample_limit"]),
        "joint_cache_byte_limit": int(hard["joint_cache_byte_limit"]),
    }
    if not math.isfinite(normalized_hard["fold_max_error"]) or \
            normalized_hard["fold_max_error"] < 0.0 or not \
            normalized_hard["joint_clip_factors"] or any(
                not math.isfinite(value) or value <= 0.0
                for value in normalized_hard["joint_clip_factors"]) or any(
                normalized_hard[name] <= 0 for name in (
                    "joint_search_rounds", "joint_cache_sample_limit",
                    "joint_cache_byte_limit")):
        raise ValueError("hard-deployment launch values are invalid")
    if tuple(payload["qat"]) != LAUNCH_QAT_ORDER:
        raise ValueError("selected QAT launch order changed")
    qat = {}
    for method in LAUNCH_QAT_ORDER:
        row = payload["qat"][method]
        if set(row) != QAT_SETTING_FIELDS:
            raise KeyError("%s QAT launch fields changed" % method)
        normalized = dict((name, row[name]) for name in QAT_SETTING_FIELDS)
        for name in (
                "epochs", "batch_size", "validation_batch_size",
                "scheduler_patience", "patience", "log_interval"):
            if isinstance(normalized[name], bool) or not isinstance(
                    normalized[name], int) or normalized[name] <= 0:
                raise ValueError("%s %s must be a positive integer" %
                                 (method, name))
        for name in ("workers", "seed"):
            if isinstance(normalized[name], bool) or not isinstance(
                    normalized[name], int) or normalized[name] < 0:
                raise ValueError("%s %s must be a nonnegative integer" %
                                 (method, name))
        for name in QAT_SETTING_FIELDS - frozenset((
                "epochs", "batch_size", "validation_batch_size", "workers",
                "scheduler_patience", "patience", "seed", "log_interval")):
            value = float(normalized[name])
            if not math.isfinite(value):
                raise ValueError("%s %s must be finite" % (method, name))
            normalized[name] = value
        qat[method] = MappingProxyType(normalized)
    qdrop_config = _absolute_path(
        payload["qdrop_config"], "QDrop config")
    if not qdrop_config.is_file():
        raise FileNotFoundError("QDrop config is missing: %s" % qdrop_config)
    static_values = payload["static_inputs"]
    if set(static_values) != STATIC_INPUT_SETTING_FIELDS:
        raise KeyError("static-input launch fields changed")
    static_inputs = {}
    for name in STATIC_INPUT_SETTING_FIELDS:
        value = static_values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("static-input %s must be nonnegative" % name)
        static_inputs[name] = int(value)
    if static_inputs["candidate_samples"] < 128 or \
            static_inputs["tail_samples"] != 32:
        raise ValueError("static-input selection dimensions changed")
    qdrop_payload = json.loads(qdrop_config.read_text(encoding="utf-8"))
    if int(qdrop_payload["formal"]["evaluation_seed"]) != \
            static_inputs["evaluation_seed"]:
        raise ValueError("static-input evaluation seed differs from QDrop")
    return LaunchSpec(
        source=source.resolve(),
        orchestrator_python=orchestrator_python,
        orchestrator_device=orchestrator_device,
        orchestrator_environment=orchestrator_environment,
        model_environments=MappingProxyType(environments),
        qdrop_config=qdrop_config,
        static_inputs=MappingProxyType(static_inputs),
        model_inputs=MappingProxyType(model_inputs),
        p3_t3_budgets=MappingProxyType(budgets),
        p3_t3_policy=MappingProxyType({
            "metric_aggregation": p3_policy["metric_aggregation"],
            "maximum_relative_rmse_loss": maximum_relative_rmse_loss,
        }),
        hard_deployment=MappingProxyType(normalized_hard),
        qat=MappingProxyType(qat),
    )


def load_launch_configuration(
        config_path: Path, launch_spec_path: Path) -> LaunchConfiguration:
    config_source = Path(config_path).resolve()
    experiment = load_selected_quantization_config(config_source)
    spec = _load_launch_spec(launch_spec_path, experiment)
    if len(set(model.device for model in experiment.models)) != len(MODEL_ORDER):
        raise ValueError("official model CUDA devices must be distinct")
    for model in experiment.models:
        _executable_path(
            model.python_executable, "%s model Python" % model.model)
    return LaunchConfiguration(
        config_path=config_source,
        experiment=experiment,
        spec=spec,
    )


def _script(name: str) -> str:
    return str((REPO_ROOT / "scripts" / name).resolve())


def _text(value) -> str:
    if isinstance(value, bool):
        raise TypeError("boolean launch value requires an explicit flag")
    return str(value)


def _deployment_arguments(spec: LaunchSpec) -> Tuple[str, ...]:
    settings = spec.hard_deployment
    values = [
        "--fold-conv-bn" if settings["fold_conv_bn"]
        else "--skip-conv-bn-fold",
        "--fold-max-error", _text(settings["fold_max_error"]),
        "--joint-clip-factors",
    ]
    values.extend(_text(value) for value in settings["joint_clip_factors"])
    values.extend((
        "--joint-search-rounds", _text(settings["joint_search_rounds"]),
        "--joint-cache-sample-limit",
        _text(settings["joint_cache_sample_limit"]),
        "--joint-cache-byte-limit",
        _text(settings["joint_cache_byte_limit"]),
    ))
    return tuple(values)


def _model_paths(configuration: LaunchConfiguration, model: str):
    root = configuration.experiment.output_root / model
    artifacts = root / "artifacts"
    p3_root = artifacts / "p3_t3_mixed_ptq"
    ptq_root = artifacts / "selected_ptq"
    trace_root = artifacts / "hawq_trace"
    allocation_root = artifacts / "hawq_allocation"
    qat_root = artifacts / "qat"
    formal_root = root / "formal"
    return {
        "root": root,
        "p3_assignment": p3_root / "p3_t3_assignment.json",
        "ptq_matrix": ptq_root / "selected_ptq_matrix.json",
        "ptq_root": ptq_root,
        "hawq_trace": trace_root / "hawq_trace_artifact.json",
        "hawq_assignment":
            allocation_root / "hawq_mixed_le6_assignment.json",
        "qat_root": qat_root,
        "formal_root": formal_root,
        "artifact_index": formal_root / "formal_artifacts.json",
        "static_validation": root / "static_inputs_validation.json",
    }


def _model_job(
        *, model_config, spec: LaunchSpec, name: str, kind: str,
        method: Optional[str], command, inputs, output: Path,
        dependencies=(), output_policy="command_file", produced_outputs=(),
        orchestrator=False, environment_overrides=()) -> LaunchJob:
    model = model_config.model
    python = spec.orchestrator_python if orchestrator \
        else model_config.python_executable
    base_environment = spec.orchestrator_environment if orchestrator \
        else spec.model_environments[model]
    environment = dict(base_environment)
    for key, value in environment_overrides:
        if key in environment:
            raise ValueError("launch job environment override is duplicated")
        environment[str(key)] = str(value)
    return LaunchJob(
        job_id="%s:%s" % (model, name),
        model=model,
        name=name,
        kind=kind,
        method=method,
        device=model_config.device,
        command=(str(python),) + tuple(str(value) for value in command),
        environment=MappingProxyType(environment),
        inputs=tuple(Path(path).resolve() for path in inputs),
        output=Path(output).resolve(),
        dependencies=tuple(
            dependency if ":" in dependency
            else "%s:%s" % (model, dependency)
            for dependency in dependencies),
        output_policy=output_policy,
        produced_outputs=tuple(
            Path(path).resolve() for path in produced_outputs),
    )


def _qat_command(
        configuration: LaunchConfiguration, model_config, method: str,
        output: Path, paths) -> Tuple[str, ...]:
    settings = configuration.spec.qat[method]
    command = [
        _script("train_nyu_selected_qat.py"),
        "--config", str(configuration.config_path),
        "--launch-spec", str(configuration.spec.source),
        "--model", model_config.model,
        "--method", method,
        "--device", model_config.device,
        "--output", str(output),
    ]
    if method == "hawq_mixed_le6":
        command.extend((
            "--hawq-assignment", str(paths["hawq_assignment"]),
            "--hawq-trace-artifact", str(paths["hawq_trace"]),
        ))
    elif method == "mixed_task_aware":
        command.extend((
            "--p3-t3-assignment", str(paths["p3_assignment"]),))
    for field, flag in QAT_CLI_FIELDS:
        command.extend((flag, _text(settings[field])))
    command.extend(_deployment_arguments(configuration.spec))
    return tuple(command)


def _formal_prediction_paths(root: Path, method: str, indices):
    return tuple(
        root / "predictions" / method / ("sample_%05d.npz" % index)
        for index in indices)


def _build_model_jobs(
        configuration: LaunchConfiguration, model_config) -> Tuple[LaunchJob, ...]:
    spec = configuration.spec
    model = model_config.model
    inputs = spec.model_inputs[model]
    paths = _model_paths(configuration, model)
    saved_args_path = model_config.run_dir / "args.json"
    saved_meta_path = model_config.run_dir / "meta.json"
    saved_args = json.loads(saved_args_path.read_text(encoding="utf-8"))
    train_list = Path(saved_args["train_list"]).resolve()
    evaluation_list = Path(saved_args["eval_list"]).resolve()
    static_outputs = (
        inputs.calibration_indices,
        inputs.evaluation_protocol,
        inputs.weight_cost_rows,
        inputs.activation_cost_rows,
    )
    preparation = _model_job(
        model_config=model_config,
        spec=spec,
        name="prepare_static_inputs",
        kind="static_input_generation",
        method=None,
        command=(
            _script("prepare_nyu_three_model_static_inputs.py"),
            "--config", str(configuration.config_path),
            "--model", model,
            "--device", model_config.device,
            "--calibration-seed",
            _text(spec.static_inputs["calibration_seed"]),
            "--evaluation-seed",
            _text(spec.static_inputs["evaluation_seed"]),
            "--candidate-samples",
            _text(spec.static_inputs["candidate_samples"]),
            "--tail-samples", _text(spec.static_inputs["tail_samples"]),
            "--calibration-metadata",
            str(model_config.calibration_metadata),
            "--calibration-indices", str(inputs.calibration_indices),
            "--evaluation-protocol", str(inputs.evaluation_protocol),
            "--weight-cost-rows", str(inputs.weight_cost_rows),
            "--activation-cost-rows", str(inputs.activation_cost_rows),
        ),
        inputs=(
            configuration.config_path,
            spec.source,
            model_config.checkpoint,
            saved_args_path,
            saved_meta_path,
            train_list,
            evaluation_list,
        ),
        output=model_config.calibration_metadata,
        output_policy="absent_parent",
        produced_outputs=static_outputs,
    )
    validation = _model_job(
        model_config=model_config,
        spec=spec,
        name="validate_static_inputs",
        kind="static_input_validation",
        method=None,
        command=(
            _script("launch_nyu_three_model_quantization.py"),
            "validate-static-inputs",
            "--config", str(configuration.config_path),
            "--launch-spec", str(spec.source),
            "--model", model,
            "--output", str(paths["static_validation"]),
        ),
        inputs=(
            configuration.config_path,
            spec.source,
            model_config.checkpoint,
            saved_args_path,
            saved_meta_path,
            train_list,
            evaluation_list,
            model_config.calibration_metadata,
        ) + static_outputs,
        output=paths["static_validation"],
        dependencies=("prepare_static_inputs",),
        output_policy="command_file",
        orchestrator=True,
    )
    common = (
        configuration.config_path,
        spec.source,
        model_config.checkpoint,
        model_config.calibration_metadata,
        paths["static_validation"],
    )
    deployment = _deployment_arguments(spec)
    budgets = spec.p3_t3_budgets[model]
    p3 = _model_job(
        model_config=model_config,
        spec=spec,
        name="p3_t3_mixed_ptq",
        kind="p3_t3_search",
        method="p3_t3_mixed_ptq",
        command=(
            _script("run_nyu_model_p3t3_search.py"),
            "--config", str(configuration.config_path),
            "--model", model,
            "--device", model_config.device,
            "--maximum-normalized-weight-cost",
            _text(budgets["maximum_normalized_weight_cost"]),
            "--maximum-normalized-activation-cost",
            _text(budgets["maximum_normalized_activation_cost"]),
            "--maximum-relative-rmse-loss",
            _text(spec.p3_t3_policy["maximum_relative_rmse_loss"]),
            "--weight-cost-rows", str(inputs.weight_cost_rows),
            "--activation-cost-rows", str(inputs.activation_cost_rows),
            "--output", str(paths["p3_assignment"].parent),
        ) + deployment,
        inputs=common + (
            inputs.weight_cost_rows, inputs.activation_cost_rows),
        output=paths["p3_assignment"],
        dependencies=("validate_static_inputs",),
        output_policy="existing_empty_parent",
    )
    ptq_manifests = tuple(
        paths["ptq_root"] / method / "hard_deployment_manifest.json"
        for method in PTQ_METHODS)
    ptq = _model_job(
        model_config=model_config,
        spec=spec,
        name="selected_ptq",
        kind="selected_ptq",
        method=None,
        command=(
            _script("run_nyu_selected_ptq.py"),
            "--config", str(configuration.config_path),
            "--model", model,
            "--device", model_config.device,
            "--qdrop-config", str(spec.qdrop_config),
            "--calibration-indices", str(inputs.calibration_indices),
            "--evaluation-protocol", str(inputs.evaluation_protocol),
            "--p3-t3-assignment", str(paths["p3_assignment"]),
            "--output", str(paths["ptq_root"]),
        ) + deployment,
        inputs=common + (
            spec.qdrop_config,
            inputs.calibration_indices,
            inputs.evaluation_protocol,
            paths["p3_assignment"],
        ),
        output=paths["ptq_matrix"],
        dependencies=("p3_t3_mixed_ptq",),
        output_policy="absent_parent",
        produced_outputs=ptq_manifests,
    )
    trace_settings = configuration.experiment.method_hyperparameters[
        "hawq_mixed_le6"]["trace"]
    trace = _model_job(
        model_config=model_config,
        spec=spec,
        name="hawq_trace",
        kind="hawq_trace",
        method="hawq_mixed_le6",
        command=(
            _script("run_nyu_model_hawq_trace.py"),
            "--phase", "trace",
            "--config", str(configuration.config_path),
            "--model", model,
            "--device", model_config.device,
            "--weight-cost-rows", str(inputs.weight_cost_rows),
            "--activation-cost-rows", str(inputs.activation_cost_rows),
            "--output", str(paths["hawq_trace"].parent),
            "--batch-size", _text(trace_settings["batch_size"]),
            "--probes-per-batch", _text(trace_settings["probes_per_batch"]),
            "--seed", _text(trace_settings["seed"]),
            "--depth-mse-weight", _text(trace_settings["depth_mse_weight"]),
            "--boundary-mse-weight",
            _text(trace_settings["boundary_mse_weight"]),
            "--boundary-threshold-m",
            _text(trace_settings["boundary_threshold_m"]),
        ),
        inputs=common + (
            inputs.weight_cost_rows, inputs.activation_cost_rows),
        output=paths["hawq_trace"],
        dependencies=("validate_static_inputs",),
        output_policy="existing_empty_parent",
    )
    allocation = _model_job(
        model_config=model_config,
        spec=spec,
        name="hawq_allocation",
        kind="hawq_allocation",
        method="hawq_mixed_le6",
        command=(
            _script("run_nyu_model_hawq_trace.py"),
            "--phase", "allocate",
            "--config", str(configuration.config_path),
            "--model", model,
            "--trace-artifact", str(paths["hawq_trace"]),
            "--output", str(paths["hawq_assignment"].parent),
        ),
        inputs=common + (paths["hawq_trace"],),
        output=paths["hawq_assignment"],
        dependencies=("hawq_trace",),
        output_policy="existing_empty_parent",
        orchestrator=True,
    )
    qat_jobs = []
    qat_names = {
        "lsqplus_w4a4": "lsqplus_w4a4_qat",
        "lsqplus_w6a6": "lsqplus_w6a6_qat",
        "hawq_mixed_le6": "hawq_mixed_le6_qat",
        "mixed_task_aware": "mixed_task_aware",
    }
    qat_dependencies = {
        "lsqplus_w4a4": ("validate_static_inputs",),
        "lsqplus_w6a6": ("validate_static_inputs",),
        "hawq_mixed_le6": ("hawq_allocation",),
        "mixed_task_aware": (
            "p3_t3_mixed_ptq", "validate_static_inputs"),
    }
    qat_inputs = {
        "lsqplus_w4a4": (),
        "lsqplus_w6a6": (),
        "hawq_mixed_le6": (
            paths["hawq_assignment"], paths["hawq_trace"]),
        "mixed_task_aware": (
            paths["p3_assignment"],
            inputs.weight_cost_rows,
            inputs.activation_cost_rows,
        ),
    }
    qat_outputs = {}
    for method in LAUNCH_QAT_ORDER:
        output = paths["qat_root"] / method / "final.pt"
        qat_outputs[method] = output
        qat_jobs.append(_model_job(
            model_config=model_config,
            spec=spec,
            name=qat_names[method],
            kind="selected_qat",
            method=method,
            command=_qat_command(
                configuration, model_config, method, output.parent, paths),
            inputs=common + qat_inputs[method],
            output=output,
            dependencies=qat_dependencies[method],
            output_policy="absent_parent",
            environment_overrides=QAT_ENVIRONMENT,
        ))
    artifact_inputs = common + (
        paths["ptq_matrix"],
        paths["p3_assignment"],
        paths["hawq_trace"],
        paths["hawq_assignment"],
    ) + ptq_manifests + tuple(qat_outputs[method]
                              for method in LAUNCH_QAT_ORDER)
    artifact_index = _model_job(
        model_config=model_config,
        spec=spec,
        name="formal_artifacts",
        kind="artifact_index",
        method=None,
        command=(
            _script("launch_nyu_three_model_quantization.py"),
            "publish-artifact-index",
            "--config", str(configuration.config_path),
            "--launch-spec", str(spec.source),
            "--model", model,
            "--output", str(paths["artifact_index"]),
        ),
        inputs=artifact_inputs,
        output=paths["artifact_index"],
        dependencies=(
            "selected_ptq",
            "hawq_mixed_le6_qat",
            "lsqplus_w6a6_qat",
            "lsqplus_w4a4_qat",
            "mixed_task_aware",
        ),
        output_policy="command_file",
        orchestrator=True,
    )
    formal_jobs = []
    previous = "formal_artifacts"
    formal_outputs = []
    for method in SELECTED_METHODS:
        output = paths["formal_root"] / "methods" / method / "formal_run.json"
        predictions = _formal_prediction_paths(
            paths["formal_root"], method, model_config.evaluation_indices)
        diagnostics = output.parent / "diagnostics.json"
        formal_outputs.extend((output, diagnostics) + predictions)
        formal_jobs.append(_model_job(
            model_config=model_config,
            spec=spec,
            name="evaluate_%s" % method,
            kind="formal_evaluation",
            method=method,
            command=(
                _script("evaluate_nyu_selected_quantization.py"),
                "evaluate",
                "--config", str(configuration.config_path),
                "--launch-spec", str(spec.source),
                "--model", model,
                "--method", method,
                "--artifact-index", str(paths["artifact_index"]),
                "--output-root", str(paths["formal_root"]),
            ),
            inputs=(
                configuration.config_path, spec.source,
                paths["artifact_index"]),
            output=output,
            dependencies=(previous,),
            output_policy="command_tree",
            produced_outputs=(diagnostics,) + predictions,
        ))
        previous = "evaluate_%s" % method
    aggregate_metrics = paths["formal_root"] / "aggregate_metrics.csv"
    selected_summary = paths["formal_root"] / "selected_method_summary.json"
    aggregate = _model_job(
        model_config=model_config,
        spec=spec,
        name="aggregate",
        kind="formal_aggregation",
        method=None,
        command=(
            _script("evaluate_nyu_selected_quantization.py"),
            "aggregate",
            "--config", str(configuration.config_path),
            "--launch-spec", str(spec.source),
            "--model", model,
            "--artifact-index", str(paths["artifact_index"]),
            "--output-root", str(paths["formal_root"]),
        ),
        inputs=(
            configuration.config_path, spec.source,
            paths["artifact_index"],
        ) + tuple(formal_outputs),
        output=selected_summary,
        dependencies=(previous,),
        output_policy="command_file",
        produced_outputs=(
            aggregate_metrics,
            paths["formal_root"] / "sample_metrics.csv",
            paths["formal_root"] / "relative_fp_loss.csv",
            paths["formal_root"] / "cost_table.csv",
            paths["formal_root"] / "diagnostics_index.csv",
        ),
    )
    plot_output = paths["formal_root"] / "figures" / \
        "pooled_rmse_comparison.png"
    plot = _model_job(
        model_config=model_config,
        spec=spec,
        name="plot",
        kind="formal_plot",
        method=None,
        command=(
            _script("plot_nyu_selected_quantization.py"),
            "--config", str(configuration.config_path),
            "--model", model,
            "--artifact-index", str(paths["artifact_index"]),
            "--output-root", str(paths["formal_root"]),
        ),
        inputs=(
            configuration.config_path, spec.source,
            paths["artifact_index"], aggregate_metrics, selected_summary,
        ) + tuple(
            path for method in SELECTED_METHODS
            for path in _formal_prediction_paths(
                paths["formal_root"], method,
                model_config.evaluation_indices)),
        output=plot_output,
        dependencies=("aggregate",),
        output_policy="command_tree",
    )
    return (preparation, validation, p3, ptq, trace, allocation) + \
        tuple(qat_jobs) + \
        (artifact_index,) + tuple(formal_jobs) + (aggregate, plot)


def build_jobs(configuration: LaunchConfiguration) -> Tuple[LaunchJob, ...]:
    jobs = []
    summary_inputs = []
    summary_dependencies = []
    for model_config in configuration.experiment.models:
        model_jobs = _build_model_jobs(configuration, model_config)
        jobs.extend(model_jobs)
        summary = _model_paths(
            configuration, model_config.model)["formal_root"] / \
            "selected_method_summary.json"
        summary_inputs.append(summary)
        summary_dependencies.append("%s:plot" % model_config.model)
    output = configuration.experiment.output_root / "cross_model_summary.json"
    jobs.append(LaunchJob(
        job_id="cross_model_summary",
        model=None,
        name="cross_model_summary",
        kind="cross_model_summary",
        method=None,
        device=configuration.spec.orchestrator_device,
        command=(
            str(configuration.spec.orchestrator_python),
            _script("launch_nyu_three_model_quantization.py"),
            "publish-cross-model-summary",
            "--summaries",
        ) + tuple(str(path) for path in summary_inputs) + (
            "--output", str(output),),
        environment=configuration.spec.orchestrator_environment,
        inputs=(configuration.config_path, configuration.spec.source) +
            tuple(summary_inputs),
        output=output.resolve(),
        dependencies=tuple(summary_dependencies),
        output_policy="command_file",
        produced_outputs=(
            configuration.experiment.output_root /
            "cross_model_summary.csv",),
    ))
    return tuple(jobs)


def build_launch_graph(configuration: LaunchConfiguration) -> LaunchGraph:
    return LaunchGraph(build_jobs(configuration))


def file_sha256(path: Path) -> str:
    return evaluation_file_sha256(Path(path))


def file_revision(path: Path) -> dict:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError("launch input is missing: %s" % source)
    before = source.stat()
    fingerprint = file_sha256(source)
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != \
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError("launch input changed during revision capture: %s" %
                           source)
    return {
        "path": str(source),
        "size_bytes": int(after.st_size),
        "sha256": fingerprint,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(destination.name + ".next")
    pending.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pending.replace(destination)


def _job_static_payload(job: LaunchJob, log_path: Path) -> dict:
    return {
        "format_version": 1,
        "job_id": job.job_id,
        "model": job.model,
        "name": job.name,
        "kind": job.kind,
        "method": job.method,
        "device": job.device,
        "command": list(job.command),
        "environment": dict(job.environment),
        "inputs": [str(path.resolve()) for path in job.inputs],
        "output_path": str(job.output.resolve()),
        "produced_outputs": [
            str(path.resolve()) for path in job.produced_outputs],
        "dependencies": list(job.dependencies),
        "log_path": str(Path(log_path).resolve()),
    }


def _job_manifest_payload(
        job: LaunchJob, log_path: Path, *, state: str,
        input_revisions, start_time, end_time, exit_status) -> dict:
    payload = _job_static_payload(job, log_path)
    payload.update({
        "input_revisions": list(input_revisions),
        "state": str(state),
        "start_time_utc": start_time,
        "end_time_utc": end_time,
        "exit_status": exit_status,
    })
    if set(payload) != JOB_MANIFEST_FIELDS:
        raise RuntimeError("launch job manifest fields changed")
    return payload


def write_planned_job_manifest(
        job: LaunchJob, manifest_path: Path, log_path: Path,
        input_revisions=None) -> None:
    revisions = _planned_input_revisions(job) \
        if input_revisions is None else tuple(input_revisions)
    _write_json(manifest_path, _job_manifest_payload(
        job,
        log_path,
        state="planned",
        input_revisions=revisions,
        start_time=None,
        end_time=None,
        exit_status=None,
    ))


def _prepare_job_output(job: LaunchJob) -> None:
    outputs = (job.output,) + tuple(job.produced_outputs)
    existing = tuple(path for path in outputs if path.exists())
    if existing:
        raise FileExistsError("launch output already exists: %s" % existing[0])
    if job.output_policy == "existing_empty_parent":
        if job.output.parent.exists():
            if any(job.output.parent.iterdir()):
                raise FileExistsError(
                    "launch output directory is not empty: %s" %
                    job.output.parent)
        else:
            job.output.parent.mkdir(parents=True, exist_ok=False)
    elif job.output_policy == "absent_parent":
        if job.output.parent.exists():
            raise FileExistsError(
                "launch command output directory already exists: %s" %
                job.output.parent)
        job.output.parent.parent.mkdir(parents=True, exist_ok=True)
    else:
        job.output.parent.parent.mkdir(parents=True, exist_ok=True)


def execute_job(
        job: LaunchJob, manifest_path: Path, log_path: Path) -> None:
    revisions = tuple(file_revision(path) for path in job.inputs)
    _prepare_job_output(job)
    start = _utc_now()
    _write_json(manifest_path, _job_manifest_payload(
        job,
        log_path,
        state="running",
        input_revisions=revisions,
        start_time=start,
        end_time=None,
        exit_status=None,
    ))
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(log_path).open("x", encoding="utf-8") as handle:
        completed = subprocess.run(
            job.command,
            cwd=str(REPO_ROOT),
            env=dict(job.environment),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    end = _utc_now()
    if completed.returncode != 0:
        _write_json(manifest_path, _job_manifest_payload(
            job,
            log_path,
            state="failed",
            input_revisions=revisions,
            start_time=start,
            end_time=end,
            exit_status=int(completed.returncode),
        ))
        raise JobExecutionError(
            "launch job %s exited with status %d" %
            (job.job_id, completed.returncode))
    expected_outputs = (job.output,) + tuple(job.produced_outputs)
    missing = tuple(path for path in expected_outputs if not path.is_file())
    if missing:
        _write_json(manifest_path, _job_manifest_payload(
            job,
            log_path,
            state="failed",
            input_revisions=revisions,
            start_time=start,
            end_time=end,
            exit_status=0,
        ))
        raise RuntimeError(
            "launch job did not publish its declared output: %s" % missing[0])
    _write_json(manifest_path, _job_manifest_payload(
        job,
        log_path,
        state="completed",
        input_revisions=revisions,
        start_time=start,
        end_time=end,
        exit_status=0,
    ))


def _job_filename(job_id: str) -> str:
    return job_id.replace(":", "__") + ".json"


def _job_manifest_path(configuration: LaunchConfiguration, job: LaunchJob):
    return configuration.experiment.output_root / "launch" / "jobs" / \
        _job_filename(job.job_id)


def _job_log_path(configuration: LaunchConfiguration, job: LaunchJob):
    return configuration.experiment.output_root / "launch" / "logs" / \
        (job.job_id.replace(":", "__") + ".log")


def _declared_outputs(job: LaunchJob):
    return (job.output,) + tuple(job.produced_outputs)


def _pending_revision(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "size_bytes": None,
        "sha256": None,
    }


def _planned_input_revisions(job, produced=(), revision_cache=None):
    generated = set(Path(path).resolve() for path in produced)
    cache = {} if revision_cache is None else revision_cache
    revisions = []
    for path in job.inputs:
        source = path.resolve()
        if source in generated or not source.is_file():
            revisions.append(_pending_revision(source))
        else:
            if source not in cache:
                cache[source] = file_revision(source)
            revisions.append(cache[source])
    return tuple(revisions)


def _plan_payload(
        configuration: LaunchConfiguration, jobs,
        revision_cache=None) -> dict:
    produced = set(
        path.resolve() for job in jobs for path in _declared_outputs(job))
    cache = {} if revision_cache is None else revision_cache
    job_rows = []
    for job in jobs:
        row = _job_static_payload(
            job, _job_log_path(configuration, job))
        row["input_revisions"] = list(_planned_input_revisions(
            job, produced, cache))
        job_rows.append(row)
    return {
        "format_version": 1,
        "config": file_revision(configuration.config_path),
        "launch_spec": file_revision(configuration.spec.source),
        "output_root": str(configuration.experiment.output_root.resolve()),
        "models": list(MODEL_ORDER),
        "methods": list(SELECTED_METHODS),
        "jobs": job_rows,
    }


def write_launch_plan(configuration: LaunchConfiguration) -> Path:
    graph = build_launch_graph(configuration)
    jobs = graph.topological_jobs()
    produced = set(
        path.resolve() for job in jobs for path in _declared_outputs(job))
    for job in jobs:
        for source in job.inputs:
            if source.resolve() not in produced and not source.is_file():
                raise FileNotFoundError(
                    "static launch input is missing: %s" % source)
    existing = tuple(path for path in produced if path.exists())
    if existing:
        raise FileExistsError("formal launch output already exists: %s" %
                              existing[0])
    launch_root = configuration.experiment.output_root / "launch"
    if launch_root.exists():
        raise FileExistsError("launch manifest root already exists: %s" %
                              launch_root)
    revision_cache = {}
    plan_payload = _plan_payload(configuration, jobs, revision_cache)
    (launch_root / "jobs").mkdir(parents=True, exist_ok=False)
    (launch_root / "logs").mkdir()
    for job in jobs:
        write_planned_job_manifest(
            job,
            _job_manifest_path(configuration, job),
            _job_log_path(configuration, job),
            _planned_input_revisions(job, produced, revision_cache),
        )
    plan_path = launch_root / "launch_plan.json"
    _write_json(plan_path, plan_payload)
    return plan_path


def _validate_completed_manifest(
        configuration: LaunchConfiguration, job: LaunchJob) -> bool:
    manifest_path = _job_manifest_path(configuration, job)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(payload) != JOB_MANIFEST_FIELDS:
        raise KeyError("launch job manifest fields changed: %s" % job.job_id)
    static = _job_static_payload(job, _job_log_path(configuration, job))
    if any(payload[name] != value for name, value in static.items()):
        raise ValueError("launch job manifest command changed: %s" % job.job_id)
    if payload["state"] == "planned":
        return False
    if payload["state"] != "completed" or payload["exit_status"] != 0:
        raise RuntimeError(
            "launch job is not resumable: %s state=%s" %
            (job.job_id, payload["state"]))
    revisions = tuple(file_revision(path) for path in job.inputs)
    if payload["input_revisions"] != list(revisions):
        raise RuntimeError("completed launch job inputs changed: %s" %
                           job.job_id)
    missing = tuple(path for path in _declared_outputs(job)
                    if not path.is_file())
    if missing:
        raise FileNotFoundError(
            "completed launch job output is missing: %s" % missing[0])
    return True


def _require_dependencies_complete(
        configuration: LaunchConfiguration, job: LaunchJob) -> None:
    for dependency in job.dependencies:
        path = configuration.experiment.output_root / "launch" / "jobs" / \
            _job_filename(dependency)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["state"] != "completed" or payload["exit_status"] != 0:
            raise RuntimeError(
                "launch dependency is incomplete: %s -> %s" %
                (job.job_id, dependency))


def _execute_model_lane(
        configuration: LaunchConfiguration, graph: LaunchGraph,
        model: str) -> None:
    for job in graph.jobs_for_model(model):
        _require_dependencies_complete(configuration, job)
        if not _validate_completed_manifest(configuration, job):
            execute_job(
                job,
                _job_manifest_path(configuration, job),
                _job_log_path(configuration, job),
            )


def execute_launch_plan(
        configuration: LaunchConfiguration, plan_path: Path) -> None:
    graph = build_launch_graph(configuration)
    jobs = graph.topological_jobs()
    expected_path = configuration.experiment.output_root / "launch" / \
        "launch_plan.json"
    if Path(plan_path).resolve() != expected_path.resolve():
        raise ValueError("launch plan path differs from configured output root")
    observed = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    expected = _plan_payload(configuration, jobs)
    if observed != expected:
        raise RuntimeError("launch plan or static input revisions changed")
    with ThreadPoolExecutor(max_workers=len(MODEL_ORDER)) as executor:
        futures = tuple(
            executor.submit(_execute_model_lane, configuration, graph, model)
            for model in MODEL_ORDER)
        for future in futures:
            future.result()
    global_jobs = tuple(job for job in graph.topological_jobs()
                        if job.model is None)
    for job in global_jobs:
        _require_dependencies_complete(configuration, job)
        if not _validate_completed_manifest(configuration, job):
            execute_job(
                job,
                _job_manifest_path(configuration, job),
                _job_log_path(configuration, job),
            )


def _validate_terminal_qat_checkpoint(
        path: Path, model: str, method: str) -> None:
    import torch
    from scripts.train_nyu_selected_qat import validate_checkpoint_payload

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    validate_checkpoint_payload(payload)
    if str(payload["model_name"]) != str(model):
        raise ValueError("terminal QAT checkpoint model changed")
    if str(payload["method"]) != str(method):
        raise ValueError("terminal QAT checkpoint method changed")
    run_state = payload["run_state"]
    if run_state["terminal"] is not True or run_state["completed"] is not True:
        raise RuntimeError("QAT checkpoint is not terminal: %s" % method)


def publish_formal_artifact_index(
        configuration: LaunchConfiguration, model: str,
        output: Path) -> Path:
    model_rows = tuple(row for row in configuration.experiment.models
                       if row.model == str(model))
    if len(model_rows) != 1:
        raise ValueError("formal artifact model is not unique")
    model_config = model_rows[0]
    paths = _model_paths(configuration, model)
    destination = Path(output).resolve()
    if destination != paths["artifact_index"].resolve():
        raise ValueError("formal artifact index output path changed")
    if destination.exists():
        raise FileExistsError("formal artifact index already exists: %s" %
                              destination)
    ptq_matrix = paths["ptq_matrix"].resolve()
    ptq_payload = json.loads(ptq_matrix.read_text(encoding="utf-8"))
    if tuple(ptq_payload["methods"]) != PTQ_METHODS or \
            str(ptq_payload["model"]) != model:
        raise ValueError("selected PTQ matrix identity changed")
    manifests = ptq_payload["hard_deployment_manifests"]
    if set(manifests) != set(PTQ_METHODS):
        raise ValueError("selected PTQ manifest method set changed")
    expected_manifests = dict(
        (method, paths["ptq_root"] / method /
         "hard_deployment_manifest.json") for method in PTQ_METHODS)
    for method in PTQ_METHODS:
        if Path(manifests[method]).resolve() != \
                expected_manifests[method].resolve():
            raise ValueError("selected PTQ manifest path changed: %s" % method)
    qat_paths = dict(
        (method, paths["qat_root"] / method / "final.pt")
        for method in QAT_METHODS)
    for method in QAT_METHODS:
        _validate_terminal_qat_checkpoint(qat_paths[method], model, method)
    primary = {
        "fp32": model_config.checkpoint,
        "rtn_w8a8": expected_manifests["rtn_w8a8"],
        "rtn_w4a4": expected_manifests["rtn_w4a4"],
        "qdrop_w6a6": expected_manifests["qdrop_w6a6"],
        "brecq_w6a6": expected_manifests["brecq_w6a6"],
        "hawq_mixed_le6": qat_paths["hawq_mixed_le6"],
        "lsqplus_w6a6": qat_paths["lsqplus_w6a6"],
        "lsqplus_w4a4": qat_paths["lsqplus_w4a4"],
        "mixed_task_aware": qat_paths["mixed_task_aware"],
        "p3_t3_mixed_ptq": expected_manifests["p3_t3_mixed_ptq"],
    }
    kinds = dict((method, "strict_ptq_manifest") for method in PTQ_METHODS)
    kinds.update(dict((method, "terminal_qat_checkpoint")
                      for method in QAT_METHODS))
    kinds["fp32"] = "official_checkpoint"
    support_paths = {
        "fp32": (),
        "rtn_w8a8": (),
        "rtn_w4a4": (),
        "qdrop_w6a6": (),
        "brecq_w6a6": (),
        "hawq_mixed_le6": (
            ("hawq_assignment", paths["hawq_assignment"]),
            ("hawq_trace_artifact", paths["hawq_trace"]),
        ),
        "lsqplus_w6a6": (),
        "lsqplus_w4a4": (),
        "mixed_task_aware": (
            ("p3_t3_assignment", paths["p3_assignment"]),),
        "p3_t3_mixed_ptq": (
            ("p3_t3_assignment", paths["p3_assignment"]),),
    }
    rows = []
    for method in SELECTED_METHODS:
        artifact = Path(primary[method]).resolve()
        rows.append({
            "method": method,
            "artifact_kind": kinds[method],
            "artifact": str(artifact),
            "artifact_sha256": file_sha256(artifact),
            "supporting_artifacts": [{
                "name": name,
                "path": str(Path(path).resolve()),
                "sha256": file_sha256(path),
            } for name, path in support_paths[method]],
        })
    preparation = configuration.spec.hard_deployment
    payload = {
        "format_version": 1,
        "model": model,
        "evaluation_indices": list(model_config.evaluation_indices),
        "evaluation_identity": ordered_evaluation_identity(
            model_config.evaluation_indices),
        "ptq_matrix": {
            "path": str(ptq_matrix),
            "sha256": file_sha256(ptq_matrix),
        },
        "methods": rows,
        "preparation": {
            "fold_conv_bn": int(preparation["fold_conv_bn"]),
            "fold_max_error": preparation["fold_max_error"],
            "joint_clip_factors": list(preparation["joint_clip_factors"]),
            "joint_search_rounds": preparation["joint_search_rounds"],
            "joint_cache_sample_limit":
                preparation["joint_cache_sample_limit"],
            "joint_cache_byte_limit":
                preparation["joint_cache_byte_limit"],
        },
    }
    _write_json(destination, payload)
    load_formal_artifact_index(
        destination, model, model_config.evaluation_indices)
    return destination


def publish_cross_model_summary(
        summaries: Sequence[Path], output: Path) -> Path:
    paths = tuple(Path(path).resolve() for path in summaries)
    if len(paths) != len(MODEL_ORDER):
        raise ValueError("cross-model summary requires three model inputs")
    rows = []
    revisions = []
    for model, path in zip(MODEL_ORDER, paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload["model"]) != model:
            raise ValueError("cross-model summary model order changed")
        if tuple(payload["methods"]) != SELECTED_METHODS:
            raise ValueError("cross-model summary method order changed")
        model_rows = tuple(payload["aggregate_metrics"])
        if tuple(str(row["method"]) for row in model_rows) != SELECTED_METHODS:
            raise ValueError("cross-model aggregate method order changed")
        for row in model_rows:
            if str(row["model"]) != model or int(row["samples"]) != 64:
                raise ValueError("cross-model aggregate identity changed")
            for field in (
                    "pooled_rmse", "mean_sample_rmse", "pooled_mae",
                    "pooled_abs_rel", "pooled_irmse"):
                value = float(row[field])
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        "cross-model aggregate metric is invalid")
            rows.append(dict(row))
        revisions.append(file_revision(path))
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError("cross-model summary already exists: %s" %
                              destination)
    _write_json(destination, {
        "format_version": 1,
        "models": list(MODEL_ORDER),
        "methods": list(SELECTED_METHODS),
        "primary_metric": "pooled_rmse",
        "diagnostic_metric": "mean_sample_rmse",
        "input_revisions": revisions,
        "aggregate_metrics": rows,
    })
    csv_path = destination.with_suffix(".csv")
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def publish_static_input_validation(
        configuration: LaunchConfiguration, model: str,
        output: Path) -> Path:
    model_rows = tuple(
        row for row in configuration.experiment.models
        if row.model == str(model))
    if len(model_rows) != 1:
        raise ValueError("static-input model is not unique")
    model_config = model_rows[0]
    launch_inputs = configuration.spec.model_inputs[model]
    paths = StaticInputPaths(
        calibration_metadata=model_config.calibration_metadata,
        calibration_indices=launch_inputs.calibration_indices,
        evaluation_protocol=launch_inputs.evaluation_protocol,
        weight_cost_rows=launch_inputs.weight_cost_rows,
        activation_cost_rows=launch_inputs.activation_cost_rows,
    )
    validated = validate_static_input_bundle(model_config, paths)
    destination = Path(output).resolve()
    expected = _model_paths(
        configuration, model)["static_validation"].resolve()
    if destination != expected:
        raise ValueError("static-input validation output path changed")
    if destination.exists():
        raise FileExistsError(
            "static-input validation output already exists: %s" %
            destination)
    payload = {
        "format_version": 1,
        "model": validated.model,
        "calibration_indices": list(validated.calibration_indices),
        "evaluation_indices": list(validated.evaluation_indices),
        "evaluation_seed": validated.evaluation_seed,
        "artifacts": {
            "calibration_metadata": {
                "path": str(paths.calibration_metadata.resolve()),
                "sha256": static_file_sha256(paths.calibration_metadata),
            },
            "calibration_indices": {
                "path": str(paths.calibration_indices.resolve()),
                "sha256": static_file_sha256(paths.calibration_indices),
            },
            "evaluation_protocol": {
                "path": str(paths.evaluation_protocol.resolve()),
                "sha256": static_file_sha256(paths.evaluation_protocol),
            },
            "weight_cost_rows": {
                "path": str(paths.weight_cost_rows.resolve()),
                "sha256": static_file_sha256(paths.weight_cost_rows),
            },
            "activation_cost_rows": {
                "path": str(paths.activation_cost_rows.resolve()),
                "sha256": static_file_sha256(paths.activation_cost_rows),
            },
        },
    }
    _write_json(destination, payload)
    return destination


def build_parser():
    parser = argparse.ArgumentParser(
        description="Launch strict selected quantization for three NYU models")
    operations = parser.add_subparsers(dest="operation", required=True)
    plan = operations.add_parser("plan")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--launch-spec", type=Path, required=True)
    execute = operations.add_parser("execute")
    execute.add_argument("--config", type=Path, required=True)
    execute.add_argument("--launch-spec", type=Path, required=True)
    execute.add_argument("--plan", type=Path, required=True)
    static = operations.add_parser("validate-static-inputs")
    static.add_argument("--config", type=Path, required=True)
    static.add_argument("--launch-spec", type=Path, required=True)
    static.add_argument("--model", choices=MODEL_ORDER, required=True)
    static.add_argument("--output", type=Path, required=True)
    artifact = operations.add_parser("publish-artifact-index")
    artifact.add_argument("--config", type=Path, required=True)
    artifact.add_argument("--launch-spec", type=Path, required=True)
    artifact.add_argument("--model", choices=MODEL_ORDER, required=True)
    artifact.add_argument("--output", type=Path, required=True)
    summary = operations.add_parser("publish-cross-model-summary")
    summary.add_argument(
        "--summaries", type=Path, nargs=len(MODEL_ORDER), required=True)
    summary.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.operation == "publish-cross-model-summary":
        publish_cross_model_summary(args.summaries, args.output)
        return
    configuration = load_launch_configuration(
        args.config, args.launch_spec)
    if args.operation == "plan":
        print(write_launch_plan(configuration))
    elif args.operation == "execute":
        execute_launch_plan(configuration, args.plan)
    elif args.operation == "validate-static-inputs":
        print(publish_static_input_validation(
            configuration, args.model, args.output))
    else:
        publish_formal_artifact_index(
            configuration, args.model, args.output)


if __name__ == "__main__":
    main()
