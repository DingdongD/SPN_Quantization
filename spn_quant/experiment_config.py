"""Strict experiment configuration for selected NYU quantization models."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Tuple


MODEL_ORDER = ("dyspn", "nlspn", "completionformer")
METHOD_ORDER = (
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
METHOD_FIELDS = {
    "rtn_w8a8": ("weight_bits", "activation_bits", "calibration_count"),
    "rtn_w4a4": ("weight_bits", "activation_bits", "calibration_count"),
    "qdrop_w6a6": (
        "weight_bits", "activation_bits", "steps", "calibration_count"),
    "brecq_w6a6": (
        "weight_bits", "activation_bits", "steps", "calibration_count"),
    "hawq_mixed_le6": (
        "bits", "maximum_average_weight_bits",
        "maximum_average_activation_bits", "calibration_count"),
    "lsqplus_w6a6": (
        "weight_bits", "activation_bits", "initialization_count"),
    "lsqplus_w4a4": (
        "weight_bits", "activation_bits", "initialization_count"),
    "mixed_task_aware": (
        "weight_bits", "activation_bits",
        "maximum_average_activation_bits"),
    "p3_t3_mixed_ptq": (
        "base_weight_bits", "base_activation_bits",
        "promotion_weight_bits", "promotion_activation_bits"),
}


def _freeze_mapping(payload: Mapping[str, object]) -> Mapping[str, object]:
    frozen = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            frozen[str(key)] = _freeze_mapping(value)
        elif isinstance(value, list):
            frozen[str(key)] = tuple(value)
        else:
            frozen[str(key)] = value
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class ModelExperimentConfig:
    model: str
    run_dir: Path
    checkpoint: Path
    python_executable: Path
    device: str
    propagation_iterations: int
    data_root: Path
    calibration_metadata: Path
    calibration_count: int
    evaluation_indices: Tuple[int, ...]
    expected_architecture_class: str
    checkpoint_architecture: str
    required_cuda_extension: str
    native_cuda_operator: Optional[str]

    def __post_init__(self) -> None:
        if self.model not in MODEL_ORDER:
            raise ValueError("unsupported selected model: %s" % self.model)
        if not self.run_dir.is_absolute():
            raise ValueError("run directory must be absolute")
        if self.checkpoint.parent != self.run_dir:
            raise ValueError("checkpoint must be inside the run directory")
        if not self.checkpoint.is_absolute():
            raise ValueError("checkpoint must be absolute")
        if not self.python_executable.is_absolute():
            raise ValueError("Python executable must be absolute")
        if not self.device.startswith("cuda:"):
            raise ValueError("model device must name an explicit CUDA device")
        if not self.data_root.is_absolute():
            raise ValueError("data root must be absolute")
        if not self.calibration_metadata.is_absolute():
            raise ValueError("calibration metadata must be absolute")
        if int(self.propagation_iterations) <= 0:
            raise ValueError("propagation iterations must be positive")
        if int(self.calibration_count) != 128:
            raise ValueError("calibration count must equal 128")
        if len(self.evaluation_indices) != 64:
            raise ValueError("evaluation indices must contain 64 samples")
        if len(set(self.evaluation_indices)) != len(self.evaluation_indices):
            raise ValueError("evaluation indices must be unique")
        if any(index < 0 for index in self.evaluation_indices):
            raise ValueError("evaluation indices must be nonnegative")
        if not self.expected_architecture_class:
            raise ValueError("expected architecture class is required")
        if not self.checkpoint_architecture:
            raise ValueError("checkpoint architecture is required")
        if not self.required_cuda_extension:
            raise ValueError("required CUDA extension is required")
        if self.model == "dyspn" and not self.native_cuda_operator:
            raise ValueError("DySPN native CUDA operator is required")
        if self.model != "dyspn" and self.native_cuda_operator is not None:
            raise ValueError("native CUDA operator is only valid for DySPN")

    def runtime_args(self) -> Namespace:
        return Namespace(
            model=self.model,
            run_dir=self.run_dir,
            checkpoint=self.checkpoint,
            expected_architecture_class=self.expected_architecture_class,
            required_cuda_extension=self.required_cuda_extension,
            propagation_iterations=self.propagation_iterations,
            data_root=self.data_root,
            device=self.device,
            checkpoint_architecture=self.checkpoint_architecture,
            native_cuda_operator=self.native_cuda_operator,
        )

@dataclass(frozen=True)
class SelectedQuantizationConfig:
    output_root: Path
    models: Tuple[ModelExperimentConfig, ...]
    method_hyperparameters: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.output_root.is_absolute():
            raise ValueError("output root must be absolute")
        if tuple(model.model for model in self.models) != MODEL_ORDER:
            raise ValueError("selected model order changed")
        if tuple(self.method_hyperparameters) != METHOD_ORDER:
            raise ValueError("selected method hyperparameter order changed")


def _parse_model(payload: Mapping[str, object]) -> ModelExperimentConfig:
    return ModelExperimentConfig(
        model=str(payload["model"]),
        run_dir=Path(payload["run_dir"]),
        checkpoint=Path(payload["checkpoint"]),
        python_executable=Path(payload["python_executable"]),
        device=str(payload["device"]),
        propagation_iterations=int(payload["propagation_iterations"]),
        data_root=Path(payload["data_root"]),
        calibration_metadata=Path(payload["calibration_metadata"]),
        calibration_count=int(payload["calibration_count"]),
        evaluation_indices=tuple(int(index) for index in payload[
            "evaluation_indices"]),
        expected_architecture_class=str(payload["expected_architecture_class"]),
        checkpoint_architecture=str(payload["checkpoint_architecture"]),
        required_cuda_extension=str(payload["required_cuda_extension"]),
        native_cuda_operator=payload["native_cuda_operator"],
    )


def _parse_method_hyperparameters(
        payload: Mapping[str, object]) -> Mapping[str, object]:
    methods = {}
    for method_name in METHOD_ORDER:
        method = payload[method_name]
        fields = METHOD_FIELDS[method_name]
        methods[method_name] = dict(
            (field_name, method[field_name]) for field_name in fields)
        if tuple(method) != fields:
            raise ValueError("method hyperparameter contract changed: %s" %
                             method_name)
    return _freeze_mapping(methods)


def load_selected_quantization_config(path: Path) -> SelectedQuantizationConfig:
    """Load the complete selected-model experiment contract without defaults."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    models = payload["models"]
    method_hyperparameters = payload["method_hyperparameters"]
    return SelectedQuantizationConfig(
        output_root=Path(payload["output_root"]),
        models=tuple(_parse_model(models[model_name])
                     for model_name in MODEL_ORDER),
        method_hyperparameters=_parse_method_hyperparameters(
            method_hyperparameters),
    )
