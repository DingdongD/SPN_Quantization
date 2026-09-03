"""Architecture-neutral semantic task loss for selected SPN QAT."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import torch


@dataclass(frozen=True)
class ModelTaskCapture:
    prediction: torch.Tensor
    initial_depth: torch.Tensor
    propagation_states: Tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class ModelTaskLossWeights:
    depth: float
    boundary: float
    teacher: float
    initial_depth: float
    propagation: float

    def __post_init__(self) -> None:
        values = (
            float(self.depth),
            float(self.boundary),
            float(self.teacher),
            float(self.initial_depth),
            float(self.propagation),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("model task loss weights must be finite")
        if any(value < 0.0 for value in values):
            raise ValueError("model task loss weights must be nonnegative")
        object.__setattr__(self, "depth", values[0])
        object.__setattr__(self, "boundary", values[1])
        object.__setattr__(self, "teacher", values[2])
        object.__setattr__(self, "initial_depth", values[3])
        object.__setattr__(self, "propagation", values[4])


@dataclass(frozen=True)
class ModelTaskLoss:
    total: torch.Tensor
    depth: torch.Tensor
    boundary: torch.Tensor
    teacher: torch.Tensor
    initial_depth: torch.Tensor
    propagation: torch.Tensor

    def as_dict(self):
        return {
            "total": self.total,
            "depth": self.depth,
            "boundary": self.boundary,
            "teacher": self.teacher,
            "initial_depth": self.initial_depth,
            "propagation": self.propagation,
        }


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.is_tensor(tensor):
        raise TypeError("%s must be a tensor" % name)
    if tensor.numel() == 0:
        raise ValueError("%s must be nonempty" % name)
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("%s must be finite" % name)


def depth_boundary_mask(
        target: torch.Tensor,
        valid: torch.Tensor,
        threshold_m: float) -> torch.Tensor:
    _require_finite("target", target)
    if valid.dtype != torch.bool:
        raise TypeError("valid depth mask must be boolean")
    if valid.shape != target.shape:
        raise ValueError("valid depth mask shape does not match target")
    threshold = float(threshold_m)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("boundary threshold must be finite and positive")

    horizontal = torch.zeros_like(valid)
    vertical = torch.zeros_like(valid)
    horizontal[..., :, 1:] = (
        (target[..., :, 1:] - target[..., :, :-1]).abs() >= threshold
    ) & valid[..., :, 1:] & valid[..., :, :-1]
    vertical[..., 1:, :] = (
        (target[..., 1:, :] - target[..., :-1, :]).abs() >= threshold
    ) & valid[..., 1:, :] & valid[..., :-1, :]
    return horizontal | vertical


def _masked_l1(
        left: torch.Tensor,
        right: torch.Tensor,
        mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any().item()):
        return left.sum() * 0.0
    return (left - right).abs()[mask].mean()


def _validate_capture(name: str, capture: ModelTaskCapture,
                      target: torch.Tensor) -> None:
    if not isinstance(capture, ModelTaskCapture):
        raise TypeError("%s must be ModelTaskCapture" % name)
    _require_finite("%s prediction" % name, capture.prediction)
    _require_finite("%s initial depth" % name, capture.initial_depth)
    if capture.prediction.shape != target.shape or \
            capture.initial_depth.shape != target.shape:
        raise ValueError("%s depth tensor shape does not match target" % name)
    if not capture.propagation_states:
        raise ValueError("%s propagation states are empty" % name)
    for index, state in enumerate(capture.propagation_states):
        _require_finite("%s propagation state %d" % (name, index), state)
        if state.shape != target.shape:
            raise ValueError(
                "%s propagation state shape does not match target" % name)


def model_task_aware_loss(
        student: ModelTaskCapture,
        teacher: ModelTaskCapture,
        target: torch.Tensor,
        valid: torch.Tensor,
        weights: ModelTaskLossWeights,
        boundary_threshold_m: float) -> ModelTaskLoss:
    """Compute task loss only from adapter-normalized semantic captures."""
    if not isinstance(weights, ModelTaskLossWeights):
        raise TypeError("weights must be ModelTaskLossWeights")
    _require_finite("target", target)
    if valid.dtype != torch.bool or valid.shape != target.shape:
        raise ValueError("valid depth mask shape and dtype are invalid")
    if not bool(valid.any().item()):
        raise ValueError("task loss requires at least one valid depth")
    if not isinstance(student, ModelTaskCapture) or not isinstance(
            teacher, ModelTaskCapture):
        raise TypeError("student and teacher must be ModelTaskCapture")
    if len(student.propagation_states) != len(
            teacher.propagation_states):
        raise ValueError(
            "student and teacher propagation state counts differ")
    _validate_capture("student", student, target)
    _validate_capture("teacher", teacher, target)

    boundary = depth_boundary_mask(target, valid, boundary_threshold_m)
    depth_loss = _masked_l1(student.prediction, target, valid)
    boundary_loss = _masked_l1(student.prediction, target, boundary)
    teacher_loss = _masked_l1(
        student.prediction, teacher.prediction.detach(), valid)
    initial_depth_loss = _masked_l1(
        student.initial_depth, teacher.initial_depth.detach(), valid)
    state_losses = tuple(
        _masked_l1(student_state, teacher_state.detach(), valid)
        for student_state, teacher_state in zip(
            student.propagation_states, teacher.propagation_states))
    propagation_loss = torch.stack(state_losses).mean()
    total = (
        weights.depth * depth_loss
        + weights.boundary * boundary_loss
        + weights.teacher * teacher_loss
        + weights.initial_depth * initial_depth_loss
        + weights.propagation * propagation_loss)
    return ModelTaskLoss(
        total=total,
        depth=depth_loss,
        boundary=boundary_loss,
        teacher=teacher_loss,
        initial_depth=initial_depth_loss,
        propagation=propagation_loss,
    )


__all__ = (
    "ModelTaskCapture",
    "ModelTaskLossWeights",
    "ModelTaskLoss",
    "depth_boundary_mask",
    "model_task_aware_loss",
)
