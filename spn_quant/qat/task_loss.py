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


def _require_finite_batch(rows, valid: torch.Tensor) -> None:
    rows = tuple(rows)
    if not rows:
        raise ValueError("semantic tensor batch must be nonempty")
    for name, tensor in rows:
        if not torch.is_tensor(tensor):
            raise TypeError("%s must be a tensor" % name)
        if tensor.numel() == 0:
            raise ValueError("%s must be nonempty" % name)
    checks = torch.stack(tuple(
        torch.isfinite(tensor).all() for name, tensor in rows) +
        (valid.any(),)).detach().cpu().tolist()
    if not bool(checks[-1]):
        raise ValueError("task loss requires at least one valid depth")
    if all(bool(value) for value in checks[:-1]):
        return
    for (name, tensor), finite in zip(rows, checks[:-1]):
        if not bool(finite):
            raise ValueError("%s must be finite" % name)
    raise RuntimeError("semantic tensor finite-state diagnosis failed")


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
    weights = mask.to(dtype=left.dtype)
    return ((left - right).abs() * weights).sum() / \
        weights.sum().clamp_min(1.0)


def _capture_tensors(name: str, capture: ModelTaskCapture,
                     target: torch.Tensor):
    if not isinstance(capture, ModelTaskCapture):
        raise TypeError("%s must be ModelTaskCapture" % name)
    if capture.prediction.shape != target.shape or \
            capture.initial_depth.shape != target.shape:
        raise ValueError("%s depth tensor shape does not match target" % name)
    if not capture.propagation_states:
        raise ValueError("%s propagation states are empty" % name)
    for index, state in enumerate(capture.propagation_states):
        if state.shape != target.shape:
            raise ValueError(
                "%s propagation state shape does not match target" % name)
    return (
        (("%s prediction" % name, capture.prediction),
         ("%s initial depth" % name, capture.initial_depth)) +
        tuple(("%s propagation state %d" % (name, index), state)
              for index, state in enumerate(capture.propagation_states)))


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
    if not torch.is_tensor(target) or target.numel() == 0:
        raise ValueError("target must be a nonempty tensor")
    if valid.dtype != torch.bool or valid.shape != target.shape:
        raise ValueError("valid depth mask shape and dtype are invalid")
    if not isinstance(student, ModelTaskCapture) or not isinstance(
            teacher, ModelTaskCapture):
        raise TypeError("student and teacher must be ModelTaskCapture")
    if len(student.propagation_states) != len(
            teacher.propagation_states):
        raise ValueError(
            "student and teacher propagation state counts differ")
    student_tensors = _capture_tensors("student", student, target)
    teacher_tensors = _capture_tensors("teacher", teacher, target)
    _require_finite_batch(
        (("target", target),) + student_tensors + teacher_tensors,
        valid,
    )

    threshold = float(boundary_threshold_m)
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
    boundary = horizontal | vertical
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
