"""Task-aware loss terms for CSPN mixed-precision QAT."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class CSPNTaskLossWeights:
    depth: float
    boundary: float
    teacher: float
    propagation: float

    def __post_init__(self) -> None:
        values = (
            float(self.depth),
            float(self.boundary),
            float(self.teacher),
            float(self.propagation),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("CSPN task loss weights must be finite")
        if any(value < 0.0 for value in values):
            raise ValueError("CSPN task loss weights must be nonnegative")
        object.__setattr__(self, "depth", values[0])
        object.__setattr__(self, "boundary", values[1])
        object.__setattr__(self, "teacher", values[2])
        object.__setattr__(self, "propagation", values[3])


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


def cspn_task_aware_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        teacher_prediction: torch.Tensor,
        student_states: Sequence[torch.Tensor],
        teacher_states: Sequence[torch.Tensor],
        weights: CSPNTaskLossWeights,
        boundary_threshold_m: float) -> Mapping[str, torch.Tensor]:
    if not isinstance(weights, CSPNTaskLossWeights):
        raise TypeError("weights must be CSPNTaskLossWeights")
    _require_finite("prediction", prediction)
    _require_finite("target", target)
    _require_finite("teacher prediction", teacher_prediction)
    if prediction.shape != target.shape or \
            teacher_prediction.shape != target.shape:
        raise ValueError("prediction, teacher, and target shapes must match")
    if valid.dtype != torch.bool or valid.shape != target.shape:
        raise ValueError("valid depth mask shape and dtype are invalid")
    if not bool(valid.any().item()):
        raise ValueError("task loss requires at least one valid depth")

    student = tuple(student_states)
    teacher = tuple(teacher_states)
    if not student or len(student) != len(teacher):
        raise ValueError("student and teacher propagation state counts differ")
    for index, state in enumerate(student):
        _require_finite("student state %d" % index, state)
        if state.shape != target.shape:
            raise ValueError("student propagation state shape does not match target")
    for index, state in enumerate(teacher):
        _require_finite("teacher state %d" % index, state)
        if state.shape != target.shape:
            raise ValueError("teacher propagation state shape does not match target")

    boundary = depth_boundary_mask(
        target, valid, boundary_threshold_m)
    depth_loss = _masked_l1(prediction, target, valid)
    boundary_loss = _masked_l1(prediction, target, boundary)
    teacher_loss = _masked_l1(
        prediction, teacher_prediction.detach(), valid)
    state_losses = tuple(
        _masked_l1(student_state, teacher_state.detach(), valid)
        for student_state, teacher_state in zip(student, teacher))
    propagation_loss = torch.stack(state_losses).mean()
    total = (
        weights.depth * depth_loss
        + weights.boundary * boundary_loss
        + weights.teacher * teacher_loss
        + weights.propagation * propagation_loss)
    return {
        "total": total,
        "depth": depth_loss,
        "boundary": boundary_loss,
        "teacher": teacher_loss,
        "propagation": propagation_loss,
    }
