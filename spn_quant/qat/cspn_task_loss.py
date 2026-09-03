"""CSPN compatibility wrapper for architecture-neutral task loss."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch

from spn_quant.qat.task_loss import (
    ModelTaskCapture,
    ModelTaskLossWeights,
    depth_boundary_mask,
    model_task_aware_loss,
)


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
    result = model_task_aware_loss(
        ModelTaskCapture(
            prediction=prediction,
            initial_depth=prediction,
            propagation_states=tuple(student_states),
        ),
        ModelTaskCapture(
            prediction=teacher_prediction,
            initial_depth=teacher_prediction,
            propagation_states=tuple(teacher_states),
        ),
        target,
        valid,
        ModelTaskLossWeights(
            depth=weights.depth,
            boundary=weights.boundary,
            teacher=weights.teacher,
            initial_depth=0.0,
            propagation=weights.propagation,
        ),
        boundary_threshold_m,
    )
    return {
        "total": result.total,
        "depth": result.depth,
        "boundary": result.boundary,
        "teacher": result.teacher,
        "propagation": result.propagation,
    }


__all__ = (
    "CSPNTaskLossWeights",
    "cspn_task_aware_loss",
    "depth_boundary_mask",
)
