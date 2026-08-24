"""Hutchinson Hessian trace estimation for HAWQ allocation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence, Tuple

import torch

from spn_quant.qat.cspn_task_loss import depth_boundary_mask


@dataclass(frozen=True)
class HutchinsonTraceConfig:
    probes: int
    seed: int

    def __post_init__(self) -> None:
        if int(self.probes) <= 0:
            raise ValueError("Hutchinson probe count must be positive")
        object.__setattr__(self, "probes", int(self.probes))
        object.__setattr__(self, "seed", int(self.seed))


@dataclass(frozen=True)
class BlockTraceEstimate:
    block: str
    estimates: Tuple[float, ...]
    mean: float
    standard_error: float
    normalized_mean: float
    coefficient_of_variation: float
    parameters: int


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.is_tensor(tensor) or tensor.numel() == 0:
        raise ValueError("%s must be a nonempty tensor" % name)
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("%s must be finite" % name)


def masked_curvature_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        depth_mse_weight: float,
        boundary_mse_weight: float,
        boundary_threshold_m: float) -> torch.Tensor:
    _require_finite("curvature prediction", prediction)
    _require_finite("curvature target", target)
    if prediction.shape != target.shape:
        raise ValueError("curvature prediction and target shapes differ")
    if valid.dtype != torch.bool or valid.shape != target.shape:
        raise ValueError("curvature valid mask is invalid")
    if not bool(valid.any().item()):
        raise ValueError("curvature loss requires valid depth")
    depth_weight = float(depth_mse_weight)
    boundary_weight = float(boundary_mse_weight)
    if not math.isfinite(depth_weight) or depth_weight <= 0.0:
        raise ValueError("curvature depth weight must be positive")
    if not math.isfinite(boundary_weight) or boundary_weight < 0.0:
        raise ValueError("curvature boundary weight must be nonnegative")
    difference = prediction - target
    depth_mse = difference.square()[valid].mean()
    boundary = depth_boundary_mask(
        target, valid, float(boundary_threshold_m))
    boundary_mse = difference.square()[boundary].mean() \
        if bool(boundary.any().item()) else difference.sum() * 0.0
    return depth_weight * depth_mse + boundary_weight * boundary_mse


def _rademacher_like(parameter: torch.Tensor,
                      generator: torch.Generator) -> torch.Tensor:
    values = torch.randint(
        0, 2, parameter.shape, generator=generator,
        device=parameter.device, dtype=torch.int64)
    return values.to(parameter.dtype).mul_(2.0).sub_(1.0)


def estimate_block_trace_samples(
        blocks: Sequence[Tuple[str, torch.Tensor]],
        loss_fn: Callable[[], torch.Tensor],
        config: HutchinsonTraceConfig
        ) -> Tuple[Tuple[str, Tuple[float, ...]], ...]:
    if not isinstance(config, HutchinsonTraceConfig):
        raise TypeError("trace config must be HutchinsonTraceConfig")
    declared = tuple((str(name), parameter) for name, parameter in blocks)
    if not declared:
        raise ValueError("Hutchinson estimation requires parameter blocks")
    names = tuple(name for name, parameter in declared)
    if len(names) != len(set(names)):
        raise ValueError("Hutchinson block names contain duplicates")
    parameters = tuple(parameter for name, parameter in declared)
    if len(set(id(parameter) for parameter in parameters)) != len(parameters):
        raise ValueError("Hutchinson parameters contain duplicates")
    device = parameters[0].device
    if any(parameter.device != device for parameter in parameters):
        raise ValueError("Hutchinson parameters must share one device")
    for name, parameter in declared:
        _require_finite("Hutchinson parameter %s" % name, parameter)
        if not parameter.requires_grad:
            raise ValueError("Hutchinson parameter must require gradients")

    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    estimates = [[] for parameter in parameters]
    for _ in range(config.probes):
        loss = loss_fn()
        _require_finite("Hutchinson loss", loss)
        if loss.numel() != 1:
            raise ValueError("Hutchinson loss must be scalar")
        gradients = torch.autograd.grad(
            loss, parameters, create_graph=True)
        vectors = tuple(
            _rademacher_like(parameter, generator)
            for parameter in parameters)
        for index, (parameter, gradient, vector) in enumerate(zip(
                parameters, gradients, vectors)):
            inner = (gradient * vector).sum()
            hessian_vector, = torch.autograd.grad(
                inner,
                parameter,
                retain_graph=index + 1 < len(parameters),
            )
            estimate = (vector * hessian_vector).sum()
            _require_finite(
                "Hutchinson estimate %s" % names[index], estimate)
            estimates[index].append(float(estimate.detach().item()))

    return tuple(
        (name, tuple(values))
        for (name, parameter), values in zip(declared, estimates))


def estimate_block_traces(
        blocks: Sequence[Tuple[str, torch.Tensor]],
        loss_fn: Callable[[], torch.Tensor],
        config: HutchinsonTraceConfig) -> Tuple[BlockTraceEstimate, ...]:
    declared = tuple((str(name), parameter) for name, parameter in blocks)
    samples = estimate_block_trace_samples(declared, loss_fn, config)
    parameters = dict(declared)
    output = []
    for name, values in samples:
        parameter = parameters[name]
        current = torch.tensor(values, dtype=torch.float64)
        mean = float(current.mean().item())
        if mean < 0.0:
            raise ValueError(
                "Hutchinson block has negative mean trace: %s" % name)
        standard_error = float(
            current.std(unbiased=True).div(math.sqrt(len(values))).item()) \
            if len(values) > 1 else 0.0
        coefficient = 0.0 if mean == 0.0 else \
            float(current.std(unbiased=False).item()) / mean
        output.append(BlockTraceEstimate(
            block=name,
            estimates=tuple(values),
            mean=mean,
            standard_error=standard_error,
            normalized_mean=mean / float(parameter.numel()),
            coefficient_of_variation=coefficient,
            parameters=int(parameter.numel()),
        ))
    return tuple(output)
