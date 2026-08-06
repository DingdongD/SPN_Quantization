"""Strict weight-only AdaRound and BRECQ reconstruction.

This module follows the original reconstruction contract: optimize adaptive
rounding on cached mini-batches, use sum-reduced rounding regularization with a
beta schedule, and optionally use diagonal/full Fisher output weighting. It is
weight-only by design; activation reconstruction stays in the semantic
extension until activation-edge ownership is deployment-identical.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from spn_quant.adaptive_rounding import (
    AdaptiveRoundingConfig,
    AdaptiveRoundingController,
    LinearTemperatureDecay,
    is_supported_weight_module,
)
from spn_quant.deployment_contract import export_rounding_contracts


@dataclass(frozen=True)
class StrictReconstructionConfig:
    steps: int = 20000
    batch_size: int = 32
    learning_rate: float = 1.0e-3
    round_loss_weight: float = 1.0e-2
    warmup_fraction: float = 0.2
    beta_start: float = 20.0
    beta_end: float = 2.0
    loss: str = "mse"
    p: float = 2.0
    seed: int = 2026

    def __post_init__(self) -> None:
        if int(self.steps) <= 0:
            raise ValueError("steps must be positive")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
        if float(self.round_loss_weight) < 0.0:
            raise ValueError("round_loss_weight cannot be negative")
        if not 0.0 <= float(self.warmup_fraction) < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if self.loss not in ("mse", "fisher_diag", "fisher_full"):
            raise ValueError(
                "unknown strict reconstruction loss: %s" % self.loss)
        if float(self.p) <= 0.0:
            raise ValueError("p must be positive")

    def manifest(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrictCalibrationRecord:
    inputs: Tuple[Any, ...]
    reference: Any
    gradients: Optional[Any] = None


@dataclass
class StrictReconstructionResult:
    before_loss: float
    after_loss: float
    best_soft_loss: float
    history: List[Dict[str, float]]
    weight_manifest: List[Dict[str, Any]]
    weight_contracts: Dict[str, Dict[str, Any]]

    def manifest(self) -> Dict[str, Any]:
        return {
            "before_loss": self.before_loss,
            "after_loss": self.after_loss,
            "best_soft_loss": self.best_soft_loss,
            "steps": len(self.history),
            "weight_sites": len(self.weight_manifest),
            "contract_sites": len(self.weight_contracts),
        }


def _map_nested(value: Any, function):
    if torch.is_tensor(value):
        return function(value)
    if isinstance(value, Mapping):
        return type(value)((
            key, _map_nested(item, function))
            for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_map_nested(item, function) for item in value)
    if isinstance(value, list):
        return [_map_nested(item, function) for item in value]
    return value


def move_to(value: Any, device: torch.device) -> Any:
    return _map_nested(value, lambda tensor: tensor.to(device))


def detach_cpu(value: Any) -> Any:
    return _map_nested(
        value, lambda tensor: tensor.detach().cpu().clone())


def _stack_nested(values: Sequence[Any]) -> Any:
    if not values:
        raise ValueError("cannot stack an empty nested value")
    first = values[0]
    if torch.is_tensor(first):
        if first.ndim == 0:
            return torch.stack(list(values), dim=0)
        return torch.cat(list(values), dim=0)
    if isinstance(first, Mapping):
        return type(first)((
            key, _stack_nested([value[key] for value in values]))
            for key in first)
    if isinstance(first, tuple):
        return tuple(
            _stack_nested([value[index] for value in values])
            for index in range(len(first)))
    if isinstance(first, list):
        return [
            _stack_nested([value[index] for value in values])
            for index in range(len(first))]
    if all(value == first for value in values):
        return first
    raise TypeError(
        "non-tensor calibration values differ across records")


def _tensor_triplets(
        reference: Any, candidate: Any,
        gradients: Optional[Any] = None
        ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    if torch.is_tensor(reference):
        if not torch.is_tensor(candidate):
            raise TypeError(
                "candidate structure does not match reference")
        gradient = gradients if torch.is_tensor(gradients) else None
        yield reference, candidate, gradient
        return
    if isinstance(reference, Mapping):
        if not isinstance(candidate, Mapping):
            raise TypeError(
                "candidate structure does not match reference")
        for key in reference:
            gradient = (
                gradients.get(key)
                if isinstance(gradients, Mapping) else None)
            yield from _tensor_triplets(
                reference[key], candidate[key], gradient)
        return
    if isinstance(reference, (tuple, list)):
        if (not isinstance(candidate, (tuple, list)) or
                len(reference) != len(candidate)):
            raise TypeError(
                "candidate structure does not match reference")
        for index, (left, right) in enumerate(zip(reference, candidate)):
            gradient = (
                gradients[index]
                if isinstance(gradients, (tuple, list)) else None)
            yield from _tensor_triplets(left, right, gradient)
        return
    raise TypeError(
        "strict reconstruction output must contain tensors")


def strict_reconstruction_loss(
        reference: Any, candidate: Any,
        gradients: Optional[Any] = None,
        mode: str = "mse", p: float = 2.0) -> torch.Tensor:
    losses = []
    for ref, pred, gradient in _tensor_triplets(
            reference, candidate, gradients):
        difference = (pred - ref).float()
        if mode == "mse":
            losses.append(
                difference.abs().pow(float(p)).mean())
            continue
        if gradient is None:
            raise ValueError(
                "Fisher reconstruction requires output gradients")
        grad = gradient.float()
        if mode == "fisher_diag":
            weighted = difference.pow(2) * grad.pow(2)
            if weighted.ndim > 1:
                losses.append(weighted.sum(dim=1).mean())
            else:
                losses.append(weighted.mean())
        elif mode == "fisher_full":
            a = difference.abs()
            g = grad.abs()
            dimensions = tuple(range(1, a.ndim))
            if not dimensions:
                losses.append((a * g).pow(2).mean() / 100.0)
            else:
                batch_dot = (a * g).sum(dim=dimensions)
                shape = [a.shape[0]] + [1] * (a.ndim - 1)
                losses.append(
                    (batch_dot.reshape(shape) * a * g).mean() /
                    100.0)
        else:
            raise ValueError(
                "unknown reconstruction loss: %s" % mode)
    if not losses:
        raise ValueError(
            "strict reconstruction output contained no tensors")
    return torch.stack(losses).mean()


def _rounding_regularization(
        controller: AdaptiveRoundingController,
        beta: float) -> torch.Tensor:
    losses = []
    for parametrization in controller.parametrizations.values():
        values = parametrization.soft_rounding()
        losses.append(
            1.0 - ((values - 0.5).abs() * 2.0).pow(
                float(beta)))
    if not losses:
        raise RuntimeError(
            "no adaptive rounding parameters installed")
    return sum(loss.sum() for loss in losses)


class StrictBlockReconstructor(object):
    """Original-style weight-rounding reconstruction for a layer or block."""

    def __init__(
            self, block: nn.Module,
            weight_config: AdaptiveRoundingConfig = AdaptiveRoundingConfig(),
            reconstruction_config: StrictReconstructionConfig =
            StrictReconstructionConfig(),
            contract_prefix: str = "") -> None:
        self.block = block
        self.weight_config = weight_config
        self.config = reconstruction_config
        self.contract_prefix = str(contract_prefix)
        self.rounding = AdaptiveRoundingController(
            block, weight_config)
        self._requires_grad = {}
        self._was_training = bool(block.training)

    @staticmethod
    def _device(module: nn.Module) -> torch.device:
        for parameter in module.parameters():
            return parameter.device
        return torch.device("cpu")

    def _weight_names(self) -> List[str]:
        names = []
        if is_supported_weight_module(self.block):
            names.append("")
        for name, module in self.block.named_modules():
            if name and is_supported_weight_module(module):
                names.append(name)
        if not names:
            raise ValueError(
                "block contains no supported weight modules")
        return sorted(set(names))

    def _freeze_parameters(self) -> None:
        self._requires_grad = {
            name: bool(parameter.requires_grad)
            for name, parameter in self.block.named_parameters()
        }
        for parameter in self.block.parameters():
            parameter.requires_grad_(False)

    def _restore_parameters(self) -> None:
        for name, parameter in self.block.named_parameters():
            original = name.replace(
                "parametrizations.weight.original", "weight")
            parameter.requires_grad_(
                self._requires_grad.get(original, False))
        self.block.train(self._was_training)

    def _batch(
            self, records: Sequence[StrictCalibrationRecord],
            generator: torch.Generator) -> StrictCalibrationRecord:
        count = min(int(self.config.batch_size), len(records))
        indices = torch.randperm(
            len(records), generator=generator)[:count].tolist()
        selected = [records[index] for index in indices]
        return StrictCalibrationRecord(
            inputs=_stack_nested([
                record.inputs for record in selected]),
            reference=_stack_nested([
                record.reference for record in selected]),
            gradients=(
                _stack_nested([
                    record.gradients for record in selected])
                if selected[0].gradients is not None else None),
        )

    def _evaluate(
            self, records: Sequence[StrictCalibrationRecord]) -> float:
        device = self._device(self.block)
        losses = []
        was_training = self.block.training
        self.block.eval()
        with torch.no_grad():
            for record in records:
                candidate = self.block(
                    *move_to(record.inputs, device))
                losses.append(float(strict_reconstruction_loss(
                    move_to(record.reference, device),
                    candidate,
                    move_to(record.gradients, device)
                    if record.gradients is not None else None,
                    mode=self.config.loss,
                    p=self.config.p).item()))
        self.block.train(was_training)
        return sum(losses) / max(len(losses), 1)

    def fit(
            self, records: Sequence[StrictCalibrationRecord]
            ) -> StrictReconstructionResult:
        records = list(records)
        if not records:
            raise ValueError(
                "calibration records cannot be empty")
        if (self.config.loss != "mse" and
                any(record.gradients is None for record in records)):
            raise ValueError(
                "Fisher reconstruction requires gradients for every record")
        self._freeze_parameters()
        self.rounding.install(self._weight_names())
        self.rounding.set_soft_targets(False)
        before = self._evaluate(records)
        self.rounding.set_soft_targets(True)

        optimizer = torch.optim.Adam(
            list(self.rounding.parameters()),
            lr=float(self.config.learning_rate))
        schedule = LinearTemperatureDecay(
            self.config.steps,
            self.config.warmup_fraction,
            self.config.beta_start,
            self.config.beta_end)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.config.seed))
        device = self._device(self.block)
        history = []
        best_soft = float("inf")
        self.block.eval()

        for step in range(self.config.steps):
            record = self._batch(records, generator)
            optimizer.zero_grad(set_to_none=True)
            candidate = self.block(
                *move_to(record.inputs, device))
            reconstruction = strict_reconstruction_loss(
                move_to(record.reference, device),
                candidate,
                move_to(record.gradients, device)
                if record.gradients is not None else None,
                mode=self.config.loss,
                p=self.config.p)
            beta = schedule(step)
            if (beta is None or
                    self.config.round_loss_weight == 0.0):
                round_loss = reconstruction.new_tensor(0.0)
            else:
                round_loss = _rounding_regularization(
                    self.rounding, beta)
            total = (
                reconstruction +
                float(self.config.round_loss_weight) * round_loss)
            if not torch.isfinite(total):
                raise FloatingPointError(
                    "non-finite strict reconstruction loss")
            total.backward()
            optimizer.step()
            best_soft = min(
                best_soft, float(total.detach().item()))
            history.append({
                "step": float(step + 1),
                "total_loss": float(total.detach().item()),
                "reconstruction_loss": float(
                    reconstruction.detach().item()),
                "round_loss": float(round_loss.detach().item()),
                "beta": (
                    float(beta) if beta is not None
                    else float("nan")),
            })

        self.rounding.set_soft_targets(False)
        after = self._evaluate(records)
        contracts = export_rounding_contracts(
            self.rounding, prefix=self.contract_prefix)
        weight_manifest = self.rounding.harden()
        self._restore_parameters()
        return StrictReconstructionResult(
            before_loss=before,
            after_loss=after,
            best_soft_loss=best_soft,
            history=history,
            weight_manifest=weight_manifest,
            weight_contracts=contracts,
        )

    def close(self, restore_weights: bool = False) -> None:
        if restore_weights and self.rounding.parametrizations:
            self.rounding.remove()
        self._restore_parameters()
