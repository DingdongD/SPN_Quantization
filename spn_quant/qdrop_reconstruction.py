"""Official-aligned joint weight and activation QDrop reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping

import torch
import torch.nn as nn

from spn_quant.adaptive_rounding import (
    AdaptiveRoundingController,
    CosineTemperatureDecay,
    is_supported_weight_module,
)
from spn_quant.deployment_contract import export_rounding_contracts
from spn_quant.strict_reconstruction import (
    _rounding_regularization,
    _stack_nested,
    move_to,
    strict_reconstruction_loss,
)


class QDropReconstructionError(RuntimeError):
    pass


@dataclass
class QDropCalibrationRecord:
    quantized_inputs: tuple[object, ...]
    full_precision_inputs: tuple[object, ...]
    reference: object


@dataclass(frozen=True)
class QDropOptimizerConfig:
    steps: int
    batch_size: int
    weight_learning_rate: float
    activation_learning_rate: float
    round_loss_weight: float
    warmup_fraction: float
    beta_start: float
    beta_end: float
    loss_power: float
    quant_probability: float
    seed: int

    def __post_init__(self):
        if int(self.steps) <= 0:
            raise ValueError("QDrop steps must be positive")
        if int(self.batch_size) <= 0:
            raise ValueError("QDrop batch size must be positive")
        if not math.isfinite(float(self.weight_learning_rate)) or \
                float(self.weight_learning_rate) <= 0.0:
            raise ValueError("QDrop weight learning rate must be positive")
        if not math.isfinite(float(self.activation_learning_rate)) or \
                float(self.activation_learning_rate) <= 0.0:
            raise ValueError("QDrop activation learning rate must be positive")
        if not math.isfinite(float(self.round_loss_weight)) or \
                float(self.round_loss_weight) < 0.0:
            raise ValueError("QDrop round loss weight cannot be negative")
        if not 0.0 <= float(self.warmup_fraction) < 1.0:
            raise ValueError("QDrop warmup fraction must be in [0, 1)")
        if not math.isfinite(float(self.beta_start)) or \
                not math.isfinite(float(self.beta_end)) or \
                float(self.beta_start) < float(self.beta_end) or \
                float(self.beta_end) <= 0.0:
            raise ValueError("QDrop beta range is invalid")
        if not math.isfinite(float(self.loss_power)) or \
                float(self.loss_power) <= 0.0:
            raise ValueError("QDrop loss power must be positive")
        if not math.isfinite(float(self.quant_probability)) or \
                not 0.0 <= float(self.quant_probability) <= 1.0:
            raise ValueError("QDrop quant probability must be in [0, 1]")


@dataclass
class QDropReconstructionResult:
    before_loss: float
    after_loss: float
    history: list[dict[str, float]]
    weight_contracts: dict[str, dict[str, object]]
    activation_contracts: dict[str, dict[str, object]]


def mix_qdrop_inputs(quantized, full_precision, quant_probability,
                     generator):
    probability = float(quant_probability)
    if not math.isfinite(probability) or \
            probability < 0.0 or probability > 1.0:
        raise ValueError("quant_probability must be in [0, 1]")
    if torch.is_tensor(quantized):
        if not torch.is_tensor(full_precision):
            raise TypeError("QDrop input structure does not match")
        if quantized.shape != full_precision.shape:
            raise ValueError("QDrop input tensor shape does not match")
        if quantized.dtype != full_precision.dtype:
            raise TypeError("QDrop input tensor dtype does not match")
        if not quantized.is_floating_point():
            raise TypeError("QDrop input tensors must be floating point")
        mask = torch.rand(
            quantized.shape,
            generator=generator,
            device="cpu",
            dtype=torch.float32).to(device=quantized.device) < probability
        return torch.where(mask, quantized, full_precision)
    if isinstance(quantized, Mapping):
        if not isinstance(full_precision, Mapping):
            raise TypeError("QDrop input structure does not match")
        if tuple(quantized) != tuple(full_precision):
            raise ValueError("QDrop input dictionary keys do not match")
        return type(quantized)((
            key,
            mix_qdrop_inputs(
                quantized[key], full_precision[key], probability, generator),
        ) for key in quantized)
    if isinstance(quantized, tuple):
        if not isinstance(full_precision, tuple):
            raise TypeError("QDrop input structure does not match")
        if len(quantized) != len(full_precision):
            raise ValueError("QDrop input tuple length does not match")
        return tuple(
            mix_qdrop_inputs(left, right, probability, generator)
            for left, right in zip(quantized, full_precision))
    if isinstance(quantized, list):
        if not isinstance(full_precision, list):
            raise TypeError("QDrop input structure does not match")
        if len(quantized) != len(full_precision):
            raise ValueError("QDrop input list length does not match")
        return [
            mix_qdrop_inputs(left, right, probability, generator)
            for left, right in zip(quantized, full_precision)]
    if type(quantized) is not type(full_precision) or \
            quantized != full_precision:
        raise ValueError("QDrop non-tensor input values do not match")
    return quantized


def _iter_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for key in value:
            yield from _iter_tensors(value[key])
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)


class QDropBlockReconstructor(object):
    def __init__(self, block, target, activation_bank,
                 weight_config, optimizer_config, contract_prefix):
        if not isinstance(block, nn.Module):
            raise TypeError("QDrop reconstruction block must be nn.Module")
        self.block = block
        self.target = str(target)
        self.activation_bank = activation_bank
        self.weight_config = weight_config
        self.config = optimizer_config
        self.contract_prefix = str(contract_prefix)
        if not self.target:
            raise ValueError("QDrop reconstruction target cannot be empty")
        self.rounding = AdaptiveRoundingController(
            self.block, self.weight_config)
        self._requires_grad = {}
        self._was_training = bool(self.block.training)

    @staticmethod
    def _device(module):
        for parameter in module.parameters():
            return parameter.device
        return torch.device("cpu")

    def _weight_names(self):
        names = []
        if is_supported_weight_module(self.block):
            names.append("")
        for name, module in self.block.named_modules():
            if name and is_supported_weight_module(module):
                names.append(name)
        if not names:
            raise ValueError("QDrop block contains no supported weights")
        return tuple(sorted(set(names)))

    def _freeze_parameters(self):
        self._requires_grad = dict(
            (name, bool(parameter.requires_grad))
            for name, parameter in self.block.named_parameters())
        for parameter in self.block.parameters():
            parameter.requires_grad_(False)

    def _restore_parameters(self):
        for name, parameter in self.block.named_parameters():
            original = name.replace(
                "parametrizations.weight.original", "weight")
            parameter.requires_grad_(self._requires_grad[original])
        self.block.train(self._was_training)

    @staticmethod
    def _validate_records(records):
        if not records:
            raise ValueError("QDrop calibration records cannot be empty")
        for record in records:
            if not isinstance(record.quantized_inputs, tuple) or \
                    not isinstance(record.full_precision_inputs, tuple):
                raise TypeError("QDrop block inputs must be tuples")
            mix_qdrop_inputs(
                record.quantized_inputs,
                record.full_precision_inputs,
                0.5,
                torch.Generator().manual_seed(0))
            tensors = list(_iter_tensors(record.reference))
            if not tensors:
                raise TypeError("QDrop reference contains no tensors")
            if any(not bool(torch.isfinite(tensor).all().item())
                   for tensor in tensors):
                raise ValueError("QDrop reference contains non-finite values")

    def _batch(self, records, generator):
        count = min(int(self.config.batch_size), len(records))
        indices = torch.randperm(
            len(records), generator=generator)[:count].tolist()
        selected = [records[index] for index in indices]
        quantized = _stack_nested([
            record.quantized_inputs for record in selected])
        full_precision = _stack_nested([
            record.full_precision_inputs for record in selected])
        inputs = mix_qdrop_inputs(
            quantized,
            full_precision,
            self.config.quant_probability,
            generator)
        reference = _stack_nested([
            record.reference for record in selected])
        return inputs, reference

    def _evaluate(self, records):
        device = self._device(self.block)
        losses = []
        was_training = self.block.training
        self.block.eval()
        with torch.no_grad():
            for record in records:
                candidate = self.block(
                    *move_to(record.quantized_inputs, device))
                loss = strict_reconstruction_loss(
                    move_to(record.reference, device),
                    candidate,
                    mode="mse",
                    p=self.config.loss_power)
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError(
                        "non-finite QDrop deterministic loss")
                losses.append(float(loss.item()))
        self.block.train(was_training)
        return sum(losses) / float(len(losses))

    @staticmethod
    def _require_gradients(parameters, family):
        missing = [index for index, parameter in enumerate(parameters)
                   if parameter.grad is None]
        if missing:
            raise RuntimeError(
                "QDrop %s parameters have no gradients: %s" %
                (family, missing))

    def fit(self, records):
        records = list(records)
        self._validate_records(records)
        self._freeze_parameters()
        self.rounding.install(self._weight_names())
        self.activation_bank.reconstruct(
            self.target, quant_probability=1.0)
        self.rounding.set_soft_targets(False)
        before = self._evaluate(records)
        self.rounding.set_soft_targets(True)
        self.activation_bank.set_quant_probability(
            self.target, self.config.quant_probability)

        weight_parameters = tuple(self.rounding.parameters())
        activation_parameters = tuple(
            self.activation_bank.parameters_for(self.target))
        overlap = set(id(parameter) for parameter in weight_parameters) & \
            set(id(parameter) for parameter in activation_parameters)
        if overlap:
            raise RuntimeError("QDrop optimizers have overlapping parameters")
        weight_optimizer = torch.optim.Adam(
            weight_parameters,
            lr=float(self.config.weight_learning_rate))
        activation_optimizer = torch.optim.Adam(
            activation_parameters,
            lr=float(self.config.activation_learning_rate))
        activation_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            activation_optimizer,
            T_max=int(self.config.steps),
            eta_min=0.0)
        beta_schedule = CosineTemperatureDecay(
            self.config.steps,
            self.config.warmup_fraction,
            self.config.beta_start,
            self.config.beta_end)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.config.seed))
        device = self._device(self.block)
        history = []
        self.block.eval()

        for step in range(self.config.steps):
            inputs, reference = self._batch(records, generator)
            weight_optimizer.zero_grad(set_to_none=True)
            activation_optimizer.zero_grad(set_to_none=True)
            candidate = self.block(*move_to(inputs, device))
            reconstruction = strict_reconstruction_loss(
                move_to(reference, device),
                candidate,
                mode="mse",
                p=self.config.loss_power)
            beta = beta_schedule(step)
            if beta is None or self.config.round_loss_weight == 0.0:
                round_loss = reconstruction.new_tensor(0.0)
            else:
                round_loss = _rounding_regularization(
                    self.rounding, beta)
            total = reconstruction + \
                float(self.config.round_loss_weight) * round_loss
            if not bool(torch.isfinite(total).item()):
                raise FloatingPointError("non-finite QDrop reconstruction loss")
            total.backward()
            self._require_gradients(weight_parameters, "weight")
            self._require_gradients(activation_parameters, "activation")
            weight_optimizer.step()
            activation_optimizer.step()
            activation_scheduler.step()
            for parameter in activation_parameters:
                if not bool(torch.isfinite(parameter).all().item()):
                    raise FloatingPointError(
                        "non-finite QDrop activation parameter")
            history.append({
                "step": float(step + 1),
                "total_loss": float(total.detach().item()),
                "reconstruction_loss": float(reconstruction.detach().item()),
                "round_loss": float(round_loss.detach().item()),
                "beta": float(beta) if beta is not None else float("nan"),
                "activation_learning_rate": float(
                    activation_scheduler.get_last_lr()[0]),
            })

        self.rounding.set_soft_targets(False)
        self.activation_bank.set_quant_probability(self.target, 1.0)
        after = self._evaluate(records)
        if not math.isfinite(after) or after > before:
            raise QDropReconstructionError(
                "hard QDrop result is worse than its initial state: "
                "before=%.9f after=%.9f" % (before, after))
        weight_contracts = export_rounding_contracts(
            self.rounding, prefix=self.contract_prefix)
        self.rounding.harden()
        self.activation_bank.freeze_target(self.target)
        all_activation_contracts = self.activation_bank.contracts()
        activation_contracts = dict(
            (site.site, all_activation_contracts[site.site])
            for site in self.activation_bank.plan.activation_sites
            if site.owner_name == self.target)
        self._restore_parameters()
        return QDropReconstructionResult(
            before_loss=before,
            after_loss=after,
            history=history,
            weight_contracts=weight_contracts,
            activation_contracts=activation_contracts,
        )
