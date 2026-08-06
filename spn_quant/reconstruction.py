"""Layer and semantic-block reconstruction for low-bit PTQ.

AdaRound is represented by a one-module reconstruction. BRECQ uses the same
adaptive rounding parameters but optimizes all supported weights inside a block
jointly, optionally with learnable activation step sizes and Fisher-weighted
output reconstruction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from spn_quant.adaptive_rounding import (
    AdaptiveRoundingConfig,
    AdaptiveRoundingController,
    LinearTemperatureDecay,
    is_supported_weight_module,
)


@dataclass(frozen=True)
class ReconstructionConfig:
    steps: int = 1000
    learning_rate: float = 1.0e-3
    activation_learning_rate: float = 1.0e-4
    round_loss_weight: float = 1.0e-2
    warmup_fraction: float = 0.2
    beta_start: float = 20.0
    beta_end: float = 2.0
    loss: str = "mse"
    activation_bits: Optional[int] = None
    qdrop_probability: float = 0.0
    seed: int = 2026
    hard_eval_interval: int = 50

    def __post_init__(self) -> None:
        if int(self.steps) <= 0:
            raise ValueError("steps must be positive")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
        if float(self.activation_learning_rate) <= 0.0:
            raise ValueError("activation_learning_rate must be positive")
        if float(self.round_loss_weight) < 0.0:
            raise ValueError("round_loss_weight cannot be negative")
        if self.loss not in ("mse", "fisher"):
            raise ValueError("loss must be mse or fisher")
        if self.activation_bits is not None and int(self.activation_bits) < 2:
            raise ValueError("activation_bits must be at least 2")
        if not 0.0 <= float(self.qdrop_probability) < 1.0:
            raise ValueError("qdrop_probability must be in [0, 1)")
        if int(self.hard_eval_interval) <= 0:
            raise ValueError("hard_eval_interval must be positive")

    def manifest(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationRecord:
    inputs: Tuple[Any, ...]
    reference: Any
    gradients: Optional[Any] = None


@dataclass
class ReconstructionResult:
    before_loss: float
    after_loss: float
    best_loss: float
    history: List[Dict[str, float]]
    weight_manifest: List[Dict[str, Any]]
    activation_manifest: List[Dict[str, Any]]

    def manifest(self) -> Dict[str, Any]:
        return {
            "before_loss": self.before_loss,
            "after_loss": self.after_loss,
            "best_loss": self.best_loss,
            "steps": len(self.history),
            "weight_sites": len(self.weight_manifest),
            "activation_sites": len(self.activation_manifest),
        }


def _map_nested(value: Any, function: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    if torch.is_tensor(value):
        return function(value)
    if isinstance(value, Mapping):
        return type(value)((key, _map_nested(item, function))
                           for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_map_nested(item, function) for item in value)
    if isinstance(value, list):
        return [_map_nested(item, function) for item in value]
    return value


def detach_cpu(value: Any) -> Any:
    return _map_nested(value, lambda tensor: tensor.detach().cpu().clone())


def move_to(value: Any, device: torch.device) -> Any:
    return _map_nested(value, lambda tensor: tensor.to(device))


def _retain_grad(value: Any) -> None:
    def retain(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.requires_grad:
            tensor.retain_grad()
        return tensor
    _map_nested(value, retain)


def _extract_grad(value: Any) -> Any:
    def get(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.grad is None:
            return torch.zeros_like(tensor, memory_format=torch.preserve_format)
        return tensor.grad.detach().clone()
    return _map_nested(value, get)


def _tensor_triplets(reference: Any, candidate: Any,
                     gradients: Optional[Any] = None
                     ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    if torch.is_tensor(reference):
        if not torch.is_tensor(candidate):
            raise TypeError("candidate structure does not match reference")
        gradient = gradients if torch.is_tensor(gradients) else None
        yield reference, candidate, gradient
        return
    if isinstance(reference, Mapping):
        if not isinstance(candidate, Mapping):
            raise TypeError("candidate structure does not match reference")
        for key in sorted(reference):
            if key not in candidate:
                raise KeyError("candidate output is missing key: %s" % key)
            gradient = gradients.get(key) if isinstance(gradients, Mapping) else None
            for item in _tensor_triplets(reference[key], candidate[key], gradient):
                yield item
        return
    if isinstance(reference, (list, tuple)):
        if not isinstance(candidate, (list, tuple)) or len(reference) != len(candidate):
            raise TypeError("candidate structure does not match reference")
        for index, (left, right) in enumerate(zip(reference, candidate)):
            gradient = gradients[index] if isinstance(gradients, (list, tuple)) else None
            for item in _tensor_triplets(left, right, gradient):
                yield item
        return
    raise TypeError("reconstruction output must contain tensors")


def reconstruction_loss(reference: Any, candidate: Any,
                        gradients: Optional[Any] = None,
                        mode: str = "mse") -> torch.Tensor:
    losses = []
    for ref, pred, gradient in _tensor_triplets(reference, candidate, gradients):
        difference_sq = (pred - ref).float().pow(2)
        if mode == "fisher":
            if gradient is None:
                raise ValueError("Fisher reconstruction requires output gradients")
            difference_sq = difference_sq * gradient.float().pow(2)
        losses.append(difference_sq.mean())
    if not losses:
        raise ValueError("reconstruction output contained no tensors")
    return torch.stack(losses).mean()


class LearnedActivationQuantizer(nn.Module):
    """Learnable symmetric/unsigned activation scale with an STE round."""

    def __init__(self, bits: int, maximum: float, unsigned: bool) -> None:
        super(LearnedActivationQuantizer, self).__init__()
        self.bits = int(bits)
        self.unsigned = bool(unsigned)
        if self.unsigned:
            self.qmin = 0
            self.qmax = 2 ** self.bits - 1
        else:
            self.qmax = 2 ** (self.bits - 1) - 1
            self.qmin = -self.qmax
        initial = max(float(maximum) / float(max(self.qmax, 1)), 1.0e-8)
        self.log_scale = nn.Parameter(torch.tensor(math.log(initial), dtype=torch.float32))

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp().clamp_min(1.0e-8)

    def forward(self, tensor: torch.Tensor, qdrop_probability: float = 0.0) -> torch.Tensor:
        scale = self.scale.to(device=tensor.device, dtype=tensor.dtype)
        scaled = tensor / scale
        rounded = scaled + (torch.round(scaled) - scaled).detach()
        codes = rounded.clamp(self.qmin, self.qmax)
        quantized = codes * scale
        if self.training and qdrop_probability > 0.0:
            mask_shape = [tensor.shape[0]] + [1] * (tensor.ndim - 1)
            keep_fp = torch.rand(mask_shape, device=tensor.device) < qdrop_probability
            quantized = torch.where(keep_fp, tensor, quantized)
        return quantized

    def manifest(self, site: str) -> Dict[str, Any]:
        scale = float(self.scale.detach().cpu().item())
        return {
            "site": site,
            "bits": self.bits,
            "unsigned": int(self.unsigned),
            "scale": scale,
            "maximum": scale * self.qmax,
            "qmin": self.qmin,
            "qmax": self.qmax,
        }


class ActivationReconstructionController(nn.Module):
    """Observe and learn the first tensor input scale of selected modules."""

    def __init__(self, root: nn.Module, module_names: Sequence[str], bits: int,
                 qdrop_probability: float = 0.0) -> None:
        super(ActivationReconstructionController, self).__init__()
        self.root = root
        self.module_names = tuple(module_names)
        self.bits = int(bits)
        self.qdrop_probability = float(qdrop_probability)
        self.mode = "bypass"
        self.ranges = {}  # type: Dict[str, List[float]]
        self.quantizers = nn.ModuleList()
        self._quantizer_index = {}  # type: Dict[str, int]
        self.handles = []
        modules = dict(root.named_modules())
        for name in self.module_names:
            if name not in modules:
                raise KeyError("unknown activation site module: %s" % name)
            module = modules[name]
            self.handles.append(module.register_forward_pre_hook(
                self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: Tuple[Any, ...]):
            del module
            if not inputs or not torch.is_tensor(inputs[0]):
                return None
            tensor = inputs[0]
            if self.mode == "observe":
                current = self.ranges.setdefault(name, [float("inf"), float("-inf")])
                current[0] = min(current[0], float(tensor.detach().min().item()))
                current[1] = max(current[1], float(tensor.detach().max().item()))
                return None
            if self.mode != "quantize":
                return None
            quantizer = self.quantizers[self._quantizer_index[name]]
            quantized = quantizer(
                tensor, qdrop_probability=self.qdrop_probability)
            return (quantized,) + tuple(inputs[1:])
        return hook

    def observe(self) -> None:
        self.mode = "observe"
        self.ranges = {}

    def freeze(self) -> None:
        missing = [name for name in self.module_names if name not in self.ranges]
        if missing:
            raise RuntimeError("unobserved activation sites: %s" % missing)
        for name in self.module_names:
            minimum, maximum = self.ranges[name]
            unsigned = minimum >= 0.0
            extent = maximum if unsigned else max(abs(minimum), abs(maximum))
            self._quantizer_index[name] = len(self.quantizers)
            self.quantizers.append(LearnedActivationQuantizer(
                self.bits, extent, unsigned=unsigned))
        self.mode = "quantize"

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.quantizers.parameters()

    def manifest(self) -> List[Dict[str, Any]]:
        return [self.quantizers[self._quantizer_index[name]].manifest(name)
                for name in sorted(self._quantizer_index)]

    def close(self) -> None:
        self.mode = "bypass"
        for handle in self.handles:
            handle.remove()
        self.handles = []


class ModuleIOCache(object):
    """Capture positional module inputs, outputs, and optional output gradients."""

    def __init__(self, model: nn.Module, module: nn.Module,
                 expected_calls: int = 1) -> None:
        self.model = model
        self.module = module
        self.expected_calls = int(expected_calls)
        self._inputs = []
        self._outputs = []
        self._handles = [
            module.register_forward_pre_hook(self._pre_hook),
            module.register_forward_hook(self._post_hook),
        ]

    def _pre_hook(self, module: nn.Module, inputs: Tuple[Any, ...]) -> None:
        del module
        self._inputs.append(inputs)

    def _post_hook(self, module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        del module, inputs
        self._outputs.append(output)
        _retain_grad(output)

    def capture(self, model_args: Tuple[Any, ...],
                loss_closure: Optional[Callable[[Any], torch.Tensor]] = None
                ) -> List[CalibrationRecord]:
        self._inputs = []
        self._outputs = []
        self.model.zero_grad(set_to_none=True)
        if loss_closure is None:
            with torch.no_grad():
                model_output = self.model(*model_args)
            del model_output
        else:
            model_output = self.model(*model_args)
            loss = loss_closure(model_output)
            if loss.ndim != 0:
                loss = loss.mean()
            loss.backward()
        if len(self._inputs) != self.expected_calls or len(self._outputs) != self.expected_calls:
            raise RuntimeError("expected %d module calls but captured inputs=%d outputs=%d" % (
                self.expected_calls, len(self._inputs), len(self._outputs)))
        records = []
        for inputs, output in zip(self._inputs, self._outputs):
            gradients = _extract_grad(output) if loss_closure is not None else None
            records.append(CalibrationRecord(
                inputs=detach_cpu(inputs), reference=detach_cpu(output),
                gradients=detach_cpu(gradients) if gradients is not None else None))
        self.model.zero_grad(set_to_none=True)
        return records

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


class SemanticBlockReconstructor(object):
    """Joint AdaRound/BRECQ optimization over one module or semantic block."""

    def __init__(self, block: nn.Module,
                 weight_config: AdaptiveRoundingConfig = AdaptiveRoundingConfig(),
                 reconstruction_config: ReconstructionConfig = ReconstructionConfig()) -> None:
        self.block = block
        self.weight_config = weight_config
        self.config = reconstruction_config
        self.rounding = AdaptiveRoundingController(block, weight_config)
        self.activations = None  # type: Optional[ActivationReconstructionController]
        self._requires_grad = {}  # type: Dict[str, bool]
        self._was_training = bool(block.training)

    def _freeze_block_parameters(self) -> None:
        self._requires_grad = dict(
            (name, bool(parameter.requires_grad))
            for name, parameter in self.block.named_parameters())
        for parameter in self.block.parameters():
            parameter.requires_grad_(False)

    def _restore_block_parameters(self) -> None:
        for name, parameter in self.block.named_parameters():
            original_name = name.replace("parametrizations.weight.original", "weight")
            parameter.requires_grad_(self._requires_grad.get(original_name, False))
        self.block.train(self._was_training)

    def _weight_module_names(self) -> List[str]:
        output = []
        if is_supported_weight_module(self.block):
            output.append("")
        for name, module in self.block.named_modules():
            if name and is_supported_weight_module(module):
                output.append(name)
        return sorted(set(output))

    def _install(self) -> List[str]:
        names = self._weight_module_names()
        if not names:
            raise ValueError("block contains no supported weight modules")
        self.rounding.install(names)
        if self.config.activation_bits is not None:
            self.activations = ActivationReconstructionController(
                self.block, names, bits=self.config.activation_bits,
                qdrop_probability=self.config.qdrop_probability)
        return names

    @staticmethod
    def _device(module: nn.Module) -> torch.device:
        for parameter in module.parameters():
            return parameter.device
        return torch.device("cpu")

    def _evaluate(self, records: Sequence[CalibrationRecord]) -> float:
        device = self._device(self.block)
        losses = []
        was_training = self.block.training
        self.block.eval()
        with torch.no_grad():
            for record in records:
                inputs = move_to(record.inputs, device)
                reference = move_to(record.reference, device)
                gradients = (move_to(record.gradients, device)
                             if record.gradients is not None else None)
                candidate = self.block(*inputs)
                losses.append(float(reconstruction_loss(
                    reference, candidate, gradients, self.config.loss).item()))
        self.block.train(was_training)
        return sum(losses) / max(len(losses), 1)

    def _snapshot_parameters(self) -> Dict[str, List[torch.Tensor]]:
        snapshot = {
            "alpha": [item.alpha.detach().clone()
                      for item in self.rounding.parametrizations.values()],
            "activation": [],
        }
        if self.activations is not None:
            snapshot["activation"] = [
                parameter.detach().clone()
                for parameter in self.activations.parameters()]
        return snapshot

    def _restore_parameters(self, snapshot: Dict[str, List[torch.Tensor]]) -> None:
        with torch.no_grad():
            for parameter, value in zip(
                    self.rounding.parameters(), snapshot.get("alpha", [])):
                parameter.copy_(value.to(parameter.device))
            if self.activations is not None:
                for parameter, value in zip(
                        self.activations.parameters(),
                        snapshot.get("activation", [])):
                    parameter.copy_(value.to(parameter.device))

    def fit(self, records: Sequence[CalibrationRecord],
            harden: bool = True) -> ReconstructionResult:
        records = list(records)
        if not records:
            raise ValueError("calibration records cannot be empty")
        torch.manual_seed(int(self.config.seed))
        self._freeze_block_parameters()
        self._install()
        device = self._device(self.block)
        if self.activations is not None:
            self.activations.observe()
            was_training = self.block.training
            self.block.eval()
            with torch.no_grad():
                for record in records:
                    self.block(*move_to(record.inputs, device))
            self.block.train(was_training)
            self.activations.freeze()

        self.rounding.set_soft_targets(False)
        if self.activations is not None:
            self.activations.eval()
        before = self._evaluate(records)
        best_hard_loss = before
        best_snapshot = self._snapshot_parameters()
        self.rounding.set_soft_targets(True)
        if self.activations is not None:
            self.activations.train()

        parameter_groups = [{
            "params": list(self.rounding.parameters()),
            "lr": float(self.config.learning_rate),
        }]
        if self.activations is not None:
            parameter_groups.append({
                "params": list(self.activations.parameters()),
                "lr": float(self.config.activation_learning_rate),
            })
        optimizer = torch.optim.Adam(parameter_groups)
        schedule = LinearTemperatureDecay(
            self.config.steps, self.config.warmup_fraction,
            self.config.beta_start, self.config.beta_end)
        history = []
        best = float("inf")
        iterator = itertools.cycle(records)
        self.block.eval()

        for step in range(self.config.steps):
            record = next(iterator)
            inputs = move_to(record.inputs, device)
            reference = move_to(record.reference, device)
            gradients = (move_to(record.gradients, device)
                         if record.gradients is not None else None)
            optimizer.zero_grad(set_to_none=True)
            candidate = self.block(*inputs)
            reconstruction = reconstruction_loss(
                reference, candidate, gradients, self.config.loss)
            beta = schedule(step)
            if beta is None or self.config.round_loss_weight == 0.0:
                round_loss = reconstruction.new_tensor(0.0)
            else:
                round_loss = self.rounding.regularization(beta)
            total = reconstruction + self.config.round_loss_weight * round_loss
            if not torch.isfinite(total):
                raise FloatingPointError("non-finite reconstruction loss at step %d" % step)
            total.backward()
            optimizer.step()
            total_value = float(total.detach().item())
            best = min(best, total_value)
            history.append({
                "step": float(step + 1),
                "total_loss": total_value,
                "reconstruction_loss": float(reconstruction.detach().item()),
                "round_loss": float(round_loss.detach().item()),
                "beta": float(beta) if beta is not None else float("nan"),
            })
            should_evaluate_hard = (
                (step + 1) % int(self.config.hard_eval_interval) == 0 or
                step + 1 == self.config.steps)
            if should_evaluate_hard:
                self.rounding.set_soft_targets(False)
                if self.activations is not None:
                    self.activations.eval()
                hard_loss = self._evaluate(records)
                if hard_loss < best_hard_loss:
                    best_hard_loss = hard_loss
                    best_snapshot = self._snapshot_parameters()
                self.rounding.set_soft_targets(True)
                if self.activations is not None:
                    self.activations.train()

        self._restore_parameters(best_snapshot)
        self.rounding.set_soft_targets(False)
        if self.activations is not None:
            self.activations.eval()
        after = self._evaluate(records)
        weight_manifest = self.rounding.manifest()
        activation_manifest = (self.activations.manifest()
                               if self.activations is not None else [])
        if harden:
            weight_manifest = self.rounding.harden()
        self._restore_block_parameters()
        return ReconstructionResult(
            before_loss=before, after_loss=after, best_loss=best,
            history=history, weight_manifest=weight_manifest,
            activation_manifest=activation_manifest)

    def close(self, restore_weights: bool = False) -> None:
        if restore_weights and self.rounding.parametrizations:
            self.rounding.remove()
        if self.activations is not None:
            self.activations.close()
        self._restore_block_parameters()
