"""Orthogonal channel rotation for low-bit convolution activations."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from spn_quant.adapters.cspn import RotationBoundary


def random_orthogonal_matrix(channels: int, seed: int) -> torch.Tensor:
    channels = int(channels)
    if channels <= 0:
        raise ValueError("rotation channels must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(
        channels, channels, generator=generator, dtype=torch.float64)
    orthogonal, triangular = torch.linalg.qr(matrix)
    signs = torch.where(
        torch.diag(triangular) < 0.0,
        torch.full((channels,), -1.0, dtype=torch.float64),
        torch.ones(channels, dtype=torch.float64),
    )
    return (orthogonal * signs.unsqueeze(0)).to(torch.float32)


def hadamard_rotation_matrix(channels: int, seed: int) -> torch.Tensor:
    channels = int(channels)
    if channels <= 0 or channels & (channels - 1):
        raise ValueError("Hadamard channels must be a power of two")
    matrix = torch.ones(1, 1, dtype=torch.float32)
    while matrix.shape[0] < channels:
        matrix = torch.cat((
            torch.cat((matrix, matrix), dim=1),
            torch.cat((matrix, -matrix), dim=1),
        ), dim=0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    signs = torch.randint(
        0, 2, (channels,), generator=generator,
        dtype=torch.int64).to(torch.float32).mul_(2.0).sub_(1.0)
    return matrix.mul(signs.unsqueeze(0)).div(math.sqrt(channels))


def rotate_channels(tensor: torch.Tensor,
                    rotation: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError("channel rotation requires an NCHW tensor")
    if tensor.shape[1] != rotation.shape[1]:
        raise ValueError("rotation channel count does not match the tensor")
    return torch.einsum("oc,nchw->nohw", rotation.to(tensor), tensor)


def absorb_input_rotation(
        module: nn.Conv2d, rotation: torch.Tensor,
        channel_start: int = 0,
        channel_count: Optional[int] = None) -> None:
    if module.groups != 1:
        raise ValueError("rotation requires groups=1")
    if rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
        raise ValueError("rotation matrix must be square")
    count = int(rotation.shape[0]) \
        if channel_count is None else int(channel_count)
    start = int(channel_start)
    stop = start + count
    if count != rotation.shape[0] or start < 0 or stop > module.in_channels:
        raise ValueError("rotation slice does not match convolution channels")
    transformed = transform_input_weight(
        module.weight.data, rotation, start, count)
    module.weight.data.copy_(transformed)


def transform_input_weight(
        weight: torch.Tensor, rotation: torch.Tensor,
        channel_start: int = 0,
        channel_count: Optional[int] = None) -> torch.Tensor:
    if weight.ndim != 4:
        raise ValueError("input rotation requires convolution weight")
    if rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
        raise ValueError("rotation matrix must be square")
    count = int(rotation.shape[0]) \
        if channel_count is None else int(channel_count)
    start = int(channel_start)
    stop = start + count
    if count != rotation.shape[0] or start < 0 or stop > weight.shape[1]:
        raise ValueError("rotation slice does not match weight channels")
    output = weight.clone()
    output[:, start:stop] = torch.einsum(
        "oihw,ji->ojhw", weight[:, start:stop], rotation.to(weight))
    return output


class SignedActivationQuantizer:
    def __init__(self, bits: int, channel_maximum: torch.Tensor,
                 group_size: Optional[int]) -> None:
        self.bits = int(bits)
        if self.bits < 2:
            raise ValueError("signed activation bits must be at least two")
        if channel_maximum.ndim != 1 or channel_maximum.numel() == 0:
            raise ValueError("channel maxima must be a nonempty vector")
        self.channels = int(channel_maximum.numel())
        self.group_size = self.channels \
            if group_size is None else int(group_size)
        if self.group_size <= 0 or self.channels % self.group_size:
            raise ValueError("group size must divide the channel count")
        self.qmax = 2 ** (self.bits - 1) - 1
        maxima = []
        for start in range(0, self.channels, self.group_size):
            maxima.append(channel_maximum[
                start:start + self.group_size].max())
        grouped = torch.stack(maxima).to(torch.float32)
        self.scales = torch.where(
            grouped > 0.0, grouped / float(self.qmax),
            torch.ones_like(grouped))

    def quantize_with_codes(
            self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if tensor.ndim != 4 or tensor.shape[1] != self.channels:
            raise ValueError("quantizer requires the calibrated NCHW channels")
        values = []
        codes = []
        for group_index, start in enumerate(
                range(0, self.channels, self.group_size)):
            stop = start + self.group_size
            scale = self.scales[group_index].to(tensor)
            current_codes = torch.round(
                tensor[:, start:stop] / scale).clamp(
                    -self.qmax, self.qmax)
            values.append(current_codes * scale)
            codes.append(current_codes.to(torch.int32))
        return torch.cat(values, dim=1), torch.cat(codes, dim=1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.quantize_with_codes(tensor)[0]


class RotationBoundaryObserver:
    def __init__(self, channels: int, capacity: int = 1000000,
                 per_update: int = 8192) -> None:
        self.channels = int(channels)
        self.capacity = int(capacity)
        self.per_update = int(per_update)
        if self.channels <= 0 or self.capacity <= 0 or self.per_update <= 0:
            raise ValueError("observer dimensions must be positive")
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self.channel_absmax = torch.zeros(self.channels, dtype=torch.float32)
        self.channel_square_sum = torch.zeros(
            self.channels, dtype=torch.float64)
        self.scalar_count = 0
        self.scalar_sum = 0.0
        self.scalar_square_sum = 0.0
        self.scalar_cube_sum = 0.0
        self.scalar_fourth_sum = 0.0
        self.sample_chunks = []
        self.sample_values = 0

    @property
    def observed(self) -> bool:
        return self.scalar_count > 0

    def update(self, tensor: torch.Tensor) -> None:
        if tensor.ndim != 4 or tensor.shape[1] != self.channels:
            raise ValueError("rotation observer requires calibrated NCHW channels")
        detached = tensor.detach()
        if not torch.isfinite(detached).all():
            raise ValueError("rotation calibration activation must be finite")
        values = detached.to(torch.float64)
        self.minimum = min(self.minimum, float(values.min().item()))
        self.maximum = max(self.maximum, float(values.max().item()))
        moved = values.movedim(1, 0).reshape(self.channels, -1)
        absmax = moved.abs().amax(dim=1).to(torch.float32).cpu()
        self.channel_absmax = torch.maximum(self.channel_absmax, absmax)
        self.channel_square_sum += moved.square().sum(dim=1).cpu()
        self.scalar_count += int(values.numel())
        self.scalar_sum += float(values.sum().item())
        self.scalar_square_sum += float(values.square().sum().item())
        self.scalar_cube_sum += float(values.pow(3).sum().item())
        self.scalar_fourth_sum += float(values.pow(4).sum().item())

        remaining = self.capacity - self.sample_values
        positions = min(
            moved.shape[1], self.per_update // self.channels,
            remaining // self.channels)
        if positions > 0:
            indices = torch.linspace(
                0, moved.shape[1] - 1, positions,
                device=moved.device).round().to(torch.long)
            sample = moved.index_select(1, indices).to(torch.float32).cpu()
            self.sample_chunks.append(sample)
            self.sample_values += int(sample.numel())

    def quantizer(self, bits: int,
                  group_size: Optional[int]) -> SignedActivationQuantizer:
        if not self.observed:
            raise RuntimeError("rotation boundary was not observed")
        if self.minimum >= 0.0:
            raise RuntimeError("rotation boundary activation is not signed")
        return SignedActivationQuantizer(
            bits, self.channel_absmax, group_size)

    def _samples(self) -> torch.Tensor:
        if not self.sample_chunks:
            raise RuntimeError("rotation observer has no retained samples")
        return torch.cat(self.sample_chunks, dim=1).unsqueeze(0).unsqueeze(2)

    def statistics(self, quantizer: SignedActivationQuantizer) -> Dict[str, float]:
        samples = self._samples()
        quantized, codes = quantizer.quantize_with_codes(samples)
        absolute = samples.abs().reshape(-1)
        percentiles = torch.quantile(
            absolute, torch.tensor(
                [0.75, 0.99, 0.999, 0.9999], dtype=torch.float32))
        signal = float(samples.to(torch.float64).square().sum().item())
        error = float((quantized - samples).to(
            torch.float64).square().sum().item())
        sqnr = float("inf") if error == 0.0 else \
            10.0 * math.log10(signal / error)

        count = float(self.scalar_count)
        mean = self.scalar_sum / count
        second = self.scalar_square_sum / count
        third = self.scalar_cube_sum / count
        fourth = self.scalar_fourth_sum / count
        variance = second - mean * mean
        central_fourth = fourth - 4.0 * mean * third + \
            6.0 * mean * mean * second - 3.0 * mean ** 4
        kurtosis = central_fourth / (variance * variance) \
            if variance > 0.0 else 0.0
        channel_norm = self.channel_square_sum.sqrt()
        imbalance = float(
            channel_norm.max().item() / channel_norm.mean().item())
        return {
            "minimum": self.minimum,
            "maximum": max(abs(self.minimum), abs(self.maximum)),
            "p75": float(percentiles[0].item()),
            "p99": float(percentiles[1].item()),
            "p99_9": float(percentiles[2].item()),
            "p99_99": float(percentiles[3].item()),
            "kurtosis": kurtosis,
            "channel_imbalance": imbalance,
            "sqnr": sqnr,
            "zero_code_ratio": float((codes == 0).float().mean().item()),
            "saturation_ratio": float(
                (codes.abs() == quantizer.qmax).float().mean().item()),
        }


class CSPNRotationController:
    METHODS = ("identity", "random", "hadamard")

    def __init__(self, model: nn.Module,
                 boundaries: Sequence[RotationBoundary], seed: int) -> None:
        self.model = model
        self.boundaries = tuple(boundaries)
        self.seed = int(seed)
        self.modules = dict(model.named_modules())
        self.mode = "bypass"
        self.frozen = False
        self.base_weights = {}
        self.source_weights = {}
        self.active_methods = {}
        self.active_quantizers = {}
        self.quantize_enabled = False
        self.channels = {}
        self.rotations = {}
        self.observers = {}
        self.handles = []
        for boundary_index, boundary in enumerate(self.boundaries):
            consumer = boundary.consumers[0]
            conv = self.modules[consumer.module]
            channels = conv.in_channels \
                if consumer.channel_count is None else consumer.channel_count
            self.channels[boundary.name] = int(channels)
            self.rotations[boundary.name] = {
                "identity": torch.eye(
                    channels, dtype=torch.float32,
                    device=conv.weight.device),
                "random": random_orthogonal_matrix(
                    channels, self.seed + boundary_index).to(
                        conv.weight.device),
                "hadamard": hadamard_rotation_matrix(
                    channels, self.seed + boundary_index).to(
                        conv.weight.device),
            }
            self.observers[boundary.name] = dict(
                (method, RotationBoundaryObserver(channels))
                for method in self.METHODS)
            module = self.modules[boundary.module]
            self.handles.append(module.register_forward_pre_hook(
                self._make_hook(boundary)))
            for current in boundary.consumers:
                current_module = self.modules[current.module]
                self.source_weights[current.module] = \
                    current_module.weight.detach().clone()

    def _make_hook(self, boundary: RotationBoundary):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
            del module
            current = inputs[boundary.argument_index]
            if self.mode == "observe":
                for method in self.METHODS:
                    rotation = self.rotations[boundary.name][method]
                    self.observers[boundary.name][method].update(
                        rotate_channels(current, rotation))
                return None
            if self.mode != "quantize":
                return None
            method = self.active_methods[boundary.name]
            rotation = self.rotations[boundary.name][method]
            transformed = rotate_channels(current, rotation)
            if self.quantize_enabled:
                transformed = self.active_quantizers[boundary.name](transformed)
            updated = list(inputs)
            updated[boundary.argument_index] = transformed
            return tuple(updated)
        return hook

    def observe(self) -> None:
        self.mode = "observe"

    def freeze(self) -> None:
        for boundary in self.boundaries:
            for method in self.METHODS:
                observer = self.observers[boundary.name][method]
                observer.quantizer(bits=4, group_size=None)
        self.frozen = True
        self.mode = "bypass"

    def _restore_weights(self) -> None:
        for name in self.base_weights:
            module = self.modules[name]
            module.weight.data.copy_(self.base_weights[name].to(module.weight))
        self.base_weights = {}

    def _validate_methods(self, methods: Mapping[str, str]) -> None:
        expected = set(boundary.name for boundary in self.boundaries)
        if set(methods) != expected:
            raise ValueError("rotation configuration must name every boundary")
        for boundary in self.boundaries:
            method = methods[boundary.name]
            if method not in self.METHODS:
                raise ValueError("unknown rotation method: %s" % method)

    def weight_source_overrides(
            self, methods: Mapping[str, str]) -> Dict[str, torch.Tensor]:
        self._validate_methods(methods)
        output = dict(
            (name, weight.clone())
            for name, weight in self.source_weights.items())
        for boundary in self.boundaries:
            rotation = self.rotations[boundary.name][methods[boundary.name]]
            for consumer in boundary.consumers:
                output[consumer.module] = transform_input_weight(
                    output[consumer.module], rotation,
                    consumer.channel_start, consumer.channel_count)
        return output

    def configure(self, methods: Mapping[str, str], bits: int,
                  group_size: Optional[int], quantize: bool = True,
                  absorb_weights: bool = True) -> None:
        self._validate_methods(methods)
        if quantize and not self.frozen:
            raise RuntimeError("rotation calibration must be frozen")
        self._restore_weights()
        consumers = set(
            consumer.module for boundary in self.boundaries
            for consumer in boundary.consumers)
        self.base_weights = dict(
            (name, self.modules[name].weight.detach().clone())
            for name in consumers)
        self.active_methods = {}
        self.active_quantizers = {}
        for boundary in self.boundaries:
            method = methods[boundary.name]
            rotation = self.rotations[boundary.name][method]
            if absorb_weights:
                for consumer in boundary.consumers:
                    absorb_input_rotation(
                        self.modules[consumer.module], rotation,
                        consumer.channel_start, consumer.channel_count)
            self.active_methods[boundary.name] = method
            if quantize:
                self.active_quantizers[boundary.name] = \
                    self.observers[boundary.name][method].quantizer(
                        bits, group_size)
        self.quantize_enabled = bool(quantize)
        self.mode = "quantize"

    def statistics(self, methods: Mapping[str, str], bits: int,
                   group_size: Optional[int]) -> Sequence[Dict[str, float]]:
        rows = []
        for boundary in self.boundaries:
            method = methods[boundary.name]
            observer = self.observers[boundary.name][method]
            quantizer = observer.quantizer(bits, group_size)
            row = observer.statistics(quantizer)
            row.update({
                "boundary": boundary.name,
                "method": method,
                "bits": int(bits),
                "group_size": quantizer.group_size,
            })
            rows.append(row)
        return rows

    def disable(self) -> None:
        self._restore_weights()
        self.mode = "bypass"
        self.active_methods = {}
        self.active_quantizers = {}
        self.quantize_enabled = False

    def close(self) -> None:
        self.disable()
        for handle in self.handles:
            handle.remove()
        self.handles = []
