#!/usr/bin/env python3
"""Framework-independent RTN fake quantization for Conv2d and Linear modules."""

from __future__ import division

import math

import torch
import torch.nn as nn


def symmetric_weight_qdq(weight, bits):
    if bits < 2:
        raise ValueError("weight bits must be at least 2")
    if weight.ndim < 2:
        raise ValueError("weight must have an output-channel dimension")
    qmax = 2 ** (bits - 1) - 1
    flat = weight.reshape(weight.shape[0], -1)
    maximum = flat.abs().max(dim=1)[0]
    safe_maximum = torch.where(maximum > 0, maximum, torch.ones_like(maximum))
    shape = [weight.shape[0]] + [1] * (weight.ndim - 1)
    scale = (safe_maximum / float(qmax)).reshape(shape)
    codes = torch.round(weight / scale).clamp(-qmax, qmax)
    return codes * scale, scale


class MinMaxObserver(object):
    def __init__(self):
        self.minimum = float("inf")
        self.maximum = float("-inf")

    @property
    def observed(self):
        return math.isfinite(self.minimum) and math.isfinite(self.maximum)

    def update(self, tensor):
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            return
        detached = tensor.detach()
        self.minimum = min(self.minimum, float(detached.min().item()))
        self.maximum = max(self.maximum, float(detached.max().item()))

    def quantizer(self, bits):
        if not self.observed:
            raise RuntimeError("cannot freeze an observer without samples")
        return AffineTensorQuantizer(bits, self.minimum, self.maximum)


class QuantizationStats(object):
    def __init__(self):
        self.numel = 0
        self.saturated = 0
        self.signal_sq = 0.0
        self.error_sq = 0.0
        self.dot = 0.0
        self.reference_sq = 0.0
        self.quantized_sq = 0.0
        self.sign_flips = 0

    def update(self, reference, quantized, saturated=0):
        reference = reference.detach().double()
        quantized = quantized.detach().double()
        difference = reference - quantized
        self.numel += reference.numel()
        self.saturated += int(saturated)
        self.signal_sq += float(torch.sum(reference * reference).item())
        self.error_sq += float(torch.sum(difference * difference).item())
        self.dot += float(torch.sum(reference * quantized).item())
        self.reference_sq += float(torch.sum(reference * reference).item())
        self.quantized_sq += float(torch.sum(quantized * quantized).item())
        self.sign_flips += int(torch.sum(reference * quantized < 0).item())

    @property
    def mse(self):
        return self.error_sq / max(self.numel, 1)

    @property
    def sqnr_db(self):
        if self.error_sq == 0.0:
            return float("inf")
        if self.signal_sq == 0.0:
            return float("-inf")
        return 10.0 * math.log10(self.signal_sq / self.error_sq)

    @property
    def cosine(self):
        denominator = math.sqrt(self.reference_sq * self.quantized_sq)
        if denominator == 0.0:
            return 1.0 if self.reference_sq == self.quantized_sq else 0.0
        return self.dot / denominator

    @property
    def saturation_rate(self):
        return self.saturated / float(max(self.numel, 1))

    @property
    def sign_flip_rate(self):
        return self.sign_flips / float(max(self.numel, 1))


class AffineTensorQuantizer(object):
    def __init__(self, bits, minimum, maximum):
        if bits < 2:
            raise ValueError("activation bits must be at least 2")
        self.bits = int(bits)
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1
        self.minimum = min(float(minimum), 0.0)
        self.maximum = max(float(maximum), 0.0)
        extent = self.maximum - self.minimum
        self.scale = extent / float(self.qmax - self.qmin) if extent > 0 else 1.0
        zero_point = round(self.qmin - self.minimum / self.scale)
        self.zero_point = min(max(float(zero_point), self.qmin), self.qmax)

    @property
    def representable_min(self):
        return (self.qmin - self.zero_point) * self.scale

    @property
    def representable_max(self):
        return (self.qmax - self.zero_point) * self.scale

    def __call__(self, tensor, stats=None):
        scaled = tensor / self.scale + self.zero_point
        rounded = torch.round(scaled)
        saturated = int(torch.sum((rounded < self.qmin) | (rounded > self.qmax)).item())
        codes = rounded.clamp(self.qmin, self.qmax)
        quantized = (codes - self.zero_point) * self.scale
        if stats is not None:
            stats.update(tensor, quantized, saturated=saturated)
        return quantized


class RTNInstrumentor(object):
    """Attach calibration and QDQ hooks while preserving official module types."""

    def __init__(self, model, group_fn):
        self.model = model
        self.mode = "bypass"
        self.frozen = False
        self.w_bits = None
        self.a_bits = None
        self.enabled_groups = set()
        self.modules = {}
        self.groups = {}
        self.original_weights = {}
        self.observers = {}
        self.quantizers = {}
        self.stats = {}
        self.handles = []

        for name, module in model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            group = group_fn(name, module)
            if group is None:
                continue
            self.modules[name] = module
            self.groups[name] = str(group)
            self.original_weights[name] = module.weight.detach().cpu().clone()
            self.observers[(name, "input")] = MinMaxObserver()
            self.observers[(name, "output")] = MinMaxObserver()
            self.handles.append(module.register_forward_pre_hook(self._make_pre_hook(name)))
            self.handles.append(module.register_forward_hook(self._make_post_hook(name)))

    def _make_pre_hook(self, name):
        def hook(module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                return None
            tensor = inputs[0]
            if self.mode == "observe":
                self.observers[(name, "input")].update(tensor)
                return None
            if self.mode != "quantize" or self.groups[name] not in self.enabled_groups:
                return None
            quantizer = self.quantizers.get((name, "input"))
            if quantizer is None:
                return None
            quantized = quantizer(tensor, self.stats[(name, "input")])
            return (quantized,) + tuple(inputs[1:])
        return hook

    def _make_post_hook(self, name):
        def hook(module, inputs, output):
            if not torch.is_tensor(output):
                return None
            if self.mode == "observe":
                self.observers[(name, "output")].update(output)
                return None
            if self.mode != "quantize" or self.groups[name] not in self.enabled_groups:
                return None
            quantizer = self.quantizers.get((name, "output"))
            if quantizer is None:
                return None
            return quantizer(output, self.stats[(name, "output")])
        return hook

    def _restore_weights(self):
        with torch.no_grad():
            for name, module in self.modules.items():
                original = self.original_weights[name]
                module.weight.copy_(original.to(device=module.weight.device,
                                                dtype=module.weight.dtype))

    def observe(self):
        self._restore_weights()
        self.mode = "observe"
        self.frozen = False
        self.quantizers = {}
        self.stats = {}

    def freeze(self):
        observed = [key for key, observer in self.observers.items() if observer.observed]
        if not observed:
            raise RuntimeError("no activation tensors were observed")
        self.frozen = True
        self.mode = "bypass"

    def configure(self, w_bits, a_bits, enabled_groups):
        if not self.frozen:
            raise RuntimeError("calibration must be frozen before quantization")
        self._restore_weights()
        self.w_bits = int(w_bits)
        self.a_bits = int(a_bits)
        self.enabled_groups = set(enabled_groups)
        unknown = self.enabled_groups - set(self.groups.values())
        if unknown:
            raise ValueError("unknown module groups: %s" % sorted(unknown))
        self.quantizers = {}
        self.stats = {}

        with torch.no_grad():
            for name, module in self.modules.items():
                if self.groups[name] not in self.enabled_groups:
                    continue
                original = self.original_weights[name]
                quantized, _ = symmetric_weight_qdq(original, self.w_bits)
                weight_stats = QuantizationStats()
                weight_stats.update(original, quantized)
                self.stats[(name, "weight")] = weight_stats
                module.weight.copy_(quantized.to(device=module.weight.device,
                                                 dtype=module.weight.dtype))
                for kind in ("input", "output"):
                    observer = self.observers[(name, kind)]
                    if observer.observed:
                        self.quantizers[(name, kind)] = observer.quantizer(self.a_bits)
                        self.stats[(name, kind)] = QuantizationStats()
        self.mode = "quantize"

    def disable(self):
        self._restore_weights()
        self.mode = "bypass"
        self.enabled_groups = set()

    def module_groups(self):
        return dict(self.groups)

    def statistics(self):
        rows = []
        for (name, kind), stats in sorted(self.stats.items()):
            rows.append({
                "module": name,
                "group": self.groups[name],
                "kind": kind,
                "numel": stats.numel,
                "mse": stats.mse,
                "sqnr_db": stats.sqnr_db,
                "cosine": stats.cosine,
                "saturation_rate": stats.saturation_rate,
                "sign_flip_rate": stats.sign_flip_rate,
            })
        return rows

    def close(self):
        self.disable()
        for handle in self.handles:
            handle.remove()
        self.handles = []
