"""Quantization contracts for the official CSPN RGBD stem."""

from __future__ import annotations

from dataclasses import dataclass
import math
import types
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from spn_quant.integer_ops import (
    int8_mm_int32,
    integer_im2col,
    requantize_int32,
)


STEM_CONFIGS = (
    "STRICT_W4A4",
    "STEM_W8A8",
    "STEM_FP16",
    "STEM_BRANCH_A4",
)


def split_rgb_depth(tensor: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(tensor) or tensor.ndim != 4:
        raise ValueError("CSPN stem input must have NCHW shape")
    if int(tensor.shape[1]) != 4:
        raise ValueError("CSPN stem input must contain four channels")
    return tensor[:, :3], tensor[:, 3:4]


@dataclass(frozen=True)
class CSPNStemIntegerResult:
    output: torch.Tensor
    rgb_codes: torch.Tensor
    depth_codes: torch.Tensor
    weight_codes: torch.Tensor
    rgb_accumulator: torch.Tensor
    depth_accumulator: torch.Tensor
    rgb_requantized: torch.Tensor
    depth_requantized: torch.Tensor
    accumulator: torch.Tensor
    rgb_scale: float
    depth_scale: float
    weight_scales: torch.Tensor
    target_accumulator_scales: torch.Tensor


class _SignalStatistics(object):
    def __init__(self, signal: str) -> None:
        self.signal = str(signal)
        self.updates = 0
        self.elements = 0
        self.signal_energy = 0.0
        self.error_energy = 0.0
        self.reference_zeros = 0
        self.quantized_zeros = 0
        self.new_zeros = 0
        self.nonzero = 0
        self.saturated = 0
        self.clipping_energy = 0.0

    def update(self, reference: torch.Tensor, quantized: torch.Tensor,
               codes: torch.Tensor = None, qmin: int = None,
               qmax: int = None, minimum: float = None,
               maximum: float = None) -> None:
        if reference.shape != quantized.shape:
            raise ValueError("stem statistic tensors must have equal shapes")
        if not bool(torch.isfinite(reference).all().item()) or \
                not bool(torch.isfinite(quantized).all().item()):
            raise ValueError("stem statistic tensors must be finite")
        reference64 = reference.detach().to(torch.float64)
        quantized64 = quantized.detach().to(torch.float64)
        difference = reference64 - quantized64
        self.updates += 1
        self.elements += int(reference.numel())
        self.signal_energy += float((reference64 ** 2).sum().item())
        self.error_energy += float((difference ** 2).sum().item())
        reference_zero = reference64 == 0.0
        quantized_zero = quantized64 == 0.0
        nonzero = ~reference_zero
        self.reference_zeros += int(reference_zero.sum().item())
        self.quantized_zeros += int(quantized_zero.sum().item())
        self.new_zeros += int((nonzero & quantized_zero).sum().item())
        self.nonzero += int(nonzero.sum().item())
        if codes is not None:
            if qmin is None or qmax is None:
                raise ValueError("stem code limits are required")
            self.saturated += int(
                ((codes == int(qmin)) | (codes == int(qmax))).sum().item())
        if minimum is not None and maximum is not None:
            clipped = reference64.clamp(float(minimum), float(maximum))
            self.clipping_energy += float(
                ((reference64 - clipped) ** 2).sum().item())

    def row(self, config: str) -> Dict[str, object]:
        if self.updates == 0 or self.elements == 0:
            raise RuntimeError("stem statistics have no observations")
        mse = self.error_energy / float(self.elements)
        if self.error_energy == 0.0:
            sqnr = 300.0
        elif self.signal_energy == 0.0:
            sqnr = -300.0
        else:
            sqnr = 10.0 * math.log10(
                self.signal_energy / self.error_energy)
        return {
            "config": str(config),
            "signal": self.signal,
            "updates": self.updates,
            "elements": self.elements,
            "mse": mse,
            "sqnr_db": sqnr,
            "reference_zero_ratio": self.reference_zeros /
            float(self.elements),
            "quantized_zero_ratio": self.quantized_zeros /
            float(self.elements),
            "new_zero_rate": 0.0 if self.nonzero == 0 else
            self.new_zeros / float(self.nonzero),
            "saturation_rate": self.saturated / float(self.elements),
            "clipping_error_share": 0.0 if self.error_energy == 0.0 else
            self.clipping_energy / self.error_energy,
        }


class CSPNStemController(object):
    def __init__(self, module: nn.Conv2d) -> None:
        if not isinstance(module, nn.Conv2d):
            raise TypeError("CSPN stem controller requires Conv2d")
        if int(module.in_channels) != 4:
            raise ValueError("CSPN stem Conv must have four input channels")
        if module.bias is not None:
            raise ValueError("CSPN stem Conv must be bias-free")
        if int(module.groups) != 1:
            raise ValueError("CSPN stem Conv requires groups=1")
        if tuple(module.dilation) != (1, 1):
            raise ValueError("CSPN stem Conv requires dilation=1")
        self.module = module
        self.original_forward = module.forward
        self.original_weight = module.weight.detach().to(
            device="cpu", dtype=torch.float32).contiguous().clone()
        self.phase = "bypass"
        self.config = None
        self.observations = 0
        self.rgb_maximum = 0.0
        self.depth_maximum = 0.0
        self.merged_maximum = 0.0
        self.weight_bits = None
        self.weight_codes = None
        self.weight_scales = None
        self.input_scale = None
        self.rgb_scale = None
        self.depth_scale = None
        self._last_integer_result = None
        self._statistics = {}  # type: Dict[str, _SignalStatistics]

        def forward(current: nn.Conv2d, tensor: torch.Tensor) -> torch.Tensor:
            del current
            return self._forward(tensor)

        module.forward = types.MethodType(forward, module)

    def _validate_input(self, tensor: torch.Tensor) -> None:
        split_rgb_depth(tensor)
        if not tensor.is_floating_point():
            raise TypeError("CSPN stem input must be floating point")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("CSPN stem input must be finite")

    def _float_convolution(self, tensor: torch.Tensor,
                           weight: torch.Tensor = None) -> torch.Tensor:
        current_weight = self.original_weight.to(
            device=tensor.device, dtype=tensor.dtype) \
            if weight is None else weight
        return F.conv2d(
            tensor, current_weight, None,
            self.module.stride, self.module.padding,
            self.module.dilation, self.module.groups)

    def _observe_input(self, tensor: torch.Tensor) -> None:
        rgb, depth = split_rgb_depth(tensor)
        self.rgb_maximum = max(
            self.rgb_maximum, float(rgb.detach().amax().item()))
        self.depth_maximum = max(
            self.depth_maximum, float(depth.detach().amax().item()))
        self.merged_maximum = max(
            self.merged_maximum, float(tensor.detach().amax().item()))
        self.observations += 1

    @staticmethod
    def _unsigned_qdq(tensor: torch.Tensor, maximum: float, bits: int
                      ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        qmax = (1 << int(bits)) - 1
        scale = float(maximum) / float(qmax) \
            if float(maximum) > 0.0 else 1.0
        codes = torch.round(tensor / scale).clamp(
            0, qmax).to(torch.uint8)
        return codes.to(tensor.dtype) * scale, codes, scale

    def _quantize_weight(self, bits: int) -> None:
        qmax = (1 << (int(bits) - 1)) - 1
        flat = self.original_weight.reshape(
            self.original_weight.shape[0], -1)
        maximum = flat.abs().amax(dim=1)
        scales = torch.where(
            maximum > 0.0, maximum / float(qmax),
            torch.ones_like(maximum))
        codes = torch.round(
            self.original_weight /
            scales.reshape(-1, 1, 1, 1)).clamp(
                -qmax, qmax).to(torch.int8)
        self.weight_bits = int(bits)
        self.weight_codes = codes
        self.weight_scales = scales.reshape(-1, 1, 1, 1)

    def _quantized_weight(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.weight_codes.to(
            device=tensor.device, dtype=tensor.dtype) * \
            self.weight_scales.to(device=tensor.device, dtype=tensor.dtype)

    def _partial_convolution(self, codes: torch.Tensor,
                             weight_codes: torch.Tensor) -> torch.Tensor:
        patches, output_shape = integer_im2col(
            codes.to(torch.int8), self.module.kernel_size,
            self.module.stride, self.module.padding, self.module.dilation)
        products = int8_mm_int32(
            patches, weight_codes.reshape(
                weight_codes.shape[0], -1).transpose(0, 1).contiguous())
        return products.reshape(
            codes.shape[0], output_shape[0], output_shape[1],
            weight_codes.shape[0]).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _checked_int32(values: torch.Tensor, name: str) -> torch.Tensor:
        limits = torch.iinfo(torch.int32)
        if values.numel() and (
                int(values.min().item()) < limits.min or
                int(values.max().item()) > limits.max):
            raise OverflowError("%s exceeds INT32" % name)
        return values.to(torch.int32)

    def _reference_partials(self, tensor: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb, depth = split_rgb_depth(tensor)
        weight = self.original_weight.to(
            device=tensor.device, dtype=tensor.dtype)
        rgb_output = F.conv2d(
            rgb, weight[:, :3], None, self.module.stride,
            self.module.padding, self.module.dilation, 1)
        depth_output = F.conv2d(
            depth, weight[:, 3:], None, self.module.stride,
            self.module.padding, self.module.dilation, 1)
        return rgb_output, depth_output

    def _update_statistics(
            self, tensor: torch.Tensor, quantized_rgb: torch.Tensor,
            quantized_depth: torch.Tensor, rgb_codes: torch.Tensor,
            depth_codes: torch.Tensor, rgb_partial: torch.Tensor,
            depth_partial: torch.Tensor, output: torch.Tensor,
            rgb_bits: int, depth_bits: int) -> None:
        rgb, depth = split_rgb_depth(tensor)
        reference_rgb_partial, reference_depth_partial = \
            self._reference_partials(tensor)
        reference_output = reference_rgb_partial + reference_depth_partial
        self._statistics["rgb_input"].update(
            rgb, quantized_rgb, rgb_codes, 0, (1 << rgb_bits) - 1,
            0.0, self.rgb_maximum)
        self._statistics["depth_input"].update(
            depth, quantized_depth, depth_codes, 0,
            (1 << depth_bits) - 1, 0.0, self.depth_maximum)
        self._statistics["rgb_partial"].update(
            reference_rgb_partial, rgb_partial)
        self._statistics["depth_partial"].update(
            reference_depth_partial, depth_partial)
        self._statistics["stem_output"].update(reference_output, output)

    def _merged_forward(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        quantized, codes, scale = self._unsigned_qdq(
            tensor, self.merged_maximum, bits)
        weight = self._quantized_weight(tensor)
        output = self._float_convolution(quantized, weight)
        rgb, depth = split_rgb_depth(quantized)
        rgb_codes, depth_codes = split_rgb_depth(codes)
        rgb_partial = F.conv2d(
            rgb, weight[:, :3], None, self.module.stride,
            self.module.padding, self.module.dilation, 1)
        depth_partial = F.conv2d(
            depth, weight[:, 3:], None, self.module.stride,
            self.module.padding, self.module.dilation, 1)
        self.input_scale = scale
        self._update_statistics(
            tensor, rgb, depth, rgb_codes, depth_codes,
            rgb_partial, depth_partial, output, bits, bits)
        return output

    def _fp16_forward(self, tensor: torch.Tensor) -> torch.Tensor:
        weight = self.original_weight.to(
            device=tensor.device, dtype=torch.float16)
        output = F.conv2d(
            tensor.to(torch.float16), weight, None,
            self.module.stride, self.module.padding,
            self.module.dilation, self.module.groups).to(torch.float32)
        rgb, depth = split_rgb_depth(tensor)
        rgb_partial = F.conv2d(
            rgb.half(), weight[:, :3], None, self.module.stride,
            self.module.padding, self.module.dilation, 1).float()
        depth_partial = F.conv2d(
            depth.half(), weight[:, 3:], None, self.module.stride,
            self.module.padding, self.module.dilation, 1).float()
        rgb_codes = torch.ones_like(rgb, dtype=torch.uint8)
        depth_codes = torch.ones_like(depth, dtype=torch.uint8)
        self._update_statistics(
            tensor, rgb, depth, rgb_codes, depth_codes,
            rgb_partial, depth_partial, output, 8, 8)
        return output

    def _branch_forward(self, tensor: torch.Tensor) -> torch.Tensor:
        rgb, depth = split_rgb_depth(tensor)
        quantized_rgb, rgb_codes, rgb_scale = self._unsigned_qdq(
            rgb, self.rgb_maximum, 4)
        quantized_depth, depth_codes, depth_scale = self._unsigned_qdq(
            depth, self.depth_maximum, 4)
        weight_codes = self.weight_codes.to(tensor.device)
        rgb_accumulator = self._partial_convolution(
            rgb_codes, weight_codes[:, :3])
        depth_accumulator = self._partial_convolution(
            depth_codes, weight_codes[:, 3:])
        weight_scales = self.weight_scales.to(
            device=tensor.device, dtype=torch.float64).reshape(1, -1, 1, 1)
        rgb_accumulator_scales = weight_scales * rgb_scale
        depth_accumulator_scales = weight_scales * depth_scale
        target_scales = torch.maximum(
            rgb_accumulator_scales, depth_accumulator_scales)
        limits = torch.iinfo(torch.int32)
        rgb_requantized = requantize_int32(
            rgb_accumulator, rgb_accumulator_scales,
            target_scales, limits.min, limits.max)
        depth_requantized = requantize_int32(
            depth_accumulator, depth_accumulator_scales,
            target_scales, limits.min, limits.max)
        accumulator = self._checked_int32(
            rgb_requantized.to(torch.int64) +
            depth_requantized.to(torch.int64), "CSPN stem accumulator")
        output = accumulator.to(torch.float32) * \
            target_scales.to(torch.float32)
        rgb_partial = rgb_requantized.to(torch.float32) * \
            target_scales.to(torch.float32)
        depth_partial = depth_requantized.to(torch.float32) * \
            target_scales.to(torch.float32)
        self.rgb_scale = rgb_scale
        self.depth_scale = depth_scale
        self._last_integer_result = CSPNStemIntegerResult(
            output=output,
            rgb_codes=rgb_codes,
            depth_codes=depth_codes,
            weight_codes=weight_codes,
            rgb_accumulator=rgb_accumulator,
            depth_accumulator=depth_accumulator,
            rgb_requantized=rgb_requantized,
            depth_requantized=depth_requantized,
            accumulator=accumulator,
            rgb_scale=rgb_scale,
            depth_scale=depth_scale,
            weight_scales=weight_scales,
            target_accumulator_scales=target_scales,
        )
        self._update_statistics(
            tensor, quantized_rgb, quantized_depth,
            rgb_codes, depth_codes, rgb_partial, depth_partial,
            output, 4, 4)
        return output

    def _forward(self, tensor: torch.Tensor) -> torch.Tensor:
        self._validate_input(tensor)
        if self.phase == "bypass":
            return self.original_forward(tensor)
        if self.phase == "observe":
            self._observe_input(tensor)
            return self.original_forward(tensor)
        if self.phase != "quantize" or self.config is None:
            raise RuntimeError("CSPN stem controller phase is invalid")
        if self.config == "STRICT_W4A4":
            return self._merged_forward(tensor, 4)
        if self.config == "STEM_W8A8":
            return self._merged_forward(tensor, 8)
        if self.config == "STEM_FP16":
            return self._fp16_forward(tensor)
        if self.config == "STEM_BRANCH_A4":
            return self._branch_forward(tensor)
        raise RuntimeError("CSPN stem configuration is invalid")

    def observe(self) -> None:
        self.phase = "observe"
        self.config = None
        self.observations = 0
        self.rgb_maximum = 0.0
        self.depth_maximum = 0.0
        self.merged_maximum = 0.0

    def freeze(self) -> None:
        if self.phase != "observe" or self.observations == 0:
            raise RuntimeError("CSPN stem freeze requires observation")
        self.phase = "bypass"

    def reset_statistics(self) -> None:
        self._statistics = dict(
            (name, _SignalStatistics(name)) for name in (
                "rgb_input", "depth_input", "rgb_partial",
                "depth_partial", "stem_output"))
        self._last_integer_result = None

    def configure(self, name: str) -> None:
        if self.observations == 0 or self.phase == "observe":
            raise RuntimeError("CSPN stem must be frozen before configuration")
        if name not in STEM_CONFIGS:
            raise ValueError("unknown CSPN stem configuration: %s" % name)
        self.config = str(name)
        self.input_scale = None
        self.rgb_scale = None
        self.depth_scale = None
        if name == "STEM_FP16":
            self.weight_bits = 16
            self.weight_codes = None
            self.weight_scales = None
        else:
            self._quantize_weight(8 if name == "STEM_W8A8" else 4)
        self.reset_statistics()
        self.phase = "quantize"

    def contract(self) -> Dict[str, object]:
        if self.phase != "quantize" or self.config is None:
            raise RuntimeError("CSPN stem configuration is not active")
        row = {
            "config": self.config,
            "weight_bits": self.weight_bits,
            "rgb_maximum": self.rgb_maximum,
            "depth_maximum": self.depth_maximum,
            "merged_maximum": self.merged_maximum,
        }
        if self.config == "STEM_FP16":
            row.update({
                "activation_bits": 16,
                "activation_scales": 0,
            })
        elif self.config == "STEM_BRANCH_A4":
            row.update({
                "activation_bits": 4,
                "activation_scales": 2,
                "rgb_scale": self.rgb_maximum / 15.0
                if self.rgb_maximum > 0.0 else 1.0,
                "depth_scale": self.depth_maximum / 15.0
                if self.depth_maximum > 0.0 else 1.0,
            })
        else:
            bits = 8 if self.config == "STEM_W8A8" else 4
            row.update({
                "activation_bits": bits,
                "activation_scales": 1,
                "input_scale": self.merged_maximum /
                float((1 << bits) - 1)
                if self.merged_maximum > 0.0 else 1.0,
            })
        return row

    def last_integer_result(self) -> CSPNStemIntegerResult:
        if self._last_integer_result is None:
            raise RuntimeError("CSPN stem has no integer result")
        return self._last_integer_result

    def statistics(self):
        if not self._statistics or any(
                value.updates == 0 for value in self._statistics.values()):
            raise RuntimeError("CSPN stem statistics have no observations")
        return [self._statistics[name].row(self.config) for name in (
            "rgb_input", "depth_input", "rgb_partial",
            "depth_partial", "stem_output")]

    def disable(self) -> None:
        self.phase = "bypass"
        self.config = None
        self.reset_statistics()

    def close(self) -> None:
        self.disable()
        self.module.forward = self.original_forward
