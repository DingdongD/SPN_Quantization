"""Split-branch integer convolution for CompletionFormer fusion blocks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from spn_quant.integer_ops import (
    int8_mm_int32,
    integer_im2col,
    requantize_int32,
)
from spn_quant.scale_search import CalibrationCache, CoordinateScaleSearch


def split_concat_branches(tensor: torch.Tensor,
                          channels: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(tensor) or tensor.ndim != 4:
        raise ValueError("concat tensor must have NCHW shape")
    channels = int(channels)
    if channels <= 0 or tensor.shape[1] != 2 * channels:
        raise ValueError("concat tensor must contain two equal channel branches")
    return tensor[:, :channels], tensor[:, channels:]


@dataclass(frozen=True)
class SplitConcatIntegerResult:
    output: torch.Tensor
    output_codes: torch.Tensor
    transformer_codes: torch.Tensor
    cnn_codes: torch.Tensor
    weight_codes: torch.Tensor
    transformer_accumulator: torch.Tensor
    cnn_accumulator: torch.Tensor
    transformer_requantized: torch.Tensor
    cnn_requantized: torch.Tensor
    accumulator: torch.Tensor
    bias_codes: torch.Tensor
    transformer_scale: float
    cnn_scale: float
    transformer_accumulator_scales: torch.Tensor
    cnn_accumulator_scales: torch.Tensor
    target_accumulator_scales: torch.Tensor
    output_scale: float


class _ErrorAccumulator(object):
    def __init__(self) -> None:
        self.signal_sq = 0.0
        self.error_sq = 0.0
        self.elements = 0

    def update(self, reference: torch.Tensor, candidate: torch.Tensor) -> None:
        reference64 = reference.detach().double()
        difference = reference64 - candidate.detach().double()
        self.signal_sq += float((reference64 ** 2).sum().item())
        self.error_sq += float((difference ** 2).sum().item())
        self.elements += int(reference.numel())

    @property
    def sqnr_db(self) -> float:
        if self.error_sq == 0.0:
            return float("inf")
        if self.signal_sq == 0.0:
            return float("-inf")
        return 10.0 * math.log10(self.signal_sq / self.error_sq)

    @property
    def mse(self) -> float:
        if self.elements == 0:
            raise RuntimeError("concat statistics have no observations")
        return self.error_sq / float(self.elements)


class SplitConcatConvController(object):
    def __init__(self, name: str, module: nn.Conv2d, branch_channels: int,
                 weight_bits: int, activation_bits: int, output_bits: int,
                 clip_factors: Sequence[float], search_rounds: int,
                 cache_sample_limit: int, cache_byte_limit: int) -> None:
        if not isinstance(module, nn.Conv2d):
            raise TypeError("split concat controller requires Conv2d")
        self.name = str(name)
        self.module = module
        self.branch_channels = int(branch_channels)
        self.weight_bits = int(weight_bits)
        self.activation_bits = int(activation_bits)
        self.output_bits = int(output_bits)
        self.clip_factors = tuple(float(value) for value in clip_factors)
        self.search_rounds = int(search_rounds)
        if not self.name:
            raise ValueError("concat controller name must be nonempty")
        if self.branch_channels <= 0 or \
                module.in_channels != 2 * self.branch_channels:
            raise ValueError("concat Conv input channels do not match branches")
        if module.groups != 1:
            raise ValueError("split concat Conv requires groups=1")
        if tuple(module.dilation) != (1, 1):
            raise ValueError("split concat Conv requires dilation=1")
        for bits in (self.weight_bits, self.activation_bits, self.output_bits):
            if bits < 2 or bits > 8:
                raise ValueError("concat integer bits must be in [2, 8]")

        self.original_weight = module.weight.detach().to(
            device="cpu", dtype=torch.float32).contiguous().clone()
        self.original_bias = None if module.bias is None else \
            module.bias.detach().to(
                device="cpu", dtype=torch.float32).contiguous().clone()
        self.cache = CalibrationCache(
            sample_limit=cache_sample_limit, byte_limit=cache_byte_limit)
        self.search = CoordinateScaleSearch(
            parameter_names=("transformer", "cnn", "accumulator", "output"),
            factors=self.clip_factors, rounds=self.search_rounds)
        self.phase = "observe"
        self.device = None
        self.observations = 0
        self.cached_samples = 0
        self.transformer_maximum = 0.0
        self.cnn_maximum = 0.0
        self.output_maximum = 0.0
        self.weight_codes = None
        self.weight_scales = None
        self.selected_factors = {}  # type: Dict[str, float]
        self._search_rows = []  # type: List[Dict[str, object]]
        self._transformer_stats = _ErrorAccumulator()
        self._cnn_stats = _ErrorAccumulator()
        self._partial_stats = _ErrorAccumulator()
        self._output_stats = _ErrorAccumulator()
        self._transformer_zeros = 0
        self._cnn_zeros = 0
        self._transformer_saturated = 0
        self._cnn_saturated = 0
        self._transformer_elements = 0
        self._cnn_elements = 0
        self._quantized_updates = 0

    def _validate(self, merged: torch.Tensor,
                  output: torch.Tensor = None) -> None:
        transformer, cnn = split_concat_branches(
            merged, self.branch_channels)
        del transformer, cnn
        if not bool(torch.isfinite(merged).all().item()):
            raise ValueError("concat input must be finite")
        if output is not None:
            expected_height = (
                merged.shape[2] + 2 * self.module.padding[0] -
                self.module.kernel_size[0]) // self.module.stride[0] + 1
            expected_width = (
                merged.shape[3] + 2 * self.module.padding[1] -
                self.module.kernel_size[1]) // self.module.stride[1] + 1
            expected = (
                merged.shape[0], self.module.out_channels,
                expected_height, expected_width)
            if tuple(output.shape) != expected:
                raise ValueError("concat Conv output shape is invalid")
            if not bool(torch.isfinite(output).all().item()):
                raise ValueError("concat Conv output must be finite")

    def observe(self, merged: torch.Tensor, fp_output: torch.Tensor) -> None:
        if self.phase != "observe":
            raise RuntimeError("concat observation phase is closed")
        self._validate(merged, fp_output)
        if self.device is None:
            self.device = merged.device
        elif self.device != merged.device:
            raise ValueError("concat calibration device changed")
        transformer, cnn = split_concat_branches(
            merged, self.branch_channels)
        self.transformer_maximum = max(
            self.transformer_maximum,
            float(transformer.detach().abs().max().item()))
        self.cnn_maximum = max(
            self.cnn_maximum, float(cnn.detach().abs().max().item()))
        self.output_maximum = max(
            self.output_maximum, float(fp_output.detach().abs().max().item()))
        if self.cached_samples < self.cache.sample_limit:
            self.cache.append((merged, fp_output))
            self.cached_samples += 1
        self.observations += 1

    def _freeze_weight(self) -> None:
        qmax = (1 << (self.weight_bits - 1)) - 1
        flat = self.original_weight.reshape(self.original_weight.shape[0], -1)
        maximum = flat.abs().amax(dim=1)
        scales = torch.where(
            maximum > 0.0, maximum / float(qmax), torch.ones_like(maximum))
        codes = torch.round(
            self.original_weight.to(torch.float64) /
            scales.to(torch.float64).reshape(-1, 1, 1, 1)).clamp(
                -qmax, qmax).to(torch.int8)
        self.weight_codes = codes
        self.weight_scales = scales.to(torch.float64)

    @staticmethod
    def _activation_codes(tensor: torch.Tensor, maximum: float,
                          bits: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        qmax = (1 << (int(bits) - 1)) - 1
        scale = float(maximum) / float(qmax) if float(maximum) > 0.0 else 1.0
        codes = torch.round(tensor.to(torch.float64) / scale).clamp(
            -qmax, qmax).to(torch.int8)
        reconstructed = codes.to(tensor.dtype) * scale
        return codes, reconstructed, scale

    @staticmethod
    def _checked_int32(values: torch.Tensor, name: str) -> torch.Tensor:
        limits = torch.iinfo(torch.int32)
        if values.numel() and (int(values.min().item()) < limits.min or
                               int(values.max().item()) > limits.max):
            raise OverflowError("%s exceeds INT32" % name)
        return values.to(torch.int32)

    def _partial_convolution(self, codes: torch.Tensor,
                             weight_codes: torch.Tensor) -> torch.Tensor:
        patches, output_shape = integer_im2col(
            codes, kernel_size=self.module.kernel_size,
            stride=self.module.stride, padding=self.module.padding,
            dilation=self.module.dilation)
        products = int8_mm_int32(
            patches, weight_codes.reshape(
                weight_codes.shape[0], -1).transpose(0, 1).contiguous())
        return products.reshape(
            codes.shape[0], output_shape[0], output_shape[1],
            weight_codes.shape[0]).permute(0, 3, 1, 2).contiguous()

    def _execute(self, merged: torch.Tensor,
                 factors: Dict[str, float]) -> SplitConcatIntegerResult:
        self._validate(merged)
        transformer, cnn = split_concat_branches(
            merged, self.branch_channels)
        transformer_codes, _, transformer_scale = self._activation_codes(
            transformer,
            self.transformer_maximum * factors["transformer"],
            self.activation_bits)
        cnn_codes, _, cnn_scale = self._activation_codes(
            cnn, self.cnn_maximum * factors["cnn"], self.activation_bits)
        weight_codes = self.weight_codes.to(merged.device)
        transformer_weight = weight_codes[:, :self.branch_channels]
        cnn_weight = weight_codes[:, self.branch_channels:]
        transformer_accumulator = self._partial_convolution(
            transformer_codes, transformer_weight)
        cnn_accumulator = self._partial_convolution(cnn_codes, cnn_weight)

        weight_scales = self.weight_scales.to(merged.device).reshape(
            1, -1, 1, 1)
        transformer_accumulator_scales = weight_scales * transformer_scale
        cnn_accumulator_scales = weight_scales * cnn_scale
        target_accumulator_scales = weight_scales * max(
            transformer_scale, cnn_scale) * factors["accumulator"]
        limits = torch.iinfo(torch.int32)
        transformer_requantized = requantize_int32(
            transformer_accumulator, transformer_accumulator_scales,
            target_accumulator_scales, limits.min, limits.max)
        cnn_requantized = requantize_int32(
            cnn_accumulator, cnn_accumulator_scales,
            target_accumulator_scales, limits.min, limits.max)
        combined = transformer_requantized.to(torch.int64) + \
            cnn_requantized.to(torch.int64)

        if self.original_bias is None:
            bias_codes = torch.zeros(
                self.module.out_channels, device=merged.device,
                dtype=torch.int32)
        else:
            bias = self.original_bias.to(merged.device).reshape(1, -1, 1, 1)
            bias_codes64 = torch.round(
                bias.to(torch.float64) /
                target_accumulator_scales.to(torch.float64)).to(torch.int64)
            bias_codes = self._checked_int32(
                bias_codes64, "concat bias").reshape(-1)
            combined = combined + bias_codes.reshape(1, -1, 1, 1)
        accumulator = self._checked_int32(combined, "concat accumulator")

        output_qmax = (1 << (self.output_bits - 1)) - 1
        output_maximum = self.output_maximum * factors["output"]
        output_scale = output_maximum / float(output_qmax) \
            if output_maximum > 0.0 else 1.0
        output_codes = requantize_int32(
            accumulator, target_accumulator_scales,
            torch.tensor(output_scale, device=merged.device),
            -output_qmax, output_qmax)
        output = output_codes.to(torch.float32) * output_scale
        return SplitConcatIntegerResult(
            output=output,
            output_codes=output_codes,
            transformer_codes=transformer_codes,
            cnn_codes=cnn_codes,
            weight_codes=weight_codes,
            transformer_accumulator=transformer_accumulator,
            cnn_accumulator=cnn_accumulator,
            transformer_requantized=transformer_requantized,
            cnn_requantized=cnn_requantized,
            accumulator=accumulator,
            bias_codes=bias_codes,
            transformer_scale=transformer_scale,
            cnn_scale=cnn_scale,
            transformer_accumulator_scales=
                transformer_accumulator_scales,
            cnn_accumulator_scales=cnn_accumulator_scales,
            target_accumulator_scales=target_accumulator_scales,
            output_scale=output_scale)

    def _objective(self, factors: Dict[str, float]) -> float:
        error = 0.0
        signal = 0.0
        for merged_cpu, target_cpu in self.cache.samples():
            merged = merged_cpu.to(self.device)
            target = target_cpu.to(self.device)
            candidate = self._execute(merged, factors).output
            error += float(((candidate.double() - target.double()) ** 2).sum().item())
            signal += float((target.double() ** 2).sum().item())
        return error / max(signal, torch.finfo(torch.float64).tiny)

    def _select_scales(self) -> None:
        result = self.search.run(
            {"transformer": 1.0, "cnn": 1.0,
             "accumulator": 1.0, "output": 1.0},
            self._objective, sample_count=self.cached_samples)
        self.selected_factors = dict(result.values)
        self._search_rows = []
        for source in result.rows:
            row = dict(source)
            row["module"] = self.name
            row["activation_bits"] = self.activation_bits
            row["output_bits"] = self.output_bits
            self._search_rows.append(row)
        self.phase = "frozen"

    def freeze(self) -> None:
        if self.phase != "observe":
            raise RuntimeError("concat controller is not observing")
        if self.observations == 0 or self.cached_samples == 0:
            raise RuntimeError("concat controller has no observations")
        self._freeze_weight()
        self._select_scales()

    def reconfigure(self, activation_bits: int, output_bits: int) -> None:
        if self.phase == "observe":
            raise RuntimeError("concat controller must be frozen first")
        activation_bits = int(activation_bits)
        output_bits = int(output_bits)
        for bits in (activation_bits, output_bits):
            if bits < 2 or bits > 8:
                raise ValueError("concat integer bits must be in [2, 8]")
        self.activation_bits = activation_bits
        self.output_bits = output_bits
        self._select_scales()

    def enable(self) -> None:
        if self.phase not in ("frozen", "disabled"):
            raise RuntimeError("concat controller must be frozen before enabling")
        self.phase = "quantize"

    def disable(self) -> None:
        if self.phase == "observe":
            raise RuntimeError("concat controller must be frozen before disabling")
        self.phase = "disabled"

    def execute(self, merged: torch.Tensor) -> SplitConcatIntegerResult:
        if self.phase != "quantize":
            raise RuntimeError("concat quantization is not enabled")
        return self._execute(merged, self.selected_factors)

    def _reference_output(self, merged: torch.Tensor) -> torch.Tensor:
        weight = self.original_weight.to(merged.device, dtype=merged.dtype)
        bias = None if self.original_bias is None else \
            self.original_bias.to(merged.device, dtype=merged.dtype)
        return F.conv2d(
            merged, weight, bias, self.module.stride, self.module.padding,
            self.module.dilation, self.module.groups)

    def quantize(self, merged: torch.Tensor) -> torch.Tensor:
        result = self.execute(merged)
        transformer, cnn = split_concat_branches(
            merged, self.branch_channels)
        transformer_reconstructed = result.transformer_codes.to(
            transformer.dtype) * result.transformer_scale
        cnn_reconstructed = result.cnn_codes.to(cnn.dtype) * result.cnn_scale
        transformer_source = result.transformer_accumulator.to(
            torch.float32) * result.transformer_accumulator_scales
        transformer_target = result.transformer_requantized.to(
            torch.float32) * result.target_accumulator_scales
        cnn_source = result.cnn_accumulator.to(
            torch.float32) * result.cnn_accumulator_scales
        cnn_target = result.cnn_requantized.to(
            torch.float32) * result.target_accumulator_scales
        reference_output = self._reference_output(merged)

        self._transformer_stats.update(transformer, transformer_reconstructed)
        self._cnn_stats.update(cnn, cnn_reconstructed)
        self._partial_stats.update(transformer_source, transformer_target)
        self._partial_stats.update(cnn_source, cnn_target)
        self._output_stats.update(reference_output, result.output)
        self._transformer_zeros += int(
            (result.transformer_codes == 0).sum().item())
        self._cnn_zeros += int((result.cnn_codes == 0).sum().item())
        transformer_limit = self.transformer_maximum * \
            self.selected_factors["transformer"]
        cnn_limit = self.cnn_maximum * self.selected_factors["cnn"]
        self._transformer_saturated += int(
            (transformer.abs() > transformer_limit).sum().item())
        self._cnn_saturated += int(
            (cnn.abs() > cnn_limit).sum().item())
        self._transformer_elements += int(result.transformer_codes.numel())
        self._cnn_elements += int(result.cnn_codes.numel())
        self._quantized_updates += 1
        return result.output

    def manifest(self) -> Dict[str, object]:
        if not self.selected_factors:
            raise RuntimeError("concat controller is not frozen")
        transformer_scale = self.transformer_maximum * \
            self.selected_factors["transformer"] / \
            float((1 << (self.activation_bits - 1)) - 1) \
            if self.transformer_maximum > 0.0 else 1.0
        cnn_scale = self.cnn_maximum * self.selected_factors["cnn"] / \
            float((1 << (self.activation_bits - 1)) - 1) \
            if self.cnn_maximum > 0.0 else 1.0
        target = self.weight_scales * max(
            transformer_scale, cnn_scale) * \
            self.selected_factors["accumulator"]
        output_maximum = self.output_maximum * self.selected_factors["output"]
        output_scale = output_maximum / \
            float((1 << (self.output_bits - 1)) - 1) \
            if output_maximum > 0.0 else 1.0
        target_values = [float(value) for value in target.tolist()]
        return {
            "module": self.name,
            "weight_bits": self.weight_bits,
            "activation_bits": self.activation_bits,
            "output_bits": self.output_bits,
            "transformer_scale": transformer_scale,
            "cnn_scale": cnn_scale,
            "weight_scales": [
                float(value) for value in self.weight_scales.tolist()],
            "transformer_accumulator_scales": [
                float(value) for value in
                (self.weight_scales * transformer_scale).tolist()],
            "cnn_accumulator_scales": [
                float(value) for value in
                (self.weight_scales * cnn_scale).tolist()],
            "target_accumulator_scales": target_values,
            "bias_scales": list(target_values),
            "output_scale": output_scale,
            "transformer_factor": self.selected_factors["transformer"],
            "cnn_factor": self.selected_factors["cnn"],
            "accumulator_factor": self.selected_factors["accumulator"],
            "output_factor": self.selected_factors["output"],
            "calibration_updates": self.observations,
            "cached_samples": self.cached_samples,
        }

    def statistics(self) -> List[Dict[str, object]]:
        if self._quantized_updates == 0:
            raise RuntimeError("concat controller has no quantized observations")
        return [{
            "module": self.name,
            "transformer_sqnr_db": self._transformer_stats.sqnr_db,
            "cnn_sqnr_db": self._cnn_stats.sqnr_db,
            "partial_requantization_mse": self._partial_stats.mse,
            "output_sqnr_db": self._output_stats.sqnr_db,
            "block_mse": self._output_stats.mse,
            "transformer_zero_ratio": self._transformer_zeros /
                float(self._transformer_elements),
            "cnn_zero_ratio": self._cnn_zeros / float(self._cnn_elements),
            "transformer_saturation_ratio": self._transformer_saturated /
                float(self._transformer_elements),
            "cnn_saturation_ratio": self._cnn_saturated /
                float(self._cnn_elements),
            "updates": self._quantized_updates,
        }]

    def reset_statistics(self) -> None:
        self._transformer_stats = _ErrorAccumulator()
        self._cnn_stats = _ErrorAccumulator()
        self._partial_stats = _ErrorAccumulator()
        self._output_stats = _ErrorAccumulator()
        self._transformer_zeros = 0
        self._cnn_zeros = 0
        self._transformer_saturated = 0
        self._cnn_saturated = 0
        self._transformer_elements = 0
        self._cnn_elements = 0
        self._quantized_updates = 0

    def search_rows(self) -> List[Dict[str, object]]:
        return [dict(row) for row in self._search_rows]
