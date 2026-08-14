"""Structured Conv2d Im2Col diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spn_quant.activation_resolution import BoundedChannelSampler


def _pair(value) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    return int(value), int(value)


@dataclass(frozen=True)
class ConvIm2ColLayout:
    module: str
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    dilation: Tuple[int, int]

    @classmethod
    def from_module(cls, name: str, module: nn.Conv2d):
        if not isinstance(module, nn.Conv2d):
            raise TypeError("Im2Col diagnostics require Conv2d: %s" % name)
        if int(module.groups) != 1:
            raise ValueError(
                "Im2Col diagnostics require groups=1: %s" % name)
        return cls(
            module=str(name),
            in_channels=int(module.in_channels),
            out_channels=int(module.out_channels),
            kernel_size=_pair(module.kernel_size),
            stride=_pair(module.stride),
            padding=_pair(module.padding),
            dilation=_pair(module.dilation),
        )

    @property
    def k_size(self) -> int:
        return self.in_channels * self.kernel_size[0] * self.kernel_size[1]

    def output_shape(self, inputs: torch.Tensor) -> Tuple[int, int]:
        self._validate_inputs(inputs)
        height = (
            int(inputs.shape[2]) + 2 * self.padding[0]
            - self.dilation[0] * (self.kernel_size[0] - 1) - 1
        ) // self.stride[0] + 1
        width = (
            int(inputs.shape[3]) + 2 * self.padding[1]
            - self.dilation[1] * (self.kernel_size[1] - 1) - 1
        ) // self.stride[1] + 1
        if height <= 0 or width <= 0:
            raise ValueError("Im2Col output shape must be positive")
        return height, width

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        if not torch.is_tensor(inputs) or inputs.ndim != 4:
            raise ValueError("Im2Col inputs must be rank-4 tensors")
        if int(inputs.shape[1]) != self.in_channels:
            raise ValueError("Im2Col input channel count changed")
        if not bool(torch.isfinite(inputs).all().item()):
            raise ValueError("Im2Col inputs must be finite")

    def unfold(self, inputs: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(inputs)
        patches = F.unfold(
            inputs,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )
        expected_tokens = self.output_shape(inputs)[0] * \
            self.output_shape(inputs)[1]
        if tuple(patches.shape[1:]) != (self.k_size, expected_tokens):
            raise RuntimeError("Im2Col output layout changed")
        return patches

    def unfold_token_range(self, inputs: torch.Tensor,
                           start: int, stop: int) -> torch.Tensor:
        self._validate_inputs(inputs)
        height, width = self.output_shape(inputs)
        tokens = height * width
        start = int(start)
        stop = int(stop)
        if start < 0 or stop <= start or stop > tokens:
            raise ValueError("Im2Col token range is invalid")
        padded = F.pad(inputs, (
            self.padding[1], self.padding[1],
            self.padding[0], self.padding[0]))
        effective_height = self.dilation[0] * \
            (self.kernel_size[0] - 1) + 1
        effective_width = self.dilation[1] * \
            (self.kernel_size[1] - 1) + 1
        windows = padded.unfold(
            2, effective_height, self.stride[0]).unfold(
            3, effective_width, self.stride[1])
        windows = windows[
            ..., ::self.dilation[0], ::self.dilation[1]]
        indices = torch.arange(start, stop, device=inputs.device)
        rows = torch.div(indices, width, rounding_mode="floor")
        columns = indices.remainder(width)
        selected = windows[:, :, rows, columns, :, :]
        return selected.permute(0, 1, 3, 4, 2).reshape(
            inputs.shape[0], self.k_size, stop - start)

    def flatten_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(weight) or tuple(weight.shape) != (
                self.out_channels, self.in_channels,
                self.kernel_size[0], self.kernel_size[1]):
            raise ValueError("Conv2d weight shape changed: %s" % self.module)
        if not bool(torch.isfinite(weight).all().item()):
            raise ValueError("Conv2d weights must be finite")
        return weight.reshape(self.out_channels, self.k_size)

    def decode_k(self, index: int) -> Tuple[int, int, int]:
        index = int(index)
        if index < 0 or index >= self.k_size:
            raise ValueError("K index is outside the Conv2d layout")
        offset_count = self.kernel_size[0] * self.kernel_size[1]
        channel = index // offset_count
        offset = index % offset_count
        return channel, offset // self.kernel_size[1], \
            offset % self.kernel_size[1]


def local_output_error(
        fp_patches: torch.Tensor,
        quantized_patches: torch.Tensor,
        fp_weight: torch.Tensor,
        quantized_weight: torch.Tensor) -> torch.Tensor:
    if fp_patches.shape != quantized_patches.shape or \
            fp_patches.ndim != 3:
        raise ValueError("FP and quantized patches must share [B, K, M]")
    if fp_weight.shape != quantized_weight.shape or fp_weight.ndim != 2:
        raise ValueError("FP and quantized weights must share [Cout, K]")
    if int(fp_patches.shape[1]) != int(fp_weight.shape[1]):
        raise ValueError("patch and weight K dimensions differ")
    tensors = (
        fp_patches, quantized_patches, fp_weight, quantized_weight)
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
        raise ValueError("local Conv output diagnostics require finite tensors")
    fp_output = torch.einsum("bkm,ok->bom", fp_patches, fp_weight)
    quantized_output = torch.einsum(
        "bkm,ok->bom", quantized_patches, quantized_weight)
    return (quantized_output - fp_output).square().sum(dim=1)


def _ratio(numerator: float, denominator: float) -> float:
    if denominator > 0.0:
        return numerator / denominator
    return 0.0 if numerator == 0.0 else float("inf")


def _sqnr(signal: float, error: float) -> float:
    if error == 0.0:
        return float("inf")
    if signal == 0.0:
        return float("-inf")
    return 10.0 * math.log10(signal / error)


class ConvIm2ColAccumulator:
    """Accumulate Conv input and weight diagnostics over structured K."""

    def __init__(self, layout: ConvIm2ColLayout,
                 percentile_capacity: int, token_topk: int) -> None:
        self.layout = layout
        self.percentile_capacity = int(percentile_capacity)
        self.token_topk = int(token_topk)
        if self.percentile_capacity <= 0:
            raise ValueError("percentile capacity must be positive")
        if self.token_topk <= 0:
            raise ValueError("token top-k must be positive")
        size = self.layout.k_size
        self.elements = torch.zeros(size, dtype=torch.int64)
        self.reference_zeros = torch.zeros(size, dtype=torch.int64)
        self.quantized_zeros = torch.zeros(size, dtype=torch.int64)
        self.nonzero_elements = torch.zeros(size, dtype=torch.int64)
        self.new_zero_elements = torch.zeros(size, dtype=torch.int64)
        self.saturated = torch.zeros(size, dtype=torch.int64)
        self.signal_energy = torch.zeros(size, dtype=torch.float64)
        self.activation_abs_sum = torch.zeros(size, dtype=torch.float64)
        self.error_energy = torch.zeros(size, dtype=torch.float64)
        self.zero_collapse_energy = torch.zeros(size, dtype=torch.float64)
        self.rounding_energy = torch.zeros(size, dtype=torch.float64)
        self.clipping_energy = torch.zeros(size, dtype=torch.float64)
        self.maximum_abs = torch.zeros(size, dtype=torch.float64)
        self.sampler = BoundedChannelSampler(
            size, self.percentile_capacity)
        self.weight_initialized = False
        self.fp_weight = None
        self.quantized_weight = None
        self.weight_signal_energy = torch.zeros(size, dtype=torch.float64)
        self.weight_error_energy = torch.zeros(size, dtype=torch.float64)
        self.weight_abs_sum = torch.zeros(size, dtype=torch.float64)
        self.weight_maximum_abs = torch.zeros(size, dtype=torch.float64)
        self.weight_elements = self.layout.out_channels
        self._spatial = {}  # type: Dict[int, Dict[str, np.ndarray]]
        self._top_tokens = []  # type: List[Dict[str, object]]
        self.local_output_signal_energy = 0.0
        self.local_output_error_energy = 0.0
        self.local_output_elements = 0
        self.local_error_sampler = BoundedChannelSampler(
            1, self.percentile_capacity)

    def _initialize_weight(self, fp_weight: torch.Tensor,
                           quantized_weight: torch.Tensor) -> None:
        fp = self.layout.flatten_weight(fp_weight).detach().cpu()
        quantized = self.layout.flatten_weight(
            quantized_weight).detach().cpu()
        if self.weight_initialized:
            if not torch.equal(fp, self.fp_weight) or \
                    not torch.equal(quantized, self.quantized_weight):
                raise ValueError("Conv diagnostic weights changed between samples")
            return
        difference = quantized.to(torch.float64) - fp.to(torch.float64)
        values = fp.to(torch.float64)
        self.weight_signal_energy = values.square().sum(dim=0)
        self.weight_error_energy = difference.square().sum(dim=0)
        self.weight_abs_sum = values.abs().sum(dim=0)
        self.weight_maximum_abs = values.abs().amax(dim=0)
        self.fp_weight = fp
        self.quantized_weight = quantized
        self.weight_initialized = True

    @staticmethod
    def _channel_first(patches: torch.Tensor) -> torch.Tensor:
        return patches.permute(1, 0, 2).reshape(patches.shape[1], -1)

    def update(self, reference: torch.Tensor, quantized: torch.Tensor,
               codes: torch.Tensor, quantizer: object,
               fp_weight: torch.Tensor, quantized_weight: torch.Tensor,
               sample_index: int, token_chunk: int) -> None:
        if reference.shape != quantized.shape or reference.shape != codes.shape:
            raise ValueError("Conv input reference, QDQ and codes must share shape")
        if int(reference.shape[0]) != 1:
            raise ValueError("spatial diagnostics require batch size one")
        if quantizer.format != "uniform" or int(quantizer.zero_point) != 0:
            raise ValueError("Im2Col diagnostics require zero-preserving uniform QDQ")
        token_chunk = int(token_chunk)
        if token_chunk <= 0:
            raise ValueError("token chunk must be positive")
        self._initialize_weight(fp_weight, quantized_weight)
        scale = torch.as_tensor(
            quantizer.scale_for(reference), device=reference.device,
            dtype=reference.dtype)
        if not bool(torch.isfinite(scale).all().item()) or \
                bool((scale <= 0).any().item()):
            raise ValueError("activation scale must be finite and positive")
        unrounded = reference / scale
        zero_mask = (reference != 0) & (codes == 0)
        clipping_mask = ((unrounded < int(quantizer.qmin)) |
                         (unrounded > int(quantizer.qmax))) & ~zero_mask
        rounding_mask = ~(zero_mask | clipping_mask)
        height, width = self.layout.output_shape(reference)
        tokens = height * width
        patch_rms = torch.empty(tokens, dtype=torch.float32)
        patch_p99 = torch.empty(tokens, dtype=torch.float32)
        patch_maximum = torch.empty(tokens, dtype=torch.float32)
        patch_error = torch.empty(tokens, dtype=torch.float64)
        patch_new_zeros = torch.empty(tokens, dtype=torch.int64)
        patch_saturation = torch.empty(tokens, dtype=torch.int64)
        patch_output_error = torch.empty(tokens, dtype=torch.float32)
        fp_flat_weight = self.layout.flatten_weight(fp_weight)
        quantized_flat_weight = self.layout.flatten_weight(quantized_weight)

        for start in range(0, tokens, token_chunk):
            stop = min(start + token_chunk, tokens)
            fp = self.layout.unfold_token_range(reference, start, stop)
            qdq = self.layout.unfold_token_range(quantized, start, stop)
            chunk_codes = self.layout.unfold_token_range(
                codes.to(reference.dtype), start, stop)
            chunk_zero = self.layout.unfold_token_range(
                zero_mask.to(reference.dtype), start, stop).bool()
            chunk_clip = self.layout.unfold_token_range(
                clipping_mask.to(reference.dtype), start, stop).bool()
            chunk_round = self.layout.unfold_token_range(
                rounding_mask.to(reference.dtype), start, stop).bool()
            error = (qdq - fp).to(torch.float64).square()
            signal = fp.to(torch.float64).square()
            channel_fp = self._channel_first(fp)
            channel_error = self._channel_first(error)
            count = int(channel_fp.shape[1])
            self.elements += count
            self.reference_zeros += self._channel_first(
                fp == 0).sum(dim=1).cpu()
            self.quantized_zeros += self._channel_first(
                chunk_codes == 0).sum(dim=1).cpu()
            self.nonzero_elements += self._channel_first(
                fp != 0).sum(dim=1).cpu()
            self.new_zero_elements += self._channel_first(
                chunk_zero).sum(dim=1).cpu()
            self.saturated += self._channel_first(
                (chunk_codes == int(quantizer.qmin)) |
                (chunk_codes == int(quantizer.qmax))).sum(dim=1).cpu()
            self.signal_energy += self._channel_first(signal).sum(dim=1).cpu()
            self.activation_abs_sum += channel_fp.abs().to(
                torch.float64).sum(dim=1).cpu()
            self.error_energy += channel_error.sum(dim=1).cpu()
            self.zero_collapse_energy += (
                channel_error * self._channel_first(chunk_zero)
            ).sum(dim=1).cpu()
            self.clipping_energy += (
                channel_error * self._channel_first(chunk_clip)
            ).sum(dim=1).cpu()
            self.rounding_energy += (
                channel_error * self._channel_first(chunk_round)
            ).sum(dim=1).cpu()
            self.maximum_abs = torch.maximum(
                self.maximum_abs,
                channel_fp.abs().amax(dim=1).to(torch.float64).cpu())
            self.sampler.update(channel_fp)

            absolute = fp[0].abs().transpose(0, 1)
            patch_rms[start:stop] = torch.sqrt(
                absolute.to(torch.float64).square().mean(dim=1)
            ).to(torch.float32).cpu()
            patch_p99[start:stop] = torch.quantile(
                absolute, 0.99, dim=1).cpu()
            patch_maximum[start:stop] = absolute.amax(dim=1).cpu()
            patch_error[start:stop] = error.sum(dim=1)[0].cpu()
            patch_new_zeros[start:stop] = chunk_zero.sum(dim=1)[0].cpu()
            patch_saturation[start:stop] = (
                (chunk_codes == int(quantizer.qmin)) |
                (chunk_codes == int(quantizer.qmax))
            ).sum(dim=1)[0].cpu()
            fp_output = torch.einsum("bkm,ok->bom", fp, fp_flat_weight)
            quantized_output = torch.einsum(
                "bkm,ok->bom", qdq, quantized_flat_weight)
            output_error = (
                quantized_output - fp_output).square().sum(dim=1)
            patch_output_error[start:stop] = output_error[0].cpu()
            self.local_output_signal_energy += float(
                fp_output.to(torch.float64).square().sum().item())
            self.local_output_error_energy += float(
                output_error.to(torch.float64).sum().item())
            self.local_output_elements += int(fp_output.numel())
            self.local_error_sampler.update(output_error.reshape(1, -1))

        arrays = {
            "patch_rms": patch_rms.reshape(height, width).numpy(),
            "patch_p99": patch_p99.reshape(height, width).numpy(),
            "patch_maximum_abs": patch_maximum.reshape(height, width).numpy(),
            "activation_error": patch_error.reshape(height, width).numpy(),
            "activation_new_zero_count": patch_new_zeros.reshape(
                height, width).numpy(),
            "activation_saturation_count": patch_saturation.reshape(
                height, width).numpy(),
            "local_output_error": patch_output_error.reshape(
                height, width).numpy(),
        }
        sample_index = int(sample_index)
        if sample_index in self._spatial:
            raise ValueError("spatial sample was recorded more than once")
        self._spatial[sample_index] = arrays
        count = min(self.token_topk, tokens)
        values, indices = torch.topk(patch_output_error, count)
        candidates = list(self._top_tokens)
        for value, index in zip(values.tolist(), indices.tolist()):
            candidates.append({
                "module": self.layout.module,
                "sample_index": sample_index,
                "output_row": int(index) // width,
                "output_col": int(index) % width,
                "local_output_error": float(value),
                "patch_rms": float(patch_rms[index].item()),
                "activation_error": float(patch_error[index].item()),
                "activation_new_zero_count": int(
                    patch_new_zeros[index].item()),
            })
        self._top_tokens = sorted(
            candidates,
            key=lambda row: (
                -float(row["local_output_error"]),
                int(row["sample_index"]), int(row["output_row"]),
                int(row["output_col"])),
        )[:self.token_topk]

    def _row(self, indices: torch.Tensor) -> Dict[str, object]:
        elements = int(self.elements.index_select(0, indices).sum().item())
        nonzero = int(
            self.nonzero_elements.index_select(0, indices).sum().item())
        signal = float(
            self.signal_energy.index_select(0, indices).sum().item())
        error = float(
            self.error_energy.index_select(0, indices).sum().item())
        weight_signal = float(
            self.weight_signal_energy.index_select(0, indices).sum().item())
        weight_error = float(
            self.weight_error_energy.index_select(0, indices).sum().item())
        return {
            "elements": elements,
            "activation_reference_zero_count": int(
                self.reference_zeros.index_select(0, indices).sum().item()),
            "activation_quantized_zero_count": int(
                self.quantized_zeros.index_select(0, indices).sum().item()),
            "activation_nonzero_count": nonzero,
            "activation_new_zero_count": int(
                self.new_zero_elements.index_select(0, indices).sum().item()),
            "activation_saturation_count": int(
                self.saturated.index_select(0, indices).sum().item()),
            "activation_signal_energy": signal,
            "activation_mean_abs": float(
                self.activation_abs_sum.index_select(
                    0, indices).sum().item()) / float(elements),
            "activation_rms": math.sqrt(signal / float(elements)),
            "activation_error_energy": error,
            "activation_zero_collapse_energy": float(
                self.zero_collapse_energy.index_select(0, indices).sum().item()),
            "activation_rounding_energy": float(
                self.rounding_energy.index_select(0, indices).sum().item()),
            "activation_clipping_energy": float(
                self.clipping_energy.index_select(0, indices).sum().item()),
            "activation_sqnr_db": _sqnr(signal, error),
            "activation_new_zero_rate": _ratio(
                float(self.new_zero_elements.index_select(
                    0, indices).sum().item()), float(nonzero)),
            "activation_saturation_rate": _ratio(
                float(self.saturated.index_select(0, indices).sum().item()),
                float(elements)),
            "activation_maximum_abs": float(
                self.maximum_abs.index_select(0, indices).max().item()),
            "weight_elements": self.weight_elements * int(indices.numel()),
            "weight_signal_energy": weight_signal,
            "weight_rms": math.sqrt(
                weight_signal /
                float(self.weight_elements * int(indices.numel()))),
            "weight_error_energy": weight_error,
            "weight_sqnr_db": _sqnr(weight_signal, weight_error),
            "weight_mean_abs": float(
                self.weight_abs_sum.index_select(0, indices).sum().item()) /
            float(self.weight_elements * int(indices.numel())),
            "weight_maximum_abs": float(
                self.weight_maximum_abs.index_select(0, indices).max().item()),
        }

    def exact_state(self) -> Dict[str, object]:
        return {
            "elements": tuple(self.elements.tolist()),
            "reference_zeros": tuple(self.reference_zeros.tolist()),
            "quantized_zeros": tuple(self.quantized_zeros.tolist()),
            "nonzero_elements": tuple(self.nonzero_elements.tolist()),
            "new_zero_elements": tuple(self.new_zero_elements.tolist()),
            "saturated": tuple(self.saturated.tolist()),
            "signal_energy": tuple(self.signal_energy.tolist()),
            "error_energy": tuple(self.error_energy.tolist()),
            "zero_collapse_energy": tuple(
                self.zero_collapse_energy.tolist()),
            "rounding_energy": tuple(self.rounding_energy.tolist()),
            "clipping_energy": tuple(self.clipping_energy.tolist()),
            "maximum_abs": tuple(self.maximum_abs.tolist()),
            "activation_abs_sum": tuple(self.activation_abs_sum.tolist()),
            "weight_signal_energy": tuple(
                self.weight_signal_energy.tolist()),
            "weight_error_energy": tuple(self.weight_error_energy.tolist()),
        }

    def channel_offset_rows(self) -> List[Dict[str, object]]:
        if int(self.elements.sum().item()) == 0:
            raise RuntimeError("cannot summarize empty Im2Col diagnostics")
        percentiles = self.sampler.percentiles((0.75, 0.99, 0.999))
        rows = []
        for index in range(self.layout.k_size):
            channel, kernel_row, kernel_col = self.layout.decode_k(index)
            indices = torch.tensor([index], dtype=torch.long)
            row = {
                "module": self.layout.module,
                "channel": channel,
                "kernel_row": kernel_row,
                "kernel_col": kernel_col,
                "kernel_offset": kernel_row * self.layout.kernel_size[1] +
                kernel_col,
                "activation_p75": float(percentiles[index, 0].item()),
                "activation_p99": float(percentiles[index, 1].item()),
                "activation_p99_9": float(percentiles[index, 2].item()),
            }
            row.update(self._row(indices))
            rows.append(row)
        return rows

    def channel_rows(self) -> List[Dict[str, object]]:
        offset_count = self.layout.kernel_size[0] * self.layout.kernel_size[1]
        rows = []
        for channel in range(self.layout.in_channels):
            indices = torch.arange(
                channel * offset_count, (channel + 1) * offset_count)
            row = {"module": self.layout.module, "channel": channel}
            row.update(self._row(indices))
            rows.append(row)
        return rows

    def offset_rows(self) -> List[Dict[str, object]]:
        offset_count = self.layout.kernel_size[0] * self.layout.kernel_size[1]
        rows = []
        for offset in range(offset_count):
            indices = torch.arange(
                offset, self.layout.k_size, step=offset_count)
            row = {
                "module": self.layout.module,
                "kernel_row": offset // self.layout.kernel_size[1],
                "kernel_col": offset % self.layout.kernel_size[1],
                "kernel_offset": offset,
            }
            row.update(self._row(indices))
            rows.append(row)
        return rows

    def layer_row(self) -> Dict[str, object]:
        indices = torch.arange(self.layout.k_size)
        row = {
            "module": self.layout.module,
            "in_channels": self.layout.in_channels,
            "out_channels": self.layout.out_channels,
            "kernel_height": self.layout.kernel_size[0],
            "kernel_width": self.layout.kernel_size[1],
        }
        row.update(self._row(indices))
        local_errors = [
            float(row["local_output_error"]) for row in self._top_tokens]
        row["maximum_local_output_error"] = max(local_errors) \
            if local_errors else 0.0
        local_percentiles = self.local_error_sampler.global_percentiles(
            (0.75, 0.99, 0.999))
        row.update({
            "local_output_elements": self.local_output_elements,
            "local_output_signal_energy": self.local_output_signal_energy,
            "local_output_error_energy": self.local_output_error_energy,
            "local_output_sqnr_db": _sqnr(
                self.local_output_signal_energy,
                self.local_output_error_energy),
            "local_output_error_p75": float(local_percentiles[0].item()),
            "local_output_error_p99": float(local_percentiles[1].item()),
            "local_output_error_p99_9": float(local_percentiles[2].item()),
        })
        return row

    def spatial_arrays(self, sample_index: int) -> Dict[str, np.ndarray]:
        sample_index = int(sample_index)
        if sample_index not in self._spatial:
            raise ValueError("spatial sample was not recorded")
        return dict(self._spatial[sample_index])

    def clear_spatial_arrays(self, sample_index: int) -> None:
        sample_index = int(sample_index)
        if sample_index not in self._spatial:
            raise ValueError("spatial sample was not recorded")
        del self._spatial[sample_index]

    def top_token_rows(self) -> List[Dict[str, object]]:
        return [dict(row) for row in self._top_tokens]


class CSPNW8A8Im2ColRecorder:
    """Bridge hardware-aligned activation QDQ events to Conv diagnostics."""

    def __init__(self, modules: Dict[str, nn.Module],
                 original_weights: Dict[str, torch.Tensor],
                 percentile_capacity: int, token_topk: int,
                 token_chunk: int) -> None:
        if set(modules) != set(original_weights):
            raise ValueError("diagnostic module and weight identities differ")
        self.modules = dict(modules)
        self.original_weights = dict(original_weights)
        self.token_chunk = int(token_chunk)
        if self.token_chunk <= 0:
            raise ValueError("token chunk must be positive")
        self.accumulators = {}
        self.status = {}
        for name in sorted(self.modules):
            module = self.modules[name]
            if isinstance(module, nn.Conv2d):
                if int(module.groups) != 1:
                    raise ValueError(
                        "CSPN Im2Col diagnostics require groups=1: %s" % name)
                layout = ConvIm2ColLayout.from_module(name, module)
                self.accumulators[name] = ConvIm2ColAccumulator(
                    layout, percentile_capacity, token_topk)
                self.status[name] = "collected_conv2d"
            elif isinstance(module, nn.ConvTranspose2d):
                self.status[name] = "excluded_conv_transpose2d"
            elif isinstance(module, nn.Linear):
                self.status[name] = "excluded_linear"
            else:
                raise TypeError(
                    "unknown instrumented module type: %s" % type(module).__name__)
        if not self.accumulators:
            raise ValueError("CSPN Im2Col diagnostics found no Conv2d modules")
        self.sample_index = None
        self.observed = set()

    def begin_sample(self, sample_index: int) -> None:
        if self.sample_index is not None:
            raise RuntimeError("previous diagnostic sample is still active")
        self.sample_index = int(sample_index)
        self.observed = set()

    def record(self, module: str, kind: str, call_index: int, group: str,
               reference: torch.Tensor, quantized: torch.Tensor,
               codes: torch.Tensor, quantizer: object,
               channel_dim: int) -> None:
        del group
        if kind != "input" or module not in self.accumulators:
            return
        if self.sample_index is None:
            raise RuntimeError("begin_sample must precede Conv diagnostics")
        if int(call_index) != 0:
            raise ValueError("Conv input was invoked more than once: %s" % module)
        if int(channel_dim) != 1:
            raise ValueError("Conv input channel dimension must be one")
        if module in self.observed:
            raise ValueError("Conv input was recorded more than once: %s" % module)
        layer = self.modules[module]
        self.accumulators[module].update(
            reference, quantized, codes, quantizer,
            self.original_weights[module].to(
                device=reference.device, dtype=reference.dtype),
            layer.weight.detach(), self.sample_index, self.token_chunk)
        self.observed.add(module)

    def end_sample(self) -> Dict[str, Dict[str, np.ndarray]]:
        if self.sample_index is None:
            raise RuntimeError("no diagnostic sample is active")
        expected = set(self.accumulators)
        if self.observed != expected:
            raise RuntimeError(
                "Conv diagnostic coverage mismatch: missing=%s" %
                sorted(expected - self.observed))
        return dict(
            (name, self.accumulators[name].spatial_arrays(self.sample_index))
            for name in sorted(self.accumulators))

    def clear_sample(self) -> None:
        if self.sample_index is None:
            raise RuntimeError("no diagnostic sample is active")
        if self.observed != set(self.accumulators):
            raise RuntimeError("cannot clear an incomplete diagnostic sample")
        for accumulator in self.accumulators.values():
            accumulator.clear_spatial_arrays(self.sample_index)
        self.sample_index = None
        self.observed = set()

    def module_manifest_rows(self) -> List[Dict[str, object]]:
        rows = []
        for name in sorted(self.modules):
            module = self.modules[name]
            row = {
                "module": name,
                "module_type": type(module).__name__,
                "status": self.status[name],
            }
            if name in self.accumulators:
                layout = self.accumulators[name].layout
                row.update({
                    "in_channels": layout.in_channels,
                    "out_channels": layout.out_channels,
                    "kernel_height": layout.kernel_size[0],
                    "kernel_width": layout.kernel_size[1],
                    "stride_height": layout.stride[0],
                    "stride_width": layout.stride[1],
                    "padding_height": layout.padding[0],
                    "padding_width": layout.padding[1],
                    "dilation_height": layout.dilation[0],
                    "dilation_width": layout.dilation[1],
                })
            rows.append(row)
        return rows

    def channel_offset_rows(self) -> List[Dict[str, object]]:
        return [
            row for name in sorted(self.accumulators)
            for row in self.accumulators[name].channel_offset_rows()]

    def channel_rows(self) -> List[Dict[str, object]]:
        return [
            row for name in sorted(self.accumulators)
            for row in self.accumulators[name].channel_rows()]

    def offset_rows(self) -> List[Dict[str, object]]:
        return [
            row for name in sorted(self.accumulators)
            for row in self.accumulators[name].offset_rows()]

    def layer_rows(self) -> List[Dict[str, object]]:
        return [
            self.accumulators[name].layer_row()
            for name in sorted(self.accumulators)]

    def top_token_rows(self) -> List[Dict[str, object]]:
        return [
            row for name in sorted(self.accumulators)
            for row in self.accumulators[name].top_token_rows()]
