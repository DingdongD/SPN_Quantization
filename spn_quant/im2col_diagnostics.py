"""Structured Conv2d Im2Col diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
