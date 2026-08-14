"""Complete Conv2d Im2Col matrix capture and visualization data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from spn_quant.im2col_diagnostics import ConvIm2ColLayout


@dataclass(frozen=True)
class ConvMatrixGeometry:
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    dilation: Tuple[int, int]

    @classmethod
    def from_module(cls, module: nn.Conv2d):
        if not isinstance(module, nn.Conv2d) or int(module.groups) != 1:
            raise TypeError("full matrix capture requires groups=1 Conv2d")
        layout = ConvIm2ColLayout.from_module("capture", module)
        return cls(
            layout.in_channels, layout.out_channels, layout.kernel_size,
            layout.stride, layout.padding, layout.dilation)

    def layout(self, module: str) -> ConvIm2ColLayout:
        return ConvIm2ColLayout(
            module=str(module), in_channels=self.in_channels,
            out_channels=self.out_channels, kernel_size=self.kernel_size,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation)


@dataclass(frozen=True)
class ConvUnfoldedMatrices:
    reference_weight: torch.Tensor
    quantized_weight: torch.Tensor
    absolute_weight_error: torch.Tensor
    reference_activation: torch.Tensor
    quantized_activation: torch.Tensor
    absolute_activation_error: torch.Tensor


def _validate_fp32(name: str, tensor: torch.Tensor) -> None:
    if not torch.is_tensor(tensor) or tensor.dtype != torch.float32:
        raise TypeError("%s must be a float32 tensor" % name)
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("%s must be finite" % name)


@dataclass(frozen=True)
class ConvMatrixCapture:
    module: str
    sample_index: int
    geometry: ConvMatrixGeometry
    reference_input: torch.Tensor
    quantized_input: torch.Tensor
    original_weight: torch.Tensor
    quantized_weight: torch.Tensor
    activation_bits: int
    activation_unsigned: bool
    activation_scale: torch.Tensor

    def __post_init__(self) -> None:
        if not self.module:
            raise ValueError("capture module must be nonempty")
        if int(self.sample_index) < 0:
            raise ValueError("capture sample index must be nonnegative")
        for name, tensor in (
                ("reference input", self.reference_input),
                ("quantized input", self.quantized_input),
                ("original weight", self.original_weight),
                ("quantized weight", self.quantized_weight),
                ("activation scale", self.activation_scale)):
            _validate_fp32(name, tensor)
        if self.reference_input.device.type != "cpu" or \
                self.quantized_input.device.type != "cpu" or \
                self.original_weight.device.type != "cpu" or \
                self.quantized_weight.device.type != "cpu" or \
                self.activation_scale.device.type != "cpu":
            raise ValueError("capture tensors must reside on CPU")
        if self.reference_input.shape != self.quantized_input.shape or \
                self.reference_input.ndim != 4 or \
                int(self.reference_input.shape[0]) != 1 or \
                int(self.reference_input.shape[1]) != \
                self.geometry.in_channels:
            raise ValueError("capture input shape changed")
        expected_weight = (
            self.geometry.out_channels, self.geometry.in_channels,
            self.geometry.kernel_size[0], self.geometry.kernel_size[1])
        if tuple(self.original_weight.shape) != expected_weight or \
                self.original_weight.shape != self.quantized_weight.shape:
            raise ValueError("capture weight shape changed")
        if int(self.activation_bits) <= 0:
            raise ValueError("activation bits must be positive")
        if self.activation_scale.numel() == 0 or \
                bool((self.activation_scale <= 0).any().item()):
            raise ValueError("activation scale must be positive")

    @classmethod
    def from_tensors(
            cls, module: str, sample_index: int, layer: nn.Conv2d,
            reference_input: torch.Tensor, quantized_input: torch.Tensor,
            original_weight: torch.Tensor, quantized_weight: torch.Tensor,
            activation_bits: int, activation_unsigned: bool,
            activation_scale: torch.Tensor):
        tensors = (
            reference_input, quantized_input, original_weight,
            quantized_weight, activation_scale)
        for tensor in tensors:
            if not torch.is_tensor(tensor) or tensor.dtype != torch.float32:
                raise TypeError("capture tensors must be float32")
        return cls(
            module=str(module), sample_index=int(sample_index),
            geometry=ConvMatrixGeometry.from_module(layer),
            reference_input=reference_input.detach().cpu().clone(),
            quantized_input=quantized_input.detach().cpu().clone(),
            original_weight=original_weight.detach().cpu().clone(),
            quantized_weight=quantized_weight.detach().cpu().clone(),
            activation_bits=int(activation_bits),
            activation_unsigned=bool(activation_unsigned),
            activation_scale=activation_scale.detach().cpu().clone())

    def matrices(self) -> ConvUnfoldedMatrices:
        layout = self.geometry.layout(self.module)
        reference_activation = layout.unfold(
            self.reference_input)[0].transpose(0, 1).contiguous()
        quantized_activation = layout.unfold(
            self.quantized_input)[0].transpose(0, 1).contiguous()
        reference_weight = layout.flatten_weight(
            self.original_weight).contiguous()
        quantized_weight = layout.flatten_weight(
            self.quantized_weight).contiguous()
        return ConvUnfoldedMatrices(
            reference_weight=reference_weight,
            quantized_weight=quantized_weight,
            absolute_weight_error=(
                reference_weight - quantized_weight).abs(),
            reference_activation=reference_activation,
            quantized_activation=quantized_activation,
            absolute_activation_error=(
                reference_activation - quantized_activation).abs())

    def save(self, path: Path) -> None:
        path = Path(path)
        if path.exists():
            raise FileExistsError(str(path))
        if not path.parent.is_dir():
            raise FileNotFoundError(str(path.parent))
        with path.open("xb") as handle:
            np.savez_compressed(
                handle,
                module=np.asarray(self.module),
                sample_index=np.asarray(self.sample_index, dtype=np.int64),
                in_channels=np.asarray(
                    self.geometry.in_channels, dtype=np.int64),
                out_channels=np.asarray(
                    self.geometry.out_channels, dtype=np.int64),
                kernel_size=np.asarray(
                    self.geometry.kernel_size, dtype=np.int64),
                stride=np.asarray(self.geometry.stride, dtype=np.int64),
                padding=np.asarray(self.geometry.padding, dtype=np.int64),
                dilation=np.asarray(self.geometry.dilation, dtype=np.int64),
                reference_input=self.reference_input.numpy(),
                quantized_input=self.quantized_input.numpy(),
                original_weight=self.original_weight.numpy(),
                quantized_weight=self.quantized_weight.numpy(),
                activation_bits=np.asarray(
                    self.activation_bits, dtype=np.int64),
                activation_unsigned=np.asarray(
                    self.activation_unsigned, dtype=np.bool_),
                activation_scale=self.activation_scale.numpy())

    @classmethod
    def load(cls, path: Path):
        expected = {
            "module", "sample_index", "in_channels", "out_channels",
            "kernel_size", "stride", "padding", "dilation",
            "reference_input", "quantized_input", "original_weight",
            "quantized_weight", "activation_bits",
            "activation_unsigned", "activation_scale"}
        with np.load(Path(path), allow_pickle=False) as source:
            if set(source.files) != expected:
                raise ValueError("matrix capture schema changed")
            geometry = ConvMatrixGeometry(
                in_channels=int(source["in_channels"].item()),
                out_channels=int(source["out_channels"].item()),
                kernel_size=tuple(int(value) for value in source[
                    "kernel_size"].tolist()),
                stride=tuple(int(value) for value in source["stride"].tolist()),
                padding=tuple(
                    int(value) for value in source["padding"].tolist()),
                dilation=tuple(
                    int(value) for value in source["dilation"].tolist()))
            return cls(
                module=str(source["module"].item()),
                sample_index=int(source["sample_index"].item()),
                geometry=geometry,
                reference_input=torch.from_numpy(
                    source["reference_input"].copy()),
                quantized_input=torch.from_numpy(
                    source["quantized_input"].copy()),
                original_weight=torch.from_numpy(
                    source["original_weight"].copy()),
                quantized_weight=torch.from_numpy(
                    source["quantized_weight"].copy()),
                activation_bits=int(source["activation_bits"].item()),
                activation_unsigned=bool(
                    source["activation_unsigned"].item()),
                activation_scale=torch.from_numpy(
                    source["activation_scale"].copy()))


@dataclass(frozen=True)
class FullLineCurtain:
    matrix_shape: Tuple[int, int]
    orientation: str
    lines: Tuple[np.ndarray, ...]
    rendered_elements: int

    def reconstruct(self) -> np.ndarray:
        matrix = np.empty(self.matrix_shape, dtype=np.float32)
        if self.orientation == "rows_along_x":
            for row, line in enumerate(self.lines):
                matrix[row] = line[:, 2]
        elif self.orientation == "columns_along_y":
            for column, line in enumerate(self.lines):
                matrix[:, column] = line[:, 2]
        else:
            raise ValueError("unknown line curtain orientation")
        return matrix


def full_line_curtain(matrix: np.ndarray) -> FullLineCurtain:
    if not isinstance(matrix, np.ndarray) or matrix.dtype != np.float32:
        raise TypeError("line curtain matrix must be float32")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("line curtain matrix must be nonempty rank two")
    if not bool(np.isfinite(matrix).all()):
        raise ValueError("line curtain matrix must be finite")
    rows, columns = matrix.shape
    lines = []
    if rows <= columns:
        x = np.arange(columns, dtype=np.float32)
        for row in range(rows):
            lines.append(np.column_stack((
                x, np.full(columns, row, dtype=np.float32), matrix[row])))
        orientation = "rows_along_x"
    else:
        y = np.arange(rows, dtype=np.float32)
        for column in range(columns):
            lines.append(np.column_stack((
                np.full(rows, column, dtype=np.float32), y,
                matrix[:, column])))
        orientation = "columns_along_y"
    return FullLineCurtain(
        matrix_shape=(rows, columns), orientation=orientation,
        lines=tuple(lines), rendered_elements=int(matrix.size))


class FullMatrixCaptureRecorder:
    """Capture declared native Conv inputs from hardware-aligned QDQ events."""

    def __init__(self, modules: Dict[str, nn.Module],
                 original_weights: Dict[str, torch.Tensor], selection) -> None:
        if set(modules) != set(original_weights):
            raise ValueError("capture module and weight identities differ")
        self.modules = dict(modules)
        self.original_weights = dict(original_weights)
        self.selection = dict(
            (int(index), set(str(name) for name in names))
            for index, names in selection.items())
        if not self.selection or any(not names for names in self.selection.values()):
            raise ValueError("capture selection must be nonempty")
        requested = set().union(*self.selection.values())
        if not requested.issubset(set(self.modules)):
            raise ValueError("capture selection contains unknown modules")
        for name in requested:
            if not isinstance(self.modules[name], nn.Conv2d):
                raise TypeError("full matrix capture requires Conv2d modules")
        self.sample_index = None
        self.captures = {}

    def begin_sample(self, sample_index: int) -> None:
        sample_index = int(sample_index)
        if self.sample_index is not None:
            raise RuntimeError("previous capture sample is active")
        if sample_index not in self.selection:
            raise ValueError("capture sample is not selected")
        self.sample_index = sample_index
        self.captures = {}

    def record(self, module: str, kind: str, call_index: int, group: str,
               reference: torch.Tensor, quantized: torch.Tensor,
               codes: torch.Tensor, quantizer: object,
               channel_dim: int) -> None:
        del group, codes
        if self.sample_index is None:
            raise RuntimeError("begin_sample must precede matrix capture")
        if kind != "input" or module not in self.selection[self.sample_index]:
            return
        if int(call_index) != 0 or int(channel_dim) != 1:
            raise ValueError("selected Conv input identity changed")
        if module in self.captures:
            raise ValueError("selected Conv input was captured twice")
        if quantizer.format != "uniform" or int(quantizer.zero_point) != 0:
            raise ValueError("matrix capture requires zero-preserving uniform QDQ")
        scale = torch.as_tensor(
            quantizer.scale_for(reference), device=reference.device,
            dtype=reference.dtype)
        layer = self.modules[module]
        self.captures[module] = ConvMatrixCapture.from_tensors(
            module, self.sample_index, layer, reference, quantized,
            self.original_weights[module].to(
                device=reference.device, dtype=reference.dtype),
            layer.weight.detach(), int(quantizer.bits),
            bool(quantizer.unsigned), scale)

    def end_sample(self) -> Dict[str, ConvMatrixCapture]:
        if self.sample_index is None:
            raise RuntimeError("no capture sample is active")
        expected = self.selection[self.sample_index]
        if set(self.captures) != expected:
            raise RuntimeError(
                "matrix capture coverage mismatch: missing=%s" %
                sorted(expected - set(self.captures)))
        return dict(self.captures)

    def clear_sample(self) -> None:
        if self.sample_index is None:
            raise RuntimeError("no capture sample is active")
        if set(self.captures) != self.selection[self.sample_index]:
            raise RuntimeError("cannot clear incomplete matrix capture")
        self.sample_index = None
        self.captures = {}
