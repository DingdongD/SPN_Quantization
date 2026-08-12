"""Add/Concat quantization policies with explicit requantization boundaries."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from spn_quant.integer_ops import add_requantized_int32
from spn_quant.runtime import EdgeQDQRuntime


_VALID_POLICIES = frozenset(("shared", "independent", "grouped", "residual"))
_VALID_OPERATIONS = frozenset(("add", "concat"))


class TensorMinMaxObserver(object):
    def __init__(self) -> None:
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self.samples = 0
        self.elements = 0
        self.signal_energy = 0.0

    @property
    def observed(self) -> bool:
        return self.samples > 0

    def update(self, tensor: torch.Tensor) -> None:
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            return
        detached = tensor.detach()
        self.minimum = min(self.minimum, float(detached.min().item()))
        self.maximum = max(self.maximum, float(detached.max().item()))
        self.samples += 1
        self.elements += int(detached.numel())
        self.signal_energy += float(
            detached.to(torch.float64).square().sum().item())

    @property
    def rms(self) -> float:
        if self.elements == 0:
            raise RuntimeError("cannot summarize an unobserved merge branch")
        return math.sqrt(self.signal_energy / float(self.elements))

    def quantizer(self, bits: int, unsigned: Optional[bool] = None
                  ) -> "UniformActivationQuantizer":
        if not self.observed:
            raise RuntimeError("cannot freeze an unobserved merge")
        if unsigned is None:
            unsigned = self.minimum >= 0.0
        maximum = self.maximum if unsigned else max(abs(self.minimum), abs(self.maximum))
        return UniformActivationQuantizer(bits, maximum, unsigned=unsigned)


class UniformActivationQuantizer(object):
    def __init__(self, bits: int, maximum: float, unsigned: bool) -> None:
        bits = int(bits)
        if bits < 1:
            raise ValueError("activation bits must be positive")
        self.bits = bits
        self.unsigned = bool(unsigned)
        if self.unsigned:
            self.qmin = 0
            self.qmax = 2 ** bits - 1
        else:
            if bits < 2:
                raise ValueError("signed activation bits must be at least 2")
            self.qmax = 2 ** (bits - 1) - 1
            self.qmin = -self.qmax
        self.maximum = max(float(maximum), 0.0)
        self.scale = self.maximum / float(self.qmax) if self.maximum > 0 else 1.0
        self.zero_point = 0

    def quantize_with_codes(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        codes = torch.round(tensor / self.scale).clamp(self.qmin, self.qmax)
        return codes * self.scale, codes.to(torch.int32)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.quantize_with_codes(tensor)[0]

    def qparams(self) -> Dict[str, Any]:
        return {
            "bits": self.bits,
            "unsigned": self.unsigned,
            "qmin": self.qmin,
            "qmax": self.qmax,
            "scale": self.scale,
            "zero_point": self.zero_point,
        }


class GroupwiseMinMaxObserver(object):
    def __init__(self, axis: int, group_size: int) -> None:
        if int(group_size) <= 0:
            raise ValueError("group_size must be positive")
        self.axis = int(axis)
        self.group_size = int(group_size)
        self.observers = []  # type: List[TensorMinMaxObserver]
        self.channels = None  # type: Optional[int]

    @property
    def observed(self) -> bool:
        return bool(self.observers) and all(item.observed for item in self.observers)

    def _slices(self, tensor: torch.Tensor) -> Iterable[torch.Tensor]:
        axis = self.axis if self.axis >= 0 else tensor.ndim + self.axis
        if axis < 0 or axis >= tensor.ndim:
            raise ValueError("group axis is outside tensor rank")
        channels = int(tensor.shape[axis])
        if self.channels is None:
            self.channels = channels
            count = int(math.ceil(channels / float(self.group_size)))
            self.observers = [TensorMinMaxObserver() for _ in range(count)]
        elif channels != self.channels:
            raise ValueError("grouped merge channel count changed across calibration")
        for start in range(0, channels, self.group_size):
            index = [slice(None)] * tensor.ndim
            index[axis] = slice(start, min(start + self.group_size, channels))
            yield tensor[tuple(index)]

    def update(self, tensor: torch.Tensor) -> None:
        for observer, group in zip(self.observers or self._initialize(tensor),
                                   self._slices(tensor)):
            observer.update(group)

    def _initialize(self, tensor: torch.Tensor) -> List[TensorMinMaxObserver]:
        list(self._slices(tensor))
        return self.observers

    def quantizer(self, bits: int) -> "GroupwiseActivationQuantizer":
        if not self.observed:
            raise RuntimeError("cannot freeze an unobserved grouped merge")
        return GroupwiseActivationQuantizer(
            [observer.quantizer(bits) for observer in self.observers],
            axis=self.axis, group_size=self.group_size, channels=int(self.channels))


class GroupwiseActivationQuantizer(object):
    def __init__(self, quantizers: Sequence[UniformActivationQuantizer],
                 axis: int, group_size: int, channels: int) -> None:
        self.quantizers = tuple(quantizers)
        self.axis = int(axis)
        self.group_size = int(group_size)
        self.channels = int(channels)
        self.bits = self.quantizers[0].bits if self.quantizers else 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        axis = self.axis if self.axis >= 0 else tensor.ndim + self.axis
        if int(tensor.shape[axis]) != self.channels:
            raise ValueError("grouped merge channel count changed at inference")
        groups = []
        for group_index, start in enumerate(range(0, self.channels, self.group_size)):
            index = [slice(None)] * tensor.ndim
            index[axis] = slice(start, min(start + self.group_size, self.channels))
            groups.append(self.quantizers[group_index](tensor[tuple(index)]))
        return torch.cat(groups, dim=axis)

    def qparams(self) -> Dict[str, Any]:
        return {
            "bits": self.bits,
            "axis": self.axis,
            "group_size": self.group_size,
            "groups": len(self.quantizers),
            "scales": ";".join(str(item.scale) for item in self.quantizers),
            "unsigned_groups": ";".join(
                "1" if item.unsigned else "0" for item in self.quantizers),
        }


class MergeSiteController(object):
    def __init__(self, name: str, operation: str, policy: str = "shared",
                 axis: int = 1, group_size: Optional[int] = None,
                 runtime: Optional[EdgeQDQRuntime] = None,
                 residual_branch_bits: Tuple[int, int] = (4, 8),
                 residual_output_bits: int = 8) -> None:
        if operation not in _VALID_OPERATIONS:
            raise ValueError("unknown merge operation: %s" % operation)
        if policy not in _VALID_POLICIES:
            raise ValueError("unknown merge policy: %s" % policy)
        if policy == "grouped" and (group_size is None or int(group_size) <= 0):
            raise ValueError("grouped merge policy requires group_size")
        if policy == "residual" and operation != "add":
            raise ValueError("residual merge policy requires add")
        self.name = str(name)
        self.operation = operation
        self.policy = policy
        self.axis = int(axis)
        self.group_size = None if group_size is None else int(group_size)
        self.runtime = runtime or EdgeQDQRuntime()
        self.residual_branch_bits = tuple(
            int(bits) for bits in residual_branch_bits)
        self.residual_output_bits = int(residual_output_bits)
        if len(self.residual_branch_bits) != 2 or \
                any(bits < 2 or bits > 8
                    for bits in self.residual_branch_bits):
            raise ValueError("residual branch bits must contain two 2-8 bit values")
        if self.residual_output_bits < 2 or self.residual_output_bits > 8:
            raise ValueError("residual output bits must be between 2 and 8")
        self.bits = None  # type: Optional[int]
        self.shared_observer = TensorMinMaxObserver()
        self.branch_observers = []  # type: List[TensorMinMaxObserver]
        self.group_observer = (GroupwiseMinMaxObserver(axis, int(group_size))
                               if policy == "grouped" else None)
        self.shared_quantizer = None  # type: Optional[UniformActivationQuantizer]
        self.branch_quantizers = []  # type: List[UniformActivationQuantizer]
        self.group_quantizer = None  # type: Optional[GroupwiseActivationQuantizer]
        self.output_observer = TensorMinMaxObserver()
        self.output_quantizer = None  # type: Optional[UniformActivationQuantizer]
        self.branch_count = None  # type: Optional[int]
        self.residual_nonzero = [0, 0]
        self.residual_new_zeros = [0, 0]

    def _check_branches(self, branches: Sequence[torch.Tensor]) -> None:
        if len(branches) < 2:
            raise ValueError("merge requires at least two tensor branches")
        if self.branch_count is None:
            self.branch_count = len(branches)
        elif len(branches) != self.branch_count:
            raise ValueError("merge branch count changed across calls")
        if self.policy == "residual" and len(branches) != 2:
            raise ValueError("residual merge requires update and base branches")

    def observe(self, branches: Sequence[torch.Tensor],
                merged: Optional[torch.Tensor] = None) -> None:
        self._check_branches(branches)
        if self.policy == "shared":
            for branch in branches:
                self.shared_observer.update(branch)
        elif self.policy in ("independent", "residual"):
            if not self.branch_observers:
                self.branch_observers = [TensorMinMaxObserver() for _ in branches]
            for observer, branch in zip(self.branch_observers, branches):
                observer.update(branch)
        elif self.operation == "concat":
            if merged is None:
                merged = torch.cat(tuple(branches), dim=self.axis)
            self.group_observer.update(merged)
        else:
            for branch in branches:
                self.group_observer.update(branch)
        if self.operation == "add":
            if merged is None:
                merged = branches[0]
                for branch in branches[1:]:
                    merged = merged + branch
            self.output_observer.update(merged)

    def freeze(self, bits: int) -> None:
        self.bits = int(bits)
        if self.policy == "shared":
            self.shared_quantizer = self.shared_observer.quantizer(bits)
        elif self.policy == "independent":
            if not self.branch_observers:
                raise RuntimeError("cannot freeze an unobserved merge")
            self.branch_quantizers = [item.quantizer(bits)
                                      for item in self.branch_observers]
        elif self.policy == "residual":
            if len(self.branch_observers) != 2:
                raise RuntimeError("cannot freeze an unobserved residual merge")
            self.branch_quantizers = [
                observer.quantizer(branch_bits)
                for observer, branch_bits in zip(
                    self.branch_observers, self.residual_branch_bits)
            ]
        else:
            self.group_quantizer = self.group_observer.quantizer(bits)
        if self.operation == "add":
            output_bits = self.residual_output_bits \
                if self.policy == "residual" else bits
            self.output_quantizer = self.output_observer.quantizer(output_bits)

    def quantize_branches(self, branches: Sequence[torch.Tensor]
                          ) -> Tuple[torch.Tensor, ...]:
        self._check_branches(branches)
        if self.policy == "grouped" and self.operation == "concat":
            raise RuntimeError("grouped concat quantizes the merged output")
        output = []
        for index, branch in enumerate(branches):
            if self.policy == "shared":
                quantizer = self.shared_quantizer
            elif self.policy in ("independent", "residual"):
                quantizer = self.branch_quantizers[index]
            else:
                quantizer = self.group_quantizer
            if quantizer is None:
                raise RuntimeError("merge quantizer is not frozen")
            output.append(self.runtime.process(
                "%s:branch#%d" % (self.name, index), branch, quantizer,
                force=True))
        return tuple(output)

    def quantize_output(self, merged: torch.Tensor) -> torch.Tensor:
        if self.operation == "add":
            quantizer = self.output_quantizer
        elif self.policy == "grouped" and self.operation == "concat":
            quantizer = self.group_quantizer
        else:
            raise RuntimeError("this merge policy does not requantize its output")
        if quantizer is None:
            raise RuntimeError("merge output quantizer is not frozen")
        return self.runtime.process(
            "%s:output" % self.name, merged, quantizer, force=True)

    def merge(self, branches: Sequence[torch.Tensor]) -> torch.Tensor:
        if self.policy == "residual":
            self._check_branches(branches)
            if self.output_quantizer is None or not self.branch_quantizers:
                raise RuntimeError("residual merge quantizers are not frozen")
            branch_codes = []
            for index, (branch, quantizer) in enumerate(zip(
                    branches, self.branch_quantizers)):
                _, codes = self.runtime.process_with_codes(
                    "%s:branch#%d" % (self.name, index), branch,
                    quantizer.quantize_with_codes, force=True)
                branch_codes.append((codes, quantizer.scale))
                nonzero = branch != 0
                self.residual_nonzero[index] += int(nonzero.sum().item())
                self.residual_new_zeros[index] += int(
                    (nonzero & (codes == 0)).sum().item())
            output_codes = add_requantized_int32(
                tuple(branch_codes), self.output_quantizer.scale,
                self.output_quantizer.qmin, self.output_quantizer.qmax)
            output = output_codes.to(branches[0].dtype) * \
                float(self.output_quantizer.scale)
            return self.runtime.mark_quantized(
                "%s:output" % self.name, output)
        if self.operation == "concat" and self.policy == "grouped":
            merged = torch.cat(tuple(branches), dim=self.axis)
            return self.quantize_output(merged)
        quantized = self.quantize_branches(branches)
        if self.operation == "concat":
            result = torch.cat(quantized, dim=self.axis)
            return self.runtime.mark_quantized("%s:output" % self.name, result)
        result = quantized[0]
        for branch in quantized[1:]:
            result = result + branch
        return self.quantize_output(result)

    def qparams(self) -> Dict[str, Any]:
        row = {
            "merge": self.name,
            "operation": self.operation,
            "policy": self.policy,
            "bits": self.bits,
            "axis": self.axis,
            "group_size": "" if self.group_size is None else self.group_size,
            "branches": self.branch_count,
            "unsigned": "",
            "qmin": "",
            "qmax": "",
            "scale": "",
            "zero_point": "",
        }
        if self.policy == "shared" and self.shared_quantizer is not None:
            row.update(self.shared_quantizer.qparams())
        elif self.policy in ("independent", "residual") and \
                self.branch_quantizers:
            row["scales"] = ";".join(str(item.scale)
                                      for item in self.branch_quantizers)
            row["unsigned_branches"] = ";".join(
                "1" if item.unsigned else "0" for item in self.branch_quantizers)
            row["branch_bits"] = ";".join(
                str(item.bits) for item in self.branch_quantizers)
        elif self.group_quantizer is not None:
            row.update(self.group_quantizer.qparams())
        if self.output_quantizer is not None:
            row["output_scale"] = self.output_quantizer.scale
            row["output_unsigned"] = self.output_quantizer.unsigned
            row["output_bits"] = self.output_quantizer.bits
        if self.policy == "residual" and len(self.branch_observers) == 2:
            update = self.branch_observers[0]
            base = self.branch_observers[1]
            row["base_to_update_rms_ratio"] = \
                base.rms / update.rms if update.rms > 0.0 else float("inf")
            row["update_to_base_energy_ratio"] = \
                update.signal_energy / base.signal_energy \
                if base.signal_energy > 0.0 else float("inf")
            zero_rates = []
            for nonzero, new_zeros in zip(
                    self.residual_nonzero, self.residual_new_zeros):
                zero_rates.append(
                    float(new_zeros) / float(nonzero) if nonzero else 0.0)
            row["branch_new_zero_rates"] = ";".join(
                str(value) for value in zero_rates)
        return row
