"""Calibration and diagnostics for propagation-domain quantization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

import torch

from spn_quant.propagation.fixed_point import (
    Q13_ONE,
    direct_signed_codes_q13,
    normalize_signed_codes_q13,
    softmax_codes_q13,
    symmetric_qdq,
    unsigned_unit_qdq,
)


FLOAT_STATE_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


@dataclass(frozen=True)
class PropagationQuantConfig:
    affinity_bits: int = 4
    confidence_bits: int = 8
    offset_bits: int = 4
    state_bits: int = 4
    coefficient_fraction_bits: int = 13

    def __post_init__(self) -> None:
        for name in ("affinity_bits", "confidence_bits", "offset_bits",
                     "state_bits"):
            bits = int(getattr(self, name))
            if bits < 2 or bits > 16:
                raise ValueError("%s must be between 2 and 16" % name)
        if int(self.coefficient_fraction_bits) != 13:
            raise ValueError("INT16 propagation coefficients require Q13")


class PropagationQuantController(object):
    def __init__(self) -> None:
        self.mode = "bypass"
        self.frozen = False
        self.config = None  # type: Optional[PropagationQuantConfig]
        self.float_state_dtype = None  # type: Optional[torch.dtype]
        self.maximum = {}  # type: Dict[str, float]
        self._statistics = []  # type: List[Dict[str, float]]
        self.statistics_enabled = True

    def set_runtime_statistics(self, enabled: bool) -> None:
        self.statistics_enabled = bool(enabled)

    def observe(self) -> None:
        self.mode = "observe"
        self.frozen = False
        self.config = None
        self.float_state_dtype = None
        self.maximum = {}
        self._statistics = []

    def observe_signal(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if self.mode != "observe":
            raise RuntimeError("propagation controller is not observing")
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            raise ValueError("observed propagation signal must be nonempty")
        finite = tensor.detach()[torch.isfinite(tensor.detach())]
        if finite.numel() != tensor.numel():
            raise ValueError("propagation calibration signal must be finite")
        maximum = float(finite.abs().max().item())
        self.maximum[str(name)] = max(self.maximum.get(str(name), 0.0), maximum)
        return tensor

    def freeze(self) -> None:
        if not self.maximum:
            raise RuntimeError("no propagation signals were observed")
        self.frozen = True
        self.mode = "bypass"

    def configure(self, config: PropagationQuantConfig) -> None:
        if not self.frozen:
            raise RuntimeError("propagation calibration must be frozen")
        if not isinstance(config, PropagationQuantConfig):
            raise TypeError("config must be PropagationQuantConfig")
        self.config = config
        self.float_state_dtype = None
        self._statistics = []
        self.mode = "quantize"

    def configure_float(self, state_dtype: str) -> None:
        if not self.frozen:
            raise RuntimeError("propagation calibration must be frozen")
        self.float_state_dtype = FLOAT_STATE_DTYPES[state_dtype]
        self.config = None
        self._statistics = []
        self.mode = "float"

    def configure_fp16(self) -> None:
        self.configure_float("fp16")

    def load_float_state_dict(self, state) -> None:
        if set(state) != {"mode"} or state["mode"] not in FLOAT_STATE_DTYPES:
            raise ValueError("propagation floating-point state is invalid")
        self.maximum = {}
        self.frozen = True
        self.configure_float(str(state["mode"]))

    def disable(self) -> None:
        self.mode = "bypass"
        self.config = None
        self.float_state_dtype = None

    def capture(self) -> None:
        self.mode = "capture"
        self.config = None
        self.float_state_dtype = None

    def begin_forward(self) -> None:
        self._statistics = []

    def _require_quantize(self) -> PropagationQuantConfig:
        if self.mode != "quantize" or self.config is None:
            raise RuntimeError("propagation quantization is not configured")
        return self.config

    def _symmetric(self, signal: str, tensor: torch.Tensor, bits: int,
                   iteration: Optional[int] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        maximum = self.maximum.get(signal)
        if maximum is None:
            raise RuntimeError("missing calibration for %s" % signal)
        quantized, codes, scale = symmetric_qdq(tensor, bits, maximum)
        qmax = (1 << (int(bits) - 1)) - 1
        self._record_qdq(signal, tensor, quantized, codes, qmax, iteration)
        return quantized, codes, scale

    def _record_qdq(self, signal: str, reference: torch.Tensor,
                    quantized: torch.Tensor, codes: torch.Tensor, qmax: int,
                    iteration: Optional[int]) -> None:
        if not self.statistics_enabled:
            return
        difference = quantized - reference
        finite = torch.isfinite(quantized)
        row = {
            "signal": str(signal),
            "iteration": -1 if iteration is None else int(iteration),
            "numel": int(reference.numel()),
            "mse": float(torch.mean(difference.float().square()).item()),
            "zeroed_rate": float(torch.mean(
                ((reference != 0) & (codes == 0)).float()).item()),
            "saturation_rate": float(torch.mean(
                (codes.abs() == int(qmax)).float()).item()),
            "nonfinite_ratio": float(torch.mean((~finite).float()).item()),
        }
        self._statistics.append(row)

    def record_float_state(self, reference: torch.Tensor,
                           stored: torch.Tensor, iteration: int) -> None:
        if self.mode != "float" or self.float_state_dtype is None:
            raise RuntimeError("floating-point propagation is not configured")
        if not self.statistics_enabled:
            return
        restored = stored.float()
        difference = restored - reference.float()
        finite = torch.isfinite(restored)
        self._statistics.append({
            "signal": "state",
            "iteration": int(iteration),
            "numel": int(reference.numel()),
            "mse": float(torch.mean(difference.square()).item()),
            "zeroed_rate": float(torch.mean(
                ((reference != 0) & (restored == 0)).float()).item()),
            "saturation_rate": 0.0,
            "nonfinite_ratio": float(torch.mean((~finite).float()).item()),
        })

    def record_float_signed_constraints(
            self, center: torch.Tensor, neighbor: torch.Tensor,
            dim: int) -> None:
        if self.mode != "float":
            raise RuntimeError("floating-point propagation is not configured")
        if not self.statistics_enabled:
            return
        coefficient_sum = center.float() + neighbor.float().sum(
            dim=int(dim), keepdim=True)
        sum_error = (coefficient_sum - 1.0).abs()
        tolerance = torch.finfo(torch.float32).eps * neighbor.shape[int(dim)]
        contraction = neighbor.float().abs().sum(
            dim=int(dim), keepdim=True) > 1.0 + tolerance
        self._statistics.append({
            "signal": "affinity_constraints",
            "iteration": 0,
            "numel": int(neighbor.numel()),
            "coefficient_sum_max_error": float(sum_error.max().item()),
            "contraction_violation_rate": float(
                contraction.float().mean().item()),
        })

    def record_float_softmax_constraints(self, affinity: torch.Tensor,
                                         dim: int) -> None:
        if self.mode != "float":
            raise RuntimeError("floating-point propagation is not configured")
        if not self.statistics_enabled:
            return
        coefficient_sum = affinity.float().sum(
            dim=int(dim), keepdim=True)
        sum_error = (coefficient_sum - 1.0).abs()
        tolerance = torch.finfo(torch.float32).eps * affinity.shape[int(dim)]
        contraction = affinity.float().abs().sum(
            dim=int(dim), keepdim=True) > 1.0 + tolerance
        self._statistics.append({
            "signal": "affinity_constraints",
            "iteration": 0,
            "numel": int(affinity.numel()),
            "coefficient_sum_max_error": float(sum_error.max().item()),
            "contraction_violation_rate": float(
                contraction.float().mean().item()),
        })

    def signed_affinity(self, tensor: torch.Tensor, denominator_floor: bool,
                        eps: float = 1e-4
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        config = self._require_quantize()
        _, raw_codes, scale = self._symmetric(
            "affinity_raw", tensor, config.affinity_bits)
        values, center, neighbor = normalize_signed_codes_q13(
            raw_codes, scale, denominator_floor, eps)
        self._record_constraints(center, neighbor)
        return values.to(tensor.dtype), center, neighbor

    def direct_signed_affinity(self, tensor: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor,
                                          torch.Tensor]:
        config = self._require_quantize()
        _, raw_codes, scale = self._symmetric(
            "affinity_raw", tensor, config.affinity_bits)
        values, center, neighbor = direct_signed_codes_q13(raw_codes, scale)
        self._record_constraints(center, neighbor)
        return values.to(tensor.dtype), center, neighbor

    def softmax_affinity(self, tensor: torch.Tensor, dim: int
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        config = self._require_quantize()
        _, raw_codes, scale = self._symmetric(
            "affinity_raw", tensor, config.affinity_bits)
        values, codes = softmax_codes_q13(raw_codes, scale, dim=dim)
        sum_error = (codes.to(torch.int32).sum(
            dim=dim, keepdim=True, dtype=torch.int32) - Q13_ONE).abs()
        self._statistics.append({
            "signal": "affinity_constraints",
            "iteration": 0,
            "numel": int(codes.numel()),
            "coefficient_sum_max_error": float(sum_error.max().item()) /
            float(Q13_ONE),
            "contraction_violation_rate": 0.0,
        })
        return values.to(tensor.dtype), codes

    def _record_constraints(self, center: torch.Tensor,
                            neighbor: torch.Tensor) -> None:
        if not self.statistics_enabled:
            return
        sum_error = (center.to(torch.int32) + neighbor.to(torch.int32).sum(
            dim=1, keepdim=True) - Q13_ONE).abs()
        contraction = neighbor.to(torch.int32).abs().sum(
            dim=1, keepdim=True) > Q13_ONE
        self._statistics.append({
            "signal": "affinity_constraints",
            "iteration": 0,
            "numel": int(neighbor.numel()),
            "coefficient_sum_max_error": float(sum_error.max().item()) /
            float(Q13_ONE),
            "contraction_violation_rate": float(
                contraction.float().mean().item()),
        })

    def quantize_confidence(self, tensor: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        config = self._require_quantize()
        quantized, codes = unsigned_unit_qdq(
            tensor, bits=config.confidence_bits)
        self._record_qdq(
            "confidence", tensor, quantized, codes,
            (1 << config.confidence_bits) - 1, None)
        return quantized, codes

    def quantize_offset(self, tensor: torch.Tensor) -> torch.Tensor:
        config = self._require_quantize()
        return self._symmetric(
            "offset", tensor, config.offset_bits)[0]

    def quantize_state(self, tensor: torch.Tensor, iteration: int
                       ) -> torch.Tensor:
        config = self._require_quantize()
        return self._symmetric(
            "state", tensor, config.state_bits, int(iteration))[0]

    def quantize_state_with_codes(
            self, tensor: torch.Tensor, iteration: int
            ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        config = self._require_quantize()
        return self._symmetric(
            "state", tensor, config.state_bits, int(iteration))

    def state_from_codes(self, reference: torch.Tensor,
                         codes: torch.Tensor, scale: float,
                         iteration: int) -> torch.Tensor:
        config = self._require_quantize()
        qmax = (1 << (config.state_bits - 1)) - 1
        if codes.dtype != torch.int32:
            raise TypeError("propagation state codes must be INT32")
        if bool((codes.abs() > qmax).any().item()):
            raise ValueError("propagation state code exceeds configured bits")
        quantized = codes.to(reference.dtype) * float(scale)
        self._record_qdq(
            "state", reference, quantized, codes, qmax, int(iteration))
        return quantized

    def statistics(self) -> List[Dict[str, float]]:
        return [dict(row) for row in self._statistics]

    def quantization_state_dict(self):
        config = self._require_quantize()
        if not self.frozen:
            raise RuntimeError("propagation quantization state is not frozen")
        maxima = tuple(sorted(
            (str(name), float(value)) for name, value in self.maximum.items()))
        if not maxima or any(
                not name or not math.isfinite(value) or value < 0.0
                for name, value in maxima):
            raise ValueError("propagation quantization maxima are invalid")
        return {
            "maximum": maxima,
            "config": {
                "affinity_bits": config.affinity_bits,
                "confidence_bits": config.confidence_bits,
                "offset_bits": config.offset_bits,
                "state_bits": config.state_bits,
                "coefficient_fraction_bits":
                    config.coefficient_fraction_bits,
            },
            "frozen": True,
        }

    def load_quantization_state_dict(self, state) -> None:
        if set(state) != {"maximum", "config", "frozen"}:
            raise ValueError("propagation quantization state fields changed")
        if state["frozen"] is not True:
            raise ValueError("propagation quantization state must be frozen")
        config_fields = {
            "affinity_bits", "confidence_bits", "offset_bits", "state_bits",
            "coefficient_fraction_bits",
        }
        if set(state["config"]) != config_fields:
            raise ValueError("propagation quantization config fields changed")
        maxima = tuple(
            (str(row[0]), float(row[1])) for row in state["maximum"])
        if not maxima or tuple(sorted(maxima)) != maxima or len(maxima) != len(
                set(name for name, value in maxima)) or any(
                    not name or not math.isfinite(value) or value < 0.0
                    for name, value in maxima):
            raise ValueError("propagation quantization maxima are invalid")
        config = PropagationQuantConfig(**dict(
            (name, int(state["config"][name])) for name in config_fields))
        self.maximum = dict(maxima)
        self.frozen = True
        self.configure(config)
