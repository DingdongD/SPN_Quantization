"""Calibration and diagnostics for propagation-domain quantization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

import torch

from spn_quant.propagation.fixed_point import (
    Q13_ONE,
    normalize_signed_codes_q13,
    symmetric_qdq,
    unsigned_unit_qdq,
)


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
        self.maximum = {}  # type: Dict[str, float]
        self._statistics = []  # type: List[Dict[str, float]]

    def observe(self) -> None:
        self.mode = "observe"
        self.frozen = False
        self.config = None
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
        self._statistics = []
        self.mode = "quantize"

    def disable(self) -> None:
        self.mode = "bypass"
        self.config = None

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

    def signed_affinity(self, tensor: torch.Tensor, denominator_floor: bool,
                        eps: float = 1e-4
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        config = self._require_quantize()
        _, raw_codes, scale = self._symmetric(
            "affinity_raw", tensor, config.affinity_bits)
        values, center, neighbor = normalize_signed_codes_q13(
            raw_codes, scale, denominator_floor, eps)
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
        return values.to(tensor.dtype), center, neighbor

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

    def statistics(self) -> List[Dict[str, float]]:
        return [dict(row) for row in self._statistics]
