"""Integer Attention reference controller for official CompletionFormer."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence

import torch

from spn_quant.integer_ops import (
    batched_int8_mm_int32,
    batched_uint8_int8_mm_int32,
    quantize_unsigned,
)
from spn_quant.scale_search import CalibrationCache, CoordinateScaleSearch


@dataclass(frozen=True)
class AttentionIntegerResult:
    context: torch.Tensor
    q_codes: torch.Tensor
    k_codes: torch.Tensor
    v_codes: torch.Tensor
    score_accumulator: torch.Tensor
    probability_codes: torch.Tensor
    context_accumulator: torch.Tensor
    probability: torch.Tensor
    probability_scale: float


class _ErrorAccumulator(object):
    def __init__(self) -> None:
        self.signal_sq = 0.0
        self.error_sq = 0.0
        self.elements = 0

    def update(self, reference: torch.Tensor, candidate: torch.Tensor) -> None:
        difference = reference.detach().double() - candidate.detach().double()
        self.signal_sq += float((reference.detach().double() ** 2).sum().item())
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
            raise RuntimeError("attention statistics have no observations")
        return self.error_sq / float(self.elements)


class IntegerAttentionController(object):
    def __init__(self, name: str, num_heads: int, head_dim: int,
                 head_scale: float, qkv_bits: int, probability_bits: int,
                 clip_factors: Sequence[float], search_rounds: int,
                 cache_sample_limit: int, cache_byte_limit: int) -> None:
        self.name = str(name)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.head_scale = float(head_scale)
        self.qkv_bits = int(qkv_bits)
        self.probability_bits = int(probability_bits)
        self.clip_factors = tuple(float(value) for value in clip_factors)
        self.search_rounds = int(search_rounds)
        if not self.name:
            raise ValueError("attention controller name must be nonempty")
        if self.num_heads <= 0 or self.head_dim <= 0:
            raise ValueError("attention head shape must be positive")
        if not math.isfinite(self.head_scale) or self.head_scale <= 0.0:
            raise ValueError("attention head scale must be finite and positive")
        if self.qkv_bits < 2 or self.qkv_bits > 8:
            raise ValueError("QKV bits must be in [2, 8]")
        if self.probability_bits != 8:
            raise ValueError("integer Attention requires unsigned A8 probability")

        self.cache = CalibrationCache(
            sample_limit=cache_sample_limit, byte_limit=cache_byte_limit)
        self.search = CoordinateScaleSearch(
            parameter_names=("q", "k", "v"),
            factors=self.clip_factors, rounds=self.search_rounds)
        self.phase = "observe"
        self.observations = 0
        self.cached_samples = 0
        self.device = None
        self.maxima = dict((name, None) for name in ("q", "k", "v"))
        self.scales = {}  # type: Dict[str, torch.Tensor]
        self.selected_factors = {}  # type: Dict[str, float]
        self._search_rows = []  # type: List[Dict[str, object]]
        self._q_stats = _ErrorAccumulator()
        self._k_stats = _ErrorAccumulator()
        self._v_stats = _ErrorAccumulator()
        self._score_stats = _ErrorAccumulator()
        self._context_stats = _ErrorAccumulator()
        self._probability_kl_sum = 0.0
        self._probability_elements = 0
        self._probability_zeros = 0
        self._probability_saturated = 0
        self._quantized_updates = 0

    def _validate(self, q: torch.Tensor, k: torch.Tensor,
                  v: torch.Tensor, context: torch.Tensor = None) -> None:
        for name, tensor in (("q", q), ("k", k), ("v", v)):
            if not torch.is_tensor(tensor) or tensor.ndim != 4:
                raise ValueError("%s must have B,H,N,D shape" % name)
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError("%s must be finite" % name)
            if tensor.shape[1] != self.num_heads or \
                    tensor.shape[3] != self.head_dim:
                raise ValueError("%s head shape does not match controller" % name)
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError("QKV batch dimensions must match")
        if k.shape[2] != v.shape[2]:
            raise ValueError("K and V token dimensions must match")
        if q.device != k.device or q.device != v.device:
            raise ValueError("QKV devices must match")
        if context is not None:
            if context.shape != q.shape:
                raise ValueError("attention context shape must match Q")
            if not bool(torch.isfinite(context).all().item()):
                raise ValueError("attention context must be finite")

    @staticmethod
    def _head_maximum(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().abs().amax(dim=(0, 2, 3)).to(
            device="cpu", dtype=torch.float64)

    def observe(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                fp_context: torch.Tensor) -> None:
        if self.phase != "observe":
            raise RuntimeError("attention observation phase is closed")
        self._validate(q, k, v, fp_context)
        if self.device is None:
            self.device = q.device
        elif self.device != q.device:
            raise ValueError("attention calibration device changed")
        for role, tensor in (("q", q), ("k", k), ("v", v)):
            maximum = self._head_maximum(tensor)
            previous = self.maxima[role]
            self.maxima[role] = maximum if previous is None else \
                torch.maximum(previous, maximum)
        if self.cached_samples < self.cache.sample_limit:
            self.cache.append((q, k, v, fp_context))
            self.cached_samples += 1
        self.observations += 1

    def _role_codes(self, tensor: torch.Tensor, role: str,
                    factor: float) -> tuple[torch.Tensor, torch.Tensor,
                                            torch.Tensor]:
        maximum = self.maxima[role].to(device=tensor.device) * float(factor)
        qmax = (1 << (self.qkv_bits - 1)) - 1
        scale = torch.where(
            maximum > 0.0, maximum / float(qmax), torch.ones_like(maximum))
        scale_view = scale.reshape(1, self.num_heads, 1, 1)
        codes = torch.round(
            tensor.to(torch.float64) / scale_view).clamp(
                -qmax, qmax).to(torch.int8)
        reconstructed = codes.to(tensor.dtype) * scale_view.to(tensor.dtype)
        return codes, reconstructed, scale

    def _execute(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 factors: Dict[str, float]) -> AttentionIntegerResult:
        self._validate(q, k, v)
        q_codes, _, q_scale = self._role_codes(q, "q", factors["q"])
        k_codes, _, k_scale = self._role_codes(k, "k", factors["k"])
        v_codes, _, v_scale = self._role_codes(v, "v", factors["v"])
        score_accumulator = batched_int8_mm_int32(
            q_codes, k_codes.transpose(-2, -1).contiguous())
        score_scale = (q_scale * k_scale * self.head_scale).reshape(
            1, self.num_heads, 1, 1)
        score = score_accumulator.to(torch.float32) * score_scale.to(
            device=q.device, dtype=torch.float32)
        score_fp16 = score.to(torch.float16)
        if not bool(torch.isfinite(score_fp16).all().item()):
            raise ValueError("quantized attention score exceeds FP16")
        probability = torch.softmax(score_fp16, dim=-1)
        if not bool(torch.isfinite(probability).all().item()):
            raise ValueError("quantized attention probability is nonfinite")
        probability_codes, probability_scale = quantize_unsigned(
            probability, bits=self.probability_bits, maximum=1.0)
        context_accumulator = batched_uint8_int8_mm_int32(
            probability_codes, v_codes)
        context_scale = (v_scale * probability_scale).reshape(
            1, self.num_heads, 1, 1)
        context = context_accumulator.to(torch.float32) * context_scale.to(
            device=q.device, dtype=torch.float32)
        return AttentionIntegerResult(
            context=context,
            q_codes=q_codes,
            k_codes=k_codes,
            v_codes=v_codes,
            score_accumulator=score_accumulator,
            probability_codes=probability_codes,
            context_accumulator=context_accumulator,
            probability=probability,
            probability_scale=probability_scale)

    def _objective(self, factors: Dict[str, float]) -> float:
        error = 0.0
        signal = 0.0
        for q_cpu, k_cpu, v_cpu, target_cpu in self.cache.samples():
            q = q_cpu.to(self.device)
            k = k_cpu.to(self.device)
            v = v_cpu.to(self.device)
            target = target_cpu.to(self.device)
            candidate = self._execute(q, k, v, factors).context
            error += float(((candidate.double() - target.double()) ** 2).sum().item())
            signal += float((target.double() ** 2).sum().item())
        denominator = max(signal, torch.finfo(torch.float64).tiny)
        return error / denominator

    def _select_scales(self) -> None:
        result = self.search.run(
            {"q": 1.0, "k": 1.0, "v": 1.0},
            self._objective, sample_count=self.cached_samples)
        self.selected_factors = dict(result.values)
        qmax = (1 << (self.qkv_bits - 1)) - 1
        self.scales = dict(
            (role, torch.where(
                self.maxima[role] > 0.0,
                self.maxima[role] * self.selected_factors[role] / float(qmax),
                torch.ones_like(self.maxima[role])))
            for role in ("q", "k", "v"))
        self._search_rows = []
        for source in result.rows:
            row = dict(source)
            row["module"] = self.name
            row["qkv_bits"] = self.qkv_bits
            self._search_rows.append(row)
        self.phase = "frozen"

    def freeze(self) -> None:
        if self.phase != "observe":
            raise RuntimeError("attention controller is not observing")
        if self.observations == 0 or self.cached_samples == 0:
            raise RuntimeError("attention controller has no observations")
        self._select_scales()

    def reconfigure(self, qkv_bits: int) -> None:
        if self.phase == "observe":
            raise RuntimeError("attention controller must be frozen first")
        qkv_bits = int(qkv_bits)
        if qkv_bits < 2 or qkv_bits > 8:
            raise ValueError("QKV bits must be in [2, 8]")
        self.qkv_bits = qkv_bits
        self._select_scales()

    def enable(self) -> None:
        if self.phase not in ("frozen", "disabled"):
            raise RuntimeError("attention controller must be frozen before enabling")
        self.phase = "quantize"

    def disable(self) -> None:
        if self.phase == "observe":
            raise RuntimeError("attention controller must be frozen before disabling")
        self.phase = "disabled"

    def execute(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor) -> AttentionIntegerResult:
        if self.phase != "quantize":
            raise RuntimeError("attention quantization is not enabled")
        return self._execute(q, k, v, self.selected_factors)

    def quantize(self, q: torch.Tensor, k: torch.Tensor,
                 v: torch.Tensor) -> torch.Tensor:
        result = self.execute(q, k, v)
        q_scale = self.scales["q"].to(q.device).reshape(
            1, self.num_heads, 1, 1)
        k_scale = self.scales["k"].to(k.device).reshape(
            1, self.num_heads, 1, 1)
        v_scale = self.scales["v"].to(v.device).reshape(
            1, self.num_heads, 1, 1)
        q_reconstructed = result.q_codes.to(q.dtype) * q_scale.to(q.dtype)
        k_reconstructed = result.k_codes.to(k.dtype) * k_scale.to(k.dtype)
        v_reconstructed = result.v_codes.to(v.dtype) * v_scale.to(v.dtype)
        reference_score = torch.matmul(q, k.transpose(-2, -1)) * self.head_scale
        candidate_score = result.score_accumulator.to(torch.float32) * (
            q_scale * k_scale * self.head_scale).to(torch.float32)
        reference_probability = torch.softmax(reference_score, dim=-1)
        probability = result.probability.to(reference_probability.dtype)
        tiny = torch.finfo(reference_probability.dtype).tiny
        kl = reference_probability * (
            torch.log(reference_probability.clamp_min(tiny)) -
            torch.log(probability.clamp_min(tiny)))
        reference_context = torch.matmul(reference_probability, v)

        self._q_stats.update(q, q_reconstructed)
        self._k_stats.update(k, k_reconstructed)
        self._v_stats.update(v, v_reconstructed)
        self._score_stats.update(reference_score, candidate_score)
        self._context_stats.update(reference_context, result.context)
        self._probability_kl_sum += float(kl.sum().item())
        self._probability_elements += int(kl.numel())
        self._probability_zeros += int(
            (result.probability_codes == 0).sum().item())
        self._probability_saturated += int(
            (result.probability_codes == 255).sum().item())
        self._quantized_updates += 1
        return result.context

    def manifest(self) -> List[Dict[str, object]]:
        if not self.scales:
            raise RuntimeError("attention controller is not frozen")
        rows = []
        for head in range(self.num_heads):
            rows.append({
                "module": self.name,
                "head": head,
                "qkv_bits": self.qkv_bits,
                "probability_bits": self.probability_bits,
                "q_scale": float(self.scales["q"][head].item()),
                "k_scale": float(self.scales["k"][head].item()),
                "v_scale": float(self.scales["v"][head].item()),
                "probability_scale": 1.0 / 255.0,
                "q_factor": self.selected_factors["q"],
                "k_factor": self.selected_factors["k"],
                "v_factor": self.selected_factors["v"],
                "calibration_updates": self.observations,
                "cached_samples": self.cached_samples,
            })
        return rows

    def statistics(self) -> List[Dict[str, object]]:
        if self._probability_elements == 0:
            raise RuntimeError("attention controller has no quantized observations")
        return [{
            "module": self.name,
            "q_sqnr_db": self._q_stats.sqnr_db,
            "k_sqnr_db": self._k_stats.sqnr_db,
            "v_sqnr_db": self._v_stats.sqnr_db,
            "score_sqnr_db": self._score_stats.sqnr_db,
            "context_mse": self._context_stats.mse,
            "probability_kl": self._probability_kl_sum /
                float(self._probability_elements),
            "probability_zero_ratio": self._probability_zeros /
                float(self._probability_elements),
            "probability_saturation_ratio": self._probability_saturated /
                float(self._probability_elements),
            "updates": self._quantized_updates,
        }]

    def reset_statistics(self) -> None:
        self._q_stats = _ErrorAccumulator()
        self._k_stats = _ErrorAccumulator()
        self._v_stats = _ErrorAccumulator()
        self._score_stats = _ErrorAccumulator()
        self._context_stats = _ErrorAccumulator()
        self._probability_kl_sum = 0.0
        self._probability_elements = 0
        self._probability_zeros = 0
        self._probability_saturated = 0
        self._quantized_updates = 0

    def search_rows(self) -> List[Dict[str, object]]:
        return [dict(row) for row in self._search_rows]
