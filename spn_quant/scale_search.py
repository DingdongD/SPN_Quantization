"""Deterministic calibration storage and coordinate scale search."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, List, Sequence, Tuple

import torch


class CalibrationCache(object):
    def __init__(self, sample_limit: int, byte_limit: int) -> None:
        self.sample_limit = int(sample_limit)
        self.byte_limit = int(byte_limit)
        if self.sample_limit <= 0:
            raise ValueError("calibration cache sample limit must be positive")
        if self.byte_limit <= 0:
            raise ValueError("calibration cache byte limit must be positive")
        self._samples = []  # type: List[Tuple[torch.Tensor, ...]]
        self._bytes = 0

    @property
    def stored_bytes(self) -> int:
        return self._bytes

    def append(self, tensors: Tuple[torch.Tensor, ...]) -> None:
        if not isinstance(tensors, tuple) or not tensors:
            raise TypeError("calibration cache expects a nonempty tensor tuple")
        if len(self._samples) >= self.sample_limit:
            raise RuntimeError("calibration cache sample limit exceeded")

        stored = []
        for tensor in tensors:
            if not torch.is_tensor(tensor):
                raise TypeError("calibration cache entries must be tensors")
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError("calibration cache tensor must be finite")
            stored.append(tensor.detach().to(
                device="cpu", dtype=torch.float32).contiguous().clone())
        entry = tuple(stored)
        entry_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in entry)
        if self._bytes + entry_bytes > self.byte_limit:
            raise RuntimeError("calibration cache byte limit exceeded")
        self._samples.append(entry)
        self._bytes += entry_bytes

    def samples(self) -> Tuple[Tuple[torch.Tensor, ...], ...]:
        return tuple(self._samples)


@dataclass(frozen=True)
class ScaleSearchResult:
    values: Dict[str, float]
    objective: float
    rows: List[Dict[str, object]]


class CoordinateScaleSearch(object):
    def __init__(self, parameter_names: Sequence[str],
                 factors: Sequence[float], rounds: int) -> None:
        self.parameter_names = tuple(str(name) for name in parameter_names)
        self.factors = tuple(float(factor) for factor in factors)
        self.rounds = int(rounds)
        if not self.parameter_names:
            raise ValueError("scale search requires parameters")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("scale search parameter names must be unique")
        if not self.factors:
            raise ValueError("scale search requires factors")
        if any(not math.isfinite(factor) or factor <= 0.0
               for factor in self.factors):
            raise ValueError("scale search factors must be finite and positive")
        if self.rounds <= 0:
            raise ValueError("scale search rounds must be positive")

    @staticmethod
    def _objective(evaluate: Callable[[Dict[str, float]], float],
                   values: Dict[str, float]) -> float:
        objective = float(evaluate(dict(values)))
        if not math.isfinite(objective):
            raise ValueError("scale search objective must be finite")
        return objective

    def run(self, initial_values: Dict[str, float],
            evaluate: Callable[[Dict[str, float]], float],
            sample_count: int) -> ScaleSearchResult:
        if set(initial_values) != set(self.parameter_names):
            missing = set(self.parameter_names) - set(initial_values)
            if missing:
                raise KeyError(sorted(missing)[0])
            raise ValueError("scale search received unknown parameters")
        sample_count = int(sample_count)
        if sample_count <= 0:
            raise ValueError("scale search sample count must be positive")

        base = dict((name, float(initial_values[name]))
                    for name in self.parameter_names)
        if any(not math.isfinite(base[name]) or base[name] <= 0.0
               for name in self.parameter_names):
            raise ValueError("initial scale values must be finite and positive")
        values = dict(base)
        objective = self._objective(evaluate, values)
        rows = []  # type: List[Dict[str, object]]

        for round_index in range(self.rounds):
            for name in self.parameter_names:
                best_values = dict(values)
                best_objective = objective
                best_factor = values[name] / base[name]
                start = len(rows)
                for factor in self.factors:
                    candidate = dict(values)
                    candidate[name] = base[name] * factor
                    candidate_objective = self._objective(evaluate, candidate)
                    rows.append({
                        "round": round_index,
                        "parameter": name,
                        "factor": factor,
                        "value": candidate[name],
                        "objective": candidate_objective,
                        "selected": False,
                        "sample_count": sample_count,
                    })
                    if candidate_objective < best_objective:
                        best_values = candidate
                        best_objective = candidate_objective
                        best_factor = factor
                selected_rows = [
                    index for index in range(start, len(rows))
                    if rows[index]["factor"] == best_factor
                ]
                if len(selected_rows) != 1:
                    raise RuntimeError("current scale factor is absent from candidates")
                rows[selected_rows[0]]["selected"] = True
                values = best_values
                objective = best_objective

        return ScaleSearchResult(
            values=dict(values), objective=objective, rows=rows)
