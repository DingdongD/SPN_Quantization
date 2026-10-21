"""Thin, reproducible adapter around LLMCompass's systolic cycle model."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import importlib
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterator, Tuple

import pandas as pd

from .workload import GemmWorkload


LLMCOMPASS_REVISION = "62321b1ee28ddbdba8a2eb7475d7caa30f75e8be"
REPO_ROOT = Path(__file__).resolve().parents[2]
LLMCOMPASS_ROOT = REPO_ROOT / "external" / "LLMCompass"


@dataclass(frozen=True)
class ArrayShape:
    height: int
    width: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "width", int(self.width))
        if self.height <= 0 or self.width <= 0:
            raise ValueError("array dimensions must be positive")

    @property
    def pe_count(self) -> int:
        return self.height * self.width

    @property
    def label(self) -> str:
        return "%dx%d" % (self.height, self.width)


@dataclass(frozen=True)
class GemmSimulationResult:
    name: str
    kind: str
    execution_index: int
    m: int
    n: int
    k: int
    repetitions: int
    array: ArrayShape
    mac_per_cycle: float
    dataflow: str
    macs: int
    cycles: int
    ideal_cycles: int
    effective_utilization: float
    ideal_cycle_overhead: float
    backend: str
    simulation_source: str

    @property
    def array_pe_count(self) -> int:
        return self.array.pe_count


@lru_cache(maxsize=1)
def _matmul_class() -> Any:
    if not (LLMCOMPASS_ROOT / "software_model" / "matmul.py").is_file():
        raise FileNotFoundError(
            "LLMCompass submodule is not initialized: %s" % LLMCOMPASS_ROOT)
    root = str(LLMCOMPASS_ROOT)
    prior_utils = sys.modules.get("utils")
    prior_utils_path = "" if prior_utils is None else str(
        getattr(prior_utils, "__file__", ""))
    displaced_utils = prior_utils is not None and not prior_utils_path.startswith(root)
    if displaced_utils:
        del sys.modules["utils"]
    sys.path.insert(0, root)
    try:
        module = importlib.import_module("software_model.matmul")
    finally:
        try:
            sys.path.remove(root)
        except ValueError:  # pragma: no cover - defensive against import hooks
            pass
        if displaced_utils:
            sys.modules["utils"] = prior_utils
    return module.Matmul


def _empty_lookup() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples([], names=(
        "M", "N", "K", "ArrayHeight", "ArrayWidth", "Dataflow"))
    return pd.DataFrame(index=index, columns=("cycle_count", "util_rate"))


def _uses_analytical_dimensions(m: int, n: int, k: int,
                                height: int, width: int) -> bool:
    maximum = max(height, width)
    if m >= height and n >= width:
        return m * n * k / height / width / maximum >= 64
    if m >= height and n < width:
        return k * m / height / maximum >= 64
    if m < height and n >= width:
        return k * n / width / maximum >= 64
    return k / maximum >= 64


@contextmanager
def _temporary_llmcompass_workdir() -> Iterator[Path]:
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="spn-llmcompass-") as directory:
        root = Path(directory)
        (root / "systolic_array_model" / "temp").mkdir(parents=True)
        os.chdir(str(root))
        try:
            yield root
        finally:
            os.chdir(str(previous))


@lru_cache(maxsize=None)
def _direct_cycle_count(m: int, n: int, k: int, height: int, width: int,
                        mac_per_cycle: float, dataflow: str) -> int:
    """Invoke the upstream primitive once for one physical-array tile."""
    matmul_class = _matmul_class()
    arguments = (
        _empty_lookup(), int(m), int(n), int(k), int(height), int(width),
        float(mac_per_cycle), str(dataflow),
    )
    if _uses_analytical_dimensions(m, n, k, height, width):
        return int(matmul_class.simulate_systolic_array_cycle_count(*arguments))
    with _temporary_llmcompass_workdir():
        return int(matmul_class.simulate_systolic_array_cycle_count(*arguments))


@lru_cache(maxsize=None)
def _cycle_count(m: int, n: int, k: int, height: int, width: int,
                 mac_per_cycle: float, dataflow: str) -> Tuple[int, str]:
    """Model a GEMM, tiling Scale-Sim fallbacks to the physical array.

    LLMCompass's fast path accepts full GEMM dimensions. Its Scale-Sim
    fallback materializes demand matrices proportional to the full spatial
    dimension, which is unnecessary for low-K image layers. LLMCompass uses
    this primitive at the L1-tile level internally, so the fallback is applied
    to at most one physical array tile and multiplied over full and tail tiles.
    """
    if _uses_analytical_dimensions(m, n, k, height, width):
        return (_direct_cycle_count(
            m, n, k, height, width, mac_per_cycle, dataflow), "analytical")
    full_m, tail_m = divmod(int(m), int(height))
    full_n, tail_n = divmod(int(n), int(width))
    m_tiles = []
    n_tiles = []
    if full_m:
        m_tiles.append((int(height), full_m))
    if tail_m:
        m_tiles.append((tail_m, 1))
    if full_n:
        n_tiles.append((int(width), full_n))
    if tail_n:
        n_tiles.append((tail_n, 1))
    total = 0
    for tile_m, count_m in m_tiles:
        for tile_n, count_n in n_tiles:
            total += count_m * count_n * _direct_cycle_count(
                tile_m, tile_n, k, height, width,
                mac_per_cycle, dataflow)
    if total <= 0:
        raise RuntimeError("tiled LLMCompass cycle count is non-positive")
    return total, "scalesim_tiled"


def clear_simulation_cache() -> None:
    _cycle_count.cache_clear()
    _direct_cycle_count.cache_clear()


def simulation_cache_info() -> Any:
    return _cycle_count.cache_info()


def simulate_gemm(workload: GemmWorkload, array: ArrayShape,
                  mac_per_cycle: float = 1, dataflow: str = "os"
                  ) -> GemmSimulationResult:
    """Run one GEMM through the upstream LLMCompass array-cycle primitive."""
    rate = float(mac_per_cycle)
    if rate <= 0:
        raise ValueError("MACs per cycle must be positive")
    if dataflow not in ("os", "ws", "is"):
        raise ValueError("unsupported dataflow: %s" % dataflow)
    cycles_per_repetition, simulation_source = _cycle_count(
        workload.m, workload.n, workload.k, array.height, array.width,
        rate, dataflow)
    cycles = cycles_per_repetition * workload.repetitions
    macs = workload.macs
    ideal_cycles = int(math.ceil(
        macs / float(array.pe_count * rate)))
    if cycles < ideal_cycles:
        # LLMCompass's narrow-N analytical heuristic divides N/array_width by
        # 0.98; when N occupies more than 98% of the columns this can imply
        # utilization above the physical peak. Preserve the upstream estimate
        # unless it violates that hard lower bound, then record the adjustment.
        cycles = ideal_cycles
        simulation_source += "_ideal_floor"
    effective = macs / float(cycles * array.pe_count * rate)
    if effective > 1.0:
        if effective - 1.0 > 1e-12:
            raise RuntimeError(
                "LLMCompass returned physically invalid utilization: %.12f" %
                effective)
        effective = 1.0
    if effective <= 0:
        raise RuntimeError("LLMCompass returned non-positive utilization")
    return GemmSimulationResult(
        name=workload.name, kind=workload.kind,
        execution_index=workload.execution_index,
        m=workload.m, n=workload.n, k=workload.k,
        repetitions=workload.repetitions, array=array,
        mac_per_cycle=rate, dataflow=dataflow, macs=macs, cycles=cycles,
        ideal_cycles=ideal_cycles, effective_utilization=effective,
        ideal_cycle_overhead=cycles / float(ideal_cycles),
        backend="llmcompass",
        simulation_source=simulation_source)
