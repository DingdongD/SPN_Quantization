"""Common dense-workload contract for cross-mapper SPN validation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Sequence, Tuple

from .workload import GemmWorkload, WorkloadProfile, lower_layer_to_gemms


@dataclass(frozen=True)
class DenseGemmInstance:
    """One original model layer lowered to a dense GEMM."""

    model_name: str
    layer_name: str
    kind: str
    execution_index: int
    workload: GemmWorkload

    @property
    def signature(self) -> Tuple[int, int, int, int]:
        return (
            self.workload.m,
            self.workload.n,
            self.workload.k,
            self.workload.repetitions,
        )


@dataclass(frozen=True)
class UniqueGemm:
    """A GEMM shape shared by one or more original model layers."""

    gemm_id: str
    workload: GemmWorkload
    instances: Tuple[DenseGemmInstance, ...]

    def __post_init__(self) -> None:
        if not self.gemm_id or not self.instances:
            raise ValueError("unique GEMM identity and instances are required")
        expected = (
            self.workload.m,
            self.workload.n,
            self.workload.k,
            self.workload.repetitions,
        )
        if any(row.signature != expected for row in self.instances):
            raise ValueError("unique GEMM contains a different shape")


@dataclass(frozen=True)
class BackendResult:
    """Backend-native metrics for one GEMM and architecture point."""

    backend: str
    architecture_id: str
    gemm_id: str
    macs: int
    cycles: int
    energy_uj: float
    utilization: float
    search_seconds: float
    native_objective: str

    def __post_init__(self) -> None:
        for name in ("backend", "architecture_id", "gemm_id",
                     "native_objective"):
            if not str(getattr(self, name)):
                raise ValueError("%s must be non-empty" % name)
        for name in ("macs", "cycles"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError("%s must be positive" % name)
            object.__setattr__(self, name, value)
        energy = float(self.energy_uj)
        if not math.isfinite(energy) or energy <= 0:
            raise ValueError("energy must be positive")
        object.__setattr__(self, "energy_uj", energy)
        utilization = float(self.utilization)
        if not math.isfinite(utilization) or not 0 < utilization <= 1:
            raise ValueError("utilization must lie in (0, 1]")
        object.__setattr__(self, "utilization", utilization)
        runtime = float(self.search_seconds)
        if not math.isfinite(runtime) or runtime < 0:
            raise ValueError("search time cannot be negative")
        object.__setattr__(self, "search_seconds", runtime)

    @property
    def edp_uj_cycles(self) -> float:
        return self.energy_uj * self.cycles


def dense_gemm_instances(
        profile: WorkloadProfile) -> Tuple[DenseGemmInstance, ...]:
    """Lower only array-mappable layers; irregular SPN operations are absent."""
    rows = []
    for layer in profile.layers:
        if not layer.array_mappable:
            continue
        gemms = lower_layer_to_gemms(layer)
        if not gemms:
            raise ValueError(
                "array-mappable layer produced no GEMM: %s" % layer.name)
        if sum(row.macs for row in gemms) != layer.macs:
            raise ValueError("GEMM lowering changed MAC count: %s" % layer.name)
        for workload in gemms:
            rows.append(DenseGemmInstance(
                model_name=profile.model_name,
                layer_name=layer.name,
                kind=layer.kind,
                execution_index=layer.execution_index,
                workload=workload,
            ))
    return tuple(sorted(
        rows,
        key=lambda row: (
            row.execution_index,
            row.layer_name,
            row.workload.m,
            row.workload.n,
            row.workload.k,
            row.workload.repetitions,
        ),
    ))


def deduplicate_gemms(
        instances: Sequence[DenseGemmInstance]) -> Tuple[UniqueGemm, ...]:
    """Group identical GEMM shapes without losing original layer identities."""
    grouped: Dict[Tuple[int, int, int, int], list[DenseGemmInstance]] = {}
    for row in instances:
        grouped.setdefault(row.signature, []).append(row)
    unique = []
    for index, signature in enumerate(sorted(grouped)):
        rows = tuple(sorted(
            grouped[signature],
            key=lambda row: (
                row.model_name, row.execution_index, row.layer_name)))
        representative = rows[0].workload
        unique.append(UniqueGemm(
            gemm_id="gemm_%04d" % index,
            workload=GemmWorkload(
                name="gemm_%04d" % index,
                kind=representative.kind,
                m=signature[0],
                n=signature[1],
                k=signature[2],
                repetitions=signature[3],
            ),
            instances=rows,
        ))
    return tuple(unique)


def validate_backend_result(
        unique: UniqueGemm, result: BackendResult) -> BackendResult:
    """Validate backend identity and arithmetic conservation."""
    if result.gemm_id != unique.gemm_id:
        raise ValueError("backend result GEMM identity mismatch")
    if result.macs != unique.workload.macs:
        raise ValueError("backend result MAC count mismatch")
    return result
