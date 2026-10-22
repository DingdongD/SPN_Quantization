#!/usr/bin/env python3
"""Run one common GEMM request through a pinned FactorFlow checkout."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterator


@contextmanager
def _factorflow_imports(
        root: Path, mapper_name: str) -> Iterator[Dict[str, Any]]:
    path = str(root.resolve())
    if not (root / "engine.py").is_file():
        raise FileNotFoundError("FactorFlow checkout is missing: %s" % root)
    prior = tuple(sys.path)
    sys.path.insert(0, path)
    try:
        from settings import Settings
        Settings.MAPPER = str(mapper_name)
        from arch import Arch
        from computations import gemm_coupling
        from engine import run_engine
        from factors import Shape
        from levels import ComputeLevel, FanoutLevel, MemLevel
        yield {
            "Arch": Arch,
            "ComputeLevel": ComputeLevel,
            "FanoutLevel": FanoutLevel,
            "MemLevel": MemLevel,
            "Settings": Settings,
            "Shape": Shape,
            "gemm_coupling": gemm_coupling,
            "run_engine": run_engine,
        }
    finally:
        sys.path[:] = prior


def _architecture(api: Dict[str, Any], spec: Dict[str, Any]) -> Any:
    MemLevel = api["MemLevel"]
    FanoutLevel = api["FanoutLevel"]
    ComputeLevel = api["ComputeLevel"]
    Arch = api["Arch"]
    return Arch([
        MemLevel(
            name="DRAM", size=2**63 - 1,
            value_access_energy=spec["dram_energy_pj_per_value"],
            bandwidth=spec["dram_bandwidth_values_per_cycle"],
            factors_constraints={}, bypasses=[]),
        MemLevel(
            name="ActivationBuffer", size=spec["activation_entries"],
            read_value_access_energy=(
                spec["activation_read_energy_pj_per_value"]),
            write_value_access_energy=(
                spec["activation_write_energy_pj_per_value"]),
            bandwidth=spec["activation_bandwidth_values_per_cycle"],
            factors_constraints={}, bypasses=["w"]),
        MemLevel(
            name="WeightBuffer", size=spec["weight_entries"],
            read_value_access_energy=spec["weight_read_energy_pj_per_value"],
            write_value_access_energy=spec["weight_write_energy_pj_per_value"],
            bandwidth=spec["weight_bandwidth_values_per_cycle"],
            factors_constraints={}, bypasses=["in", "out"]),
        FanoutLevel(
            name="ArrayRows", dim="M", mesh=spec["array_height"],
            pe_to_pe=False, factors_constraints={}),
        FanoutLevel(
            name="ArrayCols", dim="N", mesh=spec["array_width"],
            pe_to_pe=False, factors_constraints={}),
        MemLevel(
            name="Accumulator", size=1,
            value_access_energy=1e-12, bandwidth=2,
            factors_constraints={}, bypasses=["in", "w"]),
        MemLevel(
            name="OperandRegisters", size=2,
            value_access_energy=1e-12, bandwidth=2,
            dataflow_constraints=["M", "N", "K"],
            factors_constraints={"M": 1, "N": 1}, bypasses=["out"]),
        ComputeLevel(
            name="Compute", dim="K", mesh=1,
            compute_energy=spec["mac_energy_pj"], cycles=1,
            factors_constraints={"K": 1}),
    ], coupling=api["gemm_coupling"], name=spec["architecture_id"])


def _mapping(arch: Any) -> list[Dict[str, Any]]:
    rows = []
    for level in arch:
        rows.append({
            "level": level.name,
            "dataflow": list(level.dataflow),
            "factors": {
                dim: int(level.factors.dimProduct(dim))
                for dim in arch.coupling.dims
            },
            "tile_sizes": {
                dim: int(level.tile_sizes[dim])
                for dim in arch.coupling.dims
            },
        })
    return rows


def run(request: Dict[str, Any], factorflow_root: Path) -> Dict[str, Any]:
    workload = request["workload"]
    spec = request["architecture"]
    mapper = request.get("mapper", {})
    with _factorflow_imports(
            factorflow_root, mapper.get("algorithm", "linear")) as api:
        settings = api["Settings"]
        settings.VERBOSE = False
        settings.MULTITHREADED = bool(mapper.get("multithreaded", False))
        settings.PADDED_MAPPINGS = bool(mapper.get("padded_mappings", True))
        settings.FREE_DRAINS = False
        arch = _architecture(api, spec)
        comp = api["Shape"](
            M=int(workload["m"]), K=int(workload["k"]),
            N=int(workload["n"]))
        edp, mops, energy, latency, utilization, runtime, mapped = (
            api["run_engine"](
                arch, comp, api["gemm_coupling"], False, verbose=False))
    repetitions = int(workload["repetitions"])
    cycles = int(math.ceil(float(latency))) * repetitions
    energy_uj = float(energy) * repetitions
    effective_utilization = (
        int(workload["macs"]) /
        (cycles * int(spec["array_height"]) * int(spec["array_width"])))
    if not 0 < effective_utilization <= 1:
        raise ValueError("FactorFlow returned invalid effective utilization")
    reads, writes = mops
    return {
        "format_version": 1,
        "backend": "factorflow",
        "backend_revision": request["backend_revision"],
        "gemm_id": workload["gemm_id"],
        "macs": int(workload["macs"]),
        "cycles": cycles,
        "energy_uj": energy_uj,
        "edp_uj_cycles": energy_uj * cycles,
        "utilization": effective_utilization,
        "native_spatial_utilization": float(utilization),
        "search_seconds": float(runtime),
        "native_objective": "wart",
        "mops_reads": int(reads) * repetitions,
        "mops_writes": int(writes) * repetitions,
        "native_edp_j_cycles": float(edp) * repetitions * repetitions,
        "mapping": _mapping(mapped),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factorflow-root", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if "requests" in payload:
        result = {
            "format_version": 1,
            "results": [
                run(request, args.factorflow_root)
                for request in payload["requests"]],
        }
    else:
        result = run(payload, args.factorflow_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
