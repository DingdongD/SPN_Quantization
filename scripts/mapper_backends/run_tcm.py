#!/usr/bin/env python3
"""Run one common GEMM request through pinned AccelForge single-Einsum TCM."""

from __future__ import annotations

import argparse
from contextlib import chdir
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any, Dict


def _architecture_yaml(spec: Dict[str, Any]) -> str:
    """Render explicit-energy hardware without invoking component estimators."""
    operand_bits = int(spec["operand_bits"])
    act_values_per_cycle = (
        float(spec["activation_bandwidth_bits_per_cycle"]) / operand_bits)
    weight_values_per_cycle = (
        float(spec["weight_bandwidth_bits_per_cycle"]) / operand_bits)
    dram_values_per_cycle = (
        float(spec["dram_bandwidth_bits_per_cycle"]) / operand_bits)
    return f"""arch:
  nodes:
  - !Memory
    name: MainMemory
    size: inf
    leak_power: 0
    area: 0
    separate_read_write_ports: true
    bits_per_action: {operand_bits}
    tensors: {{keep: All}}
    actions:
    - {{name: read, energy: {spec['dram_energy_j_per_value']!r}, throughput: {dram_values_per_cycle!r}, latency: 0}}
    - {{name: write, energy: {spec['dram_energy_j_per_value']!r}, throughput: {dram_values_per_cycle!r}, latency: 0}}

  - !Memory
    name: ActivationBuffer
    size: {int(spec['activation_buffer_bits'])}
    leak_power: 0
    area: 0
    separate_read_write_ports: true
    bits_per_action: {operand_bits}
    tensors: {{keep: input | output}}
    actions:
    - {{name: read, energy: {spec['activation_read_energy_j_per_value']!r}, throughput: {act_values_per_cycle!r}, latency: 0}}
    - {{name: write, energy: {spec['activation_write_energy_j_per_value']!r}, throughput: {act_values_per_cycle!r}, latency: 0}}

  - !Memory
    name: WeightBuffer
    size: {int(spec['weight_buffer_bits'])}
    leak_power: 0
    area: 0
    separate_read_write_ports: true
    bits_per_action: {operand_bits}
    tensors: {{keep: weight}}
    actions:
    - {{name: read, energy: {spec['weight_read_energy_j_per_value']!r}, throughput: {weight_values_per_cycle!r}, latency: 0}}
    - {{name: write, energy: {spec['weight_write_energy_j_per_value']!r}, throughput: {weight_values_per_cycle!r}, latency: 0}}

  - !Container
    name: MACArray
    spatial:
    - {{name: ArrayRows, fanout: {int(spec['array_height'])}, min_usage: 0}}
    - {{name: ArrayColumns, fanout: {int(spec['array_width'])}, min_usage: 0}}

  - !Compute
    name: MAC
    leak_power: 0
    area: 0
    actions:
    - {{name: compute, energy: {spec['mac_energy_j']!r}, throughput: 1, latency: 0}}
"""


def _workload_yaml(payload: Dict[str, Any]) -> str:
    workload = payload["workload"]
    ranks = workload["rank_sizes"]
    einsum = workload["einsums"][0]
    return f"""workload:
  rank_sizes: {{M: {int(ranks['M'])}, N: {int(ranks['N'])}, K: {int(ranks['K'])}}}
  bits_per_value: {{All: 16}}
  einsums:
  - name: {einsum['name']}
    tensor_accesses:
    - {{name: A, projection: [m, k]}}
    - {{name: B, projection: [k, n]}}
    - {{name: C, projection: [m, n], output: true}}
    renames: {{input: A, weight: B, output: C}}
"""


def _as_float(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("TCM returned multiple Pareto mappings")
        value = value[0]
    return float(value)


def _mapping_text(mapped: Any) -> str:
    mapping = mapped.mapping()
    if hasattr(mapping, "compact_str"):
        return str(mapping.compact_str())
    return str(mapping)


def run(request: Dict[str, Any]) -> Dict[str, Any]:
    import accelforge as af
    from accelforge.frontend.spec import Spec
    from accelforge.mapper import Metrics
    from accelforge.util.parallel import set_n_parallel_jobs

    workload = request["workload"]
    architecture = request["architecture"]
    with tempfile.TemporaryDirectory(prefix="spn-tcm-") as tmp:
        root = Path(tmp)
        arch_path = root / "architecture.yaml"
        workload_path = root / "workload.yaml"
        arch_path.write_text(_architecture_yaml(architecture), encoding="utf-8")
        workload_path.write_text(_workload_yaml(workload), encoding="utf-8")
        spec = Spec.from_yaml(arch_path, workload_path)

    mapper = request["mapper"]
    optimization_level = int(mapper["optimization_level"])
    spec.mapper.set_optimization_level(optimization_level)
    spec.mapper.metrics = Metrics.ENERGY_DELAY_PRODUCT
    spec.mapper.info_metrics = Metrics.all_metrics()
    spec.mapper.max_fused_loops = int(mapper["max_fused_loops"])
    spec.mapper.explore_imperfect_spatial_loops = bool(
        mapper["explore_imperfect_spatial_loops"])
    spec.mapper.explore_imperfect_temporal_loops = bool(
        mapper["explore_imperfect_temporal_loops"])
    spec.mapper.tiling_coarseness = float(mapper["tiling_coarseness"])
    parallel_jobs = int(mapper.get("parallel_jobs", 3))
    set_n_parallel_jobs(parallel_jobs)
    started = time.perf_counter()
    # AccelForge renders ``mapping.svg`` in its current directory while
    # evaluating the winning mappings. Keep that backend side effect isolated.
    with tempfile.TemporaryDirectory(prefix="spn-tcm-render-") as render_tmp:
        with chdir(render_tmp):
            mapped = spec.map_workload_to_arch(print_progress=False)
    elapsed = time.perf_counter() - started
    candidate_count = len(mapped)
    if candidate_count < 1:
        raise ValueError("TCM EDP search returned no valid mapping")
    if candidate_count > 1:
        latencies = [float(value) for value in mapped.latency(
            list_if_one_mapping=True)]
        energies = [float(value) for value in mapped.energy(
            list_if_one_mapping=True)]
        best = min(
            range(candidate_count),
            key=lambda index: (
                energies[index] * latencies[index],
                latencies[index], energies[index], index))
        mapped = mapped[best]

    repetitions = int(workload["repetitions"])
    latency = _as_float(mapped.latency())
    energy_j = _as_float(mapped.energy())
    if not math.isfinite(latency) or latency <= 0:
        raise ValueError("TCM returned non-positive latency")
    if not math.isfinite(energy_j) or energy_j <= 0:
        raise ValueError("TCM returned non-positive energy")
    cycles_per_repeat = int(math.ceil(latency))
    cycles = cycles_per_repeat * repetitions
    energy_uj = energy_j * 1e6 * repetitions
    utilization = (
        int(workload["macs"]) /
        (cycles * int(architecture["array_height"]) *
         int(architecture["array_width"])))
    if not 0 < utilization <= 1:
        raise ValueError("TCM returned invalid effective utilization")
    resources = {
        str(key): _as_float(value)
        for key, value in mapped.resource_usage().items()
    }
    actions = {
        str(key): _as_float(value)
        for key, value in mapped.actions(
            per_einsum=True, per_component=True,
            per_tensor=True).items()
    }
    return {
        "format_version": 1,
        "backend": "tcm",
        "backend_revision": request["backend_revision"],
        "gemm_id": workload["workload"]["einsums"][0]["name"],
        "macs": int(workload["macs"]),
        "cycles": cycles,
        "energy_uj": energy_uj,
        "edp_uj_cycles": energy_uj * cycles,
        "utilization": utilization,
        "search_seconds": elapsed,
        "native_objective": "energy_delay_product_o%d" % optimization_level,
        "native_parallel_jobs": parallel_jobs,
        "native_optimization_level": optimization_level,
        "native_explore_imperfect_spatial_loops": (
            spec.mapper.explore_imperfect_spatial_loops),
        "native_explore_imperfect_temporal_loops": (
            spec.mapper.explore_imperfect_temporal_loops),
        "native_tiling_coarseness": spec.mapper.tiling_coarseness,
        "native_candidate_count": candidate_count,
        "native_total_mappings": int(mapped.total_mappings),
        "native_valid_mappings": int(mapped.valid_mappings),
        "latency_cycles_per_repeat": latency,
        "resource_usage_fraction": resources,
        "actions": actions,
        "mapping": _mapping_text(mapped),
        "accelforge_version": getattr(af, "__version__", "unknown"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser


def _self_test_request() -> Dict[str, Any]:
    return {
        "format_version": 1,
        "backend": "tcm",
        "backend_revision": "self-test",
        "workload": {
            "workload": {
                "rank_sizes": {"M": 8, "N": 8, "K": 8},
                "bits_per_value": {"All": 16},
                "einsums": [{"name": "self_test"}],
            },
            "repetitions": 1,
            "macs": 512,
        },
        "architecture": {
            "architecture_id": "self_test__4x4",
            "array_height": 4,
            "array_width": 4,
            "operand_bits": 16,
            "accumulator_bits": 32,
            "activation_buffer_bits": 4096 * 8,
            "weight_buffer_bits": 4096 * 8,
            "activation_bandwidth_bits_per_cycle": 256.0,
            "weight_bandwidth_bits_per_cycle": 256.0,
            "dram_bandwidth_bits_per_cycle": 128.0,
            "activation_read_energy_j_per_value": 1e-12,
            "activation_write_energy_j_per_value": 1e-12,
            "weight_read_energy_j_per_value": 1e-12,
            "weight_write_energy_j_per_value": 1e-12,
            "dram_energy_j_per_value": 30e-12,
            "mac_energy_j": 1e-12,
        },
        "mapper": {
            "algorithm": "accelforge_single_einsum_tcm",
            "objective": "energy_delay_product",
            "max_fused_loops": 1,
            "optimization_level": 3,
            "explore_imperfect_spatial_loops": False,
            "explore_imperfect_temporal_loops": False,
            "tiling_coarseness": 1.0,
            "parallel_jobs": 3,
        },
    }


def main() -> int:
    args = _parser().parse_args()
    if args.self_test:
        request = _self_test_request()
    else:
        if args.input is None or args.output is None:
            raise ValueError("--input and --output are required outside self-test")
        request = json.loads(args.input.read_text(encoding="utf-8"))
    if "requests" in request:
        result = {
            "format_version": 1,
            "results": [run(row) for row in request["requests"]],
        }
    else:
        result = run(request)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
