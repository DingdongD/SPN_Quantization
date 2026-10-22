import csv
import json

import pytest

from scripts.run_spn_mapper_validation import run_validation, validate_artifacts
from spn_quant.hardware.cacti import CactiSramResult
from spn_quant.hardware.joint_dse import ArchitecturePoint, CostAssumptions
from spn_quant.hardware.llmcompass_adapter import ArrayShape
from spn_quant.hardware.mapper_validation import BackendResult
from spn_quant.hardware.memory_dse import MemoryArchitecture
from spn_quant.hardware.workload import LayerWorkload, WorkloadProfile


MODELS = ("cspn", "dyspn", "nlspn", "completionformer")


def _macro(capacity=64 * 1024):
    return CactiSramResult(
        capacity_bytes=capacity, access_time_ns=1.0, cycle_time_ns=1.0,
        area_mm2=1.0, read_energy_nj=0.4, write_energy_nj=0.6,
        output_width_bits=256, bank_count=4)


def _memory():
    macro = _macro()
    return MemoryArchitecture(
        dtype="bf16", operand_bytes=2, output_bytes=2,
        accumulator_bytes=4, clock_hz=1e9,
        activation_buffer_bytes=macro.capacity_bytes,
        weight_buffer_bytes=macro.capacity_bytes,
        lpddr_transfers_per_second=4.266e9,
        lpddr_bus_width_bits=32, lpddr_channels=1,
        dram_efficiencies=(0.7,), primary_dram_efficiency=0.7,
        activation_cacti=macro, weight_cacti=macro)


def _profiles():
    profiles = {}
    for index, model in enumerate(MODELS):
        profiles[model] = WorkloadProfile(
            model_name=model, source_config="cfg", checkpoint="ckpt",
            sample_index=0, input_shapes=(("image", (1, 4)),),
            layers=(
                LayerWorkload.linear(
                    "%s_dense" % model, (1, 4), (1, 8),
                    execution_index=0),
                LayerWorkload.irregular(
                    "%s_prop" % model, "propagation", (1, 8), (1, 8),
                    macs=8 + index, execution_index=1),
            ))
    return profiles


def _points():
    memory = _memory()
    return (
        ArchitecturePoint(
            "balanced__8x8", "balanced", ArrayShape(8, 8), memory),
        ArchitecturePoint(
            "balanced__16x16", "balanced", ArrayShape(16, 16), memory),
    )


def _cost():
    return CostAssumptions(
        technology_nm=32,
        logic_transistor_density_million_per_mm2=10,
        bf16_mac_energy_pj=1,
        lpddr_energy_pj_per_byte=100,
        lpddr_channel_area_mm2=1,
        fixed_logic_area_mm2=1,
        implementation_area_margin=0.2)


def _executor(backend):
    def execute(unique, point, assumptions):
        del assumptions
        cycles = max(1, unique.workload.macs // point.array.pe_count)
        return BackendResult(
            backend=backend, architecture_id=point.architecture_id,
            gemm_id=unique.gemm_id, macs=unique.workload.macs,
            cycles=cycles, energy_uj=unique.workload.macs / 1e6,
            utilization=min(
                1.0, unique.workload.macs /
                (cycles * point.array.pe_count)),
            search_seconds=0.01, native_objective="test")
    return execute


def _read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_validation_artifacts_cover_three_backends_and_exclude_propagation(tmp_path):
    output = run_validation(
        _profiles(), _points(), _cost(), tmp_path / "analysis",
        executors={backend: _executor(backend) for backend in (
            "llmcompass", "tcm", "factorflow")})

    coverage = _read(output / "workload_coverage.csv")
    layers = _read(output / "layer_results.csv")
    statuses = _read(output / "backend_status.csv")
    architectures = _read(output / "architecture_manifest.csv")

    assert sum(int(row["included_layer_count"]) for row in coverage) == 4
    assert sum(int(row["excluded_propagation_count"]) for row in coverage) == 4
    assert {row["kind"] for row in layers} == {"linear"}
    assert len(layers) == 4 * 2 * 3
    assert {row["backend"] for row in statuses} == {
        "llmcompass", "tcm", "factorflow"}
    assert all(row["status"] == "passed" for row in statuses)
    assert len(architectures) == 2
    assert {row["array_shape"] for row in architectures} == {"8x8", "16x16"}
    assert all(int(row["activation_buffer_bytes"]) == 64 * 1024
               for row in architectures)


def test_validation_deduplicates_shapes_and_preserves_model_instances(tmp_path):
    calls = {backend: 0 for backend in ("llmcompass", "tcm", "factorflow")}

    def counted(backend):
        execute = _executor(backend)

        def wrapper(*args):
            calls[backend] += 1
            return execute(*args)
        return wrapper

    output = run_validation(
        _profiles(), _points(), _cost(), tmp_path / "analysis",
        executors={backend: counted(backend) for backend in calls})

    assert calls == {backend: 2 for backend in calls}
    assert len(_read(output / "unique_gemm_results.csv")) == 2 * 3
    assert len(_read(output / "model_summary.csv")) == 4 * 2 * 3
    assert len(_read(output / "architecture_ranking.csv")) == 2 * 3


def test_artifact_validation_rejects_a_propagation_result(tmp_path):
    output = run_validation(
        _profiles(), _points(), _cost(), tmp_path / "analysis",
        executors={backend: _executor(backend) for backend in (
            "llmcompass", "tcm", "factorflow")})
    rows = _read(output / "layer_results.csv")
    rows[0]["kind"] = "propagation"
    with (output / "layer_results.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="propagation"):
        validate_artifacts(output)


def test_artifact_validation_rejects_incomplete_unique_results(tmp_path):
    output = run_validation(
        _profiles(), _points(), _cost(), tmp_path / "analysis",
        executors={backend: _executor(backend) for backend in (
            "llmcompass", "tcm", "factorflow")})
    rows = _read(output / "unique_gemm_results.csv")[:-1]
    with (output / "unique_gemm_results.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    metadata = json.loads(
        (output / "validation.json").read_text(encoding="utf-8"))
    assert metadata["unique_gemm_count"] == 1
    with pytest.raises(ValueError, match="unique GEMM result count"):
        validate_artifacts(output)
