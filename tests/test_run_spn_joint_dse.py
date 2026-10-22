import csv
import json
from pathlib import Path

import pytest

from scripts.run_spn_joint_dse import _plot, run, validate_artifacts
from spn_quant.hardware.joint_dse import CandidateEstimate
from spn_quant.hardware.llmcompass_adapter import ArrayShape
from spn_quant.hardware.workload import LayerWorkload, WorkloadProfile


MODELS = ("cspn", "dyspn", "nlspn", "completionformer")
CACTI_REVISION = "1ffd8dfb10303d306ecd8d215320aea07651e878"


def _write_compute(path):
    path.write_text(json.dumps({
        "format_version": 1,
        "dtype": "bf16",
        "word_bytes": 2,
        "mac_per_cycle": 1,
        "dataflow": "os",
        "array_shapes": [[8, 8], [8, 16]],
        "iso_pe_groups": {"128": [[8, 16]]},
        "low_utilization_threshold": 0.25,
    }), encoding="utf-8")
    return path


def _write_macro(path, capacity, width, area):
    path.write_text(json.dumps({
        "format_version": 1,
        "source": "https://github.com/HewlettPackard/cacti",
        "cacti_revision": CACTI_REVISION,
        "compatibility_patch":
            "patches/cacti/0001-initialize-input-parameters.patch",
        "config": {
            "capacity_bytes": capacity,
            "technology_nm": 32,
            "output_width_bits": width,
        },
        "result": {
            "capacity_bytes": capacity,
            "access_time_ns": 1.0,
            "cycle_time_ns": 1.0,
            "area_mm2": area,
            "read_energy_nj": 0.5,
            "write_energy_nj": 0.75,
            "output_width_bits": width,
            "bank_count": 8,
        },
    }), encoding="utf-8")
    return path


def _write_config(path, compute, small, large):
    profiles = [
        ("conservative", "conservative", "small", "small", 1, 0.6),
        ("balanced", "balanced", "large", "large", 1, 0.7),
        ("aggressive", "aggressive", "large", "large", 2, 0.8),
        ("activation_heavy", "activation-heavy", "large", "small", 1, 0.7),
        ("weight_heavy", "weight-heavy", "small", "large", 1, 0.7),
    ]
    path.write_text(json.dumps({
        "format_version": 1,
        "dtype": "bf16",
        "operand_bytes": 2,
        "output_bytes": 2,
        "accumulator_bytes": 4,
        "clock_hz": 1.0e9,
        "compute_dse_config": str(compute),
        "sram_macros": {"small": str(small), "large": str(large)},
        "lpddr4": {
            "transfers_per_second": 4.266e9,
            "bus_width_bits": 32,
            "source": "test LPDDR4-4266",
        },
        "architecture_profiles": [{
            "id": profile_id,
            "class": class_name,
            "activation_sram": activation,
            "weight_sram": weight,
            "lpddr_channels": channels,
            "sustained_efficiency": efficiency,
        } for (profile_id, class_name, activation, weight, channels,
               efficiency) in profiles],
        "cost_model": {
            "technology_nm": 32,
            "logic_transistor_density_million_per_mm2": 10.0,
            "bf16_mac_energy_pj": 1.0,
            "lpddr_energy_pj_per_byte": 100.0,
            "lpddr_channel_area_mm2": 3.0,
            "fixed_logic_area_mm2": 1.0,
            "implementation_area_margin": 0.2,
        },
        "constraints": {},
    }), encoding="utf-8")
    return path


def _write_profiles(root):
    paths = {}
    for index, model in enumerate(MODELS):
        profile = WorkloadProfile(
            model_name=model, source_config="source.json",
            checkpoint="/%s.pt" % model, sample_index=386,
            input_shapes=(("arg0", (1, 4, 8, 8)),),
            layers=(LayerWorkload.linear(
                "linear", [1, 32], [1, 16 + index], execution_index=0),),
            metadata=(("checkpoint_sha256", "%064d" % index),))
        paths[model] = root / (model + ".json")
        profile.write_json(paths[model])
    return paths


def _inputs(tmp_path):
    compute = _write_compute(tmp_path / "compute.json")
    small = _write_macro(tmp_path / "small.json", 32 * 1024, 256, 1.0)
    large = _write_macro(tmp_path / "large.json", 64 * 1024, 512, 2.0)
    config = _write_config(
        tmp_path / "joint.json", compute, small, large)
    return config, _write_profiles(tmp_path)


def _csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def test_joint_dse_writes_complete_validated_artifacts(tmp_path):
    config, workloads = _inputs(tmp_path)

    output = run(config, workloads, tmp_path / "report")

    assert {path.name for path in output.iterdir()} == {
        "joint_architecture_catalog.csv", "joint_candidate_summary.csv",
        "joint_layer_results.csv", "joint_model_summary.csv",
        "joint_pareto_frontier.csv", "joint_pareto.png", "metadata.json",
    }
    layers = _csv(output / "joint_layer_results.csv")
    models = _csv(output / "joint_model_summary.csv")
    candidates = _csv(output / "joint_candidate_summary.csv")
    assert len(layers) == 5 * 2 * 4
    assert len(models) == 5 * 2 * 4
    assert len(candidates) == 5 * 2
    assert {row["architecture_class"] for row in candidates} == {
        "conservative", "balanced", "aggressive",
        "activation-heavy", "weight-heavy"}
    assert all(float(row["total_energy_uj"]) > 0 for row in layers)
    assert all(int(row["model_count"]) == 4 for row in candidates)
    assert all(float(row["worst_normalized_latency"]) >= 1
               for row in candidates)
    assert validate_artifacts(output) == "SPN_JOINT_DSE_VALIDATION_OK"


def test_joint_validator_rejects_capacity_violation(tmp_path):
    config, workloads = _inputs(tmp_path)
    output = run(config, workloads, tmp_path / "report")
    path = output / "joint_layer_results.csv"
    rows = list(_csv(path))
    fields = tuple(rows[0])
    rows[0]["activation_capacity_bytes"] = str(
        int(rows[0]["activation_buffer_bytes"]) + 1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="activation capacity"):
        validate_artifacts(output)


def test_joint_parallel_results_match_serial_results(tmp_path):
    config, workloads = _inputs(tmp_path)
    serial = run(config, workloads, tmp_path / "serial", workers=1)
    parallel = run(config, workloads, tmp_path / "parallel", workers=2)

    for name in (
            "joint_architecture_catalog.csv", "joint_candidate_summary.csv",
            "joint_layer_results.csv", "joint_model_summary.csv",
            "joint_pareto_frontier.csv"):
        assert (serial / name).read_bytes() == (parallel / name).read_bytes()
    assert validate_artifacts(parallel) == "SPN_JOINT_DSE_VALIDATION_OK"


def test_joint_dse_rejects_cacti_node_mismatch(tmp_path):
    config, workloads = _inputs(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["cost_model"]["technology_nm"] = 7
    config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="technology node"):
        run(config, workloads, tmp_path / "report")


def test_joint_dse_accepts_named_sram_macro_overrides(tmp_path):
    config, workloads = _inputs(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    valid = {name: Path(path) for name, path in payload["sram_macros"].items()}
    payload["sram_macros"] = {
        "small": "missing-small.json", "large": "missing-large.json"}
    config.write_text(json.dumps(payload), encoding="utf-8")

    output = run(
        config, workloads, tmp_path / "report", sram_macro_paths=valid)

    assert validate_artifacts(output) == "SPN_JOINT_DSE_VALIDATION_OK"


def test_joint_dse_reports_empty_frontier_when_all_constraints_fail(tmp_path):
    config, workloads = _inputs(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["constraints"] = {"max_total_area_mm2": 0.001}
    config.write_text(json.dumps(payload), encoding="utf-8")

    output = run(config, workloads, tmp_path / "report")

    assert _csv(output / "joint_pareto_frontier.csv") == ()
    assert all(int(row["feasible"]) == 0
               for row in _csv(output / "joint_candidate_summary.csv"))
    assert validate_artifacts(output) == "SPN_JOINT_DSE_VALIDATION_OK"


def test_three_objective_pareto_projection_does_not_connect_points(
        tmp_path, monkeypatch):
    calls = []

    def record_plot(*args, **kwargs):
        calls.append((args, kwargs))
        return []

    monkeypatch.setattr("matplotlib.axes.Axes.plot", record_plot)
    candidates = tuple(CandidateEstimate(
        architecture_id="candidate_%d" % index,
        architecture_class="test", array=ArrayShape(8, 8), model_count=4,
        geomean_latency_ms=latency, mean_energy_uj=energy,
        max_average_dynamic_power_w=1.0, total_area_mm2=area,
        model_estimates=(), feasible=True, constraint_violations=(),
        worst_normalized_latency=1.0)
        for index, (area, latency, energy) in enumerate((
            (1.0, 2.0, 1.0), (2.0, 1.0, 2.0))))

    _plot(tmp_path / "pareto.png", candidates, candidates)

    assert calls == []
