import csv
import json
from pathlib import Path

import pytest

from scripts.run_spn_memory_dse import run, validate_artifacts
from spn_quant.hardware.workload import LayerWorkload, WorkloadProfile


MODELS = ("cspn", "dyspn", "nlspn", "completionformer")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_compute_config(path):
    payload = {
        "format_version": 1,
        "dtype": "bf16",
        "word_bytes": 2,
        "mac_per_cycle": 1,
        "dataflow": "os",
        "array_shapes": [[8, 8], [8, 16], [16, 8]],
        "iso_pe_groups": {"128": [[8, 16], [16, 8]]},
        "low_utilization_threshold": 0.25,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_cacti_result(path):
    payload = {
        "format_version": 1,
        "source": "https://github.com/HewlettPackard/cacti",
        "cacti_revision": "1ffd8dfb10303d306ecd8d215320aea07651e878",
        "compatibility_patch":
            "patches/cacti/0001-initialize-input-parameters.patch",
        "config": {"capacity_bytes": 4 * 1024 * 1024},
        "result": {
            "capacity_bytes": 4 * 1024 * 1024,
            "access_time_ns": 2.10289,
            "cycle_time_ns": 0.995496,
            "area_mm2": 7.0543941208,
            "read_energy_nj": 0.589761,
            "write_energy_nj": 0.672701,
            "output_width_bits": 512,
            "bank_count": 16,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_memory_config(path, compute_config, cacti_result):
    payload = {
        "format_version": 1,
        "dtype": "bf16",
        "operand_bytes": 2,
        "output_bytes": 2,
        "accumulator_bytes": 4,
        "clock_hz": 1.0e9,
        "activation_buffer_bytes": 4 * 1024 * 1024,
        "weight_buffer_bytes": 4 * 1024 * 1024,
        "lpddr4": {
            "transfers_per_second": 4.266e9,
            "bus_width_bits": 32,
            "channels": 1,
            "source": "test LPDDR4-4266 x32",
        },
        "dram_efficiencies": [0.5, 0.7, 0.9],
        "primary_dram_efficiency": 0.7,
        "cacti_result": str(cacti_result),
        "compute_dse_config": str(compute_config),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_profiles(root):
    result = {}
    for index, model in enumerate(MODELS):
        profile = WorkloadProfile(
            model_name=model, source_config="source.json",
            checkpoint="/%s.pt" % model, sample_index=386,
            input_shapes=(("arg0", (1, 4, 8, 8)),),
            layers=(
                LayerWorkload.linear(
                    "linear", [1, 64], [1, 32 + index], execution_index=0),
                LayerWorkload.irregular(
                    "%s::propagation" % model, "propagation",
                    [1, 1, 8, 8], [1, 1, 8, 8], macs=512,
                    execution_index=1, iterations=1, neighborhood_size=8),
            ),
            metadata=(("checkpoint_sha256", "%064d" % index),))
        path = root / (model + ".json")
        profile.write_json(path)
        result[model] = path
    return result


def _study_inputs(tmp_path):
    compute = _write_compute_config(tmp_path / "compute.json")
    cacti = _write_cacti_result(tmp_path / "cacti.json")
    memory = _write_memory_config(
        tmp_path / "memory.json", compute, cacti)
    return memory, _write_profiles(tmp_path)


def test_memory_analyzer_writes_complete_validated_artifact_set(tmp_path):
    config, workloads = _study_inputs(tmp_path)
    output = run(config, workloads, tmp_path / "report")

    assert {path.name for path in output.iterdir()} == {
        "bandwidth_sensitivity.csv", "bottleneck_summary.csv",
        "candidate_memory_ranking.csv", "layer_memory_results.csv",
        "memory_pareto_frontier.csv", "memory_shape_latency.png",
        "metadata.json", "model_memory_summary.csv",
    }
    with (output / "layer_memory_results.csv").open(newline="") as handle:
        layers = tuple(csv.DictReader(handle))
    with (output / "model_memory_summary.csv").open(newline="") as handle:
        summaries = tuple(csv.DictReader(handle))
    assert len(layers) == 4 * 3 * 3
    assert len(summaries) == 4 * 3 * 3
    for row in layers:
        assert int(row["activation_capacity_bytes"]) <= 4 * 1024 * 1024
        assert int(row["weight_capacity_bytes"]) <= 4 * 1024 * 1024
        assert int(row["cycles"]) >= max(
            int(row["compute_cycles"]), int(row["activation_sram_cycles"]),
            int(row["weight_sram_cycles"]), int(row["dram_cycles"]))
    assert validate_artifacts(output) == "SPN_MEMORY_DSE_VALIDATION_OK"


def test_memory_artifact_validator_rejects_capacity_violation(tmp_path):
    config, workloads = _study_inputs(tmp_path)
    output = run(config, workloads, tmp_path / "report")
    path = output / "layer_memory_results.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = tuple(rows[0])
    rows[0]["activation_capacity_bytes"] = str(4 * 1024 * 1024 + 1)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="activation capacity"):
        validate_artifacts(output)


def test_memory_analyzer_refuses_to_overwrite_output(tmp_path):
    config, workloads = _study_inputs(tmp_path)
    output = run(config, workloads, tmp_path / "report")

    with pytest.raises(FileExistsError, match="already exists"):
        run(config, workloads, output)


def test_memory_analyzer_accepts_explicit_cacti_result_override(tmp_path):
    config, workloads = _study_inputs(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    valid_cacti = Path(payload["cacti_result"])
    payload["cacti_result"] = "missing-cacti-result.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    output = run(
        config, workloads, tmp_path / "report", cacti_result=valid_cacti)

    assert validate_artifacts(output) == "SPN_MEMORY_DSE_VALIDATION_OK"


def test_four_model_memory_wrapper_canonicalizes_paths_and_runs_cacti_first():
    script = (REPO_ROOT / "scripts" /
              "run_four_model_memory_dse.sh").read_text(encoding="utf-8")
    canonicalize = 'output_root="$(realpath -m "$1")"'
    cacti_run = 'scripts/run_cacti_sram.py'
    dse_run = 'scripts/run_spn_memory_dse.py'

    assert canonicalize in script
    assert cacti_run in script
    assert dse_run in script
    assert script.index(canonicalize) < script.index(cacti_run)
    assert script.index(cacti_run) < script.index(dse_run)
