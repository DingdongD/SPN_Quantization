import csv
import json

import pytest

from scripts.run_spn_array_dse import run, validate_artifacts
from spn_quant.hardware.workload import LayerWorkload, WorkloadProfile


MODELS = ("cspn", "dyspn", "nlspn", "completionformer")


def _write_config(path):
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


def _write_profiles(root):
    result = {}
    for index, model in enumerate(MODELS):
        layers = (
            LayerWorkload.linear(
                "linear", [1, 128], [1, 32 + index], execution_index=0),
            LayerWorkload.irregular(
                "%s::propagation" % model, "propagation",
                [1, 1, 8, 8], [1, 1, 8, 8],
                macs=8 * 8 * 8, execution_index=1,
                iterations=1, neighborhood_size=8),
        )
        profile = WorkloadProfile(
            model_name=model, source_config="source.json",
            checkpoint="/%s.pt" % model, sample_index=386,
            input_shapes=(("arg0", (1, 4, 8, 8)),), layers=layers,
            metadata=(("checkpoint_sha256", "%064d" % index),))
        path = root / (model + ".json")
        profile.write_json(path)
        result[model] = path
    return result


def test_analyzer_writes_complete_artifact_set(tmp_path):
    config = _write_config(tmp_path / "config.json")
    workloads = _write_profiles(tmp_path)
    output = run(config, workloads, tmp_path / "report")
    assert {path.name for path in output.iterdir()} == {
        "candidate_ranking.csv", "layer_results.csv", "metadata.json",
        "model_summary.csv", "pareto_frontier.csv",
        "shape_utilization.png", "workload_coverage.csv"
    }
    with (output / "model_summary.csv").open(newline="") as handle:
        summary = tuple(csv.DictReader(handle))
    with (output / "layer_results.csv").open(newline="") as handle:
        layers = tuple(csv.DictReader(handle))
    assert len(summary) == 4 * 3
    assert set(row["model"] for row in summary) == set(MODELS)
    assert len(layers) == 4 * 3
    metadata = json.loads(
        (output / "metadata.json").read_text(encoding="utf-8"))
    assert set(metadata["input_sha256"]) == set(MODELS)
    assert validate_artifacts(output) == "SPN_ARRAY_DSE_VALIDATION_OK"


def test_analyzer_refuses_to_overwrite_output(tmp_path):
    config = _write_config(tmp_path / "config.json")
    workloads = _write_profiles(tmp_path)
    output = run(config, workloads, tmp_path / "report")
    with pytest.raises(FileExistsError, match="already exists"):
        run(config, workloads, output)
