import json
from pathlib import Path

import pytest

from scripts import summarize_rk3588_compressed_propagation as summary


def _run(path: Path, process: int, latency: float = 100.0) -> Path:
    payload = {
        "schema_version": 1,
        "model": "dyspn",
        "candidate": "drop_s4+bridge50",
        "precision": "w8a8",
        "model_path": "model.rknn",
        "model_sha256": "model-hash",
        "safe_onnx_sha256": "onnx-hash",
        "input_sha256": ["rgb-hash", "depth-hash"],
        "api_version": "2.5.0",
        "driver_version": "0.9.8",
        "warmup_iterations": 30,
        "timed_iterations_per_round": 50,
        "rounds": 3,
        "process_index": process,
        "raw_latency_ms": [latency + process] * 150,
        "raw_grid_sample_cpu_ms": [10.0] * 150,
        "raw_grid_sample_call_count": [6] * 150,
        "output": {"elements": 4, "minimum": 1.0,
                   "maximum": 2.0, "mean": 1.5},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_summarize_preserves_five_processes(tmp_path: Path):
    paths = [_run(tmp_path / ("run%d.json" % index), index)
             for index in range(5)]

    result = summary.summarize_processes(paths)

    assert result["process_count"] == 5
    assert result["precision"] == "w8a8"
    assert result["candidate"] == "drop_s4+bridge50"
    assert result["samples_per_process"] == 150
    assert result["process_mean_latency_ms"] == [100, 101, 102, 103, 104]
    assert result["latency_ms"]["mean_of_process_means"] == 102
    assert result["grid_sample"]["calls_per_forward"] == 6
    assert result["grid_sample"]["mean_cpu_ms"] == 10


@pytest.mark.parametrize("mutation", (
    {"rounds": 2},
    {"model_sha256": "different"},
    {"input_sha256": ["different", "depth-hash"]},
    {"api_version": "different"},
    {"precision": "fp16"},
    {"candidate": "different"},
    {"safe_onnx_sha256": "different"},
))
def test_summarize_rejects_protocol_or_hash_mismatch(
        tmp_path: Path, mutation):
    paths = [_run(tmp_path / ("run%d.json" % index), index)
             for index in range(5)]
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload.update(mutation)
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        summary.summarize_processes(paths)


def test_summarize_rejects_missing_or_nonfinite_samples(tmp_path: Path):
    paths = [_run(tmp_path / ("run%d.json" % index), index)
             for index in range(4)]
    with pytest.raises(ValueError, match="five independent"):
        summary.summarize_processes(paths)

    paths.append(_run(tmp_path / "run4.json", 4))
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload["raw_latency_ms"][0] = float("nan")
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        summary.summarize_processes(paths)


def test_enrich_board_process_binds_artifact_identity_and_output_schema():
    raw = {
        "model": "cspn",
        "precision": "w8a8",
        "output": {
            "elements": 4,
            "all_finite": True,
            "finite_minimum": 1.0,
            "finite_maximum": 2.0,
            "finite_mean": 1.5,
        },
    }

    result = summary.enrich_board_process(
        raw, candidate="drop_s4", precision="w8a8",
        model_sha256="model-hash", safe_onnx_sha256="onnx-hash",
        input_sha256=["input-hash"], process_index=0)

    assert result["candidate"] == "drop_s4"
    assert result["precision"] == "w8a8"
    assert result["model_sha256"] == "model-hash"
    assert result["safe_onnx_sha256"] == "onnx-hash"
    assert result["input_sha256"] == ["input-hash"]
    assert result["process_index"] == 0
    assert result["output"]["minimum"] == 1.0
    assert result["output"]["maximum"] == 2.0
    assert result["output"]["mean"] == 1.5

    with pytest.raises(ValueError, match="precision"):
        summary.enrich_board_process(
            raw, candidate="drop_s4", precision="fp16",
            model_sha256="model-hash", safe_onnx_sha256="onnx-hash",
            input_sha256=["input-hash"], process_index=0)
