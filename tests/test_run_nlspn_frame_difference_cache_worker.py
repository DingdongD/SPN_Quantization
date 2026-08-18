import csv
import json
import numpy as np
import pytest
import torch

from scripts import nlspn_frame_difference_cache as cache
from scripts import run_nlspn_frame_difference_cache_worker as worker


def make_tiny_payloads(starts=(1, 3)):
    payloads = []
    for start in starts:
        frame_ids = np.asarray([start, start + 1], dtype=np.int32)
        payloads.append({
            "frame_ids": frame_ids,
            "rgb": np.zeros((2, 3, 2, 3), dtype=np.float32),
            "sparse": np.zeros((2, 2, 3), dtype=np.float32),
            "gt": np.ones((2, 2, 3), dtype=np.float32),
            "valid": np.ones((2, 2, 3), dtype=bool),
        })
    return tuple(payloads)


class RecordingCalibrationEngine:
    def __init__(self):
        self.reset_calls = 0
        self.config_calls = []
        self.full_calls = 0

    def reset(self):
        self.reset_calls += 1

    def infer_i(self, rgb, sparse, local_index):
        return cache.FrameDifferenceResult(
            torch.ones(2, 3), 1.0, "I", "i_frame", {})

    def infer_full(self, rgb, sparse):
        self.full_calls += 1
        return cache.FrameDifferenceResult(
            torch.full((2, 3), 1.1), 5.0, "FULL", "full", {})

    def infer_p(self, rgb, sparse, local_index, config):
        self.config_calls.append(config)
        value = 1.0 + (
            0.0 if config.threshold is None else float(config.threshold))
        metrics = {
            "stable_fraction": 0.75,
            "changed_fraction": 0.25,
            "photometric_changed_fraction": 0.2,
            "sparse_changed_fraction": 0.1,
            "out_of_bounds_fraction": (
                0.05 if config.variant == "global_diff" else 0.0),
            "dx": 0.0,
            "dy": 0.0,
        }
        return cache.FrameDifferenceResult(
            torch.full((2, 3), value), 1.0, "P", config.variant,
            metrics)


def test_calibration_sweep_runs_complete_grid_and_resets_boundaries():
    engine = RecordingCalibrationEngine()
    payloads = make_tiny_payloads()
    full_predictions = [np.ones((2, 3), dtype=np.float32)] * 4

    rows = worker.run_calibration_sweep(
        engine, payloads, full_predictions)

    assert len(rows) == 24
    assert {row["variant"] for row in rows} == {
        "rgb_diff", "global_diff"}
    assert all(row["valid_pixels"] == 24 for row in rows)
    assert engine.reset_calls == 24 * len(payloads)
    assert len(engine.config_calls) == 24 * len(payloads)
    assert set(engine.config_calls) == set(
        cache.candidate_configs("rgb_diff") +
        cache.candidate_configs("global_diff"))
    assert {row["frame_id_min"] for row in rows} == {1}
    assert {row["frame_id_max"] for row in rows} == {4}


def test_timed_variant_resets_each_clip_and_keeps_first_predictions():
    engine = RecordingCalibrationEngine()
    payloads = make_tiny_payloads()
    config = cache.CacheConfig("rgb_diff", 4.0 / 255.0, 8)

    records, predictions = worker.run_timed_variant(
        engine, payloads, config, repeats=2)

    assert len(records) == 8
    assert len(predictions) == 4
    assert engine.reset_calls == 2 * len(payloads)
    assert [row["kind"] for row in records[:4]] == ["I", "P", "I", "P"]
    assert [row["repeat"] for row in records] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert all(row["variant"] == "rgb_diff" for row in records)
    p_rows = [row for row in records if row["kind"] == "P"]
    assert all(row["threshold"] == config.threshold for row in p_rows)
    assert all(row["dilation_radius"] == 8 for row in p_rows)
    assert all(row["stable_fraction"] == 0.75 for row in p_rows)
    assert all("dx" in row and "dy" in row for row in p_rows)


def test_timed_full_records_one_row_per_repeat_and_frame():
    engine = RecordingCalibrationEngine()
    records, predictions = worker.run_timed_full(
        engine, make_tiny_payloads(), repeats=2)
    assert len(records) == 8
    assert len(predictions) == 4
    assert all(row["variant"] == "full" for row in records)
    assert all(row["kind"] == "FULL" for row in records)
    assert engine.full_calls == 8


def test_execute_pilot_selects_on_calibration_and_summarizes_four_paths():
    engine = RecordingCalibrationEngine()
    result = worker.execute_pilot(
        engine, make_tiny_payloads(), calibration_clip_count=1,
        warmup_repeats=0, timed_repeats=2)

    assert set(result["summary"]["paths"]) == {
        "full", "zero_flow", "rgb_diff", "global_diff"}
    assert result["selected_configs"]["rgb_diff"] == cache.CacheConfig(
        "rgb_diff", 2.0 / 255.0, 8)
    assert result["selected_configs"]["global_diff"] == cache.CacheConfig(
        "global_diff", 2.0 / 255.0, 8)
    assert len(result["sweep_rows"]) == 24
    assert sum(bool(row["selected"]) for row in result["sweep_rows"]) == 2
    assert len(result["frame_rows"]) == 4 * 2 * 4
    for variant in ("zero_flow", "rgb_diff", "global_diff"):
        summary = result["summary"]["paths"][variant]
        assert set(summary["quality"]) == {"calibration", "heldout", "all"}
        assert summary["latency"]["overall"]["count"] == 8
        assert summary["latency"]["i"]["count"] == 4
        assert summary["latency"]["p"]["count"] == 4
        assert summary["speedup"] > 0.0
    assert result["summary"]["external_raft_reference"] == {
        "quality_ratio": 1.0069332411236265,
        "speedup": 0.5267047643822169,
    }


def test_final_artifacts_are_exact_atomic_report_set(tmp_path):
    result = worker.execute_pilot(
        RecordingCalibrationEngine(), make_tiny_payloads(),
        calibration_clip_count=1, warmup_repeats=0, timed_repeats=2)
    completed = worker.write_final_artifacts(
        tmp_path, {"complete": False, "checkpoint_sha256": "abc"}, result)

    assert completed["complete"] is True
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "clip_summary.csv",
        "frame_metrics.csv",
        "report.md",
        "run_metadata.json",
        "summary.json",
        "threshold_sweep.csv",
    ]
    with (tmp_path / "run_metadata.json").open() as stream:
        metadata = json.load(stream)
    assert metadata["artifact_count"] == 6
    with (tmp_path / "threshold_sweep.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 24


def test_final_writer_rejects_tensor_cache_artifact(tmp_path):
    (tmp_path / "depth.npy").write_bytes(b"not-a-real-array")
    with pytest.raises(RuntimeError, match="tensor cache"):
        worker.write_final_artifacts(
            tmp_path, {}, {"summary": {}, "sweep_rows": [{}],
                           "frame_rows": [{}], "clip_rows": [{}]})


def test_worker_cli_has_no_raft_argument():
    parser = worker.make_parser()
    destinations = {action.dest for action in parser._actions}
    assert "raft_weights" not in destinations
    cli = parser.parse_args([
        "--checkpoint", "/weights/best.pt",
        "--args-json", "/weights/args.json",
        "--output-dir", "/tmp/output",
    ])
    assert cli.calibration_clip_count == 4
    assert cli.warmup_repeats == 1
    assert cli.timed_repeats == 5
