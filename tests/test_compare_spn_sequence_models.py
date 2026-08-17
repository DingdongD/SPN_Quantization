from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import compare_spn_sequence_models as comparison
from scripts import spn_sequence_io as sequence_io


def test_worker_specs_use_native_conda_environments():
    specs = comparison.default_worker_specs(
        Path("/repo/scripts/run_spn_sequence_worker.py"))
    assert specs["dyspn"]["environment"] == "pointkan"
    assert specs["nlspn"]["environment"] == "completionformer-py37"
    assert specs["completionformer"]["environment"] == "completionformer-py37"
    assert "/workspace/external_depth_completion_models/DySPN" in (
        specs["dyspn"]["pythonpath"])
    assert "/workspace/CompletionFormer/src/model/deformconv" in (
        specs["completionformer"]["pythonpath"])


def test_build_worker_command_is_explicit_and_deterministic(tmp_path):
    spec = comparison.default_worker_specs(Path("/repo/worker.py"))["dyspn"]
    command, env = comparison.build_worker_command(
        "dyspn",
        spec,
        Path("/canonical"),
        tmp_path / "predictions.npz",
        "cuda:0",
    )
    assert command[:5] == ["conda", "run", "-n", "pointkan", "python"]
    assert command[-2:] == ["--device", "cuda:0"]
    assert env["PYTHONPATH"].startswith(spec["pythonpath"])


def test_run_or_reuse_worker_accepts_only_digest_matching_cache(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"weights")
    spec = comparison.default_worker_specs(Path("/repo/worker.py"))["dyspn"]
    spec["checkpoint"] = str(checkpoint)
    model_dir = tmp_path / "dyspn"
    result_path = model_dir / "predictions.npz"
    sequence_io.write_worker_result(
        result_path,
        "dyspn",
        np.arange(1, 6),
        np.ones((5, 228, 304), dtype=np.float32),
        "a" * 64,
        sequence_io.file_sha256(checkpoint),
        {"iteration": 6},
        1.0,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("matching cache should not launch a worker")

    result, reused = comparison.run_or_reuse_worker(
        "dyspn", spec, Path("/canonical"), model_dir, "cuda:0",
        np.arange(1, 6), "a" * 64, runner=fail_if_called)
    assert reused is True
    assert result["metadata"]["iteration"] == 6
    assert (model_dir / "worker.log").is_file()


def test_run_or_reuse_worker_records_subprocess_failure(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"weights")
    spec = comparison.default_worker_specs(Path("/repo/worker.py"))["dyspn"]
    spec["checkpoint"] = str(checkpoint)

    def failed_runner(*args, **kwargs):
        return SimpleNamespace(returncode=7, stdout="worker out", stderr="worker err")

    model_dir = tmp_path / "dyspn"
    with pytest.raises(RuntimeError, match="exit code 7"):
        comparison.run_or_reuse_worker(
            "dyspn", spec, Path("/canonical"), model_dir, "cuda:0",
            np.arange(1, 6), "a" * 64, runner=failed_runner)
    log = (model_dir / "worker.log").read_text(encoding="utf-8")
    assert "worker out" in log
    assert "worker err" in log


def test_expected_artifacts_lists_twenty_panels_and_combined_outputs(tmp_path):
    paths = comparison.expected_artifacts(tmp_path, range(1, 6))
    panels = [
        path for path in paths
        if path.name.startswith("frame_") and path.suffix == ".png"]
    frame_npz = [
        path for path in paths
        if path.name.startswith("frame_") and path.suffix == ".npz"]
    worker_logs = [path for path in paths if path.name == "worker.log"]
    worker_results = [
        path for path in paths if path.name == "predictions.npz"]
    assert len(panels) == 20
    assert len(frame_npz) == 20
    assert len(worker_logs) == 3
    assert len(worker_results) == 3
    assert tmp_path / "four_model_depth_comparison.png" in paths
    assert tmp_path / "four_model_error_comparison.png" in paths
    assert (
        tmp_path / "four_model_temporal_comparison_unregistered.png") in paths
    assert tmp_path / "four_model_frame_metrics.csv" in paths
    assert tmp_path / "four_model_temporal_metrics.csv" in paths
    assert tmp_path / "run_metadata.json" in paths


def test_collect_metrics_has_twenty_frame_and_sixteen_temporal_rows():
    frame_ids = np.arange(1, 6)
    gt = np.ones((5, 2, 3), dtype=np.float32)
    valid = np.ones_like(gt, dtype=bool)
    predictions = {name: gt.copy() for name in comparison.MODEL_ORDER}
    frame_rows, temporal_rows, temporal_maps = comparison.collect_metrics(
        frame_ids, gt, valid, predictions)
    assert len(frame_rows) == 20
    assert len(temporal_rows) == 16
    assert set(temporal_maps) == set(comparison.MODEL_ORDER)
    assert {row["model"] for row in frame_rows} == set(comparison.MODEL_ORDER)
    assert {row["alignment"] for row in temporal_rows} == {"unregistered"}


def test_parser_uses_approved_five_frame_paths():
    parsed = comparison.make_parser().parse_args([])
    assert parsed.canonical_dir.endswith(
        "cspn_predictions/BeachApartmentInterior_My_ir/frames_0001_0005")
    assert parsed.output_dir.endswith(
        "spn_model_comparison/BeachApartmentInterior_My_ir/frames_0001_0005")
    assert parsed.device == "cuda:0"
    assert parsed.force is False


def test_load_model_predictions_reuses_cspn_and_collects_workers(tmp_path):
    cspn = np.ones((5, 228, 304), dtype=np.float32)
    canonical = {
        "frame_ids": np.arange(1, 6),
        "input_digest": "a" * 64,
        "cspn_pred_clamped": cspn,
    }
    calls = []

    def fake_worker(model, spec, canonical_dir, model_dir, device,
                    frame_ids, input_digest, force=False):
        calls.append(model)
        index = comparison.MODEL_ORDER.index(model)
        pred = np.full_like(cspn, float(index + 1))
        return {
            "pred_clamped": pred,
            "metadata": {"runtime_seconds": float(index)},
            "checkpoint_digest": model * 8,
        }, False

    specs = {name: {} for name in comparison.EXTERNAL_MODELS}
    predictions, worker_info = comparison.load_model_predictions(
        canonical,
        Path("/canonical"),
        tmp_path,
        "cuda:0",
        specs,
        worker_executor=fake_worker,
    )
    assert calls == list(comparison.EXTERNAL_MODELS)
    np.testing.assert_array_equal(predictions["cspn"], cspn)
    assert set(predictions) == set(comparison.MODEL_ORDER)
    assert set(worker_info) == set(comparison.EXTERNAL_MODELS)
