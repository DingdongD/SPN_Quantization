import json
from pathlib import Path
import csv
import argparse

import numpy as np
import pytest

from scripts import run_nlspn_in_memory_gop2_benchmark as runner


def test_array_pipe_round_trip_preserves_flow():
    flow = np.arange(2 * 3 * 4, dtype=np.float32).reshape(1, 2, 3, 4)
    encoded = runner.encode_array(flow)
    decoded = runner.decode_array(encoded)
    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded, flow)


def test_raft_parity_records_hard_tolerances():
    reference = np.ones((1, 2, 228, 304), dtype=np.float32)
    result = runner.compare_raft_flows(reference, reference.copy())
    assert result["passes"] is True
    assert result["max_abs"] == 0.0
    assert result["rmse"] == 0.0
    assert result["max_abs_tolerance"] == 1e-3
    assert result["rmse_tolerance"] == 1e-4


def test_raft_parity_rejects_changed_flow():
    reference = np.zeros((1, 2, 228, 304), dtype=np.float32)
    changed = reference.copy()
    changed[..., 10, 10] = 0.01
    with pytest.raises(RuntimeError, match="parity"):
        runner.require_raft_parity(reference, changed)


def test_raft_parity_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        runner.compare_raft_flows(
            np.zeros((1, 2, 228, 304), dtype=np.float32),
            np.zeros((1, 2, 10, 10), dtype=np.float32),
        )


def test_build_worker_command_uses_old_environment_and_pythonpath(tmp_path):
    command, environment = runner.build_worker_command(
        stage="raft-parity",
        data_root=tmp_path / "data",
        scene="scene",
        checkpoint=tmp_path / "best.pt",
        args_json=tmp_path / "args.json",
        raft_weights=tmp_path / "raft.pth",
        device="cuda:0",
    )
    assert command[:6] == [
        "conda", "run", "-n", "completionformer-py37", "python",
        str(runner.WORKER_PATH)]
    assert command[command.index("--stage") + 1] == "raft-parity"
    assert str(runner.REPO_ROOT) in environment["PYTHONPATH"]
    assert str(runner.NLSPN_ROOT / "src") in environment["PYTHONPATH"]


def test_parse_worker_pipe_payload_decodes_array():
    flow = np.ones((1, 2, 3, 4), dtype=np.float32)
    line = json.dumps({
        "array": runner.encode_array(flow),
        "shape": list(flow.shape),
        "dtype": str(flow.dtype),
        "weight_sha256": "abc",
    })
    array, metadata = runner.parse_worker_pipe_payload("noise\n" + line + "\n")
    np.testing.assert_array_equal(array, flow)
    assert metadata["weight_sha256"] == "abc"


def test_validate_final_artifacts_checks_counts_digests_and_quality(tmp_path):
    metadata = {
        "complete": True,
        "artifact_count": 5,
        "frame_count": 2,
        "timed_repeats": 1,
        "timed_frames_per_path": 2,
        "checkpoint_sha256": "checkpoint",
        "raft_weight_sha256": "raft",
        "intermediate_tensor_cache": False,
    }
    summary = {
        "quality_frame_count": 2,
        "timed_repeats": 1,
        "quality": {"passes": True, "quality_ratio": 1.005},
        "speedup": 1.4,
    }
    (tmp_path / "run_metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with (tmp_path / "frame_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "path", "repeat", "frame_id", "latency_ms"])
        writer.writeheader()
        for path in ("full", "gop2"):
            writer.writerow({
                "path": path, "repeat": 0, "frame_id": 1,
                "latency_ms": 1.0})
            writer.writerow({
                "path": path, "repeat": 0, "frame_id": 2,
                "latency_ms": 1.0})
    (tmp_path / "clip_summary.csv").write_text(
        "clip,quality_ratio\n0001-0002,1.005\n")
    (tmp_path / "report.md").write_text("report\n")
    result = runner.validate_final_artifacts(
        tmp_path, expected_frame_count=2, timed_repeats=1,
        checkpoint_digest="checkpoint", raft_digest="raft")
    assert result["summary"]["speedup"] == 1.4
    assert len(result["frame_rows"]) == 4


def test_run_parity_uses_cpu_while_benchmark_device_remains_gpu(
        tmp_path, monkeypatch):
    weights = tmp_path / "raft.pth"
    weights.write_bytes(b"weights")
    digest = runner._file_sha256(weights)
    flow = np.zeros((1, 2, 228, 304), dtype=np.float32)
    observed = {}

    def fake_reference(data_root, scene, device):
        observed["reference_device"] = device
        return flow, {"weight_sha256": digest}

    def fake_worker(command, environment):
        observed["worker_device"] = command[command.index("--device") + 1]
        return json.dumps({
            "array": runner.encode_array(flow),
            "shape": list(flow.shape),
            "dtype": str(flow.dtype),
            "weight_sha256": digest,
        })

    monkeypatch.setattr(runner, "run_reference_flow", fake_reference)
    monkeypatch.setattr(runner, "_run_worker", fake_worker)
    cli = argparse.Namespace(
        data_root=tmp_path, scene="scene", device="cuda:0",
        checkpoint=tmp_path / "best.pt", args_json=tmp_path / "args.json",
        raft_weights=weights)
    parity = runner.run_parity(cli)
    assert parity["passes"] is True
    assert observed == {
        "reference_device": "cpu",
        "worker_device": "cpu",
    }
