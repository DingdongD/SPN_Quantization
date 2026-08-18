import numpy as np
import torch

from scripts import nlspn_frame_difference_cache as cache
from scripts import run_nlspn_frame_difference_cache_pilot as runner
from scripts import run_nlspn_frame_difference_cache_worker as worker


def make_tiny_payloads():
    payloads = []
    for start in (1, 3):
        payloads.append({
            "frame_ids": np.asarray([start, start + 1], dtype=np.int32),
            "rgb": np.zeros((2, 3, 2, 3), dtype=np.float32),
            "sparse": np.zeros((2, 2, 3), dtype=np.float32),
            "gt": np.ones((2, 2, 3), dtype=np.float32),
            "valid": np.ones((2, 2, 3), dtype=bool),
        })
    return tuple(payloads)


class RecordingCalibrationEngine:
    def reset(self):
        pass

    def infer_full(self, rgb, sparse):
        return cache.FrameDifferenceResult(
            torch.full((2, 3), 1.1), 5.0, "FULL", "full", {})

    def infer_i(self, rgb, sparse, local_index):
        return cache.FrameDifferenceResult(
            torch.ones(2, 3), 1.0, "I", "i_frame", {})

    def infer_p(self, rgb, sparse, local_index, config):
        value = 1.0 + (
            0.0 if config.threshold is None else float(config.threshold))
        metrics = {
            "stable_fraction": 0.75,
            "changed_fraction": 0.25,
            "photometric_changed_fraction": 0.2,
            "sparse_changed_fraction": 0.1,
            "out_of_bounds_fraction": 0.0,
            "dx": 0.0,
            "dy": 0.0,
        }
        return cache.FrameDifferenceResult(
            torch.full((2, 3), value), 1.0, "P", config.variant, metrics)


def test_worker_command_uses_legacy_environment_without_raft(tmp_path):
    command, environment = runner.build_worker_command(
        data_root=tmp_path / "data", scene="scene",
        checkpoint=tmp_path / "best.pt", args_json=tmp_path / "args.json",
        output_dir=tmp_path / "out", device="cuda:0", clips="1:2,3:4",
        calibration_clip_count=1, warmup_repeats=1, timed_repeats=5,
        seed=2026)
    assert command[:6] == [
        "conda", "run", "-n", "completionformer-py37", "python",
        str(runner.WORKER_PATH)]
    assert "--raft-weights" not in command
    assert all("raft" not in str(item).lower() for item in command)
    assert command[command.index("--calibration-clip-count") + 1] == "1"
    assert str(runner.REPO_ROOT) in environment["PYTHONPATH"]
    assert str(runner.NLSPN_ROOT / "src") in environment["PYTHONPATH"]


def test_validator_checks_exact_counts_and_checkpoint(tmp_path):
    result = worker.execute_pilot(
        RecordingCalibrationEngine(), make_tiny_payloads(),
        calibration_clip_count=1, warmup_repeats=0, timed_repeats=2)
    worker.write_final_artifacts(tmp_path, {
        "frame_count": 4,
        "timed_repeats": 2,
        "warmup_repeats": 0,
        "checkpoint_sha256": "checkpoint",
        "intermediate_tensor_cache": False,
        "raft_constructed": False,
    }, result)

    validated = runner.validate_final_artifacts(
        tmp_path, expected_frame_count=4, timed_repeats=2,
        checkpoint_digest="checkpoint")

    assert len(validated["frame_rows"]) == 32
    assert len(validated["sweep_rows"]) == 24
    assert set(validated["summary"]["paths"]) == {
        "full", "zero_flow", "rgb_diff", "global_diff"}
    for variant in ("full", "zero_flow", "rgb_diff", "global_diff"):
        rows = [row for row in validated["frame_rows"]
                if row["variant"] == variant and row["repeat"] == "0"]
        assert len(rows) == 4
