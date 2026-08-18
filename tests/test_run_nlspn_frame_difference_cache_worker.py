import numpy as np
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

    def reset(self):
        self.reset_calls += 1

    def infer_i(self, rgb, sparse, local_index):
        return cache.FrameDifferenceResult(
            torch.ones(2, 3), 1.0, "I", "i_frame", {})

    def infer_p(self, rgb, sparse, local_index, config):
        self.config_calls.append(config)
        value = 1.0 + float(config.threshold)
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
