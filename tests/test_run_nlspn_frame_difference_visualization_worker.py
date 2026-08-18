import numpy as np
import torch

from scripts import nlspn_frame_difference_cache as cache
from scripts import nlspn_frame_difference_visualization as visual
from scripts import nlspn_in_memory_gop2 as online
from scripts import run_nlspn_frame_difference_visualization_worker as worker


HEIGHT = 228
WIDTH = 304


def make_payload():
    sparse = np.zeros((5, HEIGHT, WIDTH), dtype=np.float32)
    sparse.reshape(5, -1)[:, :500] = 2.0
    return {
        "frame_ids": np.arange(1, 6, dtype=np.int32),
        "rgb": np.zeros((5, 3, HEIGHT, WIDTH), dtype=np.float32),
        "sparse": sparse,
        "gt": np.full((5, HEIGHT, WIDTH), 2.0, dtype=np.float32),
        "valid": np.ones((5, HEIGHT, WIDTH), dtype=bool),
    }


class RecordingEngine:
    def __init__(self):
        self.reset_calls = 0
        self.full_calls = 0
        self.i_calls = 0
        self.p_configs = []

    def reset(self):
        self.reset_calls += 1

    def infer_full(self, rgb, sparse):
        self.full_calls += 1
        return online.FrameResult(torch.full((HEIGHT, WIDTH), 2.0), 5.0, "FULL")

    def infer_i(self, rgb, sparse, local_index):
        self.i_calls += 1
        return cache.FrameDifferenceResult(
            torch.full((HEIGHT, WIDTH), 2.0), 4.0, "I", "i_frame", {})

    def infer_p(self, rgb, sparse, local_index, config):
        self.p_configs.append(config)
        return cache.FrameDifferenceResult(
            torch.full((HEIGHT, WIDTH), 2.0), 2.0, "P", config.variant, {})


def selected_configs():
    return {
        "rgb_diff": cache.CacheConfig("rgb_diff", 2.0 / 255.0, 8),
        "global_diff": cache.CacheConfig("global_diff", 2.0 / 255.0, 8),
    }


def test_run_inference_uses_full_and_fixed_ipipi_schedule():
    engine = RecordingEngine()
    result = worker.run_inference(engine, make_payload(), selected_configs())
    assert set(result["predictions"]) == set(visual.METHOD_ORDER)
    assert all(value.shape == (5, HEIGHT, WIDTH)
               for value in result["predictions"].values())
    assert len(result["latency_rows"]) == 20
    assert engine.reset_calls == 4
    assert engine.full_calls == 5
    assert engine.i_calls == 9
    assert [config.variant for config in engine.p_configs] == [
        "zero_flow", "zero_flow", "rgb_diff", "rgb_diff",
        "global_diff", "global_diff"]


def test_worker_cli_exposes_no_raft_argument():
    parser = worker.make_parser()
    destinations = {action.dest for action in parser._actions}
    assert "raft_weights" not in destinations
    cli = parser.parse_args([
        "--checkpoint", "/weights/best.pt",
        "--args-json", "/weights/args.json",
        "--formal-dir", "/results/pilot_256",
        "--output-dir", "/results/visualization",
    ])
    assert cli.seed == 2026
    assert cli.device == "cuda:0"
