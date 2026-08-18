import argparse

import numpy as np
import pytest

from scripts import nlspn_cross_scene_motion_windows as motion
from scripts import nlspn_frame_difference_cache as cache
from scripts import nlspn_frame_difference_visualization as visual
from scripts import nlspn_stratified_motion_sampling as sampling
from scripts import raft_small_compat
from scripts import run_nlspn_stratified_motion_worker as worker


def _windows():
    rows = []
    for scene_index, scene in enumerate(motion.SCENES):
        for stratum_index, stratum in enumerate(sampling.STRATA):
            for index in range(5):
                start = 1 + scene_index * 1000 + stratum_index * 200 + index * 10
                score = 0.01 + stratum_index * 0.1 + index * 0.005
                rows.append({
                    "selection_seed": 2026, "scene": scene,
                    "stratum": stratum,
                    "window_id": "{}/{}/{:04d}_{:04d}".format(
                        scene, stratum, start, start + 4),
                    "stratum_rank_start": stratum_index * 15,
                    "stratum_rank_end": (stratum_index + 1) * 15,
                    "stratum_score_min": 0.01 + stratum_index * 0.1,
                    "stratum_score_max": 0.09 + stratum_index * 0.1,
                    "start_frame": start, "end_frame": start + 4,
                    "frame_ids": list(range(start, start + 5)),
                    "pair_scores": [score] * 4, "motion_score": score,
                })
    return sampling.validate_selected_windows(rows, 2026)


def _payload(frame_ids):
    gt = np.full((5, 228, 304), 2.0, dtype=np.float32)
    sparse = np.zeros_like(gt)
    sparse.reshape(5, -1)[:, :500] = 2.0
    return {
        "frame_ids": np.asarray(frame_ids, dtype=np.int32),
        "rgb": np.zeros((5, 3, 228, 304), dtype=np.float32),
        "sparse": sparse, "gt": gt,
        "valid": np.ones_like(gt, dtype=bool),
    }


def _cli(tmp_path):
    manifest = tmp_path / "manifest.json"
    sampling.write_manifest_json(manifest, _windows(), 2026)
    return argparse.Namespace(
        data_root=str(tmp_path / "data"), manifest=str(manifest),
        checkpoint=str(tmp_path / "best.pt"),
        args_json=str(tmp_path / "args.json"),
        formal_dir=str(tmp_path / "formal"),
        output_root=str(tmp_path / "staging"),
        raft_weights=str(tmp_path / "raft.pt"), device="cpu", seed=2026)


def _configs():
    return {variant: cache.CacheConfig(variant, 2.0 / 255.0, 8)
            for variant in ("rgb_diff", "global_diff")}


def test_run_batch_loads_models_once_and_processes_ninety_windows(tmp_path):
    cli = _cli(tmp_path)
    calls = {"bundle": 0, "cache": 0, "raft": 0, "windows": 0}
    paths = []
    metadata_rows = []

    def bundle_builder(*args):
        calls["bundle"] += 1
        return object(), object(), {"architecture": "NLSPN"}

    def payload_loader(data_root, scene, clips, seed):
        start, end = clips[0]
        return (_payload(range(start, end + 1)),)

    def cache_runner(engine, payload, configs):
        predictions = {method: payload["gt"] + np.float32(0.1)
                       for method in visual.BASE_METHOD_ORDER}
        latency = [{"method": method, "frame_id": int(frame_id),
                    "latency_ms": 1.0}
                   for method in visual.BASE_METHOD_ORDER
                   for frame_id in payload["frame_ids"]]
        return {"predictions": predictions, "latency_rows": latency}

    def raft_runner(engine, payload):
        return {"prediction": payload["gt"] + np.float32(0.1),
                "latency_rows": [
                    {"method": "raft_gop2", "frame_id": int(frame_id),
                     "latency_ms": 2.0} for frame_id in payload["frame_ids"]]}

    def artifact_writer(path, payload, predictions, metrics, metadata):
        calls["windows"] += 1
        paths.append(path)
        metadata_rows.append(metadata)
        assert tuple(predictions) == visual.RAFT_METHOD_ORDER
        assert len(metrics) == 25
        return {"complete": True}

    def digest(path):
        return raft_small_compat.EXPECTED_WEIGHT_SHA256 \
            if str(path) == cli.raft_weights else "digest"

    result = worker.run_batch(
        cli, bundle_builder=bundle_builder, payload_loader=payload_loader,
        cache_engine_factory=lambda model, device: calls.__setitem__(
            "cache", calls["cache"] + 1) or object(),
        raft_engine_factory=lambda model, raft, device: calls.__setitem__(
            "raft", calls["raft"] + 1) or object(),
        cache_runner=cache_runner, raft_runner=raft_runner,
        artifact_writer=artifact_writer,
        config_loader=lambda path: (_configs(), "sweep"),
        directory_digest_fn=lambda path: {"formal": "same"},
        file_digest_fn=digest, input_digest_fn=lambda *args: "input")

    assert result["complete"] is True
    assert result["window_count"] == 90
    assert result["frame_metric_row_count"] == 2250
    assert calls == {"bundle": 1, "cache": 1, "raft": 1, "windows": 90}
    assert paths[0] == tmp_path / "staging/bedroom_ir/low/0001_0005"
    assert metadata_rows[-1]["raft_flow_direction"] == "current_to_previous"
    assert metadata_rows[-1]["raft_flow_updates"] == 12
    assert metadata_rows[-1]["nlspn_model_load_count"] == 1
    assert metadata_rows[-1]["raft_model_load_count"] == 1


def test_run_batch_rejects_non_official_raft_before_model_load(tmp_path):
    cli = _cli(tmp_path)
    calls = []
    with pytest.raises(RuntimeError, match="official RAFT-Small"):
        worker.run_batch(
            cli, bundle_builder=lambda *args: calls.append(1),
            directory_digest_fn=lambda path: {},
            file_digest_fn=lambda path: "wrong")
    assert calls == []


def test_worker_cli_requires_raft_and_exposes_no_fallback():
    parser = worker.make_parser()
    destinations = {action.dest for action in parser._actions}
    assert "raft_weights" in destinations
    assert "fallback" not in destinations
    assert "calibration" not in destinations
    with pytest.raises(SystemExit):
        parser.parse_args([])
