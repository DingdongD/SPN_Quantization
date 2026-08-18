import argparse
import json

import numpy as np

from scripts import nlspn_frame_difference_cache as cache
from scripts import nlspn_cross_scene_motion_windows as motion
from scripts import run_nlspn_cross_scene_motion_worker as worker


HEIGHT = 228
WIDTH = 304


def _windows():
    return [
        {
            "scene": scene,
            "start_frame": index * 10 + 1,
            "end_frame": index * 10 + 5,
            "frame_ids": list(range(index * 10 + 1, index * 10 + 6)),
            "pair_scores": [0.1, 0.2, 0.3, 0.4],
            "motion_score": 0.25,
        }
        for index, scene in enumerate(motion.SCENES)
    ]


def _payload(frame_ids):
    gt = np.full((5, HEIGHT, WIDTH), 2.0, dtype=np.float32)
    sparse = np.zeros_like(gt)
    sparse.reshape(5, -1)[:, :500] = 2.0
    return {
        "frame_ids": np.asarray(frame_ids, dtype=np.int32),
        "rgb": np.zeros((5, 3, HEIGHT, WIDTH), dtype=np.float32),
        "sparse": sparse,
        "gt": gt,
        "valid": np.ones_like(gt, dtype=bool),
    }


def test_load_manifest_requires_six_ordered_scenes(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"windows": _windows()}), encoding="utf-8")

    assert [row["scene"] for row in worker.load_manifest(path)] == \
        list(motion.SCENES)

    reversed_path = tmp_path / "reversed.json"
    reversed_path.write_text(
        json.dumps({"windows": list(reversed(_windows()))}),
        encoding="utf-8")
    try:
        worker.load_manifest(reversed_path)
    except ValueError as error:
        assert "order" in str(error)
    else:
        raise AssertionError("reversed manifest should fail")


def test_run_batch_builds_model_once_and_infers_six_scenes(tmp_path):
    manifest = tmp_path / "manifest.json"
    windows = _windows()
    manifest.write_text(json.dumps({"windows": windows}), encoding="utf-8")
    calls = {"build": 0, "load": [], "infer": [], "artifacts": []}

    def model_builder(checkpoint, args_json, device):
        calls["build"] += 1
        return object(), {"name": "fake-nlspn"}

    def payload_loader(data_root, scene, clips, seed):
        calls["load"].append(scene)
        start, end = clips[0]
        return (_payload(range(start, end + 1)),)

    def inference_runner(engine, payload, configs):
        scene = motion.SCENES[len(calls["infer"])]
        calls["infer"].append(scene)
        predictions = {name: payload["gt"] + np.float32(0.1)
                       for name in motion.METHOD_ORDER}
        latency = [
            {"method": method, "frame_id": int(frame_id), "latency_ms": 1.0}
            for method in motion.METHOD_ORDER
            for frame_id in payload["frame_ids"]
        ]
        return {"predictions": predictions, "latency_rows": latency}

    def artifact_writer(output_dir, payload, predictions, metrics, metadata):
        calls["artifacts"].append(metadata["scene"])
        return {"complete": True, "error_vmax_m": 1.0}

    captured = {}

    def root_writer(output_root, selected, summaries, metadata):
        captured["windows"] = list(selected)
        captured["summaries"] = list(summaries)
        return {"complete": True}

    configs = {
        variant: cache.CacheConfig(variant, 2.0 / 255.0, 8)
        for variant in ("rgb_diff", "global_diff")
    }
    cli = argparse.Namespace(
        data_root=str(tmp_path / "data"), manifest=str(manifest),
        checkpoint=str(tmp_path / "best.pt"),
        args_json=str(tmp_path / "args.json"),
        formal_dir=str(tmp_path / "formal"),
        output_root=str(tmp_path / "out"), device="cpu", seed=2026)

    result = worker.run_batch(
        cli, model_builder=model_builder, payload_loader=payload_loader,
        inference_runner=inference_runner, artifact_writer=artifact_writer,
        root_writer=root_writer,
        config_loader=lambda path: (configs, "sweep"),
        engine_factory=lambda model, device: object(),
        directory_digest_fn=lambda path: {"formal": "unchanged"},
        file_digest_fn=lambda path: "digest",
        input_digest_fn=lambda *values: "input")

    assert result["complete"] is True
    assert calls["build"] == 1
    assert calls["load"] == list(motion.SCENES)
    assert calls["infer"] == list(motion.SCENES)
    assert calls["artifacts"] == list(motion.SCENES)
    assert len(captured["summaries"]) == 24


def test_worker_cli_exposes_no_raft_or_calibration_arguments():
    destinations = {action.dest for action in worker.make_parser()._actions}
    assert "raft_weights" not in destinations
    assert "calibration" not in destinations
