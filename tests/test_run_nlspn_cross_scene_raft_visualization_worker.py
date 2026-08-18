import argparse
import json

import numpy as np
import pytest

from scripts import nlspn_cross_scene_motion_windows as motion
from scripts import nlspn_frame_difference_cache as cache
from scripts import nlspn_frame_difference_visualization as visual
from scripts import raft_small_compat
from scripts import run_nlspn_cross_scene_raft_visualization_worker as worker


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


def _cli(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"windows": _windows()}), encoding="utf-8")
    return argparse.Namespace(
        data_root=str(tmp_path / "data"), manifest=str(manifest),
        checkpoint=str(tmp_path / "best.pt"),
        args_json=str(tmp_path / "args.json"),
        formal_dir=str(tmp_path / "formal"),
        output_root=str(tmp_path / "staging"),
        raft_weights=str(tmp_path / "raft-small.pt"),
        device="cpu", seed=2026)


def _configs():
    return {
        variant: cache.CacheConfig(variant, 2.0 / 255.0, 8)
        for variant in ("rgb_diff", "global_diff")
    }


def test_run_batch_loads_each_model_once_and_emits_thirty_rows(tmp_path):
    cli = _cli(tmp_path)
    calls = {
        "bundle": 0, "cache_engine": 0, "raft_engine": 0,
        "scenes": [], "artifacts": [],
    }
    captured = {}

    def bundle_builder(checkpoint, args_json, raft_weights, device):
        calls["bundle"] += 1
        return object(), object(), {"name": "frozen-models"}

    def payload_loader(data_root, scene, clips, seed):
        calls["scenes"].append(scene)
        start, end = clips[0]
        return (_payload(range(start, end + 1)),)

    def cache_runner(engine, payload, configs):
        predictions = {
            method: payload["gt"] + np.float32(0.1)
            for method in visual.BASE_METHOD_ORDER
        }
        latency = [
            {"method": method, "frame_id": int(frame_id), "latency_ms": 1.0}
            for method in visual.BASE_METHOD_ORDER
            for frame_id in payload["frame_ids"]
        ]
        return {"predictions": predictions, "latency_rows": latency}

    def raft_runner(engine, payload):
        return {
            "prediction": payload["gt"] + np.float32(0.1),
            "latency_rows": [
                {"method": "raft_gop2", "frame_id": int(frame_id),
                 "latency_ms": 2.0}
                for frame_id in payload["frame_ids"]
            ],
        }

    def artifact_writer(output_dir, payload, predictions, metrics, metadata):
        calls["artifacts"].append(metadata["scene"])
        assert tuple(predictions) == visual.RAFT_METHOD_ORDER
        assert len(metrics) == 25
        assert metadata["raft_flow_direction"] == "current_to_previous"
        assert metadata["raft_flow_updates"] == 12
        return {"complete": True}

    def root_writer(output_root, windows, summaries, metadata):
        captured["summaries"] = list(summaries)
        captured["metadata"] = dict(metadata)
        return {"complete": True}

    def file_digest(path):
        if str(path) == cli.raft_weights:
            return raft_small_compat.EXPECTED_WEIGHT_SHA256
        return "digest"

    result = worker.run_batch(
        cli, bundle_builder=bundle_builder,
        payload_loader=payload_loader,
        cache_engine_factory=lambda model, device: calls.__setitem__(
            "cache_engine", calls["cache_engine"] + 1) or object(),
        raft_engine_factory=lambda model, raft, device: calls.__setitem__(
            "raft_engine", calls["raft_engine"] + 1) or object(),
        cache_runner=cache_runner, raft_runner=raft_runner,
        artifact_writer=artifact_writer, root_writer=root_writer,
        config_loader=lambda path: (_configs(), "sweep"),
        directory_digest_fn=lambda path: {"formal": "unchanged"},
        file_digest_fn=file_digest,
        input_digest_fn=lambda *values: "input")

    assert result["complete"] is True
    assert result["summary_row_count"] == 30
    assert calls["bundle"] == 1
    assert calls["cache_engine"] == 1
    assert calls["raft_engine"] == 1
    assert calls["scenes"] == list(motion.SCENES)
    assert calls["artifacts"] == list(motion.SCENES)
    assert len(captured["summaries"]) == 30
    assert captured["metadata"]["nlspn_model_load_count"] == 1
    assert captured["metadata"]["raft_model_load_count"] == 1
    assert captured["metadata"]["method_order"] == \
        list(visual.RAFT_METHOD_ORDER)


def test_run_batch_rejects_non_official_raft_digest_before_model_load(tmp_path):
    cli = _cli(tmp_path)
    calls = {"bundle": 0}

    def bundle_builder(*args):
        calls["bundle"] += 1
        raise AssertionError("bundle must not load")

    with pytest.raises(RuntimeError, match="official RAFT-Small"):
        worker.run_batch(
            cli, bundle_builder=bundle_builder,
            directory_digest_fn=lambda path: {},
            file_digest_fn=lambda path: "wrong-digest")
    assert calls["bundle"] == 0


def test_worker_cli_requires_raft_and_exposes_no_fallback():
    parser = worker.make_parser()
    destinations = {action.dest for action in parser._actions}
    assert "raft_weights" in destinations
    assert "fallback" not in destinations
    assert "calibration" not in destinations
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--manifest", "/tmp/manifest.json",
            "--checkpoint", "/tmp/best.pt",
            "--args-json", "/tmp/args.json",
            "--formal-dir", "/tmp/formal",
            "--output-root", "/tmp/staging",
        ])


def test_activate_cuda_device_sets_current_device_for_legacy_extensions(
        monkeypatch):
    calls = []
    monkeypatch.setattr(worker.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        worker.torch.cuda, "set_device", lambda device: calls.append(device))

    worker.activate_cuda_device("cuda:2")

    assert len(calls) == 1
    assert calls[0] == worker.torch.device("cuda:2")


def test_activate_cuda_device_does_not_touch_cuda_for_cpu(monkeypatch):
    calls = []
    monkeypatch.setattr(
        worker.torch.cuda, "set_device", lambda device: calls.append(device))

    worker.activate_cuda_device("cpu")

    assert calls == []
