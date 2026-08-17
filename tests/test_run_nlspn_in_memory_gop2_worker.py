import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import nlspn_in_memory_gop2 as online
from scripts import nlspn_temporal_residual as residual
from scripts import run_nlspn_in_memory_gop2_worker as worker


HEIGHT = 228
WIDTH = 304


def fake_frame_loader(data_root, scene, frame_id):
    rgb = np.full((3, HEIGHT, WIDTH), frame_id / 10.0, dtype=np.float32)
    gt = np.full((HEIGHT, WIDTH), 2.0 + frame_id / 100.0, dtype=np.float32)
    valid = np.ones((HEIGHT, WIDTH), dtype=bool)
    if frame_id >= 3:
        valid.reshape(-1)[:10] = False
        gt[~valid] = 0.0
    return rgb, gt, valid


def test_parse_clips_defaults_to_pilot_and_accepts_override():
    assert worker.parse_clips(None) == residual.PILOT_CLIPS
    assert worker.parse_clips("1:2,10:12") == ((1, 2), (10, 12))
    with pytest.raises(ValueError):
        worker.parse_clips("2:1")


def test_load_in_memory_clips_builds_valid_payloads_without_tensor_files(
        tmp_path):
    payloads = worker.load_in_memory_clips(
        tmp_path, "scene", ((1, 2), (3, 4)), seed=2026,
        frame_loader=fake_frame_loader)
    assert len(payloads) == 2
    assert payloads[0]["frame_ids"].tolist() == [1, 2]
    assert payloads[1]["frame_ids"].tolist() == [3, 4]
    for payload in payloads:
        assert payload["rgb"].shape == (2, 3, HEIGHT, WIDTH)
        assert np.all(np.count_nonzero(
            payload["sparse"], axis=(1, 2)) == 500)
    assert list(tmp_path.iterdir()) == []


class RecordingEngine:
    def __init__(self):
        self.reset_calls = 0
        self.full_calls = 0
        self.i_calls = 0
        self.p_calls = 0

    def reset(self):
        self.reset_calls += 1

    @staticmethod
    def _prediction(rgb):
        value = float(np.asarray(rgb).reshape(-1)[0])
        return torch.full((HEIGHT, WIDTH), value + 1.0)

    def infer_full(self, rgb, sparse):
        self.full_calls += 1
        return online.FrameResult(self._prediction(rgb), 10.0, "FULL")

    def infer_i(self, rgb, sparse, local_index):
        self.i_calls += 1
        return online.FrameResult(self._prediction(rgb), 10.0, "I")

    def infer_p(self, rgb, sparse, local_index):
        self.p_calls += 1
        return online.FrameResult(self._prediction(rgb), 2.0, "P")


def make_small_payloads():
    payloads = []
    for start in (1, 3):
        frame_ids = np.asarray([start, start + 1], dtype=np.int32)
        rgb = np.stack([
            np.full((3, HEIGHT, WIDTH), frame_id, dtype=np.float32)
            for frame_id in frame_ids])
        sparse = np.zeros((2, HEIGHT, WIDTH), dtype=np.float32)
        sparse.reshape(2, -1)[:, :500] = 2.0
        gt = rgb[:, 0].copy()
        valid = np.ones_like(gt, dtype=bool)
        payloads.append({
            "frame_ids": frame_ids,
            "rgb": rgb,
            "sparse": sparse,
            "gt": gt,
            "valid": valid,
        })
    return tuple(payloads)


def test_run_timed_path_resets_each_clip_and_keeps_first_predictions():
    engine = RecordingEngine()
    records, predictions = worker.run_timed_path(
        engine, make_small_payloads(), path="gop2", repeats=2)
    assert len(records) == 8
    assert len(predictions) == 4
    assert engine.reset_calls == 4
    assert engine.i_calls == 4
    assert engine.p_calls == 4
    assert [record["kind"] for record in records[:4]] == [
        "I", "P", "I", "P"]
    assert [record["repeat"] for record in records] == [
        0, 0, 0, 0, 1, 1, 1, 1]


def test_warmup_is_not_included_in_timed_records():
    engine = RecordingEngine()
    worker.run_warmup(engine, make_small_payloads(), path="full", repeats=1)
    records, _ = worker.run_timed_path(
        engine, make_small_payloads(), path="full", repeats=2)
    assert len(records) == 8
    assert engine.full_calls == 12
    assert all(record["latency_ms"] == 10.0 for record in records)


def test_execute_benchmark_collects_latency_quality_and_warmup():
    result = worker.execute_benchmark(
        RecordingEngine(), make_small_payloads(),
        warmup_repeats=1, timed_repeats=2)
    assert len(result["frame_rows"]) == 16
    assert len(result["clip_rows"]) == 2
    assert result["summary"]["latency"]["full"]["count"] == 8
    assert result["summary"]["latency"]["gop2"]["count"] == 8
    assert result["summary"]["latency"]["i"]["count"] == 4
    assert result["summary"]["latency"]["p"]["count"] == 4
    assert result["summary"]["speedup"] == pytest.approx(10.0 / 6.0)
    assert result["summary"]["quality"]["quality_ratio"] == 1.0
    assert result["summary"]["quality"]["passes"] is True
    assert result["summary"]["warmup"]["full_seconds"] >= 0.0
    quality_rows = [
        row for row in result["frame_rows"]
        if row["path"] == "gop2" and row["repeat"] == 0]
    assert len(quality_rows) == 4
    assert all(row["valid_pixels"] == HEIGHT * WIDTH
               for row in quality_rows)


def test_write_final_artifacts_writes_only_approved_files(tmp_path):
    metadata = {"complete": False, "timed_repeats": 2}
    summary = {"quality": {"passes": True}, "speedup": 1.5}
    frame_rows = [{
        "path": "full", "repeat": 0, "clip": "0001-0002",
        "frame_id": 1, "kind": "FULL", "latency_ms": 10.0,
    }]
    clip_rows = [{
        "clip": "0001-0002", "quality_ratio": 1.0,
    }]
    worker.write_final_artifacts(
        tmp_path, metadata, summary, frame_rows, clip_rows)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "clip_summary.csv",
        "frame_metrics.csv",
        "report.md",
        "run_metadata.json",
        "summary.json",
    ]
    with (tmp_path / "run_metadata.json").open() as stream:
        completed = json.load(stream)
    assert completed["complete"] is True
    assert not list(tmp_path.glob("*.npz"))
    assert not list(tmp_path.glob("*.npy"))
