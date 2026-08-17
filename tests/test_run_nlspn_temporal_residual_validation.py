import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from scripts import run_nlspn_temporal_residual_validation as runner


def test_cli_help_bootstraps_repository_imports(tmp_path):
    script = (
        Path(__file__).resolve().parents[1] / "scripts" /
        "run_nlspn_temporal_residual_validation.py")
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(tmp_path), text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "Validate causal NLSPN" in completed.stdout


def test_prepare_clip_uses_common_fixed_sparse_mask(monkeypatch, tmp_path):
    frame_ids = (1, 2)
    rgb = np.zeros((3, 228, 304), dtype=np.float32)
    gt = np.ones((228, 304), dtype=np.float32)
    valid = np.ones((228, 304), dtype=bool)
    monkeypatch.setattr(
        runner,
        "load_preprocessed_frame",
        lambda data_root, scene, frame_id: (rgb, gt, valid),
    )
    output = tmp_path / "clip_0001_0002.npz"
    payload = runner.prepare_clip(
        Path("/data"), "scene", frame_ids, output, seed=2026)
    assert output.is_file()
    assert payload["rgb"].shape == (2, 3, 228, 304)
    masks = payload["sparse"] > 0
    np.testing.assert_array_equal(masks[0], masks[1])
    assert masks[0].sum() == 500


class FakeRaft(torch.nn.Module):
    def forward(self, image1, image2):
        batch, _, height, width = image1.shape
        flow = torch.zeros((batch, 2, height, width), device=image1.device)
        flow[:, 0] = 2.0
        return [flow]


def test_predict_backward_flow_is_current_to_previous_and_crops_padding():
    rgb = np.zeros((3, 3, 228, 304), dtype=np.float32)
    flow, seconds = runner.predict_backward_flow(
        FakeRaft(), lambda current, previous: (current, previous),
        rgb, torch.device("cpu"), batch_size=2)
    assert seconds >= 0.0
    assert flow.shape == (2, 2, 228, 304)
    np.testing.assert_allclose(flow[:, 0], 2.0)


def test_finalize_metadata_requires_every_artifact(tmp_path):
    metadata = {"complete": False}
    required = (tmp_path / "summary.json", tmp_path / "pair_metrics.csv")
    required[0].write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete"):
        runner.finalize_metadata(
            tmp_path / "run_metadata.json", metadata, required)
    required[1].write_text("model,pair\n", encoding="utf-8")
    runner.finalize_metadata(
        tmp_path / "run_metadata.json", metadata, required)
    saved = json.loads(
        (tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert saved["complete"] is True


def test_worker_command_uses_completionformer_environment(tmp_path):
    command, env = runner.build_worker_command(
        "baseline", tmp_path / "clip.npz", tmp_path / "baseline.npz",
        checkpoint=Path("/weights/best.pt"),
        args_json=Path("/weights/args.json"), device="cuda:0")
    assert command[:4] == [
        "conda", "run", "-n", "completionformer-py37"]
    assert "--stage" in command and "baseline" in command
    assert "NLSPN_ECCV20/src/model/deformconv" in env["PYTHONPATH"]


def test_smoke_frames_accepts_exactly_two_increasing_ids():
    args = runner.make_parser().parse_args(
        ["--smoke-frames", "1", "2"])
    assert runner.resolve_clips(args) == ((1, 2),)
    for values in (("1",), ("2", "1"), ("1", "2", "3")):
        args = runner.make_parser().parse_args(
            ["--smoke-frames"] + list(values))
        with pytest.raises(ValueError, match="two increasing"):
            runner.resolve_clips(args)


def test_main_leaves_metadata_incomplete_when_pilot_fails(
        monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner,
        "run_pilot",
        lambda args, metadata: (_ for _ in ()).throw(
            RuntimeError("worker failed")),
    )
    with pytest.raises(RuntimeError, match="worker failed"):
        runner.main(["--output-dir", str(tmp_path)])
    metadata = json.loads(
        (tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["complete"] is False
