import numpy as np

from scripts import generate_nlspn_frame_difference_visualization as launcher
from scripts import nlspn_frame_difference_visualization as visual


def test_worker_command_uses_old_environment_and_no_raft(tmp_path):
    command, environment = launcher.build_worker_command(
        data_root=tmp_path / "data", scene="scene",
        checkpoint=tmp_path / "best.pt", args_json=tmp_path / "args.json",
        formal_dir=tmp_path / "pilot_256", output_dir=tmp_path / "out",
        device="cuda:0", seed=2026)
    assert command[:6] == [
        "conda", "run", "-n", "completionformer-py37", "python",
        str(launcher.WORKER_PATH)]
    assert all("raft" not in str(item).lower() for item in command)
    assert str(launcher.REPO_ROOT) in environment["PYTHONPATH"]
    assert str(launcher.NLSPN_ROOT / "src") in environment["PYTHONPATH"]


def test_validator_requires_exact_figures_archive_and_metrics(tmp_path):
    gt = np.full((5, 228, 304), 2.0, dtype=np.float32)
    sparse = np.zeros_like(gt)
    sparse.reshape(5, -1)[:, :500] = 2.0
    payload = {
        "frame_ids": np.arange(1, 6, dtype=np.int32),
        "rgb": np.zeros((5, 3, 228, 304), dtype=np.float32),
        "sparse": sparse,
        "gt": gt,
        "valid": np.ones_like(gt, dtype=bool),
    }
    predictions = {name: gt.copy() for name in visual.METHOD_ORDER}
    latency = [
        {"method": method, "frame_id": frame_id, "latency_ms": 1.0}
        for method in visual.METHOD_ORDER for frame_id in range(1, 6)]
    metrics = visual.collect_frame_metrics(payload, predictions, latency)
    visual.write_artifacts(tmp_path, payload, predictions, metrics, {
        "checkpoint_sha256": "checkpoint",
        "formal_sweep_sha256": "sweep",
        "frame_ids": list(range(1, 6)),
        "selected_configs": {
            "rgb_diff": {"threshold": 2.0 / 255.0,
                         "dilation_radius": 8},
            "global_diff": {"threshold": 2.0 / 255.0,
                            "dilation_radius": 8},
        },
    })

    result = launcher.validate_final_artifacts(
        tmp_path, checkpoint_digest="checkpoint", sweep_digest="sweep")
    assert result["metadata"]["complete"] is True
    assert len(result["metrics"]) == 20
    assert result["archive"]["full"].shape == (5, 228, 304)
    assert result["depth_image_size"][0] > 1000
