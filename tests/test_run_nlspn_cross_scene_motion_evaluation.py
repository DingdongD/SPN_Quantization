import itertools

from scripts import nlspn_cross_scene_motion_windows as motion
from scripts import nlspn_frame_difference_visualization as visual
from scripts import run_nlspn_cross_scene_motion_evaluation as launcher


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


def _metrics(scene, frame_ids):
    return [
        {
            "scene": scene,
            "method": method,
            "frame_id": frame_id,
            "rmse": 1.0,
            "mae": 0.5,
            "valid_pixels": 100,
            "latency_ms": 1.0,
        }
        for method, frame_id in itertools.product(motion.METHOD_ORDER, frame_ids)
    ]


def test_worker_command_uses_one_legacy_process_and_no_raft(tmp_path):
    command, environment = launcher.build_worker_command(
        data_root=tmp_path / "data", manifest=tmp_path / "manifest.json",
        checkpoint=tmp_path / "best.pt", args_json=tmp_path / "args.json",
        formal_dir=tmp_path / "formal", output_root=tmp_path / "out",
        device="cuda:0", seed=2026)

    assert command[:6] == [
        "conda", "run", "-n", "completionformer-py37", "python",
        str(launcher.WORKER_PATH)]
    assert command.count("--manifest") == 1
    assert all("raft" not in str(item).lower() for item in command)
    assert str(launcher.REPO_ROOT) in environment["PYTHONPATH"]


def test_validate_final_tree_recomputes_all_scene_summaries(tmp_path):
    windows = _windows()
    summaries = []
    metrics_by_scene = {}
    for window in windows:
        scene = window["scene"]
        metrics = _metrics(scene, window["frame_ids"])
        metrics_by_scene[scene] = metrics
        summaries.extend(motion.build_scene_summary(scene, metrics))
        scene_dir = tmp_path / scene
        scene_dir.mkdir()
        for name in visual.FINAL_ARTIFACTS:
            (scene_dir / name).write_bytes(b"artifact")
    motion.write_root_artifacts(tmp_path, windows, summaries, {
        "checkpoint_sha256": "checkpoint",
        "formal_sweep_sha256": "sweep",
        "model_load_count": 1,
        "selected_configs": {
            "rgb_diff": {"threshold": 2.0 / 255.0, "dilation_radius": 8},
            "global_diff": {"threshold": 2.0 / 255.0, "dilation_radius": 8},
        },
    })

    def scene_validator(scene_dir, checkpoint_digest, sweep_digest):
        scene = scene_dir.name
        window = next(row for row in windows if row["scene"] == scene)
        assert checkpoint_digest == "checkpoint"
        assert sweep_digest == "sweep"
        return {
            "metadata": {
                "scene": scene,
                "frame_ids": window["frame_ids"],
                "motion_score": window["motion_score"],
            },
            "metrics": metrics_by_scene[scene],
        }

    result = launcher.validate_final_tree(
        tmp_path, windows, "checkpoint", "sweep",
        scene_validator=scene_validator)

    assert result["scene_count"] == 6
    assert result["summary_row_count"] == 24
    assert result["model_load_count"] == 1
