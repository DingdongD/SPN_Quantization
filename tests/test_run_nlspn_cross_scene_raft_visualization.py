import csv
import itertools
import json

import pytest

from scripts import nlspn_cross_scene_motion_windows as motion
from scripts import nlspn_frame_difference_visualization as visual
from scripts import nlspn_temporal_residual as residual
from scripts import run_nlspn_cross_scene_raft_visualization as launcher


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


def _write_selected(path, windows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "scene", "start_frame", "end_frame", "frame_ids",
            "pair_scores", "motion_score"))
        writer.writeheader()
        for window in windows:
            row = dict(window)
            row["frame_ids"] = json.dumps(
                row["frame_ids"], separators=(",", ":"))
            row["pair_scores"] = json.dumps(
                row["pair_scores"], separators=(",", ":"))
            writer.writerow(row)


def _metrics(scene, frame_ids):
    return [
        {
            "scene": scene,
            "method": method,
            "frame_id": frame_id,
            "rmse": 1.0 + method_index * 0.001,
            "mae": 0.5,
            "valid_pixels": 100,
            "latency_ms": 1.0,
        }
        for method_index, method in enumerate(motion.RAFT_METHOD_ORDER)
        for frame_id in frame_ids
    ]


def test_load_recorded_windows_preserves_exact_six_windows(tmp_path):
    expected = _windows()
    _write_selected(tmp_path / "selected_windows.csv", expected)

    assert launcher.load_recorded_windows(tmp_path) == expected


def test_worker_command_requires_official_raft_weights(tmp_path):
    command, environment = launcher.build_worker_command(
        data_root=tmp_path / "data", manifest=tmp_path / "manifest.json",
        checkpoint=tmp_path / "best.pt", args_json=tmp_path / "args.json",
        formal_dir=tmp_path / "formal", output_root=tmp_path / "staging",
        raft_weights=tmp_path / "raft-small.pt", device="cuda:0", seed=2026)

    assert command[:6] == [
        "conda", "run", "-n", "completionformer-py37", "python",
        str(launcher.WORKER_PATH)]
    assert command[command.index("--raft-weights") + 1] == \
        str(tmp_path / "raft-small.pt")
    assert command.count("--manifest") == 1
    assert str(launcher.REPO_ROOT) in environment["PYTHONPATH"]


def _make_valid_tree(root, raft_digest="raft-digest"):
    windows = _windows()
    summaries = []
    metrics_by_scene = {}
    for window in windows:
        scene = window["scene"]
        metrics = _metrics(scene, window["frame_ids"])
        metrics_by_scene[scene] = metrics
        summaries.extend(motion.build_scene_summary(scene, metrics))
        scene_dir = root / scene
        scene_dir.mkdir(parents=True)
        for name in visual.FINAL_ARTIFACTS:
            (scene_dir / name).write_bytes(b"artifact")
    motion.write_root_artifacts(root, windows, summaries, {
        "checkpoint_sha256": "checkpoint",
        "formal_sweep_sha256": "sweep",
        "method_order": list(visual.RAFT_METHOD_ORDER),
        "nlspn_model_load_count": 1,
        "raft_model_load_count": 1,
        "raft_weight_sha256": raft_digest,
        "raft_flow_updates": 12,
        "raft_flow_direction": "current_to_previous",
        "selected_configs": {
            "rgb_diff": {"threshold": 2.0 / 255.0, "dilation_radius": 8},
            "global_diff": {"threshold": 2.0 / 255.0,
                            "dilation_radius": 8},
        },
    })
    return windows, metrics_by_scene


def test_validate_raft_tree_recomputes_thirty_rows(tmp_path):
    windows, metrics_by_scene = _make_valid_tree(tmp_path)
    selected_digest = residual.file_sha256(tmp_path / "selected_windows.csv")

    def scene_validator(scene_dir, checkpoint_digest, sweep_digest):
        scene = scene_dir.name
        window = next(row for row in windows if row["scene"] == scene)
        return {
            "method_order": visual.RAFT_METHOD_ORDER,
            "metadata": {
                "scene": scene,
                "frame_ids": window["frame_ids"],
                "motion_score": window["motion_score"],
                "method_order": list(visual.RAFT_METHOD_ORDER),
                "nlspn_model_load_count": 1,
                "raft_model_load_count": 1,
                "raft_weight_sha256": "raft-digest",
                "raft_flow_updates": 12,
                "raft_flow_direction": "current_to_previous",
            },
            "metrics": metrics_by_scene[scene],
        }

    result = launcher.validate_raft_final_tree(
        tmp_path, windows, "checkpoint", "sweep", "raft-digest",
        selected_digest, scene_validator=scene_validator)

    assert result["scene_count"] == 6
    assert result["summary_row_count"] == 30
    assert result["nlspn_model_load_count"] == 1
    assert result["raft_model_load_count"] == 1


def test_validate_raft_tree_rejects_wrong_flow_direction(tmp_path):
    windows, metrics_by_scene = _make_valid_tree(tmp_path)
    selected_digest = residual.file_sha256(tmp_path / "selected_windows.csv")

    def scene_validator(scene_dir, checkpoint_digest, sweep_digest):
        scene = scene_dir.name
        window = next(row for row in windows if row["scene"] == scene)
        return {
            "method_order": visual.RAFT_METHOD_ORDER,
            "metadata": {
                "scene": scene,
                "frame_ids": window["frame_ids"],
                "motion_score": window["motion_score"],
                "method_order": list(visual.RAFT_METHOD_ORDER),
                "nlspn_model_load_count": 1,
                "raft_model_load_count": 1,
                "raft_weight_sha256": "raft-digest",
                "raft_flow_updates": 12,
                "raft_flow_direction": "previous_to_current",
            },
            "metrics": metrics_by_scene[scene],
        }

    with pytest.raises(RuntimeError, match="RAFT metadata"):
        launcher.validate_raft_final_tree(
            tmp_path, windows, "checkpoint", "sweep", "raft-digest",
            selected_digest, scene_validator=scene_validator)
