import copy
import csv

import numpy as np
import pytest

from scripts import nlspn_cross_scene_motion_windows as motion
from scripts import nlspn_frame_difference_visualization as visual
from scripts import nlspn_stratified_motion_sampling as sampling
from scripts import nlspn_stratified_motion_statistics as statistics


def _window(scene="room3", stratum="medium",
            frame_ids=(101, 102, 103, 104, 105),
            pair_scores=(0.1, 0.2, 0.3, 0.4)):
    start, end = frame_ids[0], frame_ids[-1]
    return {
        "selection_seed": 2026,
        "scene": scene,
        "stratum": stratum,
        "window_id": "{}/{}/{:04d}_{:04d}".format(
            scene, stratum, start, end),
        "stratum_rank_start": 10,
        "stratum_rank_end": 20,
        "stratum_score_min": 0.05,
        "stratum_score_max": 0.30,
        "start_frame": start,
        "end_frame": end,
        "frame_ids": list(frame_ids),
        "pair_scores": list(pair_scores),
        "motion_score": sum(pair_scores) / 4.0,
    }


def _five_method_metrics(window, raft_ratio=1.2):
    rows = []
    for method_index, method in enumerate(visual.RAFT_METHOD_ORDER):
        for local_index, frame_id in enumerate(window["frame_ids"]):
            full_rmse = 1.0 + local_index * 0.1
            rmse = full_rmse if method == "full" else \
                full_rmse * (raft_ratio if method == "raft_gop2" else 1.1)
            rows.append({
                "window_id": window["window_id"],
                "scene": window["scene"],
                "stratum": window["stratum"],
                "method": method,
                "frame_id": frame_id,
                "frame_kind": "FULL" if method == "full" else
                    ("P" if local_index in (1, 3) else "I"),
                "rmse": rmse,
                "mae": rmse / 2.0,
                "valid_pixels": 100 + local_index,
                "latency_ms": 10.0 if method == "full" else
                    2.0 + method_index,
            })
    return rows


def test_derive_p_frame_metrics_uses_adjacent_i_to_p_scores():
    window = _window()
    result = statistics.derive_p_frame_metrics(
        [window], _five_method_metrics(window), exact_window_count=False)
    assert len(result) == 10
    raft = [row for row in result if row["method"] == "raft_gop2"]
    assert [row["frame_id"] for row in raft] == [102, 104]
    assert [row["adjacent_motion_score"] for row in raft] == \
        pytest.approx([0.1, 0.3])
    assert raft[0]["excess_rmse"] == pytest.approx(
        raft[0]["rmse"] - raft[0]["full_rmse"])
    assert raft[0]["rmse_ratio"] == pytest.approx(1.2)
    assert raft[0]["speedup"] == pytest.approx(10.0 / 6.0)


def test_derive_p_frame_metrics_uses_inclusive_one_percent_gate():
    window = _window()
    passing = statistics.derive_p_frame_metrics(
        [window], _five_method_metrics(window, raft_ratio=1.01), False)
    failing = statistics.derive_p_frame_metrics(
        [window], _five_method_metrics(window, raft_ratio=1.0100001), False)
    assert all(row["passes_1pct"] for row in passing
               if row["method"] == "raft_gop2")
    assert not any(row["passes_1pct"] for row in failing
                   if row["method"] == "raft_gop2")


@pytest.mark.parametrize("mutate, message", [
    (lambda rows: rows.pop(), "25"),
    (lambda rows: rows[6].update(frame_kind="I"), "frame kind"),
    (lambda rows: rows[0].update(rmse=0.0), "Full RMSE"),
    (lambda rows: rows[0].update(latency_ms=0.0), "latency"),
    (lambda rows: rows[0].update(rmse=float("nan")), "finite"),
])
def test_derive_p_frame_metrics_rejects_corrupt_window_metrics(
        mutate, message):
    window = _window()
    rows = _five_method_metrics(window)
    mutate(rows)
    with pytest.raises(ValueError, match=message):
        statistics.derive_p_frame_metrics([window], rows, False)


def test_derive_p_frame_metrics_rejects_duplicate_keys():
    window = _window()
    rows = _five_method_metrics(window)
    rows[-1] = copy.deepcopy(rows[-2])
    with pytest.raises(ValueError, match="unique"):
        statistics.derive_p_frame_metrics([window], rows, False)


def _complete_windows():
    rows = []
    for scene_index, scene in enumerate(motion.SCENES):
        base = 1 + scene_index * 1000
        for stratum_index, stratum in enumerate(sampling.STRATA):
            for window_index in range(5):
                start = base + stratum_index * 200 + window_index * 10
                score = 0.01 + stratum_index * 0.1 + window_index * 0.005
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
                    "pair_scores": [score] * 4,
                    "motion_score": score,
                })
    return sampling.validate_selected_windows(rows, 2026)


def _complete_p_rows():
    rows = []
    for window in _complete_windows():
        scene_index = motion.SCENES.index(window["scene"])
        stratum_index = sampling.STRATA.index(window["stratum"])
        for local_index in (1, 3):
            full_rmse = 0.2 + scene_index * 0.03 + local_index * 0.01
            for method_index, method in enumerate(visual.RAFT_METHOD_ORDER):
                excess = 0.0 if method == "full" else \
                    0.01 * method_index * (stratum_index + 1) + \
                    window["motion_score"] * 0.1
                rmse = full_rmse + excess
                latency = 10.0 if method == "full" else 2.0 + method_index
                rows.append({
                    "scene": window["scene"],
                    "stratum": window["stratum"],
                    "window_id": window["window_id"],
                    "start_frame": window["start_frame"],
                    "frame_id": window["frame_ids"][local_index],
                    "local_index": local_index,
                    "method": method,
                    "adjacent_motion_score":
                        window["pair_scores"][local_index - 1],
                    "window_motion_score": window["motion_score"],
                    "rmse": rmse, "mae": rmse / 2.0,
                    "valid_pixels": 100 + scene_index * 10,
                    "latency_ms": latency,
                    "full_rmse": full_rmse,
                    "full_latency_ms": 10.0,
                    "excess_rmse": excess,
                    "rmse_ratio": rmse / full_rmse,
                    "passes_1pct": rmse / full_rmse <= 1.01,
                    "speedup": 10.0 / latency,
                })
    return rows


def test_stratified_summary_has_fifteen_rows_and_deterministic_intervals():
    rows = _complete_p_rows()
    first = statistics.build_stratified_summary(
        rows, bootstrap_replicates=20, seed=2026)
    second = statistics.build_stratified_summary(
        rows, bootstrap_replicates=20, seed=2026)
    assert first == second
    assert len(first) == 15
    raft_high = next(row for row in first
                     if row["stratum"] == "high" and
                     row["method"] == "raft_gop2")
    assert raft_high["scene_count"] == 6
    assert raft_high["frame_count"] == 60
    assert raft_high["excess_rmse_ci_low"] <= \
        raft_high["scene_macro_excess_rmse_mean"] <= \
        raft_high["excess_rmse_ci_high"]
    assert 0.0 <= raft_high["scene_macro_pass_rate"] <= 1.0


def test_stratified_summary_separates_macro_and_pixel_pooling():
    rows = [row for row in _complete_p_rows()
            if row["stratum"] == "low" and row["method"] == "raft_gop2"]
    result = statistics.build_stratified_summary(
        rows, bootstrap_replicates=10, seed=2026, exact_geometry=False)
    assert len(result) == 1
    assert result[0]["scene_count"] == 6
    assert result[0]["pooled_rmse"] > 0
    assert result[0]["scene_macro_rmse_mean"] > 0


def test_correlation_summary_has_thirteen_finite_rows():
    result = statistics.build_correlation_summary(_complete_p_rows())
    assert len(result) == 13
    assert [(row["method"], row["target"]) for row in result[:4]] == [
        ("full", "rmse"),
        ("zero_flow", "rmse"),
        ("zero_flow", "excess_rmse"),
        ("zero_flow", "rmse_ratio")]
    assert all(np.isfinite(row["pearson_r"]) for row in result)
    assert all(np.isfinite(row["spearman_rho"]) for row in result)


def _csv_count(path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return len(list(csv.DictReader(stream)))


def test_write_root_artifacts_emits_exact_files_and_counts(tmp_path):
    windows = _complete_windows()
    p_rows = _complete_p_rows()
    summaries = statistics.build_stratified_summary(
        p_rows, bootstrap_replicates=10, seed=2026)
    correlations = statistics.build_correlation_summary(p_rows)
    completed = statistics.write_root_artifacts(
        tmp_path, windows, p_rows, summaries, correlations,
        {"complete": False, "bootstrap_replicates": 10})
    assert set(path.name for path in tmp_path.iterdir() if path.is_file()) == \
        set(statistics.ROOT_ARTIFACTS)
    assert _csv_count(tmp_path / "p_frame_metrics.csv") == 900
    assert _csv_count(tmp_path / "stratified_summary.csv") == 15
    assert _csv_count(tmp_path / "correlation_summary.csv") == 13
    assert completed["selected_window_count"] == 90
    assert completed["p_frame_row_count"] == 900
    assert (tmp_path / "motion_error_scatter.png").stat().st_size > 0
    assert (tmp_path / "stratified_error_boxplot.png").stat().st_size > 0
