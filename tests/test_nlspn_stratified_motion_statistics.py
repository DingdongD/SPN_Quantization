import copy

import pytest

from scripts import nlspn_frame_difference_visualization as visual
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
