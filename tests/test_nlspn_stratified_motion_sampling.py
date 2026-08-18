import copy

import numpy as np
import pytest

from scripts import nlspn_cross_scene_motion_windows as motion
from scripts import nlspn_stratified_motion_sampling as sampling


def _candidates(scene="room3", count=45):
    rows = []
    for index in range(count):
        start = 1 + index * 10
        score = float(index + 1) / 100.0
        rows.append({
            "scene": scene,
            "start_frame": start,
            "end_frame": start + 4,
            "frame_ids": list(range(start, start + 5)),
            "pair_scores": [score] * 4,
            "motion_score": score,
        })
    return rows


def _selected_for_six_scenes():
    result = []
    for scene_index, scene in enumerate(motion.SCENES):
        base = 1 + scene_index * 1000
        for stratum_index, stratum in enumerate(sampling.STRATA):
            rank_start = stratum_index * 15
            rank_end = rank_start + 15
            score_min = 0.01 + stratum_index * 0.1
            score_max = score_min + 0.09
            for index in range(5):
                start = base + stratum_index * 200 + index * 10
                score = score_min + index * 0.01
                result.append({
                    "selection_seed": 2026,
                    "scene": scene,
                    "stratum": stratum,
                    "window_id": "{}/{}/{:04d}_{:04d}".format(
                        scene, stratum, start, start + 4),
                    "stratum_rank_start": rank_start,
                    "stratum_rank_end": rank_end,
                    "stratum_score_min": score_min,
                    "stratum_score_max": score_max,
                    "start_frame": start,
                    "end_frame": start + 4,
                    "frame_ids": list(range(start, start + 5)),
                    "pair_scores": [score] * 4,
                    "motion_score": score,
                })
    return result


def test_rank_tertiles_are_balanced_and_stable():
    groups = sampling.rank_tertiles(_candidates(count=18))
    assert tuple(groups) == sampling.STRATA
    assert [len(groups[name]) for name in sampling.STRATA] == [6, 6, 6]
    assert [row["motion_score"] for row in groups["low"]] == \
        pytest.approx([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    assert groups["medium"][0]["stratum_rank_start"] == 6
    assert groups["high"][-1]["stratum_rank_end"] == 18


def test_rank_tertiles_use_start_frame_to_break_score_ties():
    rows = _candidates(count=6)
    for row in rows:
        row["motion_score"] = 0.25
        row["pair_scores"] = [0.25] * 4
    groups = sampling.rank_tertiles(list(reversed(rows)))
    ordered = [row["start_frame"] for name in sampling.STRATA
               for row in groups[name]]
    assert ordered == sorted(ordered)


def test_select_scene_windows_is_deterministic_and_non_overlapping():
    first = sampling.select_scene_windows(
        "room3", _candidates(), count_per_stratum=5, seed=2026)
    second = sampling.select_scene_windows(
        "room3", _candidates(), count_per_stratum=5, seed=2026)
    assert first == second
    assert len(first) == 15
    assert [sum(row["stratum"] == name for row in first)
            for name in sampling.STRATA] == [5, 5, 5]
    used = set()
    for row in first:
        assert used.isdisjoint(row["frame_ids"])
        used.update(row["frame_ids"])


def test_select_scene_windows_rejects_insufficient_medium_windows():
    rows = _candidates(count=18)
    for row in rows[6:12]:
        row["frame_ids"] = [101, 102, 103, 104, 105]
        row["start_frame"], row["end_frame"] = 101, 105
    with pytest.raises(ValueError, match="medium.*five non-overlapping"):
        sampling.select_scene_windows(
            "room3", rows, count_per_stratum=5, seed=2026)


def test_validate_manifest_requires_exact_six_by_three_by_five():
    rows = _selected_for_six_scenes()
    result = sampling.validate_selected_windows(rows, seed=2026)
    assert len(result) == 90
    assert [row["scene"] for row in result[:15]] == ["bedroom_ir"] * 15
    assert result[0]["window_id"].startswith("bedroom_ir/low/")
    assert result[-1]["selection_seed"] == 2026


def test_validate_manifest_rejects_shared_frames_across_strata():
    rows = _selected_for_six_scenes()
    rows[5]["frame_ids"] = list(rows[0]["frame_ids"])
    rows[5]["start_frame"] = rows[0]["start_frame"]
    rows[5]["end_frame"] = rows[0]["end_frame"]
    rows[5]["window_id"] = "bedroom_ir/medium/{:04d}_{:04d}".format(
        rows[0]["start_frame"], rows[0]["end_frame"])
    with pytest.raises(ValueError, match="shared frame"):
        sampling.validate_selected_windows(rows, seed=2026)


@pytest.mark.parametrize("mutation, message", [
    (lambda rows: rows.pop(), "90"),
    (lambda rows: rows[0].update(scene="unknown"), "scene"),
    (lambda rows: rows[0].update(pair_scores=[0.1] * 3), "pair"),
    (lambda rows: rows[0].update(motion_score=np.nan), "finite"),
    (lambda rows: rows[0].update(motion_score=0.7), "mean"),
])
def test_validate_manifest_rejects_corruption(mutation, message):
    rows = _selected_for_six_scenes()
    mutation(rows)
    with pytest.raises(ValueError, match=message):
        sampling.validate_selected_windows(rows, seed=2026)


def test_manifest_and_csv_round_trip_and_digest_are_deterministic(tmp_path):
    rows = _selected_for_six_scenes()
    manifest = tmp_path / "manifest.json"
    selected = tmp_path / "selected.csv"
    sampling.write_manifest_json(manifest, rows, seed=2026)
    sampling.write_selected_windows_csv(selected, rows, seed=2026)
    loaded_manifest = sampling.load_manifest_json(manifest)
    loaded_csv = sampling.load_selected_windows_csv(selected, seed=2026)
    assert loaded_manifest == loaded_csv
    assert sampling.canonical_manifest_sha256(rows, 2026) == \
        sampling.canonical_manifest_sha256(copy.deepcopy(rows), 2026)
