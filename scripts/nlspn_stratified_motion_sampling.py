"""Deterministic per-scene motion-tertile sampling for NLSPN evaluation."""

from pathlib import Path
import csv
import hashlib
import json
import os

import numpy as np

from scripts import nlspn_cross_scene_motion_windows as motion


STRATA = ("low", "medium", "high")
WINDOWS_PER_STRATUM = 5
SELECTION_SEED = 2026
SELECTED_WINDOW_FIELDS = (
    "selection_seed", "scene", "stratum", "window_id",
    "stratum_rank_start", "stratum_rank_end",
    "stratum_score_min", "stratum_score_max", "start_frame", "end_frame",
    "frame_ids", "pair_scores", "motion_score")


def _finite_float(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be finite".format(label))
    if not np.isfinite(result):
        raise ValueError("{} must be finite".format(label))
    return result


def _validate_candidate(row, require_stratum=False, seed=SELECTION_SEED):
    if not isinstance(row, dict):
        raise ValueError("window rows must be objects")
    value = dict(row)
    scene = str(value.get("scene", ""))
    if scene not in motion.SCENES:
        raise ValueError("unapproved scene")
    try:
        ids = [int(item) for item in value.get("frame_ids", ())]
    except (TypeError, ValueError):
        raise ValueError("frame IDs must be integers")
    if (len(ids) != 5 or any(item <= 0 for item in ids) or
            any(right != left + 1 for left, right in zip(ids, ids[1:]))):
        raise ValueError("frame IDs must contain five consecutive values")
    try:
        scores = [float(item) for item in value.get("pair_scores", ())]
    except (TypeError, ValueError):
        raise ValueError("pair scores must be finite")
    if len(scores) != 4:
        raise ValueError("pair scores must contain four values")
    if not np.isfinite(np.asarray(scores, dtype=np.float64)).all():
        raise ValueError("pair scores must be finite")
    score = _finite_float(value.get("motion_score"), "motion score")
    if not np.isclose(score, np.mean(scores, dtype=np.float64),
                      rtol=0.0, atol=1e-15):
        raise ValueError("motion score must equal pair-score mean")
    start = int(value.get("start_frame", -1))
    end = int(value.get("end_frame", -1))
    if start != ids[0] or end != ids[-1]:
        raise ValueError("window bounds must match frame IDs")
    result = {
        "scene": scene, "start_frame": start, "end_frame": end,
        "frame_ids": ids, "pair_scores": scores, "motion_score": score,
    }
    if require_stratum:
        stratum = str(value.get("stratum", ""))
        if stratum not in STRATA:
            raise ValueError("unapproved stratum")
        rank_start = int(value.get("stratum_rank_start", -1))
        rank_end = int(value.get("stratum_rank_end", -1))
        score_min = _finite_float(
            value.get("stratum_score_min"), "stratum score minimum")
        score_max = _finite_float(
            value.get("stratum_score_max"), "stratum score maximum")
        if rank_start < 0 or rank_end <= rank_start:
            raise ValueError("stratum rank bounds are invalid")
        if score_min > score_max or score < score_min or score > score_max:
            raise ValueError("stratum score bounds are invalid")
        declared_seed = int(value.get("selection_seed", seed))
        if declared_seed != int(seed):
            raise ValueError("selection seed differs")
        window_id = "{}/{}/{:04d}_{:04d}".format(
            scene, stratum, start, end)
        declared_id = str(value.get("window_id", window_id))
        if declared_id != window_id:
            raise ValueError("window ID differs from canonical value")
        result.update({
            "selection_seed": declared_seed,
            "stratum": stratum,
            "window_id": window_id,
            "stratum_rank_start": rank_start,
            "stratum_rank_end": rank_end,
            "stratum_score_min": score_min,
            "stratum_score_max": score_max,
        })
    return result


def rank_tertiles(candidates):
    ordered = sorted(
        (_validate_candidate(row) for row in candidates),
        key=lambda row: (row["motion_score"], row["start_frame"]))
    if len(ordered) < 3:
        raise ValueError("at least three candidates are required")
    boundaries = [0, len(ordered) // 3, 2 * len(ordered) // 3, len(ordered)]
    result = {}
    for index, name in enumerate(STRATA):
        start, end = boundaries[index], boundaries[index + 1]
        if start == end:
            raise ValueError("motion tertile is empty")
        group = []
        for row in ordered[start:end]:
            value = dict(row)
            value.update({
                "stratum": name,
                "stratum_rank_start": start,
                "stratum_rank_end": end,
                "stratum_score_min": ordered[start]["motion_score"],
                "stratum_score_max": ordered[end - 1]["motion_score"],
            })
            group.append(value)
        result[name] = group
    return result


def select_scene_windows(scene, candidates, count_per_stratum=5, seed=2026):
    if scene not in motion.SCENES:
        raise ValueError("unapproved scene")
    count = int(count_per_stratum)
    if count <= 0:
        raise ValueError("count per stratum must be positive")
    groups = rank_tertiles(candidates)
    rng = np.random.RandomState(int(seed) + motion.SCENES.index(scene))
    queues = {}
    for name in STRATA:
        order = rng.permutation(len(groups[name]))
        queues[name] = [groups[name][int(index)] for index in order]
    selected = {name: [] for name in STRATA}
    used_frames = set()
    for _ in range(count):
        for name in STRATA:
            match = None
            for row in queues[name]:
                if used_frames.isdisjoint(row["frame_ids"]):
                    match = row
                    break
            if match is None:
                raise ValueError(
                    "{} cannot provide five non-overlapping windows".format(
                        name))
            queues[name].remove(match)
            used_frames.update(match["frame_ids"])
            value = dict(match)
            value.update({
                "selection_seed": int(seed),
                "window_id": "{}/{}/{:04d}_{:04d}".format(
                    scene, name, value["start_frame"], value["end_frame"]),
            })
            selected[name].append(value)
    return [row for name in STRATA
            for row in sorted(selected[name], key=lambda item: item["start_frame"])]


def validate_selected_windows(rows, seed=SELECTION_SEED, exact_geometry=True):
    values = [_validate_candidate(row, True, seed) for row in list(rows)]
    expected_count = len(motion.SCENES) * len(STRATA) * WINDOWS_PER_STRATUM
    if exact_geometry and len(values) != expected_count:
        raise ValueError("selected manifest must contain exactly 90 windows")
    expected_order = sorted(values, key=lambda row: (
        motion.SCENES.index(row["scene"]), STRATA.index(row["stratum"]),
        row["start_frame"]))
    if values != expected_order:
        raise ValueError("selected windows are not in stable order")
    if len({row["window_id"] for row in values}) != len(values):
        raise ValueError("duplicate window ID")
    by_cell = {}
    used_by_scene = {}
    for row in values:
        cell = (row["scene"], row["stratum"])
        by_cell.setdefault(cell, []).append(row)
        used = used_by_scene.setdefault(row["scene"], set())
        if not used.isdisjoint(row["frame_ids"]):
            raise ValueError("selected windows contain shared frame IDs")
        used.update(row["frame_ids"])
    if exact_geometry:
        expected_cells = {(scene, stratum) for scene in motion.SCENES
                          for stratum in STRATA}
        if set(by_cell) != expected_cells or any(
                len(by_cell[cell]) != WINDOWS_PER_STRATUM
                for cell in expected_cells):
            raise ValueError("each scene/stratum requires five windows")
    return values


def discover_selected_windows(data_root, seed=SELECTION_SEED):
    root = Path(data_root)
    rows = []
    for scene in motion.SCENES:
        rows.extend(select_scene_windows(
            scene, motion.scan_scene_candidates(root / scene),
            WINDOWS_PER_STRATUM, seed))
    return validate_selected_windows(rows, seed)


def manifest_object(rows, seed=SELECTION_SEED):
    return {
        "schema_version": 1,
        "selection_seed": int(seed),
        "windows": validate_selected_windows(rows, seed),
    }


def _atomic_text(path, content):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(str(temporary), str(path))


def write_manifest_json(path, rows, seed=SELECTION_SEED):
    content = json.dumps(
        manifest_object(rows, seed), indent=2, sort_keys=True) + "\n"
    _atomic_text(path, content)


def load_manifest_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or set(value) != {
            "schema_version", "selection_seed", "windows"}:
        raise ValueError("manifest schema is invalid")
    if value["schema_version"] != 1:
        raise ValueError("manifest schema version is invalid")
    return validate_selected_windows(
        value["windows"], int(value["selection_seed"]))


def write_selected_windows_csv(path, rows, seed=SELECTION_SEED):
    values = validate_selected_windows(rows, seed)
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SELECTED_WINDOW_FIELDS)
        writer.writeheader()
        for row in values:
            value = dict(row)
            value["frame_ids"] = json.dumps(value["frame_ids"], separators=(",", ":"))
            value["pair_scores"] = json.dumps(value["pair_scores"], separators=(",", ":"))
            writer.writerow(value)
    os.replace(str(temporary), str(path))


def load_selected_windows_csv(path, seed=SELECTION_SEED):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = []
    for row in rows:
        value = dict(row)
        try:
            value["frame_ids"] = json.loads(value["frame_ids"])
            value["pair_scores"] = json.loads(value["pair_scores"])
        except (KeyError, TypeError, json.JSONDecodeError):
            raise ValueError("selected-window JSON fields are invalid")
        values.append(value)
    return validate_selected_windows(values, seed)


def canonical_manifest_sha256(rows, seed=SELECTION_SEED):
    content = json.dumps(
        manifest_object(rows, seed), sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(content).hexdigest()
