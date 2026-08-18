"""P-frame statistics and artifacts for stratified NLSPN motion evaluation."""

import numpy as np

from scripts import nlspn_frame_difference_visualization as visual
from scripts import nlspn_stratified_motion_sampling as sampling


P_FRAME_FIELDS = (
    "scene", "stratum", "window_id", "start_frame", "frame_id",
    "local_index", "method", "adjacent_motion_score",
    "window_motion_score", "rmse", "mae", "valid_pixels",
    "latency_ms", "full_rmse", "full_latency_ms", "excess_rmse",
    "rmse_ratio", "passes_1pct", "speedup")


def _finite_float(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be finite".format(name))
    if not np.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def _validate_frame_metric_rows(windows, frame_metrics):
    rows = list(frame_metrics)
    expected_count = len(windows) * 5 * len(visual.RAFT_METHOD_ORDER)
    if len(rows) != expected_count:
        raise ValueError(
            "each window requires exactly 25 frame metric rows")
    windows_by_id = {row["window_id"]: row for row in windows}
    keyed = {}
    for source in rows:
        row = dict(source)
        window_id = str(row.get("window_id", ""))
        if window_id not in windows_by_id:
            raise ValueError("metric window ID is unknown")
        window = windows_by_id[window_id]
        method = str(row.get("method", ""))
        if method not in visual.RAFT_METHOD_ORDER:
            raise ValueError("metric method is unknown")
        frame_id = int(row.get("frame_id", -1))
        if frame_id not in window["frame_ids"]:
            raise ValueError("metric frame ID is outside its window")
        local_index = window["frame_ids"].index(frame_id)
        expected_kind = "FULL" if method == "full" else (
            "P" if local_index in (1, 3) else "I")
        if str(row.get("frame_kind")) != expected_kind:
            raise ValueError("metric frame kind is invalid")
        rmse = _finite_float(row.get("rmse"), "RMSE")
        mae = _finite_float(row.get("mae"), "MAE")
        latency = _finite_float(row.get("latency_ms"), "latency")
        valid = int(row.get("valid_pixels", 0))
        if rmse < 0 or mae < 0 or latency <= 0 or valid <= 0:
            raise ValueError("metrics and latency must be positive")
        if method == "full" and rmse <= 0:
            raise ValueError("Full RMSE must be positive")
        normalized = {
            "window_id": window_id,
            "scene": window["scene"],
            "stratum": window["stratum"],
            "method": method,
            "frame_id": frame_id,
            "frame_kind": expected_kind,
            "rmse": rmse,
            "mae": mae,
            "valid_pixels": valid,
            "latency_ms": latency,
        }
        key = (window_id, method, frame_id)
        if key in keyed:
            raise ValueError("frame metric keys must be unique")
        keyed[key] = normalized
    expected = {(window["window_id"], method, frame_id)
                for window in windows
                for method in visual.RAFT_METHOD_ORDER
                for frame_id in window["frame_ids"]}
    if set(keyed) != expected:
        raise ValueError("frame metric keys are incomplete")
    return keyed


def derive_p_frame_metrics(windows, frame_metrics,
                           exact_window_count=True):
    windows = sampling.validate_selected_windows(
        windows, seed=sampling.SELECTION_SEED,
        exact_geometry=exact_window_count)
    keyed = _validate_frame_metric_rows(windows, frame_metrics)
    result = []
    for window in windows:
        ids = window["frame_ids"]
        for local_index in (1, 3):
            frame_id = ids[local_index]
            full = keyed[(window["window_id"], "full", frame_id)]
            if full["rmse"] <= 0:
                raise ValueError("Full RMSE must be positive")
            for method in visual.RAFT_METHOD_ORDER:
                row = keyed[(window["window_id"], method, frame_id)]
                ratio = row["rmse"] / full["rmse"]
                result.append({
                    "scene": window["scene"],
                    "stratum": window["stratum"],
                    "window_id": window["window_id"],
                    "start_frame": window["start_frame"],
                    "frame_id": frame_id,
                    "local_index": local_index,
                    "method": method,
                    "adjacent_motion_score":
                        window["pair_scores"][local_index - 1],
                    "window_motion_score": window["motion_score"],
                    "rmse": row["rmse"],
                    "mae": row["mae"],
                    "valid_pixels": row["valid_pixels"],
                    "latency_ms": row["latency_ms"],
                    "full_rmse": full["rmse"],
                    "full_latency_ms": full["latency_ms"],
                    "excess_rmse": row["rmse"] - full["rmse"],
                    "rmse_ratio": ratio,
                    "passes_1pct": bool(ratio <= 1.01),
                    "speedup": full["latency_ms"] / row["latency_ms"],
                })
    return result
