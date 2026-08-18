"""P-frame statistics and artifacts for stratified NLSPN motion evaluation."""

from pathlib import Path
import csv
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

from scripts import nlspn_frame_difference_visualization as visual
from scripts import nlspn_stratified_motion_sampling as sampling


P_FRAME_FIELDS = (
    "scene", "stratum", "window_id", "start_frame", "frame_id",
    "local_index", "method", "adjacent_motion_score",
    "window_motion_score", "rmse", "mae", "valid_pixels",
    "latency_ms", "full_rmse", "full_latency_ms", "excess_rmse",
    "rmse_ratio", "passes_1pct", "speedup")
ROOT_ARTIFACTS = (
    "selected_windows.csv", "p_frame_metrics.csv", "stratified_summary.csv",
    "correlation_summary.csv", "run_metadata.json", "report.md",
    "motion_error_scatter.png", "stratified_error_boxplot.png")


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


def _validate_p_rows(rows, exact_geometry=True):
    values = []
    for source in list(rows):
        row = dict(source)
        if row.get("scene") not in sampling.motion.SCENES:
            raise ValueError("P-frame scene is invalid")
        if row.get("stratum") not in sampling.STRATA:
            raise ValueError("P-frame stratum is invalid")
        if row.get("method") not in visual.RAFT_METHOD_ORDER:
            raise ValueError("P-frame method is invalid")
        numeric = {}
        for name in ("adjacent_motion_score", "window_motion_score", "rmse",
                     "mae", "latency_ms", "full_rmse", "full_latency_ms",
                     "excess_rmse", "rmse_ratio", "speedup"):
            numeric[name] = _finite_float(row.get(name), name)
        valid = int(row.get("valid_pixels", 0))
        local_index = int(row.get("local_index", -1))
        frame_id = int(row.get("frame_id", -1))
        if (valid <= 0 or local_index not in (1, 3) or frame_id <= 0 or
                numeric["rmse"] < 0 or numeric["mae"] < 0 or
                numeric["latency_ms"] <= 0 or numeric["full_rmse"] <= 0 or
                numeric["full_latency_ms"] <= 0 or numeric["rmse_ratio"] <= 0 or
                numeric["speedup"] <= 0):
            raise ValueError("P-frame numeric fields are invalid")
        expected_excess = numeric["rmse"] - numeric["full_rmse"]
        expected_ratio = numeric["rmse"] / numeric["full_rmse"]
        expected_speedup = numeric["full_latency_ms"] / numeric["latency_ms"]
        if not (np.isclose(numeric["excess_rmse"], expected_excess,
                           rtol=1e-12, atol=1e-12) and
                np.isclose(numeric["rmse_ratio"], expected_ratio,
                           rtol=1e-12, atol=1e-12) and
                np.isclose(numeric["speedup"], expected_speedup,
                           rtol=1e-12, atol=1e-12)):
            raise ValueError("P-frame derived fields differ")
        normalized = dict(row)
        normalized.update(numeric)
        normalized.update({
            "valid_pixels": valid, "local_index": local_index,
            "frame_id": frame_id,
            "start_frame": int(row.get("start_frame", -1)),
            "passes_1pct": bool(row.get("passes_1pct")),
        })
        if normalized["passes_1pct"] != (expected_ratio <= 1.01):
            raise ValueError("P-frame pass gate differs")
        values.append(normalized)
    keys = [(row["window_id"], row["method"], row["frame_id"])
            for row in values]
    if len(keys) != len(set(keys)):
        raise ValueError("P-frame keys must be unique")
    if exact_geometry:
        if len(values) != 900:
            raise ValueError("P-frame table must contain exactly 900 rows")
        expected_cells = {(stratum, method) for stratum in sampling.STRATA
                          for method in visual.RAFT_METHOD_ORDER}
        actual_cells = {(row["stratum"], row["method"]) for row in values}
        if actual_cells != expected_cells:
            raise ValueError("P-frame table cells are incomplete")
        for stratum, method in expected_cells:
            selected = [row for row in values
                        if row["stratum"] == stratum and
                        row["method"] == method]
            if len(selected) != 60:
                raise ValueError("each stratum/method requires 60 P rows")
    return values


def _scene_macro(selected, field):
    scene_means = []
    for scene in sampling.motion.SCENES:
        values = [row[field] for row in selected if row["scene"] == scene]
        if values:
            scene_means.append(float(np.mean(values, dtype=np.float64)))
    if not scene_means:
        raise ValueError("summary has no scene values")
    return float(np.mean(scene_means, dtype=np.float64)), scene_means


def _bootstrap_macro(selected, fields, replicates, seed):
    by_scene = {}
    for row in selected:
        by_scene.setdefault(row["scene"], {}).setdefault(
            row["window_id"], []).append(row)
    rng = np.random.RandomState(int(seed))
    samples = {field: [] for field in fields}
    for _ in range(int(replicates)):
        scene_values = {field: [] for field in fields}
        for scene in sorted(by_scene, key=sampling.motion.SCENES.index):
            windows = sorted(by_scene[scene])
            chosen = rng.choice(windows, size=len(windows), replace=True)
            drawn = [row for window_id in chosen
                     for row in by_scene[scene][str(window_id)]]
            for field in fields:
                scene_values[field].append(float(np.mean(
                    [row[field] for row in drawn], dtype=np.float64)))
        for field in fields:
            samples[field].append(float(np.mean(
                scene_values[field], dtype=np.float64)))
    return {field: (
        float(np.percentile(samples[field], 2.5)),
        float(np.percentile(samples[field], 97.5))) for field in fields}


def build_stratified_summary(p_rows, bootstrap_replicates=2000, seed=2026,
                             exact_geometry=True):
    rows = _validate_p_rows(p_rows, exact_geometry)
    cells = sorted({(row["stratum"], row["method"]) for row in rows},
                   key=lambda key: (sampling.STRATA.index(key[0]),
                                    visual.RAFT_METHOD_ORDER.index(key[1])))
    fields = ("rmse", "excess_rmse", "rmse_ratio", "passes_1pct",
              "latency_ms", "speedup")
    result = []
    for cell_index, (stratum, method) in enumerate(cells):
        selected = [row for row in rows if row["stratum"] == stratum and
                    row["method"] == method]
        macros = {field: _scene_macro(selected, field)[0] for field in fields}
        intervals = _bootstrap_macro(
            selected, fields, bootstrap_replicates, seed + cell_index)
        valid_total = sum(row["valid_pixels"] for row in selected)
        pooled_rmse = float(np.sqrt(sum(
            row["rmse"] ** 2 * row["valid_pixels"] for row in selected
        ) / valid_total))
        pooled_mae = float(sum(
            row["mae"] * row["valid_pixels"] for row in selected
        ) / valid_total)
        result.append({
            "stratum": stratum, "method": method,
            "scene_count": len({row["scene"] for row in selected}),
            "window_count": len({row["window_id"] for row in selected}),
            "frame_count": len(selected), "valid_pixels": valid_total,
            "scene_macro_rmse_mean": macros["rmse"],
            "scene_macro_excess_rmse_mean": macros["excess_rmse"],
            "scene_macro_rmse_ratio_mean": macros["rmse_ratio"],
            "scene_macro_pass_rate": macros["passes_1pct"],
            "scene_macro_latency_ms_mean": macros["latency_ms"],
            "scene_macro_speedup_mean": macros["speedup"],
            "frame_rmse_median": float(np.median(
                [row["rmse"] for row in selected])),
            "frame_excess_rmse_median": float(np.median(
                [row["excess_rmse"] for row in selected])),
            "frame_rmse_ratio_median": float(np.median(
                [row["rmse_ratio"] for row in selected])),
            "frame_excess_rmse_std": float(np.std(
                [row["excess_rmse"] for row in selected], ddof=1)),
            "pooled_rmse": pooled_rmse, "pooled_mae": pooled_mae,
            "rmse_ci_low": intervals["rmse"][0],
            "rmse_ci_high": intervals["rmse"][1],
            "excess_rmse_ci_low": intervals["excess_rmse"][0],
            "excess_rmse_ci_high": intervals["excess_rmse"][1],
            "rmse_ratio_ci_low": intervals["rmse_ratio"][0],
            "rmse_ratio_ci_high": intervals["rmse_ratio"][1],
            "pass_rate_ci_low": intervals["passes_1pct"][0],
            "pass_rate_ci_high": intervals["passes_1pct"][1],
            "latency_ci_low": intervals["latency_ms"][0],
            "latency_ci_high": intervals["latency_ms"][1],
            "speedup_ci_low": intervals["speedup"][0],
            "speedup_ci_high": intervals["speedup"][1],
        })
    return result


def build_correlation_summary(p_rows, exact_geometry=True):
    rows = _validate_p_rows(p_rows, exact_geometry)
    result = []
    for method in visual.RAFT_METHOD_ORDER:
        targets = ("rmse",) if method == "full" else (
            "rmse", "excess_rmse", "rmse_ratio")
        selected = [row for row in rows if row["method"] == method]
        x = np.asarray([row["adjacent_motion_score"] for row in selected],
                       dtype=np.float64)
        for target in targets:
            y = np.asarray([row[target] for row in selected], dtype=np.float64)
            if (not np.isfinite(x).all() or not np.isfinite(y).all() or
                    np.ptp(x) <= 0 or np.ptp(y) <= 0):
                raise ValueError("correlation inputs must be finite and nonconstant")
            pearson = stats.pearsonr(x, y)
            spearman = stats.spearmanr(x, y)
            result.append({
                "method": method, "target": target,
                "sample_count": len(selected),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
            })
    return result


def _atomic_text(path, content):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(str(temporary), str(path))


def _atomic_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("CSV rows cannot be empty")
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _save_figure(figure, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    figure.savefig(str(temporary), format="png", dpi=150, facecolor="white")
    plt.close(figure)
    os.replace(str(temporary), str(path))


def _render_scatter(path, rows):
    methods = visual.RAFT_METHOD_ORDER[1:]
    figure, axes = plt.subplots(1, len(methods), figsize=(16, 4), squeeze=False)
    colors = {"low": "#2c7bb6", "medium": "#fdae61", "high": "#d7191c"}
    for axis, method in zip(axes[0], methods):
        selected = [row for row in rows if row["method"] == method]
        for stratum in sampling.STRATA:
            subset = [row for row in selected if row["stratum"] == stratum]
            axis.scatter([row["adjacent_motion_score"] for row in subset],
                         [row["excess_rmse"] for row in subset], s=10,
                         alpha=0.65, color=colors[stratum], label=stratum)
        x = np.asarray([row["adjacent_motion_score"] for row in selected])
        y = np.asarray([row["excess_rmse"] for row in selected])
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(float(x.min()), float(x.max()), 100)
        axis.plot(grid, slope * grid + intercept, color="black", linewidth=1)
        axis.set_title(visual.METHOD_NAMES[method])
        axis.set_xlabel("Adjacent RGB motion")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Excess RMSE (m)")
    axes[0, -1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    _save_figure(figure, path)


def _render_boxplot(path, rows):
    methods = visual.RAFT_METHOD_ORDER[1:]
    figure, axes = plt.subplots(1, 2, figsize=(16, 5), squeeze=False)
    labels = []
    excess = []
    ratios = []
    for stratum in sampling.STRATA:
        for method in methods:
            selected = [row for row in rows if row["stratum"] == stratum and
                        row["method"] == method]
            labels.append("{}\n{}".format(stratum, visual.METHOD_NAMES[method]))
            excess.append([row["excess_rmse"] for row in selected])
            ratios.append([row["rmse_ratio"] for row in selected])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", matplotlib.MatplotlibDeprecationWarning)
        axes[0, 0].boxplot(excess, labels=labels, showfliers=False)
        axes[0, 1].boxplot(ratios, labels=labels, showfliers=False)
    axes[0, 0].set_ylabel("Excess RMSE (m)")
    axes[0, 1].set_ylabel("RMSE / Full")
    axes[0, 1].axhline(1.01, color="red", linestyle="--", linewidth=1)
    for axis in axes[0]:
        axis.tick_params(axis="x", labelrotation=70, labelsize=7)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    _save_figure(figure, path)


def _render_report(windows, summaries, correlations):
    lines = [
        "# NLSPN stratified motion evaluation", "",
        "- Windows: {}".format(len(windows)),
        "- P-frame method rows: 900", "",
        "## Stratified macro results", "",
        "| Stratum | Method | Excess RMSE | RMSE/full | <=1% | Speedup |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append("| {} | {} | {:.6f} | {:.6f} | {:.3f} | {:.3f} |".format(
            row["stratum"], row["method"],
            row["scene_macro_excess_rmse_mean"],
            row["scene_macro_rmse_ratio_mean"],
            row["scene_macro_pass_rate"], row["scene_macro_speedup_mean"]))
    lines.extend(["", "## Motion correlations", "",
                  "| Method | Target | Pearson r (p) | Spearman rho (p) |",
                  "|---|---|---:|---:|"])
    for row in correlations:
        lines.append("| {} | {} | {:.4f} ({:.4g}) | {:.4f} ({:.4g}) |".format(
            row["method"], row["target"], row["pearson_r"],
            row["pearson_p"], row["spearman_rho"], row["spearman_p"]))
    return "\n".join(lines) + "\n"


def write_root_artifacts(output_root, windows, p_rows, summaries,
                         correlations, metadata):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    windows = sampling.validate_selected_windows(windows, sampling.SELECTION_SEED)
    p_rows = _validate_p_rows(p_rows, True)
    if len(summaries) != 15 or len(correlations) != 13:
        raise ValueError("root statistic row counts are invalid")
    sampling.write_selected_windows_csv(
        output_root / "selected_windows.csv", windows, sampling.SELECTION_SEED)
    _atomic_csv(output_root / "p_frame_metrics.csv", p_rows)
    _atomic_csv(output_root / "stratified_summary.csv", summaries)
    _atomic_csv(output_root / "correlation_summary.csv", correlations)
    incomplete = dict(metadata)
    incomplete.update({"complete": False})
    _atomic_text(output_root / "run_metadata.json",
                 json.dumps(incomplete, indent=2, sort_keys=True) + "\n")
    _atomic_text(output_root / "report.md",
                 _render_report(windows, summaries, correlations))
    _render_scatter(output_root / "motion_error_scatter.png", p_rows)
    _render_boxplot(output_root / "stratified_error_boxplot.png", p_rows)
    completed = dict(incomplete)
    completed.update({
        "complete": True, "artifacts": list(ROOT_ARTIFACTS),
        "artifact_count": len(ROOT_ARTIFACTS),
        "selected_window_count": len(windows),
        "frame_count": len(windows) * 5,
        "p_frame_count": len(windows) * 2,
        "p_frame_row_count": len(p_rows),
        "summary_row_count": len(summaries),
        "correlation_row_count": len(correlations),
        "stratified_summary": summaries,
        "correlation_summary": correlations,
    })
    _atomic_text(output_root / "run_metadata.json",
                 json.dumps(completed, indent=2, sort_keys=True) + "\n")
    if any(not (output_root / name).is_file() or
           (output_root / name).stat().st_size <= 0 for name in ROOT_ARTIFACTS):
        raise RuntimeError("root artifacts are incomplete")
    return completed
