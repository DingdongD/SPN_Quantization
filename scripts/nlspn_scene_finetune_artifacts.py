"""Independent aggregation, visualization, and validation of fine-tuning results."""

import csv
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from scripts import run_nlspn_scene_finetune_worker as worker


DEPTH_BAND_NAMES = ("band_0_2", "band_2_4", "band_4_6", "band_6_8", "band_8_10")
ROOT_ARTIFACTS = (
    "best.pt", "args.json", "train_manifest.csv", "val_manifest.csv",
    "test_manifest.csv", "epoch_metrics.csv",
    "baseline_val_frame_metrics.csv", "test_frame_metrics.csv",
    "aggregate_metrics.csv", "depth_band_metrics.csv",
    "training_curves.png", "rmse_comparison.png",
    "depth_band_comparison.png", "run_metadata.json", "report.md",
    "worker.log")


def _finite_nonnegative(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("{} is nonfinite".format(name))
    if value < 0.0:
        raise ValueError("{} is negative".format(name))
    return np.float64(value)


def _summarize(squared, absolute, abs_rel, count):
    count = int(count)
    if count <= 0:
        raise ValueError("valid pixel count must be positive")
    return {
        "squared_error_sum": np.float64(squared),
        "absolute_error_sum": np.float64(absolute),
        "abs_rel_sum": np.float64(abs_rel),
        "valid_pixels": count,
        "rmse": math.sqrt(float(squared) / count),
        "mae": float(absolute) / count,
        "abs_rel": float(abs_rel) / count,
    }


def _validated_row(row):
    result = dict(row)
    if result.get("variant") not in ("generic", "specialized"):
        raise ValueError("test row variant is invalid")
    if result.get("scene") not in ("room3", "room7"):
        raise ValueError("test row scene is invalid")
    result["frame_id"] = int(result["frame_id"])
    result["valid_pixels"] = int(result["valid_pixels"])
    if result["valid_pixels"] <= 0:
        raise ValueError("test row valid count must be positive")
    for field in ("squared_error_sum", "absolute_error_sum", "abs_rel_sum"):
        result[field] = _finite_nonnegative(result[field], field)
    band_counts = 0
    band_squared = np.float64(0.0)
    band_absolute = np.float64(0.0)
    for band in DEPTH_BAND_NAMES:
        short = band.replace("band_", "")
        squared_field = "band_{}_squared_error_sum".format(short)
        absolute_field = "band_{}_absolute_error_sum".format(short)
        count_field = "band_{}_valid_pixels".format(short)
        if any(field not in result
               for field in (squared_field, absolute_field, count_field)):
            raise ValueError("test row is missing depth band fields")
        result[squared_field] = _finite_nonnegative(
            result[squared_field], squared_field)
        result[absolute_field] = _finite_nonnegative(
            result[absolute_field], absolute_field)
        result[count_field] = int(result[count_field])
        if result[count_field] < 0:
            raise ValueError("depth band count is negative")
        band_counts += result[count_field]
        band_squared += result[squared_field]
        band_absolute += result[absolute_field]
    if band_counts != result["valid_pixels"]:
        raise ValueError("depth band count does not equal full valid count")
    if not np.isclose(band_squared, result["squared_error_sum"], rtol=1e-9, atol=1e-9):
        raise ValueError("depth band squared sums differ from full sum")
    if not np.isclose(band_absolute, result["absolute_error_sum"], rtol=1e-9, atol=1e-9):
        raise ValueError("depth band absolute sums differ from full sum")
    return result


def aggregate_test_rows(rows, exact_geometry=True):
    rows = [_validated_row(row) for row in rows]
    identities = [(row["variant"], row["scene"], row["frame_id"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("test rows contain duplicate identities")
    if exact_geometry:
        worker.validate_test_row_identities(rows)
    elif {row["variant"] for row in rows} != {"generic", "specialized"}:
        raise ValueError("both variants are required")
    result = {"variants": {}}
    for variant in ("generic", "specialized"):
        variant_rows = [row for row in rows if row["variant"] == variant]
        if not variant_rows:
            raise ValueError("variant {} has no rows".format(variant))
        squared = np.sum([row["squared_error_sum"] for row in variant_rows], dtype=np.float64)
        absolute = np.sum([row["absolute_error_sum"] for row in variant_rows], dtype=np.float64)
        abs_rel = np.sum([row["abs_rel_sum"] for row in variant_rows], dtype=np.float64)
        count = sum(row["valid_pixels"] for row in variant_rows)
        summary = _summarize(squared, absolute, abs_rel, count)
        scenes = {}
        for scene in ("room3", "room7"):
            scene_rows = [row for row in variant_rows if row["scene"] == scene]
            if not scene_rows:
                raise ValueError("variant {} is missing scene {}".format(variant, scene))
            scenes[scene] = _summarize(
                np.sum([row["squared_error_sum"] for row in scene_rows], dtype=np.float64),
                np.sum([row["absolute_error_sum"] for row in scene_rows], dtype=np.float64),
                np.sum([row["abs_rel_sum"] for row in scene_rows], dtype=np.float64),
                sum(row["valid_pixels"] for row in scene_rows))
        summary["scenes"] = scenes
        summary["scene_macro_rmse"] = float(np.mean(
            [scenes[name]["rmse"] for name in ("room3", "room7")]))
        summary["scene_macro_mae"] = float(np.mean(
            [scenes[name]["mae"] for name in ("room3", "room7")]))
        summary["scene_macro_abs_rel"] = float(np.mean(
            [scenes[name]["abs_rel"] for name in ("room3", "room7")]))
        summary["pooled_rmse"] = summary["rmse"]
        summary["pooled_mae"] = summary["mae"]
        summary["pooled_abs_rel"] = summary["abs_rel"]
        bands = {}
        for band in DEPTH_BAND_NAMES:
            short = band.replace("band_", "")
            squared_field = "band_{}_squared_error_sum".format(short)
            absolute_field = "band_{}_absolute_error_sum".format(short)
            count_field = "band_{}_valid_pixels".format(short)
            band_count = sum(row[count_field] for row in variant_rows)
            if band_count <= 0:
                raise ValueError("depth band {} has no valid pixels".format(band))
            band_squared = np.sum(
                [row[squared_field] for row in variant_rows], dtype=np.float64)
            band_absolute = np.sum(
                [row[absolute_field] for row in variant_rows], dtype=np.float64)
            bands[band] = _summarize(
                band_squared, band_absolute, np.float64(0.0), band_count)
        summary["depth_bands"] = bands
        result["variants"][variant] = summary
    return result


def calculate_success_gate(baseline, specialized):
    baseline_rmse = float(baseline["pooled_rmse"])
    specialized_rmse = float(specialized["pooled_rmse"])
    if not math.isfinite(baseline_rmse) or baseline_rmse <= 0.0:
        raise ValueError("baseline pooled RMSE must be finite positive")
    relative = (baseline_rmse - specialized_rmse) / baseline_rmse
    pooled_passed = relative >= 0.05
    scene_ratios = {}
    scene_passed = {}
    for scene in ("room3", "room7"):
        base = float(baseline["scenes"][scene]["rmse"])
        value = float(specialized["scenes"][scene]["rmse"])
        if not math.isfinite(base) or base <= 0.0 or not math.isfinite(value):
            raise ValueError("scene RMSE values must be finite and baseline positive")
        scene_ratios[scene] = value / base
        scene_passed[scene] = scene_ratios[scene] <= 1.01
    return {
        "relative_improvement": relative,
        "pooled_improvement_passed": pooled_passed,
        "scene_rmse_ratios": scene_ratios,
        "scene_no_regression_passed": scene_passed,
        "passed": pooled_passed and all(scene_passed.values()),
    }


def validate_png(path, require_nonblank=False):
    with Image.open(str(path)) as image:
        image.verify()
    with Image.open(str(path)) as image:
        array = np.asarray(image.convert("RGB"))
    if array.size == 0:
        raise ValueError("PNG is empty")
    if require_nonblank and float(array.std()) < 1e-6:
        raise ValueError("PNG panel is blank")
    return tuple(array.shape)


def _save_figure(path, figure):
    figure.savefig(str(path), dpi=130, bbox_inches="tight")
    plt.close(figure)
    validate_png(path, require_nonblank=True)


def write_window_artifacts(root, payload, require_500_sparse=False):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("window output directory is not empty")
    root.mkdir(parents=True, exist_ok=True)
    required = ("scenes", "frame_ids", "rgb", "sparse", "gt", "valid",
                "generic", "specialized")
    if any(name not in payload for name in required):
        raise ValueError("window payload is missing required arrays")
    count = len(payload["frame_ids"])
    if count != 150 or any(len(payload[name]) != count for name in required):
        raise ValueError("window payload must contain exactly 150 aligned frames")
    for window_index in range(30):
        indices = slice(window_index * 5, window_index * 5 + 5)
        scenes = np.asarray(payload["scenes"])[indices]
        frame_ids = np.asarray(payload["frame_ids"])[indices].astype(int)
        if len(set(scenes.tolist())) != 1 or not np.all(np.diff(frame_ids) == 1):
            raise ValueError("window payload groups must be one scene and consecutive")
        sparse = np.asarray(payload["sparse"])[indices]
        if require_500_sparse and np.any(
                np.count_nonzero(sparse.reshape(5, -1), axis=1) != 500):
            raise ValueError("window input does not contain 500 sparse points")
        scene = str(scenes[0])
        directory = root / "{:02d}_{}_{:04d}_{:04d}".format(
            window_index + 1, scene, frame_ids[0], frame_ids[-1])
        directory.mkdir()
        values = {name: np.asarray(payload[name])[indices] for name in required}
        np.savez_compressed(str(directory / "predictions.npz"), **values)

        figure, axes = plt.subplots(5, 3, figsize=(9, 12))
        for row in range(5):
            for column, name in enumerate(("gt", "generic", "specialized")):
                axes[row, column].imshow(values[name][row], vmin=0, vmax=10, cmap="viridis")
                axes[row, column].axis("off")
                if row == 0:
                    axes[row, column].set_title((
                        "GT", "Generic Full NLSPN", "Specialized Full NLSPN")[column])
        _save_figure(directory / "depth_comparison.png", figure)

        figure, axes = plt.subplots(5, 2, figsize=(6, 12))
        maximum = max(
            float(np.max(np.abs(values["generic"] - values["gt"]))),
            float(np.max(np.abs(values["specialized"] - values["gt"]))), 1e-6)
        for row in range(5):
            for column, name in enumerate(("generic", "specialized")):
                error = np.abs(values[name][row] - values["gt"][row])
                axes[row, column].imshow(error, vmin=0, vmax=maximum, cmap="magma")
                axes[row, column].axis("off")
                if row == 0:
                    axes[row, column].set_title(name.title() + " absolute error")
        _save_figure(directory / "error_comparison.png", figure)
        metrics = []
        for frame_id, generic, specialized, gt, valid in zip(
                frame_ids, values["generic"], values["specialized"],
                values["gt"], values["valid"]):
            for name, prediction in (("generic", generic),
                                     ("specialized", specialized)):
                error = prediction[valid] - gt[valid]
                metrics.append({
                    "variant": name, "frame_id": int(frame_id),
                    "rmse": float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
                    "mae": float(np.mean(np.abs(error.astype(np.float64)))),
                })
        with (directory / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("variant", "frame_id", "rmse", "mae"))
            writer.writeheader()
            writer.writerows(metrics)
    return {"window_count": 30}


def _read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _render_summary_plots(root, aggregate, epoch_rows):
    epochs = [int(row["epoch"]) for row in epoch_rows]
    train = [float(row["train_loss"]) for row in epoch_rows]
    val = [float(row["val_rmse"]) for row in epoch_rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train, marker="o"); axes[0].set_title("Training loss")
    axes[1].plot(epochs, val, marker="o"); axes[1].set_title("Validation RMSE")
    _save_figure(root / "training_curves.png", figure)

    labels = ("Pooled", "room3", "room7")
    generic = aggregate["variants"]["generic"]
    specialized = aggregate["variants"]["specialized"]
    generic_values = [generic["pooled_rmse"]] + [generic["scenes"][x]["rmse"] for x in labels[1:]]
    specialized_values = [specialized["pooled_rmse"]] + [specialized["scenes"][x]["rmse"] for x in labels[1:]]
    x = np.arange(3)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(x - 0.18, generic_values, 0.36, label="Generic")
    axis.bar(x + 0.18, specialized_values, 0.36, label="Specialized")
    axis.set_xticks(x); axis.set_xticklabels(labels); axis.legend(); axis.set_ylabel("RMSE (m)")
    _save_figure(root / "rmse_comparison.png", figure)

    x = np.arange(5)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(x - 0.18, [generic["depth_bands"][b]["rmse"] for b in DEPTH_BAND_NAMES], .36, label="Generic")
    axis.bar(x + 0.18, [specialized["depth_bands"][b]["rmse"] for b in DEPTH_BAND_NAMES], .36, label="Specialized")
    axis.set_xticks(x); axis.set_xticklabels([b.replace("band_", "") for b in DEPTH_BAND_NAMES]); axis.legend()
    _save_figure(root / "depth_band_comparison.png", figure)


def write_final_artifacts(root, manifest_paths, source_digests=None,
                          checkpoint_loader=None):
    root = Path(root)
    raw = root / "raw"
    if not raw.is_dir():
        raise ValueError("raw worker directory is missing")
    worker_metadata = json.loads((raw / "worker_metadata.json").read_text())
    if not worker_metadata.get("complete"):
        raise ValueError("worker metadata complete=false")
    rows = _read_csv(raw / "test_frame_metrics.csv")
    aggregate = aggregate_test_rows(rows, exact_geometry=True)
    generic = aggregate["variants"]["generic"]
    specialized = aggregate["variants"]["specialized"]
    gate = calculate_success_gate(generic, specialized)

    shutil.copyfile(str(raw / "best.pt"), str(root / "best.pt"))
    shutil.copyfile(str(raw / "specialized_args.json"), str(root / "args.json"))
    for split in ("train", "val", "test"):
        shutil.copyfile(str(manifest_paths[split]), str(root / (split + "_manifest.csv")))
    for name in ("epoch_metrics.csv", "baseline_val_frame_metrics.csv", "test_frame_metrics.csv"):
        shutil.copyfile(str(raw / name), str(root / name))

    aggregate_rows = []
    band_rows = []
    for variant in ("generic", "specialized"):
        value = aggregate["variants"][variant]
        for scope in ("pooled", "room3", "room7"):
            metrics = value if scope == "pooled" else value["scenes"][scope]
            aggregate_rows.append({"variant": variant, "scope": scope,
                                   "rmse": metrics["rmse"], "mae": metrics["mae"],
                                   "abs_rel": metrics["abs_rel"],
                                   "valid_pixels": metrics["valid_pixels"]})
        for band in DEPTH_BAND_NAMES:
            metrics = value["depth_bands"][band]
            band_rows.append({"variant": variant, "band": band,
                              "rmse": metrics["rmse"], "mae": metrics["mae"],
                              "valid_pixels": metrics["valid_pixels"]})
    _write_csv(root / "aggregate_metrics.csv", aggregate_rows,
               ("variant", "scope", "rmse", "mae", "abs_rel", "valid_pixels"))
    _write_csv(root / "depth_band_metrics.csv", band_rows,
               ("variant", "band", "rmse", "mae", "valid_pixels"))
    epoch_rows = _read_csv(raw / "epoch_metrics.csv")
    _render_summary_plots(root, aggregate, epoch_rows)
    with np.load(str(raw / "window_predictions.npz"), allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    write_window_artifacts(root / "windows", payload, require_500_sparse=True)

    metadata = {
        "complete": True, "aggregate": aggregate, "success_gate": gate,
        "worker": worker_metadata, "source_digests": source_digests or {},
        "test_frame_count": 8000, "test_metric_row_count": len(rows),
        "window_count": 30,
    }
    (root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=float) + "\n")
    report = (
        "# Scene-specific NLSPN fine-tuning\n\n"
        "- Generic pooled RMSE: {:.9f} m\n"
        "- Specialized pooled RMSE: {:.9f} m\n"
        "- Relative improvement: {:.6%}\n"
        "- Success gate: {}\n".format(
            generic["pooled_rmse"], specialized["pooled_rmse"],
            gate["relative_improvement"], "PASS" if gate["passed"] else "FAIL"))
    (root / "report.md").write_text(report)
    if checkpoint_loader is not None:
        checkpoint_loader(root / "best.pt")
    validate_final_tree(root, checkpoint_loader=checkpoint_loader)
    return metadata


def validate_final_tree(root, checkpoint_loader=None):
    root = Path(root)
    actual_files = {path.name for path in root.iterdir() if path.is_file()}
    if actual_files != set(ROOT_ARTIFACTS):
        raise ValueError("final root files differ from exact contract")
    actual_dirs = {path.name for path in root.iterdir() if path.is_dir()}
    if actual_dirs != {"raw", "windows"}:
        raise ValueError("final root directories differ from exact contract")
    if {path.name for path in (root / "raw").iterdir()} != set(worker.RAW_ARTIFACTS):
        raise ValueError("raw artifact tree differs from exact contract")
    if (root / "best.pt").read_bytes() != (root / "raw/best.pt").read_bytes():
        raise ValueError("root and raw best checkpoints differ")
    windows = [path for path in (root / "windows").iterdir() if path.is_dir()]
    if len(windows) != 30:
        raise ValueError("final tree must contain exactly 30 windows")
    expected_window_files = {
        "depth_comparison.png", "error_comparison.png", "predictions.npz", "metrics.csv"}
    for directory in windows:
        if {path.name for path in directory.iterdir()} != expected_window_files:
            raise ValueError("window artifact tree differs from exact contract")
        validate_png(directory / "depth_comparison.png", True)
        validate_png(directory / "error_comparison.png", True)
    for name in ("training_curves.png", "rmse_comparison.png", "depth_band_comparison.png"):
        validate_png(root / name, True)
    metadata = json.loads((root / "run_metadata.json").read_text())
    if not metadata.get("complete"):
        raise ValueError("run metadata complete=false")
    aggregate_test_rows(_read_csv(root / "test_frame_metrics.csv"), True)
    if checkpoint_loader is not None:
        checkpoint_loader(root / "best.pt")
    return metadata
