#!/usr/bin/env python3
"""Export CSPN predictions and unregistered temporal diagnostics."""

import argparse
import csv
import hashlib
import json
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

from pathlib import Path
import sys

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


MAX_DEPTH = 10.0
OUTPUT_HEIGHT = 228
OUTPUT_WIDTH = 304


def load_rgb(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with path.open("rb") as stream:
        return np.asarray(Image.open(stream).convert("RGB"), dtype=np.uint8).copy()


def read_exr_depth(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError("failed to decode EXR depth: %s" % path)
    if depth.ndim != 3 or depth.shape[2] != 3:
        raise ValueError("EXR depth must contain three channels: %s" % path)
    if not (np.allclose(depth[..., 0], depth[..., 1], equal_nan=True) and
            np.allclose(depth[..., 0], depth[..., 2], equal_nan=True)):
        raise ValueError("EXR depth channels differ: %s" % path)
    return depth[..., 0].astype(np.float32, copy=True)


def sanitize_depth(depth, max_depth=MAX_DEPTH):
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= float(max_depth))
    return np.where(valid, depth, 0.0).astype(np.float32), valid


def preprocess_pair(rgb, depth, max_depth=MAX_DEPTH):
    rgb = np.asarray(rgb, dtype=np.uint8)
    depth = np.asarray(depth, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape [height, width, 3]")
    if depth.shape != rgb.shape[:2]:
        raise ValueError("rgb and depth must have identical image geometry")

    clean_depth, valid = sanitize_depth(depth, max_depth=max_depth)
    rgb_image = Image.fromarray(rgb, mode="RGB")
    depth_image = Image.fromarray(clean_depth, mode="F")
    valid_image = Image.fromarray(valid.astype(np.uint8) * 255, mode="L")

    rgb_image = TF.resize(
        rgb_image, 240, interpolation=InterpolationMode.BILINEAR)
    depth_image = TF.resize(
        depth_image, 240, interpolation=InterpolationMode.NEAREST)
    valid_image = TF.resize(
        valid_image, 240, interpolation=InterpolationMode.NEAREST)
    rgb_image = TF.center_crop(rgb_image, (OUTPUT_HEIGHT, OUTPUT_WIDTH))
    depth_image = TF.center_crop(depth_image, (OUTPUT_HEIGHT, OUTPUT_WIDTH))
    valid_image = TF.center_crop(valid_image, (OUTPUT_HEIGHT, OUTPUT_WIDTH))

    rgb_out = TF.to_tensor(rgb_image).numpy().astype(np.float32)
    depth_out = np.asarray(depth_image, dtype=np.float32).copy()
    valid_out = np.asarray(valid_image, dtype=np.uint8) > 0
    depth_out, depth_valid = sanitize_depth(depth_out, max_depth=max_depth)
    valid_out &= depth_valid
    depth_out[~valid_out] = 0.0
    return rgb_out, depth_out, valid_out


def build_shared_sparse_depths(depths, valid_masks, count=500, seed=2026):
    depths = np.asarray(depths, dtype=np.float32)
    valid_masks = np.asarray(valid_masks, dtype=bool)
    if depths.shape != valid_masks.shape or depths.ndim != 3:
        raise ValueError(
            "depths and valid_masks must have shape [frames, height, width]")
    common = np.all(valid_masks, axis=0)
    candidates = np.flatnonzero(common)
    if candidates.size < int(count):
        raise ValueError(
            "common valid pixels %d are fewer than %d" %
            (candidates.size, count))
    chosen = np.random.default_rng(seed).choice(
        candidates, size=int(count), replace=False)
    mask = np.zeros(common.size, dtype=bool)
    mask[chosen] = True
    mask = mask.reshape(common.shape)
    return (depths * mask[None]).astype(np.float32), mask


def _validated_metric_arrays(gt, pred, valid):
    gt = np.asarray(gt, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if gt.shape != pred.shape or gt.shape != valid.shape:
        raise ValueError("gt, pred, and valid must have identical shapes")
    if not np.any(valid):
        raise ValueError("metric mask contains no valid pixels")
    if not np.isfinite(gt[valid]).all() or not np.isfinite(pred[valid]).all():
        raise ValueError("metric inputs contain non-finite valid values")
    return gt, pred, valid


def frame_metrics(gt, pred, valid):
    gt, pred, valid = _validated_metric_arrays(gt, pred, valid)
    gt_valid = gt[valid].astype(np.float64)
    diff = pred[valid].astype(np.float64) - gt_valid
    result = {
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
        "abs_rel": float(np.mean(np.abs(diff) / gt_valid)),
        "valid_pixels": int(valid.sum()),
        "valid_coverage": float(valid.mean()),
    }
    if not all(np.isfinite(value) for key, value in result.items()
               if key != "valid_pixels"):
        raise ValueError("frame metrics contain non-finite values")
    return result


def temporal_metrics(depths, predictions, valid_masks, frame_ids):
    depths = np.asarray(depths, dtype=np.float32)
    predictions = np.asarray(predictions, dtype=np.float32)
    valid_masks = np.asarray(valid_masks, dtype=bool)
    if depths.shape != predictions.shape or depths.shape != valid_masks.shape:
        raise ValueError(
            "depths, predictions, and valid_masks must have identical shapes")
    if depths.ndim != 3 or depths.shape[0] != len(frame_ids):
        raise ValueError("sequence arrays and frame_ids must describe the same frames")

    rows = []
    maps = []
    for index in range(len(frame_ids) - 1):
        valid = valid_masks[index] & valid_masks[index + 1]
        gt_change = depths[index + 1] - depths[index]
        pred_change = predictions[index + 1] - predictions[index]
        residual = pred_change - gt_change
        _, _, valid = _validated_metric_arrays(gt_change, pred_change, valid)
        values = residual[valid].astype(np.float64)
        row = {
            "pair": "%04d->%04d" % (frame_ids[index], frame_ids[index + 1]),
            "from_frame": int(frame_ids[index]),
            "to_frame": int(frame_ids[index + 1]),
            "rmse": float(np.sqrt(np.mean(values ** 2))),
            "mae": float(np.mean(np.abs(values))),
            "valid_pixels": int(valid.sum()),
            "valid_coverage": float(valid.mean()),
        }
        if not np.isfinite(row["rmse"]) or not np.isfinite(row["mae"]):
            raise ValueError("temporal metrics contain non-finite values")
        rows.append(row)
        maps.append({
            "pair": row["pair"],
            "gt_change": gt_change,
            "pred_change": pred_change,
            "residual": residual,
            "valid": valid,
        })
    return rows, maps


def load_compatible_state(model, state, allowed_missing=()):
    if isinstance(state, dict) and "net" in state:
        state = state["net"]
    normalized = {}
    for key, value in state.items():
        normalized[key.removeprefix("module.")] = value

    ignored = []
    legacy_key = "post_process_layer.sum_conv.weight"
    if legacy_key in normalized:
        value = normalized.pop(legacy_key)
        if (tuple(value.shape) != (1, 8, 1, 1, 1) or
                not bool(torch.all(value == 1).item())):
            raise RuntimeError("invalid CSPN fixed sum kernel in checkpoint")
        ignored.append(legacy_key)

    incompatible = model.load_state_dict(normalized, strict=False)
    allowed_missing = set(allowed_missing)
    missing = sorted(set(incompatible.missing_keys) - allowed_missing)
    unexpected = sorted(incompatible.unexpected_keys)
    if missing:
        raise RuntimeError("missing checkpoint keys: %s" % missing)
    if unexpected:
        raise RuntimeError("unexpected checkpoint keys: %s" % unexpected)
    return {
        "ignored": ignored,
        "missing": missing,
        "allowed_missing": sorted(
            set(incompatible.missing_keys) & allowed_missing),
        "unexpected": unexpected,
    }


def cspn_model_config():
    """Return the propagation settings compatible with the legacy checkpoint."""
    return {"step": 24, "kernel": 3, "norm_type": "8sum_abs"}


def build_cspn(checkpoint, device):
    repo_root = Path(__file__).resolve().parents[1]
    models_root = repo_root / "models"
    if str(models_root) not in sys.path:
        sys.path.insert(0, str(models_root))
    import torch_resnet_cspn_nyu as cspn_model

    model = cspn_model.resnet50(
        pretrained=False, cspn_config=cspn_model_config())
    try:
        state = torch.load(
            str(checkpoint), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(checkpoint), map_location="cpu")
    allowed_missing = {
        key for key in model.state_dict() if key.endswith("._up_pool.weights")}
    report = load_compatible_state(
        model, state, allowed_missing=allowed_missing)
    return model.to(device).eval(), report


def predict_sequence(model, rgb, sparse, device):
    rgb = np.asarray(rgb, dtype=np.float32)
    sparse = np.asarray(sparse, dtype=np.float32)
    if (rgb.ndim != 4 or rgb.shape[1:] !=
            (3, OUTPUT_HEIGHT, OUTPUT_WIDTH)):
        raise ValueError("rgb must have shape [frames, 3, 228, 304]")
    if sparse.shape != (rgb.shape[0], OUTPUT_HEIGHT, OUTPUT_WIDTH):
        raise ValueError("sparse must have shape [frames, 228, 304]")

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.train_nyu_iteration_sweep import legacy_cspn_rgb

    predictions = []
    with torch.inference_mode():
        for index in range(rgb.shape[0]):
            rgb_tensor = torch.from_numpy(rgb[index:index + 1]).to(device)
            sparse_tensor = torch.from_numpy(
                sparse[index:index + 1, None]).to(device)
            model_input = torch.cat(
                (legacy_cspn_rgb(rgb_tensor), sparse_tensor), dim=1)
            if tuple(model_input.shape) != (
                    1, 4, OUTPUT_HEIGHT, OUTPUT_WIDTH):
                raise RuntimeError("unexpected CSPN input shape: %s" %
                                   (tuple(model_input.shape),))
            output = model(model_input)
            if isinstance(output, dict):
                output = output["pred"]
            prediction = output.detach().float().cpu().numpy()[0, 0]
            if not np.isfinite(prediction).all():
                raise ValueError("non-finite prediction for frame index %d" % index)
            predictions.append(prediction)
    return np.stack(predictions).astype(np.float32)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_frame_paths(data_root, scene, frame_ids):
    scene_root = Path(data_root) / scene
    pairs = []
    for frame_id in frame_ids:
        rgb_path = scene_root / "rgb" / ("%04d.jpg" % frame_id)
        depth_path = scene_root / "depth" / ("Image%04d.exr" % frame_id)
        for path in (rgb_path, depth_path):
            if not path.is_file():
                raise FileNotFoundError(str(path))
        pairs.append((rgb_path, depth_path))
    return pairs


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _masked(values, valid):
    return np.ma.array(values, mask=~np.asarray(valid, dtype=bool))


def write_frame_panel(path, frame_id, rgb, sparse, gt, pred, valid):
    error = np.abs(pred - gt)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.3), constrained_layout=True)
    axes[0].imshow(np.moveaxis(rgb, 0, -1))
    axes[0].set_title("RGB %04d" % frame_id)
    sparse_valid = sparse > 0.0
    sparse_plot = axes[1].imshow(
        _masked(sparse, sparse_valid), vmin=0.0, vmax=MAX_DEPTH, cmap="viridis")
    axes[1].set_title("Sparse depth (500)")
    gt_plot = axes[2].imshow(
        _masked(gt, valid), vmin=0.0, vmax=MAX_DEPTH, cmap="viridis")
    axes[2].set_title("Ground truth")
    axes[3].imshow(
        _masked(pred, valid), vmin=0.0, vmax=MAX_DEPTH, cmap="viridis")
    axes[3].set_title("CSPN prediction")
    error_plot = axes[4].imshow(
        _masked(error, valid), vmin=0.0, vmax=3.0, cmap="magma")
    axes[4].set_title("Absolute error")
    for axis in axes:
        axis.set_axis_off()
    fig.colorbar(sparse_plot, ax=axes[1:4], shrink=0.72, label="Depth (m)")
    fig.colorbar(error_plot, ax=axes[4], shrink=0.72, label="Error (m)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_sequence_overview(path, frame_ids, rgb, gt, pred, valid):
    fig, axes = plt.subplots(
        len(frame_ids), 4, figsize=(12, 2.55 * len(frame_ids)),
        squeeze=False, constrained_layout=True)
    depth_plot = None
    error_plot = None
    for row, frame_id in enumerate(frame_ids):
        error = np.abs(pred[row] - gt[row])
        axes[row, 0].imshow(np.moveaxis(rgb[row], 0, -1))
        depth_plot = axes[row, 1].imshow(
            _masked(gt[row], valid[row]), vmin=0.0, vmax=MAX_DEPTH,
            cmap="viridis")
        axes[row, 2].imshow(
            _masked(pred[row], valid[row]), vmin=0.0, vmax=MAX_DEPTH,
            cmap="viridis")
        error_plot = axes[row, 3].imshow(
            _masked(error, valid[row]), vmin=0.0, vmax=3.0, cmap="magma")
        axes[row, 0].set_ylabel("Frame %04d" % frame_id)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    for axis, title in zip(
            axes[0], ("RGB", "Ground truth", "CSPN", "Absolute error")):
        axis.set_title(title)
    fig.colorbar(depth_plot, ax=axes[:, 1:3], shrink=0.72, label="Depth (m)")
    fig.colorbar(error_plot, ax=axes[:, 3], shrink=0.72, label="Error (m)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_temporal_overview(path, temporal_maps):
    finite_values = []
    for item in temporal_maps:
        for key in ("gt_change", "pred_change", "residual"):
            finite_values.append(np.abs(item[key][item["valid"]]))
    limit = max(0.1, float(np.percentile(np.concatenate(finite_values), 99)))
    fig, axes = plt.subplots(
        len(temporal_maps), 3, figsize=(10, 2.6 * len(temporal_maps)),
        squeeze=False, constrained_layout=True)
    plot = None
    for row, item in enumerate(temporal_maps):
        for column, key in enumerate(("gt_change", "pred_change", "residual")):
            plot = axes[row, column].imshow(
                _masked(item[key], item["valid"]), vmin=-limit, vmax=limit,
                cmap="coolwarm")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(item["pair"])
    for axis, title in zip(
            axes[0], ("GT change", "Prediction change", "Temporal residual")):
        axis.set_title(title)
    fig.suptitle("Unregistered image-space temporal differences")
    fig.colorbar(plot, ax=axes, shrink=0.75, label="Depth change (m)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_artifacts(out_dir, frame_ids, rgb, sparse, depth, predictions,
                    valid_masks, checkpoint_sha256, model_config):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(rgb, dtype=np.float32)
    sparse = np.asarray(sparse, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    predictions = np.asarray(predictions, dtype=np.float32)
    valid_masks = np.asarray(valid_masks, dtype=bool)
    if not np.isfinite(predictions).all():
        raise ValueError("predictions contain non-finite values")
    pred_clamped = np.clip(predictions, 1e-6, MAX_DEPTH)

    manifest = {"frame_npz": [], "frame_panels": []}
    frame_rows = []
    for index, frame_id in enumerate(frame_ids):
        metrics = frame_metrics(
            depth[index], pred_clamped[index], valid_masks[index])
        row = {"frame_id": int(frame_id),
               "sparse_points": int(np.count_nonzero(sparse[index]))}
        row.update(metrics)
        frame_rows.append(row)
        error = np.abs(pred_clamped[index] - depth[index])
        npz_path = out_dir / ("frame_%04d.npz" % frame_id)
        np.savez_compressed(
            npz_path,
            frame_id=np.array(frame_id, dtype=np.int32),
            rgb=rgb[index],
            sparse=sparse[index],
            gt=depth[index],
            pred_raw=predictions[index],
            pred_clamped=pred_clamped[index],
            valid=valid_masks[index],
            abs_err=error)
        panel_path = out_dir / ("frame_%04d_panel.png" % frame_id)
        write_frame_panel(
            panel_path, frame_id, rgb[index], sparse[index], depth[index],
            pred_clamped[index], valid_masks[index])
        manifest["frame_npz"].append(str(npz_path))
        manifest["frame_panels"].append(str(panel_path))

    temporal_rows, temporal_maps = temporal_metrics(
        depth, pred_clamped, valid_masks, frame_ids)
    frame_csv = out_dir / "frame_metrics.csv"
    temporal_csv = out_dir / "temporal_metrics.csv"
    write_csv(
        frame_csv, frame_rows,
        ("frame_id", "rmse", "mae", "abs_rel", "valid_pixels",
         "valid_coverage", "sparse_points"))
    write_csv(
        temporal_csv, temporal_rows,
        ("pair", "from_frame", "to_frame", "rmse", "mae",
         "valid_pixels", "valid_coverage"))

    sequence_path = out_dir / "sequence_overview.png"
    temporal_path = out_dir / "temporal_overview_unregistered.png"
    write_sequence_overview(
        sequence_path, frame_ids, rgb, depth, pred_clamped, valid_masks)
    write_temporal_overview(temporal_path, temporal_maps)

    metadata_path = out_dir / "run_metadata.json"
    metadata = {
        "frame_ids": [int(value) for value in frame_ids],
        "sparse_count": int(np.count_nonzero(sparse[0])),
        "checkpoint_sha256": checkpoint_sha256,
        "model_config": model_config,
        "preprocessing": {
            "resize_short_side": 240,
            "center_crop": [OUTPUT_HEIGHT, OUTPUT_WIDTH],
            "valid_depth_metres": [0.0, MAX_DEPTH],
        },
        "temporal_alignment": "unregistered",
        "temporal_warning": (
            "Image-space differences are not compensated for camera motion."),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    manifest.update({
        "sequence_overview": str(sequence_path),
        "temporal_overview": str(temporal_path),
        "frame_metrics": str(frame_csv),
        "temporal_metrics": str(temporal_csv),
        "metadata": str(metadata_path),
    })
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/workspace/VoxelNet/train")
    parser.add_argument("--scene", default="BeachApartmentInterior_My_ir")
    parser.add_argument("--frames", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--checkpoint",
        default="/workspace/VoxelNet/cspn_models/best_model.pth")
    parser.add_argument("--sparse-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--out-dir",
        default=("/workspace/VoxelNet/cspn_predictions/"
                 "BeachApartmentInterior_My_ir/frames_0001_0005"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for requested device %s" % args.device)
    device = torch.device(args.device)
    pairs = resolve_frame_paths(args.data_root, args.scene, args.frames)

    rgb_frames = []
    depths = []
    valid_masks = []
    for rgb_path, depth_path in pairs:
        rgb, depth, valid = preprocess_pair(
            load_rgb(rgb_path), read_exr_depth(depth_path))
        rgb_frames.append(rgb)
        depths.append(depth)
        valid_masks.append(valid)
    rgb_frames = np.stack(rgb_frames)
    depths = np.stack(depths)
    valid_masks = np.stack(valid_masks)
    sparse, _ = build_shared_sparse_depths(
        depths, valid_masks, count=args.sparse_count, seed=args.seed)

    model, load_report = build_cspn(args.checkpoint, device)
    predictions = predict_sequence(model, rgb_frames, sparse, device)
    propagation_config = cspn_model_config()
    model_config = {
        "architecture": "CSPN ResNet-50",
        "iteration": propagation_config["step"],
        "kernel": propagation_config["kernel"],
        "norm_type": propagation_config["norm_type"],
        "checkpoint_load": load_report,
        "device": str(device),
        "seed": args.seed,
    }
    manifest = write_artifacts(
        args.out_dir, args.frames, rgb_frames, sparse, depths, predictions,
        valid_masks, checkpoint_sha256=file_sha256(args.checkpoint),
        model_config=model_config)
    print("saved %d frame predictions and %d temporal pairs to %s" %
          (len(args.frames), len(args.frames) - 1, args.out_dir), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    main()
