"""Deterministic motion-window selection for cross-scene NLSPN evaluation."""

from pathlib import Path
import re

import numpy as np
from PIL import Image


SCENES = (
    "bedroom_ir",
    "livingroom_ir",
    "room3",
    "room4",
    "room6",
    "room7",
)

THUMBNAIL_SIZE = (76, 57)
_RGB_PATTERN = re.compile(r"^(\d{4})\.jpg$")
_DEPTH_PATTERN = re.compile(r"^Image(\d{4})\.exr$")
FIXED_THRESHOLD = 2.0 / 255.0
FIXED_DILATION_RADIUS = 8


def require_fixed_configs(configs):
    """Validate and serialize the approved conservative cache configs."""
    if set(configs) != {"rgb_diff", "global_diff"}:
        raise ValueError("fixed configs require rgb_diff and global_diff")
    result = {}
    for variant in ("rgb_diff", "global_diff"):
        config = configs[variant]
        if getattr(config, "variant", None) != variant:
            raise ValueError("fixed config variant mismatch for {}".format(variant))
        threshold = float(getattr(config, "threshold", float("nan")))
        radius = int(getattr(config, "dilation_radius", -1))
        if (not np.isfinite(threshold) or
                abs(threshold - FIXED_THRESHOLD) > 1e-15):
            raise ValueError("fixed config threshold must equal 2/255")
        if radius != FIXED_DILATION_RADIUS:
            raise ValueError("fixed config dilation radius must equal 8")
        result[variant] = {
            "threshold": threshold,
            "dilation_radius": radius,
        }
    return result


def _validated_ids(frame_ids):
    ids = tuple(frame_ids)
    if not ids or any(not isinstance(item, (int, np.integer)) for item in ids):
        raise ValueError("frame IDs must be positive integers")
    ids = tuple(int(item) for item in ids)
    if any(item <= 0 for item in ids):
        raise ValueError("frame IDs must be positive")
    if tuple(sorted(ids)) != ids:
        raise ValueError("frame IDs must be sorted")
    if len(set(ids)) != len(ids):
        raise ValueError("frame IDs must be unique")
    return ids


def select_motion_window(frame_ids, thumbnails):
    """Choose the highest-mean-MAD consecutive five-frame window."""
    ids = _validated_ids(frame_ids)
    arrays = {}
    shape = None
    for frame_id in ids:
        if frame_id not in thumbnails:
            raise ValueError("missing thumbnail for frame {}".format(frame_id))
        array = np.asarray(thumbnails[frame_id])
        if array.dtype != np.float32:
            raise ValueError("thumbnails must use float32")
        if array.ndim != 2:
            raise ValueError("thumbnails must be grayscale 2-D arrays")
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise ValueError("thumbnails must have the same shape")
        if not np.all(np.isfinite(array)):
            raise ValueError("thumbnails must contain finite values")
        if np.any(array < 0.0) or np.any(array > 1.0):
            raise ValueError("thumbnails must lie in [0,1]")
        arrays[frame_id] = array

    candidates = []
    for start_index in range(max(0, len(ids) - 4)):
        window = ids[start_index:start_index + 5]
        if any(right != left + 1 for left, right in zip(window, window[1:])):
            continue
        pair_scores = [
            float(np.mean(np.abs(arrays[right] - arrays[left]), dtype=np.float64))
            for left, right in zip(window, window[1:])
        ]
        candidates.append({
            "frame_ids": list(window),
            "pair_scores": pair_scores,
            "motion_score": float(np.mean(pair_scores, dtype=np.float64)),
        })

    if not candidates:
        raise ValueError("no five consecutive complete frames are available")
    return min(
        candidates,
        key=lambda item: (-item["motion_score"], item["frame_ids"][0]),
    )


def _indexed_files(directory, pattern, label):
    if not directory.is_dir():
        raise ValueError("missing {} directory: {}".format(label, directory))
    indexed = {}
    for path in directory.iterdir():
        match = pattern.match(path.name)
        if match is None:
            continue
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError("empty {} file: {}".format(label, path))
        frame_id = int(match.group(1))
        if frame_id in indexed:
            raise ValueError("duplicate {} frame {}".format(label, frame_id))
        indexed[frame_id] = path
    if not indexed:
        raise ValueError("no {} frames found in {}".format(label, directory))
    return indexed


def _load_thumbnail(path):
    try:
        with Image.open(str(path)) as image:
            image = image.convert("L")
            resampling = getattr(Image, "Resampling", Image).BILINEAR
            image = image.resize(THUMBNAIL_SIZE, resample=resampling)
            return np.asarray(image, dtype=np.float32) / np.float32(255.0)
    except Exception as error:
        raise ValueError("unreadable RGB JPEG: {}".format(path)) from error


def scan_scene(scene_root):
    """Scan one ``rgb``/``depth`` scene and select its motion-rich window."""
    root = Path(scene_root)
    if not root.is_dir():
        raise ValueError("missing scene directory: {}".format(root))
    rgb = _indexed_files(root / "rgb", _RGB_PATTERN, "RGB")
    depth = _indexed_files(root / "depth", _DEPTH_PATTERN, "depth")
    ids = tuple(sorted(set(rgb).intersection(depth)))
    thumbnails = {frame_id: _load_thumbnail(rgb[frame_id]) for frame_id in ids}
    result = select_motion_window(ids, thumbnails)
    result.update({
        "scene": root.name,
        "start_frame": result["frame_ids"][0],
        "end_frame": result["frame_ids"][-1],
    })
    return result
