"""Data contracts for scene-disjoint NLSPN fine-tuning."""

import csv
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
from PIL import Image


RESIZE_HEIGHT = 240
CROP_SHAPE = (228, 304)

SPLIT_SCENES = {
    "train": (
        "BeachApartmentInterior_My_ir",
        "bedroom_ir",
        "livingroom_ir",
        "room4",
    ),
    "val": ("room6",),
    "test": ("room3", "room7"),
}

SCENE_FRAME_COUNTS = {
    "BeachApartmentInterior_My_ir": 2000,
    "bedroom_ir": 2000,
    "livingroom_ir": 4000,
    "room3": 4000,
    "room4": 2000,
    "room6": 2000,
    "room7": 4000,
}

MANIFEST_FIELDS = ("split", "scene", "frame_id", "rgb_path", "depth_path")

_RGB_PATTERN = re.compile(r"^(\d{4})\.jpg$")
_DEPTH_PATTERN = re.compile(r"^Image(\d{4})\.exr$")


def canonical_row(split, scene, frame_id):
    frame_id = int(frame_id)
    return {
        "split": split,
        "scene": scene,
        "frame_id": frame_id,
        "rgb_path": "{}/rgb/{:04d}.jpg".format(scene, frame_id),
        "depth_path": "{}/depth/Image{:04d}.exr".format(scene, frame_id),
    }


def _expected_rows(split):
    return [
        canonical_row(split, scene, frame_id)
        for scene in SPLIT_SCENES[split]
        for frame_id in range(1, SCENE_FRAME_COUNTS[scene] + 1)
    ]


def _identity(row):
    try:
        return row["scene"], int(row["frame_id"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("manifest row has an invalid identity")


def validate_manifest(rows, split):
    if split not in SPLIT_SCENES:
        raise ValueError("unknown manifest split: {}".format(split))
    rows = list(rows)

    identities = [_identity(row) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("manifest contains a duplicate frame identity")

    expected = _expected_rows(split)
    if len(rows) != len(expected):
        raise ValueError(
            "{} manifest row count is {}, expected {}".format(
                split, len(rows), len(expected)))

    expected_identities = [_identity(row) for row in expected]
    if identities != expected_identities:
        raise ValueError("{} manifest is not in canonical order".format(split))

    for row, canonical in zip(rows, expected):
        if tuple(row.keys()) != MANIFEST_FIELDS and set(row) != set(MANIFEST_FIELDS):
            raise ValueError("manifest row fields differ from the contract")
        if row.get("split") != canonical["split"]:
            raise ValueError("manifest row has the wrong split")
        if (row.get("rgb_path") != canonical["rgb_path"] or
                row.get("depth_path") != canonical["depth_path"]):
            raise ValueError("manifest row has a noncanonical path")
        if dict(row) != canonical:
            raise ValueError("manifest row differs from its canonical value")
    return rows


def validate_split_isolation(manifests):
    if tuple(manifests) != tuple(SPLIT_SCENES):
        raise ValueError("manifest splits are not in canonical order")
    owners = {}
    for split, rows in manifests.items():
        for row in rows:
            owners.setdefault(row["scene"], set()).add(split)
    overlap = {
        scene: sorted(splits)
        for scene, splits in owners.items()
        if len(splits) > 1
    }
    if overlap:
        raise ValueError("scene split overlap detected: {}".format(overlap))


def _parsed_ids(directory, pattern, scene, kind):
    if not directory.is_dir():
        raise ValueError("{} {} directory is missing".format(scene, kind))
    ids = set()
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match is None:
            raise ValueError(
                "{} {} directory contains noncanonical file {}".format(
                    scene, kind, path.name))
        frame_id = int(match.group(1))
        if frame_id in ids:
            raise ValueError(
                "{} {} has duplicate frame id {:04d}".format(
                    scene, kind, frame_id))
        ids.add(frame_id)
    return ids


def _validate_scene_files(root, scene):
    expected = set(range(1, SCENE_FRAME_COUNTS[scene] + 1))
    rgb_ids = _parsed_ids(root / scene / "rgb", _RGB_PATTERN, scene, "rgb")
    depth_ids = _parsed_ids(
        root / scene / "depth", _DEPTH_PATTERN, scene, "depth")

    observed = rgb_ids | depth_ids
    extra = sorted(observed - expected)
    if extra:
        raise ValueError(
            "{} has extra frame id {:04d}".format(scene, extra[0]))

    paired = rgb_ids & depth_ids
    missing_pair = sorted(observed - paired)
    if missing_pair:
        raise ValueError(
            "{} frame {:04d} is missing its rgb/depth pair".format(
                scene, missing_pair[0]))
    missing = sorted(expected - paired)
    if missing:
        raise ValueError(
            "{} frame {:04d} is missing its rgb/depth pair".format(
                scene, missing[0]))


def build_manifests(data_root):
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise ValueError("data root is not a directory: {}".format(root))

    manifests = {}
    for split, scenes in SPLIT_SCENES.items():
        for scene in scenes:
            _validate_scene_files(root, scene)
        manifests[split] = _expected_rows(split)
        validate_manifest(manifests[split], split)
    validate_split_isolation(manifests)
    return manifests


def _validate_all(manifests):
    if tuple(manifests) != tuple(SPLIT_SCENES):
        raise ValueError("manifest splits are not in canonical order")
    for split in SPLIT_SCENES:
        validate_manifest(manifests[split], split)
    validate_split_isolation(manifests)


def write_manifests(manifests, output_dir):
    _validate_all(manifests)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in SPLIT_SCENES:
        destination = output_dir / "{}.csv".format(split)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".{}-".format(split), suffix=".tmp", dir=str(output_dir))
        try:
            with os.fdopen(descriptor, "w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerows(manifests[split])
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, str(destination))
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        paths[split] = destination
    return paths


def _read_manifest(path, split):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError("manifest CSV fields differ from the contract")
        rows = []
        for row in reader:
            try:
                row["frame_id"] = int(row["frame_id"])
            except (TypeError, ValueError):
                raise ValueError("manifest frame_id is not an integer")
            rows.append(row)
    validate_manifest(rows, split)
    return rows


def _require_paths_within_root(rows, root):
    for row in rows:
        for field in ("rgb_path", "depth_path"):
            path = (root / row[field]).resolve()
            if root != path and root not in path.parents:
                raise ValueError("manifest path escapes the approved data root")
            if not path.is_file():
                raise ValueError(
                    "{} frame {:04d} is missing its rgb/depth pair".format(
                        row["scene"], row["frame_id"]))


def read_manifests(manifest_dir, data_root):
    manifest_dir = Path(manifest_dir)
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise ValueError("data root is not a directory: {}".format(root))
    manifests = {}
    for split in SPLIT_SCENES:
        rows = _read_manifest(manifest_dir / "{}.csv".format(split), split)
        _require_paths_within_root(rows, root)
        manifests[split] = [
            canonical_row(split, row["scene"], row["frame_id"])
            for row in rows
        ]
    _validate_all(manifests)
    return manifests


def canonical_manifest_sha256(manifests):
    _validate_all(manifests)
    canonical = {
        split: [
            {field: row[field] for field in MANIFEST_FIELDS}
            for row in manifests[split]
        ]
        for split in SPLIT_SCENES
    }
    payload = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_canonical_digest(manifests, expected_sha256):
    actual = canonical_manifest_sha256(manifests)
    if actual != str(expected_sha256).lower():
        raise ValueError(
            "manifest digest mismatch: expected {}, got {}".format(
                expected_sha256, actual))
    return actual


def read_rgb_jpg(path):
    with Image.open(str(path)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def depth_from_exr_array(array):
    depth = np.asarray(array)
    if depth.ndim == 2:
        return depth.astype(np.float32, copy=False)
    if depth.ndim != 3 or depth.shape[2] != 3:
        raise ValueError("EXR depth has an unsupported shape: {}".format(depth.shape))
    first = depth[:, :, 0]
    if not (
            np.allclose(first, depth[:, :, 1], rtol=0.0, atol=0.0, equal_nan=True)
            and np.allclose(
                first, depth[:, :, 2], rtol=0.0, atol=0.0, equal_nan=True)):
        raise ValueError("EXR depth channels are not equal")
    return first.astype(np.float32, copy=False)


def read_depth_exr(path):
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if array is None:
        raise ValueError("failed to read EXR depth: {}".format(path))
    return depth_from_exr_array(array)


def sanitize_depth(depth):
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth must be a two-dimensional array")
    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= 10.0)
    gt = np.where(valid, depth, 0.0).astype(np.float32, copy=False)
    return gt, valid


def _resize_and_center_crop(array, interpolation):
    height, width = array.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("input array has an empty spatial dimension")
    resized_width = int(round(float(width) * RESIZE_HEIGHT / float(height)))
    crop_height, crop_width = CROP_SHAPE
    if resized_width < crop_width:
        raise ValueError("resized input is narrower than the required crop")
    resized = cv2.resize(
        array, (resized_width, RESIZE_HEIGHT), interpolation=interpolation)
    top = (RESIZE_HEIGHT - crop_height) // 2
    left = (resized_width - crop_width) // 2
    return resized[top:top + crop_height, left:left + crop_width]


def preprocess_arrays(rgb, depth):
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("RGB input must be an HxWx3 uint8 array")
    if rgb.shape[:2] != np.asarray(depth).shape[:2]:
        raise ValueError("RGB and depth spatial shapes do not match")

    gt, valid = sanitize_depth(depth)
    rgb_out = _resize_and_center_crop(rgb, cv2.INTER_LINEAR)
    gt_out = _resize_and_center_crop(gt, cv2.INTER_NEAREST)
    valid_out = _resize_and_center_crop(
        valid.astype(np.uint8), cv2.INTER_NEAREST).astype(bool)
    gt_out = np.where(valid_out, gt_out, 0.0).astype(np.float32, copy=False)
    rgb_out = rgb_out.astype(np.float32) / 255.0
    rgb_out = np.transpose(rgb_out, (2, 0, 1))
    return (
        np.ascontiguousarray(rgb_out),
        np.ascontiguousarray(gt_out),
        np.ascontiguousarray(valid_out),
    )


def sample_seed(split, scene, frame_id, epoch, seed):
    effective_epoch = int(epoch) if split == "train" else 0
    payload = "{}|{}|{}|{}|{}".format(
        seed, split, scene, int(frame_id), effective_epoch)
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little")


def build_sparse(gt, valid, split, scene, frame_id, epoch, seed):
    gt = np.asarray(gt, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if gt.shape != valid.shape:
        raise ValueError("ground truth and validity shapes do not match")
    candidates = np.flatnonzero(valid)
    if candidates.size < 500:
        raise ValueError("preprocessed sample has fewer than 500 valid pixels")
    rng = np.random.default_rng(
        sample_seed(split, scene, frame_id, epoch, seed))
    chosen = rng.choice(candidates, 500, replace=False)
    sparse = np.zeros(gt.size, dtype=np.float32)
    sparse[chosen] = gt.reshape(-1)[chosen]
    return sparse.reshape(gt.shape)


def _augmentation_parameters(split, scene, frame_id, epoch, seed):
    if split != "train":
        return False, 1.0, 1.0, 1.0
    rng = np.random.default_rng(
        sample_seed(split, scene, frame_id, epoch, seed))
    return (
        bool(rng.random() < 0.5),
        float(rng.uniform(0.9, 1.1)),
        float(rng.uniform(0.9, 1.1)),
        float(rng.uniform(0.9, 1.1)),
    )


def augment_arrays(rgb, gt, valid, split, scene, frame_id, epoch, seed):
    rgb_out = np.asarray(rgb, dtype=np.float32).copy()
    gt_out = np.asarray(gt, dtype=np.float32).copy()
    valid_out = np.asarray(valid, dtype=bool).copy()
    if rgb_out.shape[0] != 3 or rgb_out.shape[1:] != gt_out.shape:
        raise ValueError("augmentation RGB and depth shapes do not match")
    if gt_out.shape != valid_out.shape:
        raise ValueError("augmentation depth and validity shapes do not match")

    flip, brightness, contrast, saturation = _augmentation_parameters(
        split, scene, frame_id, epoch, seed)
    if split != "train":
        return rgb_out, gt_out, valid_out
    if flip:
        rgb_out = rgb_out[:, :, ::-1]
        gt_out = gt_out[:, ::-1]
        valid_out = valid_out[:, ::-1]

    rgb_out *= brightness
    channel_mean = rgb_out.mean(axis=(1, 2), keepdims=True)
    rgb_out = channel_mean + contrast * (rgb_out - channel_mean)
    gray = (
        0.2989 * rgb_out[0:1] +
        0.5870 * rgb_out[1:2] +
        0.1140 * rgb_out[2:3])
    rgb_out = gray + saturation * (rgb_out - gray)
    rgb_out = np.clip(rgb_out, 0.0, 1.0).astype(np.float32, copy=False)
    return (
        np.ascontiguousarray(rgb_out),
        np.ascontiguousarray(gt_out),
        np.ascontiguousarray(valid_out),
    )


def _resolve_sample_path(root, relative_path):
    path = (root / relative_path).resolve()
    if root != path and root not in path.parents:
        raise ValueError("sample path escapes the approved data root")
    if not path.is_file():
        raise ValueError("sample file does not exist: {}".format(path))
    return path


class SceneDepthDataset(torch.utils.data.Dataset):
    def __init__(self, rows, data_root, split, seed=2026):
        if split not in SPLIT_SCENES:
            raise ValueError("unknown dataset split: {}".format(split))
        self.rows = []
        for row in rows:
            identity = _identity(row)
            canonical = canonical_row(split, identity[0], identity[1])
            if dict(row) != canonical:
                raise ValueError("dataset row is not canonical for its split")
            self.rows.append(canonical)
        self.root = Path(data_root).resolve()
        if not self.root.is_dir():
            raise ValueError("data root is not a directory: {}".format(self.root))
        self.split = split
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        row = self.rows[index]
        rgb_path = _resolve_sample_path(self.root, row["rgb_path"])
        depth_path = _resolve_sample_path(self.root, row["depth_path"])
        rgb, gt, valid = preprocess_arrays(
            read_rgb_jpg(rgb_path), read_depth_exr(depth_path))
        rgb, gt, valid = augment_arrays(
            rgb, gt, valid, self.split, row["scene"], row["frame_id"],
            self.epoch, self.seed)
        sparse = build_sparse(
            gt, valid, self.split, row["scene"], row["frame_id"],
            self.epoch, self.seed)
        return {
            "rgb": torch.from_numpy(rgb),
            "dep": torch.from_numpy(sparse[None]),
            "gt": torch.from_numpy(gt[None]),
            "valid": torch.from_numpy(valid[None]),
            "scene": row["scene"],
            "frame_id": row["frame_id"],
        }
