"""Data contracts for scene-disjoint NLSPN fine-tuning."""

import csv
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


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
