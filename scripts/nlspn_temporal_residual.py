"""Shared contracts and math for NLSPN temporal-residual validation."""

import hashlib
from pathlib import Path

import numpy as np


HEIGHT = 228
WIDTH = 304
SPARSE_COUNT = 500
SPARSE_SEED = 2026
MAX_DEPTH = 10.0
PILOT_CLIPS = (
    (1, 32),
    (282, 313),
    (563, 594),
    (844, 875),
    (1126, 1157),
    (1407, 1438),
    (1688, 1719),
    (1969, 2000),
)


def clip_frame_ids(clips):
    return tuple(
        frame_id
        for start, end in clips
        for frame_id in range(int(start), int(end) + 1)
    )


def clip_pairs(clips):
    return tuple(
        (frame_id, frame_id + 1)
        for start, end in clips
        for frame_id in range(int(start), int(end))
    )


def validate_clip_payload(payload):
    frame_ids = np.asarray(payload["frame_ids"])
    frame_count = frame_ids.size
    expected = {
        "rgb": (frame_count, 3, HEIGHT, WIDTH),
        "sparse": (frame_count, HEIGHT, WIDTH),
        "gt": (frame_count, HEIGHT, WIDTH),
        "valid": (frame_count, HEIGHT, WIDTH),
    }
    for key, shape in expected.items():
        value = np.asarray(payload[key])
        if value.shape != shape:
            raise ValueError("%s must have shape %r" % (key, shape))
    sparse_count = np.count_nonzero(payload["sparse"], axis=(1, 2))
    if not np.all(sparse_count == SPARSE_COUNT):
        raise ValueError("every frame must contain exactly 500 sparse points")
    if not np.isfinite(payload["rgb"]).all():
        raise ValueError("RGB contains non-finite values")
    for key in ("sparse", "gt"):
        if not np.isfinite(np.asarray(payload[key])).all():
            raise ValueError("%s contains non-finite values" % key)
    return payload


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
