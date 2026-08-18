from pathlib import Path

import numpy as np
import pytest

from scripts import spn_sequence_io as sequence_io


def write_canonical(root, sparse_masks=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    base_mask = np.zeros((228, 304), dtype=bool)
    base_mask.flat[:500] = True
    for index, frame_id in enumerate(range(1, 6)):
        mask = base_mask if sparse_masks is None else sparse_masks[index]
        gt = np.full((228, 304), 1.0 + index, dtype=np.float32)
        sparse = np.where(mask, gt, 0.0).astype(np.float32)
        np.savez_compressed(
            root / ("frame_%04d.npz" % frame_id),
            frame_id=np.asarray(frame_id),
            rgb=np.full((3, 228, 304), index / 5.0, dtype=np.float32),
            sparse=sparse,
            gt=gt,
            pred_raw=gt + 0.2,
            pred_clamped=gt + 0.2,
            valid=np.ones((228, 304), dtype=bool),
            abs_err=np.full((228, 304), 0.2, dtype=np.float32),
        )


def test_load_canonical_frames_stacks_exact_five_frame_contract(tmp_path):
    write_canonical(tmp_path)
    data = sequence_io.load_canonical_frames(tmp_path, range(1, 6))
    assert data["frame_ids"].tolist() == [1, 2, 3, 4, 5]
    assert data["rgb"].shape == (5, 3, 228, 304)
    assert data["sparse"].shape == (5, 228, 304)
    assert data["gt"].shape == (5, 228, 304)
    assert data["valid"].shape == (5, 228, 304)
    assert data["cspn_pred_raw"].shape == (5, 228, 304)
    assert all(np.count_nonzero(x) == 500 for x in data["sparse"])
    assert len(data["input_digest"]) == 64


def test_load_canonical_frames_rejects_changed_sparse_coordinates(tmp_path):
    masks = []
    for index in range(5):
        mask = np.zeros((228, 304), dtype=bool)
        mask.flat[index:index + 500] = True
        masks.append(mask)
    write_canonical(tmp_path, masks)
    with pytest.raises(ValueError, match="shared sparse coordinates"):
        sequence_io.load_canonical_frames(tmp_path, range(1, 6))


def test_worker_result_round_trip_and_digest_validation(tmp_path):
    write_canonical(tmp_path / "canonical")
    data = sequence_io.load_canonical_frames(tmp_path / "canonical")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"weights")
    output = tmp_path / "predictions.npz"
    pred = np.full((5, 228, 304), 2.0, dtype=np.float32)
    sequence_io.write_worker_result(
        output,
        "dyspn",
        data["frame_ids"],
        pred,
        data["input_digest"],
        sequence_io.file_sha256(checkpoint),
        {"iteration": 6},
        1.25,
    )
    result = sequence_io.load_worker_result(
        output,
        "dyspn",
        data["frame_ids"],
        data["input_digest"],
        sequence_io.file_sha256(checkpoint),
    )
    np.testing.assert_array_equal(result["pred_raw"], pred)
    assert result["metadata"]["iteration"] == 6


def test_worker_result_rejects_input_digest_mismatch(tmp_path):
    output = tmp_path / "predictions.npz"
    pred = np.full((5, 228, 304), 2.0, dtype=np.float32)
    sequence_io.write_worker_result(
        output, "dyspn", np.arange(1, 6), pred, "a" * 64, "b" * 64,
        {"iteration": 6}, 1.25)
    with pytest.raises(ValueError, match="input digest"):
        sequence_io.load_worker_result(
            output, "dyspn", np.arange(1, 6), "c" * 64, "b" * 64)


def test_frame_and_temporal_metrics_use_valid_pixels():
    gt = np.asarray([[[1.0, 2.0]], [[2.0, 4.0]]], dtype=np.float32)
    pred = np.asarray([[[1.0, 2.0]], [[3.0, 5.0]]], dtype=np.float32)
    valid = np.ones_like(gt, dtype=bool)
    frame = sequence_io.frame_metrics(gt[1], pred[1], valid[1])
    assert frame["rmse"] == pytest.approx(1.0)
    assert frame["mae"] == pytest.approx(1.0)
    assert frame["abs_rel"] == pytest.approx(0.375)
    temporal, maps = sequence_io.temporal_metrics(gt, pred, valid, [1, 2])
    assert temporal[0]["pair"] == "0001->0002"
    assert temporal[0]["rmse"] == pytest.approx(1.0)
    np.testing.assert_array_equal(maps[0]["residual"], [[1.0, 1.0]])
