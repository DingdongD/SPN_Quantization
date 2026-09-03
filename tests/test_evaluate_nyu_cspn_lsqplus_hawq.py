import numpy as np
import pytest

from scripts import evaluate_nyu_cspn_lsqplus_hawq as evaluator


def test_formal_matrix_is_fixed():
    assert evaluator.CONFIGURATIONS == (
        "FP32",
        "PA_RTN_W4A4",
        "PA_RTN_W6A6",
        "LSQPLUS_W4A4",
        "LSQPLUS_W6A6",
        "HAWQ_MIXED_LE6",
        "MIXED_TASK_AWARE_QAT",
    )


def _write_shards(root, indices=(0, 1)):
    gt = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    for column, config in enumerate(evaluator.CONFIGURATIONS):
        output = root / "shards" / config
        output.mkdir(parents=True)
        for index in indices:
            pred = gt + np.float32(column * 0.01)
            np.savez_compressed(
                output / ("sample_%05d.npz" % index),
                sample_index=np.int64(index),
                gt=gt,
                pred=pred,
                rgb=np.zeros((2, 2, 3), dtype=np.float32),
                sparse=np.zeros((2, 2), dtype=np.float32),
            )


def test_aggregate_rejects_missing_prediction(tmp_path):
    _write_shards(tmp_path)
    (tmp_path / "shards" / "LSQPLUS_W4A4" /
     "sample_00001.npz").unlink()

    with pytest.raises(RuntimeError, match="coverage"):
        evaluator.aggregate_shards(tmp_path, (0, 1))


def test_aggregate_records_nonpositive_prediction_without_clamping(tmp_path):
    _write_shards(tmp_path)
    path = tmp_path / "shards" / "HAWQ_MIXED_LE6" / "sample_00000.npz"
    with np.load(path) as payload:
        values = dict((key, payload[key]) for key in payload.files)
    values["pred"][0, 0] = 0.0
    np.savez_compressed(path, **values)

    result = evaluator.aggregate_shards(tmp_path, (0, 1))

    metrics = dict((row["configuration"], row) for row in result.metrics)
    assert metrics["HAWQ_MIXED_LE6"]["nonpositive_pixels"] == 1
    assert metrics["HAWQ_MIXED_LE6"]["nonpositive_ratio"] == pytest.approx(
        1.0 / 8.0)


def test_aggregate_computes_paired_fp_loss(tmp_path):
    _write_shards(tmp_path)

    result = evaluator.aggregate_shards(tmp_path, (0, 1))

    metrics = dict((row["configuration"], row) for row in result.metrics)
    relative = dict(
        (row["configuration"], row) for row in result.relative_fp_loss)
    assert metrics["FP32"]["RMSE"] == 0.0
    assert metrics["LSQPLUS_W4A4"]["RMSE"] == pytest.approx(
        0.03, abs=1e-6)
    assert relative["LSQPLUS_W4A4"]["RMSE_delta_m"] == pytest.approx(
        0.03, abs=1e-6)


def test_aggregate_rejects_gt_identity_change(tmp_path):
    _write_shards(tmp_path)
    path = tmp_path / "shards" / "PA_RTN_W6A6" / "sample_00000.npz"
    with np.load(path) as payload:
        values = dict((key, payload[key]) for key in payload.files)
    values["gt"][0, 0] = 9.0
    np.savez_compressed(path, **values)

    with pytest.raises(RuntimeError, match="GT"):
        evaluator.aggregate_shards(tmp_path, (0, 1))
