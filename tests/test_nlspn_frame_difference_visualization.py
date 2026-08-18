import csv
import itertools

import numpy as np
import pytest

from scripts import nlspn_frame_difference_cache as cache
from scripts import nlspn_frame_difference_visualization as visual


HEIGHT = 228
WIDTH = 304


def write_sweep(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "variant", "threshold", "dilation_radius", "selected"])
        writer.writeheader()
        writer.writerows(rows)


def sweep_row(variant, threshold, radius, selected):
    return {
        "variant": variant,
        "threshold": threshold,
        "dilation_radius": radius,
        "selected": selected,
    }


def make_payload(frame_ids=range(1, 6)):
    gt = np.full((5, HEIGHT, WIDTH), 2.0, dtype=np.float32)
    valid = np.ones_like(gt, dtype=bool)
    sparse = np.zeros_like(gt)
    sparse.reshape(5, -1)[:, :500] = 2.0
    return {
        "frame_ids": np.asarray(list(frame_ids), dtype=np.int32),
        "rgb": np.zeros((5, 3, HEIGHT, WIDTH), dtype=np.float32),
        "sparse": sparse,
        "gt": gt,
        "valid": valid,
    }


def make_predictions(payload=None):
    payload = make_payload() if payload is None else payload
    return {
        name: payload["gt"] + index * 0.1
        for index, name in enumerate(visual.METHOD_ORDER)
    }


def make_latency_rows(frame_ids=range(1, 6)):
    return [
        {"method": method, "frame_id": frame_id, "latency_ms": 1.0}
        for method, frame_id in itertools.product(
            ("full", "zero_flow", "rgb_diff", "global_diff"),
            frame_ids)
    ]


def make_raft_predictions(payload=None):
    payload = make_payload() if payload is None else payload
    predictions = make_predictions(payload)
    predictions["raft_gop2"] = payload["gt"] + np.float32(0.4)
    return predictions


def make_raft_latency_rows(frame_ids=range(1, 6)):
    return make_latency_rows(frame_ids) + [
        {"method": "raft_gop2", "frame_id": frame_id, "latency_ms": 3.0}
        for frame_id in frame_ids
    ]


def test_load_selected_configs_requires_exact_rgb_and_global_rows(tmp_path):
    path = tmp_path / "threshold_sweep.csv"
    write_sweep(path, [
        sweep_row("rgb_diff", 2.0 / 255.0, 8, True),
        sweep_row("global_diff", 2.0 / 255.0, 8, True),
    ])
    configs, digest = visual.load_selected_configs(path)
    assert configs["rgb_diff"] == cache.CacheConfig(
        "rgb_diff", 2.0 / 255.0, 8)
    assert configs["global_diff"] == cache.CacheConfig(
        "global_diff", 2.0 / 255.0, 8)
    assert len(digest) == 64


def test_load_selected_configs_rejects_ambiguous_rows(tmp_path):
    path = tmp_path / "threshold_sweep.csv"
    rows = [sweep_row("rgb_diff", 2.0 / 255.0, 8, True)] * 2
    rows.append(sweep_row("global_diff", 2.0 / 255.0, 8, True))
    write_sweep(path, rows)
    with pytest.raises(ValueError, match="exactly one"):
        visual.load_selected_configs(path)


def test_collect_frame_metrics_emits_twenty_rows():
    payload = make_payload()
    rows = visual.collect_frame_metrics(
        payload, make_predictions(payload), make_latency_rows())
    assert len(rows) == 20
    assert {(row["method"], row["frame_id"]) for row in rows} == set(
        itertools.product(visual.METHOD_ORDER, range(1, 6)))
    zero_rows = [row for row in rows if row["method"] == "zero_flow"]
    assert [row["frame_kind"] for row in zero_rows] == [
        "I", "P", "I", "P", "I"]
    assert all(row["sparse_count"] == 500 for row in rows)


def test_payload_accepts_any_five_consecutive_ids():
    frame_ids = range(282, 287)
    payload = make_payload(frame_ids)

    visual._validate_payload(payload)
    rows = visual.collect_frame_metrics(
        payload, make_predictions(payload), make_latency_rows(frame_ids))

    assert [row["frame_id"] for row in rows[:5]] == list(frame_ids)


def test_payload_still_rejects_nonconsecutive_ids():
    payload = make_payload((1, 2, 4, 5, 6))

    with pytest.raises(ValueError, match="consecutive"):
        visual._validate_payload(payload)


def test_depth_grid_has_fixed_limits_and_external_colorbar(tmp_path):
    payload = make_payload()
    figure = visual.render_depth_comparison(
        tmp_path / "depth.png", payload, make_predictions(payload))
    assert len(figure.axes) == 26
    panels = figure.axes[:25]
    assert all(axis.images[0].get_clim() == (0.0, 10.0)
               for axis in panels)
    assert max(axis.get_position().x1 for axis in panels) + 0.01 <= \
        figure.axes[-1].get_position().x0


def test_error_grid_uses_one_global_99th_percentile(tmp_path):
    payload = make_payload()
    predictions = make_predictions(payload)
    expected = visual.common_error_max(payload, predictions)
    figure = visual.render_error_comparison(
        tmp_path / "error.png", payload, predictions)
    assert len(figure.axes) == 21
    assert all(axis.images[0].get_clim() == (0.0, expected)
               for axis in figure.axes[:20])


def test_five_method_metrics_and_grid_geometry(tmp_path):
    frame_ids = range(282, 287)
    payload = make_payload(frame_ids)
    predictions = make_raft_predictions(payload)

    rows = visual.collect_frame_metrics(
        payload, predictions, make_raft_latency_rows(frame_ids))
    depth = visual.render_depth_comparison(
        tmp_path / "depth.png", payload, predictions)
    error = visual.render_error_comparison(
        tmp_path / "error.png", payload, predictions)

    assert visual.RAFT_METHOD_ORDER[-1] == "raft_gop2"
    assert len(rows) == 25
    assert [row["method"] for row in rows[20:]] == ["raft_gop2"] * 5
    assert len(depth.axes) == 31
    assert len(error.axes) == 26
    assert all(axis.images[0].get_clim() == (0.0, 10.0)
               for axis in depth.axes[:30])


def test_prediction_schema_rejects_partial_raft_method():
    predictions = make_predictions()
    predictions["unknown_flow"] = predictions["full"].copy()

    with pytest.raises(ValueError, match="approved method schema"):
        visual._validate_predictions(predictions)


def test_write_artifacts_creates_exact_six_files(tmp_path):
    payload = make_payload()
    predictions = make_predictions(payload)
    metrics = visual.collect_frame_metrics(
        payload, predictions, make_latency_rows())
    completed = visual.write_artifacts(
        tmp_path, payload, predictions, metrics,
        {"checkpoint_sha256": "checkpoint", "complete": False})
    assert completed["complete"] is True
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        visual.FINAL_ARTIFACTS)
    with np.load(tmp_path / "predictions.npz", allow_pickle=False) as item:
        assert item["full"].shape == (5, HEIGHT, WIDTH)
        assert item["global_diff"].shape == (5, HEIGHT, WIDTH)
        assert item["rgb"].shape == (5, 3, HEIGHT, WIDTH)
    assert completed["error_vmax_m"] == pytest.approx(
        visual.common_error_max(payload, predictions))


def test_write_artifacts_records_five_method_order_and_archive(tmp_path):
    payload = make_payload()
    predictions = make_raft_predictions(payload)
    metrics = visual.collect_frame_metrics(
        payload, predictions, make_raft_latency_rows())

    completed = visual.write_artifacts(
        tmp_path, payload, predictions, metrics,
        {"checkpoint_sha256": "checkpoint"})

    assert completed["method_order"] == list(visual.RAFT_METHOD_ORDER)
    with np.load(tmp_path / "predictions.npz", allow_pickle=False) as item:
        assert set(item.files) == {
            "frame_ids", "rgb", "sparse", "gt", "valid",
            *visual.RAFT_METHOD_ORDER}
        assert item["raft_gop2"].shape == (5, HEIGHT, WIDTH)
