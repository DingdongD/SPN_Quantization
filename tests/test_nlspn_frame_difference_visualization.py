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


def make_payload():
    gt = np.full((5, HEIGHT, WIDTH), 2.0, dtype=np.float32)
    valid = np.ones_like(gt, dtype=bool)
    sparse = np.zeros_like(gt)
    sparse.reshape(5, -1)[:, :500] = 2.0
    return {
        "frame_ids": np.arange(1, 6, dtype=np.int32),
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


def make_latency_rows():
    return [
        {"method": method, "frame_id": frame_id, "latency_ms": 1.0}
        for method, frame_id in itertools.product(
            ("full", "zero_flow", "rgb_diff", "global_diff"),
            range(1, 6))
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
