from PIL import Image
import csv
import numpy as np
import pytest

from scripts import evaluate_nyu_selected_quantization as evaluator
from scripts import plot_nyu_selected_quantization as plotter


def _write_aligned_exports(root, index=3):
    gt = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rgb = np.zeros((2, 2, 3), dtype=np.float32)
    sparse = np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float32)
    identity = evaluator.ordered_evaluation_identity(tuple(range(64)))
    for offset, method in enumerate(evaluator.SELECTED_METHODS):
        evaluator.write_prediction_export(
            root=root,
            model="dyspn",
            method=method,
            sample_index=index,
            evaluation_identity=identity,
            rgb=rgb,
            sparse=sparse,
            gt=gt,
            pred=gt + np.float32(offset * 0.1),
        )
    return identity


def test_aligned_sample_uses_one_depth_and_error_range(tmp_path):
    identity = _write_aligned_exports(tmp_path)

    sample = plotter.load_aligned_sample(
        tmp_path, "dyspn", 3, identity, evaluator.SELECTED_METHODS)
    ranges = plotter.shared_sample_ranges(sample)

    assert ranges.depth_min == pytest.approx(1.0)
    assert ranges.depth_max == pytest.approx(4.9)
    assert ranges.error_min == pytest.approx(0.0)
    assert ranges.error_max == pytest.approx(0.9)


def test_aligned_sample_rejects_changed_rgb_sparse_or_gt(tmp_path):
    identity = _write_aligned_exports(tmp_path)
    path = evaluator.prediction_path(tmp_path, "rtn_w8a8", 3)
    with np.load(path, allow_pickle=False) as source:
        payload = dict((key, source[key]) for key in source.files)
    payload["sparse"][0, 0] = 9.0
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError, match="aligned input"):
        plotter.load_aligned_sample(
            tmp_path, "dyspn", 3, identity,
            evaluator.SELECTED_METHODS)


def test_prediction_panel_contains_all_selected_methods(tmp_path):
    identity = _write_aligned_exports(tmp_path)
    sample = plotter.load_aligned_sample(
        tmp_path, "dyspn", 3, identity, evaluator.SELECTED_METHODS)
    output = tmp_path / "panel.png"

    plotter.render_prediction_panel(sample, output)

    image = Image.open(output).convert("RGB")
    assert image.width > 1000
    assert image.height > 200
    assert any(low != high for low, high in image.getextrema())


def test_metric_figure_uses_pooled_rmse_as_primary(tmp_path):
    path = tmp_path / "aggregate_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "method", "configuration", "pooled_rmse",
            "mean_sample_rmse"))
        writer.writeheader()
        for position, method in enumerate(evaluator.SELECTED_METHODS):
            writer.writerow({
                "method": method,
                "configuration": evaluator.METHOD_LABELS[method],
                "pooled_rmse": 0.5 + position * 0.01,
                "mean_sample_rmse": 0.4 + position * 0.01,
            })
    output = tmp_path / "metric.png"

    plotter.render_metric_comparison(path, output)

    image = Image.open(output).convert("RGB")
    assert image.width > 1000
    assert image.height > 200
    assert any(low != high for low, high in image.getextrema())
