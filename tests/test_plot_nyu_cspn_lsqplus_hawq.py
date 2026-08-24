from PIL import Image
import numpy as np

from scripts import evaluate_nyu_cspn_lsqplus_hawq as evaluator
from scripts import plot_nyu_cspn_lsqplus_hawq as plotter


def _write_shards(root, indices):
    gt = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    for column, config in enumerate(evaluator.CONFIGURATIONS):
        output = root / "shards" / config
        output.mkdir(parents=True)
        for index in indices:
            np.savez_compressed(
                output / ("sample_%05d.npz" % index),
                sample_index=np.int64(index),
                gt=gt,
                pred=gt + np.float32(column * 0.01),
                rgb=np.zeros((2, 2, 3), dtype=np.float32),
                sparse=np.zeros((2, 2), dtype=np.float32),
            )


def test_plotter_generates_nonblank_unified_figures(tmp_path):
    _write_shards(tmp_path, tuple(range(8)))
    result = evaluator.aggregate_shards(tmp_path, tuple(range(8)))
    evaluator.write_aggregation(tmp_path, result)

    plotter.generate_figures(tmp_path, tuple(range(8)), detail_samples=8)

    expected = (
        "quantization_rmse_comparison.png",
        "prediction_details.png",
        "prediction_contact_sheet.png",
    )
    for name in expected:
        image = Image.open(tmp_path / "figures" / name).convert("RGB")
        extrema = image.getextrema()
        assert image.width > 100 and image.height > 100
        assert any(low != high for low, high in extrema)


def test_display_labels_are_exact_and_horizontal():
    assert plotter.DISPLAY_LABELS == (
        "FP32", "PA-RTN W4A4", "PA-RTN W6A6",
        "LSQ+ W4A4", "LSQ+ W6A6", "HAWQ Mixed<=6",
        "Mixed Task-aware QAT")
    assert plotter.XTICK_ROTATION == 0
