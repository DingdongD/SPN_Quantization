import numpy as np
import pytest

from scripts import export_nlspn_invalid_region_predictions as exporter


def test_depth_to_millimetres_rounds_and_clips():
    depth = np.array(
        [[-1.0, 0.0, 1.234, 1.2346, 10.0, 11.0]], dtype=np.float32)
    actual = exporter.depth_to_millimetres(depth)
    assert actual.dtype == np.uint16
    assert actual.tolist() == [[0, 0, 1234, 1235, 10000, 10000]]


def test_compose_fill_uses_gt_only_where_valid():
    gt = np.array([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    valid = np.array([[True, False], [True, False]])
    prediction = np.array([[8.0, 8.1], [8.2, 8.3]], dtype=np.float32)
    actual = exporter.compose_gt_with_prediction(gt, valid, prediction)
    expected = np.array([[1.0, 8.1], [3.0, 8.3]], dtype=np.float32)
    np.testing.assert_array_equal(actual, expected)


def test_invalid_mask_is_white_only_for_invalid_gt():
    valid = np.array([[True, False], [False, True]])
    actual = exporter.invalid_mask(valid)
    assert actual.dtype == np.uint8
    assert actual.tolist() == [[0, 255], [255, 0]]


def test_colorize_depth_has_fixed_zero_to_ten_metre_scale():
    depth = np.array([[0.0, 5.0, 10.0, -1.0, 11.0]], dtype=np.float32)
    actual = exporter.colorize_depth(depth)
    assert actual.shape == (1, 5, 3)
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual[0, 0], actual[0, 3])
    np.testing.assert_array_equal(actual[0, 2], actual[0, 4])
    assert not np.array_equal(actual[0, 0], actual[0, 1])
    assert not np.array_equal(actual[0, 1], actual[0, 2])


@pytest.mark.parametrize(
    "function_name", ["depth_to_millimetres", "colorize_depth"])
def test_prediction_products_reject_nonfinite_values(function_name):
    function = getattr(exporter, function_name)
    with pytest.raises(ValueError, match="finite"):
        function(np.array([[np.nan]], dtype=np.float32))
