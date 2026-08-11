import numpy as np
import pytest

from scripts.plot_nyu_qdrop_w4a4 import (
    select_visual_samples,
    validate_prediction_arrays,
)


def test_selects_median_and_worst_seed_mean_error_samples():
    rows = [
        {"method": "qdrop", "seed": seed, "sample_index": index,
         "RMSE": value}
        for seed in (1005, 1006, 1007)
        for index, value in ((3, 0.1), (7, 0.3), (9, 0.2))
    ]

    assert select_visual_samples(rows, (1005, 1006, 1007)) == (9, 7)


def test_prediction_arrays_require_equal_finite_shapes():
    arrays = {
        "rgb": np.ones((2, 3, 3), dtype=np.float32),
        "sparse": np.ones((2, 3), dtype=np.float32),
        "gt": np.ones((2, 3), dtype=np.float32),
        "fp32": np.ones((2, 3), dtype=np.float32),
        "rtn": np.ones((2, 3), dtype=np.float32),
        "brecq": np.ones((2, 3), dtype=np.float32),
        "qdrop_1005": np.ones((2, 3), dtype=np.float32),
    }
    validate_prediction_arrays(arrays)

    broken = dict(arrays)
    broken["rtn"] = np.ones((3, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        validate_prediction_arrays(broken)
    broken = dict(arrays)
    broken["brecq"] = np.full((2, 3), np.nan, dtype=np.float32)
    with pytest.raises(ValueError):
        validate_prediction_arrays(broken)
