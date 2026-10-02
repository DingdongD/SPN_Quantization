import math

import pytest

from scripts.audit_four_model_rmse_protocol import (
    SampleError, aggregate_sample_errors)


def test_aggregate_reports_mean_sample_and_pooled_pixel_rmse():
    rows = (
        SampleError(squared_error_sum=1.0, valid_pixels=1),
        SampleError(squared_error_sum=9.0, valid_pixels=3),
    )

    result = aggregate_sample_errors(rows)

    assert result["sample_count"] == 2
    assert result["valid_pixels"] == 4
    assert result["mean_sample_rmse_m"] == pytest.approx(
        (1.0 + math.sqrt(3.0)) / 2.0)
    assert result["pooled_pixel_rmse_m"] == pytest.approx(math.sqrt(2.5))


def test_aggregate_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        aggregate_sample_errors(())
