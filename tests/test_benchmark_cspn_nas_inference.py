import pytest
import torch

from scripts import benchmark_cspn_nas_inference as benchmark


def test_parse_csv_choices_preserves_order_and_rejects_unknown_values():
    assert benchmark.parse_csv_choices(
        "fp32,fp16", benchmark.PRECISIONS, "precision") == ["fp32", "fp16"]

    with pytest.raises(ValueError, match="unsupported precision"):
        benchmark.parse_csv_choices(
            "fp32,int4", benchmark.PRECISIONS, "precision")


def test_parse_positive_steps_deduplicates_values():
    assert benchmark.parse_positive_steps("24,12,12,8") == [24, 12, 8]

    with pytest.raises(ValueError, match="positive"):
        benchmark.parse_positive_steps("24,0")


def test_prediction_error_reports_rmse_and_maximum():
    reference = torch.tensor([0.0, 2.0])
    candidate = torch.tensor([1.0, 0.0])

    result = benchmark.prediction_error(reference, candidate)

    assert result["rmse"] == pytest.approx((2.5) ** 0.5)
    assert result["max_abs"] == pytest.approx(2.0)
    assert result["finite"] is True
