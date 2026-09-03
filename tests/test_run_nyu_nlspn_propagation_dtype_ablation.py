import pytest
import torch

from scripts.run_nyu_nlspn_propagation_dtype_ablation import (
    MODE_NAMES,
    _aggregate,
    _metrics,
    _state_rows,
)


def test_ablation_modes_are_fixed_and_include_all_propagation_variants():
    assert MODE_NAMES == (
        "PA_W8A8",
        "W8A8_FP32_PROP",
        "W8A8_BF16_STATE",
        "W8A8_FP16_STATE",
    )


def test_metrics_and_aggregate_use_pooled_pixel_sums():
    prediction = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    target = torch.tensor([[[1.0, 1.0], [2.0, 4.0]]])
    row = _metrics(prediction, target)
    row["sample_index"] = 12
    aggregate = _aggregate((row,))

    assert row["valid_pixels"] == 4
    assert aggregate["pooled_rmse"] == pytest.approx(
        (2.0 / 4.0) ** 0.5)
    assert aggregate["pooled_mae"] == pytest.approx(0.5)


def test_state_rows_compare_each_iteration_to_fp32_reference():
    reference = (torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2) * 2.0)
    actual = (reference[0].to(torch.bfloat16), reference[1].to(torch.bfloat16))
    rows = _state_rows("W8A8_BF16_STATE", actual, reference)

    assert len(rows) == 2
    assert rows[0]["iteration"] == 1
    assert rows[0]["state_dtype"] == "bfloat16"
    assert rows[0]["state_rmse_vs_fp32_prop"] == pytest.approx(0.0)
