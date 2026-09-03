import pytest

from scripts.run_nyu_four_model_fp4_fp8 import (
    GROUP_NAMES,
    _format_score_gain,
    _group_budgets,
)
from spn_quant.fp_formats import FORMAT_SPECS


def _budget_payload():
    return {
        "fp_format_group_budgets": {
            "weight_average_bits": dict((group, 6.0)
                                         for group in GROUP_NAMES),
            "activation_average_bits": dict((group, 5.0)
                                             for group in GROUP_NAMES),
            "activation_minimum_fp8_fraction": dict((group, 0.0)
                                                    for group in GROUP_NAMES),
        },
        "fp_format_global_budgets": {
            "weight_average_bits": 6.0,
            "activation_average_bits": 6.0,
        },
    }


def test_group_budget_payload_is_explicit_and_independent():
    budgets = _group_budgets(_budget_payload())
    assert budgets["encoder"] == {
        "weight_average_bits": 6.0,
        "activation_average_bits": 5.0,
        "activation_minimum_fp8_fraction": 0.0,
    }
    assert budgets["attention"]["weight_average_bits"] == 6.0
    assert budgets["attention"]["activation_average_bits"] == 5.0


def test_group_budget_payload_rejects_missing_group():
    payload = _budget_payload()
    del payload["fp_format_group_budgets"]["weight_average_bits"]["concat"]
    with pytest.raises(KeyError, match="coverage"):
        _group_budgets(payload)


def test_format_score_gain_uses_fp4_to_fp8_task_error_reduction():
    scores = {
        "layer": {4: 10.0, 6: 4.0, 8: 1.5},
    }
    assert _format_score_gain(scores, ("layer",))["layer"] == 8.5


def test_fp6_is_an_explicit_six_bit_format():
    assert FORMAT_SPECS["fp6_e3m2"].bits == 6
