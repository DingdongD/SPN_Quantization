import math

from scripts.run_nyu_unified_fp16_task_aware_allocation import json_safe


def test_json_safe_converts_nonfinite_values_to_null():
    value = {
        "finite": 1.0,
        "infinite": float("inf"),
        "nested": [float("-inf"), 2.0],
    }

    actual = json_safe(value)

    assert actual == {
        "finite": 1.0,
        "infinite": None,
        "nested": [None, 2.0],
    }
    assert math.isfinite(actual["finite"])
