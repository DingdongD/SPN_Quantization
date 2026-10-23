import importlib.util
from pathlib import Path

import pytest


def _attribute_activities(*args):
    module_path = Path(__file__).resolve().parents[1] / "scripts" / \
        "analyze_nsys_nyu_vanilla_breakdown.py"
    if not module_path.exists():
        pytest.fail("strict Nsight breakdown analyzer is not implemented")
    spec = importlib.util.spec_from_file_location("strict_nsys_breakdown", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.attribute_activities(*args)


def test_correlation_id_maps_async_kernel_to_launching_operator():
    ranges = [
        {"start": 100, "end": 300, "text": "SPN_PROFILE_INFERENCE", "globalTid": 7},
        {"start": 120, "end": 160, "text": "aten::conv2d", "globalTid": 7},
    ]
    runtimes = [
        {"start": 130, "end": 140, "correlationId": 41, "globalTid": 7},
    ]
    activities = [
        # The device interval executes after the CPU operator range has closed.
        {"start": 500, "end": 550, "correlationId": 41,
         "name": "opaque_kernel", "activity": "kernel"},
    ]

    forwards = _attribute_activities(ranges, runtimes, activities)

    assert len(forwards) == 1
    assert forwards[0][0]["category"] == "dense_compute"
    assert forwards[0][0]["duration_ns"] == 50


def test_propagation_scope_takes_precedence_and_activity_is_counted_once():
    ranges = [
        {"start": 100, "end": 300, "text": "SPN_PROFILE_INFERENCE", "globalTid": 7},
        {"start": 120, "end": 250, "text": "SPN_PROPAGATION", "globalTid": 7},
        {"start": 130, "end": 180, "text": "aten::conv2d", "globalTid": 7},
    ]
    runtimes = [
        {"start": 140, "end": 150, "correlationId": 42, "globalTid": 7},
    ]
    activities = [
        {"start": 400, "end": 475, "correlationId": 42,
         "name": "conv_kernel", "activity": "kernel"},
    ]

    forwards = _attribute_activities(ranges, runtimes, activities)

    assert [row["category"] for row in forwards[0]] == ["propagation"]
    assert sum(row["duration_ns"] for row in forwards[0]) == 75


def test_memcpy_is_layout_memory_even_without_aten_range():
    ranges = [
        {"start": 100, "end": 300, "text": "SPN_PROFILE_INFERENCE", "globalTid": 7},
    ]
    runtimes = [
        {"start": 200, "end": 210, "correlationId": 43, "globalTid": 7},
    ]
    activities = [
        {"start": 600, "end": 620, "correlationId": 43,
         "name": "[CUDA memcpy]", "activity": "memcpy"},
    ]

    forwards = _attribute_activities(ranges, runtimes, activities)

    assert forwards[0][0]["category"] == "layout_memory"
