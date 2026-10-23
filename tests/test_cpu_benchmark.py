from types import SimpleNamespace

import pytest
import torch

from spn_quant.cpu_benchmark import (
    audit_cpu_execution,
    classify_operator,
    measure_cpu_forward,
    summarize_profile_events,
)


class CountingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return value + 1.0


def test_measure_cpu_forward_times_only_complete_measured_forwards():
    model = CountingModel()
    timestamps = iter([0, 2_000_000, 3_000_000, 7_000_000])

    result = measure_cpu_forward(
        model, (torch.ones(1),), warmup_iterations=1,
        timed_iterations=2, rounds=1,
        perf_counter_ns=lambda: next(timestamps))

    assert model.calls == 3
    assert result["raw_latency_ms"] == [2.0, 4.0]
    assert result["latency_ms"]["mean"] == pytest.approx(3.0)


def test_cpu_execution_audit_rejects_non_cpu_input_tensor():
    with pytest.raises(RuntimeError, match="not resident on CPU"):
        audit_cpu_execution(CountingModel(), (torch.empty(1, device="meta"),))


@pytest.mark.parametrize(
    "name, ancestors, expected",
    [
        ("aten::convolution", (), "dense_compute"),
        ("aten::bmm", (), "attention"),
        ("torchvision::deform_conv2d", (), "sampling"),
        ("aten::relu", (), "elementwise_norm"),
        ("aten::gelu", (), "elementwise_norm"),
        ("aten::permute", (), "layout_memory"),
        ("aten::constant_pad_nd", (), "layout_memory"),
        ("aten::slice", (), "layout_memory"),
        ("aten::convolution", ("SPN_PROPAGATION",), "propagation"),
        ("aten::linear", ("SPN_ATTENTION",), "attention"),
    ],
)
def test_operator_classification_is_exclusive_with_scope_precedence(
        name, ancestors, expected):
    assert classify_operator(name, ancestors) == expected


def test_profile_summary_uses_self_cpu_time_and_assigns_each_operator_once():
    propagation = SimpleNamespace(name="SPN_PROPAGATION", cpu_parent=None)
    events = [
        SimpleNamespace(
            name="aten::convolution", self_cpu_time_total=100.0,
            cpu_parent=None),
        SimpleNamespace(
            name="aten::add", self_cpu_time_total=50.0,
            cpu_parent=propagation),
        SimpleNamespace(
            name="SPN_PROPAGATION", self_cpu_time_total=20.0,
            cpu_parent=None),
    ]

    summary = summarize_profile_events(events, profiled_iterations=2)

    assert summary["category_mean_self_cpu_time_us"] == {
        "dense_compute": 50.0,
        "attention": 0.0,
        "sampling": 0.0,
        "propagation": 25.0,
        "elementwise_norm": 0.0,
        "layout_memory": 0.0,
        "other_unclassified": 0.0,
    }
    assert summary["classified_event_count"] == 2
    assert summary["self_cpu_time_sum_mean_us"] == pytest.approx(75.0)


def test_profile_summary_preserves_and_audits_negative_self_time_artifacts():
    events = [
        SimpleNamespace(
            name="aten::add", self_cpu_time_total=20.0, cpu_parent=None),
        SimpleNamespace(
            name="aten::fill_", self_cpu_time_total=-16.0, cpu_parent=None),
    ]

    summary = summarize_profile_events(events, profiled_iterations=1)

    assert summary["negative_self_time_events"] == {
        "count": 1,
        "total_magnitude_us": 16.0,
        "minimum_us": -16.0,
    }
    assert summary["category_mean_self_cpu_time_us"][
        "elementwise_norm"] == 4.0
    assert summary["self_cpu_time_sum_mean_us"] == 4.0


def test_profile_summary_rejects_negative_aggregated_category():
    event = SimpleNamespace(
        name="aten::fill_", self_cpu_time_total=-101.0, cpu_parent=None)

    with pytest.raises(ValueError, match="aggregated category"):
        summarize_profile_events([event], profiled_iterations=1)
