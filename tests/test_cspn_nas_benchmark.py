import inspect
import os

import pytest
import torch
import torch.nn as nn

from spn_quant.nas.benchmark import (
    assert_gpu_idle,
    benchmark_model,
)


class CountingModel(nn.Module):
    def __init__(self, finite=True, shape=(1, 1, 4, 5)):
        super().__init__()
        self.calls = 0
        self.finite = finite
        self.shape = shape

    def forward(self, value):
        self.calls += 1
        output = torch.ones(self.shape)
        if not self.finite:
            output.flatten()[0] = float("nan")
        return output


class FakeTimer:
    def __init__(self, durations):
        self.durations = iter(durations)

    def measure(self, callback):
        callback()
        return float(next(self.durations))


def test_benchmark_defaults_match_a100_protocol():
    signature = inspect.signature(benchmark_model)

    assert signature.parameters["warmup"].default == 200
    assert signature.parameters["iterations"].default == 1000
    assert signature.parameters["repeats"].default == 5
    assert signature.parameters["precision"].default == "fp32"


def test_benchmark_counts_iterations_and_summarizes_latency():
    model = CountingModel()
    timer = FakeTimer([1, 2, 3, 4, 5, 6])

    result = benchmark_model(
        model,
        torch.zeros(1, 4, 4, 5),
        warmup=2,
        iterations=3,
        repeats=2,
        timer=timer,
        require_idle=False,
        expected_output_shape=(1, 1, 4, 5),
    )

    assert model.calls == 8
    assert result["median_ms"] == pytest.approx(3.5)
    assert result["p95_ms"] == pytest.approx(5.75)
    assert result["repetition_medians_ms"] == [2.0, 5.0]
    assert result["precision"] == "fp32"


def test_benchmark_rejects_unknown_precision():
    with pytest.raises(ValueError, match="precision"):
        benchmark_model(
            CountingModel(), torch.zeros(1, 4, 4, 5),
            warmup=0, iterations=1, repeats=1,
            timer=FakeTimer([1]), require_idle=False,
            precision="int4")


@pytest.mark.parametrize("precision", ["tf32", "fp16", "bf16"])
def test_accelerated_precision_requires_cuda(precision):
    with pytest.raises(ValueError, match="CUDA"):
        benchmark_model(
            CountingModel(), torch.zeros(1, 4, 4, 5),
            warmup=0, iterations=1, repeats=1,
            timer=FakeTimer([1]), require_idle=False,
            precision=precision)


def test_benchmark_rejects_nonfinite_or_wrong_shape():
    with pytest.raises(FloatingPointError, match="non-finite"):
        benchmark_model(
            CountingModel(finite=False), torch.zeros(1, 4, 4, 5),
            warmup=0, iterations=1, repeats=1,
            timer=FakeTimer([1]), require_idle=False,
            expected_output_shape=(1, 1, 4, 5))

    with pytest.raises(RuntimeError, match="output shape"):
        benchmark_model(
            CountingModel(shape=(1, 1, 3, 5)), torch.zeros(1, 4, 4, 5),
            warmup=0, iterations=1, repeats=1,
            timer=FakeTimer([1]), require_idle=False,
            expected_output_shape=(1, 1, 4, 5))


def test_busy_gpu_preflight_rejects_other_processes():
    environment = {
        "index": 0,
        "processes": [{"pid": os.getpid() + 1, "used_memory_mib": 1024}],
    }

    with pytest.raises(RuntimeError, match="other compute processes"):
        assert_gpu_idle(environment)
