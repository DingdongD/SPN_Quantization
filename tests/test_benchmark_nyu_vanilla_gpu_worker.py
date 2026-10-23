import gc
from pathlib import Path

import torch

from scripts import benchmark_nyu_vanilla_gpu_worker as worker


class CountingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.gc_states = []

    def forward(self, value):
        self.calls += 1
        self.gc_states.append(gc.isenabled())
        return value + 1.0


class FakeEvent:
    def __init__(self, enable_timing):
        assert enable_timing is True

    def record(self):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        assert isinstance(other, FakeEvent)
        return 2.0


def test_measure_forward_excludes_warmups_and_reports_peak_workspace(monkeypatch):
    model = CountingModel()
    monkeypatch.setattr(worker.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(worker.torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(
        worker.torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(
        worker.torch.cuda, "memory_allocated", lambda device: 10 * 1024**2)
    monkeypatch.setattr(
        worker.torch.cuda, "memory_reserved", lambda device: 12 * 1024**2)
    monkeypatch.setattr(
        worker.torch.cuda, "max_memory_allocated", lambda device: 14 * 1024**2)
    monkeypatch.setattr(
        worker.torch.cuda, "max_memory_reserved", lambda device: 16 * 1024**2)

    measurement = worker.measure_forward(
        model, (torch.ones(1),), torch.device("cuda:0"),
        warmup_iterations=2, timed_iterations=3, rounds=2,
        disable_python_gc=True)

    assert model.calls == 8
    assert model.gc_states == [False] * 8
    assert gc.isenabled() is True
    assert measurement["latency_ms"]["count"] == 6
    assert measurement["latency_ms"]["mean"] == 2.0
    assert len(measurement["rounds"]) == 2
    assert measurement["memory_mb"] == {
        "baseline_allocated": 10.0,
        "baseline_reserved": 12.0,
        "peak_allocated": 14.0,
        "peak_reserved": 16.0,
        "inference_workspace_peak": 4.0,
    }


def test_load_fixed_model_args_selects_model_and_moves_nested_tensors(
        tmp_path: Path):
    fixed_inputs = tmp_path / "fixed.pt"
    torch.save({
        "cspn": {"args": (torch.ones(1, 4, 2, 3),)},
        "completionformer": {
            "args": ({
                "rgb": torch.ones(1, 3, 2, 3),
                "dep": torch.ones(1, 1, 2, 3),
            },),
        },
    }, fixed_inputs)

    args = worker.load_fixed_model_args(
        fixed_inputs, "completionformer", torch.device("cpu"))

    assert isinstance(args, tuple)
    assert isinstance(args[0], dict)
    assert tuple(args[0]["rgb"].shape) == (1, 3, 2, 3)
    assert tuple(args[0]["dep"].shape) == (1, 1, 2, 3)
    assert args[0]["rgb"].device.type == "cpu"


def test_load_fixed_model_args_rejects_missing_model(tmp_path: Path):
    fixed_inputs = tmp_path / "fixed.pt"
    torch.save({"cspn": {"args": (torch.ones(1),)}}, fixed_inputs)

    try:
        worker.load_fixed_model_args(
            fixed_inputs, "dyspn", torch.device("cpu"))
    except ValueError as error:
        assert "dyspn" in str(error)
    else:
        raise AssertionError("missing fixed model input must be rejected")
