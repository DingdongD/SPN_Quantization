import gc
import json
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


def test_run_synchronized_energy_workload_counts_complete_forwards(monkeypatch):
    model = CountingModel()
    timeline = iter([0.0, 0.0, 0.2, 0.4, 0.6])
    monkeypatch.setattr(worker.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(worker.torch.cuda, "synchronize", lambda device: None)

    result = worker.run_synchronized_energy_workload(
        model,
        (torch.ones(1),),
        torch.device("cuda:0"),
        active_seconds=0.5,
        monotonic=lambda: next(timeline),
        monotonic_ns=lambda: 123,
    )

    assert model.calls == 3
    assert result["completed_forwards"] == 3
    assert result["raw_latency_ms"] == [2.0, 2.0, 2.0]
    assert result["active_start_monotonic_ns"] == 123
    assert result["active_end_monotonic_ns"] == 123


def test_build_power_sampler_command_keeps_sensor_process_explicit(tmp_path):
    command = worker.build_power_sampler_command(
        source="jtop",
        python_executable=Path("/usr/bin/python3"),
        output=tmp_path / "power.jsonl",
        interval_ms=50.0,
        gpu_index=0,
    )

    assert command[0] == "/usr/bin/python3"
    assert Path(command[1]).name == "sample_hardware_power.py"
    assert command[2:] == [
        "--source", "jtop",
        "--output", str(tmp_path / "power.jsonl"),
        "--interval-ms", "50.0",
        "--gpu-index", "0",
    ]


def test_power_trace_freshness_rejects_samples_from_a_previous_run(tmp_path):
    trace = tmp_path / "power.jsonl"
    trace.write_text("\n".join([
        json.dumps({"kind": "metadata", "source": "fake"}),
        json.dumps({"kind": "sample", "monotonic_ns": 99,
                    "power_w": {"board": 1.0}}),
    ]), encoding="utf-8")

    assert worker._has_power_sample_at_or_after(trace, 100) is False
    with trace.open("a", encoding="utf-8") as handle:
        handle.write("\n" + json.dumps({
            "kind": "sample", "monotonic_ns": 100,
            "power_w": {"board": 1.0}}))
    assert worker._has_power_sample_at_or_after(trace, 100) is True


def test_runtime_environment_is_set_before_model_modules_import(
        tmp_path, monkeypatch):
    monkeypatch.delenv("SPN_DATA_ROOT", raising=False)
    monkeypatch.delenv("SPN_EXTERNAL_ROOT", raising=False)
    monkeypatch.delenv("COMPLETIONFORMER_ROOT", raising=False)

    worker.configure_runtime_environment(
        tmp_path / "data", tmp_path / "external",
        tmp_path / "completionformer")

    assert worker.os.environ["SPN_DATA_ROOT"] == str(tmp_path / "data")
    assert worker.os.environ["SPN_EXTERNAL_ROOT"] == str(tmp_path / "external")
    assert worker.os.environ["COMPLETIONFORMER_ROOT"] == str(
        tmp_path / "completionformer")
