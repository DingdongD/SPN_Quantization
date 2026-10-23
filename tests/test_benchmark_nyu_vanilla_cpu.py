from types import SimpleNamespace

import pytest
import torch

from scripts import benchmark_nyu_vanilla_cpu as worker


class Nested(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Module()
        self.block.target = torch.nn.Identity()


def test_resolve_module_supports_dotted_path():
    model = Nested()
    assert worker.resolve_module(model, "block.target") is model.block.target


def test_select_attention_modules_finds_attention_classes_and_paths():
    class Attention(torch.nn.Identity):
        pass

    model = torch.nn.Module()
    model.attn = torch.nn.Identity()
    model.encoder = torch.nn.Module()
    model.encoder.block = Attention()

    selected = dict(worker.select_attention_modules(model))

    assert set(selected) == {"attn", "encoder.block"}


def test_cpu_environment_gate_rejects_visible_cuda(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(worker.torch.cuda, "is_available", lambda: True)

    with pytest.raises(RuntimeError, match="CUDA must be unavailable"):
        worker.assert_cpu_only_process()


def test_require_cpu_energy_rails_accepts_cpu_and_total():
    result = {
        "power": {
            "rails": {
                "CPU": {"active_mean_power_w": 4.0,
                        "idle": {"mean_power_w": 1.0}},
                "total_module": {"active_mean_power_w": 12.0,
                                 "idle": {"mean_power_w": 8.0}},
            }
        }
    }

    worker.require_cpu_energy_rails(result)
    assert result["cpu_rail_name"] == "CPU"


def test_require_cpu_energy_rails_accepts_lowercase_jtop_name():
    result = {"power": {"rails": {
        "cpu": {"active_mean_power_w": 4.0,
                "idle": {"mean_power_w": 1.0}},
        "total_module": {"active_mean_power_w": 12.0,
                         "idle": {"mean_power_w": 8.0}},
    }}}

    worker.require_cpu_energy_rails(result)

    assert result["cpu_rail_name"] == "cpu"


def test_require_cpu_energy_rails_rejects_missing_cpu():
    result = {"power": {"rails": {"total_module": {
        "active_mean_power_w": 12.0,
        "idle": {"mean_power_w": 8.0},
    }}}}

    with pytest.raises(RuntimeError, match="CPU rail"):
        worker.require_cpu_energy_rails(result)


def test_parser_requires_positive_threads():
    args = SimpleNamespace(cpu_threads=0)
    with pytest.raises(ValueError, match="threads"):
        worker.validate_thread_settings(args)


def test_prime_cpu_convolutions_runs_discarded_kernel_before_forward(
        monkeypatch):
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3, padding=1))
    value = torch.ones(1, 1, 4, 4)
    real_conv2d = worker.torch.nn.functional.conv2d
    calls = []

    def counted(*args, **kwargs):
        calls.append(tuple(args[0].shape))
        return real_conv2d(*args, **kwargs)

    monkeypatch.setattr(worker.torch.nn.functional, "conv2d", counted)

    result = worker.prime_cpu_convolutions(model, (value,))

    assert result["primed_conv2d_modules"] == 1
    # One discarded hook invocation plus the module's real forward.
    assert calls == [(1, 1, 4, 4), (1, 1, 4, 4)]
    assert torch.isfinite(result["output"]).all()
