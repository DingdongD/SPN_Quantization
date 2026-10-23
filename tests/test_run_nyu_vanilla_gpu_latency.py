import os
from argparse import Namespace
from pathlib import Path

import pytest

from scripts import run_nyu_vanilla_gpu_latency as launcher
from scripts.run_nyu_vanilla_gpu_latency import (
    build_worker_command,
    build_worker_environment,
    load_benchmark_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "four_spn_vanilla_gpu_latency.json"


def test_checked_in_config_covers_four_official_models_on_one_device():
    config = load_benchmark_config(CONFIG, REPO_ROOT)

    assert tuple(config["models"]) == (
        "cspn", "dyspn", "nlspn", "completionformer")
    assert config["measurement"]["batch_size"] == 1
    assert config["measurement"]["precision"] == "fp32"
    assert config["measurement"]["tf32"] is False
    assert config["measurement"]["python_gc"] is True
    assert config["measurement"]["includes_propagation"] is True
    assert config["measurement"]["warmup_iterations"] > 0
    assert config["measurement"]["rounds"] >= 2
    assert config["model_source_config"].is_file()


def test_worker_command_uses_model_python_and_complete_measurement_contract(
        tmp_path):
    config = load_benchmark_config(CONFIG, REPO_ROOT)

    command = build_worker_command(
        REPO_ROOT, CONFIG, config, "nlspn", "cuda:2",
        tmp_path / "nlspn.json")

    assert command[0] == \
        "/opt/conda/envs/completionformer-py37/bin/python"
    assert command[1].endswith("benchmark_nyu_vanilla_gpu_worker.py")
    assert command[command.index("--model") + 1] == "nlspn"
    assert command[command.index("--device") + 1] == "cuda:2"
    assert command[command.index("--warmup-iterations") + 1] == "30"
    assert command[command.index("--timed-iterations") + 1] == "50"
    assert command[command.index("--rounds") + 1] == "3"
    assert "--disable-python-gc" not in command


def test_worker_command_propagates_fixed_input_artifact(tmp_path):
    config = load_benchmark_config(CONFIG, REPO_ROOT)
    fixed_inputs = tmp_path / "inputs.pt"
    fixed_inputs.write_bytes(b"fixture")

    command = build_worker_command(
        REPO_ROOT, CONFIG, config, "cspn", "cuda:0",
        tmp_path / "cspn.json", fixed_inputs=fixed_inputs)

    assert command[command.index("--fixed-inputs") + 1] == str(fixed_inputs)


def test_worker_environment_preserves_existing_library_path_and_sets_roots():
    config = load_benchmark_config(CONFIG, REPO_ROOT)
    base = {"PATH": os.environ.get("PATH", ""), "LD_LIBRARY_PATH": "/old"}

    environment = build_worker_environment(
        base, config["models"]["completionformer"])

    assert environment["SPN_DATA_ROOT"] == "/workspace/CSPN/cspn_pytorch"
    assert environment["SPN_EXTERNAL_ROOT"] == \
        "/workspace/external_depth_completion_models"
    assert environment["COMPLETIONFORMER_ROOT"] == "/workspace/CompletionFormer"
    assert environment["LD_LIBRARY_PATH"].endswith(":/old")


def test_busy_gpu_rejection_does_not_create_partial_output(tmp_path, monkeypatch):
    output = tmp_path / "benchmark"
    monkeypatch.setattr(launcher, "_gpu_state", lambda index: {
        "index": index,
        "name": "A100",
        "uuid": "GPU-test",
        "utilization_percent": 100,
        "memory_used_mb": 2000,
        "temperature_c": 80,
    })
    args = Namespace(
        config=CONFIG, output=output, gpu_index=0, allow_busy_gpu=False)

    with pytest.raises(RuntimeError, match="selected GPU is busy"):
        launcher.run_benchmark(args)

    assert not output.exists()
