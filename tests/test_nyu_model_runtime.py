import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts.nyu_model_runtime import NYUModelRuntime
from spn_quant.experiment_config import load_selected_quantization_config


BASELINE_ROOT = Path(
    "/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines")
CONFIG = Path(__file__).resolve().parents[1] / \
    "configs/three_model_selected_quantization.json"


def runtime_args(model_name, expected_architecture_class,
                 required_cuda_extension):
    run_dir = BASELINE_ROOT / ("%s_iter%d" % (
        model_name, 6 if model_name == "dyspn" else 18))
    saved_args = json.loads((run_dir / "args.json").read_text(
        encoding="utf-8"))
    return SimpleNamespace(
        model=model_name,
        run_dir=run_dir,
        checkpoint=run_dir / "best.pt",
        expected_architecture_class=expected_architecture_class,
        required_cuda_extension=required_cuda_extension,
        propagation_iterations=saved_args["iteration"],
        data_root=Path("/workspace/CSPN/cspn_pytorch"),
    )


def test_runtime_rejects_model_checkpoint_mismatch():
    args = runtime_args("dyspn", "Model", "torchvision.ops.deform_conv2d")
    args.model = "nlspn"

    with pytest.raises(ValueError, match="checkpoint model"):
        NYUModelRuntime.from_args(args)


def test_runtime_is_built_from_the_strict_model_config():
    config = load_selected_quantization_config(CONFIG)

    runtime = NYUModelRuntime.from_config(config.models[0])

    assert runtime.model_name == "dyspn"
    assert runtime.propagation_iterations == 6
    assert runtime.data_root == Path("/workspace/CSPN/cspn_pytorch")


def test_runtime_normalizes_dyspn_inputs_and_dictionary_predictions():
    runtime = NYUModelRuntime.from_args(
        runtime_args("dyspn", "Model", "torchvision.ops.deform_conv2d"))
    sample = {
        "rgbd": torch.ones(1, 4, 2, 3),
        "depth": torch.ones(1, 1, 2, 3),
    }

    model_args, ground_truth = runtime.model_input(sample, torch.device("cpu"))

    assert len(model_args) == 2
    assert tuple(model_args[0].shape) == (1, 3, 2, 3)
    assert tuple(model_args[1].shape) == (1, 1, 2, 3)
    assert runtime.prediction({"pred": ground_truth}) is ground_truth


@pytest.mark.skipif(sys.version_info[:2] != (3, 11),
                    reason="DySPN uses the default runtime")
def test_runtime_builds_selected_official_dyspn_checkpoint(monkeypatch):
    from scripts import train_nyu_iteration_sweep as sweep

    monkeypatch.setattr(
        sweep, "EXTERNAL_ROOT",
        Path("/workspace/external_depth_completion_models"))
    runtime = NYUModelRuntime.from_args(
        runtime_args("dyspn", "Model", "torchvision.ops.deform_conv2d"))

    model = runtime.build_model(torch.device("cpu"))

    assert type(model).__name__ == "Model"
    assert runtime.prediction(model) is model
    runtime.close()


@pytest.mark.skipif(sys.version_info[:2] != (3, 7),
                    reason="NLSPN uses the official Python 3.7 runtime")
def test_runtime_builds_selected_official_nlspn_checkpoint(monkeypatch):
    from scripts import train_nyu_iteration_sweep as sweep

    monkeypatch.setattr(
        sweep, "EXTERNAL_ROOT",
        Path("/workspace/external_depth_completion_models"))
    runtime = NYUModelRuntime.from_args(
        runtime_args("nlspn", "NLSPNModel", "DCN"))

    model = runtime.build_model(torch.device("cpu"))

    assert type(model).__name__ == "NLSPNModel"
    runtime.close()


@pytest.mark.skipif(sys.version_info[:2] != (3, 7),
                    reason="CompletionFormer uses the official Python 3.7 runtime")
def test_runtime_builds_selected_official_completionformer_checkpoint(
        monkeypatch):
    from scripts import train_nyu_iteration_sweep as sweep

    monkeypatch.setattr(
        sweep, "COMPLETIONFORMER_ROOT", Path("/workspace/CompletionFormer"))
    runtime = NYUModelRuntime.from_args(
        runtime_args("completionformer", "CompletionFormer", "DCN"))

    model = runtime.build_model(torch.device("cpu"))

    assert type(model).__name__ == "CompletionFormer"
    runtime.close()
