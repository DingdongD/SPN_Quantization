import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torchvision

from scripts import nyu_model_runtime as runtime_module
from scripts.nyu_model_runtime import NYUModelRuntime
from spn_quant.experiment_config import load_selected_quantization_config


CONFIG = Path(__file__).resolve().parents[1] / \
    "configs/three_model_selected_quantization.json"


def runtime_args(model_name, expected_architecture_class,
                 required_cuda_extension):
    config = load_selected_quantization_config(CONFIG)
    model_config = next(
        model for model in config.models if model.model == model_name)
    run_dir = model_config.run_dir
    saved_args = json.loads((run_dir / "args.json").read_text(
        encoding="utf-8"))
    return SimpleNamespace(
        model=model_name,
        run_dir=run_dir,
        checkpoint=model_config.checkpoint,
        expected_architecture_class=expected_architecture_class,
        required_cuda_extension=required_cuda_extension,
        native_cuda_operator=model_config.native_cuda_operator,
        checkpoint_architecture=model_config.checkpoint_architecture,
        propagation_iterations=saved_args["iteration"],
        data_root=Path("/workspace/CSPN/cspn_pytorch"),
        device=model_config.device,
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


def test_checkpoint_runtime_skips_cspn_imagenet_bootstrap():
    runtime = NYUModelRuntime.from_args(
        runtime_args("dyspn", "Model", "torchvision.ops.deform_conv2d"))
    runtime.model_name = "cspn"
    assert runtime.saved_args.from_scratch is False

    builder_args = runtime._checkpoint_builder_args()

    assert builder_args.from_scratch is True
    assert runtime.saved_args.from_scratch is False


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


@pytest.mark.parametrize(
    "model_name, expected_dataset",
    (
        ("cspn", "CspnOfficialDataset"),
        ("dyspn", "NyuHdf5Dataset"),
    ),
)
def test_runtime_uses_the_training_dataset_pipeline(
        monkeypatch, model_name, expected_dataset):
    runtime = NYUModelRuntime.from_args(
        runtime_args("dyspn", "Model", "torchvision.ops.deform_conv2d"))
    runtime.model_name = model_name
    calls = []

    def dataset_factory(**kwargs):
        calls.append((expected_dataset, kwargs))
        return object()

    monkeypatch.setattr(
        runtime_module.sweep, expected_dataset, dataset_factory)

    runtime.build_dataset("val")

    assert calls == [(expected_dataset, {
        "csv_file": runtime.saved_args.eval_list,
        "root_dir": str(runtime.data_root),
        "split": "val",
        "n_sample": runtime.saved_args.n_sample,
        "seed": runtime.saved_args.seed,
    })]


@pytest.mark.parametrize(
    "section, field, value, message",
    (
        ("args", "model", "nlspn", "checkpoint args model"),
        ("args", "iteration", 7, "checkpoint args iteration"),
        ("meta", "architecture", "other", "checkpoint meta architecture"),
        ("meta", "iteration", 7, "checkpoint meta iteration"),
    ),
)
@pytest.mark.skipif(sys.version_info[:2] != (3, 11),
                    reason="DySPN uses the default runtime")
def test_runtime_rejects_modified_checkpoint_identity_before_state_loading(
        monkeypatch, section, field, value, message):
    class Model(torch.nn.Module):
        def load_state_dict(self, state_dict, strict=True):
            raise AssertionError("state loading must not occur")

    def build_model(saved_args, device):
        return Model(), {}

    args = runtime_args("dyspn", "Model", "torchvision.ops.deform_conv2d")
    payload = torch.load(str(args.checkpoint), map_location="cpu")
    payload[section][field] = value
    monkeypatch.setitem(runtime_module.sweep.BUILDERS, "dyspn", build_model)
    monkeypatch.setattr(runtime_module.torch, "load",
                        lambda path, map_location: payload)
    runtime = NYUModelRuntime.from_args(args)

    with pytest.raises(ValueError, match=message):
        runtime.build_model(torch.device("cuda:0"))


@pytest.mark.skipif(sys.version_info[:2] != (3, 11),
                    reason="DySPN uses the default runtime")
def test_runtime_requires_configured_cuda_device_before_building(monkeypatch):
    def build_model(saved_args, device):
        raise AssertionError("builder must not run")

    args = runtime_args("dyspn", "Model", "torchvision.ops.deform_conv2d")
    monkeypatch.setitem(runtime_module.sweep.BUILDERS, "dyspn", build_model)
    runtime = NYUModelRuntime.from_args(args)

    with pytest.raises(RuntimeError, match="configured CUDA device"):
        runtime.build_model(torch.device("cpu"))


@pytest.mark.skipif(sys.version_info[:2] != (3, 11),
                    reason="DySPN uses the default runtime")
def test_runtime_requires_available_cuda_before_building(monkeypatch):
    def build_model(saved_args, device):
        raise AssertionError("builder must not run")

    args = runtime_args("dyspn", "Model", "torchvision.ops.deform_conv2d")
    monkeypatch.setitem(runtime_module.sweep.BUILDERS, "dyspn", build_model)
    monkeypatch.setattr(runtime_module.torch.cuda, "is_available",
                        lambda: False)
    runtime = NYUModelRuntime.from_args(args)

    with pytest.raises(RuntimeError, match="CUDA is required"):
        runtime.build_model(torch.device(runtime.device))


def test_explicit_cuda_activation_sets_and_verifies_requested_index(
        monkeypatch):
    calls = []
    monkeypatch.setattr(runtime_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime_module.torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(
        runtime_module.torch.cuda, "set_device",
        lambda index: calls.append(int(index)))
    monkeypatch.setattr(
        runtime_module.torch.cuda, "current_device", lambda: calls[-1])

    device = runtime_module.activate_explicit_cuda_device(
        torch.device("cuda:2"), "official smoke")

    assert device == torch.device("cuda:2")
    assert calls == [2]


def test_explicit_cuda_activation_rejects_current_device_mismatch(
        monkeypatch):
    monkeypatch.setattr(runtime_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime_module.torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(runtime_module.torch.cuda, "set_device", lambda index: None)
    monkeypatch.setattr(runtime_module.torch.cuda, "current_device", lambda: 0)

    with pytest.raises(RuntimeError, match="current CUDA device"):
        runtime_module.activate_explicit_cuda_device(
            torch.device("cuda:2"), "official smoke")


@pytest.mark.skipif(sys.version_info[:2] != (3, 11),
                    reason="DySPN uses the default runtime")
def test_runtime_rejects_wrapper_without_native_cuda_operator(monkeypatch):
    args = runtime_args("dyspn", "Model", "torchvision.ops.deform_conv2d")
    runtime = NYUModelRuntime.from_args(args)
    assert hasattr(torchvision.ops, "deform_conv2d")
    monkeypatch.setattr(runtime_module.torch.cuda, "is_available",
                        lambda: True)
    monkeypatch.setattr(
        runtime_module.torch._C, "_dispatch_has_kernel_for_dispatch_key",
        lambda operator, dispatch_key: False)

    with pytest.raises(RuntimeError, match="native CUDA operator"):
        runtime._assert_required_cuda_extension()


@pytest.mark.skipif(sys.version_info[:2] != (3, 11),
                    reason="DySPN uses the default runtime")
def test_runtime_checks_dispatch_table_when_kernel_query_is_missing(
        monkeypatch):
    args = runtime_args("dyspn", "Model", "torchvision.ops.deform_conv2d")
    runtime = NYUModelRuntime.from_args(args)
    monkeypatch.delattr(
        runtime_module.torch._C,
        "_dispatch_has_kernel_for_dispatch_key")
    monkeypatch.setattr(
        runtime_module.torch._C, "_dispatch_dump_table",
        lambda operator: "CPU: registered at test.cpp:1 [kernel]\n")

    with pytest.raises(RuntimeError, match="native CUDA operator"):
        runtime._assert_required_cuda_extension()


@pytest.mark.skipif(sys.version_info[:2] != (3, 11),
                    reason="DySPN uses the default runtime")
def test_runtime_builds_selected_official_dyspn_checkpoint(monkeypatch):
    from scripts import train_nyu_iteration_sweep as sweep

    monkeypatch.setattr(
        sweep, "EXTERNAL_ROOT",
        Path("/workspace/external_depth_completion_models"))
    runtime = NYUModelRuntime.from_args(
        runtime_args("dyspn", "Model", "torchvision.ops.deform_conv2d"))

    model = runtime.build_model(torch.device(runtime.device))

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

    model = runtime.build_model(torch.device(runtime.device))

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

    model = runtime.build_model(torch.device(runtime.device))

    assert type(model).__name__ == "CompletionFormer"
    runtime.close()
