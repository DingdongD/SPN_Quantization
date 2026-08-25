import json
from pathlib import Path

import pytest

from spn_quant.experiment_config import load_selected_quantization_config


CONFIG = Path(__file__).resolve().parents[1] / \
    "configs/three_model_selected_quantization.json"


def test_config_requires_all_three_models(tmp_path):
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    del payload["models"]["completionformer"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(KeyError):
        load_selected_quantization_config(path)


def test_config_requires_every_selected_method_hyperparameter(tmp_path):
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    del payload["method_hyperparameters"]["qdrop_w6a6"]["steps"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(KeyError):
        load_selected_quantization_config(path)


@pytest.mark.parametrize(
    "field",
    ("maximum_average_weight_bits", "maximum_average_activation_bits"),
)
def test_hawq_mixed_le6_config_rejects_budget_above_six(tmp_path, field):
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["method_hyperparameters"]["hawq_mixed_le6"][field] = 8.0
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="mixed_le6"):
        load_selected_quantization_config(path)


def test_config_declares_official_model_runtime_contracts():
    config = load_selected_quantization_config(CONFIG)

    assert config.output_root == Path(
        "/workspace/SPN_Quantization/profile_logs/"
        "nyu_three_model_selected_quantization")
    assert tuple(model.model for model in config.models) == (
        "dyspn", "nlspn", "completionformer")
    assert tuple(model.propagation_iterations for model in config.models) == (
        6, 18, 18)
    assert tuple(model.calibration_count for model in config.models) == (
        128, 128, 128)
    assert all(len(model.evaluation_indices) == 64 for model in config.models)
    assert tuple(model.expected_architecture_class for model in config.models) == (
        "Model", "NLSPNModel", "CompletionFormer")
    assert tuple(model.checkpoint_architecture for model in config.models) == (
        "DySPN res34 v1 neighbor=5", "NLSPN resnet34", "CompletionFormer")
    assert tuple(model.required_cuda_extension for model in config.models) == (
        "torchvision.ops.deform_conv2d", "DCN", "DCN")
    assert tuple(model.python_executable for model in config.models) == (
        Path("/opt/conda/bin/python"),
        Path("/opt/conda/envs/completionformer-py37/bin/python"),
        Path("/opt/conda/envs/completionformer-py37/bin/python"),
    )
    assert config.method_hyperparameters["rtn_w8a8"]["weight_bits"] == 8
    assert config.method_hyperparameters["rtn_w8a8"]["activation_bits"] == 8
    assert config.method_hyperparameters["qdrop_w6a6"]["steps"] == 20000
    assert config.method_hyperparameters["brecq_w6a6"]["steps"] == 20000
    assert config.method_hyperparameters["hawq_mixed_le6"][
        "maximum_average_weight_bits"] == 6.0
    assert config.method_hyperparameters["hawq_mixed_le6"][
        "maximum_average_activation_bits"] == 6.0
