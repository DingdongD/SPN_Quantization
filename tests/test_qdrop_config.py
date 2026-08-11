import ast
import json
from pathlib import Path

import pytest

from spn_quant.qdrop_config import load_qdrop_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def valid_payload():
    return {
        "reference": {
            "repository": "https://github.com/wimh966/QDrop.git",
            "branch": "qdrop",
            "commit": "4a9ca007ce91b66620b911de97df36d5109ecae0",
        },
        "quantization": {
            "weight_bits": 4,
            "activation_bits": 4,
            "weight_clip_ratio": 1.0,
            "activation_scale_minimum": 1.0e-8,
        },
        "search": {
            "calibration_samples": 128,
            "reconstruction_samples": 112,
            "validation_samples": 16,
            "steps": 2000,
            "quant_probabilities": [0.5],
        },
        "reconstruction": {
            "batch_size": 32,
            "capture_batch_size": 4,
            "cache_cuda_byte_limit": 17179869184,
            "steps": 20000,
            "weight_learning_rate": 1.0e-3,
            "activation_learning_rate": 4.0e-5,
            "round_loss_weight": 1.0e-2,
            "warmup_fraction": 0.2,
            "beta_start": 20.0,
            "beta_end": 2.0,
            "loss_power": 2.0,
        },
        "formal": {
            "seeds": [1005, 1006, 1007],
            "evaluation_samples": 64,
            "evaluation_seed": 20260804,
        },
    }


def write_payload(tmp_path, payload):
    path = tmp_path / "qdrop.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_complete_qdrop_configuration(tmp_path):
    config = load_qdrop_config(write_payload(tmp_path, valid_payload()))

    assert config.reference.commit == (
        "4a9ca007ce91b66620b911de97df36d5109ecae0")
    assert config.quantization.weight_bits == 4
    assert config.quantization.activation_bits == 4
    assert config.search.quant_probabilities == (0.5,)
    assert config.reconstruction.activation_learning_rate == 4.0e-5
    assert config.reconstruction.capture_batch_size == 4
    assert config.reconstruction.cache_cuda_byte_limit == 17179869184
    assert config.formal.seeds == (1005, 1006, 1007)


def test_repository_official_config_uses_fixed_qdrop_probability():
    config = load_qdrop_config(
        REPO_ROOT / "configs" / "qdrop_w4a4_official.json")

    assert config.search.quant_probabilities == (0.5,)
    assert config.search.calibration_samples == 128
    assert config.search.reconstruction_samples == 112
    assert config.search.validation_samples == 16


def test_missing_required_field_fails(tmp_path):
    payload = valid_payload()
    del payload["reconstruction"]["activation_learning_rate"]

    with pytest.raises(KeyError, match="activation_learning_rate"):
        load_qdrop_config(write_payload(tmp_path, payload))


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    (
        ("quantization", "weight_bits", 8, "exactly W4A4"),
        ("quantization", "activation_bits", 8, "exactly W4A4"),
        ("search", "calibration_samples", 1024, "128"),
        ("search", "quant_probabilities", [0.25, 0.5, 0.75], "fixed 0.5"),
        ("reconstruction", "steps", 10000, "20000"),
        ("formal", "seeds", [1005, 1005, 1007], "unique"),
    ),
)
def test_rejects_noncanonical_protocol(
        tmp_path, section, field, value, message):
    payload = valid_payload()
    payload[section][field] = value

    with pytest.raises(ValueError, match=message):
        load_qdrop_config(write_payload(tmp_path, payload))


def test_qdrop_production_files_fail_closed_by_style():
    paths = sorted((REPO_ROOT / "spn_quant").glob("qdrop_*.py"))
    paths += sorted((REPO_ROOT / "scripts").glob("*qdrop*.py"))
    violations = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                violations.append((path.name, node.lineno, "try"))
            if isinstance(node, ast.Call) and \
                    isinstance(node.func, ast.Name) and \
                    node.func.id == "getattr":
                violations.append((path.name, node.lineno, "getattr"))
            if isinstance(node, ast.Call) and \
                    isinstance(node.func, ast.Attribute) and \
                    node.func.attr == "get":
                violations.append((path.name, node.lineno, "dict.get"))
    assert violations == []
