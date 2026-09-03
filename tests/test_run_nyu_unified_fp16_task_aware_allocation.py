import pytest

from scripts.run_nyu_unified_fp16_task_aware_allocation import (
    MODEL_NAMES,
    run_manifest_payload,
    validate_protocol_payload,
)


def test_protocol_requires_all_four_models_and_fp16_propagation():
    payload = {
        "protocol": {
            "models": list(MODEL_NAMES),
            "propagation_dtype": "fp16",
            "bit_levels": [4, 6, 8],
            "calibration_count": 128,
            "evaluation_count": 64,
            "budget_pairs": [[6.0, 6.0]],
        }
    }

    validate_protocol_payload(payload)


def test_protocol_rejects_integer_propagation():
    payload = {
        "protocol": {
            "models": list(MODEL_NAMES),
            "propagation_dtype": "bf16",
            "bit_levels": [4, 6, 8],
            "calibration_count": 128,
            "evaluation_count": 64,
            "budget_pairs": [[6.0, 6.0]],
        }
    }

    with pytest.raises(ValueError, match="FP16"):
        validate_protocol_payload(payload)


def test_protocol_rejects_missing_model():
    payload = {
        "protocol": {
            "models": ["cspn", "dyspn", "nlspn"],
            "propagation_dtype": "fp16",
            "bit_levels": [4, 6, 8],
            "calibration_count": 128,
            "evaluation_count": 64,
            "budget_pairs": [[6.0, 6.0]],
        }
    }

    with pytest.raises(ValueError, match="model order"):
        validate_protocol_payload(payload)


def test_run_manifest_payload_requires_all_model_outputs():
    payload = {
        "output_root": "/workspace/results",
        "protocol": {"models": list(MODEL_NAMES)},
    }
    manifest = run_manifest_payload(
        payload,
        tuple("/workspace/results/%s" % name for name in MODEL_NAMES))
    assert manifest["models"] == list(MODEL_NAMES)
    assert manifest["model_outputs"][-1] == \
        "/workspace/results/completionformer"

    with pytest.raises(ValueError, match="model outputs"):
        run_manifest_payload(payload, ("/workspace/results/cspn",))
