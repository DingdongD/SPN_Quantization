import pytest

from spn_quant import mixed_precision
from spn_quant.model_contracts import QuantizationBlock, QuantizationModelContract
from scripts.run_nyu_unified_fp16_task_aware_qat import (
    QAT_CONFIGURATIONS,
    assignment_counts,
    assignment_budget,
    json_safe,
    should_replace_best,
    load_best_checkpoint_for_evaluation,
    validate_training_checkpoint,
    validate_qat_payload,
)


def _payload():
    return {
        "output_root": "/workspace/qat-results",
        "source_root": "/workspace/ptq-results",
        "protocol": {
            "propagation_dtype": "fp16",
            "calibration_count": 128,
            "evaluation_count": 64,
            "models": ["cspn", "dyspn", "nlspn", "completionformer"],
            "configurations": list(QAT_CONFIGURATIONS),
        },
        "training": {
            "method": "task_aware",
            "epochs": 3,
            "batch_size": 2,
            "workers": 2,
            "learning_rate": 0.00001,
            "momentum": 0.9,
            "weight_decay": 0.0001,
            "max_gradient_norm": 10.0,
            "seed": 20260831,
            "log_interval": 10,
            "boundary_threshold_m": 0.1,
            "depth_loss_weight": 1.0,
            "boundary_loss_weight": 0.25,
            "teacher_loss_weight": 0.5,
            "initial_depth_loss_weight": 0.5,
            "propagation_loss_weight": 0.1,
            "hawq_range_momentum": 0.9,
            "fold_conv_bn": False,
            "fold_max_error": 0.05,
            "joint_clip_factors": [0.8, 1.0, 1.2],
            "joint_search_rounds": 2,
            "joint_cache_sample_limit": 128,
            "joint_cache_byte_limit": 1073741824,
        },
    }


def test_qat_protocol_requires_fp16_and_three_allocations():
    validate_qat_payload(_payload())


def test_qat_protocol_rejects_non_fp16_propagation():
    payload = _payload()
    payload["protocol"]["propagation_dtype"] = "int16"
    with pytest.raises(ValueError, match="FP16"):
        validate_qat_payload(payload)


def test_assignment_counts_are_exact_and_explicit():
    contract = QuantizationModelContract(
        model_name="test",
        blocks=(QuantizationBlock(
            "block", ("conv4", "conv6", "conv8"),
            (("activation::conv4::input", "module_input"),)),),
        prefix_groups=(("block",),),
        tail_groups=(("block",),),
        protected_roles=("propagation_state",),
        attention_edges=(),
        concat_edges=(),
        protected_modules=("propagation",),
        module_roles=(("propagation", "propagation_state"),),
    )
    assignment = mixed_precision.BitAssignment(
        weight_bits=(("conv4", 4), ("conv6", 6), ("conv8", 8)),
        activation_bits=((("activation::conv4::input", "module_input"), 4),),
        model_name=contract.model_name,
    )
    assert assignment_counts(assignment) == {
        "weight": {4: 1, 6: 1, 8: 1},
        "activation": {4: 1},
    }


def test_assignment_budget_reads_the_weighted_source_budget():
    manifest = {
        "assignments": {
            "TASK_AWARE_W5.00_A5.00": {
                "budget": {
                    "actual_weight_bits": 4.999999,
                    "actual_activation_bits": 5.000001,
                },
            },
        },
    }
    assert assignment_budget(
        manifest, "TASK_AWARE_W5.00_A5.00") == (4.999999, 5.000001)


def test_qat_keeps_first_checkpoint_when_validation_is_nonfinite():
    assert should_replace_best(None, float("inf"), float("inf"))


def test_qat_json_artifact_encodes_nonfinite_metrics_as_null():
    assert json_safe({"rmse": float("inf")}) == {"rmse": None}


def test_qat_resume_checkpoint_requires_exact_training_contract():
    assignment = {
        "model_name": "cspn",
        "weight_bits": [["conv4", 4]],
        "activation_bits": [[[
            "activation::conv4::input", "module_input"], 4]],
    }
    checkpoint = {
        "epoch": 2,
        "model_state": {"conv4.weight": object()},
        "method_state": {"activation.step": object()},
        "optimizer_state": {"state": {}, "param_groups": []},
        "history": [{"epoch": 1}, {"epoch": 2}],
        "assignment": assignment,
        "propagation_dtype": "fp16",
    }
    assert validate_training_checkpoint(
        checkpoint, assignment, 20) == 3


def test_qat_evaluation_loads_the_selected_best_checkpoint():
    class Controller:
        def __init__(self):
            self.calls = []

        def load_canonical_model_state_dict(self, state):
            self.calls.append(("model", state))

        def load_method_state_dict(self, state):
            self.calls.append(("method", state))

    controller = Controller()
    payload = {"model_state": {"weight": 1}, "method_state": {"step": 2}}
    load_best_checkpoint_for_evaluation(controller, payload)
    assert controller.calls == [
        ("model", payload["model_state"]),
        ("method", payload["method_state"]),
    ]
