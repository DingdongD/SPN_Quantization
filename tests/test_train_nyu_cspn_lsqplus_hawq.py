import json

import pytest

from scripts import train_nyu_cspn_lsqplus_hawq as runner


def test_training_methods_are_exact():
    assert runner.METHODS == (
        "lsqplus_w4a4", "lsqplus_w6a6", "hawq_mixed_le6")


def test_lsqplus_method_has_uniform_declared_bits():
    assignment = runner.uniform_method_assignment(
        runner.expected_registry(), 4)

    assert set(bits for name, bits in assignment.weight_bits) == {4}
    assert set(bits for owner, bits in assignment.activation_bits) == {4}


def _required_cli(method):
    return [
        "--method", method,
        "--config", "config.json",
        "--checkpoint", "checkpoint.pt",
        "--data-root", "data",
        "--calibration-metadata", "calibration.json",
        "--output-root", "output",
        "--device", "cuda:0",
    ]


def test_hawq_training_requires_assignment_path():
    args = runner.parse_args(_required_cli("hawq_mixed_le6"))

    with pytest.raises(ValueError, match="assignment"):
        runner.validate_method_paths(args)


def test_lsqplus_training_rejects_assignment_path():
    args = runner.parse_args(
        _required_cli("lsqplus_w4a4") +
        ["--assignment", "selected_assignment.json"])

    with pytest.raises(ValueError, match="HAWQ"):
        runner.validate_method_paths(args)


def test_load_hawq_assignment_requires_exact_registry_and_budgets(tmp_path):
    registry = runner.expected_registry()
    fixed_weights = set(
        registry.weights_by_block["stem"] +
        registry.weights_by_block["initial_depth"])
    fixed_activations = set(
        registry.activations_by_block["stem"] +
        registry.activations_by_block["initial_depth"])
    uniform = runner.uniform_method_assignment(registry, 4)
    assignment = runner.BitAssignment(
        weight_bits=tuple(
            (name, 8 if name in fixed_weights else bits)
            for name, bits in uniform.weight_bits),
        activation_bits=tuple(
            (owner, 8 if owner in fixed_activations else bits)
            for owner, bits in uniform.activation_bits),
    )
    path = tmp_path / "assignment.json"
    path.write_text(json.dumps({
        "weight_bits": [
            {"module": name, "bits": bits}
            for name, bits in assignment.weight_bits],
        "activation_bits": [
            {"module": owner[0], "kind": owner[1], "bits": bits}
            for owner, bits in assignment.activation_bits],
        "average_weight_bits": 6.0,
        "average_activation_bits": 6.0,
    }), encoding="utf-8")

    loaded = runner.load_hawq_assignment(path, registry)

    assert loaded == assignment


def test_checkpoint_payload_requires_all_training_state():
    required = {
        "model_state", "method_state", "optimizer", "scheduler",
        "epoch", "convergence", "assignment", "method_config",
        "owner_manifest", "history", "train_generator_state",
        "torch_rng_state", "numpy_rng_state", "cuda_rng_state",
    }

    runner.validate_checkpoint_payload(dict((key, object()) for key in required))
    incomplete = dict((key, object()) for key in required - {"method_state"})
    with pytest.raises(ValueError, match="checkpoint"):
        runner.validate_checkpoint_payload(incomplete)


def test_resume_contract_rejects_method_or_assignment_change():
    saved = {
        "method": "lsqplus_w4a4",
        "assignment": {"weight_bits": [["0", 4]]},
        "calibration_indices": list(range(128)),
        "owner_manifest": ["weight:0"],
    }
    expected = dict(saved)
    expected["method"] = "lsqplus_w6a6"

    with pytest.raises(ValueError, match="resume"):
        runner.validate_resume_contract(saved, expected)


def test_convergence_decision_updates_tracker_before_checkpointing():
    tracker = runner.qat_base.QATConvergenceTracker(30, 6, 0.001)

    is_best, stop = runner.update_convergence(tracker, 1, 0.2)

    assert is_best
    assert not stop
    assert tracker.best_epoch == 1
    assert tracker.best_rmse == 0.2
