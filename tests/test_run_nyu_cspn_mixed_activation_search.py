import json
from pathlib import Path
import shutil

import pytest

from scripts import run_nyu_cspn_mixed_activation_search as runner
from scripts import run_nyu_cspn_task_sensitive_bits as task_runner
from spn_quant import cspn_task_sensitive_bits as allocation


def registry():
    return task_runner.expected_registry()


def basis():
    current = registry()
    return allocation.CostBasis(
        weight_macs=tuple(
            (module, 1)
            for block in allocation.BLOCK_ORDER
            for module in current.weights_by_block[block]),
        activation_elements=tuple(
            (owner, 1)
            for block in allocation.BLOCK_ORDER
            for owner in current.activations_by_block[block]),
    )


def measured_rows(candidates):
    return tuple({
        "config": candidate.name,
        "assignment": candidate.assignment,
        "calibration_RMSE": 0.2 + index * 0.001,
        "boundary_RMSE": 0.3,
        "propagation_MSE": 0.01,
        "nonfinite_ratio": 0.0,
        "nonpositive_ratio": 0.0,
    } for index, candidate in enumerate(candidates))


def cli_values():
    return [
        "--run-dir", "/runs/cspn",
        "--checkpoint", "/models/best.pt",
        "--data-root", "/data/nyu",
        "--calibration-indices", "/runs/calibration_indices.json",
        "--calibration-metadata", "/runs/calibration_metadata.json",
        "--evaluation-protocol", "/runs/evaluation_metadata.json",
        "--precision-config", "/configs/mixed.json",
        "--out-dir", "/runs/mixed_search",
        "--devices", "cuda:0,cuda:1",
        "--seed", "20260812",
        "--fold-max-error", "0.05",
    ]


def test_parse_args_requires_every_experiment_path_and_device():
    args = runner.parse_args(cli_values())
    assert args.devices == "cuda:0,cuda:1"
    missing = cli_values()
    position = missing.index("--precision-config")
    del missing[position:position + 2]
    with pytest.raises(SystemExit):
        runner.parse_args(missing)


def test_devices_must_be_explicit_unique_cuda_devices():
    assert runner.parse_devices("cuda:2,cuda:3") == ("cuda:2", "cuda:3")
    with pytest.raises(ValueError, match="unique"):
        runner.parse_devices("cuda:0,cuda:0")
    with pytest.raises(ValueError, match="explicit CUDA"):
        runner.parse_devices("cpu")


def test_precision_config_requires_complete_approved_search_contract():
    config_path = Path("tests/.cspn_mixed_search_config.json")
    source = Path("configs/cspn_mixed_task_aware_qat.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    config = runner.load_precision_config(config_path)

    assert config["search"]["activation_bits"] == [4, 6, 8]
    del payload["loss"]["teacher"]
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(KeyError):
        runner.load_precision_config(config_path)
    config_path.unlink()


def test_index_protocol_requires_stratified_unique_128_train_samples():
    calibration = {
        "indices": list(range(128)),
        "count": 128,
        "selection": "32_tail_96_kmedoids",
    }
    evaluation = {
        "evaluation_indices": list(range(64)),
        "evaluation_samples": 64,
        "seed": 20260812,
    }

    protocol = runner.validate_index_protocol(calibration, evaluation)

    assert len(protocol.calibration_indices) == 128
    calibration["selection"] = "random"
    with pytest.raises(ValueError, match="stratified"):
        runner.validate_index_protocol(calibration, evaluation)
    calibration["selection"] = "32_tail_96_kmedoids"
    calibration["indices"][-1] = calibration["indices"][0]
    with pytest.raises(ValueError, match="unique"):
        runner.validate_index_protocol(calibration, evaluation)


def test_identity_overlap_is_rejected_when_splits_are_comparable():
    with pytest.raises(ValueError, match="overlap"):
        runner.validate_identity_isolation(
            (("train", "scene-1"),),
            (("train", "scene-1"),))
    runner.validate_identity_isolation(
        (("train", 1),), (("validation", 1),))


def test_search_selects_only_from_calibration_rows():
    calls = []

    class Evaluator:
        def calibration(self, phase, candidates):
            calls.append((phase, len(candidates)))
            return measured_rows(candidates)

        def validation(self, *args):
            raise AssertionError("validation entered assignment search")

    result = runner.run_activation_search(
        Evaluator(), registry(), basis(), 6.0)

    assert calls == [("mixed_activation", len(result.candidates))]
    assert result.selected_assignment == result.candidates[0].assignment
    assert result.audit.feasible
    assert all(
        candidate.assignment.weight_bits ==
        allocation.p3_t3_assignment(registry()).weight_bits
        for candidate in result.candidates)


def test_publish_is_atomic_and_records_complete_hashes():
    root = Path("tests/.cspn_mixed_search_publish")
    staging = Path(str(root) + ".incomplete")
    if root.exists():
        shutil.rmtree(root)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    current_basis = basis()

    class Evaluator:
        def calibration(self, phase, candidates):
            return measured_rows(candidates)

    result = runner.run_activation_search(
        Evaluator(), registry(), current_basis, 6.0)
    runner.publish_search_result(
        staging, root, result, current_basis,
        {"checkpoint_sha256": "checkpoint", "config_sha256": "config"})

    assert root.is_dir()
    assert not staging.exists()
    for name in (
            "selected_assignment.json", "cost_basis.json",
            "candidate_metrics.csv", "search_manifest.json",
            "artifact_sha256.json"):
        assert (root / name).is_file()
    hashes = json.loads(
        (root / "artifact_sha256.json").read_text(encoding="utf-8"))
    assert set(hashes) == {
        "candidate_metrics.csv", "cost_basis.json",
        "search_manifest.json", "selected_assignment.json"}
    shutil.rmtree(root)
