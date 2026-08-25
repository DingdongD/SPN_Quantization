import json
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from scripts import run_nyu_qdrop_reconstruction as runner
from scripts.run_nyu_qdrop_reconstruction import (
    _prepare_saved_args,
    algorithm_probability,
    build_seeded_batches,
    build_strict_manifest,
    configure_validation_propagation,
    build_calibration_split,
    merge_contracts,
    load_reconstruction_protocol,
    ordered_sample_identity_sha256,
    parse_args,
    resolve_legacy_sidecar_device_ownership,
    resolve_execution_order,
    select_probability_candidate,
    stack_seeded_samples,
    strict_prediction_metrics,
    strict_method,
    validate_phase_seed,
)


class RecordingPropagationAdapter(object):
    def __init__(self):
        self.config = None

    def configure(self, config):
        self.config = config


class SampleDataset(object):
    def __getitem__(self, index):
        return {
            "value": torch.full((2, 3), float(index)),
            "constant": "nyu",
        }


class CountingSampleDataset(object):
    def __init__(self):
        self.indices = []

    def __getitem__(self, index):
        self.indices.append(int(index))
        return {
            "value": torch.full((2, 3), float(index)),
            "constant": "nyu",
        }


def test_saved_run_device_cannot_override_explicit_reconstruction_device(
        monkeypatch):
    saved_args = SimpleNamespace(model="completionformer", device="cuda:0")
    monkeypatch.setattr(
        "scripts.run_nyu_qdrop_reconstruction.load_run_args",
        lambda run_dir: saved_args)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)

    prepared = _prepare_saved_args(
        "run", "data", "completionformer", torch.device("cuda:2"))

    assert prepared.device == "cuda:0"


@pytest.mark.parametrize("device", ("cpu", "cuda", "cuda:3"))
def test_reconstruction_device_must_be_available_explicit_index(
        monkeypatch, device):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)

    with pytest.raises(RuntimeError, match="device"):
        _prepare_saved_args(
            "run", "data", "nlspn", torch.device(device))


def test_seeded_samples_are_stacked_in_explicit_capture_batches():
    batch = stack_seeded_samples(
        SampleDataset(), indices=(7, 3, 9, 1), seed=61)

    assert tuple(batch) == ("value", "constant")
    assert batch["value"].shape == (4, 2, 3)
    assert batch["value"][:, 0, 0].tolist() == [7.0, 3.0, 9.0, 1.0]
    assert batch["constant"] == "nyu"


def test_seeded_batches_load_each_calibration_sample_once():
    dataset = CountingSampleDataset()

    batches = build_seeded_batches(
        dataset, indices=(7, 3, 9, 1, 8, 2), seed=61, batch_size=4)

    assert dataset.indices == [7, 3, 9, 1, 8, 2]
    assert tuple(batch.indices for batch in batches) == (
        (7, 3, 9, 1), (8, 2))
    assert batches[0].sample["value"].shape == (4, 2, 3)
    assert batches[1].sample["value"].shape == (2, 2, 3)


def test_validation_uses_the_formal_pa_constraint():
    adapter = RecordingPropagationAdapter()

    configure_validation_propagation(adapter)

    assert adapter.config.affinity_bits == 8
    assert adapter.config.confidence_bits == 8
    assert adapter.config.offset_bits == 8
    assert adapter.config.state_bits == 8
    assert adapter.config.coefficient_fraction_bits == 13


def test_calibration_split_preserves_persisted_order():
    indices = tuple(range(1000, 1128))
    first = build_calibration_split(
        calibration_indices=indices,
        reconstruction_samples=112,
        validation_samples=16,
    )
    second = build_calibration_split(
        calibration_indices=indices,
        reconstruction_samples=112,
        validation_samples=16,
    )

    assert first == second
    assert first.calibration == indices
    assert first.reconstruction == tuple(range(1000, 1112))
    assert first.validation == tuple(range(1112, 1128))
    assert set(first.reconstruction).isdisjoint(first.validation)


def test_calibration_split_rejects_duplicate_persisted_indices():
    with pytest.raises(ValueError, match="unique"):
        build_calibration_split(
            calibration_indices=(1,) * 128,
            reconstruction_samples=112,
            validation_samples=16,
        )


def test_probability_selection_orders_by_validation_loss_then_probability():
    rows = (
        {"probability": 0.75, "validation_loss": 0.3,
         "finite": 1, "failed_targets": 0},
        {"probability": 0.5, "validation_loss": 0.2,
         "finite": 1, "failed_targets": 0},
        {"probability": 0.25, "validation_loss": 0.2,
         "finite": 1, "failed_targets": 0},
    )

    selected = select_probability_candidate(
        rows, expected_probabilities=(0.25, 0.5, 0.75))

    assert selected["probability"] == 0.25


@pytest.mark.parametrize(
    "rows",
    (
        (
            {"probability": 0.25, "validation_loss": 0.2,
             "finite": 1, "failed_targets": 0},
            {"probability": 0.5, "validation_loss": 0.3,
             "finite": 1, "failed_targets": 0},
        ),
        (
            {"probability": 0.25, "validation_loss": 0.2,
             "finite": 1, "failed_targets": 0},
            {"probability": 0.5, "validation_loss": math.nan,
             "finite": 1, "failed_targets": 0},
            {"probability": 0.75, "validation_loss": 0.3,
             "finite": 1, "failed_targets": 0},
        ),
        (
            {"probability": 0.25, "validation_loss": 0.2,
             "finite": 1, "failed_targets": 0},
            {"probability": 0.5, "validation_loss": 0.3,
             "finite": 0, "failed_targets": 0},
            {"probability": 0.75, "validation_loss": 0.4,
             "finite": 1, "failed_targets": 1},
        ),
    ),
)
def test_probability_selection_fails_closed(rows):
    with pytest.raises(ValueError):
        select_probability_candidate(
            rows, expected_probabilities=(0.25, 0.5, 0.75))


def _legacy_pre_device_arguments(tmp_path):
    return [
        "--config", str(tmp_path / "qdrop.json"),
        "--run-dir", str(tmp_path / "run"),
        "--checkpoint", "best.pt",
        "--data-root", str(tmp_path / "data"),
        "--model", "dyspn",
        "--algorithm", "qdrop",
        "--precision", "W6A6",
        "--phase", "formal",
        "--seed", "1005",
        "--calibration-indices", str(tmp_path / "calibration_indices.json"),
        "--calibration-metadata", str(tmp_path / "calibration_metadata.json"),
        "--evaluation-protocol", str(tmp_path / "evaluation_protocol.json"),
        "--out-dir", str(tmp_path / "output"),
    ]


def test_legacy_parser_accepts_exact_pre_device_argument_set(tmp_path):
    with pytest.raises(SystemExit):
        parse_args([])

    args = parse_args(_legacy_pre_device_arguments(tmp_path))

    assert args.model == "dyspn"
    assert args.device is None
    assert args.algorithm == "qdrop"
    assert args.precision == "W6A6"
    assert args.phase == "formal"
    assert args.seed == 1005


def test_legacy_main_uses_sidecar_device_ownership_for_pre_device_cli(
        monkeypatch, tmp_path):
    observed = {}
    config = SimpleNamespace(
        precision=lambda name: SimpleNamespace(name=name),
        formal=SimpleNamespace(
            seeds=(1005, 1006, 1007), evaluation_seed=20260812),
    )
    monkeypatch.setattr(
        runner, "load_run_args",
        lambda run_dir: SimpleNamespace(device="cuda:0"))
    monkeypatch.setattr(runner, "load_qdrop_config", lambda path: config)

    def load_protocol(args, observed_config):
        observed["protocol_device"] = args.device
        assert observed_config is config
        return object(), object()

    def reconstruct(args, observed_config, probability, split, protocol,
                    phase, output):
        del observed_config, probability, split, protocol, phase, output
        observed["reconstruction_device"] = args.device
        return {
            "validation_loss": 0.25,
            "finite": 1,
            "failed_targets": 0,
        }

    monkeypatch.setattr(runner, "load_reconstruction_protocol", load_protocol)
    monkeypatch.setattr(runner, "run_reconstruction", reconstruct)

    runner.main(_legacy_pre_device_arguments(tmp_path))

    assert observed == {
        "protocol_device": "cuda:0",
        "reconstruction_device": "cuda:0",
    }


def test_legacy_sidecar_device_ownership_rejects_explicit_mismatch(
        monkeypatch):
    monkeypatch.setattr(
        runner, "load_run_args",
        lambda run_dir: SimpleNamespace(device="cuda:0"))
    args = SimpleNamespace(run_dir="run", device="cuda:1")

    with pytest.raises(ValueError, match="legacy sidecar device ownership"):
        resolve_legacy_sidecar_device_ownership(args)


def test_algorithm_contract_uses_deterministic_all_quantized_brecq():
    assert algorithm_probability("qdrop") == 0.5
    assert algorithm_probability("brecq") == 1.0
    assert strict_method("qdrop") == "qdrop_strict"
    assert strict_method("brecq") == "brecq_joint_strict"

    with pytest.raises(ValueError, match="algorithm"):
        algorithm_probability("rtn")


def test_validation_metrics_reject_nonpositive_without_mislabeling_nonfinite():
    gt = torch.ones(1, 1, 1, 3)
    prediction = torch.tensor([[[[1.0, 0.0, float("nan")]]]])

    metrics = strict_prediction_metrics(gt, prediction)

    assert math.isinf(metrics["RMSE"])
    assert metrics["nonfinite_pixels"] == 1
    assert metrics["nonpositive_pixels"] == 1
    assert metrics["invalid_pixels"] == 2
    assert metrics["prediction_min"] == 0.0


class BranchedModel(nn.Module):
    def __init__(self):
        super(BranchedModel, self).__init__()
        self.left = nn.Conv2d(1, 1, 1)
        self.right = nn.Conv2d(1, 1, 1)
        self.output = nn.Conv2d(1, 1, 1)
        self.unused = nn.Conv2d(1, 1, 1)

    def forward(self, value):
        right = self.right(value)
        left = self.left(value)
        return self.output(left + right)


def test_execution_order_comes_from_forward_not_manifest_sorting():
    model = BranchedModel().eval()

    order = resolve_execution_order(
        model,
        targets=("left", "output", "right"),
        model_args=(torch.ones(1, 1, 2, 2),),
    )

    assert order == ("right", "left", "output")


def test_execution_order_requires_every_target_once():
    model = BranchedModel().eval()

    with pytest.raises(RuntimeError, match="exactly once"):
        resolve_execution_order(
            model,
            targets=("left", "right", "output", "unused"),
            model_args=(torch.ones(1, 1, 2, 2),),
        )


def test_contract_merge_rejects_cross_block_overlap():
    combined = {}
    merge_contracts(combined, {"conv": {"bits": 4}}, "weight")

    with pytest.raises(RuntimeError, match="duplicate QDrop weight"):
        merge_contracts(combined, {"conv": {"bits": 4}}, "weight")


def test_formal_phase_requires_a_configured_seed():
    validate_phase_seed("formal", 1005, (1005, 1006, 1007))
    validate_phase_seed("probability-search", 61, (1005, 1006, 1007))

    with pytest.raises(ValueError, match="formal QDrop seed"):
        validate_phase_seed("formal", 61, (1005, 1006, 1007))


def test_strict_manifest_matches_edge_loader_schema():
    manifest = build_strict_manifest(
        method="brecq_joint_strict",
        model="cspn",
        contract="contract.pt",
        targets=("conv1", "conv2"),
        precision="W6A6",
        weight_bits=6,
        activation_bits=6,
        protocol={
            "checkpoint_sha256": "checkpoint",
            "calibration_indices_sha256": "calibration",
            "calibration_metadata_sha256": "metadata",
            "evaluation_protocol_sha256": "evaluation",
            "calibration_indices": list(range(128)),
            "reconstruction_indices": list(range(112)),
            "validation_indices": list(range(112, 128)),
            "evaluation_indices": list(range(64)),
            "evaluation_seed": 20260812,
        },
    )

    assert set(manifest) == {
        "format_version", "strict", "method", "model",
        "deployment_contract", "targets", "weight_bits",
        "activation_bits", "activation_policy", "precision", "protocol",
    }
    assert manifest["method"] == "brecq_joint_strict"
    assert manifest["weight_bits"] == 6
    assert manifest["activation_bits"] == 6
    assert manifest["precision"] == "W6A6"
    assert manifest["protocol"]["evaluation_seed"] == 20260812
    assert manifest["activation_policy"] == \
        "exact_semantic_edge_contract"


def _write_reconstruction_protocol(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    data_root = tmp_path / "data"
    data_root.mkdir()
    calibration_indices = tuple(range(128, 256))
    evaluation_indices = tuple(range(64))
    indices_path = tmp_path / "calibration_indices.json"
    indices_path.write_text(json.dumps({
        "indices": list(calibration_indices),
        "count": 128,
        "selection": "32_tail_96_kmedoids",
    }), encoding="utf-8")
    from spn_quant.deployment_contract import file_sha256
    metadata_path = tmp_path / "calibration_metadata.json"
    metadata_path.write_text(json.dumps({
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "data_root": str(data_root),
        "calibration_indices": list(calibration_indices),
        "evaluation_indices": list(evaluation_indices),
    }), encoding="utf-8")
    evaluation_path = tmp_path / "evaluation_protocol.json"
    evaluation_path.write_text(json.dumps({
        "checkpoint": str(checkpoint),
        "data_root": str(data_root),
        "seed": 20260812,
        "evaluation_samples": 64,
        "evaluation_indices": list(evaluation_indices),
    }), encoding="utf-8")
    args = SimpleNamespace(
        run_dir=tmp_path,
        checkpoint=checkpoint,
        data_root=data_root,
        model="nlspn",
        device="cuda:1",
        calibration_indices=indices_path,
        calibration_metadata=metadata_path,
        evaluation_protocol=evaluation_path,
    )
    config = SimpleNamespace(
        formal=SimpleNamespace(evaluation_seed=20260812),
        search=SimpleNamespace(
            reconstruction_samples=112, validation_samples=16),
    )
    return args, config, metadata_path, calibration_indices, \
        evaluation_indices


def test_reconstruction_protocol_hashes_consumed_ordered_identities(
        monkeypatch, tmp_path):
    args, config, _, calibration_indices, evaluation_indices = \
        _write_reconstruction_protocol(tmp_path)
    monkeypatch.setattr(
        "scripts.run_nyu_qdrop_reconstruction._prepare_saved_args",
        lambda run_dir, data_root, model, device: SimpleNamespace())
    monkeypatch.setattr(
        "scripts.run_nyu_qdrop_reconstruction.calibration_dataset",
        lambda saved_args: range(512))

    split, protocol = load_reconstruction_protocol(args, config)

    assert split.calibration == calibration_indices
    assert protocol["calibration_identity_sha256"] == \
        ordered_sample_identity_sha256("train", calibration_indices)
    assert protocol["reconstruction_identity_sha256"] == \
        ordered_sample_identity_sha256("train", split.reconstruction)
    assert protocol["validation_identity_sha256"] == \
        ordered_sample_identity_sha256("train", split.validation)
    assert protocol["evaluation_identity_sha256"] == \
        ordered_sample_identity_sha256("validation", evaluation_indices)


@pytest.mark.parametrize("identity", ("calibration", "evaluation"))
def test_reconstruction_protocol_rejects_metadata_identity_mismatch(
        monkeypatch, tmp_path, identity):
    args, config, metadata_path, _, _ = \
        _write_reconstruction_protocol(tmp_path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    field = identity + "_indices"
    payload[field][0], payload[field][1] = payload[field][1], payload[field][0]
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "scripts.run_nyu_qdrop_reconstruction._prepare_saved_args",
        lambda run_dir, data_root, model, device: SimpleNamespace())

    with pytest.raises(ValueError, match=identity + " identities"):
        load_reconstruction_protocol(args, config)
