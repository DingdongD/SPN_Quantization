import math

import pytest
import torch
import torch.nn as nn

from scripts.run_nyu_qdrop_reconstruction import (
    build_seeded_batches,
    build_strict_manifest,
    configure_validation_propagation,
    build_calibration_split,
    merge_contracts,
    parse_args,
    resolve_execution_order,
    select_probability_candidate,
    stack_seeded_samples,
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


def test_runner_requires_every_path_model_phase_and_seed():
    with pytest.raises(SystemExit):
        parse_args([])

    args = parse_args([
        "--config", "qdrop.json",
        "--run-dir", "run",
        "--checkpoint", "best.pt",
        "--data-root", "data",
        "--model", "dyspn",
        "--precision", "W6A6",
        "--phase", "formal",
        "--seed", "1005",
        "--calibration-indices", "calibration_indices.json",
        "--calibration-metadata", "calibration_metadata.json",
        "--evaluation-protocol", "evaluation_protocol.json",
        "--out-dir", "output",
    ])
    assert args.model == "dyspn"
    assert args.precision == "W6A6"
    assert args.phase == "formal"
    assert args.seed == 1005


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
    assert manifest["weight_bits"] == 6
    assert manifest["activation_bits"] == 6
    assert manifest["precision"] == "W6A6"
    assert manifest["protocol"]["evaluation_seed"] == 20260812
    assert manifest["activation_policy"] == \
        "exact_semantic_edge_contract"
