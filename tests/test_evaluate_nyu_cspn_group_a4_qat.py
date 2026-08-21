from pathlib import Path
from collections import OrderedDict

import numpy as np
import pytest
import torch

from scripts import evaluate_nyu_cspn_group_a4_qat as evaluator


EXPECTED = (
    "FP32",
    "PTQ_STATIC_G8_W4A4",
    "PTQ_DYNAMIC_G8_W4A4",
    "QAT_STATIC_G8_W4A4",
    "QAT_DYNAMIC_G8_W4A4",
)
MIXED_EXPECTED = (
    "FP32",
    "UNIFORM_W6A6",
    "P3_T3",
    "MIXED_TASK_AWARE_QAT",
)


def _write_predictions(root: Path, indices):
    for config in EXPECTED:
        directory = root / "predictions" / config
        directory.mkdir(parents=True)
        for index in indices:
            np.savez_compressed(
                directory / ("sample_%05d.npz" % index),
                sample_index=np.array(index), config=np.array(config))


def test_evaluation_matrix_is_fixed():
    assert evaluator.CONFIGURATIONS == EXPECTED
    assert evaluator.EXPECTED_CONFIGS == MIXED_EXPECTED


def test_prediction_coverage_requires_same_64_indices(tmp_path):
    indices = tuple(range(64))
    _write_predictions(tmp_path, indices)
    evaluator.validate_prediction_coverage(tmp_path, indices)

    (tmp_path / "predictions" / EXPECTED[-1] /
     "sample_00063.npz").unlink()
    with pytest.raises(RuntimeError, match="prediction coverage"):
        evaluator.validate_prediction_coverage(tmp_path, indices)


def test_mixed_prediction_coverage_requires_all_four_fresh_configs(tmp_path):
    indices = tuple(range(64))
    for config in MIXED_EXPECTED:
        directory = tmp_path / "predictions" / config
        directory.mkdir(parents=True)
        for index in indices:
            np.savez_compressed(
                directory / ("sample_%05d.npz" % index),
                sample_index=np.array(index), config=np.array(config))

    evaluator.validate_prediction_coverage_for(
        tmp_path, indices, evaluator.EXPECTED_CONFIGS)

    (tmp_path / "predictions" / "P3_T3" / "sample_00017.npz").unlink()
    with pytest.raises(RuntimeError, match="prediction coverage"):
        evaluator.validate_prediction_coverage_for(
            tmp_path, indices, evaluator.EXPECTED_CONFIGS)


def test_protocol_paths_are_strictly_separated():
    legacy = type("Args", (), {
        "mixed_protocol": False,
        "static_checkpoint": "static.pt",
        "dynamic_checkpoint": "dynamic.pt",
        "mixed_checkpoint": None,
        "precision_config": None,
        "assignment": None,
        "cost_basis": None,
    })()
    evaluator.validate_protocol_paths(legacy)
    mixed = type("Args", (), {
        "mixed_protocol": True,
        "static_checkpoint": None,
        "dynamic_checkpoint": None,
        "mixed_checkpoint": "mixed.pt",
        "precision_config": "config.json",
        "assignment": "assignment.json",
        "cost_basis": "cost.json",
    })()
    evaluator.validate_protocol_paths(mixed)
    mixed.cost_basis = None
    with pytest.raises(ValueError, match="all mixed"):
        evaluator.validate_protocol_paths(mixed)


def test_configuration_mode_is_explicit():
    assert evaluator.configuration_mode("FP32") is None
    assert evaluator.configuration_mode("PTQ_STATIC_G8_W4A4") == "static"
    assert evaluator.configuration_mode("QAT_DYNAMIC_G8_W4A4") == "dynamic"
    with pytest.raises(KeyError):
        evaluator.configuration_mode("unknown")


def test_mixed_checkpoint_is_loaded_only_for_mixed_qat():
    args = type("Args", (), {"mixed_checkpoint": "/runs/mixed/best.pt"})()
    assert evaluator.mixed_checkpoint_for("FP32", args) is None
    assert evaluator.mixed_checkpoint_for("UNIFORM_W6A6", args) is None
    assert evaluator.mixed_checkpoint_for("P3_T3", args) is None
    assert evaluator.mixed_checkpoint_for(
        "MIXED_TASK_AWARE_QAT", args) == Path("/runs/mixed/best.pt")


def test_mixed_assignment_contract_keeps_p3_weights_and_budget():
    from scripts import run_nyu_cspn_task_sensitive_bits as task_runner
    from spn_quant import cspn_task_sensitive_bits as allocation

    registry = task_runner.expected_registry()
    p3 = allocation.p3_t3_assignment(registry)
    mixed = allocation.BitAssignment(
        weight_bits=p3.weight_bits,
        activation_bits=tuple((owner, 4) for owner, bits in p3.activation_bits),
    )
    basis = allocation.CostBasis(
        weight_macs=tuple((name, 1) for name, bits in p3.weight_bits),
        activation_elements=tuple(
            (owner, 1) for owner, bits in p3.activation_bits),
    )

    audit = evaluator.validate_mixed_assignment_contracts(
        p3, mixed, basis, 6.0)

    assert audit.feasible
    changed = allocation.BitAssignment(
        weight_bits=tuple(
            (name, 4 if index == 0 else bits)
            for index, (name, bits) in enumerate(mixed.weight_bits)),
        activation_bits=mixed.activation_bits,
    )
    with pytest.raises(ValueError, match="weight assignments"):
        evaluator.validate_mixed_assignment_contracts(
            p3, changed, basis, 6.0)

    infeasible = allocation.BitAssignment(
        weight_bits=p3.weight_bits,
        activation_bits=tuple((owner, 8) for owner, bits in p3.activation_bits),
    )
    with pytest.raises(ValueError, match="activation budget"):
        evaluator.validate_mixed_assignment_contracts(
            p3, infeasible, basis, 6.0)


def test_precision_summary_reports_element_weighted_weight_bits():
    from spn_quant import cspn_task_sensitive_bits as allocation

    model = torch.nn.Sequential(OrderedDict((
        ("small", torch.nn.Conv2d(1, 1, 1, bias=False)),
        ("large", torch.nn.Conv2d(1, 3, 1, bias=False)),
    )))
    assignment = allocation.BitAssignment(
        weight_bits=(("small", 4), ("large", 8)),
        activation_bits=((('small', 'input'), 4),),
    )
    basis = allocation.CostBasis(
        weight_macs=(("small", 2), ("large", 6)),
        activation_elements=((('small', 'input'), 1),),
    )
    activation_budget = allocation.audit_activation_budget(
        assignment, basis, 6.0)

    summary = evaluator.precision_summary(
        model, assignment, basis, activation_budget)

    assert summary["average_weight_bits"] == 7.0
    assert summary["w8_weight_element_fraction"] == 0.75
    assert summary["w8_weight_mac_fraction"] == 0.75


def test_mixed_acceptance_uses_only_fixed64_propagation_rows():
    sample_rows = tuple({
        "config": "MIXED_TASK_AWARE_QAT",
        "RMSE": 0.1,
        "nonfinite_ratio": 0.0,
        "nonpositive_ratio": 0.0,
    } for _ in range(64))
    propagation_rows = (
        {
            "config": "MIXED_TASK_AWARE_QAT",
            "split": "calibration",
            "signal": "anchor",
            "anchor_max_error": 1.0,
        },
        {
            "config": "MIXED_TASK_AWARE_QAT",
            "split": "calibration",
            "signal": "affinity_constraints",
            "coefficient_sum_max_error": 1.0,
            "contraction_violation_rate": 1.0,
        },
        {
            "config": "MIXED_TASK_AWARE_QAT",
            "split": "evaluation",
            "signal": "anchor",
            "anchor_max_error": 0.0,
        },
        {
            "config": "MIXED_TASK_AWARE_QAT",
            "split": "evaluation",
            "signal": "affinity_constraints",
            "coefficient_sum_max_error": 0.0,
            "contraction_violation_rate": 0.0,
        },
    )
    mixed = type("Mixed", (), {
        "budget": type("Budget", (), {"average_activation_bits": 4.0})(),
        "precision_config": {"acceptance": {
            "rmse_m": 0.175,
            "average_activation_bits": 6.0,
            "nonfinite_ratio": 0.0,
            "nonpositive_ratio": 0.0,
            "anchor_max_error": 0.0,
            "coefficient_sum_max_error": 0.0,
            "contraction_violation_ratio": 0.0,
        }},
    })()

    report = evaluator.mixed_acceptance_report(
        sample_rows, propagation_rows, mixed)

    assert report["accepted"]


def test_canonical_checkpoint_rejects_parametrization_keys():
    with pytest.raises(ValueError, match="parametrization"):
        evaluator.validate_canonical_state({
            "conv.parametrizations.weight.original": np.array([1.0]),
        })


def test_qat_checkpoint_loads_after_source_range_calibration(monkeypatch):
    calls = []

    class Instrumentor:
        def refresh_parameter_sources(self):
            calls.append("refresh")

    def calibrate(*args):
        calls.append("calibrate")

    def load(*args):
        calls.append("load")
        return {"epoch": 1, "validation": {"RMSE": 0.25}}

    monkeypatch.setattr(evaluator.base, "_calibrate", calibrate)
    monkeypatch.setattr(evaluator, "_load_canonical_checkpoint", load)

    source = evaluator.prepare_deployment_state(
        "QAT_STATIC_G8_W4A4", object(), object(), object(),
        (0,), object(), 1, Instrumentor(), object(), object(), Path("qat.pt"))

    assert calls == ["calibrate", "load", "refresh"]
    assert source["epoch"] == 1


def test_visualization_upgrade_preserves_model_rgb(tmp_path):
    index = 3
    model_rgb = np.full((2, 4, 3), 0.75, dtype=np.float32)
    natural_rgb = np.full((2, 4, 3), 0.25, dtype=np.float32)
    for config in EXPECTED:
        directory = tmp_path / "predictions" / config
        directory.mkdir(parents=True)
        np.savez_compressed(
            directory / ("sample_%05d.npz" % index),
            gt=np.ones((2, 4), dtype=np.float32),
            fp32=np.ones((2, 4), dtype=np.float32),
            pred=np.ones((2, 4), dtype=np.float32),
            abs_err=np.zeros((2, 4), dtype=np.float32),
            valid_gt=np.ones((2, 4), dtype=np.bool_),
            nonfinite=np.zeros((2, 4), dtype=np.bool_),
            sample_index=np.array(index),
            model=np.array("cspn"),
            config=np.array(config),
            sparse=np.zeros((2, 4), dtype=np.float32),
            rgb=model_rgb,
        )

    class VisualizationDataset:
        def __getitem__(self, sample_index):
            assert sample_index == index
            rgb = torch.from_numpy(natural_rgb).permute(2, 0, 1)
            return {"rgbd": torch.cat((rgb, torch.zeros(1, 2, 4)), dim=0)}

    evaluator.upgrade_prediction_visuals(
        tmp_path, (index,), VisualizationDataset())
    evaluator.upgrade_prediction_visuals(
        tmp_path, (index,), VisualizationDataset())

    for config in EXPECTED:
        path = tmp_path / "predictions" / config / \
            ("sample_%05d.npz" % index)
        with np.load(path, allow_pickle=False) as source:
            assert set(source.files) == evaluator.VISUAL_PREDICTION_FIELDS
            np.testing.assert_array_equal(source["model_rgb"], model_rgb)
            np.testing.assert_array_equal(source["rgb"], natural_rgb)
    assert (tmp_path / "prediction_visualization_manifest.json").is_file()
