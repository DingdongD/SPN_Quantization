import json
from pathlib import Path
import shutil

import pytest

from scripts import run_nyu_model_p3t3_search as runner
from spn_quant import mixed_precision
from spn_quant.model_contracts import (
    QuantizationBlock,
    QuantizationModelContract,
)


def contract():
    return QuantizationModelContract(
        model_name="model_z",
        blocks=tuple(
            QuantizationBlock(
                name,
                ("weight_%s" % name,),
                (("edge_%s" % name, "input"),),
            )
            for name in ("alpha", "beta", "gamma", "delta")
        ),
        prefix_groups=(
            ("alpha",),
            ("alpha", "beta"),
            ("alpha", "beta", "gamma"),
        ),
        tail_groups=(("gamma",), ("delta",)),
        protected_roles=("propagation_state",),
        attention_edges=(),
        concat_edges=(),
        protected_modules=("protected",),
        module_roles=(("protected", "propagation_state"),),
    )


def costs():
    return mixed_precision.CostBasis(
        weight_macs=tuple(
            ("weight_%s" % name, value)
            for name, value in zip(
                ("alpha", "beta", "gamma", "delta"), (1, 2, 4, 3))
        ),
        activation_elements=tuple(
            (("edge_%s" % name, "input"), value)
            for name, value in zip(
                ("alpha", "beta", "gamma", "delta"), (1, 2, 4, 3))
        ),
    )


def score(candidate):
    prefix_scores = {
        (): 1.00,
        ("alpha",): 0.80,
        ("alpha", "beta"): 0.50,
        ("alpha", "beta", "gamma"): 0.49,
    }
    if candidate.stage == "single_block":
        return 0.95
    value = prefix_scores[candidate.prefix]
    if candidate.tail == ("gamma",):
        value -= 0.03
    elif candidate.tail == ("delta",):
        value -= 0.10
    elif candidate.tail == ("gamma", "delta"):
        value -= 0.12
    return value


class MeasuredEvaluator(object):
    def __init__(self):
        self.candidates = ()

    def __call__(self, candidates):
        self.candidates = tuple(candidates)
        rows = []
        for candidate in candidates:
            measured = score(candidate)
            for sample_index, pixels in enumerate((1, 2, 3)):
                sample_rmse = measured + sample_index * 0.01
                rows.append({
                    "config": candidate.name,
                    "sample_index": sample_index,
                    "squared_error_sum": sample_rmse ** 2 * pixels,
                    "valid_pixels": pixels,
                    "RMSE": sample_rmse,
                    "prediction_finite": True,
                    "propagation_valid": True,
                    "reproducible": True,
                })
        return tuple(rows)


def test_candidate_matrix_uses_contract_prefixes_tail_combinations_and_blocks():
    registry = mixed_precision.build_registry(contract(), costs())

    candidates = runner.build_p3_t3_candidates(contract(), registry, 4, 4, 8, 8)

    assert len(candidates) == 20
    assert tuple(candidate.stage for candidate in candidates).count(
        "single_block") == 4
    assert tuple(candidate.stage for candidate in candidates).count("prefix") == 3
    assert tuple(candidate.stage for candidate in candidates).count("tail") == 3
    assert tuple(candidate.stage for candidate in candidates).count(
        "interaction") == 9
    assert all(set(candidate.prefix + candidate.tail) <= set(registry.blocks)
               for candidate in candidates)
    assert all("stem" not in candidate.name for candidate in candidates)


def test_p3t3_search_selects_model_specific_prefix_and_budget_valid_tail():
    measured = MeasuredEvaluator()

    result = runner.search_p3_t3(
        contract=contract(),
        costs=costs(),
        evaluator=measured,
        base_weight_bits=4,
        base_activation_bits=4,
        promotion_weight_bits=8,
        promotion_activation_bits=8,
        maximum_normalized_weight_cost=1.65,
        maximum_normalized_activation_cost=1.65,
        expected_samples=3,
    )

    assert result.assignment.model_name == "model_z"
    assert result.prefix == ("alpha", "beta")
    assert result.prefix in contract().prefix_groups
    assert result.tail == ("delta",)
    assert result.tail in contract().tail_groups
    assert len(measured.candidates) == 20
    selected = next(row for row in result.candidates
                    if row.name == result.selected_candidate)
    assert selected.pooled_rmse == pytest.approx(
        ((0.40 ** 2 + 2 * 0.41 ** 2 + 3 * 0.42 ** 2) / 6) ** 0.5)
    assert selected.mean_sample_rmse == pytest.approx(0.41)
    assert selected.normalized_weight_cost == pytest.approx(1.6)
    assert selected.normalized_activation_cost == pytest.approx(1.6)
    assert selected.paired_sample_differences == pytest.approx(
        (-0.60, -0.60, -0.60))


def test_search_rejects_incomplete_measured_coverage_without_estimating_accuracy():
    class IncompleteEvaluator(MeasuredEvaluator):
        def __call__(self, candidates):
            return super().__call__(candidates)[:-1]

    with pytest.raises(ValueError, match="measured sample coverage mismatch"):
        runner.search_p3_t3(
            contract(), costs(), IncompleteEvaluator(),
            4, 4, 8, 8, 2.0, 2.0, 3)


@pytest.mark.parametrize("field,value,match", (
    ("propagation_valid", "yes", "validity flags"),
    ("RMSE", 9.0, "RMSE and squared error"),
))
def test_search_rejects_malformed_measured_rows(field, value, match):
    class MalformedEvaluator(MeasuredEvaluator):
        def __call__(self, candidates):
            rows = list(super().__call__(candidates))
            rows[0][field] = value
            return tuple(rows)

    with pytest.raises(ValueError, match=match):
        runner.search_p3_t3(
            contract(), costs(), MalformedEvaluator(),
            4, 4, 8, 8, 2.0, 2.0, 3)


def test_invalid_or_unreproducible_rows_cannot_be_selected():
    class InvalidBestEvaluator(MeasuredEvaluator):
        def __call__(self, candidates):
            rows = list(super().__call__(candidates))
            for row in rows:
                if row["config"] == "INTERACTION_P2_T02":
                    row["squared_error_sum"] = 0.0
                    row["RMSE"] = 0.0
                    row["reproducible"] = False
            return tuple(rows)

    result = runner.search_p3_t3(
        contract(), costs(), InvalidBestEvaluator(),
        4, 4, 8, 8, 1.75, 1.75, 3)

    assert result.selected_candidate == "INTERACTION_P2_T01"
    rejected = next(row for row in result.candidates
                    if row.name == "INTERACTION_P2_T02")
    assert not rejected.valid


def test_assignment_artifact_persists_measured_evidence_and_tuple_payload():
    result = runner.search_p3_t3(
        contract(), costs(), MeasuredEvaluator(),
        4, 4, 8, 8, 1.65, 1.65, 3)
    root = Path(__file__).resolve().parent / ".model_p3t3_output"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()

    path = runner.write_p3_t3_assignment(root, result)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path == root / "p3_t3_assignment.json"
    assert payload["model_name"] == "model_z"
    assert payload["prefix"] == ["alpha", "beta"]
    assert payload["tail"] == ["delta"]
    assert payload["assignment"]["weight_bits"][0] == ["weight_alpha", 8]
    assert payload["assignment"]["activation_bits"][0] == [
        ["edge_alpha", "input"], 8]
    assert payload["cost_definition"]["weight_denominator"] == 40
    assert payload["cost_definition"]["activation_denominator"] == 40
    assert payload["budgets"] == {
        "maximum_normalized_activation_cost": 1.65,
        "maximum_normalized_weight_cost": 1.65,
    }
    assert payload["cost_basis"]["weight_macs"][0] == ["weight_alpha", 1]
    assert len(payload["candidates"]) == 20
    assert all("pooled_rmse" in row and "paired_sample_differences" in row
               for row in payload["candidates"])
    shutil.rmtree(root)


def test_runtime_search_builds_the_official_contract_and_closes_runtime(monkeypatch):
    class FakeRuntime(object):
        model_name = "model_z"
        device = "cuda:7"

        def __init__(self):
            self.model = object()
            self.closed = False

        def build_model(self, device):
            assert device == "cuda:7"
            return self.model

        def close(self):
            self.closed = True

    runtime = FakeRuntime()
    monkeypatch.setattr(
        runner,
        "build_model_quantization_contract",
        lambda model_name, model: contract(),
    )

    def evaluator_factory(observed_runtime, model, observed_contract, registry):
        assert observed_runtime is runtime
        assert model is runtime.model
        assert observed_contract == contract()
        assert registry.model_name == "model_z"
        return MeasuredEvaluator()

    result = runner.run_runtime_search(
        runtime, costs(), evaluator_factory,
        4, 4, 8, 8, 1.65, 1.65, 3)

    assert result.assignment.model_name == "model_z"
    assert runtime.closed
