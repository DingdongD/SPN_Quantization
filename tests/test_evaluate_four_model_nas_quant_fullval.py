from scripts import evaluate_four_model_nas_quant_fullval as fullval


def test_summary_keeps_official_and_diagnostic_rmse_metrics():
    result = fullval._summary({
        "mean_sample_rmse": 0.1,
        "pooled_rmse": 0.2,
        "sample_count": 654,
    })

    assert result == {
        "mean_sample_rmse_m": 0.1,
        "pooled_rmse_m": 0.2,
        "sample_count": 654,
    }


def test_every_model_has_one_selected_low_bit_assignment():
    for model in fullval.nas.MODEL_ORDER:
        selected, changes = fullval._selected_changes(model)
        assert selected == fullval.SELECTED_LOW_BIT[model]
        assert changes


def test_interleaved_shards_are_disjoint_and_complete():
    indices = tuple(range(654))
    shards = tuple(indices[shard_id::4] for shard_id in range(4))

    assert sum(len(shard) for shard in shards) == 654
    assert set().union(*(set(shard) for shard in shards)) == set(indices)
    assert all(set(left).isdisjoint(right)
               for position, left in enumerate(shards)
               for right in shards[position + 1:])


def test_optional_structured_candidate_is_applied(monkeypatch):
    calls = []
    monkeypatch.setattr(
        fullval.structured, "apply_structured_candidate",
        lambda model, model_name, candidate_id: calls.append(
            (model, model_name, candidate_id)) or {"candidate_id": candidate_id})
    model = object()

    assert fullval._apply_structured_candidate(
        model, "dyspn", "bridge_62p5pct") == {
            "candidate_id": "bridge_62p5pct"}
    assert calls == [(model, "dyspn", "bridge_62p5pct")]
    assert fullval._apply_structured_candidate(model, "dyspn", None) == {}


def test_selected_assignment_uses_explicit_propagation_dtype():
    class Evaluator:
        def __init__(self):
            self.calls = []

        def evaluate_precision_assignment_with_propagation_dtype(
                self, assignment, candidate_id, propagation_dtype):
            self.calls.append((assignment, candidate_id, propagation_dtype))
            return {"propagation_dtype": propagation_dtype}

    evaluator = Evaluator()
    result = fullval._evaluate_assignment(
        evaluator, "assignment", "candidate", "bf16")

    assert result == {"propagation_dtype": "bf16"}
    assert evaluator.calls == [("assignment", "candidate", "bf16")]
