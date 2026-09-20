import math

import pytest

from spn_quant.nas.search import (
    make_search_split,
    pareto_rank,
    select_successive_halving,
    validate_search_split,
)


def test_search_split_is_reproducible_complete_and_disjoint(tmp_path):
    source = tmp_path / "train.csv"
    source.write_text("Name\n" + "\n".join(f"{i:05d}.h5" for i in range(6700)))

    first = make_search_split(6700, 670, seed=20260920, source_list=source)
    second = make_search_split(6700, 670, seed=20260920, source_list=source)

    assert first == second
    assert len(first["train_indices"]) == 6030
    assert len(first["dev_indices"]) == 670
    assert set(first["train_indices"]).isdisjoint(first["dev_indices"])
    assert first["source_list_sha256"]
    validate_search_split(first, total_count=6700, source_list=source)


def test_split_validation_rejects_duplicate_indices(tmp_path):
    source = tmp_path / "train.csv"
    source.write_text("Name\na\nb\n")
    manifest = make_search_split(2, 1, seed=1, source_list=source)
    manifest["train_indices"] = list(manifest["dev_indices"])

    with pytest.raises(ValueError, match="overlap"):
        validate_search_split(manifest, total_count=2, source_list=source)


def test_pareto_rank_keeps_tradeoffs_and_marks_dominated_rows():
    rows = [
        {"candidate": "fast", "rmse": 0.150, "parameters": 12, "latency_ms": 3.0},
        {"candidate": "accurate", "rmse": 0.140, "parameters": 20, "latency_ms": 4.0},
        {"candidate": "small", "rmse": 0.155, "parameters": 8, "latency_ms": 3.5},
        {"candidate": "dominated", "rmse": 0.160, "parameters": 30, "latency_ms": 5.0},
    ]

    ranked = {row["candidate"]: row for row in pareto_rank(rows)}

    assert ranked["fast"]["pareto_rank"] == 0
    assert ranked["accurate"]["pareto_rank"] == 0
    assert ranked["small"]["pareto_rank"] == 0
    assert ranked["dominated"]["pareto_rank"] > 0


def test_successive_halving_retains_quarter_with_floor_of_eight():
    rows = [
        {
            "candidate": f"c{index:02d}",
            "rmse": 0.14 + index * 0.001,
            "parameters": 100 - index,
            "latency_ms": 2.0 + (index % 4) * 0.1,
        }
        for index in range(20)
    ]

    selected = select_successive_halving(rows)

    assert len(selected) == 8
    assert selected == select_successive_halving(list(reversed(rows)))


def test_pareto_rank_rejects_nonfinite_objectives():
    with pytest.raises(ValueError, match="non-finite"):
        pareto_rank([{
            "candidate": "bad",
            "rmse": math.nan,
            "parameters": 1,
            "latency_ms": 1,
        }])
