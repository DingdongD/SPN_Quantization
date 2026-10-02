import pytest

from scripts import run_depth_nas_quant_search as runner


def test_candidate_depths_resolves_exact_candidate():
    assert runner.candidate_depths("dyspn", "drop_s4") == (3, 4, 6, 2)
    assert runner.candidate_depths(
        "completionformer", "drop_pvt_s3") == (3, 4, 3, 4, 3, 3)
    assert runner.candidate_depths(
        "completionformer", "drop_pvt_s3_s4") == (3, 4, 3, 4, 3, 2)


def test_candidate_depths_rejects_unknown_candidate():
    with pytest.raises(ValueError, match="unknown NAS candidate"):
        runner.candidate_depths("cspn", "missing")
