from pathlib import Path

from spn_quant.rk3588_compressed_candidates import candidate_inventory


def test_inventory_freezes_four_models_and_executed_precisions():
    candidates = candidate_inventory(Path("/workspace/SPN_Quantization"))

    assert tuple(candidates) == (
        "cspn", "dyspn", "nlspn", "completionformer")
    for candidate in candidates.values():
        assert candidate.state_source.is_file()
        assert candidate.model_build_source.is_file()
        assert candidate.quantization_source.is_file()
        assert candidate.nas_candidate
        assert candidate.nas_depths
        assert candidate.research_precision_policy
        assert candidate.rk3588_execution_precisions == ("fp16", "w8a8")
        assert "w8a8" not in candidate.research_precision_policy.lower() or \
            candidate.model == "cspn"


def test_inventory_declares_model_specific_structure_and_propagation():
    candidates = candidate_inventory(Path("/workspace/SPN_Quantization"))

    assert candidates["cspn"].structured_candidate == "depth_pruned_drop_s4"
    assert candidates["cspn"].propagation_lowering == "native_convolution"
    assert candidates["nlspn"].structured_candidate == \
        "bridge62_s5hidden62_s4hidden_62p5pct"
    assert candidates["nlspn"].propagation_lowering == "grid_sample_legacy_and_packed"
    assert candidates["dyspn"].standalone_checkpoint
    assert candidates["completionformer"].standalone_checkpoint
