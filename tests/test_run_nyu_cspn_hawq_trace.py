import json

import pytest
import torch
import torch.nn as nn

from scripts import run_nyu_cspn_hawq_trace as runner
from spn_quant.hawq_trace import BlockTraceEstimate


def test_trace_cli_requires_every_formal_path():
    with pytest.raises(SystemExit):
        runner.parse_args([])


def test_trace_blocks_exclude_guidance_and_propagation():
    blocks = runner.hawq_block_contract(runner.expected_registry())
    names = tuple(block.name for block in blocks)

    assert names[0] == "encoder_stem"
    assert names[-1] == "initial_depth"
    assert not any(
        "guidance" in block.name or "propagation" in block.name
        for block in blocks)
    assert not any(
        module.startswith("gud_up_proj_layer6")
        for block in blocks for module in block.weight_modules)


def test_trace_manifest_requires_128_unique_train_indices():
    with pytest.raises(ValueError, match="128"):
        runner.validate_calibration_indices(tuple(range(127)))
    with pytest.raises(ValueError, match="unique"):
        runner.validate_calibration_indices(tuple(range(127)) + (126,))


def test_assignment_expands_block_bits_to_exact_registry():
    registry = runner.expected_registry()
    blocks = runner.hawq_block_contract(registry)
    selected = tuple(
        (block.name, 8 if block.fixed_eight else 4)
        for block in blocks)

    assignment = runner.expand_assignment(registry, selected)

    assert dict(assignment.weight_bits)["conv1_1"] == 8
    assert dict(assignment.activation_bits)[("conv1_1", "input")] == 8
    assert not any(name.startswith("gud_up_proj_layer6")
                   for name, bits in assignment.weight_bits)


def test_allocate_from_trace_summary_writes_complete_payload(tmp_path):
    model = nn.Sequential(
        nn.Conv2d(2, 1, 1, bias=False),
        nn.Conv2d(2, 1, 1, bias=False),
    )
    model[0].weight.data.copy_(torch.tensor([[[[1.0]], [[0.25]]]]))
    model[1].weight.data.copy_(torch.tensor([[[[1.0]], [[0.25]]]]))
    blocks = (
        runner.HAWQTraceBlock(
            "encoder_stem", ("0",), (("0", "input"),), False),
        runner.HAWQTraceBlock(
            "initial_depth", ("1",), (("1", "input"),), False),
    )
    traces = (
        BlockTraceEstimate("0", (100.0,), 100.0, 0.0, 50.0, 0.0, 2),
        BlockTraceEstimate("1", (1.0,), 1.0, 0.0, 0.5, 0.0, 2),
    )
    activation_elements = ((('0', 'input'), 1), (('1', 'input'), 1))

    result = runner.allocate_from_trace_summary(
        model, blocks, traces, activation_elements,
        bits=(4, 6, 8), maximum_weight_bits=6.0,
        maximum_activation_bits=6.0)
    runner.write_allocation_artifacts(tmp_path, result)

    assert result.assignment.block_bits == (
        ("encoder_stem", 8), ("initial_depth", 4))
    expected = {
        "trace_summary.csv",
        "candidate_costs.csv",
        "cost_basis.json",
        "selected_assignment.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    payload = json.loads(
        (tmp_path / "selected_assignment.json").read_text(encoding="utf-8"))
    assert payload["average_weight_bits"] == 6.0
    assert payload["average_activation_bits"] == 6.0


def test_trace_summary_rejects_undeclared_module_coverage():
    model = nn.Sequential(
        nn.Conv2d(1, 1, 1, bias=False),
        nn.Conv2d(1, 1, 1, bias=False),
    )
    trace = BlockTraceEstimate(
        "0", (1.0,), 1.0, 0.0, 1.0, 0.0, 1)

    with pytest.raises(ValueError, match="coverage"):
        runner._summarize_traces(
            (trace,), {"0": model[0], "1": model[1]})
