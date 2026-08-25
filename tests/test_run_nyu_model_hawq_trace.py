import json

import pytest
import torch
import torch.nn as nn

from scripts import run_nyu_model_hawq_trace as runner
from spn_quant.hawq_trace import BlockTraceEstimate
from spn_quant.model_contracts import (
    QuantizationBlock,
    QuantizationModelContract,
)


class TinyDepthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(2, 1, 1, bias=False)
        self.decoder = nn.Conv2d(2, 1, 1, bias=False)
        self.protected = nn.Conv2d(2, 1, 1, bias=False)
        self.encoder.weight.data.copy_(torch.tensor([[[[1.0]], [[0.25]]]]))
        self.decoder.weight.data.copy_(torch.tensor([[[[0.5]], [[0.125]]]]))

    def forward(self, value):
        return {
            "official_prediction":
                self.encoder(value) + self.decoder(value) +
                self.protected(value),
        }


class TinyRuntime(object):
    def __init__(self):
        self.device = torch.device("cpu")
        self.model_name = "completionformer"
        self.prediction_calls = 0

    def model_input(self, batch, device):
        return (batch["input"].to(device),), batch["target"].to(device)

    def prediction(self, output):
        self.prediction_calls += 1
        return output["official_prediction"]


def contract():
    return QuantizationModelContract(
        model_name="completionformer",
        blocks=(
            QuantizationBlock(
                "encoder", ("encoder",),
                (("activation::encoder::input", "module_input"),)),
            QuantizationBlock(
                "decoder", ("decoder",),
                (("attention::decoder::q", "q"),)),
        ),
        prefix_groups=(("encoder",),),
        tail_groups=(("decoder",),),
        protected_roles=("propagation_state",),
        attention_edges=("attention::decoder::q",),
        concat_edges=(),
        protected_modules=("protected",),
        module_roles=(("protected", "propagation_state"),),
    )


def test_trace_parameter_blocks_follow_contract_and_exclude_protected_modules():
    model = TinyDepthModel()

    blocks = runner.build_trace_parameter_blocks(model, contract())

    assert tuple(block.name for block in blocks) == contract().block_names
    assert tuple(block.module_names for block in blocks) == (
        ("encoder",), ("decoder",))
    assert all(
        model.protected.weight is not parameter
        for block in blocks for parameter in block.parameters)


def test_cost_blocks_require_exact_macs_and_include_attention_traffic():
    blocks = runner.build_cost_blocks(
        TinyDepthModel(),
        contract(),
        weight_macs=(("encoder", 20), ("decoder", 30)),
        activation_traffic=(
            (("activation::encoder::input", "module_input"), 100),
            (("attention::decoder::q", "q"), 300),
        ),
    )

    assert blocks[0].weight_parameters == 2
    assert blocks[0].weight_macs == 20
    assert blocks[0].activation_traffic == 100
    assert blocks[1].activation_traffic == 300

    with pytest.raises(ValueError, match="activation cost coverage"):
        runner.build_cost_blocks(
            TinyDepthModel(), contract(),
            weight_macs=(("encoder", 20), ("decoder", 30)),
            activation_traffic=((
                ("activation::encoder::input", "module_input"), 100),),
        )


def test_cost_blocks_allow_contract_weight_block_without_activation_owner():
    weight_only_contract = QuantizationModelContract(
        model_name="completionformer",
        blocks=(
            QuantizationBlock("encoder", ("encoder",), ()),
            QuantizationBlock(
                "decoder", ("decoder",),
                (("attention::decoder::q", "q"),)),
        ),
        prefix_groups=(("encoder",),),
        tail_groups=(("decoder",),),
        protected_roles=("propagation_state",),
        attention_edges=("attention::decoder::q",),
        concat_edges=(),
        protected_modules=("protected",),
        module_roles=(("protected", "propagation_state"),),
    )

    blocks = runner.build_cost_blocks(
        TinyDepthModel(), weight_only_contract,
        weight_macs=(("encoder", 20), ("decoder", 30)),
        activation_traffic=((
            ("attention::decoder::q", "q"), 300),),
    )

    assert blocks[0].activation_traffic == 0
    assert blocks[1].activation_traffic == 300


def test_trace_uses_runtime_prediction_for_exact_128_calibration_identities():
    model = TinyDepthModel()
    runtime = TinyRuntime()
    dataset = tuple({
        "input": torch.tensor([[[1.0]], [[0.5]]]),
        "target": torch.ones(1, 1, 1),
    } for _ in range(128))
    settings = runner.HAWQTraceSettings(
        batch_size=128,
        probes_per_batch=1,
        seed=17,
        depth_mse_weight=1.0,
        boundary_mse_weight=0.0,
        boundary_threshold_m=0.5,
    )

    traced = runner.trace_calibration_batches(
        runtime, model, contract(), dataset, tuple(range(128)), settings)

    assert tuple(row.block for row in traced.traces) == contract().block_names
    assert runtime.prediction_calls == 1
    assert traced.calibration_indices == tuple(range(128))
    assert len(traced.raw_rows) == 2

    with pytest.raises(ValueError, match="128"):
        runner.trace_calibration_batches(
            runtime, model, contract(), dataset, tuple(range(127)), settings)


def test_allocation_persists_honest_objective_and_separate_assignments(
        tmp_path):
    model = TinyDepthModel()
    traces = (
        BlockTraceEstimate(
            "encoder", (200.0,), 200.0, 0.0, 100.0, 0.0, 2),
        BlockTraceEstimate(
            "decoder", (2.0,), 2.0, 0.0, 1.0, 0.0, 2),
    )
    weight_macs = (("encoder", 20), ("decoder", 20))
    activation_traffic = (
        (("activation::encoder::input", "module_input"), 30),
        (("attention::decoder::q", "q"), 10),
    )

    result = runner.allocate_contract_hawq(
        model, contract(), traces, weight_macs, activation_traffic,
        bits=(4, 6, 8), maximum_weight_bits=6.0,
        maximum_activation_bits=6.0)
    with pytest.raises(ValueError, match="calibration identity"):
        runner.write_hawq_assignment(
            tmp_path, contract(), result,
            tuple(range(128)), "calibration-identity")
    identity = runner.ordered_sample_identity_sha256(
        "train", tuple(range(128)))
    path = runner.write_hawq_assignment(
        tmp_path, contract(), result,
        tuple(range(128)), identity)

    assert result.assignment.average_weight_bits <= 6.0
    assert result.assignment.average_weight_mac_bits <= 6.0
    assert result.assignment.average_activation_bits <= 6.0
    assert result.assignment.weight_block_bits != \
        result.assignment.activation_block_bits
    assert all(
        row.cost == row.normalized_trace * row.quantization_error
        for row in result.objective_components)
    assert path.name == "hawq_mixed_le6_assignment.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "block_bits" not in payload["assignment"]
    assert payload["assignment"]["weight_bits"]
    assert payload["assignment"]["activation_bits"]
    assert payload["average_weight_bits"] <= 6.0
    assert payload["average_weight_mac_bits"] <= 6.0
    assert payload["average_activation_bits"] <= 6.0
    assert payload["objective"]["activation_sensitivity"] == \
        "not_estimated"
    assert payload["constraints"]["weight_parameter_residual"] >= 0.0
    assert payload["constraints"]["weight_mac_residual"] >= 0.0
    assert payload["constraints"]["activation_traffic_residual"] >= 0.0
    attention_rows = tuple(
        row for row in payload["cost_basis"]["activation_traffic"]
        if row["site"] == "attention::decoder::q")
    assert attention_rows == ({
        "site": "attention::decoder::q",
        "role": "q",
        "elements": 10,
    },)


def test_calibration_metadata_preserves_ordered_fixed_identity(tmp_path):
    path = tmp_path / "calibration_metadata.json"
    path.write_text(json.dumps({
        "calibration_indices": list(range(128)),
        "evaluation_indices": list(range(64)),
        "calibration_source": {"selection": "32_tail_96_kmedoids"},
    }), encoding="utf-8")

    identity = runner.load_calibration_identity(
        path, tuple(range(64)), dataset_size=256)

    assert identity.indices == tuple(range(128))
    assert identity.sha256 == runner.ordered_sample_identity_sha256(
        "train", tuple(range(128)))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evaluation_indices"][0], payload["evaluation_indices"][1] = \
        payload["evaluation_indices"][1], payload["evaluation_indices"][0]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation identities"):
        runner.load_calibration_identity(
            path, tuple(range(64)), dataset_size=256)
