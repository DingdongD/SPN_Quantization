import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch
import torch.nn as nn

from scripts import run_nyu_model_hawq_trace as runner
from spn_quant.hawq_trace import BlockTraceEstimate
from spn_quant.model_contracts import (
    QuantizationBlock,
    QuantizationModelContract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON37 = Path("/opt/conda/envs/completionformer-py37/bin/python")


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


def trace_settings(seed=17):
    return runner.HAWQTraceSettings(
        batch_size=128,
        probes_per_batch=1,
        seed=seed,
        depth_mse_weight=1.0,
        boundary_mse_weight=0.0,
        boundary_threshold_m=0.5,
    )


def test_trace_module_import_and_help_do_not_require_scipy_in_python37():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT)

    imported = subprocess.run(
        (str(PYTHON37), "-c",
         "import scripts.run_nyu_model_hawq_trace"),
        cwd=str(REPO_ROOT), env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    helped = subprocess.run(
        (str(PYTHON37),
         str(REPO_ROOT / "scripts/run_nyu_model_hawq_trace.py"), "--help"),
        cwd=str(REPO_ROOT), env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    assert imported.returncode == 0, imported.stderr
    assert helped.returncode == 0, helped.stderr
    assert "--phase" in helped.stdout
    assert "trace" in helped.stdout
    assert "allocate" in helped.stdout


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


def test_trace_uses_runtime_prediction_for_exact_128_calibration_identities(
        tmp_path):
    model = TinyDepthModel()
    runtime = TinyRuntime()
    dataset = tuple({
        "input": torch.tensor([[[1.0]], [[0.5]]]),
        "target": torch.ones(1, 1, 1),
    } for _ in range(128))
    settings = trace_settings()
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"official-checkpoint")
    checkpoint_identity = runner.capture_checkpoint_identity(checkpoint)

    traced = runner.trace_calibration_batches(
        runtime, model, contract(), dataset, tuple(range(128)), settings,
        checkpoint_identity)

    assert tuple(row.block for row in traced.traces) == contract().block_names
    assert runtime.prediction_calls == 1
    assert traced.calibration_indices == tuple(range(128))
    assert traced.settings == settings
    assert traced.checkpoint_identity == checkpoint_identity
    assert len(traced.raw_rows) == 2

    with pytest.raises(ValueError, match="128"):
        runner.trace_calibration_batches(
            runtime, model, contract(), dataset, tuple(range(127)), settings,
            checkpoint_identity)


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


def test_trace_artifact_roundtrip_is_complete_and_identity_bound(tmp_path):
    model = TinyDepthModel()
    traces = (
        BlockTraceEstimate(
            "encoder", (200.0,), 200.0, 0.0, 100.0, 0.0, 2),
        BlockTraceEstimate(
            "decoder", (2.0,), 2.0, 0.0, 1.0, 0.0, 2),
    )
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"official-checkpoint")
    traced = runner.HAWQTraceRun(
        traces=traces,
        raw_rows=(
            {"batch_start": 0, "block": "encoder",
             "probe": 0, "estimate": 200.0},
            {"batch_start": 0, "block": "decoder",
             "probe": 0, "estimate": 2.0},
        ),
        calibration_indices=tuple(range(128)),
        settings=trace_settings(),
        checkpoint_identity=runner.capture_checkpoint_identity(checkpoint),
    )
    identity = runner.ordered_sample_identity_sha256(
        "train", traced.calibration_indices)
    trace_output = tmp_path / "trace"
    trace_output.mkdir()
    allocation_output = tmp_path / "allocation"
    allocation_output.mkdir()

    trace_path = runner.write_trace_artifact(
        trace_output,
        model=model,
        contract=contract(),
        traced=traced,
        weight_macs=(("encoder", 20), ("decoder", 20)),
        activation_traffic=(
            (("activation::encoder::input", "module_input"), 30),
            (("attention::decoder::q", "q"), 10),
        ),
        bits=(4, 6, 8),
        model_name="completionformer",
        calibration_identity=identity,
    )
    assignment_path = runner.allocate_trace_artifact(
        trace_path,
        allocation_output,
        expected_model_name="completionformer",
        expected_checkpoint=checkpoint,
        expected_calibration_indices=traced.calibration_indices,
        expected_calibration_identity=identity,
        expected_trace_settings=trace_settings(),
        bits=(4, 6, 8),
        maximum_weight_bits=6.0,
        maximum_activation_bits=6.0,
    )

    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assignment_payload = json.loads(
        assignment_path.read_text(encoding="utf-8"))
    assert trace_payload["artifact_kind"] == "nyu_contract_hawq_trace"
    assert trace_payload["checkpoint"]["path"] == str(checkpoint.resolve())
    assert trace_payload["checkpoint"]["size_bytes"] == len(
        b"official-checkpoint")
    assert trace_payload["checkpoint"]["sha256"] == \
        traced.checkpoint_identity.sha256
    assert trace_payload["trace_settings"] == {
        "batch_size": 128,
        "probes_per_batch": 1,
        "seed": 17,
        "depth_mse_weight": 1.0,
        "boundary_mse_weight": 0.0,
        "boundary_threshold_m": 0.5,
    }
    assert trace_payload["contract"]["blocks"][0]["name"] == "encoder"
    assert trace_payload["traces"][0]["block"] == "encoder"
    assert trace_payload["cost_basis"]["weight_macs"]
    assert trace_payload["cost_basis"]["activation_traffic"]
    assert trace_payload["objective"]["components"]
    assert assignment_payload["average_weight_bits"] <= 6.0
    assert assignment_payload["average_activation_bits"] <= 6.0

    checkpoint.write_bytes(b"changed-checkpoint")
    identity_output = tmp_path / "identity-rejected"
    identity_output.mkdir()
    with pytest.raises(ValueError, match="checkpoint identity"):
        runner.allocate_trace_artifact(
            trace_path,
            identity_output,
            expected_model_name="completionformer",
            expected_checkpoint=checkpoint,
            expected_calibration_indices=traced.calibration_indices,
            expected_calibration_identity=identity,
            expected_trace_settings=trace_settings(),
            bits=(4, 6, 8),
            maximum_weight_bits=6.0,
            maximum_activation_bits=6.0,
        )
    checkpoint.write_bytes(b"official-checkpoint")
    mismatch_output = tmp_path / "settings-mismatch"
    mismatch_output.mkdir()
    with pytest.raises(ValueError, match="trace settings"):
        runner.allocate_trace_artifact(
            trace_path,
            mismatch_output,
            expected_model_name="completionformer",
            expected_checkpoint=checkpoint,
            expected_calibration_indices=traced.calibration_indices,
            expected_calibration_identity=identity,
            expected_trace_settings=trace_settings(seed=18),
            bits=(4, 6, 8),
            maximum_weight_bits=6.0,
            maximum_activation_bits=6.0,
        )
    settings_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    settings_payload["trace_settings"]["seed"] = 18
    settings_tampered = tmp_path / "settings-tampered.json"
    settings_tampered.write_text(
        json.dumps(settings_payload), encoding="utf-8")
    settings_output = tmp_path / "settings-rejected"
    settings_output.mkdir()
    with pytest.raises(ValueError, match="trace settings"):
        runner.allocate_trace_artifact(
            settings_tampered,
            settings_output,
            expected_model_name="completionformer",
            expected_checkpoint=checkpoint,
            expected_calibration_indices=traced.calibration_indices,
            expected_calibration_identity=identity,
            expected_trace_settings=trace_settings(),
            bits=(4, 6, 8),
            maximum_weight_bits=6.0,
            maximum_activation_bits=6.0,
        )
    tampered = tmp_path / "tampered.json"
    trace_payload["objective"]["components"].pop()
    tampered.write_text(json.dumps(trace_payload), encoding="utf-8")
    rejected_output = tmp_path / "rejected"
    rejected_output.mkdir()
    with pytest.raises(ValueError, match="objective component coverage"):
        runner.allocate_trace_artifact(
            tampered,
            rejected_output,
            expected_model_name="completionformer",
            expected_checkpoint=checkpoint,
            expected_calibration_indices=traced.calibration_indices,
            expected_calibration_identity=identity,
            expected_trace_settings=trace_settings(),
            bits=(4, 6, 8),
            maximum_weight_bits=6.0,
            maximum_activation_bits=6.0,
        )

    trace_rows = trace_output / "hawq_trace_rows.csv"
    changed_rows = trace_rows.read_text(encoding="utf-8").replace(
        "0,encoder,0,200.0", "1,encoder,0,200.0", 1)
    trace_rows.write_text(changed_rows, encoding="utf-8")
    row_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    row_payload["files"]["trace_rows"]["sha256"] = hashlib.sha256(
        trace_rows.read_bytes()).hexdigest()
    trace_path.write_text(json.dumps(row_payload), encoding="utf-8")
    row_output = tmp_path / "rows-rejected"
    row_output.mkdir()
    with pytest.raises(ValueError, match="row coverage"):
        runner.allocate_trace_artifact(
            trace_path,
            row_output,
            expected_model_name="completionformer",
            expected_checkpoint=checkpoint,
            expected_calibration_indices=traced.calibration_indices,
            expected_calibration_identity=identity,
            expected_trace_settings=trace_settings(),
            bits=(4, 6, 8),
            maximum_weight_bits=6.0,
            maximum_activation_bits=6.0,
        )


def test_trace_publication_rejects_checkpoint_replaced_after_capture(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint-loaded-by-model")
    captured = runner.capture_checkpoint_identity(checkpoint)
    traced = runner.HAWQTraceRun(
        traces=(
            BlockTraceEstimate(
                "encoder", (200.0,), 200.0, 0.0, 100.0, 0.0, 2),
            BlockTraceEstimate(
                "decoder", (2.0,), 2.0, 0.0, 1.0, 0.0, 2),
        ),
        raw_rows=(
            {"batch_start": 0, "block": "encoder",
             "probe": 0, "estimate": 200.0},
            {"batch_start": 0, "block": "decoder",
             "probe": 0, "estimate": 2.0},
        ),
        calibration_indices=tuple(range(128)),
        settings=trace_settings(),
        checkpoint_identity=captured,
    )
    checkpoint.write_bytes(b"replacement-checkpoint")
    output = tmp_path / "trace"
    output.mkdir()

    with pytest.raises(ValueError, match="checkpoint identity changed"):
        runner.write_trace_artifact(
            output,
            model=TinyDepthModel(),
            contract=contract(),
            traced=traced,
            weight_macs=(("encoder", 20), ("decoder", 20)),
            activation_traffic=(
                (("activation::encoder::input", "module_input"), 30),
                (("attention::decoder::q", "q"), 10),
            ),
            bits=(4, 6, 8),
            model_name="completionformer",
            calibration_identity=runner.ordered_sample_identity_sha256(
                "train", tuple(range(128))),
        )
    assert not tuple(output.iterdir())


@pytest.mark.parametrize(
    ("maximum_weight_bits", "maximum_activation_bits"),
    ((8.0, 6.0), (6.0, 8.0)),
)
def test_named_mixed_le6_allocation_rejects_budget_above_six(
        tmp_path, maximum_weight_bits, maximum_activation_bits):
    output = tmp_path / "allocation"
    output.mkdir()

    with pytest.raises(ValueError, match="mixed_le6"):
        runner.allocate_trace_artifact(
            tmp_path / "unused-trace.json",
            output,
            expected_model_name="completionformer",
            expected_checkpoint=tmp_path / "unused-checkpoint.pt",
            expected_calibration_indices=tuple(range(128)),
            expected_calibration_identity=runner.ordered_sample_identity_sha256(
                "train", tuple(range(128))),
            expected_trace_settings=trace_settings(),
            bits=(4, 6, 8),
            maximum_weight_bits=maximum_weight_bits,
            maximum_activation_bits=maximum_activation_bits,
        )


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
