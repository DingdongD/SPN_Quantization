import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import run_nyu_selected_ptq as runner
from scripts.run_nyu_rtn_quantization import build_contract_rtn_plan
from spn_quant.mixed_precision import BitAssignment
from spn_quant.model_contracts import (
    QuantizationBlock,
    QuantizationModelContract,
)


def contract():
    return QuantizationModelContract(
        model_name="model_z",
        blocks=(
            QuantizationBlock(
                "encoder", ("encoder.conv",),
                (("activation::encoder.conv::input", "module_input"),)),
            QuantizationBlock(
                "decoder", ("decoder.conv",),
                (("attention::decoder.attn::q", "q"),
                 ("concat::decoder.merge::cnn_input", "cnn_input"))),
        ),
        prefix_groups=(("encoder",),),
        tail_groups=(("decoder",),),
        protected_roles=("propagation_state",),
        attention_edges=("attention::decoder.attn::q",),
        concat_edges=("concat::decoder.merge::cnn_input",),
        protected_modules=("propagation.conv",),
        module_roles=(("propagation.conv", "propagation_state"),),
    )


def assignment():
    return BitAssignment(
        weight_bits=(("encoder.conv", 8), ("decoder.conv", 4)),
        activation_bits=(
            (("activation::encoder.conv::input", "module_input"), 8),
            (("attention::decoder.attn::q", "q"), 4),
            (("concat::decoder.merge::cnn_input", "cnn_input"), 4),
        ),
        model_name="model_z",
    )


def test_selected_ptq_matrix_is_exact():
    assert runner.selected_ptq_methods() == (
        "rtn_w8a8",
        "rtn_w4a4",
        "qdrop_w6a6",
        "brecq_w6a6",
        "p3_t3_mixed_ptq",
    )


@pytest.mark.parametrize(
    ("method", "weight_bits", "activation_bits"),
    (("rtn_w8a8", 8, 8), ("rtn_w4a4", 4, 4)),
)
def test_uniform_rtn_plan_uses_only_contract_owned_sites(
        method, weight_bits, activation_bits):
    plan = build_contract_rtn_plan(
        contract(), method, weight_bits, activation_bits, None)

    assert plan.module_names == ("encoder.conv", "decoder.conv")
    assert dict(plan.weight_bits) == {
        "encoder.conv": weight_bits,
        "decoder.conv": weight_bits,
    }
    assert set(dict(plan.activation_bits)) == {
        ("activation::encoder.conv::input", "module_input"),
        ("attention::decoder.attn::q", "q"),
        ("concat::decoder.merge::cnn_input", "cnn_input"),
    }
    assert set(contract().protected_modules).isdisjoint(plan.module_names)
    assert plan.attention_edges == contract().attention_edges
    assert plan.concat_edges == contract().concat_edges


def test_selected_method_plan_rejects_calibration_or_field_drift():
    with pytest.raises(ValueError, match="calibration count"):
        runner._method_plan(
            "rtn_w8a8",
            {"weight_bits": 8, "activation_bits": 8,
             "calibration_count": 64},
            contract(),
            Path("unused.json"),
        )
    with pytest.raises(ValueError, match="field contract"):
        runner._method_plan(
            "qdrop_w6a6",
            {"weight_bits": 6, "activation_bits": 6, "steps": 20000,
             "calibration_count": 128, "precision": "fallback"},
            contract(),
            Path("unused.json"),
        )


def test_p3_t3_plan_requires_complete_exact_assignment():
    plan = build_contract_rtn_plan(
        contract(), "p3_t3_mixed_ptq", 4, 4, assignment())

    assert tuple(name for name, bits in plan.weight_bits) == \
        contract().weight_modules
    assert dict(plan.weight_bits) == dict(assignment().weight_bits)
    assert dict(plan.activation_bits) == dict(assignment().activation_bits)

    incomplete = BitAssignment(
        weight_bits=(("encoder.conv", 8),),
        activation_bits=assignment().activation_bits,
        model_name="model_z",
    )
    with pytest.raises(ValueError, match="weight assignment coverage"):
        build_contract_rtn_plan(
            contract(), "p3_t3_mixed_ptq", 4, 4, incomplete)


def test_load_p3_t3_assignment_rejects_precision_or_model_drift(tmp_path):
    payload = {
        "model_name": "model_z",
        "precision": {
            "base_weight_bits": 4,
            "base_activation_bits": 4,
            "promotion_weight_bits": 8,
            "promotion_activation_bits": 8,
        },
        "assignment": {
            "model_name": "model_z",
            "weight_bits": [["decoder.conv", 4], ["encoder.conv", 8]],
            "activation_bits": [
                [["activation::encoder.conv::input", "module_input"], 8],
                [["attention::decoder.attn::q", "q"], 4],
                [["concat::decoder.merge::cnn_input", "cnn_input"], 4],
            ],
        },
    }
    path = tmp_path / "p3_t3_assignment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = runner.load_p3_t3_assignment(
        path, contract(), {
            "base_weight_bits": 4,
            "base_activation_bits": 4,
            "promotion_weight_bits": 8,
            "promotion_activation_bits": 8,
        })

    assert loaded == assignment()
    payload["assignment"]["weight_bits"][0][1] = 6
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="precision values"):
        runner.load_p3_t3_assignment(
            path, contract(), {
                "base_weight_bits": 4,
                "base_activation_bits": 4,
                "promotion_weight_bits": 8,
                "promotion_activation_bits": 8,
            })
    payload["assignment"]["weight_bits"][0][1] = 4
    payload["model_name"] = "other"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="model"):
        runner.load_p3_t3_assignment(
            path, contract(), {
                "base_weight_bits": 4,
                "base_activation_bits": 4,
                "promotion_weight_bits": 8,
                "promotion_activation_bits": 8,
            })


def hard_manifest(tmp_path, method, state_name):
    weights = tmp_path / (method + "_hard_weights.pt")
    torch.save({"state": state_name}, weights)
    state = tmp_path / (state_name + ".json")
    state.write_text("{}", encoding="utf-8")
    contract_path = tmp_path / (method + "_contract.pt")
    torch.save({"strict": 1}, contract_path)
    method_bits = {
        "rtn_w8a8": 8,
        "rtn_w4a4": 4,
        "qdrop_w6a6": 6,
        "brecq_w6a6": 6,
    }
    weight_bits = method_bits[method] \
        if method in method_bits else [
            ["encoder.conv", 8], ["decoder.conv", 4]]
    activation_bits = method_bits[method] \
        if method in method_bits else [
            [["activation::encoder.conv::input", "module_input"], 8],
            [["attention::decoder.attn::q", "q"], 4],
            [["concat::decoder.merge::cnn_input", "cnn_input"], 4],
        ]
    payload = {
        "format_version": 1,
        "strict": 1,
        "method": method,
        "model": "model_z",
        "weight_bits": weight_bits,
        "activation_bits": activation_bits,
        "module_names": ["encoder.conv", "decoder.conv"],
        "activation_owners": [
            ["activation::encoder.conv::input", "module_input"],
            ["attention::decoder.attn::q", "q"],
            ["concat::decoder.merge::cnn_input", "cnn_input"],
        ],
        "protected_modules": ["propagation.conv"],
        "materialized_hard_weights": 1,
        "hard_weights": str(weights),
        "hard_weights_sha256": runner.file_sha256(weights),
        "deployment_contract": str(contract_path),
        "deployment_contract_sha256": runner.file_sha256(contract_path),
        "optimization_state": str(state),
        "optimization_state_sha256": runner.file_sha256(state),
        "calibration_identity": "calibration-sha",
        "evaluation_identity": "evaluation-sha",
    }
    path = tmp_path / (method + "_hard_deployment_manifest.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_reconstruction_manifests_require_materialized_hard_weights(tmp_path):
    path = hard_manifest(tmp_path, "qdrop_w6a6", "qdrop_state")

    payload = runner.validate_hard_deployment_manifest(
        path, "qdrop_w6a6", contract(), "calibration-sha",
        "evaluation-sha")

    assert payload["materialized_hard_weights"] == 1
    payload["materialized_hard_weights"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="materialized hard weights"):
        runner.validate_hard_deployment_manifest(
            path, "qdrop_w6a6", contract(), "calibration-sha",
            "evaluation-sha")


def test_qdrop_and_brecq_keep_shared_calibration_and_distinct_states(tmp_path):
    qdrop = hard_manifest(tmp_path, "qdrop_w6a6", "qdrop_state")
    brecq = hard_manifest(tmp_path, "brecq_w6a6", "brecq_state")

    manifests = runner.validate_selected_reconstruction_pair(
        qdrop, brecq, contract(), "calibration-sha", "evaluation-sha")

    assert manifests[0]["calibration_identity"] == \
        manifests[1]["calibration_identity"]
    assert manifests[0]["optimization_state"] != \
        manifests[1]["optimization_state"]


def test_matrix_orchestration_uses_runtime_contract_and_exact_method_order(
        tmp_path):
    assignment_payload = {
        "model_name": "model_z",
        "precision": {
            "base_weight_bits": 4,
            "base_activation_bits": 4,
            "promotion_weight_bits": 8,
            "promotion_activation_bits": 8,
        },
        "assignment": {
            "model_name": "model_z",
            "weight_bits": [["decoder.conv", 4], ["encoder.conv", 8]],
            "activation_bits": [
                [["activation::encoder.conv::input", "module_input"], 8],
                [["attention::decoder.attn::q", "q"], 4],
                [["concat::decoder.merge::cnn_input", "cnn_input"], 4],
            ],
        },
    }
    assignment_path = tmp_path / "p3_t3_assignment.json"
    assignment_path.write_text(
        json.dumps(assignment_payload), encoding="utf-8")
    methods = {
        "rtn_w8a8": {
            "weight_bits": 8, "activation_bits": 8,
            "calibration_count": 128,
        },
        "rtn_w4a4": {
            "weight_bits": 4, "activation_bits": 4,
            "calibration_count": 128,
        },
        "qdrop_w6a6": {
            "weight_bits": 6, "activation_bits": 6,
            "steps": 20000, "calibration_count": 128,
        },
        "brecq_w6a6": {
            "weight_bits": 6, "activation_bits": 6,
            "steps": 20000, "calibration_count": 128,
        },
        "p3_t3_mixed_ptq": assignment_payload["precision"],
    }
    events = []

    class Runtime(object):
        model_name = "model_z"
        device = torch.device("cuda:7")

        def build_model(self, device):
            assert device == self.device
            events.append(("build", self.model_name))
            return object()

        def close(self):
            events.append(("close", self.model_name))

    def execute(runtime, model, observed_contract, plan, method, output,
                method_config, calibration_identity, evaluation_identity):
        del runtime, model, method_config
        assert observed_contract == contract()
        assert plan.method == method
        assert calibration_identity == "calibration-sha"
        assert evaluation_identity == "evaluation-sha"
        events.append(("execute", method))
        output.mkdir(parents=True)
        return hard_manifest(output, method, method + "_state")

    dependencies = runner.SelectedPTQDependencies(
        runtime_factory=lambda model_config: Runtime(),
        contract_builder=lambda model_name, model: contract(),
        method_executor=execute,
    )

    result = runner.run_selected_ptq_matrix(
        model_config=SimpleNamespace(model="model_z"),
        method_hyperparameters=methods,
        p3_t3_assignment=assignment_path,
        output=tmp_path / "matrix",
        calibration_identity="calibration-sha",
        evaluation_identity="evaluation-sha",
        dependencies=dependencies,
    )

    assert tuple(result) == runner.selected_ptq_methods()
    assert [event[1] for event in events if event[0] == "execute"] == \
        list(runner.selected_ptq_methods())
    assert len([event for event in events if event[0] == "build"]) == 5
    assert len([event for event in events if event[0] == "close"]) == 5


@pytest.mark.parametrize(
    ("model_name", "device", "method", "algorithm"),
    (
        ("nlspn", "cuda:1", "qdrop_w6a6", "qdrop"),
        ("completionformer", "cuda:2", "brecq_w6a6", "brecq"),
    ),
)
def test_production_reconstruction_receives_selected_runtime_device(
        monkeypatch, tmp_path, model_name, device, method, algorithm):
    observed = {}

    def reconstruct(args, config, probability, split, protocol, phase,
                    output, observed_contract, selected_device,
                    fold_conv_bn, fold_max_error):
        del config, probability, split, protocol, phase, output
        observed["device"] = args.device
        observed["selected_device"] = str(selected_device)
        observed["algorithm"] = args.algorithm
        observed["contract"] = observed_contract
        observed["fold_conv_bn"] = fold_conv_bn
        observed["fold_max_error"] = fold_max_error
        return {"hard_deployment_manifest": str(tmp_path / "hard.json")}

    monkeypatch.setattr(
        "scripts.run_nyu_qdrop_reconstruction.run_contract_reconstruction",
        reconstruct,
    )
    qdrop_config = SimpleNamespace(
        reconstruction=SimpleNamespace(steps=20000),
        formal=SimpleNamespace(evaluation_seed=20260812),
        precision=lambda name: SimpleNamespace(
            name=name, weight_bits=6, activation_bits=6),
    )
    executor = runner.ProductionMethodExecutor(
        qdrop_args=SimpleNamespace(device=device),
        qdrop_config=qdrop_config,
        qdrop_split=object(),
        qdrop_protocol={},
        rtn_settings=SimpleNamespace(
            fold_conv_bn=False, fold_max_error=0.05),
        selected_device=device,
    )
    runtime = SimpleNamespace(
        model_name=model_name, device=torch.device(device))
    observed_contract = SimpleNamespace(model_name=model_name)

    result = executor(
        runtime, object(), observed_contract, object(), method,
        tmp_path / method,
        {"weight_bits": 6, "activation_bits": 6, "steps": 20000},
        "calibration-sha",
        "evaluation-sha",
    )

    assert result == tmp_path / "hard.json"
    assert observed == {
        "device": device,
        "selected_device": device,
        "algorithm": algorithm,
        "contract": observed_contract,
        "fold_conv_bn": False,
        "fold_max_error": 0.05,
    }


@pytest.mark.parametrize(
    ("fold_max_error", "clip_factors"),
    (
        (math.nan, (1.0,)),
        (math.inf, (1.0,)),
        (0.05, (math.nan,)),
        (0.05, (math.inf,)),
    ),
)
def test_selected_cli_rejects_nonfinite_rtn_controls(
        monkeypatch, tmp_path, fold_max_error, clip_factors):
    assignment_path = tmp_path / "assignment.json"
    assignment_path.write_text("{}", encoding="utf-8")
    model_config = SimpleNamespace(model="nlspn", device="cuda:1")
    selected = SimpleNamespace(models=(model_config,))
    monkeypatch.setattr(
        runner, "load_selected_quantization_config", lambda path: selected)
    arguments = [
        "--config", str(tmp_path / "selected.json"),
        "--model", "nlspn",
        "--device", "cuda:1",
        "--qdrop-config", str(tmp_path / "qdrop.json"),
        "--calibration-indices", str(tmp_path / "indices.json"),
        "--evaluation-protocol", str(tmp_path / "evaluation.json"),
        "--p3-t3-assignment", str(assignment_path),
        "--output", str(tmp_path / "matrix"),
        "--fold-conv-bn",
        "--fold-max-error", str(fold_max_error),
        "--joint-clip-factors",
    ]
    arguments.extend(str(value) for value in clip_factors)
    arguments.extend([
        "--joint-search-rounds", "1",
        "--joint-cache-sample-limit", "1",
        "--joint-cache-byte-limit", "1",
    ])

    with pytest.raises(ValueError, match="calibration values"):
        runner.run_cli(arguments)
