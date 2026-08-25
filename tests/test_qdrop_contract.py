import copy
import json
import math

import pytest
import torch
import torch.nn as nn

import spn_quant

from scripts.run_nyu_rtn_quantization import (
    build_propagation_configurations,
    instrumentor_options,
)
from spn_quant.adaptive_rounding import (
    AdaptiveRoundingConfig,
    AdaptiveRoundingController,
)
from spn_quant.deployment_contract import (
    dequantize_weight_contract,
    export_rounding_contracts,
    tensor_sha256,
)
from spn_quant.qdrop_activation import (
    ExactActivationQuantizer,
    QDropActivationQuantizer,
)
from spn_quant.qdrop_contract import (
    QDropContractInstrumentor,
    build_qdrop_contract,
    load_qdrop_contract,
    save_qdrop_contract,
)
from spn_quant.qdrop_targets import (
    EXCLUDED_PROPAGATION_SITES,
    QDropActivationSite,
    QDropTargetPlan,
)
from scripts.run_nyu_qdrop_reconstruction import (
    build_contract_reconstruction_plan,
    write_hard_reconstruction_artifacts,
)
from spn_quant.model_contracts import (
    QuantizationBlock,
    QuantizationModelContract,
)


class UniformQuantizer(object):
    def __init__(self, scale):
        self.scale = float(scale)


class Observer(object):
    def __init__(self):
        self.samples = 4
        self.observed = False


class FakeInstrumentor(object):
    def __init__(self, model):
        self.model = model
        self.modules = {"conv": model.conv}
        self.groups = {"conv": "encoder"}
        self.original_weights = {
            "conv": model.conv.weight.detach().cpu().clone()}
        self.original_biases = {
            "conv": model.conv.bias.detach().cpu().clone()}
        self.quantizers = {}
        self.relu_quantizers = {}
        self.stats = {}
        self.weight_scales = {}
        self.handles = []
        self.observers = {("conv", "input"): Observer()}
        self.relu_observers = {}
        self.mode = "bypass"
        self.frozen = True

    def _restore_parameters(self):
        with torch.no_grad():
            self.model.conv.weight.copy_(self.original_weights["conv"])
            self.model.conv.bias.copy_(self.original_biases["conv"])

    def configure(self, w_bits, a_bits, enabled_groups, **kwargs):
        del w_bits, a_bits, enabled_groups
        self._restore_parameters()
        self.quantizers = {
            ("conv", "input"): UniformQuantizer(0.5),
            ("conv", "output"): UniformQuantizer(0.25),
        }
        self.relu_quantizers = {"relu#0": UniformQuantizer(0.25)}
        self.mode = "quantize"
        with torch.no_grad():
            self.model.conv.weight.zero_()
            if kwargs["quantize_bias"]:
                self.model.conv.bias.zero_()

    def manifest(self):
        return []

    def metadata(self):
        return {"activation_mode": "uniform"}

    def statistics(self):
        return []

    def observe(self):
        self.mode = "observe"

    def freeze(self):
        self.frozen = True

    def disable(self):
        self.mode = "bypass"

    def close(self):
        pass

    def _relu_owner(self, key):
        assert key == "relu#0"
        return "conv", "encoder"


class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv = nn.Conv2d(2, 3, 1, bias=True)

    def forward(self, value):
        return self.conv(value)


class TransposeModel(nn.Module):
    def __init__(self):
        super(TransposeModel, self).__init__()
        self.conv = nn.ConvTranspose2d(2, 3, 1, bias=True)

    def forward(self, value):
        return self.conv(value)


class TwoConvModel(nn.Module):
    def __init__(self):
        super(TwoConvModel, self).__init__()
        self.conv1 = nn.Conv2d(2, 2, 1, bias=True)
        self.conv2 = nn.Conv2d(2, 3, 1, bias=True)

    def forward(self, value):
        return self.conv2(self.conv1(value))


class TwoConvInstrumentor(FakeInstrumentor):
    def __init__(self, model):
        self.model = model
        self.modules = {
            "conv1": model.conv1,
            "conv2": model.conv2,
        }
        self.groups = {"conv1": "encoder", "conv2": "encoder"}
        self.original_weights = {
            name: module.weight.detach().cpu().clone()
            for name, module in self.modules.items()
        }
        self.original_biases = {
            name: module.bias.detach().cpu().clone()
            for name, module in self.modules.items()
        }
        self.quantizers = {}
        self.relu_quantizers = {}
        self.stats = {}
        self.weight_scales = {}
        self.handles = []
        self.observers = {
            (name, "input"): Observer() for name in self.modules}
        self.relu_observers = {}
        self.mode = "bypass"
        self.frozen = True

    def _restore_parameters(self):
        with torch.no_grad():
            for name, module in self.modules.items():
                module.weight.copy_(self.original_weights[name])
                module.bias.copy_(self.original_biases[name])

    def configure(self, w_bits, a_bits, enabled_groups, **kwargs):
        del w_bits, a_bits, enabled_groups
        self._restore_parameters()
        self.quantizers = {
            (name, "input"): UniformQuantizer(0.5)
            for name in self.modules
        }
        self.mode = "quantize"
        with torch.no_grad():
            for module in self.modules.values():
                module.weight.zero_()


def test_reconstruction_excludes_protected_contract_roles():
    model = TwoConvModel()
    plan = QDropTargetPlan(
        model="cspn",
        blocks=("conv1", "conv2"),
        activation_sites=(
            QDropActivationSite(
                site="activation::conv1::input",
                owner_name="conv1",
                owner_kind="module_input",
                role="module_input",
                signed=True,
                symmetric=True,
            ),
            QDropActivationSite(
                site="activation::conv2::input",
                owner_name="conv2",
                owner_kind="module_input",
                role="module_input",
                signed=True,
                symmetric=True,
            ),
        ),
        excluded_sites=EXCLUDED_PROPAGATION_SITES,
    )
    model_contract = QuantizationModelContract(
        model_name="cspn",
        blocks=(QuantizationBlock(
            "conv1", ("conv1",),
            (("activation::conv1::input", "module_input"),)),),
        prefix_groups=(("conv1",),),
        tail_groups=(("conv1",),),
        protected_roles=("propagation_state",),
        attention_edges=(),
        concat_edges=(),
        protected_modules=("conv2",),
        module_roles=(("conv2", "propagation_state"),),
    )

    reconstruction = build_contract_reconstruction_plan(
        model_contract, model, plan)

    protected = set(model_contract.protected_modules)
    assert protected.isdisjoint(reconstruction.module_names)
    assert reconstruction.block_names == ("conv1",)
    assert reconstruction.activation_owners == (
        ("activation::conv1::input", "module_input"),)
    assert reconstruction.weight_names_by_block == (
        ("conv1", ("",)),)


def test_reconstruction_block_uses_explicit_generic_weight_allowlist():
    class Block(nn.Module):
        def __init__(self):
            super(Block, self).__init__()
            self.generic = nn.Conv2d(2, 2, 1)
            self.protected = nn.Conv2d(2, 2, 1)

        def forward(self, value):
            return self.generic(value) + self.protected(value)

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.block = Block()

        def forward(self, value):
            return self.block(value)

    model = Model()
    resolved = QDropTargetPlan(
        model="cspn",
        blocks=("block",),
        activation_sites=(QDropActivationSite(
            site="activation::block::input",
            owner_name="block",
            owner_kind="module_input",
            role="module_input",
            signed=True,
            symmetric=True,
        ),),
        excluded_sites=EXCLUDED_PROPAGATION_SITES,
    )
    model_contract = QuantizationModelContract(
        model_name="cspn",
        blocks=(QuantizationBlock(
            "block", ("block.generic",),
            (("activation::block::input", "module_input"),)),),
        prefix_groups=(("block",),),
        tail_groups=(("block",),),
        protected_roles=("propagation_state",),
        attention_edges=(),
        concat_edges=(),
        protected_modules=("block.protected",),
        module_roles=(("block.protected", "propagation_state"),),
    )

    reconstruction = build_contract_reconstruction_plan(
        model_contract, model, resolved)

    assert reconstruction.weight_names_by_block == (
        ("block", ("generic",)),)


def test_hard_reconstruction_artifact_matches_materialized_weight_contract(
        tmp_path):
    model = TwoConvModel()
    target_plan = QDropTargetPlan(
        model="cspn",
        blocks=("conv1",),
        activation_sites=(QDropActivationSite(
            site="activation::conv1::input",
            owner_name="conv1",
            owner_kind="module_input",
            role="module_input",
            signed=True,
            symmetric=True,
        ),),
        excluded_sites=EXCLUDED_PROPAGATION_SITES,
    )
    model_contract = QuantizationModelContract(
        model_name="cspn",
        blocks=(QuantizationBlock(
            "conv1", ("conv1",),
            (("activation::conv1::input", "module_input"),)),),
        prefix_groups=(("conv1",),),
        tail_groups=(("conv1",),),
        protected_roles=("propagation_state",),
        attention_edges=(),
        concat_edges=(),
        protected_modules=("conv2",),
        module_roles=(("conv2", "propagation_state"),),
    )
    reconstruction = build_contract_reconstruction_plan(
        model_contract, model, target_plan)
    deployment_contract = tmp_path / "qdrop_strict_contract.pt"
    torch.save({"strict": 1}, deployment_contract)
    optimization_state = tmp_path / "qdrop_reconstruction_history.json"
    optimization_state.write_text("{}", encoding="utf-8")
    weight_contracts = {
        "conv1": {
            "dequantized_sha256": tensor_sha256(model.conv1.weight),
        },
    }

    manifest_path = write_hard_reconstruction_artifacts(
        output=tmp_path,
        model=model,
        plan=reconstruction,
        selected_method="qdrop_w6a6",
        weight_bits=6,
        activation_bits=6,
        weight_contracts=weight_contracts,
        deployment_contract=deployment_contract,
        optimization_state=optimization_state,
        calibration_identity="calibration-sha",
        evaluation_identity="evaluation-sha",
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(
        payload["hard_weights"], map_location="cpu", weights_only=False)

    assert payload["materialized_hard_weights"] == 1
    assert payload["module_names"] == ["conv1"]
    assert "conv1.weight" in checkpoint["state_dict"]
    assert not any("parametrizations.weight" in name
                   for name in checkpoint["state_dict"])

    weight_contracts["conv1"]["dequantized_sha256"] = "modified"
    with pytest.raises(RuntimeError, match="materialized weight"):
        write_hard_reconstruction_artifacts(
            output=tmp_path / "invalid",
            model=model,
            plan=reconstruction,
            selected_method="qdrop_w6a6",
            weight_bits=6,
            activation_bits=6,
            weight_contracts=weight_contracts,
            deployment_contract=deployment_contract,
            optimization_state=optimization_state,
            calibration_identity="calibration-sha",
            evaluation_identity="evaluation-sha",
        )


def make_contract(tmp_path, bits=4, method="qdrop_strict"):
    torch.manual_seed(43)
    source = ConvModel()
    rounding = AdaptiveRoundingController(
        source, AdaptiveRoundingConfig(bits=bits))
    rounding.install(("conv",))
    weight_contracts = export_rounding_contracts(rounding)
    site = QDropActivationSite(
        site="activation::conv::input",
        owner_name="conv",
        owner_kind="module_input",
        role="module_input",
        signed=True,
        symmetric=True,
    )
    plan = QDropTargetPlan(
        model="cspn",
        blocks=("conv",),
        activation_sites=(site,),
        excluded_sites=EXCLUDED_PROPAGATION_SITES,
    )
    quantizer = QDropActivationQuantizer(
        site=site.site,
        bits=bits,
        signed=True,
        symmetric=True,
        scale_minimum=1.0e-8,
        seed=47,
    )
    quantizer.initialize(torch.tensor((-1.0, 1.0)))
    quantizer.start_reconstruction(1.0)
    quantizer.freeze()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"net": source.state_dict()}, checkpoint)
    payload = build_qdrop_contract(
        method=method,
        source_checkpoint=checkpoint,
        graph_contract={
            "fold": 1,
            "folded_pairs": [],
            "unfolded_fanout_pairs": [],
            "unfolded_conv_bn_pairs": [],
        },
        weight_contracts=weight_contracts,
        activation_contracts={site.site: quantizer.contract()},
        targets=plan,
        metadata={"seed": 47},
    )
    return source, plan, payload


def test_joint_contract_preserves_brecq_method_identity(tmp_path):
    _, _, payload = make_contract(
        tmp_path, bits=6, method="brecq_joint_strict")
    path = save_qdrop_contract(tmp_path / "brecq.pt", payload)

    loaded = load_qdrop_contract(path)

    assert loaded["method"] == "brecq_joint_strict"
    assert loaded["weight_bits"] == 6
    assert loaded["activation_bits"] == 6


def test_joint_contract_rejects_unknown_method(tmp_path):
    with pytest.raises(ValueError, match="method"):
        make_contract(tmp_path, method="rtn_strict")


def test_qdrop_contract_round_trip_preserves_exact_w4_a4_fields(tmp_path):
    _, plan, payload = make_contract(tmp_path)
    path = save_qdrop_contract(tmp_path / "qdrop.pt", payload)

    loaded = load_qdrop_contract(path)

    assert loaded["format_version"] == 3
    assert loaded["method"] == "qdrop_strict"
    assert loaded["weight_bits"] == 4
    assert loaded["activation_bits"] == 4
    assert loaded["activation_policy"] == "exact_semantic_edge_contract"
    assert loaded["target_plan"]["blocks"] == list(plan.blocks)
    assert torch.equal(
        loaded["weight_contracts"]["conv"]["codes"],
        payload["weight_contracts"]["conv"]["codes"])
    activation = loaded["activation_contracts"][
        "activation::conv::input"]
    assert "fingerprint" not in loaded
    assert "fingerprint" not in activation
    assert ExactActivationQuantizer.from_contract(activation).phase == "frozen"


def test_qdrop_contract_round_trip_preserves_exact_w6_a6_fields(tmp_path):
    source, plan, payload = make_contract(tmp_path, bits=6)
    path = save_qdrop_contract(tmp_path / "qdrop_w6a6.pt", payload)

    loaded = load_qdrop_contract(path)

    assert loaded["weight_bits"] == 6
    assert loaded["activation_bits"] == 6
    assert loaded["target_plan"]["blocks"] == list(plan.blocks)
    assert torch.equal(
        loaded["weight_contracts"]["conv"]["codes"],
        payload["weight_contracts"]["conv"]["codes"])
    activation = loaded["activation_contracts"][
        "activation::conv::input"]
    assert ExactActivationQuantizer.from_contract(activation).bits == 6

    target = ConvModel()
    with torch.no_grad():
        target.conv.weight.copy_(
            source.conv.parametrizations.weight.original.detach())
        target.conv.bias.copy_(source.conv.bias.detach())
    base = FakeInstrumentor(target)
    proxy = QDropContractInstrumentor(base, loaded)
    proxy.configure(
        w_bits=6,
        a_bits=6,
        enabled_groups={"encoder"},
        activation_overrides={},
        activation_bit_overrides={},
        quantize_bias=False,
    )

    assert isinstance(
        base.quantizers[("conv", "input")], ExactActivationQuantizer)
    with pytest.raises(ValueError, match="contract W6A6"):
        proxy.configure(
            w_bits=4,
            a_bits=4,
            enabled_groups={"encoder"},
            activation_overrides={},
            activation_bit_overrides={},
            quantize_bias=False,
        )


def test_qdrop_contract_types_are_available_from_stable_package_api():
    assert spn_quant.QDropContractInstrumentor is QDropContractInstrumentor
    assert spn_quant.QDropActivationQuantizer is QDropActivationQuantizer
    assert spn_quant.QDropBlockReconstructor.__name__ == (
        "QDropBlockReconstructor")


@pytest.mark.parametrize(
    "mutation",
    (
        "method",
        "activation_bits",
        "activation_scale",
        "activation_owner",
    ),
)
def test_qdrop_contract_rejects_tampering(tmp_path, mutation):
    _, _, payload = make_contract(tmp_path)
    tampered = copy.deepcopy(payload)
    if mutation == "method":
        tampered["method"] = "brecq_strict"
    elif mutation == "activation_bits":
        tampered["activation_bits"] = 8
    elif mutation == "activation_scale":
        tampered["activation_contracts"][
            "activation::conv::input"]["scale"] = torch.tensor(0.0)
    elif mutation == "activation_owner":
        tampered["target_plan"]["activation_sites"][0][
            "owner_name"] = "other"
    path = tmp_path / (mutation + ".pt")
    torch.save(tampered, path)

    with pytest.raises((KeyError, ValueError, RuntimeError)):
        load_qdrop_contract(path)


def test_contract_instrumentor_replays_exact_a4_and_recomputes_bias(tmp_path):
    source, _, payload = make_contract(tmp_path)
    target = ConvModel()
    original_weight = source.conv.parametrizations.weight.original.detach()
    with torch.no_grad():
        target.conv.weight.copy_(original_weight)
        target.conv.bias.copy_(source.conv.bias.detach())
    original_bias = target.conv.bias.detach().clone()
    base = FakeInstrumentor(target)
    observer_samples = base.observers[("conv", "input")].samples
    proxy = QDropContractInstrumentor(base, payload)

    proxy.configure(
        w_bits=4,
        a_bits=4,
        enabled_groups={"encoder"},
        activation_overrides={},
        activation_bit_overrides={},
        quantize_bias=True,
    )

    quantizer = base.quantizers[("conv", "input")]
    assert isinstance(quantizer, ExactActivationQuantizer)
    assert ("conv", "output") not in base.quantizers
    assert "relu#0" not in base.relu_quantizers
    expected_weight = dequantize_weight_contract(
        target.conv, payload["weight_contracts"]["conv"])
    torch.testing.assert_close(target.conv.weight.detach().cpu(), expected_weight)
    weight_scale = payload["weight_contracts"]["conv"]["output_scale"]
    bias_scale = weight_scale * quantizer.scale
    expected_bias = torch.round(original_bias / bias_scale) * bias_scale
    torch.testing.assert_close(target.conv.bias.detach().cpu(), expected_bias)
    assert base.observers[("conv", "input")].samples == observer_samples
    assert proxy.metadata()["activation_execution"] == (
        "exact_integer_code_contract")
    assert any(row["kind"] == "exact_activation_contract"
               for row in proxy.manifest())


def test_exact_replay_does_not_require_generic_transpose_observation(tmp_path):
    torch.manual_seed(49)
    source = TransposeModel()
    rounding = AdaptiveRoundingController(
        source, AdaptiveRoundingConfig(bits=4))
    rounding.install(("conv",))
    site = QDropActivationSite(
        site="activation::conv::input",
        owner_name="conv",
        owner_kind="module_input",
        role="module_input",
        signed=True,
        symmetric=True,
    )
    plan = QDropTargetPlan(
        model="dyspn",
        blocks=("conv",),
        activation_sites=(site,),
        excluded_sites=EXCLUDED_PROPAGATION_SITES,
    )
    quantizer = QDropActivationQuantizer(
        site=site.site,
        bits=4,
        signed=True,
        symmetric=True,
        scale_minimum=1.0e-8,
        seed=51,
    )
    quantizer.initialize(torch.tensor((-1.0, 1.0)))
    quantizer.start_reconstruction(1.0)
    quantizer.freeze()
    checkpoint = tmp_path / "transpose.pt"
    torch.save({"net": source.state_dict()}, checkpoint)
    payload = build_qdrop_contract(
        method="qdrop_strict",
        source_checkpoint=checkpoint,
        graph_contract={
            "fold": 1,
            "folded_pairs": [],
            "unfolded_fanout_pairs": [],
            "unfolded_conv_bn_pairs": [],
        },
        weight_contracts=export_rounding_contracts(rounding),
        activation_contracts={site.site: quantizer.contract()},
        targets=plan,
        metadata={"seed": 51},
    )
    target = TransposeModel()
    with torch.no_grad():
        target.conv.weight.copy_(
            source.conv.parametrizations.weight.original)
        target.conv.bias.copy_(source.conv.bias)
    base = FakeInstrumentor(target)
    proxy = QDropContractInstrumentor(base, payload)

    proxy.configure(
        w_bits=4,
        a_bits=4,
        enabled_groups={"encoder"},
        activation_overrides={},
        activation_bit_overrides={},
        quantize_bias=True,
    )

    assert isinstance(base.quantizers[("conv", "input")],
                      ExactActivationQuantizer)
    expected_weight = dequantize_weight_contract(
        target.conv, payload["weight_contracts"]["conv"])
    torch.testing.assert_close(target.conv.weight.detach().cpu(), expected_weight)


def test_propagation_runtime_supplies_explicit_qdrop_options(tmp_path):
    source, _, payload = make_contract(tmp_path)
    target = ConvModel()
    with torch.no_grad():
        target.conv.weight.copy_(
            source.conv.parametrizations.weight.original)
        target.conv.bias.copy_(source.conv.bias)
    proxy = QDropContractInstrumentor(FakeInstrumentor(target), payload)
    config = build_propagation_configurations(["encoder"])[2]

    proxy.configure(
        config["w_bits"], config["a_bits"], config["groups"],
        **instrumentor_options(config))



def test_exact_replay_keeps_bias_fp_for_explicitly_unquantized_input(tmp_path):
    torch.manual_seed(53)
    source = TwoConvModel()
    rounding = AdaptiveRoundingController(
        source, AdaptiveRoundingConfig(bits=4))
    rounding.install(("conv1", "conv2"))
    site = QDropActivationSite(
        site="activation::conv2::input",
        owner_name="conv2",
        owner_kind="module_input",
        role="module_input",
        signed=True,
        symmetric=True,
    )
    plan = QDropTargetPlan(
        model="cspn",
        blocks=("conv1", "conv2"),
        activation_sites=(site,),
        excluded_sites=EXCLUDED_PROPAGATION_SITES,
    )
    quantizer = QDropActivationQuantizer(
        site=site.site,
        bits=4,
        signed=True,
        symmetric=True,
        scale_minimum=1.0e-8,
        seed=59,
    )
    quantizer.initialize(torch.tensor((-1.0, 1.0)))
    quantizer.start_reconstruction(1.0)
    quantizer.freeze()
    checkpoint = tmp_path / "two_conv.pt"
    torch.save({"net": source.state_dict()}, checkpoint)
    payload = build_qdrop_contract(
        method="qdrop_strict",
        source_checkpoint=checkpoint,
        graph_contract={
            "fold": 1,
            "folded_pairs": [],
            "unfolded_fanout_pairs": [],
            "unfolded_conv_bn_pairs": [],
        },
        weight_contracts=export_rounding_contracts(rounding),
        activation_contracts={site.site: quantizer.contract()},
        targets=plan,
        metadata={"seed": 59},
    )
    target = TwoConvModel()
    with torch.no_grad():
        for name in ("conv1", "conv2"):
            source_module = dict(source.named_modules())[name]
            target_module = dict(target.named_modules())[name]
            target_module.weight.copy_(
                source_module.parametrizations.weight.original.detach())
            target_module.bias.copy_(source_module.bias.detach())
    original_conv1_bias = target.conv1.bias.detach().clone()
    original_conv2_bias = target.conv2.bias.detach().clone()
    proxy = QDropContractInstrumentor(TwoConvInstrumentor(target), payload)

    proxy.configure(
        w_bits=4,
        a_bits=4,
        enabled_groups={"encoder"},
        activation_overrides={},
        activation_bit_overrides={},
        quantize_bias=True,
    )

    torch.testing.assert_close(target.conv1.bias, original_conv1_bias)
    assert not torch.equal(target.conv2.bias, original_conv2_bias)
    fp_rows = [
        row for row in proxy.manifest()
        if row["kind"] == "explicit_fp_bias"]
    assert [row["module"] for row in fp_rows] == ["conv1"]
    assert fp_rows[0]["reason"] == "no_input_activation_contract"
    assert proxy.metadata()["explicit_fp_bias_sites"] == 1


def test_exact_replay_restores_noncontracted_same_group_weights(tmp_path):
    torch.manual_seed(61)
    source = TwoConvModel()
    rounding = AdaptiveRoundingController(
        source, AdaptiveRoundingConfig(bits=4))
    rounding.install(("conv2",))
    site = QDropActivationSite(
        site="activation::conv2::input",
        owner_name="conv2",
        owner_kind="module_input",
        role="module_input",
        signed=True,
        symmetric=True,
    )
    plan = QDropTargetPlan(
        model="cspn",
        blocks=("conv2",),
        activation_sites=(site,),
        excluded_sites=EXCLUDED_PROPAGATION_SITES,
    )
    quantizer = QDropActivationQuantizer(
        site=site.site,
        bits=4,
        signed=True,
        symmetric=True,
        scale_minimum=1.0e-8,
        seed=67,
    )
    quantizer.initialize(torch.tensor((-1.0, 1.0)))
    quantizer.start_reconstruction(1.0)
    quantizer.freeze()
    checkpoint = tmp_path / "partial.pt"
    torch.save({"net": source.state_dict()}, checkpoint)
    payload = build_qdrop_contract(
        method="qdrop_strict",
        source_checkpoint=checkpoint,
        graph_contract={
            "fold": 1,
            "folded_pairs": [],
            "unfolded_fanout_pairs": [],
            "unfolded_conv_bn_pairs": [],
        },
        weight_contracts=export_rounding_contracts(rounding),
        activation_contracts={site.site: quantizer.contract()},
        targets=plan,
        metadata={"seed": 67},
    )
    target = TwoConvModel()
    with torch.no_grad():
        target.conv1.weight.copy_(source.conv1.weight)
        target.conv1.bias.copy_(source.conv1.bias)
        target.conv2.weight.copy_(
            source.conv2.parametrizations.weight.original)
        target.conv2.bias.copy_(source.conv2.bias)
    original_conv1 = target.conv1.weight.detach().clone()
    proxy = QDropContractInstrumentor(TwoConvInstrumentor(target), payload)

    proxy.configure(
        w_bits=4,
        a_bits=4,
        enabled_groups={"encoder"},
        activation_overrides={},
        activation_bit_overrides={},
        quantize_bias=True,
    )

    torch.testing.assert_close(target.conv1.weight, original_conv1)
    expected_conv2 = dequantize_weight_contract(
        target.conv2, payload["weight_contracts"]["conv2"])
    torch.testing.assert_close(
        target.conv2.weight.detach().cpu(), expected_conv2)


def test_exact_replay_reports_contracted_activation_statistics(tmp_path):
    source, _, payload = make_contract(tmp_path)
    target = ConvModel()
    with torch.no_grad():
        target.conv.weight.copy_(
            source.conv.parametrizations.weight.original)
        target.conv.bias.copy_(source.conv.bias)
    base = FakeInstrumentor(target)
    proxy = QDropContractInstrumentor(base, payload)
    proxy.configure(
        w_bits=4,
        a_bits=4,
        enabled_groups={"encoder"},
        activation_overrides={},
        activation_bit_overrides={},
        quantize_bias=True,
    )

    base.quantizers[("conv", "input")](
        torch.tensor([[[[-1.0, 0.0], [0.5, 1.0]]]]))
    rows = [
        row for row in proxy.statistics()
        if row["kind"] == "exact_activation_contract"]

    assert len(rows) == 1
    assert rows[0]["module"] == "activation::conv::input"
    assert rows[0]["calls"] == 1
    assert rows[0]["numel"] == 4
    assert 0.0 <= rows[0]["zero_code_rate"] <= 1.0
    assert 0.0 <= rows[0]["saturation_rate"] <= 1.0
    assert math.isfinite(rows[0]["sqnr_db"])
