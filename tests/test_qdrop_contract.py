import copy

import pytest
import torch
import torch.nn as nn

import spn_quant

from spn_quant.adaptive_rounding import (
    AdaptiveRoundingConfig,
    AdaptiveRoundingController,
)
from spn_quant.deployment_contract import (
    dequantize_weight_contract,
    export_rounding_contracts,
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


class UniformQuantizer(object):
    def __init__(self, scale):
        self.scale = float(scale)


class Observer(object):
    def __init__(self):
        self.samples = 4


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
        self.lognp_quantizers = {}
        self.lognp_relu_quantizers = {}
        self.stats = {}
        self.weight_scales = {}
        self.handles = []
        self.observers = {("conv", "input"): Observer()}
        self.relu_observers = {}
        self.activation_mode = "uniform"
        self.mode = "bypass"
        self.frozen = True

    def _restore_parameters(self):
        with torch.no_grad():
            self.model.conv.weight.copy_(self.original_weights["conv"])
            self.model.conv.bias.copy_(self.original_biases["conv"])

    def configure(self, w_bits, a_bits, enabled_groups, **kwargs):
        del w_bits, a_bits, enabled_groups
        self._restore_parameters()
        self.quantizers = {("conv", "input"): UniformQuantizer(0.5)}
        self.relu_quantizers = {}
        self.activation_mode = kwargs["activation_mode"]
        self.mode = "quantize"
        with torch.no_grad():
            self.model.conv.weight.zero_()
            if kwargs["quantize_bias"]:
                self.model.conv.bias.zero_()

    def manifest(self):
        return []

    def metadata(self):
        return {"activation_mode": self.activation_mode}

    def observe(self, activation_mode="uniform"):
        self.activation_mode = activation_mode

    def freeze(self):
        self.frozen = True

    def disable(self):
        self.mode = "bypass"

    def close(self):
        pass


class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv = nn.Conv2d(2, 3, 1, bias=True)

    def forward(self, value):
        return self.conv(value)


def make_contract(tmp_path):
    torch.manual_seed(43)
    source = ConvModel()
    rounding = AdaptiveRoundingController(
        source, AdaptiveRoundingConfig(bits=4))
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
        bits=4,
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


def test_qdrop_contract_round_trip_preserves_exact_w4_a4_fields(tmp_path):
    _, plan, payload = make_contract(tmp_path)
    path = save_qdrop_contract(tmp_path / "qdrop.pt", payload)

    loaded = load_qdrop_contract(path)

    assert loaded["format_version"] == 2
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
    assert activation["fingerprint"] == payload["activation_contracts"][
        "activation::conv::input"]["fingerprint"]
    assert ExactActivationQuantizer.from_contract(activation).phase == "frozen"


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
        "bundle_fingerprint",
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
            "activation::conv::input"]["scale"] *= 2.0
    elif mutation == "activation_owner":
        tampered["target_plan"]["activation_sites"][0][
            "owner_name"] = "other"
    else:
        tampered["fingerprint"] = "0" * 64
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
        activation_mode="uniform",
        activation_overrides={},
        activation_bit_overrides={},
        activation_format_overrides={},
        smooth_channel_maxima={},
        weight_clip_ratio=1.0,
        quantize_bias=True,
    )

    quantizer = base.quantizers[("conv", "input")]
    assert isinstance(quantizer, ExactActivationQuantizer)
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
