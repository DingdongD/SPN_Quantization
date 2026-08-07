import torch
import torch.nn as nn

from spn_quant.adaptive_rounding import (
    AdaptiveRoundingConfig,
    AdaptiveRoundingController,
)
from spn_quant.deployment_contract import (
    StrictContractInstrumentor,
    build_deployment_contract,
    dequantize_weight_contract,
    export_rounding_contracts,
    load_deployment_contract,
    save_deployment_contract,
    tensor_sha256,
    validate_graph_preparation,
)


class _Quantizer(object):
    def __init__(self, scale):
        self.scale = float(scale)


class _FakeInstrumentor(object):
    def __init__(self, model):
        self.model = model
        self.modules = {"conv": model.conv}
        self.groups = {"conv": "encoder"}
        self.original_weights = {
            "conv": model.conv.weight.detach().cpu().clone(),
        }
        self.original_biases = {
            "conv": model.conv.bias.detach().cpu().clone(),
        }
        self.quantizers = {}
        self.relu_quantizers = {}
        self.stats = {}
        self.weight_scales = {}
        self.handles = []
        self.observers = {}

    def _restore_parameters(self):
        with torch.no_grad():
            self.model.conv.weight.copy_(
                self.original_weights["conv"])
            self.model.conv.bias.copy_(
                self.original_biases["conv"])

    def configure(self, w_bits, a_bits, enabled_groups, **kwargs):
        del w_bits, a_bits, enabled_groups
        self._restore_parameters()
        self.quantizers[("conv", "input")] = _Quantizer(0.25)
        with torch.no_grad():
            self.model.conv.weight.zero_()
            if kwargs["quantize_bias"]:
                self.model.conv.bias.zero_()

    def manifest(self):
        return []

    def metadata(self):
        return {}


class _ConvModel(nn.Module):
    def __init__(self):
        super(_ConvModel, self).__init__()
        self.conv = nn.Conv2d(2, 3, kernel_size=1, bias=True)

    def forward(self, tensor):
        return self.conv(tensor)


def test_conv_contract_round_trip_matches_hard_rounding():
    torch.manual_seed(1)
    module = nn.Conv2d(3, 4, kernel_size=3, bias=True)
    controller = AdaptiveRoundingController(
        module, AdaptiveRoundingConfig(bits=4))
    controller.install([""])

    entry = export_rounding_contracts(
        controller, prefix="conv")["conv"]
    dequantized = dequantize_weight_contract(module, entry)

    controller.set_soft_targets(False)
    torch.testing.assert_close(
        module.weight.detach().cpu(), dequantized)
    assert tensor_sha256(dequantized) == entry["dequantized_sha256"]


def test_grouped_conv_transpose_contract_uses_output_channel_scales():
    torch.manual_seed(2)
    module = nn.ConvTranspose2d(
        4, 6, kernel_size=3, groups=2, bias=False)
    controller = AdaptiveRoundingController(
        module, AdaptiveRoundingConfig(bits=4))
    controller.install([""])

    entry = export_rounding_contracts(
        controller, prefix="up")["up"]
    dequantized = dequantize_weight_contract(module, entry)

    assert entry["output_scale"].numel() == 6
    controller.set_soft_targets(False)
    torch.testing.assert_close(
        module.weight.detach().cpu(), dequantized)


def test_contract_bundle_checks_graph_and_source_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"strict checkpoint")
    graph = {
        "fold": 1,
        "folded_pairs": [{"conv": "conv", "bn": "bn"}],
        "unfolded_fanout_pairs": [],
        "unfolded_conv_bn_pairs": [],
    }
    payload = build_deployment_contract(
        source_checkpoint=checkpoint,
        graph_contract=graph,
        weight_contracts={"conv": {"dummy": 1}},
        method="adaround_strict",
        targets=["conv"])
    path = save_deployment_contract(
        tmp_path / "contract.pt", payload)
    loaded = load_deployment_contract(path)

    validate_graph_preparation({
        "folded_pairs": [{"conv": "conv", "bn": "bn"}],
        "unfolded_fanout_pairs": [],
        "unfolded_conv_bn_pairs": [],
    }, loaded["graph_contract"])
    assert loaded["source_checkpoint_sha256"] == (
        payload["source_checkpoint_sha256"])


def test_contract_instrumentor_replays_codes_after_base_rtn():
    torch.manual_seed(7)
    source = _ConvModel()
    controller = AdaptiveRoundingController(
        source, AdaptiveRoundingConfig(bits=4))
    controller.install(["conv"])
    entries = export_rounding_contracts(controller)

    target = _ConvModel()
    with torch.no_grad():
        target.conv.weight.copy_(
            source.conv.parametrizations.weight.original.detach())
        target.conv.bias.copy_(source.conv.bias.detach())
    proxy = StrictContractInstrumentor(
        _FakeInstrumentor(target), {
            "format_version": 1,
            "weight_contracts": entries,
        })

    proxy.configure(
        4, 8, {"encoder"}, activation_mode="uniform",
        quantize_bias=True)
    expected = dequantize_weight_contract(
        target.conv, entries["conv"])

    torch.testing.assert_close(
        target.conv.weight.detach().cpu(), expected)
    assert proxy.metadata()["weight_execution"] == (
        "exact_integer_code_contract")


def test_contract_instrumentor_manifest_serializes_scale_shape():
    torch.manual_seed(9)
    source = _ConvModel()
    controller = AdaptiveRoundingController(
        source, AdaptiveRoundingConfig(bits=4))
    controller.install(["conv"])
    entries = export_rounding_contracts(controller)

    target = _ConvModel()
    with torch.no_grad():
        target.conv.weight.copy_(
            source.conv.parametrizations.weight.original.detach())
        target.conv.bias.copy_(source.conv.bias.detach())
    proxy = StrictContractInstrumentor(
        _FakeInstrumentor(target), {
            "format_version": 1,
            "weight_contracts": entries,
        })
    proxy.configure(
        4, 8, {"encoder"}, activation_mode="uniform",
        quantize_bias=True)

    rows = proxy.manifest()

    assert rows[0]["scale"] == "tensor:(3, 1, 1, 1)"


def test_weight_only_contract_preserves_fp32_bias_and_e2m1_activation():
    torch.manual_seed(11)
    source = _ConvModel()
    controller = AdaptiveRoundingController(
        source, AdaptiveRoundingConfig(bits=4))
    controller.install(["conv"])
    entries = export_rounding_contracts(controller)

    target = _ConvModel()
    with torch.no_grad():
        target.conv.weight.copy_(
            source.conv.parametrizations.weight.original.detach())
        target.conv.bias.copy_(source.conv.bias.detach())
    expected_bias = target.conv.bias.detach().clone()
    proxy = StrictContractInstrumentor(
        _FakeInstrumentor(target), {
            "format_version": 1,
            "weight_contracts": entries,
        })

    proxy.configure(
        4, 4, {"encoder"}, activation_mode="e2m1",
        quantize_bias=False)

    torch.testing.assert_close(target.conv.bias, expected_bias)
    assert ("conv", "bias") not in proxy.instrumentor.stats
