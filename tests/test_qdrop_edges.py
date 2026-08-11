import pytest
import torch
import torch.nn as nn

from scripts.hardware_aligned_quantization import HardwareAlignedInstrumentor
from spn_quant.qdrop_edges import QDropActivationBank
from spn_quant.qdrop_targets import (
    EXCLUDED_PROPAGATION_SITES,
    QDropActivationSite,
    QDropTargetPlan,
)


class ConvReluConv(nn.Module):
    def __init__(self):
        super(ConvReluConv, self).__init__()
        self.conv1 = nn.Conv2d(2, 2, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(2, 1, 1, bias=False)

    def forward(self, value):
        return self.conv2(self.relu(self.conv1(value)))


def make_plan(site):
    return QDropTargetPlan(
        model="cspn",
        blocks=("conv1", "conv2"),
        activation_sites=(site,),
        excluded_sites=EXCLUDED_PROPAGATION_SITES,
    )


def configured_instrumentor(model, value):
    instrumentor = HardwareAlignedInstrumentor(
        model,
        group_fn=lambda name, module: "encoder",
        fused_relu_producers={"relu#0": "conv1"},
    )
    instrumentor.observe()
    with torch.no_grad():
        model(value)
    instrumentor.freeze()
    instrumentor.configure(
        w_bits=4,
        a_bits=4,
        enabled_groups={"encoder"},
        activation_mode="uniform",
    )
    return instrumentor


def test_bank_replaces_one_hardware_boundary_and_restores_it():
    torch.manual_seed(3)
    model = ConvReluConv().eval()
    value = torch.randn(1, 2, 3, 3)
    instrumentor = configured_instrumentor(model, value)
    boundary = ("conv2", "input")
    original = instrumentor.quantizers[boundary]
    site = QDropActivationSite(
        site="activation::conv2::input",
        owner_name="conv2",
        owner_kind="module_input",
        role="module_input",
        signed=False,
        symmetric=False,
    )
    bank = QDropActivationBank(
        plan=make_plan(site),
        instrumentor=instrumentor,
        bits=4,
        scale_minimum=1.0e-8,
        seed=17,
    )

    bank.initialize()
    bank.reconstruct("conv2", quant_probability=1.0)
    assert instrumentor.quantizers[boundary] is bank.quantizers[site.site]
    with torch.no_grad():
        model(value)
    assert bank.quantizers[site.site].statistics()["calls"] == 1
    assert tuple(bank.parameters_for("conv2"))

    bank.freeze_target("conv2")
    bank.disable_randomness()
    assert bank.quantizers[site.site].phase == "frozen"
    assert set(bank.contracts()) == {site.site}
    bank.close()
    assert instrumentor.quantizers[boundary] is original


def test_bank_rejects_unknown_or_already_replaced_boundary():
    model = ConvReluConv().eval()
    value = torch.randn(1, 2, 2, 2)
    instrumentor = configured_instrumentor(model, value)
    site = QDropActivationSite(
        site="activation::missing::input",
        owner_name="conv2",
        owner_kind="module_input",
        role="module_input",
        signed=True,
        symmetric=True,
    )
    bank = QDropActivationBank(
        plan=make_plan(site),
        instrumentor=instrumentor,
        bits=4,
        scale_minimum=1.0e-8,
        seed=19,
    )

    with pytest.raises(KeyError, match="missing QDrop hardware boundary"):
        bank.initialize()


def test_bank_cannot_disable_randomness_before_every_site_is_frozen():
    model = ConvReluConv().eval()
    value = torch.randn(1, 2, 2, 2)
    instrumentor = configured_instrumentor(model, value)
    sites = (
        QDropActivationSite(
            site="activation::conv1::input",
            owner_name="conv1",
            owner_kind="module_input",
            role="module_input",
            signed=True,
            symmetric=True),
        QDropActivationSite(
            site="activation::conv2::input",
            owner_name="conv2",
            owner_kind="module_input",
            role="module_input",
            signed=False,
            symmetric=False),
    )
    plan = QDropTargetPlan(
        model="cspn",
        blocks=("conv1", "conv2"),
        activation_sites=sites,
        excluded_sites=EXCLUDED_PROPAGATION_SITES,
    )
    bank = QDropActivationBank(
        plan=plan,
        instrumentor=instrumentor,
        bits=4,
        scale_minimum=1.0e-8,
        seed=23,
    )
    bank.initialize()
    bank.reconstruct("conv2", quant_probability=1.0)
    bank.freeze_target("conv2")

    with pytest.raises(RuntimeError, match="not frozen"):
        bank.disable_randomness()
