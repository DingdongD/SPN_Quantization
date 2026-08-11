import pytest
import torch
import torch.nn as nn

from scripts.hardware_aligned_quantization import HardwareAlignedInstrumentor
from spn_quant.adaptive_rounding import AdaptiveRoundingConfig
from spn_quant.qdrop_edges import QDropActivationBank
from spn_quant.qdrop_reconstruction import (
    QDropBlockReconstructor,
    QDropCalibrationRecord,
    QDropOptimizerConfig,
    QDropReconstructionError,
    mix_qdrop_inputs,
)
from spn_quant.qdrop_targets import (
    EXCLUDED_PROPAGATION_SITES,
    QDropActivationSite,
    QDropTargetPlan,
)


def test_mix_inputs_matches_official_elementwise_semantics():
    quantized = torch.tensor([[1.0, 2.0, 3.0]])
    full_precision = torch.tensor([[10.0, 20.0, 30.0]])
    generator = torch.Generator().manual_seed(5)

    mixed = mix_qdrop_inputs(
        quantized, full_precision, 0.5, generator)
    expected_mask = torch.rand(
        quantized.shape,
        generator=torch.Generator().manual_seed(5)) < 0.5

    torch.testing.assert_close(
        mixed, torch.where(expected_mask, quantized, full_precision))


def test_mix_inputs_preserves_nested_structure_and_probability_endpoints():
    quantized = (
        torch.tensor([[1.0, 2.0]]),
        [torch.tensor([[3.0]]), {"scale": torch.tensor([[4.0]])}],
        "constant",
    )
    full_precision = (
        torch.tensor([[10.0, 20.0]]),
        [torch.tensor([[30.0]]), {"scale": torch.tensor([[40.0]])}],
        "constant",
    )

    all_fp = mix_qdrop_inputs(
        quantized, full_precision, 0.0,
        torch.Generator().manual_seed(7))
    all_quantized = mix_qdrop_inputs(
        quantized, full_precision, 1.0,
        torch.Generator().manual_seed(7))

    torch.testing.assert_close(all_fp[0], full_precision[0])
    torch.testing.assert_close(all_fp[1][1]["scale"],
                               full_precision[1][1]["scale"])
    torch.testing.assert_close(all_quantized[0], quantized[0])
    torch.testing.assert_close(all_quantized[1][0], quantized[1][0])
    assert all_quantized[2] == "constant"


@pytest.mark.parametrize(
    ("quantized", "full_precision", "message"),
    (
        (torch.ones(1, 2), torch.ones(1, 3), "shape"),
        ({"left": torch.ones(1)}, {"right": torch.ones(1)}, "keys"),
        ((torch.ones(1),), [torch.ones(1)], "structure"),
        (("left",), ("right",), "non-tensor"),
    ),
)
def test_mix_inputs_rejects_mismatched_records(
        quantized, full_precision, message):
    with pytest.raises((TypeError, ValueError), match=message):
        mix_qdrop_inputs(
            quantized, full_precision, 0.5,
            torch.Generator().manual_seed(11))


class TinyBlock(nn.Module):
    def __init__(self):
        super(TinyBlock, self).__init__()
        self.linear1 = nn.Linear(4, 4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4, 2)

    def forward(self, value):
        return self.linear2(self.relu(self.linear1(value)))


class TinyModel(nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        self.block = TinyBlock()

    def forward(self, value):
        return self.block(value)


def make_optimizer_config(steps=12):
    return QDropOptimizerConfig(
        steps=steps,
        batch_size=4,
        weight_learning_rate=1.0e-3,
        activation_learning_rate=4.0e-5,
        round_loss_weight=1.0e-4,
        warmup_fraction=0.2,
        beta_start=20.0,
        beta_end=2.0,
        loss_power=2.0,
        quant_probability=0.5,
        seed=31,
    )


def make_reconstruction_fixture():
    torch.manual_seed(37)
    model = TinyModel().eval()
    values = torch.randn(8, 4)
    with torch.no_grad():
        references = model(values).detach().clone()
    original_state = dict(
        (name, value.detach().clone())
        for name, value in model.state_dict().items())
    instrumentor = HardwareAlignedInstrumentor(
        model,
        group_fn=lambda name, module: "encoder",
        fused_relu_producers={"block.relu#0": "block.linear1"},
    )
    instrumentor.observe()
    with torch.no_grad():
        model(values)
    instrumentor.freeze()
    instrumentor.configure(
        w_bits=4,
        a_bits=4,
        enabled_groups={"encoder"},
        activation_mode="uniform",
    )
    site = QDropActivationSite(
        site="activation::block.linear2::input",
        owner_name="block",
        owner_kind="module_input",
        role="module_input",
        signed=False,
        symmetric=False,
    )
    plan = QDropTargetPlan(
        model="cspn",
        blocks=("block",),
        activation_sites=(site,),
        excluded_sites=EXCLUDED_PROPAGATION_SITES,
    )
    bank = QDropActivationBank(
        plan=plan,
        instrumentor=instrumentor,
        bits=4,
        scale_minimum=1.0e-8,
        seed=41,
    )
    bank.initialize()
    model.load_state_dict(original_state)
    records = [
        QDropCalibrationRecord(
            quantized_inputs=(values[index:index + 1].clone(),),
            full_precision_inputs=(values[index:index + 1].clone(),),
            reference=references[index:index + 1].clone(),
        )
        for index in range(values.shape[0])
    ]
    return model, bank, records


def test_joint_reconstruction_hardens_weights_and_activation_contracts():
    model, bank, records = make_reconstruction_fixture()
    original_biases = {
        "linear1": model.block.linear1.bias.detach().clone(),
        "linear2": model.block.linear2.bias.detach().clone(),
    }
    reconstructor = QDropBlockReconstructor(
        block=model.block,
        target="block",
        activation_bank=bank,
        weight_config=AdaptiveRoundingConfig(bits=4),
        optimizer_config=make_optimizer_config(),
        contract_prefix="block",
    )

    result = reconstructor.fit(records)

    assert result.after_loss <= result.before_loss
    assert set(result.weight_contracts) == {
        "block.linear1", "block.linear2"}
    assert set(result.activation_contracts) == {
        "activation::block.linear2::input"}
    torch.testing.assert_close(model.block.linear1.bias,
                               original_biases["linear1"], rtol=0, atol=0)
    torch.testing.assert_close(model.block.linear2.bias,
                               original_biases["linear2"], rtol=0, atol=0)
    assert not any("parametrizations.weight" in name
                   for name, _ in model.block.named_parameters())
    assert bank.quantizers[
        "activation::block.linear2::input"].phase == "frozen"


def test_hard_solution_worse_than_initial_raises_without_export():
    class ControlledReconstructor(QDropBlockReconstructor):
        def __init__(self, *args, **kwargs):
            super(ControlledReconstructor, self).__init__(*args, **kwargs)
            self.evaluations = iter((1.0, 2.0))

        def _evaluate(self, records):
            del records
            return next(self.evaluations)

    model, bank, records = make_reconstruction_fixture()
    reconstructor = ControlledReconstructor(
        block=model.block,
        target="block",
        activation_bank=bank,
        weight_config=AdaptiveRoundingConfig(bits=4),
        optimizer_config=make_optimizer_config(steps=1),
        contract_prefix="block",
    )

    with pytest.raises(QDropReconstructionError, match="worse"):
        reconstructor.fit(records)
    assert bank.contracts() == {}
