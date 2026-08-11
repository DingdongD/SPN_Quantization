import math

import pytest
import torch
import torch.nn as nn

import spn_quant.qdrop_reconstruction as qdrop_reconstruction

from scripts.hardware_aligned_quantization import HardwareAlignedInstrumentor
from spn_quant.adaptive_rounding import AdaptiveRoundingConfig
from spn_quant.qdrop_edges import QDropActivationBank
from spn_quant.qdrop_reconstruction import (
    QDropBlockReconstructor,
    QDropCalibrationRecord,
    QDropOptimizerConfig,
    QDropReconstructionError,
    cache_storage_device,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_mix_inputs_uses_the_tensor_and_generator_cuda_device():
    quantized = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
    full_precision = torch.tensor([[10.0, 20.0, 30.0]], device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(5)

    mixed = mix_qdrop_inputs(
        quantized, full_precision, 0.5, generator)

    assert mixed.device.type == "cuda"
    assert torch.isfinite(mixed).all()


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
        cache_cuda_byte_limit=1024,
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


def test_cuda_cache_uses_explicit_capacity_limit():
    compute_device = torch.device("cuda:2")

    assert cache_storage_device(
        total_bytes=1024,
        compute_device=compute_device,
        cuda_byte_limit=1024,
    ) == compute_device
    assert cache_storage_device(
        total_bytes=1025,
        compute_device=compute_device,
        cuda_byte_limit=1024,
    ) == torch.device("cpu")
    assert cache_storage_device(
        total_bytes=1025,
        compute_device=torch.device("cpu"),
        cuda_byte_limit=1024,
    ) == torch.device("cpu")


def make_reconstruction_fixture(with_activation=True):
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
        activation_sites=(site,) if with_activation else (),
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


def test_input_excluded_first_block_reconstructs_weights_without_a4_site():
    model, bank, records = make_reconstruction_fixture(with_activation=False)
    config = QDropOptimizerConfig(
        steps=1,
        batch_size=4,
        cache_cuda_byte_limit=1024,
        weight_learning_rate=1.0e-12,
        activation_learning_rate=4.0e-5,
        round_loss_weight=0.0,
        warmup_fraction=0.2,
        beta_start=20.0,
        beta_end=2.0,
        loss_power=2.0,
        quant_probability=0.5,
        seed=67,
    )
    reconstructor = QDropBlockReconstructor(
        block=model.block,
        target="block",
        activation_bank=bank,
        weight_config=AdaptiveRoundingConfig(bits=4),
        optimizer_config=config,
        contract_prefix="block",
    )

    result = reconstructor.fit(records)

    assert result.activation_contracts == {}
    assert set(result.weight_contracts) == {
        "block.linear1", "block.linear2"}


def test_reconstruction_stacks_calibration_records_once(monkeypatch):
    model, bank, records = make_reconstruction_fixture()
    reconstructor = QDropBlockReconstructor(
        block=model.block,
        target="block",
        activation_bank=bank,
        weight_config=AdaptiveRoundingConfig(bits=4),
        optimizer_config=make_optimizer_config(steps=2),
        contract_prefix="block",
    )
    original = qdrop_reconstruction._stack_nested
    calls = []

    def counted(values):
        calls.append(len(values))
        return original(values)

    monkeypatch.setattr(qdrop_reconstruction, "_stack_nested", counted)

    reconstructor.fit(records)

    assert calls == [8, 8, 8]
    assert records == []


def test_oversized_cuda_cache_keeps_records_segmented_without_stacking(
        monkeypatch):
    model, bank, records = make_reconstruction_fixture()
    reconstructor = QDropBlockReconstructor(
        block=model.block,
        target="block",
        activation_bank=bank,
        weight_config=AdaptiveRoundingConfig(bits=4),
        optimizer_config=make_optimizer_config(steps=1),
        contract_prefix="block",
    )
    reconstructor.config = QDropOptimizerConfig(
        steps=1,
        batch_size=4,
        cache_cuda_byte_limit=1,
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
    monkeypatch.setattr(
        reconstructor, "_device", lambda module: torch.device("cuda"))
    calls = []
    original = qdrop_reconstruction._stack_nested

    def counted(values):
        calls.append(len(values))
        return original(values)

    monkeypatch.setattr(qdrop_reconstruction, "_stack_nested", counted)

    cache = reconstructor._cache(records)

    assert calls == []
    assert cache.storage_device == torch.device("cpu")
    assert len(cache.records) == 8
    assert records == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_segmented_cache_batches_and_evaluates_on_cuda():
    model, bank, records = make_reconstruction_fixture()
    model.cuda()
    reconstructor = QDropBlockReconstructor(
        block=model.block,
        target="block",
        activation_bank=bank,
        weight_config=AdaptiveRoundingConfig(bits=4),
        optimizer_config=QDropOptimizerConfig(
            steps=1,
            batch_size=4,
            cache_cuda_byte_limit=1,
            weight_learning_rate=1.0e-3,
            activation_learning_rate=4.0e-5,
            round_loss_weight=1.0e-4,
            warmup_fraction=0.2,
            beta_start=20.0,
            beta_end=2.0,
            loss_power=2.0,
            quant_probability=0.5,
            seed=31,
        ),
        contract_prefix="block",
    )

    cache = reconstructor._cache(records)
    generator = torch.Generator(device="cuda").manual_seed(31)
    inputs, reference = reconstructor._batch(cache, generator)

    assert cache.segmented
    assert inputs[0].device.type == "cuda"
    assert reference.device.type == "cuda"
    assert math.isfinite(reconstructor._evaluate(cache))
