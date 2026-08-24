import torch

from spn_quant.qat.hawq import HAWQActivationQuantizer
from spn_quant.qat.quantizers import PerOutputChannelWeightFakeQuantizer


def test_hawq_activation_updates_then_freezes_running_range():
    quantizer = HAWQActivationQuantizer(
        bits=4, unsigned=False, range_momentum=0.5)
    quantizer.train()
    quantizer(torch.tensor([-2.0, 4.0]))
    quantizer(torch.tensor([-4.0, 2.0]))
    observed = (quantizer.minimum.clone(), quantizer.maximum.clone())

    quantizer.freeze_range()
    quantizer(torch.tensor([-100.0, 100.0]))

    torch.testing.assert_close(observed[0], torch.tensor([-3.0]))
    torch.testing.assert_close(observed[1], torch.tensor([3.0]))
    assert torch.equal(quantizer.minimum, observed[0])
    assert torch.equal(quantizer.maximum, observed[1])


def test_hawq_affine_codes_include_zero_point():
    quantizer = HAWQActivationQuantizer(4, False, 0.5)
    quantizer.initialize_range(torch.tensor([-1.0, 3.0]))

    _, codes = quantizer.quantize_with_codes(torch.tensor([0.0]))

    assert 0 <= int(codes.item()) <= 15
    assert int(quantizer.zero_point.item()) != 0


def test_hawq_unsigned_range_starts_at_zero():
    quantizer = HAWQActivationQuantizer(6, True, 0.9)
    quantizer.initialize_range(torch.tensor([0.25, 4.0]))

    assert float(quantizer.minimum.item()) == 0.0
    assert float(quantizer.maximum.item()) == 4.0


def test_hawq_state_reload_preserves_frozen_output():
    source = HAWQActivationQuantizer(6, False, 0.95)
    source.initialize_range(torch.tensor([-2.0, 5.0]))
    source.freeze_range()
    target = HAWQActivationQuantizer(6, False, 0.95)
    target.load_state_dict(source.state_dict())
    current = torch.linspace(-3.0, 6.0, 100)

    torch.testing.assert_close(source(current), target(current))
    assert not target.running_range


def test_existing_weight_fake_quantizer_accepts_w6():
    quantizer = PerOutputChannelWeightFakeQuantizer(6, 0)

    output = quantizer(torch.randn(3, 2, 1, 1))

    assert output.shape == (3, 2, 1, 1)
