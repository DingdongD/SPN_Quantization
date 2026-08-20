import pytest
import torch
import torch.nn as nn

from spn_quant.activation_boundaries import (
    ActivationBoundaryObserver,
    CSPNActivationBoundaryController,
)
from spn_quant.adapters import ActivationBoundary, ActivationConsumer


class DecoderEntryBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(8, 4, 3, padding=1)
        self.sc_conv1 = nn.Conv2d(8, 4, 1)

    def forward(self, value):
        return self.conv1(value) + self.sc_conv1(value)


class ConcatBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 4, 1)
        self.conv1_1 = nn.Conv2d(8, 4, 1)

    def forward(self, value, signed_skip):
        relu_branch = torch.relu(self.conv1(value))
        return self.conv1_1(torch.cat((relu_branch, signed_skip), dim=1))


class BoundaryToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_entry = DecoderEntryBlock()
        self.concat = ConcatBlock()

    def forward(self, value, signed_skip):
        return self.concat(self.decoder_entry(value), signed_skip)


def toy_boundaries():
    return (
        ActivationBoundary(
            "decoder_entry", "decoder_entry", 0, (
                ActivationConsumer("decoder_entry.conv1", 0, None),
                ActivationConsumer("decoder_entry.sc_conv1", 0, None),
            )),
        ActivationBoundary(
            "layer4_signed_skip", "concat", 1, (
                ActivationConsumer("concat.conv1_1", 4, 4),
            )),
    )


def calibrated_controller():
    model = BoundaryToyModel().eval()
    controller = CSPNActivationBoundaryController(model, toy_boundaries())
    controller.observe()
    model(torch.randn(1, 8, 5, 5), torch.randn(1, 4, 5, 5))
    controller.freeze()
    return model, controller


def test_boundary_observer_reports_tail_and_qdq_metrics():
    observer = ActivationBoundaryObserver(channels=4)
    observer.update(torch.tensor([
        [[[-8.0]], [[-1.0]], [[2.0]], [[4.0]]],
    ]))
    quantizer = observer.quantizer(bits=4, group_size=None)

    row = observer.statistics(quantizer)

    assert {
        "maximum", "p75", "p99", "p99_9", "p99_99", "kurtosis",
        "channel_imbalance", "sqnr", "zero_code_ratio",
        "saturation_ratio",
    } <= set(row)


def test_boundary_controller_uses_per_boundary_bits_groups_and_scales():
    _, controller = calibrated_controller()

    controller.configure_specs(
        bit_widths={"decoder_entry": 8, "layer4_signed_skip": 4},
        group_sizes={"decoder_entry": 4, "layer4_signed_skip": None},
        scale_factors={"decoder_entry": 0.5,
                       "layer4_signed_skip": 1.0},
        quantize=True)

    decoder = controller.active_quantizers["decoder_entry"]
    signed_skip = controller.active_quantizers["layer4_signed_skip"]
    assert decoder.bits == 8
    assert signed_skip.bits == 4
    expected = controller.observers[
        "decoder_entry"].channel_absmax.reshape(2, 4).amax(1)
    torch.testing.assert_close(decoder.scales, expected * 0.5 / 127.0)
    controller.close()


def test_boundary_controller_accepts_exact_group_range_overrides():
    _, controller = calibrated_controller()

    controller.configure_specs_with_ranges(
        bit_widths={"decoder_entry": 4, "layer4_signed_skip": 4},
        group_sizes={"decoder_entry": 4, "layer4_signed_skip": 4},
        scale_factors={"decoder_entry": 1.0,
                       "layer4_signed_skip": 1.0},
        maximum_overrides={
            "decoder_entry": torch.tensor([0.7, 1.4]),
            "layer4_signed_skip": torch.tensor([2.1]),
        },
        quantize=True)

    torch.testing.assert_close(
        controller.active_quantizers["decoder_entry"].scales,
        torch.tensor([0.1, 0.2]))
    torch.testing.assert_close(
        controller.active_quantizers["layer4_signed_skip"].scales,
        torch.tensor([0.3]))
    controller.close()


def test_boundary_controller_requires_every_range_override():
    _, controller = calibrated_controller()

    with pytest.raises(ValueError, match="ranges must name every boundary"):
        controller.configure_specs_with_ranges(
            bit_widths={"decoder_entry": 4, "layer4_signed_skip": 4},
            group_sizes={"decoder_entry": 4, "layer4_signed_skip": 4},
            scale_factors={"decoder_entry": 1.0,
                           "layer4_signed_skip": 1.0},
            maximum_overrides={
                "decoder_entry": torch.tensor([1.0, 1.0]),
            },
            quantize=True)
    controller.close()


def test_boundary_controller_records_owned_qdq():
    class Recorder(object):
        def __init__(self):
            self.rows = []

        def record(self, *args):
            self.rows.append(args)

    model, controller = calibrated_controller()
    controller.configure_specs(
        bit_widths={"decoder_entry": 4, "layer4_signed_skip": 4},
        group_sizes={"decoder_entry": 4, "layer4_signed_skip": None},
        scale_factors={"decoder_entry": 1.0,
                       "layer4_signed_skip": 1.0},
        quantize=True)
    recorder = Recorder()
    controller.set_activation_recorder(recorder)

    model(torch.randn(1, 8, 5, 5), torch.randn(1, 4, 5, 5))

    assert [(row[0], row[1], row[3]) for row in recorder.rows] == [
        ("boundary.decoder_entry", "boundary", "decoder"),
        ("boundary.layer4_signed_skip", "boundary", "decoder"),
    ]
    controller.close()
