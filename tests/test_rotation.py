import pytest
import torch
import torch.nn as nn

from spn_quant.rotation import (
    CSPNRotationController,
    RotationBoundaryObserver,
    absorb_input_rotation,
    hadamard_rotation_matrix,
    random_orthogonal_matrix,
    rotate_channels,
)
from spn_quant.adapters import RotationBoundary, RotationConsumer


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


class RotationToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_entry = DecoderEntryBlock()
        self.concat = ConcatBlock()

    def forward(self, value, signed_skip):
        return self.concat(self.decoder_entry(value), signed_skip)


def toy_boundaries():
    return (
        RotationBoundary(
            "decoder_entry", "decoder_entry", 0, (
                RotationConsumer("decoder_entry.conv1", 0, None),
                RotationConsumer("decoder_entry.sc_conv1", 0, None),
            )),
        RotationBoundary(
            "layer4_signed_skip", "concat", 1, (
                RotationConsumer("concat.conv1_1", 4, 4),
            )),
    )


def test_random_rotation_is_deterministic_and_orthogonal():
    first = random_orthogonal_matrix(8, seed=17)
    second = random_orthogonal_matrix(8, seed=17)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(
        first @ first.t(), torch.eye(8), rtol=1e-5, atol=1e-6)


def test_hadamard_rotation_is_orthogonal():
    rotation = hadamard_rotation_matrix(8, seed=17)

    torch.testing.assert_close(
        rotation @ rotation.t(), torch.eye(8), rtol=1e-5, atol=1e-6)


def test_hadamard_requires_power_of_two_channels():
    with pytest.raises(ValueError, match="power of two"):
        hadamard_rotation_matrix(6, seed=17)


def test_absorbed_conv_matches_rotated_input():
    torch.manual_seed(3)
    conv = nn.Conv2d(8, 5, 3, padding=1, bias=True)
    value = torch.randn(2, 8, 7, 9)
    rotation = random_orthogonal_matrix(8, seed=3)
    reference = conv(value)

    absorb_input_rotation(conv, rotation)
    actual = conv(rotate_channels(value, rotation))

    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)


def test_absorb_rotation_updates_only_concat_slice():
    torch.manual_seed(5)
    conv = nn.Conv2d(12, 5, 1, bias=False)
    left = torch.rand(2, 4, 3, 3)
    right = torch.randn(2, 8, 3, 3)
    rotation = random_orthogonal_matrix(8, seed=5)
    reference = conv(torch.cat((left, right), dim=1))

    absorb_input_rotation(
        conv, rotation, channel_start=4, channel_count=8)
    actual = conv(torch.cat(
        (left, rotate_channels(right, rotation)), dim=1))

    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("boundary,method", [
    ("decoder_entry", "random"),
    ("decoder_entry", "hadamard"),
    ("layer4_signed_skip", "random"),
    ("layer4_signed_skip", "hadamard"),
])
def test_controller_preserves_fp_equivalence(boundary, method):
    torch.manual_seed(11)
    model = RotationToyModel()
    controller = CSPNRotationController(
        model, toy_boundaries(), seed=7)
    value = torch.randn(1, 8, 5, 5)
    signed_skip = torch.randn(1, 4, 5, 5)
    reference = model(value, signed_skip)
    methods = {
        "decoder_entry": "identity",
        "layer4_signed_skip": "identity",
    }
    methods[boundary] = method

    controller.configure(
        methods, bits=4, group_size=None, quantize=False)
    actual = model(value, signed_skip)

    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)
    controller.close()


def test_boundary_observer_reports_tail_and_qdq_metrics():
    observer = RotationBoundaryObserver(channels=4)
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


def test_group_quantizer_requires_divisible_channel_count():
    observer = RotationBoundaryObserver(channels=6)
    observer.update(torch.randn(1, 6, 2, 2))

    with pytest.raises(ValueError, match="divide"):
        observer.quantizer(bits=4, group_size=4)


def test_controller_builds_rotated_fp_weight_sources_without_mutation():
    torch.manual_seed(23)
    model = RotationToyModel()
    controller = CSPNRotationController(
        model, toy_boundaries(), seed=7)
    original = {
        name: module.weight.detach().clone()
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    }
    methods = {
        "decoder_entry": "random",
        "layer4_signed_skip": "hadamard",
    }

    sources = controller.weight_source_overrides(methods)

    assert set(sources) == {
        "decoder_entry.conv1",
        "decoder_entry.sc_conv1",
        "concat.conv1_1",
    }
    for name in sources:
        assert not torch.equal(sources[name], original[name])
        torch.testing.assert_close(
            dict(model.named_modules())[name].weight, original[name])
    controller.close()


def test_identity_rotation_has_no_weight_source_overrides():
    torch.manual_seed(47)
    model = RotationToyModel()
    controller = CSPNRotationController(
        model, toy_boundaries(), seed=7)
    sources = controller.weight_source_overrides({
        "decoder_entry": "identity",
        "layer4_signed_skip": "identity",
    })

    assert sources == {}
    controller.close()


def test_single_rotation_overrides_only_its_consumers():
    model = RotationToyModel()
    controller = CSPNRotationController(
        model, toy_boundaries(), seed=7)

    sources = controller.weight_source_overrides({
        "decoder_entry": "random",
        "layer4_signed_skip": "identity",
    })

    assert set(sources) == {
        "decoder_entry.conv1", "decoder_entry.sc_conv1"}
    controller.close()


def test_identity_controller_forward_is_bit_exact():
    torch.manual_seed(53)
    model = RotationToyModel()
    controller = CSPNRotationController(
        model, toy_boundaries(), seed=7)
    value = torch.randn(1, 8, 5, 5)
    signed_skip = torch.randn(1, 4, 5, 5)
    reference = model(value, signed_skip)
    controller.rotations["decoder_entry"]["identity"] = \
        random_orthogonal_matrix(8, seed=71)
    controller.rotations["layer4_signed_skip"]["identity"] = \
        random_orthogonal_matrix(4, seed=73)

    controller.configure(
        {
            "decoder_entry": "identity",
            "layer4_signed_skip": "identity",
        }, bits=4, group_size=None, quantize=False)
    candidate = model(value, signed_skip)

    assert torch.equal(candidate, reference)
    controller.close()


def test_controller_does_not_reabsorb_already_transformed_w4_weight():
    torch.manual_seed(29)
    model = RotationToyModel()
    controller = CSPNRotationController(
        model, toy_boundaries(), seed=7)
    controller.observe()
    model(torch.randn(1, 8, 5, 5), torch.randn(1, 4, 5, 5))
    controller.freeze()
    methods = {
        "decoder_entry": "random",
        "layer4_signed_skip": "hadamard",
    }
    sources = controller.weight_source_overrides(methods)
    for name, source in sources.items():
        dict(model.named_modules())[name].weight.data.copy_(source)
    before = dict(
        (name, dict(model.named_modules())[name].weight.detach().clone())
        for name in sources)

    controller.configure(
        methods, bits=4, group_size=None,
        quantize=True, absorb_weights=False)

    for name in sources:
        torch.testing.assert_close(
            dict(model.named_modules())[name].weight, before[name])
    controller.close()


def test_controller_validation_failure_does_not_change_active_weights():
    torch.manual_seed(31)
    model = RotationToyModel()
    controller = CSPNRotationController(
        model, toy_boundaries(), seed=7)
    controller.observe()
    model(torch.randn(1, 8, 5, 5), torch.randn(1, 4, 5, 5))
    controller.freeze()
    methods = {
        "decoder_entry": "random",
        "layer4_signed_skip": "hadamard",
    }
    controller.configure(
        methods, bits=4, group_size=None, quantize=True)
    before = {
        name: module.weight.detach().clone()
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    }

    with pytest.raises(ValueError, match="divide"):
        controller.configure(
            methods, bits=4, group_size=3, quantize=True)

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            torch.testing.assert_close(module.weight, before[name])
    assert controller.mode == "quantize"
    assert controller.active_methods == methods
    controller.close()


def test_controller_accepts_explicit_group_size_per_boundary():
    torch.manual_seed(37)
    model = RotationToyModel()
    controller = CSPNRotationController(
        model, toy_boundaries(), seed=7)
    controller.observe()
    model(torch.randn(1, 8, 5, 5), torch.randn(1, 4, 5, 5))
    controller.freeze()
    methods = {
        "decoder_entry": "identity",
        "layer4_signed_skip": "identity",
    }

    controller.configure_group_sizes(
        methods, bits=4,
        group_sizes={"decoder_entry": 4, "layer4_signed_skip": None},
        quantize=True, absorb_weights=False)

    assert controller.active_quantizers["decoder_entry"].group_size == 4
    assert controller.active_quantizers["layer4_signed_skip"].group_size == 4
    controller.close()


def test_controller_records_owned_boundary_qdq():
    class Recorder(object):
        def __init__(self):
            self.rows = []

        def record(self, *args):
            self.rows.append(args)

    torch.manual_seed(41)
    model = RotationToyModel()
    controller = CSPNRotationController(
        model, toy_boundaries(), seed=7)
    value = torch.randn(1, 8, 5, 5)
    signed_skip = torch.randn(1, 4, 5, 5)
    controller.observe()
    model(value, signed_skip)
    controller.freeze()
    controller.configure_group_sizes(
        {
            "decoder_entry": "identity",
            "layer4_signed_skip": "identity",
        },
        bits=4,
        group_sizes={"decoder_entry": 4, "layer4_signed_skip": None},
        quantize=True, absorb_weights=False)
    recorder = Recorder()
    controller.set_activation_recorder(recorder)

    model(value, signed_skip)

    assert [(row[0], row[1], row[3]) for row in recorder.rows] == [
        ("rotation.decoder_entry", "boundary", "decoder"),
        ("rotation.layer4_signed_skip", "boundary", "decoder"),
    ]
    assert all(row[-1] == 1 for row in recorder.rows)
    controller.close()


def test_controller_accepts_bit_width_and_scale_per_boundary():
    torch.manual_seed(43)
    model = RotationToyModel()
    controller = CSPNRotationController(
        model, toy_boundaries(), seed=7)
    controller.observe()
    model(torch.randn(1, 8, 5, 5), torch.randn(1, 4, 5, 5))
    controller.freeze()

    controller.configure_specs(
        {
            "decoder_entry": "identity",
            "layer4_signed_skip": "identity",
        },
        bit_widths={"decoder_entry": 8, "layer4_signed_skip": 4},
        group_sizes={"decoder_entry": 4, "layer4_signed_skip": None},
        scale_factors={"decoder_entry": 0.5,
                       "layer4_signed_skip": 1.0},
        quantize=True, absorb_weights=False)

    decoder = controller.active_quantizers["decoder_entry"]
    signed_skip = controller.active_quantizers["layer4_signed_skip"]
    assert decoder.bits == 8
    assert signed_skip.bits == 4
    expected = controller.observers[
        "decoder_entry"]["identity"].channel_absmax.reshape(2, 4).amax(1)
    torch.testing.assert_close(decoder.scales, expected * 0.5 / 127.0)
    controller.close()
