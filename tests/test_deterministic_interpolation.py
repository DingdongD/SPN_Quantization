from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts import train_nyu_selected_qat as qat_runner
from spn_quant.deterministic_interpolation import (
    deterministic_adaptive_max_pool2d_1,
    deterministic_bilinear2d,
    install_completionformer_qat_interpolation,
)


@pytest.mark.parametrize("align_corners", (False, True))
@pytest.mark.parametrize("input_size,output_size", (
    ((3, 5), (6, 9)),
    ((4, 7), (2, 3)),
    ((1, 3), (4, 5)),
))
def test_deterministic_bilinear_matches_official_forward_and_gradient(
        align_corners, input_size, output_size):
    source = torch.randn(2, 3, *input_size, dtype=torch.float64)
    gradient = torch.randn(2, 3, *output_size, dtype=torch.float64)
    reference_input = source.clone().requires_grad_(True)
    deterministic_input = source.clone().requires_grad_(True)

    reference = F.interpolate(
        reference_input,
        size=output_size,
        mode="bilinear",
        align_corners=align_corners,
    )
    actual = deterministic_bilinear2d(
        deterministic_input, output_size, align_corners)
    reference.backward(gradient)
    actual.backward(gradient)

    torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        deterministic_input.grad,
        reference_input.grad,
        rtol=1e-12,
        atol=1e-12,
    )


def test_deterministic_bilinear_backward_has_no_scalar_tensor_writes(
        monkeypatch):
    original = torch.Tensor.__setitem__
    writes = []

    def counted(tensor, key, value):
        writes.append((key, value))
        return original(tensor, key, value)

    monkeypatch.setattr(torch.Tensor, "__setitem__", counted)
    source = torch.randn(1, 2, 3, 5, requires_grad=True)
    output = deterministic_bilinear2d(source, (6, 9), False)
    output.sum().backward()

    assert writes == []


def test_deterministic_bilinear_backward_uses_dense_axis_products():
    source = torch.randn(1, 2, 8, 12, requires_grad=True)
    output = deterministic_bilinear2d(source, (16, 24), False)

    with torch.autograd.profiler.profile() as profiler:
        output.sum().backward()

    operations = set(row.key for row in profiler.key_averages())
    assert "aten::matmul" in operations
    assert "aten::index_select" not in operations


class InterpolateConvBNReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode = "bilinear"
        self.align_corners = False
        self.conv = nn.Identity()
        self.bn = None
        self.relu = None

    def forward(self, value, size):
        return F.interpolate(
            value,
            size=size,
            mode=self.mode,
            align_corners=self.align_corners,
        )


class PVT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed1 = SimpleNamespace(num_patches=4)
        self.block1 = nn.ModuleList((PVTBlock(),))

    def _get_pos_embed(self, pos_embed, patch_embed, height, width):
        if height * width == self.patch_embed1.num_patches:
            return pos_embed
        return F.interpolate(
            pos_embed.reshape(
                1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
            size=(height, width),
            mode="bilinear",
        ).reshape(1, -1, height * width).permute(0, 2, 1)


class Backbone(nn.Module):
    def __init__(self, interpolate_decoder=True):
        super().__init__()
        self.use_interpolate_decoder = interpolate_decoder
        self.concat_align_corners = True
        self.former = PVT()
        self.dec2 = nn.Sequential(nn.Identity(), ResidualBlock())
        if interpolate_decoder:
            self.decoder = InterpolateConvBNReLU()

    def _concat(self, decoder, encoder, dim=1):
        decoder = F.interpolate(
            decoder,
            size=encoder.shape[-2:],
            mode="bilinear",
            align_corners=self.concat_align_corners,
        )
        return torch.cat((decoder, encoder), dim=dim)


class CompletionFormer(nn.Module):
    def __init__(self, interpolate_decoder=True):
        super().__init__()
        self.backbone = Backbone(interpolate_decoder)


class ChannelAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ca = ChannelAttention()


class PVTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.resblock = ResidualBlock()


def test_completionformer_installer_binds_all_declared_interpolation_paths():
    model = CompletionFormer()

    manifest = install_completionformer_qat_interpolation(model)

    assert manifest == {
        "implementation": "official_forward_deterministic_input_gradient",
        "decoder_modules": ["backbone.decoder"],
        "adaptive_max_pool_modules": [
            "backbone.former.block1.0.resblock.ca.max_pool",
            "backbone.dec2.1.ca.max_pool",
        ],
        "concat_method": "backbone._concat",
        "position_method": "backbone.former._get_pos_embed",
    }
    decoder = torch.randn(1, 2, 2, 3, requires_grad=True)
    encoder = torch.randn(1, 1, 4, 5)
    output = model.backbone._concat(
        model.backbone.decoder(decoder, encoder.shape[-2:]),
        encoder,
    )
    output.square().mean().backward()
    assert torch.isfinite(decoder.grad).all()


def test_selected_qat_installs_deterministic_ops_only_for_completionformer():
    model = CompletionFormer()

    manifest = qat_runner.install_deterministic_qat_operators(
        "completionformer", model)

    assert manifest["decoder_modules"] == ["backbone.decoder"]
    assert qat_runner.install_deterministic_qat_operators(
        "cspn", nn.Linear(2, 2)) is None
    assert qat_runner.install_deterministic_qat_operators(
        "dyspn", nn.Linear(2, 2)) is None
    assert qat_runner.install_deterministic_qat_operators(
        "nlspn", nn.Linear(2, 2)) is None


def test_completionformer_installer_accepts_declared_transpose_conv_decoder():
    model = CompletionFormer(interpolate_decoder=False)

    manifest = install_completionformer_qat_interpolation(model)

    assert manifest["decoder_modules"] == []


def test_deterministic_adaptive_max_pool_matches_official_tie_gradient():
    source = torch.tensor([[[[3.0, 3.0], [1.0, 2.0]]]], requires_grad=True)
    reference_input = source.detach().clone().requires_grad_(True)

    actual = deterministic_adaptive_max_pool2d_1(source)
    reference = F.adaptive_max_pool2d(reference_input, 1)
    actual.backward(torch.ones_like(actual))
    reference.backward(torch.ones_like(reference))

    assert torch.equal(actual, reference)
    assert torch.equal(source.grad, reference_input.grad)
    assert torch.equal(
        source.grad,
        torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
    )
