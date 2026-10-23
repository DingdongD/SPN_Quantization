from types import SimpleNamespace

import torch

from scripts.export_completionformer_rknn_fp16_stable import (
    stabilize_completionformer_module,
)


def test_completionformer_module_stabilization_preserves_eval_function():
    torch.manual_seed(4)
    patch3 = SimpleNamespace(
        proj=torch.nn.Conv2d(3, 4, 1),
        norm=torch.nn.LayerNorm(4, eps=1e-5),
    )
    patch4 = SimpleNamespace(
        proj=torch.nn.Conv2d(4, 5, 1),
        norm=torch.nn.LayerNorm(5, eps=1e-5),
    )
    deconv = torch.nn.ConvTranspose2d(4, 3, 1, bias=False)
    batch_norm = torch.nn.BatchNorm2d(3, eps=1e-5).eval()
    batch_norm.running_mean.copy_(torch.tensor([0.3, -0.4, 0.8]))
    batch_norm.running_var.copy_(torch.tensor([0.7, 1.2, 1.8]))
    model = SimpleNamespace(backbone=SimpleNamespace(
        former=SimpleNamespace(patch_embed3=patch3, patch_embed4=patch4),
        dec4=torch.nn.Sequential(torch.nn.Sequential(deconv, batch_norm)),
    ))

    patch3_input = torch.randn(2, 3, 3, 4)
    patch4_input = torch.randn(2, 4, 3, 4)
    decoder_input = torch.randn(2, 4, 3, 4)

    def patch_output(block, value):
        projected = block.proj(value).flatten(2).transpose(1, 2)
        return block.norm(projected)

    expected_patch3 = patch_output(patch3, patch3_input)
    expected_patch4 = patch_output(patch4, patch4_input)
    expected_decoder = batch_norm(deconv(decoder_input))

    assert stabilize_completionformer_module(model, scale=0.5) == 3

    torch.testing.assert_close(
        patch_output(patch3, patch3_input), expected_patch3,
        rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        patch_output(patch4, patch4_input), expected_patch4,
        rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        batch_norm(deconv(decoder_input)), expected_decoder,
        rtol=2e-5, atol=2e-6)
