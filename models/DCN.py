"""Inference-compatible DCNv2 API backed by torchvision's native CUDA op.

The original NLSPN/CompletionFormer extension is tied to Python 3.7 and an
older PyTorch ABI.  The forward signature below is the one used by those
repositories; torchvision infers convolution and deformable group counts from
the tensor shapes and implements the same modulated deformable convolution.
"""

from __future__ import annotations

from torchvision.ops import deform_conv2d


def modulated_deform_conv_forward(
        input, weight, bias, offset, mask,
        kernel_h, kernel_w, stride_h, stride_w,
        pad_h, pad_w, dilation_h, dilation_w,
        group, deformable_group, im2col_step):
    del kernel_h, kernel_w, group, deformable_group, im2col_step
    return deform_conv2d(
        input=input,
        offset=offset,
        weight=weight,
        bias=bias,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation_h, dilation_w),
        mask=mask,
    )


def modulated_deform_conv_backward(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(
        "the torchvision DCN compatibility backend supports inference only")
