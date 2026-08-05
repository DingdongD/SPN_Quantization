import torch
import torch.nn as nn
import torch.nn.functional as F

from cspn_aligned_hw import FastAffinityPropagate


def fake_quant_i8(x, scale):
    if scale is None:
        return x
    return (torch.clamp(torch.round(x * scale), -128, 127) - x * scale).detach() / scale + x


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvBN(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(x))


class SampleDown1x1BN(nn.Module):
    """Retrainable RHB-friendly replacement for stride-2 projection.

    Host/runtime performs the even-grid sample. RHB executes only 1x1 Conv.
    This is not exact to stride-2 3x3 convolution and must be trained as a
    hardware-aligned model family.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.proj(x[:, :, ::2, ::2]))


class BasicBlockAligned(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample_variant="sample_1x1"):
        super().__init__()
        if stride == 2 and downsample_variant == "sample_1x1":
            self.conv1 = nn.Sequential(
                SampleDown1x1BN(in_ch, out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = ConvBNReLU(in_ch, out_ch, kernel_size=3, stride=stride)
        self.conv2 = ConvBN(out_ch, out_ch, kernel_size=3, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            if stride == 2 and downsample_variant == "sample_1x1":
                self.shortcut = SampleDown1x1BN(in_ch, out_ch)
            else:
                self.shortcut = ConvBN(in_ch, out_ch, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.conv2(self.conv1(x))
        return self.relu(out + identity)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv0 = ConvBNReLU(in_ch + skip_ch, out_ch, kernel_size=3)
        self.block = BasicBlockAligned(out_ch, out_ch, stride=1)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(self.conv0(x))


class ResNetTinyAlignedBackboneHeads(nn.Module):
    """ResNet18-style CSPN-tiny backbone adapted for RHB deployment.

    The topology keeps residual stages and decoder skip fusion from the remote
    CSPN-tiny direction, while using hardware-aligned downsample blocks that can
    later be scheduled as Host sample + RHB 1x1 Conv. BatchNorm is retained for
    training quality and is expected to be folded into Conv during export.
    """

    def __init__(
        self,
        width=24,
        downsample_variant="sample_1x1",
        fused_contract_qat=False,
        qat_scales=None,
    ):
        super().__init__()
        self.fused_contract_qat = bool(fused_contract_qat)
        self.qat_scales = qat_scales or {}
        c1, c2, c3, c4 = width, width * 2, width * 4, width * 4
        self.stem = nn.Sequential(
            ConvBNReLU(4, c1, kernel_size=3, stride=1),
            BasicBlockAligned(c1, c1, stride=1, downsample_variant=downsample_variant),
        )
        self.stage2 = nn.Sequential(
            BasicBlockAligned(c1, c2, stride=2, downsample_variant=downsample_variant),
            BasicBlockAligned(c2, c2, stride=1, downsample_variant=downsample_variant),
        )
        self.stage3 = nn.Sequential(
            BasicBlockAligned(c2, c3, stride=2, downsample_variant=downsample_variant),
            BasicBlockAligned(c3, c3, stride=1, downsample_variant=downsample_variant),
        )
        self.stage4 = nn.Sequential(
            BasicBlockAligned(c3, c4, stride=2, downsample_variant=downsample_variant),
            BasicBlockAligned(c4, c4, stride=1, downsample_variant=downsample_variant),
        )
        self.dec3 = DecoderBlock(c4, c3, c3)
        self.dec2 = DecoderBlock(c3, c2, c2)
        self.dec1 = DecoderBlock(c2, c1, c1)
        self.refine = BasicBlockAligned(c1, c1, stride=1, downsample_variant=downsample_variant)
        self.depth_head = nn.Sequential(
            ConvBNReLU(c1, c1, kernel_size=3),
            nn.Conv2d(c1, 1, kernel_size=3, padding=1, bias=True),
        )
        self.guidance_head = nn.Sequential(
            ConvBNReLU(c1, c1, kernel_size=3),
            nn.Conv2d(c1, 8, kernel_size=3, padding=1, bias=True),
        )
        self._init_weights()

    def _fq(self, name, x):
        return fake_quant_i8(x, self.qat_scales.get(name))

    def _init_weights(self):
        for op in self.modules():
            if isinstance(op, nn.Conv2d):
                nn.init.kaiming_normal_(op.weight, mode="fan_out", nonlinearity="relu")
                if op.bias is not None:
                    nn.init.zeros_(op.bias)
            elif isinstance(op, nn.BatchNorm2d):
                nn.init.ones_(op.weight)
                nn.init.zeros_(op.bias)

    def forward(self, x):
        s1 = self.stem(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        x = self.dec3(s4, s3)
        if self.fused_contract_qat:
            dec2_input = torch.cat(
                [F.interpolate(x, size=s2.shape[-2:], mode="bilinear", align_corners=False), s2],
                dim=1,
            )
            dec2_input = self._fq("dec2_input", dec2_input)
            x = self._fq("dec2", self.dec2.block(self.dec2.conv0(dec2_input)))

            dec1_input = torch.cat(
                [F.interpolate(x, size=s1.shape[-2:], mode="bilinear", align_corners=False), s1],
                dim=1,
            )
            dec1_input = self._fq("dec1_input", dec1_input)
            x = self._fq("dec1", self.dec1.block(self.dec1.conv0(dec1_input)))
            x = self._fq("refined", self.refine(self._fq("dec1", x)))
        else:
            x = self.dec2(x, s2)
            x = self.dec1(x, s1)
            x = self.refine(x)
        return self.depth_head(x), self.guidance_head(x)


class ResNetTinyAlignedCSPNHW(nn.Module):
    def __init__(
        self,
        width=24,
        cspn_step=4,
        cspn_norm_type="8sum",
        downsample_variant="sample_1x1",
        fused_contract_qat=False,
        qat_scales=None,
    ):
        super().__init__()
        self.backbone = ResNetTinyAlignedBackboneHeads(
            width=width,
            downsample_variant=downsample_variant,
            fused_contract_qat=fused_contract_qat,
            qat_scales=qat_scales,
        )
        self.cspn = FastAffinityPropagate(cspn_step, 3, norm_type=cspn_norm_type)

    def forward_heads(self, x):
        return self.backbone(x)

    def forward(self, x):
        sparse_depth = x[:, 3:4]
        raw_depth, raw_guidance = self.forward_heads(x)
        return self.cspn(raw_guidance, raw_depth, sparse_depth)
