"""Encoder-searchable CSPN with the released decoder and propagation path."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from cspn import Affinity_Propagate
from spn_quant.nas.spec import EncoderSpec
from torch_resnet_cspn_nyu import (
    BasicBlock,
    Gudi_UpProj_Block,
    Gudi_UpProj_Block_Cat,
    Simple_Gudi_UpConv_Block_Last_Layer,
)


class ProjectionStage(nn.Module):
    """Preserve a stage downsample while omitting its residual blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.projection(value)))


def _adapter(in_channels: int, out_channels: int) -> nn.Module:
    if in_channels == out_channels:
        return nn.Identity()
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)


def _residual_stage(
    in_channels: int,
    out_channels: int,
    depth: int,
    stride: int,
) -> nn.Sequential:
    if depth == 0:
        if stride != 2:
            raise ValueError("projection-only stages must downsample")
        return nn.Sequential(ProjectionStage(in_channels, out_channels))
    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1,
                stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    blocks = [BasicBlock(in_channels, out_channels, stride, downsample)]
    blocks.extend(BasicBlock(out_channels, out_channels) for _ in range(1, depth))
    return nn.Sequential(*blocks)


class CSPNEncoderNAS(nn.Module):
    decoder_channels = (64, 64, 128, 512)

    def __init__(
        self,
        spec: EncoderSpec,
        cspn_step: int = 24,
        cspn_norm_type: str = "8sum",
    ) -> None:
        super().__init__()
        self.encoder_spec = spec
        widths = spec.widths
        depths = spec.depths

        self.conv1_1 = nn.Conv2d(
            4, spec.stem_width, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(spec.stem_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = _residual_stage(
            spec.stem_width, widths[0], depths[0], stride=1)
        self.layer2 = _residual_stage(
            widths[0], widths[1], depths[1], stride=2)
        self.layer3 = _residual_stage(
            widths[1], widths[2], depths[2], stride=2)
        self.layer4 = _residual_stage(
            widths[2], widths[3], depths[3], stride=2)

        self.stem_skip_adapter = _adapter(spec.stem_width, 64)
        self.stage1_skip_adapter = _adapter(widths[0], 64)
        self.stage2_skip_adapter = _adapter(widths[1], 128)
        self.bottleneck_adapter = _adapter(widths[3], 512)

        self.conv2 = nn.Conv2d(
            512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.gud_up_proj_layer1 = Gudi_UpProj_Block(512, 256, 15, 19)
        self.gud_up_proj_layer2 = Gudi_UpProj_Block_Cat(256, 128, 29, 38)
        self.gud_up_proj_layer3 = Gudi_UpProj_Block_Cat(128, 64, 57, 76)
        self.gud_up_proj_layer4 = Gudi_UpProj_Block_Cat(64, 64, 114, 152)
        self.gud_up_proj_layer5 = Simple_Gudi_UpConv_Block_Last_Layer(
            64, 1, 228, 304)
        self.gud_up_proj_layer6 = Simple_Gudi_UpConv_Block_Last_Layer(
            64, 8, 228, 304)
        self.post_process_layer = Affinity_Propagate(
            cspn_step, 3, norm_type=cspn_norm_type)
        self._initialize_new_modules()

    def _initialize_new_modules(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_encoder(
        self, value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stem = self.conv1_1(value)
        stem_skip = self.stem_skip_adapter(stem)
        value = self.maxpool(self.relu(self.bn1(stem)))
        value = self.layer1(value)
        stage1_skip = self.stage1_skip_adapter(value)
        value = self.layer2(value)
        stage2_skip = self.stage2_skip_adapter(value)
        value = self.layer3(value)
        value = self.layer4(value)
        bottleneck = self.bottleneck_adapter(value)
        return stem_skip, stage1_skip, stage2_skip, bottleneck

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        sparse_depth = value[:, 3:4].clone()
        stem_skip, stage1_skip, stage2_skip, value = self.forward_encoder(value)
        value = self.bn2(self.conv2(value))
        value = self.gud_up_proj_layer1(value)
        value = self.gud_up_proj_layer2(value, stage2_skip)
        value = self.gud_up_proj_layer3(value, stage1_skip)
        value = self.gud_up_proj_layer4(value, stem_skip)
        guidance = self.gud_up_proj_layer6(value)
        initial_depth = self.gud_up_proj_layer5(value)
        return self.post_process_layer(guidance, initial_depth, sparse_depth)


def build_cspn_nas(
    spec: EncoderSpec,
    cspn_step: int = 24,
    cspn_norm_type: str = "8sum",
) -> CSPNEncoderNAS:
    return CSPNEncoderNAS(
        spec=spec,
        cspn_step=cspn_step,
        cspn_norm_type=cspn_norm_type,
    )
