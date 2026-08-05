from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34


TensorDict = Dict[str, torch.Tensor]


@dataclass
class DySPNHWConfig:
    input_height: int = 128
    input_width: int = 128
    network: str = "resnet18"
    from_scratch: bool = True
    pretrained_resnet_path: Optional[str] = None
    iteration: int = 6
    num_neighbor: int = 5
    max_depth: float = 10.0
    frontend_rhb_quant: bool = False
    frontend_rgb_scale: float = 12.0
    frontend_dep_scale: float = 16.0
    frontend_fe1_scale: float = 1.0
    frontend_fe1_down_scale: float = 1.0
    frontend_fe2_scale: float = 1.0
    frontend_fe3_scale: float = 1.0
    frontend_quant_semantics: str = "raw"
    tail_rhb_quant: bool = False
    tail_quant_points: str = ""
    tail_quant_semantics: str = "raw"
    tail_fe4_scale: float = 1.0
    tail_fe5_scale: float = 1.0
    tail_fd4_scale: float = 1.0
    tail_fd3_scale: float = 1.0
    tail_fd2_scale: float = 1.0
    tail_fd1_scale: float = 1.0
    tail_gd_fd1_scale: float = 1.0
    tail_guide_scale: float = 1.0
    branch_contract_qat: bool = False
    branch_qat_scales: Optional[Dict[str, float]] = None
    branch_quant_semantics: str = "dequant"
    collect_branch_stats: bool = False


def conv_bn_relu(
    ch_in: int,
    ch_out: int,
    kernel: int,
    stride: int = 1,
    padding: int = 0,
    bn: bool = True,
    relu: bool = True,
) -> nn.Sequential:
    layers: List[nn.Module] = [nn.Conv2d(ch_in, ch_out, kernel, stride, padding, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def ste_round_i8(x: torch.Tensor, scale: float = 1.0, semantics: str = "raw") -> torch.Tensor:
    """Straight-through int8 quantization.

    `raw` returns the integer tensor consumed by RHB-style downstream kernels.
    `dequant` returns q / scale and is useful for software QDQ diagnostics, but
    it is not the current RHB submodel boundary contract.
    """
    q = torch.clamp(torch.round(x * float(scale)), -128.0, 127.0)
    if semantics == "raw":
        target = q
    elif semantics == "dequant":
        target = q / float(scale)
    else:
        raise ValueError(f"unsupported quant semantics: {semantics}")
    return x + (target - x).detach()


class ResizeConvBNReLU(nn.Module):
    """Compiler-aligned replacement for ConvTranspose2d.

    DySPN's reference decoder uses ConvTranspose2d. Current RHB deployment rules
    prefer Host resize + RHB Conv/BN/ReLU. This is not an exact replacement for a
    pretrained reference checkpoint; train/fine-tune this HW model directly.
    """

    def __init__(self, ch_in: int, ch_out: int, scale_factor: int = 2) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = conv_bn_relu(ch_in, ch_out, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return self.conv(x)


def _build_resnet_layers(network: str, pretrained_path: Optional[str]) -> nn.Module:
    if network == "resnet18":
        net = resnet18(weights=None)
    elif network == "resnet34":
        net = resnet34(weights=None)
    else:
        raise ValueError(f"unsupported network: {network}")
    if pretrained_path:
        state = torch.load(pretrained_path, map_location="cpu")
        net.load_state_dict(state)
    return net


def load_dyspn_hw_checkpoint(model: nn.Module, ckpt_path: Optional[str] = None, strict: bool = True) -> None:
    path = ckpt_path or os.environ.get("DYSPN_HW_CKPT", "")
    if not path:
        return
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    clean = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("basenet."):
            key = key[len("basenet.") :]
        clean[key] = value
    missing, unexpected = model.load_state_dict(clean, strict=strict)
    if missing or unexpected:
        print(f"[DYSPN_HW_CKPT] missing={missing} unexpected={unexpected}")


class DySPNHostPropagation(nn.Module):
    """DySPN propagation kept on Host.

    The RHB-friendly part is `conv_offset_aff`, a Conv2d over guide channels.
    The grid_sample loop is not considered RHB-safe in the current rule set and
    is intentionally kept in Host glue.
    """

    def __init__(self, iteration: int = 6, num_neighbor: int = 5, mode: str = "yx") -> None:
        super().__init__()
        if num_neighbor not in (1, 3, 5, 9):
            raise ValueError("num_neighbor must be one of 1, 3, 5, 9")
        self.iteration = iteration
        self.num_neighbor = num_neighbor
        self.mode = mode
        self.ch = iteration * num_neighbor
        self.conv_offset_aff = nn.Conv2d(self.ch, 3 * self.ch, kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.zeros_(self.conv_offset_aff.weight)
        nn.init.zeros_(self.conv_offset_aff.bias)

        if num_neighbor == 9:
            offsets = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        elif num_neighbor == 5:
            offsets = [(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)]
        elif num_neighbor == 3:
            offsets = [(-1, 0), (0, 0), (1, 0)]
        else:
            offsets = [(0, 0)]
        self.register_buffer("base_offsets", torch.tensor(offsets, dtype=torch.float32).view(num_neighbor, 1, 1, 2))

    def _refgrid(self, offset: torch.Tensor) -> torch.Tensor:
        b, _, h, w = offset.shape
        offset = offset.view(b, self.iteration, self.num_neighbor, 2, h, w).permute(0, 1, 2, 4, 5, 3)
        ref_y = torch.linspace(-h + 1, h - 1, h, device=offset.device, dtype=offset.dtype)
        ref_x = torch.linspace(-w + 1, w - 1, w, device=offset.device, dtype=offset.dtype)
        base = self.base_offsets.to(device=offset.device, dtype=offset.dtype)
        if self.mode == "yx":
            grid_x = ((offset[..., 1] + base[..., 0]) * 2.0 + ref_x.view(1, 1, 1, 1, w)) / w
            grid_y = ((offset[..., 0] + base[..., 1]) * 2.0 + ref_y.view(1, 1, 1, h, 1)) / h
        else:
            grid_x = ((offset[..., 0] + base[..., 0]) * 2.0 + ref_x.view(1, 1, 1, 1, w)) / w
            grid_y = ((offset[..., 1] + base[..., 1]) * 2.0 + ref_y.view(1, 1, 1, h, 1)) / h
        return torch.stack((grid_x, grid_y), dim=-1)

    def forward(self, pred_init: torch.Tensor, guide: torch.Tensor, sparse_depth: torch.Tensor, confidence: torch.Tensor) -> TensorDict:
        b, _, h, w = pred_init.shape
        offset_aff = self.conv_offset_aff(guide)
        offset, aff = torch.split(offset_aff, [2 * self.ch, self.ch], dim=1)
        confidence = torch.sigmoid(confidence) * sparse_depth.sign()
        grids = torch.unbind(self._refgrid(offset).float(), dim=1)
        aff = aff.view(b, self.iteration, self.num_neighbor, h, w)
        aff = torch.chunk(torch.softmax(aff, dim=2), self.iteration, dim=1)

        feat = pred_init.float()
        intermediates: List[torch.Tensor] = []
        for i in range(self.iteration):
            out = 0.0
            for j in range(self.num_neighbor):
                out = out + F.grid_sample(
                    feat,
                    grids[i][:, j],
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                ) * aff[i][:, :, j]
            feat = (1.0 - confidence) * out + confidence * sparse_depth
            intermediates.append(feat)
        return {
            "pred": feat.clamp(0.0, 10.0),
            "pred_init": pred_init,
            "offset_aff": offset_aff,
            "list_feat": intermediates,
        }


class DySPNHWAlignedModel(nn.Module):
    """DySPN-style model aligned to the current RHB black-box rules."""

    def __init__(self, config: Optional[DySPNHWConfig] = None) -> None:
        super().__init__()
        self.config = config or DySPNHWConfig()
        self.branch_qat_scales = self.config.branch_qat_scales or {}
        self.branch_quant_trace: TensorDict = {}
        self.out_channel = self.config.iteration * self.config.num_neighbor + 1 + 1

        self.conv1_rgb = conv_bn_relu(3, 48, 3, 1, 1)
        self.conv1_dep = conv_bn_relu(1, 16, 3, 1, 1)
        self.conv1 = conv_bn_relu(64, 64, 3, 2, 1)

        net = _build_resnet_layers(
            self.config.network,
            None if self.config.from_scratch else self.config.pretrained_resnet_path,
        )
        self.conv2 = net.layer1
        self.conv3 = net.layer2
        self.conv4 = net.layer3
        self.conv5 = net.layer4

        self.dec5 = ResizeConvBNReLU(512, 256)
        self.dec4 = ResizeConvBNReLU(256 + 256, 128)
        self.dec3 = ResizeConvBNReLU(128 + 128, 64)
        self.dec2 = ResizeConvBNReLU(64 + 64, 64)

        self.gd_dec1 = conv_bn_relu(64 + 64, 128, 3, 1, 1)
        self.gd_dec0 = conv_bn_relu(128, self.out_channel, 3, 1, 1, bn=False, relu=False)
        self.propagation = DySPNHostPropagation(self.config.iteration, self.config.num_neighbor, mode="yx")

    def _frontend_input(self, rgb: torch.Tensor, dep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.config.frontend_rhb_quant:
            return rgb, dep
        return (
            ste_round_i8(rgb, self.config.frontend_rgb_scale, self.config.frontend_quant_semantics),
            ste_round_i8(dep, self.config.frontend_dep_scale, self.config.frontend_quant_semantics),
        )

    def _frontend_boundary(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        if not self.config.frontend_rhb_quant:
            return x
        return ste_round_i8(x, scale, self.config.frontend_quant_semantics)

    def _tail_quant_enabled(self, name: str) -> bool:
        if not self.config.tail_rhb_quant:
            return False
        points = {
            item.strip()
            for item in self.config.tail_quant_points.split(",")
            if item.strip()
        }
        return name in points


    def _tail_boundary(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if not self._tail_quant_enabled(name):
            return x
        scale = float(getattr(self.config, f"tail_{name}_scale"))
        return ste_round_i8(x, scale, self.config.tail_quant_semantics)

    def _branch_boundary(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if self.config.collect_branch_stats:
            self.branch_quant_trace[name] = x.detach()
        if not self.config.branch_contract_qat:
            return x
        scale = self.branch_qat_scales.get(name)
        if scale is None:
            raise KeyError(f"missing branch QAT scale for {name}")
        return ste_round_i8(x, float(scale), self.config.branch_quant_semantics)

    def _run_residual_stage(self, stage: nn.Sequential, x: torch.Tensor, name: str) -> torch.Tensor:
        """Run torchvision BasicBlocks with explicit branch quantization points."""
        for index, block in enumerate(stage):
            prefix = f"{name}.b{index}"
            identity = x

            out = block.conv1(x)
            out = block.bn1(out)
            out = block.relu(out)
            out = self._branch_boundary(f"{prefix}.main1", out)

            out = block.conv2(out)
            out = block.bn2(out)
            out = self._branch_boundary(f"{prefix}.main2", out)

            if block.downsample is not None:
                identity = block.downsample(x)
            identity = self._branch_boundary(f"{prefix}.identity", identity)

            out = self._branch_boundary(f"{prefix}.postadd", out + identity)
            out = block.relu(out)
            x = self._branch_boundary(f"{prefix}.output", out)
        return x

    def _concat(self, fd: torch.Tensor, fe: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if fd.shape[-2:] != fe.shape[-2:]:
            fd = F.interpolate(fd, size=fe.shape[-2:], mode="nearest")
        return torch.cat((fd, fe), dim=dim)

    def _branch_concat(self, name: str, fd: torch.Tensor, fe: torch.Tensor) -> torch.Tensor:
        if fd.shape[-2:] != fe.shape[-2:]:
            fd = F.interpolate(fd, size=fe.shape[-2:], mode="nearest")
        fd = self._branch_boundary(f"{name}.decoder", fd)
        fe = self._branch_boundary(f"{name}.skip", fe)
        return self._branch_boundary(f"{name}.concat", torch.cat((fd, fe), dim=1))

    def encode_decode(self, rgb: torch.Tensor, dep: torch.Tensor) -> TensorDict:
        self.branch_quant_trace = {}
        rgb, dep = self._frontend_input(rgb, dep)
        rgb = self._branch_boundary("input.rgb", rgb)
        dep = self._branch_boundary("input.dep", dep)
        fe1_rgb = self.conv1_rgb(rgb)
        fe1_dep = self.conv1_dep(dep)
        fe1_rgb = self._frontend_boundary(fe1_rgb, self.config.frontend_fe1_scale)
        fe1_dep = self._frontend_boundary(fe1_dep, self.config.frontend_fe1_scale)
        fe1_rgb = self._branch_boundary("stem.rgb", fe1_rgb)
        fe1_dep = self._branch_boundary("stem.dep", fe1_dep)
        fe1 = self._branch_boundary("stem.concat", torch.cat((fe1_rgb, fe1_dep), dim=1))
        fe1_down = self._frontend_boundary(self.conv1(fe1), self.config.frontend_fe1_down_scale)
        fe1_down = self._branch_boundary("stem.down", fe1_down)

        explicit_stages = self.config.branch_contract_qat or self.config.collect_branch_stats
        if explicit_stages:
            fe2 = self._run_residual_stage(self.conv2, fe1_down, "conv2")
            fe3 = self._run_residual_stage(self.conv3, fe2, "conv3")
            fe4 = self._run_residual_stage(self.conv4, fe3, "conv4")
            fe5 = self._run_residual_stage(self.conv5, fe4, "conv5")
        else:
            fe2 = self.conv2(fe1_down)
            fe3 = self.conv3(fe2)
            fe4 = self.conv4(fe3)
            fe5 = self.conv5(fe4)
        fe2 = self._frontend_boundary(fe2, self.config.frontend_fe2_scale)
        fe3 = self._frontend_boundary(fe3, self.config.frontend_fe3_scale)
        fe4 = self._tail_boundary("fe4", fe4)
        fe5 = self._tail_boundary("fe5", fe5)

        fd4 = self._branch_boundary("dec5.output", self._tail_boundary("fd4", self.dec5(fe5)))
        fd3_in = self._branch_concat("dec4", fd4, fe4)
        fd3 = self._branch_boundary("dec4.output", self._tail_boundary("fd3", self.dec4(fd3_in)))
        fd2_in = self._branch_concat("dec3", fd3, fe3)
        fd2 = self._branch_boundary("dec3.output", self._tail_boundary("fd2", self.dec3(fd2_in)))
        fd1_in = self._branch_concat("dec2", fd2, fe2)
        fd1 = self._branch_boundary("dec2.output", self._tail_boundary("fd1", self.dec2(fd1_in)))
        gd_fd1_in = self._branch_concat("gd_dec1", fd1, fe1)
        gd_fd1 = self._branch_boundary(
            "gd_dec1.output", self._tail_boundary("gd_fd1", self.gd_dec1(gd_fd1_in))
        )
        guide = self._branch_boundary("head.guide", self._tail_boundary("guide", self.gd_dec0(gd_fd1)))
        return {
            "guide": guide,
            "gd_fd1": gd_fd1,
            "fd1": fd1,
            "fd2": fd2,
            "fd3": fd3,
            "fd4": fd4,
            "fe1": fe1,
            "fe1_down": fe1_down,
            "fe2": fe2,
            "fe3": fe3,
            "fe4": fe4,
            "fe5": fe5,
        }

    def forward(self, rgb: torch.Tensor, dep: torch.Tensor) -> TensorDict:
        tensors = self.encode_decode(rgb, dep)
        guide = tensors["guide"]
        n = self.config.iteration * self.config.num_neighbor
        prop = self.propagation(
            pred_init=guide[:, n + 1 : n + 2],
            guide=guide[:, :n],
            sparse_depth=dep,
            confidence=guide[:, n : n + 1],
        )
        tensors.update(prop)
        return tensors


ifmap_sz = [(3, 128, 128), (1, 128, 128)]
input_layouts = ["BCHW", "BCHW"]
op_version = 18
batch_size = 1
