import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=True)

    def forward(self, x):
        return F.relu(self.conv(x))


class Downsample1x1ReLU(nn.Module):
    """Hardware-aligned replacement for small-spatial stride-2 ConvReLU.

    It intentionally changes the model family and should be used with
    retraining. The Host/runtime performs a fixed even-grid sample through the
    tensor slice; RHB only needs to execute a 1x1 Conv + ReLU at 8x8.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        return F.relu(self.proj(x[:, :, ::2, ::2]))


class FastAffinityPropagate(nn.Module):
    """Device-agnostic CSPN propagation implemented with tensor ops.

    The reference CSPN implementation creates a Conv3d and a CUDA weight tensor
    inside every forward call. That works for single-GPU smoke runs, but it adds
    repeated module allocation overhead and hard-codes CUDA. This version keeps
    the same 8-neighbor math while using reductions on the current tensor device.
    """

    def __init__(self, prop_time, prop_kernel, norm_type="8sum"):
        super().__init__()
        if prop_kernel != 3:
            raise ValueError("only 3x3 CSPN propagation is supported")
        if norm_type not in {"8sum", "8sum_abs"}:
            raise ValueError(f"unsupported cspn norm_type: {norm_type}")
        self.prop_time = prop_time
        self.prop_kernel = prop_kernel
        self.norm_type = norm_type

    @staticmethod
    def _pad_chunks(chunks):
        pads = [
            (0, 2, 0, 2),
            (1, 1, 0, 2),
            (2, 0, 0, 2),
            (0, 2, 1, 1),
            (2, 0, 1, 1),
            (0, 2, 2, 0),
            (1, 1, 2, 0),
            (2, 0, 2, 0),
        ]
        return torch.cat([F.pad(chunk, pad).unsqueeze(1) for chunk, pad in zip(chunks, pads)], dim=1)

    @classmethod
    def _pad_neighbors(cls, x):
        return cls._pad_chunks([x] * 8)

    def affinity_normalization(self, guidance):
        if "abs" in self.norm_type:
            guidance = torch.abs(guidance)
        chunks = torch.chunk(guidance, 8, dim=1)
        gates = self._pad_chunks(chunks)
        abs_weight = torch.sum(torch.abs(gates), dim=1, keepdim=True).clamp_min(1.0e-6)
        gates = gates / abs_weight
        gate_sum = torch.sum(gates, dim=1)[:, :, 1:-1, 1:-1]
        return gates, gate_sum

    def forward(self, guidance, blur_depth, sparse_depth=None):
        gate_wb, gate_sum = self.affinity_normalization(guidance)
        raw_depth_input = blur_depth
        result_depth = blur_depth
        sparse_mask = sparse_depth.sign() if sparse_depth is not None else None

        for _ in range(self.prop_time):
            result_neighbors = self._pad_neighbors(result_depth)
            result_depth = torch.sum(gate_wb * result_neighbors, dim=1)[:, :, 1:-1, 1:-1]
            result_depth = (1.0 - gate_sum) * raw_depth_input + result_depth
            if sparse_mask is not None:
                result_depth = (1.0 - sparse_mask) * result_depth + sparse_mask * raw_depth_input
        return result_depth


class AlignedBackboneHeads(nn.Module):
    def __init__(self, base_ch=8, down20_variant="stride3x3"):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        self.stem0 = ConvReLU(4, c1, 3, 1)
        self.stem1 = ConvReLU(c1, c1, 3, 1)
        self.down10 = ConvReLU(c1, c2, 3, 2)
        self.down11 = ConvReLU(c2, c2, 3, 1)
        if down20_variant == "sample_1x1":
            self.down20 = Downsample1x1ReLU(c2, c3)
        elif down20_variant == "stride3x3":
            self.down20 = ConvReLU(c2, c3, 3, 2)
        else:
            raise ValueError(f"unsupported down20_variant: {down20_variant}")
        self.down21 = ConvReLU(c3, c3, 3, 1)
        self.down22 = ConvReLU(c3, c3, 3, 1)

        self.up10 = ConvReLU(c3 + c2, c2, 3, 1)
        self.up11 = ConvReLU(c2, c2, 3, 1)
        self.up20 = ConvReLU(c2 + c1, c1, 3, 1)
        self.up21 = ConvReLU(c1, c1, 3, 1)

        self.depth0 = ConvReLU(c1, c1, 3, 1)
        self.depth1 = nn.Conv2d(c1, 1, kernel_size=3, padding=1, bias=True)
        self.aff0 = ConvReLU(c1, c1, 3, 1)
        self.aff1 = nn.Conv2d(c1, 8, kernel_size=3, padding=1, bias=True)
        self._init_weights()

    def _init_weights(self):
        for op in self.modules():
            if isinstance(op, nn.Conv2d):
                nn.init.xavier_uniform_(op.weight, gain=0.1)
                if op.bias is not None:
                    nn.init.uniform_(op.bias, -0.1, 0.1)

    def forward(self, x):
        s1 = self.stem1(self.stem0(x))
        s2 = self.down11(self.down10(s1))
        b = self.down22(self.down21(self.down20(s2)))

        u1 = F.interpolate(b, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        u1 = torch.cat([u1, s2], dim=1)
        u1 = self.up11(self.up10(u1))

        u2 = F.interpolate(u1, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        u2 = torch.cat([u2, s1], dim=1)
        feat = self.up21(self.up20(u2))

        raw_depth = self.depth1(self.depth0(feat))
        raw_guidance = self.aff1(self.aff0(feat))
        return raw_depth, raw_guidance


class AlignedCSPNHW(nn.Module):
    def __init__(self, base_ch=8, cspn_step=4, cspn_norm_type="8sum", down20_variant="stride3x3"):
        super().__init__()
        self.backbone = AlignedBackboneHeads(base_ch=base_ch, down20_variant=down20_variant)
        self.cspn = FastAffinityPropagate(cspn_step, 3, norm_type=cspn_norm_type)

    def forward_heads(self, x):
        return self.backbone(x)

    def forward(self, x):
        sparse_depth = x[:, 3:4]
        raw_depth, raw_guidance = self.forward_heads(x)
        depth = self.cspn(raw_guidance, raw_depth, sparse_depth)
        return depth
