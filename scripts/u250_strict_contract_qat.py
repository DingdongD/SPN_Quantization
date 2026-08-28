#!/usr/bin/env python3
"""Software-only QAT/evaluation for the deployed U250 CSPN contract.

This module deliberately lives outside the existing generic QAT implementation.
It models the protocol exercised by the current U250 strict graph: signed INT8
QDQ at every recurrent boundary, a fixed 14/127 metre depth scale, a 1/127
normalised affinity scale, hard sparse-depth preservation, and 24 recurrent
iterations.  It does not change RTL, the DS compiler, the bitstream, or the
existing QAT worktree.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = Path(os.environ.get("SPN_EXTERNAL_ROOT", str(REPO_ROOT / "external")))
OFFICIAL_ROOT = Path(os.environ.get("CSPN_OFFICIAL_ROOT", "/workspace/CSPN-master/cspn_pytorch"))
# Prefer the worktree's corrected ResNet18/tiny decoder.  The official source
# is only a fallback for files not carried by this isolated worktree.
sys.path.insert(0, str(OFFICIAL_ROOT / "models"))
sys.path.insert(0, str(REPO_ROOT / "models"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

DEPTH_SCALE = 14.0 / 127.0
GATE_SCALE = 1.0 / 127.0
QMIN, QMAX = -127.0, 127.0
NEIGHBOR_PADS = ((0, 2, 0, 2), (1, 1, 0, 2), (2, 0, 0, 2),
                 (0, 2, 1, 1), (2, 0, 1, 1), (0, 2, 2, 0),
                 (1, 1, 2, 0), (2, 0, 2, 0))


def qdq_ste(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Symmetric INT8 fake quantisation with a straight-through gradient."""
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    q = torch.clamp(torch.round(x / scale), QMIN, QMAX) * scale
    return x + (q - x).detach()


def _pad_neighbors(x: torch.Tensor) -> torch.Tensor:
    # NCHW -> N,8,C,H+2,W+2. This is the same order as cspn.py.
    return torch.cat([F.pad(x, pad).unsqueeze(1) for pad in NEIGHBOR_PADS], dim=1)


def _pad_guidance(x: torch.Tensor) -> torch.Tensor:
    """Pad guidance channel i into physical neighbour slot i."""
    return torch.cat([
        F.pad(x[:, i:i + 1], pad).unsqueeze(1)
        for i, pad in enumerate(NEIGHBOR_PADS)
    ], dim=1)


class U250StrictPropagation(nn.Module):
    """Differentiable simulator of the fixed U250 K24 propagation protocol."""

    def __init__(self, steps: int = 24, depth_scale: float = DEPTH_SCALE,
                 gate_scale: float = GATE_SCALE):
        super().__init__()
        if steps <= 0:
            raise ValueError("steps must be positive")
        self.steps = int(steps)
        self.depth_scale = float(depth_scale)
        self.gate_scale = float(gate_scale)

    def _normalise_gate(self, guidance: torch.Tensor):
        # Build the physical 8-neighbour layout, perform the external abs/sum
        # normalisation, and quantise both the gate and gate-sum boundaries.
        gate = _pad_guidance(guidance)
        denom = gate.abs().sum(dim=1, keepdim=True).clamp_min(1e-6)
        gate = gate / denom
        gate = qdq_ste(gate, self.gate_scale)
        gate_sum = gate.sum(dim=1, keepdim=False)[:, :, 1:-1, 1:-1]
        gate_sum = qdq_ste(gate_sum, self.gate_scale)
        return gate, gate_sum

    def forward(self, guidance: torch.Tensor, raw_depth: torch.Tensor,
                sparse_depth: torch.Tensor | None = None) -> torch.Tensor:
        if guidance.ndim != 4 or guidance.shape[1] != 8:
            raise ValueError("guidance must be N,8,H,W")
        if raw_depth.ndim != 4 or raw_depth.shape[1] != 1:
            raise ValueError("raw_depth must be N,1,H,W")
        if guidance.shape[0] != raw_depth.shape[0] or guidance.shape[-2:] != raw_depth.shape[-2:]:
            raise ValueError("guidance/raw_depth shapes do not match")
        gate, gate_sum = self._normalise_gate(guidance)
        raw = qdq_ste(raw_depth, self.depth_scale)
        state = raw
        mask = None
        if sparse_depth is not None:
            if sparse_depth.shape != raw.shape:
                raise ValueError("sparse_depth/raw_depth shapes do not match")
            mask = (sparse_depth.abs() > 1e-4).to(raw.dtype)
        for _ in range(self.steps):
            neighbors = qdq_ste(_pad_neighbors(state), self.depth_scale)
            weighted = qdq_ste(gate * neighbors, self.depth_scale)
            neighbor_sum = qdq_ste(weighted.sum(dim=1)[:, :, 1:-1, 1:-1], self.depth_scale)
            gate_raw = qdq_ste(gate_sum * raw, self.depth_scale)
            state = qdq_ste(raw + neighbor_sum - gate_raw, self.depth_scale)
            if mask is not None:
                state = mask * raw + (1.0 - mask) * state
                state = qdq_ste(state, self.depth_scale)
        return state


class StrictH5Dataset(Dataset):
    """Reads the remote 128-sample train/32-sample val NYU subsets."""

    def __init__(self, manifest: str | Path, root: str | Path, train: bool,
                 n_sample: int = 500, seed: int = 123):
        payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.root = Path(root)
        self.train = bool(train)
        self.n_sample = int(n_sample)
        self.seed = int(seed)
        self.files = [item["filename"] for item in payload["samples"]]

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _to_tensor(arr):
        return torch.from_numpy(np.asarray(arr, dtype=np.float32).copy())

    def __getitem__(self, index):
        with h5py.File(str(self.root / self.files[index]), "r") as f:
            rgb = f["rgb"][:].transpose(1, 2, 0)
            depth = f["depth"][:]
        rgb_img = Image.fromarray(rgb, mode="RGB")
        depth_img = Image.fromarray(depth.astype("float32"), mode="F")
        if self.train:
            # Keep augmentation modest and deterministic enough for a small
            # feasibility run; the geometry matches the official CSPN loader.
            scale = 1.2
            size = int(240 * scale)
            rgb_img = TF.resize(rgb_img, size, interpolation=transforms.InterpolationMode.BILINEAR)
            depth_img = TF.resize(depth_img, size, interpolation=transforms.InterpolationMode.NEAREST)
            rgb_img = TF.center_crop(rgb_img, (228, 304))
            depth_img = TF.center_crop(depth_img, (228, 304))
            rgb_img = transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15)(rgb_img)
            rgb = TF.to_tensor(rgb_img)
            depth = self._to_tensor(np.array(depth_img)).unsqueeze(0) / scale
        else:
            rgb_img = TF.resize(rgb_img, 240, interpolation=transforms.InterpolationMode.BILINEAR)
            depth_img = TF.resize(depth_img, 240, interpolation=transforms.InterpolationMode.NEAREST)
            rgb = TF.to_tensor(TF.center_crop(rgb_img, (228, 304)))
            depth = self._to_tensor(np.array(TF.center_crop(depth_img, (228, 304)))).unsqueeze(0)
        mean = rgb.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        std = rgb.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
        rgb = (rgb - mean) / std
        generator = torch.Generator().manual_seed(self.seed + int(index))
        valid = torch.nonzero((depth > 1e-4).flatten(), as_tuple=False).flatten()
        count = min(self.n_sample, int(valid.numel()))
        sparse = torch.zeros_like(depth)
        if count:
            selected = valid[torch.randperm(valid.numel(), generator=generator)[:count]]
            sparse.flatten()[selected] = depth.flatten()[selected]
        return torch.cat((rgb, sparse), dim=0), depth


def build_model(checkpoint: str | Path, device: torch.device):
    import torch_resnet_cspn_nyu as cspn_model
    model = cspn_model.resnet18(pretrained=False, cspn_config={"step": 24, "kernel": 3, "norm_type": "8sum"})
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = payload.get("net", payload)
    state = dict(state)
    # The official CSPN checkpoint stores a fixed all-ones Conv3D reduction;
    # strict propagation computes the reduction explicitly, so it is not a
    # model parameter in this isolated simulator.
    state.pop("post_process_layer.sum_conv.weight", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The generic QAT checkpoint intentionally omits the dynamically created
    # sum_conv. It is irrelevant because strict propagation replaces it.
    if unexpected or any(not key.startswith("post_process_layer.sum_conv") for key in missing):
        raise RuntimeError(f"checkpoint mismatch missing={missing} unexpected={unexpected}")
    model.post_process_layer = U250StrictPropagation()
    return model.to(device), {"missing": list(missing), "unexpected": list(unexpected)}


def metric(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    valid = gt > 1e-4
    diff = (pred - gt)[valid]
    mse = float(diff.square().mean().item())
    return {"RMSE": math.sqrt(mse), "MAE": float(diff.abs().mean().item()),
            "valid": int(diff.numel())}


def evaluate(model, loader, device):
    model.eval()
    sums = {"sq": 0.0, "abs": 0.0, "valid": 0}
    with torch.no_grad():
        for x, gt in loader:
            x, gt = x.to(device), gt.to(device)
            pred = model(x)
            valid = gt > 1e-4
            diff = (pred - gt)[valid]
            sums["sq"] += float(diff.square().sum().item())
            sums["abs"] += float(diff.abs().sum().item())
            sums["valid"] += int(diff.numel())
    return {"RMSE": math.sqrt(sums["sq"] / sums["valid"]),
            "MAE": sums["abs"] / sums["valid"], "valid": sums["valid"]}


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    model, load_report = build_model(args.checkpoint, device)
    train_ds = StrictH5Dataset(args.train_manifest, args.train_root, True, args.n_sample, args.seed)
    val_ds = StrictH5Dataset(args.val_manifest, args.val_root, False, args.n_sample, args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    baseline = evaluate(model, val_loader, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, gt in train_loader:
            x, gt = x.to(device), gt.to(device)
            pred = model(x)
            valid = gt > 1e-4
            loss = (pred[valid] - gt[valid]).abs().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
        current = evaluate(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_l1": total / max(1, len(train_loader)), **current}
        history.append(row)
        print(json.dumps(row), flush=True)
    final = history[-1] if history else baseline
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"net": model.state_dict(), "contract": {
        "name": "U250 strict K24 software contract",
        "steps": 24, "depth_scale": DEPTH_SCALE, "gate_scale": GATE_SCALE,
        "state_qdq_each_stage": True, "sparse_depth_hard_preserve": True,
    }, "baseline": baseline, "history": history}, output / "strict_contract_qat.pt")
    report = {"checkpoint": str(args.checkpoint), "load_report": load_report,
              "contract": {"steps": 24, "depth_scale": DEPTH_SCALE, "gate_scale": GATE_SCALE,
                           "input_shape": [1, 4, 228, 304], "mode": "software-only; no U250 DMA"},
              "baseline_val": baseline, "final_val": final, "history": history,
              "dataset": {"train": len(train_ds), "val": len(val_ds), "n_sample": args.n_sample},
              "note": "This worktree is isolated from the existing generic QAT worktree."}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--train-root", required=True)
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--val-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--n-sample", type=int, default=500)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=123)
    run(p.parse_args())


if __name__ == "__main__":
    main()
