"""Pure-memory online GOP2 engine for frozen NLSPN."""

from __future__ import division

from dataclasses import dataclass
import time

import numpy as np
import torch

from scripts import nlspn_temporal_residual as residual
from scripts import raft_small_compat


@dataclass
class OnlineState:
    previous_rgb: torch.Tensor
    previous_depth: torch.Tensor
    previous_guidance: torch.Tensor
    previous_confidence: torch.Tensor
    local_index: int


@dataclass
class FrameResult:
    prediction: torch.Tensor
    latency_ms: float
    kind: str


def latency_summary(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("latency values must be finite and non-empty")
    if np.any(values < 0.0):
        raise ValueError("latency values cannot be negative")
    mean = float(np.mean(values))
    if mean <= 0.0:
        raise ValueError("mean latency must be positive")
    return {
        "count": int(values.size),
        "total_ms": float(np.sum(values)),
        "mean_ms": mean,
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "fps": float(1000.0 / mean),
    }


def _quality_terms(full, gop2, gt, valid):
    full = np.asarray(full, dtype=np.float64)
    gop2 = np.asarray(gop2, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if not (full.shape == gop2.shape == gt.shape == valid.shape):
        raise ValueError("quality arrays must have identical shapes")
    if not np.any(valid):
        raise ValueError("quality mask is empty")
    if not all(np.isfinite(value[valid]).all()
               for value in (full, gop2, gt)):
        raise ValueError("quality arrays contain non-finite values")
    full_error = full[valid] - gt[valid]
    gop2_error = gop2[valid] - gt[valid]
    return (
        float(np.sum(full_error ** 2)),
        float(np.sum(gop2_error ** 2)),
        int(np.count_nonzero(valid)),
    )


def _quality_from_terms(full_squared_error, gop2_squared_error,
                        valid_pixels):
    valid_pixels = int(valid_pixels)
    if valid_pixels <= 0:
        raise ValueError("valid pixel count must be positive")
    rmse_full = float(np.sqrt(full_squared_error / valid_pixels))
    if rmse_full <= 0.0:
        raise ValueError("full reference RMSE must be positive")
    rmse_gop2 = float(np.sqrt(gop2_squared_error / valid_pixels))
    ratio = float(rmse_gop2 / rmse_full)
    return {
        "rmse_full": rmse_full,
        "rmse_gop2": rmse_gop2,
        "quality_ratio": ratio,
        "passes": bool(ratio <= 1.01),
        "full_squared_error": float(full_squared_error),
        "gop2_squared_error": float(gop2_squared_error),
        "valid_pixels": valid_pixels,
    }


def pooled_quality_summary(full, gop2, gt, valid):
    return _quality_from_terms(*_quality_terms(full, gop2, gt, valid))


def frame_quality_rows(frame_ids, clips, full, gop2, gt, valid):
    frame_ids = np.asarray(frame_ids).reshape(-1)
    clips = np.asarray(clips).reshape(-1)
    full = np.asarray(full)
    gop2 = np.asarray(gop2)
    gt = np.asarray(gt)
    valid = np.asarray(valid)
    if not (full.ndim == 3 and full.shape == gop2.shape == gt.shape ==
            valid.shape and full.shape[0] == frame_ids.size == clips.size):
        raise ValueError("frame quality inputs are incompatible")
    rows = []
    for index, frame_id in enumerate(frame_ids):
        full_sse, gop2_sse, count = _quality_terms(
            full[index], gop2[index], gt[index], valid[index])
        quality = _quality_from_terms(full_sse, gop2_sse, count)
        rows.append({
            "frame_id": int(frame_id),
            "clip": str(clips[index]),
            "frame_kind": frame_kind(index),
            "rmse_full": quality["rmse_full"],
            "rmse_gop2": quality["rmse_gop2"],
            "quality_ratio": quality["quality_ratio"],
            "passes_1pct": quality["passes"],
            "full_squared_error": full_sse,
            "gop2_squared_error": gop2_sse,
            "valid_pixels": count,
        })
    return rows


def clip_quality_rows(frame_rows):
    frame_rows = list(frame_rows)
    if not frame_rows:
        raise ValueError("frame rows cannot be empty")
    order = []
    grouped = {}
    for row in frame_rows:
        clip = str(row["clip"])
        if clip not in grouped:
            order.append(clip)
            grouped[clip] = [0.0, 0.0, 0, 0]
        values = grouped[clip]
        values[0] += float(row["full_squared_error"])
        values[1] += float(row["gop2_squared_error"])
        values[2] += int(row["valid_pixels"])
        values[3] += 1
    result = []
    for clip in order:
        full_sse, gop2_sse, count, frames = grouped[clip]
        quality = _quality_from_terms(full_sse, gop2_sse, count)
        result.append({
            "clip": clip,
            "frame_count": frames,
            "rmse_full": quality["rmse_full"],
            "rmse_gop2": quality["rmse_gop2"],
            "quality_ratio": quality["quality_ratio"],
            "passes_1pct": quality["passes"],
            "valid_pixels": count,
        })
    return result


def benchmark_summary(full_latencies, gop2_latencies, gop2_kinds, quality):
    gop2_latencies = list(gop2_latencies)
    gop2_kinds = list(gop2_kinds)
    if len(gop2_latencies) != len(gop2_kinds):
        raise ValueError("GOP2 latency and kind counts differ")
    if not gop2_kinds or any(kind not in ("I", "P")
                             for kind in gop2_kinds):
        raise ValueError("GOP2 frame kinds are invalid")
    full_summary = latency_summary(full_latencies)
    gop2_summary = latency_summary(gop2_latencies)
    i_summary = latency_summary([
        value for value, kind in zip(gop2_latencies, gop2_kinds)
        if kind == "I"])
    p_summary = latency_summary([
        value for value, kind in zip(gop2_latencies, gop2_kinds)
        if kind == "P"])
    return {
        "latency": {
            "full": full_summary,
            "gop2": gop2_summary,
            "i": i_summary,
            "p": p_summary,
        },
        "speedup": float(
            full_summary["total_ms"] / gop2_summary["total_ms"]),
        "quality": dict(quality),
    }


def frame_kind(local_index):
    if isinstance(local_index, bool) or int(local_index) != local_index:
        raise ValueError("local frame index must be an integer")
    local_index = int(local_index)
    if local_index < 0:
        raise ValueError("local frame index must be non-negative")
    return "I" if local_index % 2 == 0 else "P"


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _as_cpu_tensor(value, name):
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    elif isinstance(value, torch.Tensor):
        tensor = value
    else:
        raise TypeError("%s must be a NumPy array or Torch tensor" % name)
    if tensor.device.type != "cpu":
        raise ValueError("%s must reside in CPU memory" % name)
    return tensor


def _validate_inputs(rgb, sparse):
    rgb = _as_cpu_tensor(rgb, "RGB")
    sparse = _as_cpu_tensor(sparse, "sparse depth")
    if tuple(rgb.shape) != (3, residual.HEIGHT, residual.WIDTH):
        raise ValueError("RGB must have shape [3, 228, 304]")
    if tuple(sparse.shape) != (residual.HEIGHT, residual.WIDTH):
        raise ValueError("sparse depth must have shape [228, 304]")
    if not torch.isfinite(rgb).all() or not torch.isfinite(sparse).all():
        raise ValueError("online input contains non-finite values")
    if torch.any(sparse < 0):
        raise ValueError("sparse depth cannot be negative")
    if int(torch.count_nonzero(sparse > 0).item()) != residual.SPARSE_COUNT:
        raise ValueError("sparse depth must contain exactly 500 points")
    return rgb, sparse


class InMemoryGOP2Engine(object):
    def __init__(self, nlspn, raft, device):
        self.nlspn = nlspn
        self.raft = raft
        self.device = torch.device(device)
        self.state = None

    def reset(self):
        self.state = None

    def infer_full(self, rgb_cpu, sparse_cpu):
        rgb_cpu, sparse_cpu = _validate_inputs(rgb_cpu, sparse_cpu)
        _synchronize(self.device)
        started = time.perf_counter()
        with torch.no_grad():
            current_rgb = rgb_cpu[None].to(
                device=self.device, dtype=torch.float32)
            current_sparse = sparse_cpu[None, None].to(
                device=self.device, dtype=torch.float32)
            output = self.nlspn({
                "rgb": current_rgb,
                "dep": current_sparse,
            })
            if not isinstance(output, dict) or "pred" not in output:
                raise ValueError("NLSPN output is missing prediction")
            prediction = output["pred"]
            if tuple(prediction.shape) != (
                    1, 1, residual.HEIGHT, residual.WIDTH):
                raise ValueError("NLSPN prediction shape is invalid")
            if not torch.isfinite(prediction).all():
                raise ValueError("NLSPN prediction contains non-finite values")
            prediction_cpu = prediction.detach().to("cpu")[0, 0]
        _synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return FrameResult(prediction_cpu, latency_ms, "FULL")

    def infer_i(self, rgb_cpu, sparse_cpu, local_index):
        if frame_kind(local_index) != "I":
            raise ValueError("infer_i requires an I-frame local index")
        rgb_cpu, sparse_cpu = _validate_inputs(rgb_cpu, sparse_cpu)
        _synchronize(self.device)
        started = time.perf_counter()
        with torch.no_grad():
            current_rgb = rgb_cpu[None].to(
                device=self.device, dtype=torch.float32)
            current_sparse = sparse_cpu[None, None].to(
                device=self.device, dtype=torch.float32)
            output = self.nlspn({
                "rgb": current_rgb,
                "dep": current_sparse,
            })
            required = ("pred", "guidance", "confidence")
            if not isinstance(output, dict) or not all(
                    key in output for key in required):
                raise ValueError("NLSPN output is missing online state fields")
            prediction = output["pred"]
            if tuple(prediction.shape) != (
                    1, 1, residual.HEIGHT, residual.WIDTH):
                raise ValueError("NLSPN prediction shape is invalid")
            if not all(torch.isfinite(output[key]).all() for key in required):
                raise ValueError("NLSPN output contains non-finite values")
            self.state = OnlineState(
                previous_rgb=current_rgb.detach(),
                previous_depth=prediction.detach(),
                previous_guidance=output["guidance"].detach(),
                previous_confidence=output["confidence"].detach(),
                local_index=int(local_index),
            )
            prediction_cpu = prediction.detach().to("cpu")[0, 0]
        _synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return FrameResult(prediction_cpu, latency_ms, "I")

    def infer_p(self, rgb_cpu, sparse_cpu, local_index):
        if frame_kind(local_index) != "P":
            raise ValueError("infer_p requires a P-frame local index")
        if self.state is None:
            raise RuntimeError("P frame requires live temporal state")
        if self.state.local_index != int(local_index) - 1:
            raise RuntimeError("P frame temporal state is stale or misordered")
        rgb_cpu, sparse_cpu = _validate_inputs(rgb_cpu, sparse_cpu)
        _synchronize(self.device)
        started = time.perf_counter()
        with torch.no_grad():
            current_rgb = rgb_cpu[None].to(
                device=self.device, dtype=torch.float32)
            current_sparse = sparse_cpu[None, None].to(
                device=self.device, dtype=torch.float32)
            flow = raft_small_compat.predict_backward_flow(
                self.raft, current_rgb, self.state.previous_rgb)
            base, _ = residual.backward_warp(
                self.state.previous_depth, flow)
            guidance, _ = residual.backward_warp(
                self.state.previous_guidance, flow)
            confidence, _ = residual.backward_warp(
                self.state.previous_confidence, flow)
            seed = torch.zeros_like(base)
            sparse_mask = current_sparse > 0.0
            seed[sparse_mask] = (
                current_sparse[sparse_mask] - base[sparse_mask])
            dense = self.nlspn.prop_layer(
                seed, guidance, confidence, None, current_rgb)[0]
            prediction = torch.clamp(
                base + dense, min=0.0, max=residual.MAX_DEPTH)
            values = (flow, base, guidance, confidence, dense, prediction)
            if not all(torch.isfinite(value).all() for value in values):
                raise ValueError("P-frame inference contains non-finite values")
            self.state = OnlineState(
                previous_rgb=current_rgb.detach(),
                previous_depth=prediction.detach(),
                previous_guidance=guidance.detach(),
                previous_confidence=confidence.detach(),
                local_index=int(local_index),
            )
            prediction_cpu = prediction.detach().to("cpu")[0, 0]
        _synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return FrameResult(prediction_cpu, latency_ms, "P")
