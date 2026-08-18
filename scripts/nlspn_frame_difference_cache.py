"""Causal frame-difference cache primitives for frozen NLSPN."""

from __future__ import division

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from scripts import nlspn_temporal_residual as residual


VARIANTS = ("zero_flow", "rgb_diff", "global_diff")
THRESHOLDS = tuple(value / 255.0 for value in (2.0, 4.0, 8.0, 16.0))
DILATION_RADII = (2, 4, 8)
SPARSE_GATE_METERS = 0.02


@dataclass(frozen=True)
class CacheConfig:
    variant: str
    threshold: object
    dilation_radius: object

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError("unknown frame-difference cache variant")
        if self.variant == "zero_flow":
            if self.threshold is not None or self.dilation_radius is not None:
                raise ValueError("zero_flow does not accept mask parameters")
            return
        if self.threshold is None or not 0.0 <= float(self.threshold) <= 1.0:
            raise ValueError("threshold must be within [0, 1]")
        if (self.dilation_radius is None or
                isinstance(self.dilation_radius, bool) or
                int(self.dilation_radius) != self.dilation_radius or
                int(self.dilation_radius) <= 0):
            raise ValueError("dilation radius must be a positive integer")


@dataclass(frozen=True)
class TranslatedState:
    previous_rgb: torch.Tensor
    previous_depth: torch.Tensor
    previous_guidance: torch.Tensor
    previous_confidence: torch.Tensor
    in_bounds: torch.Tensor


def candidate_configs(variant):
    if variant not in VARIANTS:
        raise ValueError("unknown frame-difference cache variant")
    if variant == "zero_flow":
        return (CacheConfig("zero_flow", None, None),)
    return tuple(
        CacheConfig(variant, threshold, radius)
        for threshold in THRESHOLDS
        for radius in DILATION_RADII)


def _validate_bchw(value, name, channels=None):
    if not isinstance(value, torch.Tensor) or value.ndim != 4:
        raise ValueError("%s must be a BCHW tensor" % name)
    if channels is not None and value.shape[1] != channels:
        raise ValueError("%s has an invalid channel count" % name)
    if value.dtype != torch.float32:
        raise ValueError("%s must use float32" % name)
    if not torch.isfinite(value).all():
        raise ValueError("%s contains non-finite values" % name)
    return value


def blurred_rgb_delta(current, previous):
    current = _validate_bchw(current, "current RGB", channels=3)
    previous = _validate_bchw(previous, "previous RGB", channels=3)
    if current.shape != previous.shape:
        raise ValueError("RGB tensors must have identical geometry")
    if (torch.any(current < 0.0) or torch.any(current > 1.0) or
            torch.any(previous < 0.0) or torch.any(previous > 1.0)):
        raise ValueError("RGB tensors must be within [0, 1]")
    current_blur = functional.avg_pool2d(
        current, kernel_size=3, stride=1, padding=1)
    previous_blur = functional.avg_pool2d(
        previous, kernel_size=3, stride=1, padding=1)
    return torch.max(torch.abs(current_blur - previous_blur), dim=1,
                     keepdim=True)[0]


def _dilate(mask, radius):
    if not isinstance(mask, torch.Tensor) or mask.ndim != 4:
        raise ValueError("mask must be BCHW")
    if mask.dtype != torch.bool:
        raise ValueError("mask must be boolean")
    if isinstance(radius, bool) or int(radius) != radius or int(radius) <= 0:
        raise ValueError("dilation radius must be a positive integer")
    radius = int(radius)
    return functional.max_pool2d(
        mask.to(torch.float32), kernel_size=2 * radius + 1,
        stride=1, padding=radius).to(torch.bool)


def photometric_changed(delta, threshold, radius):
    delta = _validate_bchw(delta, "RGB delta", channels=1)
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within [0, 1]")
    return _dilate(delta > threshold, radius)


def sparse_changed(sparse, base, radius):
    sparse = _validate_bchw(sparse, "sparse depth", channels=1)
    base = _validate_bchw(base, "base depth", channels=1)
    if sparse.shape != base.shape:
        raise ValueError("sparse and base depth must have identical geometry")
    if torch.any(sparse < 0.0):
        raise ValueError("sparse depth cannot be negative")
    changed = ((sparse > 0.0) &
               (torch.abs(sparse - base) > SPARSE_GATE_METERS))
    return _dilate(changed, radius)


def compose_stable_mask(photo_changed, depth_changed, out_of_bounds=None):
    if (not isinstance(photo_changed, torch.Tensor) or
            not isinstance(depth_changed, torch.Tensor) or
            photo_changed.dtype != torch.bool or
            depth_changed.dtype != torch.bool or
            photo_changed.shape != depth_changed.shape):
        raise ValueError("changed masks must be same-shape boolean tensors")
    changed = photo_changed | depth_changed
    if out_of_bounds is not None:
        if (not isinstance(out_of_bounds, torch.Tensor) or
                out_of_bounds.dtype != torch.bool or
                out_of_bounds.shape != changed.shape):
            raise ValueError("out-of-bounds mask geometry is invalid")
        changed = changed | out_of_bounds
    return ~changed


def blend_cached_depth(previous, candidate, stable):
    previous = _validate_bchw(previous, "previous depth", channels=1)
    candidate = _validate_bchw(candidate, "candidate depth", channels=1)
    if (previous.shape != candidate.shape or
            not isinstance(stable, torch.Tensor) or
            stable.dtype != torch.bool or stable.shape != previous.shape):
        raise ValueError("cache blend tensors must have identical B1HW geometry")
    return torch.where(stable, previous, candidate)


def estimate_backward_translation(current, previous, downsample=4):
    current = _validate_bchw(current, "current RGB", channels=3)
    previous = _validate_bchw(previous, "previous RGB", channels=3)
    if current.shape != previous.shape or current.shape[0] != 1:
        raise ValueError("phase correlation requires matching batch-one RGB")
    if (isinstance(downsample, bool) or int(downsample) != downsample or
            int(downsample) <= 0):
        raise ValueError("downsample must be a positive integer")
    downsample = int(downsample)
    coefficients = current.new_tensor((0.2989, 0.5870, 0.1140)).reshape(
        1, 3, 1, 1)
    current_gray = torch.sum(current * coefficients, dim=1, keepdim=True)
    previous_gray = torch.sum(previous * coefficients, dim=1, keepdim=True)
    current_small = functional.avg_pool2d(
        current_gray, kernel_size=downsample, stride=downsample)
    previous_small = functional.avg_pool2d(
        previous_gray, kernel_size=downsample, stride=downsample)
    if (float(torch.std(current_small).item()) == 0.0 or
            float(torch.std(previous_small).item()) == 0.0):
        raise ValueError("phase correlation requires nonconstant RGB")
    current_fft = torch.fft.rfft2(current_small[0, 0])
    previous_fft = torch.fft.rfft2(previous_small[0, 0])
    cross_power = previous_fft * torch.conj(current_fft)
    cross_power = cross_power / torch.clamp(torch.abs(cross_power), min=1e-12)
    correlation = torch.fft.irfft2(
        cross_power, s=current_small.shape[-2:])
    peak = int(torch.argmax(correlation).item())
    height, width = current_small.shape[-2:]
    shift_y, shift_x = divmod(peak, width)
    if shift_x > width // 2:
        shift_x -= width
    if shift_y > height // 2:
        shift_y -= height
    return float(shift_x * downsample), float(shift_y * downsample)


def constant_backward_flow(dx, dy, height, width, device, dtype):
    if int(height) <= 0 or int(width) <= 0:
        raise ValueError("flow geometry must be positive")
    if not dtype.is_floating_point:
        raise ValueError("flow dtype must be floating point")
    flow = torch.empty(
        1, 2, int(height), int(width), device=device, dtype=dtype)
    flow[:, 0].fill_(float(dx))
    flow[:, 1].fill_(float(dy))
    if not torch.isfinite(flow).all():
        raise ValueError("flow contains non-finite values")
    return flow


def translation_state(state, flow):
    required = (
        "previous_rgb", "previous_depth", "previous_guidance",
        "previous_confidence")
    if state is None or not all(hasattr(state, name) for name in required):
        raise ValueError("translation requires complete online state")
    tensors = [getattr(state, name) for name in required]
    geometry = tensors[0].shape[0], tensors[0].shape[-2], tensors[0].shape[-1]
    if any(tensor.ndim != 4 or
           (tensor.shape[0], tensor.shape[-2], tensor.shape[-1]) != geometry
           for tensor in tensors):
        raise ValueError("online state geometry is inconsistent")
    if tuple(flow.shape) != (geometry[0], 2, geometry[1], geometry[2]):
        raise ValueError("translation flow geometry is invalid")
    warped = [residual.backward_warp(tensor, flow) for tensor in tensors]
    in_bounds = warped[0][1]
    if any(not torch.equal(item[1], in_bounds) for item in warped[1:]):
        raise RuntimeError("translated state has inconsistent bounds")
    return TranslatedState(
        previous_rgb=warped[0][0],
        previous_depth=warped[1][0],
        previous_guidance=warped[2][0],
        previous_confidence=warped[3][0],
        in_bounds=in_bounds)
