#!/usr/bin/env python3
"""Streaming activation and QDQ histogram statistics."""

from __future__ import division

import math

import numpy as np
import torch

from scripts.activation_outlier_analysis import BoundedActivationSampler


def _finite_tensor(name, tensor):
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("%s contains nonfinite values" % name)


def _safe_ratio(numerator, denominator):
    if denominator > 0.0:
        return float(numerator) / float(denominator)
    if numerator > 0.0:
        return float("inf")
    return 1.0


def quantizer_descriptor(quantizer, channel_dim):
    scale = torch.as_tensor(quantizer.scale).detach().double().cpu()
    if scale.numel() == 0:
        raise ValueError("quantizer scale is empty")
    _finite_tensor("quantizer scale", scale)
    scale_values = scale.reshape(-1).numpy().astype(np.float64)
    return {
        "format": quantizer.format,
        "bits": int(quantizer.bits),
        "unsigned": bool(quantizer.unsigned),
        "qmin": int(quantizer.qmin),
        "qmax": int(quantizer.qmax),
        "zero_point": int(quantizer.zero_point),
        "channel_dim": int(channel_dim),
        "granularity": "per_tensor" if scale_values.size == 1 else
            "per_channel",
        "scale": scale_values,
    }


class RangeCollector(object):
    """Collect exact QDQ statistics and bounded activation magnitudes."""

    def __init__(self, capacity, per_update):
        self.sampler = BoundedActivationSampler(
            capacity=int(capacity), per_update=int(per_update))
        self.elements = 0
        self.updates = 0
        self.reference_zeros = 0
        self.zero_codes = 0
        self.saturated_values = 0
        self.endpoint_codes = 0
        self.signal_energy = 0.0
        self.error_energy = 0.0
        self.reference_minimum = float("inf")
        self.reference_maximum = float("-inf")
        self.error_minimum = float("inf")
        self.error_maximum = float("-inf")
        self.quantizer = None
        self.channel_dim = None

    @property
    def channel_absmax(self):
        if self.sampler.channel_absmax is None:
            raise RuntimeError("range collector has no channel observations")
        return self.sampler.channel_absmax

    def _validate_quantizer(self, quantizer):
        if quantizer.format != "uniform":
            raise ValueError(
                "activation histograms require uniform quantizers")
        if self.quantizer is None:
            self.quantizer = quantizer
            return
        if quantizer is not self.quantizer:
            raise ValueError("quantizer changed during range collection")

    def update(self, reference, quantized, codes, quantizer, channel_dim):
        if reference.shape != quantized.shape or reference.shape != codes.shape:
            raise ValueError("reference, quantized, and codes must have one shape")
        _finite_tensor("reference", reference)
        _finite_tensor("quantized", quantized)
        self._validate_quantizer(quantizer)
        channel_dim = int(channel_dim)
        if self.channel_dim is None:
            self.channel_dim = channel_dim
        elif self.channel_dim != channel_dim:
            raise ValueError("channel dimension changed during collection")

        detached = reference.detach()
        reconstructed = quantized.detach()
        integer_codes = codes.detach()
        difference = detached.double() - reconstructed.double()
        scale = quantizer.scale_for(detached)
        _finite_tensor("quantizer scale", torch.as_tensor(scale))
        normalized = detached / scale

        self.elements += int(detached.numel())
        self.updates += 1
        self.reference_zeros += int((detached == 0).sum().item())
        self.zero_codes += int((integer_codes == 0).sum().item())
        self.saturated_values += int(
            ((normalized < quantizer.qmin) |
             (normalized > quantizer.qmax)).sum().item())
        self.endpoint_codes += int(
            ((integer_codes == quantizer.qmin) |
             (integer_codes == quantizer.qmax)).sum().item())
        self.signal_energy += float((detached.double() ** 2).sum().item())
        self.error_energy += float((difference ** 2).sum().item())
        self.reference_minimum = min(
            self.reference_minimum, float(detached.min().item()))
        self.reference_maximum = max(
            self.reference_maximum, float(detached.max().item()))
        self.error_minimum = min(
            self.error_minimum, float(difference.min().item()))
        self.error_maximum = max(
            self.error_maximum, float(difference.max().item()))
        self.sampler.update(detached, channel_axis=channel_dim)

    def _scale_values(self):
        if self.quantizer is None:
            raise RuntimeError("range collector has no quantizer")
        scale = torch.as_tensor(self.quantizer.scale).detach().double().cpu()
        if scale.numel() == 0:
            raise RuntimeError("quantizer scale is empty")
        if not bool(torch.isfinite(scale).all().item()):
            raise ValueError("quantizer scale contains nonfinite values")
        return scale.reshape(-1)

    def _magnitude_normalizer(self, sampled):
        if sampled["p99"] > 0.0:
            return float(sampled["p99"]), "p99"

        values = self.sampler.values()
        positive = values[values > 0]
        if positive.numel() > 0:
            return float(positive.min().item()), "minimum_positive_sample"

        channel = self.channel_absmax
        positive = channel[channel > 0]
        if positive.numel() > 0:
            return float(positive.min().item()), "minimum_positive_channel"

        return float(self._scale_values().min().item()), \
            "quantizer_scale_all_zero"

    def summary(self):
        if self.elements == 0:
            raise RuntimeError("range collector has no observations")
        sampled = self.sampler.statistics()
        channel = self.channel_absmax
        channel_median = float(torch.quantile(channel, 0.5).item())
        magnitude_normalizer, magnitude_normalizer_kind = \
            self._magnitude_normalizer(sampled)
        sqnr_db = float("inf") if self.error_energy == 0.0 else \
            10.0 * math.log10(self.signal_energy / self.error_energy) \
            if self.signal_energy > 0.0 else float("-inf")
        return {
            "elements": self.elements,
            "updates": self.updates,
            "reference_zeros": self.reference_zeros,
            "zero_codes": self.zero_codes,
            "saturated_values": self.saturated_values,
            "endpoint_codes": self.endpoint_codes,
            "signal_energy": self.signal_energy,
            "error_energy": self.error_energy,
            "sqnr_db": sqnr_db,
            "reference_minimum": self.reference_minimum,
            "reference_maximum": self.reference_maximum,
            "error_minimum": self.error_minimum,
            "error_maximum": self.error_maximum,
            "p75": sampled["p75"],
            "p90": sampled["p90"],
            "p99": sampled["p99"],
            "p99_9": sampled["p99_9"],
            "p99_99": sampled["p99_99"],
            "maximum_abs": sampled["maximum"],
            "magnitude_normalizer": magnitude_normalizer,
            "magnitude_normalizer_kind": magnitude_normalizer_kind,
            "p99_9_over_p99": _safe_ratio(
                sampled["p99_9"], sampled["p99"]),
            "p99_99_over_p99": _safe_ratio(
                sampled["p99_99"], sampled["p99"]),
            "max_over_p99_99": _safe_ratio(
                sampled["maximum"], sampled["p99_99"]),
            "channel_max_over_median": _safe_ratio(
                float(channel.max().item()), channel_median),
            "reference_zero_ratio": self.reference_zeros /
                float(self.elements),
            "zero_code_ratio": self.zero_codes / float(self.elements),
            "saturation_ratio": self.saturated_values /
                float(self.elements),
            "endpoint_code_ratio": self.endpoint_codes /
                float(self.elements),
        }

    def signed_edges(self, bin_count):
        scales = self._scale_values()
        minimum = float((scales * self.quantizer.qmin).min().item())
        maximum = float((scales * self.quantizer.qmax).max().item())
        if minimum == maximum:
            raise ValueError("signed histogram range is degenerate")
        return np.linspace(minimum, maximum, int(bin_count) + 1,
                           dtype=np.float64)

    def magnitude_edges(self, bin_count):
        row = self.summary()
        return float(row["magnitude_normalizer"]) * np.power(
            2.0, np.linspace(-16.0, 8.0, int(bin_count) + 1,
                             dtype=np.float64))

    def error_edges(self, bin_count):
        extent = max(abs(self.error_minimum), abs(self.error_maximum))
        if extent == 0.0:
            extent = float(self._scale_values().min().item()) * 0.5
        return np.linspace(-extent, extent, int(bin_count) + 1,
                           dtype=np.float64)


class HistogramAccumulator(object):
    """Accumulate fixed-bin activation and QDQ histograms on the source device."""

    def __init__(self, signed_edges, magnitude_edges, error_edges, qmin, qmax):
        self.signed_edges = np.asarray(signed_edges, dtype=np.float64)
        self.magnitude_edges = np.asarray(magnitude_edges, dtype=np.float64)
        self.error_edges = np.asarray(error_edges, dtype=np.float64)
        self.qmin = int(qmin)
        self.qmax = int(qmax)
        self.reference_counts = np.zeros(
            self.signed_edges.size - 1, dtype=np.int64)
        self.magnitude_counts = np.zeros(
            self.magnitude_edges.size - 1, dtype=np.int64)
        self.error_counts = np.zeros(
            self.error_edges.size - 1, dtype=np.int64)
        self.code_counts = np.zeros(
            self.qmax - self.qmin + 1, dtype=np.int64)
        self.reference_underflow = 0
        self.reference_overflow = 0
        self.magnitude_underflow = 0
        self.magnitude_overflow = 0
        self.error_underflow = 0
        self.error_overflow = 0
        self.reference_zeros = 0
        self.updates = 0

    @classmethod
    def from_range(cls, collector, bin_count):
        if collector.quantizer is None:
            raise RuntimeError("range collector has no quantizer")
        return cls(
            collector.signed_edges(bin_count),
            collector.magnitude_edges(bin_count),
            collector.error_edges(bin_count),
            collector.quantizer.qmin,
            collector.quantizer.qmax)

    @staticmethod
    def _counts(values, edges):
        device_edges = torch.as_tensor(
            edges, device=values.device, dtype=values.dtype)
        underflow = int((values < device_edges[0]).sum().item())
        overflow = int((values > device_edges[-1]).sum().item())
        selected = values[
            (values >= device_edges[0]) & (values <= device_edges[-1])]
        indices = torch.bucketize(selected, device_edges[1:-1])
        counts = torch.bincount(
            indices, minlength=device_edges.numel() - 1)
        return counts.cpu().numpy().astype(np.int64), underflow, overflow

    @property
    def reference_total(self):
        return int(self.reference_counts.sum()) + \
            self.reference_underflow + self.reference_overflow + \
            self.reference_zeros

    @property
    def magnitude_total(self):
        return int(self.magnitude_counts.sum()) + \
            self.magnitude_underflow + self.magnitude_overflow + \
            self.reference_zeros

    @property
    def error_total(self):
        return int(self.error_counts.sum()) + \
            self.error_underflow + self.error_overflow

    @property
    def code_total(self):
        return int(self.code_counts.sum())

    def update(self, reference, quantized, codes):
        if reference.shape != quantized.shape or reference.shape != codes.shape:
            raise ValueError("reference, quantized, and codes must have one shape")
        _finite_tensor("reference", reference)
        _finite_tensor("quantized", quantized)
        flat_reference = reference.detach().reshape(-1)
        flat_error = (reference.detach() - quantized.detach()).reshape(-1)
        flat_codes = codes.detach().reshape(-1).to(torch.int64)
        if int(flat_codes.min().item()) < self.qmin or \
                int(flat_codes.max().item()) > self.qmax:
            raise ValueError("quantization codes exceed configured range")

        zero_mask = flat_reference == 0
        nonzero_reference = flat_reference[~zero_mask]
        self.reference_zeros += int(zero_mask.sum().item())
        counts, underflow, overflow = self._counts(
            nonzero_reference, self.signed_edges)
        self.reference_counts += counts
        self.reference_underflow += underflow
        self.reference_overflow += overflow

        counts, underflow, overflow = self._counts(
            nonzero_reference.abs(), self.magnitude_edges)
        self.magnitude_counts += counts
        self.magnitude_underflow += underflow
        self.magnitude_overflow += overflow

        counts, underflow, overflow = self._counts(
            flat_error, self.error_edges)
        self.error_counts += counts
        self.error_underflow += underflow
        self.error_overflow += overflow

        code_indices = flat_codes - self.qmin
        code_counts = torch.bincount(
            code_indices, minlength=self.code_counts.size)
        self.code_counts += code_counts.cpu().numpy().astype(np.int64)
        self.updates += 1
