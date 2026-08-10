#!/usr/bin/env python3
"""Streaming activation and QDQ histogram statistics."""

from __future__ import division

import math
import csv
from pathlib import Path

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
        for name, edges in (
                ("signed", self.signed_edges),
                ("magnitude", self.magnitude_edges),
                ("error", self.error_edges)):
            if edges.ndim != 1 or edges.size < 2:
                raise ValueError("%s histogram edges are invalid" % name)
            if not np.isfinite(edges).all():
                raise ValueError("%s histogram edges are nonfinite" % name)
            if not bool(np.all(np.diff(edges) > 0)):
                raise ValueError("%s histogram edges are not increasing" % name)
        self.qmin = int(qmin)
        self.qmax = int(qmax)
        if self.qmin >= self.qmax:
            raise ValueError("quantizer code range is invalid")
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
        _finite_tensor("quantization error", flat_error)
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


INPUT_STEMS = {
    "dyspn": {
        "base.conv1_rgb.0": "input_rgb",
        "base.conv1_dep.0": "input_depth",
    },
    "nlspn": {
        "conv1_rgb.0": "input_rgb",
        "conv1_dep.0": "input_depth",
    },
    "completionformer": {
        "backbone.conv1_rgb.0": "input_rgb",
        "backbone.conv1_dep.0": "input_depth",
    },
}


def site_name(module, call_index, kind):
    return "%s#%d:%s" % (module, int(call_index), kind)


def _same_quantizer(left, right):
    scalar_fields = (
        "format", "bits", "unsigned", "qmin", "qmax", "zero_point",
        "channel_dim", "granularity")
    if any(left[field] != right[field] for field in scalar_fields):
        return False
    return bool(np.array_equal(left["scale"], right["scale"]))


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class ActivationHistogramRecorder(object):
    """Record every configured activation QDQ site in two streaming passes."""

    def __init__(self, model_name, phase, capacity, per_update):
        if model_name not in ("cspn", "dyspn", "nlspn", "completionformer"):
            raise ValueError("unknown model: %s" % model_name)
        if phase != "range":
            raise ValueError("activation recorder must start in range phase")
        self.model_name = model_name
        self.phase = phase
        self.capacity = int(capacity)
        self.per_update = int(per_update)
        self.ranges = {}
        self.histograms = {}
        self.site_metadata = {}
        self.bin_count = None
        self.validated = False

    def site_names(self):
        return sorted(self.ranges)

    def _input_slices(self, module, kind, reference, quantized, codes):
        if kind != "input" or reference.ndim != 4:
            return []
        if self.model_name == "cspn" and module == "conv1_1":
            if reference.shape[1] != 4:
                raise ValueError("CSPN conv1_1 input must contain RGBD channels")
            return [
                ("input_rgb", reference[:, :3], quantized[:, :3], codes[:, :3]),
                ("input_depth", reference[:, 3:4], quantized[:, 3:4],
                 codes[:, 3:4]),
            ]
        if self.model_name == "cspn":
            return []
        stems = INPUT_STEMS[self.model_name]
        if module not in stems:
            return []
        return [(stems[module], reference, quantized, codes)]

    def _metadata(self, module, kind, call_index, group, reference,
                  quantizer, channel_dim, synthetic_slice):
        channel_dim = int(channel_dim)
        if channel_dim < -reference.ndim or channel_dim >= reference.ndim:
            raise ValueError("activation channel dimension is invalid")
        descriptor = quantizer_descriptor(quantizer, channel_dim)
        return {
            "site": site_name(module, call_index, kind),
            "module": module,
            "call_index": int(call_index),
            "kind": kind,
            "group": group,
            "synthetic_slice": int(synthetic_slice),
            "channels": int(reference.shape[channel_dim]),
            "quantizer": descriptor,
        }

    def _record_one(self, module, kind, call_index, group, reference,
                    quantized, codes, quantizer, channel_dim,
                    synthetic_slice):
        name = site_name(module, call_index, kind)
        metadata = self._metadata(
            module, kind, call_index, group, reference,
            quantizer, channel_dim, synthetic_slice)
        channel_dim = metadata["quantizer"]["channel_dim"]
        if self.phase == "range":
            if name not in self.ranges:
                self.ranges[name] = RangeCollector(
                    capacity=self.capacity, per_update=self.per_update)
                self.site_metadata[name] = metadata
            elif self.site_metadata[name]["synthetic_slice"] != \
                    int(synthetic_slice):
                raise ValueError("site metadata changed during range collection")
            self.ranges[name].update(
                reference, quantized, codes, quantizer, channel_dim)
            return
        if self.phase != "histogram":
            raise RuntimeError("activation recorder phase is invalid")
        if name not in self.histograms:
            raise ValueError("histogram pass produced unknown site: %s" % name)
        if not _same_quantizer(
                self.site_metadata[name]["quantizer"],
                metadata["quantizer"]):
            raise ValueError("quantizer changed between profiling passes: %s" % name)
        self.histograms[name].update(reference, quantized, codes)

    def record(self, module, kind, call_index, group, reference,
               quantized, codes, quantizer, channel_dim):
        self._record_one(
            module, kind, call_index, group, reference,
            quantized, codes, quantizer, channel_dim,
            synthetic_slice=False)
        for slice_module, slice_reference, slice_quantized, slice_codes in \
                self._input_slices(
                    module, kind, reference, quantized, codes):
            self._record_one(
                slice_module, kind, call_index, "input",
                slice_reference, slice_quantized, slice_codes,
                quantizer, channel_dim, synthetic_slice=True)

    def freeze_ranges(self, bin_count):
        if self.phase != "range":
            raise RuntimeError("range phase has already ended")
        if not self.ranges:
            raise RuntimeError("range pass observed no activation sites")
        self.bin_count = int(bin_count)
        if self.bin_count <= 0:
            raise ValueError("histogram bin count must be positive")
        self.histograms = dict(
            (name, HistogramAccumulator.from_range(
                self.ranges[name], self.bin_count))
            for name in self.site_names())
        self.phase = "frozen"
        return self.outlier_rows()

    def begin_histogram_pass(self):
        if self.phase != "frozen":
            raise RuntimeError("range bins must be frozen first")
        self.phase = "histogram"

    def _real_sites(self):
        return {
            name for name in self.site_names()
            if self.site_metadata[name]["synthetic_slice"] == 0
        }

    def validate(self, expected_manifest_sites, expected_updates):
        if self.phase != "histogram":
            raise RuntimeError("histogram pass has not started")
        expected = set(expected_manifest_sites)
        real = self._real_sites()
        missing = sorted(expected - real)
        if missing:
            raise ValueError("missing manifest sites: %s" % ", ".join(missing))
        unexpected = sorted(real - expected)
        if unexpected:
            raise ValueError(
                "unexpected manifest sites: %s" % ", ".join(unexpected))
        for name in self.site_names():
            ranges = self.ranges[name]
            histogram = self.histograms[name]
            if ranges.updates != int(expected_updates):
                raise ValueError("range update count mismatch: %s" % name)
            if histogram.updates != int(expected_updates):
                raise ValueError("histogram update count mismatch: %s" % name)
            if histogram.reference_total != ranges.elements or \
                    histogram.magnitude_total != ranges.elements or \
                    histogram.error_total != ranges.elements or \
                    histogram.code_total != ranges.elements:
                raise ValueError("histogram count mismatch: %s" % name)
        self.validated = True

    def outlier_rows(self):
        rows = []
        real_sites = self._real_sites()
        total_error = sum(
            self.ranges[name].error_energy for name in real_sites)
        group_error = {}
        for name in sorted(real_sites):
            group = self.site_metadata[name]["group"]
            if group not in group_error:
                group_error[group] = 0.0
            group_error[group] += self.ranges[name].error_energy
        for name in self.site_names():
            metadata = self.site_metadata[name]
            row = dict((key, value) for key, value in metadata.items()
                       if key != "quantizer")
            row.update(self.ranges[name].summary())
            row["local_error_energy_share"] = \
                self.ranges[name].error_energy / total_error \
                if total_error > 0.0 else 0.0
            row["group_error_energy_share"] = 0.0 \
                if metadata["synthetic_slice"] else \
                group_error[metadata["group"]] / total_error \
                if total_error > 0.0 else 0.0
            row["excluded_from_aggregate"] = metadata["synthetic_slice"]
            row["total_error_energy_is_zero"] = int(total_error == 0.0)
            rows.append(row)
        real_rows = [row for row in rows if row["synthetic_slice"] == 0]
        rankings = (
            ("error_energy_rank", "error_energy", True),
            ("sqnr_rank", "sqnr_db", False),
            ("tail_rank", "p99_99_over_p99", True),
            ("channel_imbalance_rank", "channel_max_over_median", True),
        )
        for row in rows:
            row["critical_selection_eligible"] = int(
                row["synthetic_slice"] == 0)
            for rank_field, _, _ in rankings:
                row[rank_field] = 0
        for rank_field, metric, descending in rankings:
            ordered = sorted(
                real_rows,
                key=lambda row: (
                    -float(row[metric]) if descending else float(row[metric]),
                    row["site"]))
            for rank, row in enumerate(ordered, 1):
                row[rank_field] = rank
        return rows

    def write(self, output_dir):
        if not self.validated:
            raise RuntimeError("activation profile must be validated before writing")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        arrays = {}
        index_rows = []
        for index, name in enumerate(self.site_names()):
            prefix = "site_%04d" % index
            histogram = self.histograms[name]
            descriptor = self.site_metadata[name]["quantizer"]
            site_arrays = {
                "signed_edges_key": histogram.signed_edges,
                "reference_counts_key": histogram.reference_counts,
                "magnitude_edges_key": histogram.magnitude_edges,
                "magnitude_counts_key": histogram.magnitude_counts,
                "error_edges_key": histogram.error_edges,
                "error_counts_key": histogram.error_counts,
                "code_values_key": np.arange(
                    histogram.qmin, histogram.qmax + 1, dtype=np.int64),
                "code_counts_key": histogram.code_counts,
                "scale_key": descriptor["scale"],
            }
            row = dict((key, value) for key, value in self.site_metadata[name].items()
                       if key != "quantizer")
            for field in descriptor:
                if field != "scale":
                    row[field] = descriptor[field]
            for field, value in site_arrays.items():
                key = "%s_%s" % (prefix, field[:-4])
                arrays[key] = value
                row[field] = key
            ranges = self.ranges[name]
            row.update({
                "updates": histogram.updates,
                "elements": ranges.elements,
                "reference_zeros": histogram.reference_zeros,
                "reference_underflow": histogram.reference_underflow,
                "reference_overflow": histogram.reference_overflow,
                "magnitude_underflow": histogram.magnitude_underflow,
                "magnitude_overflow": histogram.magnitude_overflow,
                "error_underflow": histogram.error_underflow,
                "error_overflow": histogram.error_overflow,
            })
            index_rows.append(row)
        np.savez_compressed(output / "histogram_data.npz", **arrays)
        _write_csv(output / "histogram_index.csv", index_rows)
        _write_csv(output / "outlier_summary.csv", self.outlier_rows())
