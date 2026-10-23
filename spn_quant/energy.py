"""Power-sample integration helpers for controlled inference measurements."""

from __future__ import annotations

import bisect
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PowerSample = Tuple[float, float]


def normalize_power_sample(
        monotonic_ns: int, power_mw: Dict[str, float]) -> Dict[str, object]:
    """Normalize one hardware reading to timestamped watts."""
    timestamp = int(monotonic_ns)
    if timestamp < 0 or not power_mw:
        raise ValueError("power sample requires a non-negative time and rails")
    normalized = {}
    for name, value in power_mw.items():
        milliwatts = float(value)
        if not name or not math.isfinite(milliwatts) or milliwatts < 0.0:
            raise ValueError("power rail names and readings must be valid")
        normalized[str(name)] = milliwatts / 1000.0
    return {
        "kind": "sample",
        "monotonic_ns": timestamp,
        "power_w": normalized,
    }


def validate_energy_settings(settings: Dict[str, object]) -> None:
    """Validate the controlled energy-measurement contract."""
    for key in ("warmup_iterations", "idle_seconds", "active_seconds",
                "sample_interval_ms"):
        value = float(settings.get(key, 0))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("energy setting %s must be positive" % key)
    if settings.get("synchronize_each_forward") is not True:
        raise ValueError("energy forwards must be synchronized")


def parse_power_trace(path: Path) -> Dict[str, object]:
    """Load a completed JSONL hardware-power trace."""
    metadata = None
    samples = []
    ended = False
    for line_number, line in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as error:
            raise ValueError(
                "invalid power trace JSON at line %d" % line_number) from error
        kind = record.get("kind")
        if kind == "metadata":
            if metadata is not None or samples:
                raise ValueError("power trace metadata must be first and unique")
            metadata = record
        elif kind == "sample":
            if metadata is None or ended:
                raise ValueError("power trace sample is outside its stream")
            normalized = normalize_power_sample(
                int(record["monotonic_ns"]),
                {key: float(value) * 1000.0
                 for key, value in record["power_w"].items()},
            )
            if samples and normalized["monotonic_ns"] <= \
                    samples[-1]["monotonic_ns"]:
                raise ValueError("power trace timestamps are not increasing")
            samples.append(normalized)
        elif kind == "end":
            if metadata is None or ended:
                raise ValueError("power trace has an invalid end record")
            ended = True
        else:
            raise ValueError("unknown power trace record at line %d" % line_number)
    if metadata is None or len(samples) < 2 or not ended:
        raise ValueError("power trace is incomplete")
    return {"metadata": metadata, "samples": samples}


def _validated_samples(samples: Iterable[PowerSample]) -> List[PowerSample]:
    values = [(float(timestamp), float(power)) for timestamp, power in samples]
    if len(values) < 2:
        raise ValueError("at least two power samples are required")
    for index, (timestamp, power) in enumerate(values):
        if not math.isfinite(timestamp) or not math.isfinite(power):
            raise ValueError("power samples must be finite")
        if power < 0.0:
            raise ValueError("power samples must be non-negative")
        if index and timestamp <= values[index - 1][0]:
            raise ValueError("power sample timestamps must be strictly increasing")
    return values


def _interpolate(
        left: PowerSample, right: PowerSample, timestamp: float) -> PowerSample:
    fraction = (timestamp - left[0]) / (right[0] - left[0])
    return timestamp, left[1] + fraction * (right[1] - left[1])


def _clip_samples(
        samples: Iterable[PowerSample], start: float,
        end: float) -> Tuple[List[PowerSample], int]:
    values = _validated_samples(samples)
    start = float(start)
    end = float(end)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("power window must have a finite positive duration")
    timestamps = [item[0] for item in values]
    if timestamps[0] > start or timestamps[-1] < end:
        raise ValueError("power samples do not cover the requested window")

    left_index = max(0, bisect.bisect_right(timestamps, start) - 1)
    right_index = min(len(values) - 1, bisect.bisect_left(timestamps, end))
    if timestamps[left_index] == start:
        start_sample = values[left_index]
    else:
        start_sample = _interpolate(
            values[left_index], values[left_index + 1], start)
    if timestamps[right_index] == end:
        end_sample = values[right_index]
    else:
        end_sample = _interpolate(
            values[right_index - 1], values[right_index], end)

    clipped = [start_sample]
    clipped.extend(
        item for item in values[left_index + 1:right_index]
        if start < item[0] < end)
    if clipped[-1][0] != end_sample[0]:
        clipped.append(end_sample)
    support_count = right_index - left_index + 1
    return clipped, support_count


def integrate_power_joules(
        samples: Iterable[PowerSample], start: float, end: float) -> float:
    """Integrate watts over a covered monotonic-second interval."""
    clipped, _ = _clip_samples(samples, start, end)
    energy = 0.0
    for left, right in zip(clipped, clipped[1:]):
        energy += (right[0] - left[0]) * (left[1] + right[1]) / 2.0
    return energy


def summarize_power_window(
        samples: Iterable[PowerSample], start: float,
        end: float) -> Dict[str, float]:
    """Return time-weighted power statistics for one covered interval."""
    values = list(samples)
    clipped, support_count = _clip_samples(values, start, end)
    duration = float(end) - float(start)
    energy = 0.0
    for left, right in zip(clipped, clipped[1:]):
        energy += (right[0] - left[0]) * (left[1] + right[1]) / 2.0
    return {
        "duration_seconds": duration,
        "energy_joules": energy,
        "mean_power_w": energy / duration,
        "sample_count": support_count,
    }


def energy_per_forward(
        active_energy_j: float, active_seconds: float,
        idle_mean_power_w: float,
        completed_forwards: int,
        allow_negative_incremental: bool = False) -> Dict[str, float]:
    """Calculate gross and idle-subtracted energy for completed forwards."""
    active_energy_j = float(active_energy_j)
    active_seconds = float(active_seconds)
    idle_mean_power_w = float(idle_mean_power_w)
    completed_forwards = int(completed_forwards)
    if not all(math.isfinite(value) for value in (
            active_energy_j, active_seconds, idle_mean_power_w)):
        raise ValueError("energy inputs must be finite")
    if active_energy_j < 0.0 or active_seconds <= 0.0:
        raise ValueError(
            "active energy must be non-negative and duration positive")
    if idle_mean_power_w < 0.0 or completed_forwards <= 0:
        raise ValueError("idle power must be non-negative and count positive")
    active_mean_power_w = active_energy_j / active_seconds
    incremental_energy_j = (
        active_energy_j - idle_mean_power_w * active_seconds)
    if incremental_energy_j < 0.0 and not allow_negative_incremental:
        raise ValueError("active mean power is below idle mean power")
    return {
        "active_mean_power_w": active_mean_power_w,
        "gross_j_per_forward": active_energy_j / completed_forwards,
        "incremental_j_per_forward": (
            incremental_energy_j / completed_forwards),
    }


def summarize_energy_trace(
        trace: Dict[str, object], idle_start_monotonic_ns: int,
        idle_end_monotonic_ns: int, active_start_monotonic_ns: int,
        active_end_monotonic_ns: int,
        completed_forwards: int) -> Dict[str, object]:
    """Integrate every rail in a parsed trace over idle and active windows."""
    samples = list(trace.get("samples", []))
    if len(samples) < 2:
        raise ValueError("energy trace requires at least two samples")
    rail_names = tuple(sorted(samples[0]["power_w"]))
    if not rail_names:
        raise ValueError("energy trace has no power rails")
    if any(tuple(sorted(sample["power_w"])) != rail_names
           for sample in samples):
        raise ValueError("power rail set changed within the trace")

    idle_start = int(idle_start_monotonic_ns) / 1.0e9
    idle_end = int(idle_end_monotonic_ns) / 1.0e9
    active_start = int(active_start_monotonic_ns) / 1.0e9
    active_end = int(active_end_monotonic_ns) / 1.0e9
    rails = {}
    for rail_name in rail_names:
        rail_samples = [
            (int(sample["monotonic_ns"]) / 1.0e9,
             float(sample["power_w"][rail_name]))
            for sample in samples
        ]
        idle = summarize_power_window(rail_samples, idle_start, idle_end)
        active = summarize_power_window(
            rail_samples, active_start, active_end)
        per_forward = energy_per_forward(
            active["energy_joules"], active["duration_seconds"],
            idle["mean_power_w"], completed_forwards,
            allow_negative_incremental=(
                rail_name not in ("gpu_board", "total_module")))
        rails[rail_name] = {
            "idle": idle,
            "active": active,
            **per_forward,
        }
    return {
        "source": trace.get("metadata", {}).get("source"),
        "sensor_scope": trace.get("metadata", {}).get("scope"),
        "completed_forwards": int(completed_forwards),
        "rails": rails,
    }
