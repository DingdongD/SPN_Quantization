"""Parse the small SRAM contract consumed from an official CACTI run."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class CactiSramResult:
    capacity_bytes: int
    access_time_ns: float
    cycle_time_ns: float
    area_mm2: float
    read_energy_nj: float
    write_energy_nj: float
    output_width_bits: int
    bank_count: int


def _number(text: str, label: str) -> float:
    match = re.search(
        r"^\s*" + re.escape(label) + r"\s*:\s*([-+0-9.eE]+)\s*$",
        text, flags=re.MULTILINE)
    if match is None:
        raise ValueError("CACTI output is missing %s" % label)
    value = float(match.group(1))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("CACTI output has invalid %s" % label)
    return value


def parse_cacti_output(text: str, expected_capacity_bytes: int,
                       output_width_bits: int) -> CactiSramResult:
    """Extract timing, area, and access energy from CACTI stdout."""
    expected_capacity = int(expected_capacity_bytes)
    width = int(output_width_bits)
    if expected_capacity <= 0:
        raise ValueError("expected capacity must be positive")
    if width <= 0 or width % 8:
        raise ValueError("output width must be a positive whole number of bytes")
    capacity = int(_number(text, "Cache size"))
    if capacity != expected_capacity:
        raise ValueError(
            "CACTI capacity mismatch: expected %d, observed %d" %
            (expected_capacity, capacity))
    banks = int(_number(text, "Cache banks (UCA)"))
    dimensions = re.search(
        r"^\s*Cache height x width \(mm\):\s*"
        r"([-+0-9.eE]+)\s*x\s*([-+0-9.eE]+)\s*$",
        text, flags=re.MULTILINE)
    if dimensions is None:
        raise ValueError("CACTI output is missing Cache height x width (mm)")
    height, width_mm = (float(value) for value in dimensions.groups())
    if min(height, width_mm) <= 0:
        raise ValueError("CACTI output has invalid physical dimensions")
    return CactiSramResult(
        capacity_bytes=capacity,
        access_time_ns=_number(text, "Access time (ns)"),
        cycle_time_ns=_number(text, "Cycle time (ns)"),
        area_mm2=height * width_mm,
        read_energy_nj=_number(
            text, "Total dynamic read energy per access (nJ)"),
        write_energy_nj=_number(
            text, "Total dynamic write energy per access (nJ)"),
        output_width_bits=width, bank_count=banks)
