#!/usr/bin/env python3
"""Continuously sample a hardware power interface into a JSONL trace."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal
import threading
import time
from typing import Dict, Sequence


def normalize_power_sample(
        monotonic_ns: int, power_mw: Dict[str, float]) -> Dict[str, object]:
    """Normalize a reading locally so the sampler needs no PyTorch package."""
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


def _write_record(handle, record: Dict[str, object]) -> None:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()


def sample_stream(reader, output: Path, interval_seconds: float,
                  stop_event, monotonic_ns=time.monotonic_ns) -> None:
    """Sample immediately and after every interval, including the stop edge."""
    if interval_seconds <= 0.0:
        raise ValueError("sample interval must be positive")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    last_timestamp = None
    try:
        with output.open("w", encoding="utf-8") as handle:
            _write_record(handle, {
                "kind": "metadata",
                "source": reader.source,
                "scope": reader.scope,
                "requested_interval_seconds": float(interval_seconds),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            })
            while True:
                last_timestamp = int(monotonic_ns())
                _write_record(handle, normalize_power_sample(
                    last_timestamp, reader.read_milliwatts()))
                if stop_event.is_set():
                    break
                stop_event.wait(interval_seconds)
            _write_record(handle, {
                "kind": "end",
                "monotonic_ns": last_timestamp,
            })
    finally:
        reader.close()


class NvmlPowerReader:
    source = "nvml"
    scope = "gpu_board"

    def __init__(self, gpu_index: int = 0):
        self._library = ctypes.CDLL("libnvidia-ml.so.1")
        self._library.nvmlInit_v2.restype = ctypes.c_int
        self._library.nvmlDeviceGetHandleByIndex_v2.argtypes = [
            ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        self._library.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        self._library.nvmlDeviceGetPowerUsage.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        self._library.nvmlDeviceGetPowerUsage.restype = ctypes.c_int
        self._library.nvmlShutdown.restype = ctypes.c_int
        self._check(self._library.nvmlInit_v2(), "nvmlInit_v2")
        self._handle = ctypes.c_void_p()
        self._check(self._library.nvmlDeviceGetHandleByIndex_v2(
            int(gpu_index), ctypes.byref(self._handle)),
            "nvmlDeviceGetHandleByIndex_v2")

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if int(status) != 0:
            raise RuntimeError("%s failed with NVML status %d" % (
                operation, status))

    def read_milliwatts(self) -> Dict[str, float]:
        value = ctypes.c_uint()
        self._check(self._library.nvmlDeviceGetPowerUsage(
            self._handle, ctypes.byref(value)), "nvmlDeviceGetPowerUsage")
        return {"gpu_board": float(value.value)}

    def close(self) -> None:
        self._check(self._library.nvmlShutdown(), "nvmlShutdown")


def jtop_power_milliwatts(power: Dict[str, object]) -> Dict[str, float]:
    """Extract all available jtop INA3221 rail readings in milliwatts."""
    rails = {}
    total = power.get("tot")
    if isinstance(total, dict) and total.get("power") is not None:
        rails["total_module"] = float(total["power"])
    rail_payload = power.get("rail", {})
    if isinstance(rail_payload, dict):
        for name, values in rail_payload.items():
            if isinstance(values, dict) and values.get("power") is not None:
                rails[str(name).lower()] = float(values["power"])
    if not rails:
        raise RuntimeError("jtop returned no power rails")
    return rails


class JtopPowerReader:
    source = "jtop_ina3221"
    scope = "jetson_module_and_rails"

    def __init__(self):
        from jtop import jtop
        self._jetson = jtop(interval=0.1)
        self._jetson.start()
        if not self._jetson.ok():
            self._jetson.close()
            raise RuntimeError("jtop service did not become ready")

    def read_milliwatts(self) -> Dict[str, float]:
        return jtop_power_milliwatts(self._jetson.power)

    def close(self) -> None:
        self._jetson.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("nvml", "jtop"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-ms", type=float, default=50.0)
    parser.add_argument("--gpu-index", type=int, default=0)
    return parser


def main(argv: Sequence[str] = None) -> int:
    args = _parser().parse_args(argv)
    reader = (NvmlPowerReader(args.gpu_index) if args.source == "nvml"
              else JtopPowerReader())
    stop_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    sample_stream(reader, args.output, args.interval_ms / 1000.0, stop_event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
