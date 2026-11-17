#!/usr/bin/env python3
"""Compile one RKNN-safe SPN graph as FP16 and calibrated W8A8."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Sequence


@dataclass(frozen=True)
class Precision:
    name: str
    quantize: bool
    config: Dict[str, str]


def precision_plan():
    return (
        Precision("fp16", False, {"float_dtype": "float16"}),
        Precision("w8a8", True, {
            "quantized_dtype": "w8a8",
            "quantized_algorithm": "normal",
            "quantized_method": "channel",
        }),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_calibration(path: Path, expected_count: int = 128) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    lines = [line for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if len(lines) != int(expected_count):
        raise ValueError("calibration manifest must contain %d samples" %
                         expected_count)
    return path.resolve()


def compile_variant(source: Path, output: Path, precision: Precision,
                    calibration: Optional[Path]) -> dict:
    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    try:
        config = dict(target_platform="rk3588", optimization_level=3)
        config.update(precision.config)
        calls = {
            "config": rknn.config(**config),
            "load_onnx": rknn.load_onnx(model=str(source)),
        }
        dataset = str(calibration) if precision.quantize else None
        calls["build"] = rknn.build(
            do_quantization=precision.quantize, dataset=dataset)
        if any(int(value) != 0 for value in calls.values()):
            raise RuntimeError("RKNN compile failed: %s" % calls)
        output.parent.mkdir(parents=True, exist_ok=True)
        calls["export"] = rknn.export_rknn(str(output))
        if int(calls["export"]) != 0:
            raise RuntimeError("RKNN export failed: %s" % calls)
    finally:
        rknn.release()
    return {
        "precision": precision.name,
        "quantized": precision.quantize,
        "config": config,
        "calibration": None if calibration is None else str(calibration),
        "status": {key: int(value) for key, value in calls.items()},
        "path": str(output.resolve()),
        "bytes": output.stat().st_size,
        "sha256": file_sha256(output),
    }


def build(args: argparse.Namespace) -> Path:
    selected = precision_plan() if args.precision == "all" else tuple(
        item for item in precision_plan() if item.name == args.precision)
    calibration = None
    if any(item.quantize for item in selected):
        calibration = validate_calibration(
            args.calibration, args.calibration_count)
    records = {}
    for precision in selected:
        destination = args.output_dir / (
            "%s_compressed_%s.rknn" % (args.model, precision.name))
        records[precision.name] = compile_variant(
            args.source, destination, precision,
            calibration if precision.quantize else None)
    manifest = {
        "schema_version": 1,
        "model": args.model,
        "target": "rk3588",
        "source": str(args.source.resolve()),
        "source_sha256": file_sha256(args.source),
        "calibration_count": args.calibration_count,
        "variants": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return args.manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(
        "cspn", "dyspn", "nlspn", "completionformer"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--calibration-count", type=int, default=128)
    parser.add_argument("--precision", choices=("all", "fp16", "w8a8"),
                        default="all")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    print(build(_parser().parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
