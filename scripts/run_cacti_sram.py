#!/usr/bin/env python3
"""Build pinned CACTI in a temporary tree and characterize one SRAM."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
CACTI_ROOT = REPO_ROOT / "external" / "cacti"
CACTI_REVISION = "1ffd8dfb10303d306ecd8d215320aea07651e878"
CACTI_PATCH = REPO_ROOT / "patches" / "cacti" / \
    "0001-initialize-input-parameters.patch"

if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from spn_quant.hardware.cacti import parse_cacti_output  # noqa: E402


@dataclass(frozen=True)
class CactiConfig:
    capacity_bytes: int
    block_bytes: int
    bank_count: int
    technology_nm: int
    output_width_bits: int
    temperature_k: int
    format_version: int = 1

    def __post_init__(self) -> None:
        integer_fields = (
            "capacity_bytes", "block_bytes", "bank_count", "technology_nm",
            "output_width_bits", "temperature_k")
        for name in integer_fields:
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError("%s must be positive" % name)
            object.__setattr__(self, name, value)
        if self.format_version != 1:
            raise ValueError("unsupported CACTI config format")
        if self.capacity_bytes & (self.capacity_bytes - 1):
            raise ValueError("CACTI capacity must be a power of two")
        if self.block_bytes & (self.block_bytes - 1):
            raise ValueError("CACTI block size must be a power of two")
        if self.output_width_bits % 8:
            raise ValueError("CACTI output width must be byte aligned")
        if self.output_width_bits != self.block_bytes * 8:
            raise ValueError("CACTI output width must equal one block")


def load_config(path: Path) -> CactiConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CactiConfig(**payload)


def cacti_cli_arguments(config: CactiConfig) -> Tuple[str, ...]:
    """Use CACTI's 63-argument interface so every scalar field is explicit."""
    values = (
        config.capacity_bytes, config.block_bytes, 1,  # capacity/line/assoc
        1, 0, 0, 0, 0, config.bank_count, config.technology_nm,
        config.output_width_bits, 0, 0, 0, 0, 0,
        0, 0, 0, 100, 0,  # optimize cycle time
        20, 100000, 100000, 100000, 100000,
        2, config.temperature_k, 1,
        0, 0, 0, 0, 1, 1,  # SRAM/peripheral/wire choices
        1, 1, 0, 1, 8192, 8, 8,
        0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
        0, 1, config.output_width_bits, 1000, 0, 1, 0, 0, 1, 1, 0,
    )
    arguments = tuple(str(value) for value in values)
    if len(arguments) != 63:  # pragma: no cover - protects upstream ABI edits
        raise RuntimeError("CACTI interface requires exactly 63 arguments")
    return arguments


def _git_revision(root: Path) -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=str(root), check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return completed.stdout.strip()


def _copy_source(destination: Path) -> None:
    shutil.copytree(
        str(CACTI_ROOT), str(destination),
        ignore=shutil.ignore_patterns(".git", "cacti", "obj_dbg", "obj_opt"))
    subprocess.run(
        ("patch", "-p1", "-i", str(CACTI_PATCH)), cwd=str(destination),
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _build_executable(build_root: Path) -> Path:
    _copy_source(build_root)
    subprocess.run(
        ("make", "-j2", "CXXFLAGS=-Wno-unknown-pragmas -O2 -DNTHREADS=1"),
        cwd=str(build_root), check=True)
    return build_root / "cacti"


def _execute(executable: Path, config: CactiConfig) -> str:
    completed = subprocess.run(
        (str(executable),) + cacti_cli_arguments(config),
        cwd=str(Path(executable).parent), check=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)
    return completed.stdout


def run_many(jobs: Sequence[Tuple[Path, Path, Path]]) -> Tuple[Path, ...]:
    """Characterize multiple SRAMs with one pinned CACTI compilation."""
    rows = tuple((Path(config), Path(raw), Path(output))
                 for config, raw, output in jobs)
    if not rows:
        raise ValueError("CACTI sweep requires at least one configuration")
    if _git_revision(CACTI_ROOT) != CACTI_REVISION:
        raise RuntimeError("CACTI submodule revision does not match the study")
    configs = tuple(load_config(config_path) for config_path, _, _ in rows)
    outputs = []
    with tempfile.TemporaryDirectory(prefix="spn-cacti-build-") as directory:
        build_root = Path(directory) / "cacti"
        executable = _build_executable(build_root)
        for (config_path, raw_output, json_output), config in zip(rows, configs):
            raw_output.parent.mkdir(parents=True, exist_ok=True)
            json_output.parent.mkdir(parents=True, exist_ok=True)
            stdout = _execute(executable, config)
            raw_output.write_text(stdout, encoding="utf-8")
            result = parse_cacti_output(
                stdout, expected_capacity_bytes=config.capacity_bytes,
                output_width_bits=config.output_width_bits)
            payload = {
                "format_version": 1,
                "source": "https://github.com/HewlettPackard/cacti",
                "cacti_revision": CACTI_REVISION,
                "compatibility_patch": str(
                    CACTI_PATCH.relative_to(REPO_ROOT)),
                "config_path": str(config_path.resolve()),
                "config": asdict(config),
                "result": asdict(result),
            }
            json_output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            outputs.append(json_output)
    return tuple(outputs)


def run(config_path: Path, raw_output: Path, json_output: Path) -> Path:
    return run_many(((config_path, raw_output, json_output),))[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser


def main(argv: Any = None) -> int:
    args = _parser().parse_args(argv)
    output = run(args.config, args.raw_output, args.json_output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
