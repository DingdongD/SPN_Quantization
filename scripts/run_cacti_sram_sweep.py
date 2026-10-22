#!/usr/bin/env python3
"""Characterize several CACTI SRAM macros with one compilation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_cacti_sram import run_many  # noqa: E402


def _job(values: Sequence[str]) -> Tuple[Path, Path, Path]:
    if len(values) != 3:
        raise ValueError("CACTI job requires CONFIG RAW_OUTPUT JSON_OUTPUT")
    return tuple(Path(value) for value in values)  # type: ignore[return-value]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job", nargs=3, action="append", metavar=("CONFIG", "RAW", "JSON"),
        required=True)
    return parser


def main(argv: Any = None) -> int:
    args = _parser().parse_args(argv)
    outputs = run_many(tuple(_job(values) for values in args.job))
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
