#!/usr/bin/env python3
"""Run the existing NYU driver with logical-edge QDQ and merge policies.

The wrapper keeps the mature experiment/evaluation path untouched. It replaces
only the hardware instrumentor and Concat adapter factories before dispatching
to ``run_nyu_rtn_quantization.main``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def parse_edge_args(argv: Optional[Sequence[str]] = None
                    ) -> Tuple[argparse.Namespace, list]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--merge-policy", choices=("shared", "independent", "grouped"),
        default="shared")
    parser.add_argument("--merge-group-size", type=int, default=None)
    options, remaining = parser.parse_known_args(argv)
    if options.merge_policy == "grouped":
        if options.merge_group_size is None or options.merge_group_size <= 0:
            parser.error("--merge-policy grouped requires --merge-group-size > 0")
    elif options.merge_group_size is not None:
        parser.error("--merge-group-size is only valid with grouped policy")
    return options, remaining


def normalize_merge_manifest(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep legacy hardware-manifest columns for non-scalar scale layouts."""
    output = []
    for source in rows:
        row = dict(source)
        for key in ("unsigned", "qmin", "qmax", "scale", "zero_point"):
            row.setdefault(key, "")
        output.append(row)
    return output


def install_edge_backend(runner, options):
    from scripts.hardware_aligned_quantization import HardwareAlignedInstrumentor
    from scripts.hardware_merge_adapters import CallIndexedConcatAdapter
    from spn_quant.runtime import EdgeAwareInstrumentorAdapter, EdgeQDQRuntime

    shared_runtime = EdgeQDQRuntime()

    def instrumentor_factory(*args, **kwargs):
        return EdgeAwareInstrumentorAdapter(
            HardwareAlignedInstrumentor(*args, **kwargs),
            runtime=shared_runtime)

    class RunnerConcatAdapter(CallIndexedConcatAdapter):
        def __init__(self, model):
            super(RunnerConcatAdapter, self).__init__(
                model, policy=options.merge_policy,
                group_size=options.merge_group_size,
                runtime=shared_runtime, manage_runtime=False)

        def manifest(self):
            return normalize_merge_manifest(
                super(RunnerConcatAdapter, self).manifest())

    runner.HardwareAlignedInstrumentor = instrumentor_factory
    runner.CallIndexedConcatAdapter = RunnerConcatAdapter
    return runner


def main(argv: Optional[Sequence[str]] = None) -> None:
    options, remaining = parse_edge_args(argv)
    from scripts import run_nyu_rtn_quantization as runner
    install_edge_backend(runner, options)
    original = list(sys.argv)
    try:
        sys.argv = [original[0]] + list(remaining)
        runner.main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    main(sys.argv[1:])
