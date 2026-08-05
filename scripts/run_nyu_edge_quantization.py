#!/usr/bin/env python3
"""Run NYU quantization with logical-edge QDQ and semantic model adapters."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def parse_edge_args(argv: Optional[Sequence[str]] = None
                    ) -> Tuple[argparse.Namespace, list]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--merge-policy", choices=("shared", "independent", "grouped"),
        default="shared")
    parser.add_argument("--merge-group-size", type=int, default=None)
    parser.add_argument(
        "--no-strict-semantic-sites", action="store_true",
        help="allow missing semantic sites during bring-up")
    options, remaining = parser.parse_known_args(argv)
    if options.merge_policy == "grouped":
        if options.merge_group_size is None or options.merge_group_size <= 0:
            parser.error("--merge-policy grouped requires --merge-group-size > 0")
    elif options.merge_group_size is not None:
        parser.error("--merge-group-size is only valid with grouped policy")
    return options, remaining


def normalize_merge_manifest(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for source in rows:
        row = dict(source)
        for key in ("unsigned", "qmin", "qmax", "scale", "zero_point"):
            row.setdefault(key, "")
        output.append(row)
    return output


def install_edge_backend(runner, options):
    from scripts.hardware_aligned_quantization import HardwareAlignedInstrumentor
    from spn_quant.adapters import install_model_semantic_adapter
    from spn_quant.runtime import EdgeAwareInstrumentorAdapter, EdgeQDQRuntime

    shared_runtime = EdgeQDQRuntime()
    active = {"semantic_adapter": None}

    def instrumentor_factory(*args, **kwargs):
        return EdgeAwareInstrumentorAdapter(
            HardwareAlignedInstrumentor(*args, **kwargs),
            runtime=shared_runtime)

    def semantic_adapter_factory(model):
        adapter = install_model_semantic_adapter(
            model, runtime=shared_runtime,
            merge_policy=options.merge_policy,
            group_size=options.merge_group_size,
            strict=not options.no_strict_semantic_sites)
        original_manifest = adapter.manifest
        adapter.manifest = lambda: normalize_merge_manifest(original_manifest())
        active["semantic_adapter"] = adapter
        return adapter

    original_write_json = runner.write_json

    def write_json_with_semantics(path, payload):
        adapter = active.get("semantic_adapter")
        if adapter is not None:
            rows = adapter.semantic_manifest()
            runner.write_csv(Path(path).parent / "semantic_site_manifest.csv", rows)
            roles = sorted(set(row["role"] for row in rows))
            operational = sum(int(row.get("meta_operational", 0)) for row in rows)
            observed = sum(int(row.get("observed", 0)) for row in rows)
            payload["semantic_quantization"] = {
                "model": adapter.MODEL_NAME,
                "sites": len(rows),
                "roles": roles,
                "observed_sites": observed,
                "operational_sites": operational,
                "strict": int(not options.no_strict_semantic_sites),
                "default_transform": "none",
                "lognp_default": 0,
            }
        return original_write_json(path, payload)

    runner.HardwareAlignedInstrumentor = instrumentor_factory
    runner.CallIndexedConcatAdapter = semantic_adapter_factory
    runner.write_json = write_json_with_semantics
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
