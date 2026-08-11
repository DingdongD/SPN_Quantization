#!/usr/bin/env python3
"""Run NYU quantization with logical-edge QDQ and semantic model adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
    parser.add_argument(
        "--reconstruction-manifest", default=None,
        help="reconstruction manifest with activation ranges or a strict contract")
    parser.add_argument(
        "--deployment-contract", default=None,
        help="direct strict deployment-contract .pt path")
    options, remaining = parser.parse_known_args(argv)
    if options.merge_policy == "grouped":
        if options.merge_group_size is None or options.merge_group_size <= 0:
            parser.error("--merge-policy grouped requires --merge-group-size > 0")
    elif options.merge_group_size is not None:
        parser.error("--merge-group-size is only valid with grouped policy")
    if options.reconstruction_manifest and options.deployment_contract:
        parser.error(
            "use either --reconstruction-manifest or --deployment-contract")
    return options, remaining


def normalize_merge_manifest(
        rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for source in rows:
        row = dict(source)
        for key in ("unsigned", "qmin", "qmax", "scale", "zero_point"):
            row.setdefault(key, "")
        output.append(row)
    return output


def _resolve_relative(path: str, parent: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else parent / value


def _load_strict_contract(path: Path):
    from spn_quant.deployment_contract import load_deployment_contract
    from spn_quant.qdrop_contract import (
        QDROP_CONTRACT_VERSION,
        load_qdrop_contract,
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload["format_version"]) == QDROP_CONTRACT_VERSION:
        return load_qdrop_contract(path)
    return load_deployment_contract(path)


def load_reconstruction_manifest(
        path: Optional[str], direct_contract: Optional[str] = None
        ) -> Optional[Dict[str, Any]]:
    if path is None and direct_contract is None:
        return None
    if direct_contract is not None:
        contract_path = Path(direct_contract)
        contract = _load_strict_contract(contract_path)
        if int(contract["strict"]) != 1:
            raise ValueError("strict deployment contract required")
        if contract["method"] == "qdrop_strict":
            return {
                "path": "",
                "strict": 1,
                "method": "qdrop_strict",
                "activation_bits": 4,
                "activation_policy": "exact_semantic_edge_contract",
                "weight_bits": 4,
                "targets": list(contract["target_plan"]["blocks"]),
                "strict_contract_path": str(contract_path),
                "strict_contract": contract,
                "qdrop_contract": contract,
            }
        first = next(iter(contract["weight_contracts"].values()))
        return {
            "path": "",
            "strict": 1,
            "method": contract["method"],
            "activation_bits": 0,
            "activation_policy": "evaluation_backend_owned",
            "weight_bits": int(first["bits"]),
            "targets": list(contract["targets"]),
            "strict_contract_path": str(contract_path),
            "strict_contract": contract,
        }

    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(str(manifest_path))
    payload = json.loads(
        manifest_path.read_text(encoding="utf-8"))
    if "strict" not in payload or int(payload["strict"]) != 1:
        raise ValueError("strict reconstruction manifest required")
    if payload["method"] == "qdrop_strict":
        required = {
            "format_version", "strict", "method", "model",
            "deployment_contract", "targets", "weight_bits",
            "activation_bits", "activation_policy",
        }
        if set(payload) != required:
            raise KeyError("strict QDrop manifest fields mismatch")
        if int(payload["format_version"]) != 2 or \
                int(payload["weight_bits"]) != 4 or \
                int(payload["activation_bits"]) != 4:
            raise ValueError("strict QDrop manifest requires W4A4")
        if payload["activation_policy"] != \
                "exact_semantic_edge_contract":
            raise ValueError("strict QDrop activation policy mismatch")
        contract_path = _resolve_relative(
            str(payload["deployment_contract"]), manifest_path.parent)
        contract = _load_strict_contract(contract_path)
        if contract["method"] != "qdrop_strict" or \
                int(contract["format_version"]) != 2:
            raise ValueError("strict QDrop contract mismatch")
        if list(contract["target_plan"]["blocks"]) != \
                list(payload["targets"]):
            raise ValueError("strict QDrop targets mismatch")
        if str(contract["target_plan"]["model"]) != str(payload["model"]):
            raise ValueError("strict QDrop model mismatch")
        return {
            "path": str(manifest_path),
            "strict": 1,
            "method": "qdrop_strict",
            "activation_bits": 4,
            "activation_policy": "exact_semantic_edge_contract",
            "weight_bits": 4,
            "targets": list(payload["targets"]),
            "strict_contract_path": str(contract_path),
            "strict_contract": contract,
            "qdrop_contract": contract,
        }
    if int(payload["activation_bits"]) != 0 or payload["activation_manifest"]:
        raise ValueError(
            "strict weight reconstruction cannot carry activation overrides")
    contract_path = _resolve_relative(
        str(payload["deployment_contract"]), manifest_path.parent)
    strict_contract = _load_strict_contract(contract_path)
    if int(strict_contract["strict"]) != 1:
        raise ValueError("strict deployment contract required")
    if strict_contract["method"] != payload["method"]:
        raise ValueError("strict reconstruction method mismatch")
    if list(strict_contract["targets"]) != list(payload["targets"]):
        raise ValueError("strict reconstruction targets mismatch")
    return {
        "path": str(manifest_path),
        "strict": 1,
        "method": payload["method"],
        "activation_bits": int(payload["activation_bits"]),
        "activation_policy": "evaluation_backend_owned",
        "weight_bits": int(payload["weight_bits"]),
        "targets": list(payload["targets"]),
        "strict_contract_path": str(contract_path),
        "strict_contract": strict_contract,
    }


def install_edge_backend(runner, options):
    from scripts.hardware_aligned_quantization import HardwareAlignedInstrumentor
    from spn_quant.adapters import install_model_semantic_adapter
    from spn_quant.deployment_contract import (
        StrictContractInstrumentor,
        file_sha256,
        validate_graph_preparation,
    )
    from spn_quant.qdrop_contract import QDropContractInstrumentor
    from spn_quant.runtime import EdgeAwareInstrumentorAdapter, EdgeQDQRuntime

    shared_runtime = EdgeQDQRuntime()
    active = {"semantic_adapter": None}
    reconstruction = load_reconstruction_manifest(
        options.reconstruction_manifest,
        options.deployment_contract)
    strict_contract = (
        reconstruction["strict_contract"]
        if reconstruction is not None else None)

    def instrumentor_factory(*args, **kwargs):
        base = HardwareAlignedInstrumentor(*args, **kwargs)
        if strict_contract is not None:
            group_fn = (
                args[1] if len(args) > 1
                else kwargs["group_fn"])
            if strict_contract["method"] == "qdrop_strict":
                base = QDropContractInstrumentor(
                    base, strict_contract,
                    group_fn=group_fn)
            else:
                base = StrictContractInstrumentor(
                    base, strict_contract,
                    group_fn=group_fn)
        return EdgeAwareInstrumentorAdapter(
            base, runtime=shared_runtime)

    def semantic_adapter_factory(model):
        adapter = install_model_semantic_adapter(
            model, runtime=shared_runtime,
            merge_policy=options.merge_policy,
            group_size=options.merge_group_size,
            strict=not options.no_strict_semantic_sites)
        original_manifest = adapter.manifest
        adapter.manifest = lambda: normalize_merge_manifest(
            original_manifest())
        active["semantic_adapter"] = adapter
        return adapter

    if strict_contract is not None:
        original_build_model = runner.build_model

        def build_model_with_contract(
                saved_args, checkpoint, device):
            actual = file_sha256(checkpoint)
            expected = strict_contract[
                "source_checkpoint_sha256"]
            if actual != expected:
                raise RuntimeError(
                    "strict deployment contract source-checkpoint mismatch; "
                    "use the original checkpoint recorded by reconstruction")
            return original_build_model(
                saved_args, checkpoint, device)

        original_prepare = runner.prepare_hardware_model

        def prepare_hardware_model_with_contract(
                *args, **kwargs):
            preparation = original_prepare(*args, **kwargs)
            validate_graph_preparation(
                preparation,
                strict_contract["graph_contract"])
            return preparation

        runner.build_model = build_model_with_contract
        runner.prepare_hardware_model = (
            prepare_hardware_model_with_contract)

    original_write_json = runner.write_json

    def write_json_with_semantics(path, payload):
        adapter = active["semantic_adapter"]
        if adapter is not None:
            rows = adapter.semantic_manifest()
            runner.write_csv(
                Path(path).parent /
                "semantic_site_manifest.csv",
                rows)
            roles = sorted(set(
                row["role"] for row in rows))
            operational = sum(int(
                row["meta_operational"]) for row in rows)
            observed = sum(int(
                row["observed"]) for row in rows)
            payload["semantic_quantization"] = {
                "model": adapter.MODEL_NAME,
                "sites": len(rows),
                "roles": roles,
                "observed_sites": observed,
                "operational_sites": operational,
                "strict": int(
                    not options.no_strict_semantic_sites),
                "default_transform": "none",
                "lognp_default": 0,
            }
        if reconstruction is not None:
            payload["reconstruction"] = {
                "manifest": reconstruction["path"],
                "method": reconstruction["method"],
                "weight_bits": reconstruction["weight_bits"],
                "activation_bits": reconstruction[
                    "activation_bits"],
                "activation_policy": reconstruction[
                    "activation_policy"],
                "targets": reconstruction["targets"],
                "strict_deployment_contract": reconstruction[
                    "strict_contract_path"],
                "exact_weight_contract": 1,
                "exact_activation_contract": int(
                    reconstruction["method"] == "qdrop_strict"),
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
