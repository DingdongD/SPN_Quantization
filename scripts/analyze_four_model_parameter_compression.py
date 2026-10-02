#!/usr/bin/env python3
"""Measure NAS parameter counts and mixed-bit packed-weight storage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.nyu_model_runtime import NYUModelRuntime  # noqa: E402
from scripts import run_nyu_four_model_int_mixed_precision as quant  # noqa: E402
from scripts import screen_four_model_depth_nas as nas  # noqa: E402
from scripts import screen_structured_channel_nas as structured  # noqa: E402
from spn_quant.model_contracts import build_model_quantization_contract  # noqa: E402


PROPAGATION_PREFIXES = {
    "cspn": ("post_process_layer",),
    "dyspn": ("dyspn_",),
    "nlspn": ("prop_layer",),
    "completionformer": ("prop_layer",),
}


def apply_structured_candidate(model, model_name, candidate_id):
    if candidate_id is None:
        return {}
    return structured.apply_structured_candidate(
        model, model_name, candidate_id)


def _under(name: str, prefix: str) -> bool:
    if prefix.endswith("_"):
        return name.startswith(prefix)
    return name == prefix or name.startswith(prefix + ".")


def parameter_storage(model: torch.nn.Module, contract,
                      assignment: Mapping[str, object],
                      model_name: str) -> dict:
    unit_bits = {str(name): int(bits) for name, bits in
                 assignment["weight_bits"].items()}
    unit_by_name = {unit.name: unit for unit in contract.search_units}
    if set(unit_bits) != set(unit_by_name):
        raise ValueError("assignment units differ from model contract")

    module_bits = {}
    for unit_name, bits in unit_bits.items():
        for member in unit_by_name[unit_name].members:
            if member in module_bits:
                raise ValueError("weight module has multiple bit assignments")
            module_bits[member] = bits

    parameter_rows = []
    bit_totals = {}
    parameters = dict(model.named_parameters())
    expected_weights = {member + ".weight" for member in module_bits}
    missing = expected_weights - set(parameters)
    if missing:
        raise ValueError("contract weight parameters are missing: %s" %
                         sorted(missing))
    for name, parameter in parameters.items():
        module_name = name.rsplit(".", 1)[0] if "." in name else ""
        if name in expected_weights:
            bits = module_bits[module_name]
            storage_class = "integer_weight"
        elif any(_under(name, prefix)
                 for prefix in PROPAGATION_PREFIXES[model_name]):
            bits = 16
            storage_class = "fp16_propagation"
        else:
            bits = 32
            storage_class = "fp32_auxiliary"
        count = int(parameter.numel())
        bit_totals[bits] = bit_totals.get(bits, 0) + count
        parameter_rows.append({
            "name": name,
            "parameters": count,
            "bits": bits,
            "storage_class": storage_class,
        })
    total_parameters = sum(row["parameters"] for row in parameter_rows)
    total_bits = sum(row["parameters"] * row["bits"]
                     for row in parameter_rows)
    return {
        "parameters": total_parameters,
        "packed_weight_bits": total_bits,
        "packed_weight_bytes": total_bits / 8.0,
        "parameter_weighted_bits": total_bits / float(total_parameters),
        "parameter_counts_by_bits": {
            str(bits): count for bits, count in sorted(bit_totals.items())},
        "parameter_rows": parameter_rows,
    }


def run(config_path: Path, model_name: str, candidate_id: str,
        device: str, fullval_summary: Path, output: Path,
        nas_checkpoint: Path | None = None,
        structured_candidate: str | None = None) -> Path:
    config = quant._load_run_config(config_path)
    source = quant._model_source(config_path, config)
    payload = source["models"][model_name]
    selected = json.loads(fullval_summary.read_text(encoding="utf-8"))
    if selected["model"] != model_name or \
            selected["nas_candidate"] != candidate_id:
        raise ValueError("full-validation identity mismatch")
    runtime = NYUModelRuntime.from_args(quant._runtime_args(payload, device))
    quant.configure_runtime_execution(runtime)
    try:
        model = runtime.build_model(runtime.device)
        original_parameters = nas.parameter_count(model)
        paths = nas.STAGE_PATHS[model_name]
        stages = nas.stage_modules(model, paths)
        depths = dict(nas.CANDIDATE_DEPTHS[model_name])[candidate_id]
        nas.apply_depths(model, paths, stages, depths)
        if nas_checkpoint is not None:
            checkpoint = torch.load(
                str(nas_checkpoint), map_location=runtime.device)
            model.load_state_dict(checkpoint["net"], strict=True)
        structured_report = apply_structured_candidate(
            model, model_name, structured_candidate)
        contract = build_model_quantization_contract(model_name, model)
        storage = parameter_storage(
            model, contract, selected["selected_low_bit_assignment"],
            model_name)
        original_fp32_bits = original_parameters * 32
        nas_fp32_bits = storage["parameters"] * 32
        result = {
            "format_version": 1,
            "model": model_name,
            "nas_candidate": candidate_id,
            "structured_candidate": (
                "" if structured_candidate is None else
                structured_candidate),
            "structured_pruning": structured_report,
            "original_fp32_parameters": original_parameters,
            "nas_parameters": storage["parameters"],
            "nas_parameter_reduction_pct": 100.0 * (
                1.0 - storage["parameters"] / float(original_parameters)),
            "nas_only_parameter_compression_ratio":
                original_parameters / float(storage["parameters"]),
            "nas_fp32_bytes": nas_fp32_bits / 8.0,
            **storage,
            "quantization_storage_compression_vs_nas_fp32":
                nas_fp32_bits / float(storage["packed_weight_bits"]),
            "total_storage_compression_vs_original_fp32":
                original_fp32_bits / float(storage["packed_weight_bits"]),
            "total_storage_reduction_vs_original_fp32_pct": 100.0 * (
                1.0 - storage["packed_weight_bits"] /
                float(original_fp32_bits)),
            "storage_policy": {
                "integer_conv_linear_weights": "assigned W4/W6/W8",
                "propagation_parameters": "FP16",
                "unassigned_bias_norm_and_protected_parameters": "FP32",
                "scale_and_container_overhead_included": False,
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
        return output
    finally:
        runtime.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=nas.MODEL_ORDER, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--fullval-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nas-checkpoint", type=Path)
    parser.add_argument("--structured-candidate")
    args = parser.parse_args(argv)
    print(run(args.config, args.model, args.candidate, args.device,
              args.fullval_summary, args.output, args.nas_checkpoint,
              args.structured_candidate))


if __name__ == "__main__":
    main()
