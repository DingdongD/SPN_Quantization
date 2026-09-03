#!/usr/bin/env python3
"""Measure NLSPN propagation storage dtype under a uniform W8A8 model."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.nyu_model_runtime import NYUModelRuntime  # noqa: E402
from scripts.run_nyu_model_p3t3_search import (  # noqa: E402
    HardDeploymentP3T3Evaluator,
    HardDeploymentSettings,
    _read_activation_cost_rows,
    _read_weight_cost_rows,
)
from spn_quant import mixed_precision  # noqa: E402
from spn_quant.experiment_config import (  # noqa: E402
    load_selected_quantization_config,
)
from spn_quant.model_contracts import (  # noqa: E402
    build_model_quantization_contract,
)


MODE_NAMES = (
    "PA_W8A8",
    "W8A8_FP32_PROP",
    "W8A8_BF16_STATE",
    "W8A8_FP16_STATE",
)
FLOAT_STATE_NAMES = {
    "W8A8_FP32_PROP": "fp32",
    "W8A8_BF16_STATE": "bf16",
    "W8A8_FP16_STATE": "fp16",
}


def _metrics(prediction: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, float]:
    valid = torch.isfinite(ground_truth) & (ground_truth > 1e-4)
    if int(valid.sum().item()) <= 0:
        raise ValueError("evaluation sample has no valid depth pixels")
    values = prediction[valid]
    target = ground_truth[valid]
    if not bool(torch.isfinite(values).all().item()):
        raise RuntimeError("quantized prediction is non-finite")
    if bool((values <= 1e-4).any().item()):
        raise RuntimeError("quantized prediction is non-positive")
    difference = values.double() - target.double()
    inverse_difference = values.double().reciprocal() - \
        target.double().reciprocal()
    return {
        "squared_error_sum": float(difference.square().sum().item()),
        "absolute_error_sum": float(difference.abs().sum().item()),
        "abs_rel_sum": float(
            (difference.abs() / target.double()).sum().item()),
        "irmse_squared_sum": float(inverse_difference.square().sum().item()),
        "valid_pixels": int(valid.sum().item()),
    }


def _aggregate(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    valid_pixels = sum(int(row["valid_pixels"]) for row in rows)
    if valid_pixels <= 0:
        raise ValueError("metric rows contain no valid pixels")
    squared_error_sum = sum(row["squared_error_sum"] for row in rows)
    absolute_error_sum = sum(row["absolute_error_sum"] for row in rows)
    abs_rel_sum = sum(row["abs_rel_sum"] for row in rows)
    irmse_squared_sum = sum(row["irmse_squared_sum"] for row in rows)
    sample_rmse = tuple(
        (row["sample_index"], row["squared_error_sum"] / row["valid_pixels"])
        for row in rows)
    return {
        "pooled_rmse": (squared_error_sum / valid_pixels) ** 0.5,
        "mean_sample_rmse": sum(
            value ** 0.5 for index, value in sample_rmse) / len(sample_rmse),
        "pooled_mae": absolute_error_sum / valid_pixels,
        "pooled_abs_rel": abs_rel_sum / valid_pixels,
        "pooled_irmse": (irmse_squared_sum / valid_pixels) ** 0.5,
        "valid_pixels": valid_pixels,
    }


def _state_rows(mode: str, states: Sequence[torch.Tensor],
                reference: Sequence[torch.Tensor]) -> List[Dict[str, object]]:
    if len(states) != len(reference):
        raise RuntimeError("propagation iteration count differs from FP32 reference")
    rows = []
    for iteration, (actual, expected) in enumerate(zip(states, reference), 1):
        difference = actual.float() - expected.float()
        rows.append({
            "mode": mode,
            "iteration": iteration,
            "state_dtype": str(actual.dtype).replace("torch.", ""),
            "state_rmse_vs_fp32_prop": float(
                difference.float().square().mean().sqrt().item()),
            "state_max_abs_vs_fp32_prop": float(
                difference.float().abs().max().item()),
            "state_absmax": float(actual.float().abs().max().item()),
        })
    return rows


def _build_settings(model_config, launch_payload, device: str):
    hard = launch_payload["hard_deployment"]
    return HardDeploymentSettings(
        device=device,
        calibration_metadata=model_config.calibration_metadata,
        calibration_count=model_config.calibration_count,
        evaluation_indices=model_config.evaluation_indices,
        base_weight_bits=8,
        base_activation_bits=8,
        promotion_weight_bits=8,
        promotion_activation_bits=8,
        fold_conv_bn=bool(hard["fold_conv_bn"]),
        fold_max_error=float(hard["fold_max_error"]),
        joint_clip_factors=tuple(float(value) for value in hard[
            "joint_clip_factors"]),
        joint_search_rounds=int(hard["joint_search_rounds"]),
        joint_cache_sample_limit=int(hard["joint_cache_sample_limit"]),
        joint_cache_byte_limit=int(hard["joint_cache_byte_limit"]),
    )


def _evaluate_mode(evaluator: HardDeploymentP3T3Evaluator, mode: str):
    evaluator.configure_uniform_w8a8()
    if mode in FLOAT_STATE_NAMES:
        evaluator.propagation_adapter.configure_float(FLOAT_STATE_NAMES[mode])
    with torch.no_grad():
        prediction, ground_truth = evaluator._forward(
            evaluator.evaluation_batch)
    rows = []
    for position, (sample_index, batch) in enumerate(
            evaluator.evaluation_batches):
        del batch
        metrics = _metrics(prediction[position], ground_truth[position])
        metrics["mode"] = mode
        metrics["sample_index"] = int(sample_index)
        rows.append(metrics)
    return rows, tuple(evaluator.propagation_adapter.last_states()), tuple(
        evaluator.propagation_adapter.statistics())


def run(config_path: Path, launch_path: Path, model_name: str,
        device: str, output: Path) -> Path:
    if model_name != "nlspn":
        raise ValueError("this ablation only supports nlspn")
    if output.exists():
        raise FileExistsError("ablation output already exists: %s" % output)
    selected = load_selected_quantization_config(config_path)
    model_config = next(
        model for model in selected.models if model.model == model_name)
    configured_device = model_config.device
    model_config = replace(model_config, device=device)
    launch_payload = json.loads(Path(launch_path).read_text(encoding="utf-8"))
    model_inputs = launch_payload["model_inputs"][model_name]
    costs = mixed_precision.CostBasis(
        weight_macs=_read_weight_cost_rows(Path(model_inputs[
            "weight_cost_rows"])),
        activation_elements=_read_activation_cost_rows(Path(model_inputs[
            "activation_cost_rows"])),
    )
    runtime = NYUModelRuntime.from_config(model_config)
    model = runtime.build_model(runtime.device)
    contract = build_model_quantization_contract(model_name, model)
    registry = mixed_precision.build_registry(contract, costs)
    evaluator = HardDeploymentP3T3Evaluator(
        runtime, model, contract, registry,
        _build_settings(model_config, launch_payload, device))
    output.mkdir(parents=True)
    mode_rows = []
    state_rows = []
    propagation_statistics = {}
    states_by_mode = {}
    reference_states = None
    try:
        for mode in MODE_NAMES:
            rows, states, statistics = _evaluate_mode(evaluator, mode)
            mode_rows.extend(rows)
            states_by_mode[mode] = states
            propagation_statistics[mode] = statistics
            if mode == "W8A8_FP32_PROP":
                reference_states = states
    finally:
        evaluator.close()
        runtime.close()
    if reference_states is None or set(states_by_mode) != set(MODE_NAMES):
        raise RuntimeError("FP32 propagation reference was not evaluated")
    for mode in MODE_NAMES:
        state_rows.extend(_state_rows(
            mode, states_by_mode[mode], reference_states))
    summary = []
    for mode in MODE_NAMES:
        rows = tuple(row for row in mode_rows if row["mode"] == mode)
        aggregate = _aggregate(rows)
        reference = _aggregate(tuple(
            row for row in mode_rows if row["mode"] == "W8A8_FP32_PROP"))
        aggregate["mode"] = mode
        aggregate["delta_vs_fp32_prop"] = \
            aggregate["pooled_rmse"] - reference["pooled_rmse"]
        aggregate["relative_delta_vs_fp32_prop"] = \
            aggregate["pooled_rmse"] / reference["pooled_rmse"] - 1.0
        summary.append(aggregate)
    (output / "metadata.json").write_text(json.dumps({
        "model": model_name,
        "device": device,
        "configured_device": configured_device,
        "calibration_count": model_config.calibration_count,
        "evaluation_indices": list(model_config.evaluation_indices),
        "ordinary_quantization": "uniform W8A8",
        "propagation_modes": list(MODE_NAMES),
        "reference_mode": "W8A8_FP32_PROP",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = tuple(summary[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    with (output / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = tuple(mode_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(mode_rows)
    with (output / "propagation_state_metrics.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        fields = tuple(state_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(state_rows)
    (output / "propagation_statistics.json").write_text(
        json.dumps(propagation_statistics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main():
    args = build_parser().parse_args(tuple(sys.argv[1:]))
    print(run(args.config, args.launch_config, args.model, args.device,
              args.output))


if __name__ == "__main__":
    main()
