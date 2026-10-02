#!/usr/bin/env python3
"""Evaluate selected NAS and calibrated low-bit models on full NYU val."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.nyu_model_runtime import NYUModelRuntime  # noqa: E402
from scripts import run_depth_nas_targeted_lowbit as targeted  # noqa: E402
from scripts import run_nyu_four_model_int_mixed_precision as quant  # noqa: E402
from scripts import run_nyu_rtn_quantization as rtn_runner  # noqa: E402
from scripts import screen_four_model_depth_nas as nas  # noqa: E402
from scripts import screen_structured_channel_nas as structured  # noqa: E402
from scripts.run_nyu_model_p3t3_search import HardDeploymentP3T3Evaluator  # noqa: E402
from spn_quant.model_contracts import build_model_quantization_contract  # noqa: E402


SELECTED_LOW_BIT = {
    "cspn": "NAS_decoder_stage2_W4_decoder_stage3_A6",
    "dyspn": "NAS_encoder_stage4_W4A4_encoder_stage3_W6",
    "nlspn": "NAS_encoder_stage5_W6_encoder_stage4_A6_encoder_tail_A6",
    "completionformer": "NAS_transformer_fusion_W4_initial_depth_A6",
}


def _selected_changes(model_name: str):
    selected = SELECTED_LOW_BIT[model_name]
    matches = [changes for name, changes
               in targeted.TARGET_ASSIGNMENTS[model_name]
               if name == selected]
    if len(matches) != 1:
        raise ValueError("selected low-bit assignment is not unique")
    return selected, matches[0]


def _summary(payload: dict) -> dict:
    return {
        "mean_sample_rmse_m": float(payload["mean_sample_rmse"]),
        "pooled_rmse_m": float(payload["pooled_rmse"]),
        "sample_count": int(payload["sample_count"]),
    }


def _apply_structured_candidate(model, model_name, candidate_id):
    if candidate_id is None:
        return {}
    return structured.apply_structured_candidate(
        model, model_name, candidate_id)


def run(config_path: Path, model_name: str, candidate_id: str,
        device: str, output: Path, original_audit: Path,
        nas_checkpoint: Path | None = None, shard_id: int = 0,
        shard_count: int = 1,
        structured_candidate: str | None = None) -> Path:
    if output.exists():
        raise FileExistsError("full-validation output already exists: %s" % output)
    config = quant._load_run_config(config_path)
    source = quant._model_source(config_path, config)
    payload = source["models"][model_name]
    original = json.loads(original_audit.read_text(encoding="utf-8"))
    if original["model"] != model_name:
        raise ValueError("original audit model identity mismatch")
    if shard_count < 1 or shard_id < 0 or shard_id >= shard_count:
        raise ValueError("invalid evaluation shard")
    original_rmse = float(
        original["full_validation"]["mean_sample_rmse_m"])

    runtime = NYUModelRuntime.from_args(quant._runtime_args(payload, device))
    quant.configure_runtime_execution(runtime)
    evaluator = None
    try:
        model = runtime.build_model(runtime.device)
        paths = nas.STAGE_PATHS[model_name]
        stages = nas.stage_modules(model, paths)
        depths = dict(nas.CANDIDATE_DEPTHS[model_name])[candidate_id]
        nas.apply_depths(model, paths, stages, depths)
        if nas_checkpoint is not None:
            checkpoint = torch.load(
                str(nas_checkpoint), map_location=runtime.device)
            if checkpoint["model"] != model_name or \
                    checkpoint["candidate_id"] != candidate_id:
                raise ValueError("NAS checkpoint identity mismatch")
            model.load_state_dict(checkpoint["net"], strict=True)
        structured_report = _apply_structured_candidate(
            model, model_name, structured_candidate)

        contract = build_model_quantization_contract(model_name, model)
        calibration = json.loads(Path(
            payload["calibration_metadata"]).read_text(encoding="utf-8"))
        trainset = runtime.build_dataset("train")
        sample = rtn_runner.seeded_sample(
            trainset, int(calibration["calibration_indices"][0]),
            int(runtime.saved_args.seed))
        batch = rtn_runner.batch_from_sample(sample)
        model_args, ground_truth = runtime.model_input(batch, runtime.device)
        del ground_truth
        costs = quant.measure_unit_costs(model, contract, model_args)

        evaluator = HardDeploymentP3T3Evaluator(
            runtime, model, contract, quant._RegistryView(contract.block_names),
            quant._hard_settings(
                payload, device, config["hard_deployment"]))
        evaluation_indices = tuple(
            range(len(evaluator.valset)))[shard_id::shard_count]
        evaluator.replace_evaluation_indices(evaluation_indices)
        reference = _summary(evaluator.reference())

        fp16_units = targeted.ANCHOR_FP16_UNITS.get(model_name, ())
        anchor_assignment = targeted._assignment(
            contract, (), fp16_units)
        anchor_payload = evaluator.evaluate_precision_assignment(
            anchor_assignment, "NAS_W8_ANCHOR")
        anchor = _summary(anchor_payload)

        low_bit_id, changes = _selected_changes(model_name)
        low_bit_assignment = targeted._assignment(
            contract, changes, fp16_units)
        low_bit_payload = evaluator.evaluate_precision_assignment(
            low_bit_assignment, low_bit_id)
        low_bit = _summary(low_bit_payload)

        for stage in (reference, anchor, low_bit):
            stage["relative_to_original_fp32_pct"] = 100.0 * (
                stage["mean_sample_rmse_m"] / original_rmse - 1.0)
            stage["relative_to_nas_fp32_pct"] = 100.0 * (
                stage["mean_sample_rmse_m"] /
                reference["mean_sample_rmse_m"] - 1.0)

        result = {
            "format_version": 1,
            "metric": "mean of per-image RMSE over full NYU validation",
            "model": model_name,
            "original_fp32_rmse_m": original_rmse,
            "nas_candidate": candidate_id,
            "nas_depths": list(depths),
            "nas_checkpoint": "" if nas_checkpoint is None else
                str(nas_checkpoint.resolve()),
            "structured_candidate": (
                "" if structured_candidate is None else
                structured_candidate),
            "structured_pruning": structured_report,
            "calibration_indices": list(calibration["calibration_indices"]),
            "evaluation_count": len(evaluation_indices),
            "evaluation_indices": list(evaluation_indices),
            "shard_id": shard_id,
            "shard_count": shard_count,
            "nas_fp32": reference,
            "nas_w8a8": anchor,
            "selected_low_bit_id": low_bit_id,
            "selected_low_bit": low_bit,
            "selected_low_bit_assignment":
                low_bit_assignment.canonical_payload(),
            "precision_costs": {
                "weight_macs": [list(row) for row in costs.weight_macs],
                "activation_elements": [list(row)
                                        for row in costs.activation_elements],
            },
        }
        output.mkdir(parents=True)
        (output / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        for name, payload in (("nas_w8a8", anchor_payload),
                              ("selected_low_bit", low_bit_payload)):
            rows = payload["sample_rows"]
            with (output / (name + "_samples.csv")).open(
                    "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        return output / "summary.json"
    finally:
        if evaluator is not None:
            evaluator.close()
        runtime.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=nas.MODEL_ORDER, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-audit", type=Path, required=True)
    parser.add_argument("--nas-checkpoint", type=Path)
    parser.add_argument("--structured-candidate")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args(argv)
    print(run(args.config, args.model, args.candidate, args.device,
              args.output, args.original_audit, args.nas_checkpoint,
              args.shard_id, args.shard_count,
              args.structured_candidate))


if __name__ == "__main__":
    main()
