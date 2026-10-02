#!/usr/bin/env python3
"""Measure prioritized low-bit assignments on selected depth-NAS subnets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.nyu_model_runtime import NYUModelRuntime  # noqa: E402
from scripts import run_nyu_four_model_int_mixed_precision as quant  # noqa: E402
from scripts import run_nyu_rtn_quantization as rtn_runner  # noqa: E402
from scripts import screen_four_model_depth_nas as nas  # noqa: E402
from scripts.run_nyu_model_p3t3_search import HardDeploymentP3T3Evaluator  # noqa: E402
from spn_quant.model_contracts import build_model_quantization_contract  # noqa: E402


TARGET_ASSIGNMENTS = {
    "cspn": (
        ("NAS_decoder_stage2_W4", (("decoder_stage2", 4, 8),)),
        ("NAS_decoder_stage3_A6", (("decoder_stage3", 8, 6),)),
        ("NAS_decoder_stage2_W4_decoder_stage3_A6",
         (("decoder_stage2", 4, 8), ("decoder_stage3", 8, 6))),
    ),
    "dyspn": (
        ("NAS_encoder_stage4_W4A4", (("encoder_stage4", 4, 4),)),
        ("NAS_encoder_stage3_W6", (("encoder_stage3", 6, 8),)),
        ("NAS_encoder_stage4_W4A4_encoder_stage3_W6",
         (("encoder_stage4", 4, 4), ("encoder_stage3", 6, 8))),
    ),
    "completionformer": (
        ("NAS_transformer_fusion_W4", (("transformer_fusion", 4, 8),)),
        ("NAS_initial_depth_A6", (("initial_depth", 8, 6),)),
        ("NAS_transformer_fusion_W4_initial_depth_A6",
         (("transformer_fusion", 4, 8), ("initial_depth", 8, 6))),
    ),
    "nlspn": (
        ("NAS_encoder_stage5_W6", (("encoder_stage5", 6, 8),)),
        ("NAS_encoder_stage4_A6", (("encoder_stage4", 8, 6),)),
        ("NAS_encoder_tail_A6", (("encoder_tail", 8, 6),)),
        ("NAS_encoder_stage5_W6_encoder_stage4_A6_encoder_tail_A6",
         (("encoder_stage5", 6, 8), ("encoder_stage4", 8, 6),
          ("encoder_tail", 8, 6))),
    ),
}


ANCHOR_FP16_UNITS = {"nlspn": ("initial_depth", "early_boundary")}


def _assignment(contract, changes, fp16_units=()):
    assignment = quant.uniform_assignment(contract, 8, 8)
    for unit in fp16_units:
        assignment = quant.promote_fp16(assignment, contract, unit)
    for unit, weight_bits, activation_bits in changes:
        assignment = quant._replace_unit(
            assignment, unit, weight_bits, activation_bits, False, contract)
    return assignment


def run(config_path: Path, baseline_ptq_root: Path, model_name: str,
        candidate_id: str, device: str, output: Path,
        nas_checkpoint: Path | None = None) -> Path:
    if model_name not in TARGET_ASSIGNMENTS:
        raise ValueError("no targeted assignments for model: %s" % model_name)
    if output.exists():
        raise FileExistsError("targeted output already exists: %s" % output)
    config = quant._load_run_config(config_path)
    source = quant._model_source(config_path, config)
    payload = source["models"][model_name]
    original_manifest = json.loads((
        baseline_ptq_root / model_name / "manifest.json").read_text(
            encoding="utf-8"))
    original_reference = float(original_manifest["reference_pooled_rmse"])
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
            checkpoint = torch.load(str(nas_checkpoint), map_location=runtime.device)
            if checkpoint["model"] != model_name or \
                    checkpoint["candidate_id"] != candidate_id:
                raise ValueError("NAS checkpoint identity mismatch")
            model.load_state_dict(checkpoint["net"], strict=True)
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
            quant._hard_settings(payload, device, config["hard_deployment"]),
        )
        reference = evaluator.reference()
        nas_reference = float(reference["pooled_rmse"])
        sample_count = int(reference["sample_count"])
        fp16_units = ANCHOR_FP16_UNITS.get(model_name, ())
        anchor = _assignment(contract, (), fp16_units)
        candidates = [("NAS_W8_ANCHOR", anchor)]
        candidates.extend((name, _assignment(contract, changes, fp16_units))
                          for name, changes in TARGET_ASSIGNMENTS[model_name])
        records = [quant._measure(
            name, "targeted", assignment, evaluator, costs,
            nas_reference, sample_count)
            for name, assignment in candidates]
        rows = []
        for record in records:
            measured = record.candidate
            rows.append({
                "model": model_name,
                "nas_candidate": candidate_id,
                "candidate_id": measured.candidate_id,
                "pooled_rmse_m": measured.pooled_rmse,
                "relative_to_nas_fp_pct": 100.0 * measured.relative_loss,
                "relative_to_original_fp_pct": 100.0 * (
                    measured.pooled_rmse / original_reference - 1.0),
                "average_weight_bits": measured.average_weight_bits,
                "average_activation_bits": measured.average_activation_bits,
                "fp16_mac_fraction": measured.fp16_mac_fraction,
                "fp16_activation_fraction": measured.fp16_activation_fraction,
                "valid": record.valid,
                "passes_total_2pct": record.valid and
                    measured.pooled_rmse <= original_reference * 1.02,
            })
        output.mkdir(parents=True)
        with (output / "targeted_lowbit.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (output / "assignments.json").write_text(json.dumps({
            record.candidate.candidate_id:
                record.candidate.assignment.canonical_payload()
            for record in records
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output / "manifest.json").write_text(json.dumps({
            "format_version": 1,
            "model": model_name,
            "nas_candidate": candidate_id,
            "nas_depths": list(depths),
            "original_fp32_rmse_m": original_reference,
            "nas_fp32_rmse_m": nas_reference,
            "evaluation_indices": list(payload["evaluation_indices"]),
            "calibration_indices": list(calibration["calibration_indices"]),
            "propagation_dtype": "fp16",
            "nas_checkpoint": "" if nas_checkpoint is None else
                str(nas_checkpoint.resolve()),
            "rows": rows,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output / "targeted_lowbit.csv"
    finally:
        if evaluator is not None:
            evaluator.close()
        runtime.close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-ptq-root", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(TARGET_ASSIGNMENTS), required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nas-checkpoint", type=Path)
    args = parser.parse_args(argv)
    print(run(args.config, args.baseline_ptq_root, args.model,
              args.candidate, args.device, args.output,
              args.nas_checkpoint))


if __name__ == "__main__":
    main()
