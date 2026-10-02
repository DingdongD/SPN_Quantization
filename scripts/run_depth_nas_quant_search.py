#!/usr/bin/env python3
"""Run the official calibrated quantization search on one depth-NAS subnet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.nyu_model_runtime import NYUModelRuntime  # noqa: E402
from scripts import run_nyu_four_model_int_mixed_precision as quant  # noqa: E402
from scripts import run_nyu_rtn_quantization as rtn_runner  # noqa: E402
from scripts import screen_four_model_depth_nas as nas  # noqa: E402
from scripts.run_nyu_model_p3t3_search import HardDeploymentP3T3Evaluator  # noqa: E402
from spn_quant.model_contracts import build_model_quantization_contract  # noqa: E402


def candidate_depths(model_name: str, candidate_id: str) -> tuple[int, ...]:
    matches = [depths for name, depths in nas.CANDIDATE_DEPTHS[model_name]
               if name == candidate_id]
    if len(matches) != 1:
        raise ValueError("unknown NAS candidate %s/%s" %
                         (model_name, candidate_id))
    return tuple(int(value) for value in matches[0])


def run(config_path: Path, model_name: str, candidate_id: str,
        device: str, phase: str, output: Path) -> Path:
    if output.exists():
        raise FileExistsError("quantization output already exists: %s" % output)
    config = quant._load_run_config(config_path)
    source = quant._model_source(config_path, config)
    model_payload = source["models"][model_name]
    runtime = NYUModelRuntime.from_args(quant._runtime_args(model_payload, device))
    quant.configure_runtime_execution(runtime)
    evaluator = None
    try:
        model = runtime.build_model(runtime.device)
        paths = nas.STAGE_PATHS[model_name]
        full_stages = nas.stage_modules(model, paths)
        depths = candidate_depths(model_name, candidate_id)
        nas.apply_depths(model, paths, full_stages, depths)
        contract = build_model_quantization_contract(model_name, model)
        calibration = json.loads(Path(
            model_payload["calibration_metadata"]).read_text(encoding="utf-8"))
        first_index = int(calibration["calibration_indices"][0])
        trainset = runtime.build_dataset("train")
        sample = rtn_runner.seeded_sample(
            trainset, first_index, int(runtime.saved_args.seed))
        batch = rtn_runner.batch_from_sample(sample)
        model_args, ground_truth = runtime.model_input(batch, runtime.device)
        del ground_truth
        costs = quant.measure_unit_costs(model, contract, model_args)
        evaluator = HardDeploymentP3T3Evaluator(
            runtime, model, contract, quant._RegistryView(contract.block_names),
            quant._hard_settings(model_payload, device,
                                 config["hard_deployment"]),
        )
        result = quant.run_constrained_search(
            contract=contract,
            costs=costs,
            evaluator=evaluator,
            settings=quant._search_settings(config["search"]),
            boundary_order=tuple(config["boundary_order"][model_name]),
            interaction_pairs=tuple(
                tuple(pair) for pair in config["interaction_pairs"][model_name]),
            phase=phase,
        )
        output.mkdir(parents=True)
        quant.write_search_artifacts(output, result)
        (output / "fp32_reference.json").write_text(json.dumps(
            quant.reference_artifact(result), indent=2,
            sort_keys=True) + "\n", encoding="utf-8")
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = Path(model_payload["checkpoint"]).resolve()
        manifest.update({
            "checkpoint": {"path": str(checkpoint),
                           "sha256": quant._file_sha256(checkpoint)},
            "architecture_class": type(model).__name__,
            "nas_candidate_id": candidate_id,
            "nas_depths": list(depths),
            "nas_stage_paths": list(paths),
            "calibration_indices": list(calibration["calibration_indices"]),
            "evaluation_indices": list(model_payload["evaluation_indices"]),
            "propagation_iterations": int(model_payload["propagation_iterations"]),
            "propagation_dtype": "fp16",
            "precision_costs": {
                "weight_macs": [list(row) for row in costs.weight_macs],
                "activation_elements": [list(row)
                                        for row in costs.activation_elements],
            },
        })
        manifest_path.write_text(json.dumps(
            manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest_path
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
    parser.add_argument("--phase", choices=("anchors", "ptq-search"),
                        default="anchors")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run(args.config, args.model, args.candidate, args.device,
              args.phase, args.output))


if __name__ == "__main__":
    main()
