#!/usr/bin/env python3
"""Evaluate a fine-tuned depth-NAS checkpoint against its original baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.nyu_model_runtime import NYUModelRuntime  # noqa: E402
from scripts import run_nyu_four_model_int_mixed_precision as quant  # noqa: E402
from scripts import screen_four_model_depth_nas as nas  # noqa: E402


def run(config_path: Path, model_name: str, candidate_id: str,
        checkpoint: Path, device: str, output: Path) -> Path:
    if output.exists():
        raise FileExistsError("evaluation output already exists: %s" % output)
    config = quant._load_run_config(config_path)
    source = quant._model_source(config_path, config)
    payload = source["models"][model_name]
    runtime = NYUModelRuntime.from_args(quant._runtime_args(payload, device))
    quant.configure_runtime_execution(runtime)
    try:
        model = runtime.build_model(runtime.device)
        baseline_parameters = nas.parameter_count(model)
        dataset = runtime.build_dataset("val")
        indices = tuple(int(value) for value in payload["evaluation_indices"])
        baseline, references = nas.evaluate_candidate(
            runtime, model, dataset, indices, None)
        paths = nas.STAGE_PATHS[model_name]
        full_stages = nas.stage_modules(model, paths)
        depths = dict(nas.CANDIDATE_DEPTHS[model_name])[candidate_id]
        nas.apply_depths(model, paths, full_stages, depths)
        checkpoint_payload = torch.load(str(checkpoint), map_location=runtime.device)
        if checkpoint_payload["model"] != model_name or \
                checkpoint_payload["candidate_id"] != candidate_id:
            raise ValueError("fine-tuned checkpoint identity mismatch")
        model.load_state_dict(checkpoint_payload["net"], strict=True)
        candidate, _ = nas.evaluate_candidate(
            runtime, model, dataset, indices, references)
        result = {
            "format_version": 1,
            "model": model_name,
            "candidate_id": candidate_id,
            "checkpoint": str(checkpoint.resolve()),
            "depths": list(depths),
            "evaluation_indices": list(indices),
            "baseline": baseline,
            "candidate": candidate,
            "relative_rmse_pct": 100.0 * (
                candidate["pooled_rmse_m"] / baseline["pooled_rmse_m"] - 1.0),
            "parameter_reduction_pct": 100.0 * (
                1.0 - nas.parameter_count(model) /
                float(baseline_parameters)),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
        return output
    finally:
        runtime.close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=nas.MODEL_ORDER, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run(args.config, args.model, args.candidate, args.checkpoint,
              args.device, args.output))


if __name__ == "__main__":
    main()
