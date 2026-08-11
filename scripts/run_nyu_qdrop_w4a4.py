#!/usr/bin/env python3
"""Orchestrate and aggregate strict QDrop W4A4 NYU evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_nyu_predictions import load_run_args  # noqa: E402
from scripts.run_nyu_edge_quantization import (  # noqa: E402
    load_reconstruction_manifest,
)
from spn_quant.deployment_contract import file_sha256  # noqa: E402
from spn_quant.qdrop_config import load_qdrop_config  # noqa: E402


MODEL_ORDER = ("cspn", "dyspn", "nlspn", "completionformer")
BASE_METHODS = ("fp32", "rtn", "brecq")
QDROP_EVALUATION_BACKEND = "propagation"
QDROP_EVALUATION_CONFIG = "PA_Constraint"
MAX_MODELS_PER_WAVE = len(MODEL_ORDER)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    rows = list(rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def write_json(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def build_run_matrix(phase, seeds, devices):
    seeds = tuple(int(seed) for seed in seeds)
    devices = tuple(str(device) for device in devices)
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("QDrop formal matrix requires three unique seeds")
    if len(devices) != len(MODEL_ORDER):
        raise ValueError("QDrop matrix requires one device per model")
    if phase not in ("probability-search", "formal", "evaluate"):
        raise ValueError("unknown QDrop orchestration phase")
    active_seeds = (seeds[0],) \
        if phase == "probability-search" else seeds
    return tuple({
        "phase": phase,
        "model": model,
        "seed": seed,
        "device": devices[model_index],
        "weight_bits": 4,
        "activation_bits": 4,
    } for model_index, model in enumerate(MODEL_ORDER)
      for seed in active_seeds)


def build_execution_waves(rows):
    rows = tuple(rows)
    seed_order = tuple(dict.fromkeys(int(row["seed"]) for row in rows))
    waves = []
    for seed in seed_order:
        wave = tuple(row for row in rows if int(row["seed"]) == seed)
        if set(row["model"] for row in wave) != set(MODEL_ORDER):
            raise ValueError("QDrop execution wave does not cover all models")
        if len(set(row["device"] for row in wave)) != len(wave):
            raise ValueError("QDrop execution wave repeats a CUDA device")
        waves.extend(
            wave[start:start + MAX_MODELS_PER_WAVE]
            for start in range(0, len(wave), MAX_MODELS_PER_WAVE))
    return tuple(waves)


def validate_aligned_sample_rows(rows, model, indices, seeds):
    indices = tuple(int(index) for index in indices)
    seeds = tuple(int(seed) for seed in seeds)
    if len(set(indices)) != len(indices):
        raise ValueError("QDrop evaluation indices contain duplicates")
    expected = set()
    for method in BASE_METHODS:
        for index in indices:
            expected.add((method, 0, index))
    for seed in seeds:
        for index in indices:
            expected.add(("qdrop", seed, index))
    observed = []
    for row in rows:
        if row["model"] != model:
            raise ValueError("QDrop sample row model mismatch")
        key = (
            str(row["method"]), int(row["seed"]),
            int(row["sample_index"]))
        observed.append(key)
        values = np.asarray([
            float(row["RMSE"]), float(row["MAE"]),
            float(row["ABS_REL"]),
        ], dtype=np.float64)
        if not np.isfinite(values).all() or \
                int(row["nonfinite_pixels"]) != 0:
            raise ValueError("QDrop sample metrics contain non-finite values")
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("QDrop methods do not share the required samples")


def validate_qdrop_layer_rows(rows):
    exact = [
        row for row in rows
        if row["kind"] == "exact_activation_contract"]
    if not exact:
        raise ValueError("QDrop layer metrics contain no exact A4 sites")
    for row in exact:
        if int(row["calls"]) <= 0 or int(row["numel"]) <= 0:
            raise ValueError("QDrop exact A4 site was not executed")
        ratios = np.asarray([
            float(row["zero_code_rate"]),
            float(row["saturation_rate"]),
        ], dtype=np.float64)
        if not np.isfinite(ratios).all() or \
                bool(np.any(ratios < 0.0)) or bool(np.any(ratios > 1.0)):
            raise ValueError("QDrop exact A4 ratios are invalid")
        if np.isnan(float(row["sqnr_db"])):
            raise ValueError("QDrop exact A4 SQNR is invalid")


def aggregate_seed_rows(sample_rows):
    keys = sorted(set(
        (row["model"], row["method"], int(row["seed"]))
        for row in sample_rows))
    rows = []
    for model, method, seed in keys:
        current = [
            row for row in sample_rows
            if row["model"] == model and row["method"] == method and
            int(row["seed"]) == seed]
        rows.append({
            "model": model,
            "method": method,
            "seed": seed,
            "samples": len(current),
            "mean_rmse": float(np.mean([
                float(row["RMSE"]) for row in current])),
            "mean_mae": float(np.mean([
                float(row["MAE"]) for row in current])),
            "mean_abs_rel": float(np.mean([
                float(row["ABS_REL"]) for row in current])),
            "nonfinite_ratio": float(np.mean([
                int(row["nonfinite_pixels"]) > 0 for row in current])),
        })
    return rows


def aggregate_model_metrics(seed_rows, seeds):
    seeds = tuple(int(seed) for seed in seeds)
    models = sorted(set(row["model"] for row in seed_rows))
    summaries = []
    acceptance = []
    for model in models:
        qdrop = sorted(
            (row for row in seed_rows
             if row["model"] == model and row["method"] == "qdrop"),
            key=lambda row: int(row["seed"]))
        if tuple(int(row["seed"]) for row in qdrop) != seeds:
            raise ValueError("QDrop model summary has an incomplete seed set")
        baselines = {}
        for method in BASE_METHODS:
            matches = [
                row for row in seed_rows
                if row["model"] == model and row["method"] == method]
            if len(matches) != 1:
                raise ValueError("QDrop baseline summary is incomplete")
            baselines[method] = matches[0]
        values = np.asarray(
            [float(row["mean_rmse"]) for row in qdrop],
            dtype=np.float64)
        nonfinite = max(float(row["nonfinite_ratio"]) for row in qdrop)
        mean_rmse = float(values.mean())
        summaries.append({
            "model": model,
            "method": "qdrop",
            "seeds": len(qdrop),
            "mean_rmse": mean_rmse,
            "std_rmse": float(values.std(ddof=0)),
            "min_rmse": float(values.min()),
            "max_rmse": float(values.max()),
            "nonfinite_ratio": nonfinite,
        })
        fp32 = float(baselines["fp32"]["mean_rmse"])
        brecq = float(baselines["brecq"]["mean_rmse"])
        acceptance.append({
            "model": model,
            "finite": int(nonfinite == 0.0),
            "better_than_brecq": int(mean_rmse < brecq),
            "within_fp32_10pct": int(
                (mean_rmse - fp32) / fp32 <= 0.10),
            "fp32_rmse": fp32,
            "brecq_rmse": brecq,
            "qdrop_mean_rmse": mean_rmse,
            "delta_vs_brecq": mean_rmse - brecq,
            "relative_fp32_degradation": (mean_rmse - fp32) / fp32,
        })
    return summaries, acceptance


def _mapping(values, label):
    rows = {}
    for value in values:
        parts = str(value).split("=", 1)
        if len(parts) != 2 or parts[0] not in MODEL_ORDER or not parts[1]:
            raise ValueError("invalid %s mapping: %s" % (label, value))
        if parts[0] in rows:
            raise ValueError("duplicate %s mapping: %s" % (label, parts[0]))
        rows[parts[0]] = Path(parts[1]).resolve()
    if set(rows) != set(MODEL_ORDER):
        raise ValueError("%s mappings must cover all models" % label)
    return rows


def _validate_inputs(model_runs, brecq_manifests, baseline_root,
                     dcn_extension):
    dcn = Path(dcn_extension).resolve()
    if not dcn.is_file() or not dcn.name.startswith("DCN") or \
            dcn.suffix != ".so":
        raise FileNotFoundError("verified DCN extension is missing: %s" % dcn)
    baseline_root = Path(baseline_root).resolve()
    for model in MODEL_ORDER:
        run_dir = model_runs[model]
        checkpoint = run_dir / "best.pt"
        if not (run_dir / "args.json").is_file() or not checkpoint.is_file():
            raise FileNotFoundError("incomplete QDrop model run: %s" % run_dir)
        saved_args = load_run_args(run_dir)
        if saved_args.model != model:
            raise RuntimeError("QDrop run model mismatch: %s" % model)
        baseline_metadata = read_json(
            baseline_root / "stress" / "rtn" / model / "metadata.json")
        expected_hash = baseline_metadata[
            "model_provenance"]["checkpoint_sha256"]
        if file_sha256(checkpoint) != expected_hash:
            raise RuntimeError("QDrop checkpoint hash mismatch: %s" % model)
        reconstruction = load_reconstruction_manifest(
            str(brecq_manifests[model]))
        if reconstruction["method"] != "brecq_strict":
            raise RuntimeError("strict BRECQ manifest mismatch: %s" % model)
        contract = reconstruction["strict_contract"]
        if contract["source_checkpoint_sha256"] != expected_hash:
            raise RuntimeError("BRECQ checkpoint hash mismatch: %s" % model)
    return dcn


def _reconstruction_command(row, args, model_runs):
    model = row["model"]
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_nyu_qdrop_reconstruction.py"),
        "--config", str(Path(args.config).resolve()),
        "--run-dir", str(model_runs[model]),
        "--checkpoint", "best.pt",
        "--data-root", str(Path(args.data_root).resolve()),
        "--model", model,
        "--phase", row["phase"],
        "--seed", str(row["seed"]),
        "--out-dir", str(Path(args.out_dir).resolve() / model),
    ]


def _prepare_process(row, environment):
    current = dict(environment)
    current["CUDA_VISIBLE_DEVICES"] = row["device"].split(":", 1)[1]
    command = list(row["command"])
    if "--device" in command:
        command[command.index("--device") + 1] = "cuda:0"
    return command, current


def run_execution_wave(wave, environment):
    running = []
    for row in wave:
        command, current = _prepare_process(row, environment)
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=current)
        running.append((row, command, process))
    while running:
        completed = [
            entry for entry in running if entry[2].poll() is not None]
        failed = [entry for entry in completed if entry[2].returncode != 0]
        if failed:
            for _, _, process in running:
                if process.poll() is None:
                    process.terminate()
            for _, _, process in running:
                process.wait()
            row, command, process = failed[0]
            raise subprocess.CalledProcessError(
                process.returncode, command,
                output="QDrop command failed for %s seed %d" %
                (row["model"], row["seed"]))
        running = [entry for entry in running if entry not in completed]
        if running and not completed:
            time.sleep(0.2)


def _evaluation_command(row, args, model_runs, baseline_root):
    model = row["model"]
    qdrop_root = Path(args.out_dir).resolve() / model / \
        ("formal_seed_%d" % row["seed"])
    output = Path(args.out_dir).resolve() / "evaluation" / "qdrop" / \
        model / ("seed_%d" % row["seed"])
    sample_metrics = baseline_root / "stress" / "rtn" / model / \
        "sample_metrics.csv"
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_nyu_edge_quantization.py"),
        "--reconstruction-manifest",
        str(qdrop_root / "qdrop_strict_manifest.json"),
        "--run-dir", str(model_runs[model]),
        "--checkpoint", "best.pt",
        "--sample-metrics", str(sample_metrics),
        "--data-root", str(Path(args.data_root).resolve()),
        "--out-dir", str(output),
        "--device", row["device"],
        "--seed", str(config.formal.evaluation_seed),
        "--calibration-samples", "64",
        "--max-eval-samples", "64",
        "--config-names", "FP32", QDROP_EVALUATION_CONFIG,
        "--export-prediction-configs", "FP32", QDROP_EVALUATION_CONFIG,
        "--quant-backend", QDROP_EVALUATION_BACKEND,
    ]


def _baseline_rows(baseline_root, model, method, config):
    path = baseline_root / "stress" / method / model / "sample_metrics.csv"
    rows = [row for row in read_csv(path) if row["config"] == config]
    return rows


def _method_rows(source, model, method, seed):
    rows = []
    for row in source:
        rows.append({
            "model": model,
            "method": method,
            "seed": int(seed),
            "sample_index": int(row["sample_index"]),
            "RMSE": float(row["RMSE"]),
            "MAE": float(row["MAE"]),
            "ABS_REL": float(row["ABS_REL"]),
            "nonfinite_pixels": int(float(row["nonfinite_pixels"])),
        })
    return rows


def aggregate_outputs(root, baseline_root, seeds):
    root = Path(root).resolve()
    baseline_root = Path(baseline_root).resolve()
    sample_rows = []
    layer_rows = []
    propagation_rows = []
    shared_indices = None
    for model in MODEL_ORDER:
        metadata = read_json(
            baseline_root / "stress" / "rtn" / model / "metadata.json")
        indices = tuple(int(index) for index in metadata["evaluation_indices"])
        if shared_indices is None:
            shared_indices = indices
        elif indices != shared_indices:
            raise ValueError("baseline evaluation indices differ by model")
        sample_rows.extend(_method_rows(
            _baseline_rows(baseline_root, model, "rtn", "FP32"),
            model, "fp32", 0))
        sample_rows.extend(_method_rows(
            _baseline_rows(
                baseline_root, model, "rtn", "HW_W4A4_full"),
            model, "rtn", 0))
        sample_rows.extend(_method_rows(
            _baseline_rows(
                baseline_root, model, "brecq", "HW_W4A4_full"),
            model, "brecq", 0))
        for seed in seeds:
            output = root / "evaluation" / "qdrop" / model / \
                ("seed_%d" % seed)
            qdrop_metadata = read_json(output / "metadata.json")
            if tuple(int(index) for index in
                     qdrop_metadata["evaluation_indices"]) != indices:
                raise ValueError("QDrop evaluation indices are not aligned")
            source = [
                row for row in read_csv(output / "sample_metrics.csv")
                if row["config"] == QDROP_EVALUATION_CONFIG]
            sample_rows.extend(_method_rows(
                source, model, "qdrop", seed))
            current_layer_rows = [
                row for row in read_csv(
                    output / "layer_quantization_metrics.csv")
                if row["config"] == QDROP_EVALUATION_CONFIG]
            validate_qdrop_layer_rows(current_layer_rows)
            for row in current_layer_rows:
                current = dict(row)
                current.update({
                    "method": "qdrop", "seed": seed})
                layer_rows.append(current)
            propagation_path = output / "propagation_quantization_metrics.csv"
            if propagation_path.is_file():
                for row in read_csv(propagation_path):
                    if row["config"] == QDROP_EVALUATION_CONFIG:
                        current = dict(row)
                        current.update({
                            "method": "qdrop", "seed": seed})
                        propagation_rows.append(current)
        current = [row for row in sample_rows if row["model"] == model]
        validate_aligned_sample_rows(current, model, indices, seeds)
    seed_rows = aggregate_seed_rows(sample_rows)
    model_rows, acceptance = aggregate_model_metrics(seed_rows, seeds)
    write_csv(root / "qdrop_w4a4_sample_metrics.csv", sample_rows)
    write_csv(root / "qdrop_w4a4_seed_summary.csv", seed_rows)
    write_csv(root / "qdrop_w4a4_model_summary.csv", model_rows)
    write_csv(root / "qdrop_w4a4_layer_metrics.csv", layer_rows)
    write_csv(
        root / "qdrop_w4a4_propagation_metrics.csv", propagation_rows)
    write_csv(root / "qdrop_w4a4_acceptance.csv", acceptance)
    return {
        "evaluation_indices": list(shared_indices),
        "sample_rows": len(sample_rows),
        "seed_rows": len(seed_rows),
        "model_rows": len(model_rows),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--phase",
        choices=("probability-search", "formal", "evaluate"),
        required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--dcn-extension", required=True)
    parser.add_argument("--model-run", action="append", required=True)
    parser.add_argument("--brecq-manifest", action="append", required=True)
    parser.add_argument("--devices", nargs=4, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_qdrop_config(args.config)
    model_runs = _mapping(args.model_run, "model run")
    brecq_manifests = _mapping(
        args.brecq_manifest, "BRECQ manifest")
    baseline_root = Path(args.baseline_root).resolve()
    dcn = _validate_inputs(
        model_runs, brecq_manifests, baseline_root,
        args.dcn_extension)
    matrix = build_run_matrix(
        args.phase, config.formal.seeds, args.devices)
    commands = []
    for row in matrix:
        if args.phase == "evaluate":
            command = _evaluation_command(
                row, args, model_runs, baseline_root)
        else:
            command = _reconstruction_command(
                row, args, model_runs)
        commands.append(dict(row, command=command))
    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / ("%s_commands.json" % args.phase), commands)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "%s:%s" % (dcn.parent, REPO_ROOT)
    environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    for wave in build_execution_waves(commands):
        run_execution_wave(wave, environment)
    if args.phase == "evaluate":
        result = aggregate_outputs(
            root, baseline_root, config.formal.seeds)
        write_json(root / "qdrop_w4a4_evaluation.json", {
            "baseline_root": str(baseline_root),
            "model_runs": dict(
                (model, str(model_runs[model])) for model in MODEL_ORDER),
            "brecq_manifests": dict(
                (model, str(brecq_manifests[model]))
                for model in MODEL_ORDER),
            "seeds": list(config.formal.seeds),
            "qdrop_evaluation_backend": QDROP_EVALUATION_BACKEND,
            "qdrop_evaluation_config": QDROP_EVALUATION_CONFIG,
            "evaluation_indices": result["evaluation_indices"],
            "sample_rows": result["sample_rows"],
            "seed_rows": result["seed_rows"],
            "model_rows": result["model_rows"],
        })


if __name__ == "__main__":
    main()
