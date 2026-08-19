#!/usr/bin/env python3
"""Run the unified CSPN BRECQ and QDrop precision evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from spn_quant.deployment_contract import file_sha256  # noqa: E402
from spn_quant.qdrop_config import load_qdrop_config  # noqa: E402


MODEL_ORDER = ("cspn",)
PRECISION_ORDER = ("W4A4", "W6A6")
QDROP_EVALUATION_BACKEND = "propagation"
QDROP_EVALUATION_CONFIGS = {
    "W4A4": "PA_W4A4_PROP_A8",
    "W6A6": "PA_W6A6_PROP_A8",
}
P3_T3_CONFIG = "CONTEXT_P3_T3_W8A8"


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


def _precision_bits(precision):
    if precision == "W4A4":
        return 4, 4
    if precision == "W6A6":
        return 6, 6
    raise ValueError("unsupported reconstruction precision: %s" % precision)


def build_run_matrix(phase, seeds, devices):
    seeds = tuple(int(seed) for seed in seeds)
    devices = tuple(str(device) for device in devices)
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("QDrop formal matrix requires three unique seeds")
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("execution devices must be nonempty and unique")
    if any(not device.startswith("cuda:") for device in devices):
        raise ValueError("execution devices must be explicit CUDA devices")
    rows = []
    if phase == "brecq":
        jobs = tuple(("brecq", precision, 0)
                     for precision in PRECISION_ORDER)
    elif phase == "formal":
        jobs = tuple(("qdrop", precision, seed)
                     for precision in PRECISION_ORDER for seed in seeds)
    else:
        raise ValueError("unknown reconstruction matrix phase: %s" % phase)
    for index, (method, precision, seed) in enumerate(jobs):
        weight_bits, activation_bits = _precision_bits(precision)
        rows.append({
            "phase": phase,
            "model": "cspn",
            "method": method,
            "precision": precision,
            "seed": seed,
            "device": devices[index % len(devices)],
            "weight_bits": weight_bits,
            "activation_bits": activation_bits,
        })
    return tuple(rows)


def build_execution_waves(rows):
    rows = tuple(rows)
    if not rows:
        return ()
    device_count = len(set(row["device"] for row in rows))
    waves = tuple(
        rows[start:start + device_count]
        for start in range(0, len(rows), device_count))
    for wave in waves:
        if len(set(row["device"] for row in wave)) != len(wave):
            raise ValueError("execution wave repeats a CUDA device")
    return waves


def validate_aligned_sample_rows(rows, model, indices, seeds, precisions):
    indices = tuple(int(index) for index in indices)
    seeds = tuple(int(seed) for seed in seeds)
    precisions = tuple(str(precision) for precision in precisions)
    expected = set()
    for index in indices:
        expected.add(("fp32", "FP32", 0, index))
        expected.add(("p3_t3", "P3T3", 0, index))
        for precision in precisions:
            expected.add(("rtn", precision, 0, index))
            expected.add(("brecq", precision, 0, index))
            for seed in seeds:
                expected.add(("qdrop", precision, seed, index))
    observed = []
    for row in rows:
        if row["model"] != model:
            raise ValueError("evaluation row model mismatch")
        key = (
            str(row["method"]), str(row["precision"]),
            int(row["seed"]), int(row["sample_index"]))
        observed.append(key)
        nonfinite_pixels = int(row["nonfinite_pixels"])
        metrics = np.asarray([
            float(row["RMSE"]), float(row["MAE"]),
            float(row["ABS_REL"]), float(row["IRMSE"]),
        ], dtype=np.float64)
        if nonfinite_pixels == 0 and not np.isfinite(metrics).all():
            raise ValueError("finite prediction has non-finite metrics")
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("evaluation methods do not share exact sample coverage")


def validate_qdrop_layer_rows(rows):
    exact = [
        row for row in rows
        if row["kind"] == "exact_activation_contract"]
    if not exact:
        raise ValueError("QDrop layer metrics contain no exact activation sites")
    for row in exact:
        if int(row["calls"]) <= 0 or int(row["numel"]) <= 0:
            raise ValueError("QDrop exact activation site was not executed")
        ratios = np.asarray([
            float(row["zero_code_rate"]),
            float(row["saturation_rate"]),
        ], dtype=np.float64)
        if not np.isfinite(ratios).all() or \
                bool(np.any(ratios < 0.0)) or bool(np.any(ratios > 1.0)):
            raise ValueError("QDrop exact activation ratios are invalid")
        if np.isnan(float(row["sqnr_db"])):
            raise ValueError("QDrop exact activation SQNR is invalid")


def aggregate_seed_rows(sample_rows):
    keys = sorted(set(
        (row["model"], row["method"], row["precision"], int(row["seed"]))
        for row in sample_rows))
    rows = []
    for model, method, precision, seed in keys:
        current = [
            row for row in sample_rows
            if row["model"] == model and row["method"] == method and
            row["precision"] == precision and int(row["seed"]) == seed]
        nonfinite_ratio = float(np.mean([
            int(row["nonfinite_pixels"]) > 0 for row in current]))
        rows.append({
            "model": model,
            "method": method,
            "precision": precision,
            "seed": seed,
            "samples": len(current),
            "mean_rmse": float(np.mean([
                float(row["RMSE"]) for row in current])),
            "mean_mae": float(np.mean([
                float(row["MAE"]) for row in current])),
            "mean_abs_rel": float(np.mean([
                float(row["ABS_REL"]) for row in current])),
            "mean_irmse": float(np.mean([
                float(row["IRMSE"]) for row in current])),
            "nonfinite_ratio": nonfinite_ratio,
        })
    return rows


def aggregate_model_metrics(seed_rows, seeds, precisions=PRECISION_ORDER):
    seeds = tuple(int(seed) for seed in seeds)
    summaries = []
    acceptance = []
    fp32_matches = [
        row for row in seed_rows
        if row["method"] == "fp32" and row["precision"] == "FP32"]
    if len(fp32_matches) != 1:
        raise ValueError("FP32 summary coverage is incomplete")
    fp32_rmse = float(fp32_matches[0]["mean_rmse"])
    for precision in precisions:
        qdrop = sorted(
            (row for row in seed_rows
             if row["method"] == "qdrop" and
             row["precision"] == precision),
            key=lambda row: int(row["seed"]))
        if tuple(int(row["seed"]) for row in qdrop) != seeds:
            raise ValueError("QDrop seed summary is incomplete")
        brecq = [
            row for row in seed_rows
            if row["method"] == "brecq" and row["precision"] == precision]
        rtn = [
            row for row in seed_rows
            if row["method"] == "rtn" and row["precision"] == precision]
        if len(brecq) != 1 or len(rtn) != 1:
            raise ValueError("precision baseline summary is incomplete")
        values = np.asarray([
            float(row["mean_rmse"]) for row in qdrop], dtype=np.float64)
        mean_rmse = float(values.mean())
        nonfinite = max(float(row["nonfinite_ratio"]) for row in qdrop)
        summaries.append({
            "model": "cspn",
            "method": "qdrop",
            "precision": precision,
            "seeds": len(qdrop),
            "mean_rmse": mean_rmse,
            "std_rmse": float(values.std(ddof=0)),
            "min_rmse": float(values.min()),
            "max_rmse": float(values.max()),
            "nonfinite_ratio": nonfinite,
        })
        brecq_rmse = float(brecq[0]["mean_rmse"])
        acceptance.append({
            "model": "cspn",
            "precision": precision,
            "finite": int(nonfinite == 0.0),
            "better_than_brecq": int(mean_rmse < brecq_rmse),
            "better_than_rtn": int(mean_rmse < float(rtn[0]["mean_rmse"])),
            "within_fp32_10pct": int(
                np.isfinite(mean_rmse) and
                (mean_rmse - fp32_rmse) / fp32_rmse <= 0.10),
            "fp32_rmse": fp32_rmse,
            "brecq_rmse": brecq_rmse,
            "qdrop_mean_rmse": mean_rmse,
            "delta_vs_brecq": mean_rmse - brecq_rmse,
            "relative_fp32_degradation":
                (mean_rmse - fp32_rmse) / fp32_rmse,
        })
    return summaries, acceptance


def _prepare_process(row, environment):
    current = dict(environment)
    current["CUDA_VISIBLE_DEVICES"] = row["device"].split(":", 1)[1]
    command = list(row["command"])
    if "--device" in command:
        command[command.index("--device") + 1] = "cuda:0"
    return command, current, row["working_directory"]


def run_execution_wave(wave, environment):
    running = []
    for row in wave:
        command, current, working_directory = _prepare_process(
            row, environment)
        process = subprocess.Popen(
            command, cwd=working_directory, env=current)
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
                output="%s %s seed %d failed" %
                (row["method"], row["precision"], row["seed"]))
        running = [entry for entry in running if entry not in completed]
        if running and not completed:
            time.sleep(0.2)


def _brecq_command(row, args):
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_nyu_strict_reconstruction.py"),
        "--run-dir", str(Path(args.run_dir).resolve()),
        "--checkpoint", str(Path(args.checkpoint).resolve()),
        "--data-root", str(Path(args.data_root).resolve()),
        "--calibration-indices", str(Path(args.calibration_indices).resolve()),
        "--calibration-metadata", str(Path(args.calibration_metadata).resolve()),
        "--evaluation-protocol", str(Path(args.evaluation_protocol).resolve()),
        "--method", "brecq_strict",
        "--qdrop-target-plan",
        "--w-bits", str(row["weight_bits"]),
        "--steps", "20000",
        "--batch-size", "32",
        "--eval-samples", "0",
        "--device", row["device"],
        "--seed", "20260812",
        "--out-dir", str(
            Path(args.out_dir).resolve() / "reconstruction" /
            row["precision"] / "brecq"),
    ]


def _qdrop_command(row, args):
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_nyu_qdrop_reconstruction.py"),
        "--config", str(Path(args.config).resolve()),
        "--run-dir", str(Path(args.run_dir).resolve()),
        "--checkpoint", str(Path(args.checkpoint).resolve()),
        "--data-root", str(Path(args.data_root).resolve()),
        "--model", "cspn",
        "--precision", row["precision"],
        "--phase", "formal",
        "--seed", str(row["seed"]),
        "--calibration-indices", str(Path(args.calibration_indices).resolve()),
        "--calibration-metadata", str(Path(args.calibration_metadata).resolve()),
        "--evaluation-protocol", str(Path(args.evaluation_protocol).resolve()),
        "--out-dir", str(
            Path(args.out_dir).resolve() / "reconstruction" /
            row["precision"] / "qdrop"),
    ]


def _sample_index_csv(args):
    root = Path(args.out_dir).resolve()
    path = root / "evaluation_indices.csv"
    evaluation = read_json(args.evaluation_protocol)
    rows = [{"sample_index": int(index)}
            for index in evaluation["evaluation_indices"]]
    if len(rows) != 64:
        raise ValueError("evaluation protocol requires exactly 64 samples")
    write_csv(path, rows)
    return path


def _brecq_manifest(root, precision):
    return root / "reconstruction" / precision / "brecq" / \
        "cspn" / "brecq_strict" / "strict_reconstruction_manifest.json"


def _qdrop_manifest(root, precision, seed):
    return root / "reconstruction" / precision / "qdrop" / \
        ("formal_seed_%d" % int(seed)) / "qdrop_strict_manifest.json"


def _evaluation_jobs(args, seeds, devices):
    root = Path(args.out_dir).resolve()
    jobs = [{
        "method": "rtn",
        "precision": "BOTH",
        "seed": 0,
        "device": devices[0],
        "manifest": "",
    }]
    for precision in PRECISION_ORDER:
        jobs.append({
            "method": "brecq",
            "precision": precision,
            "seed": 0,
            "device": devices[len(jobs) % len(devices)],
            "manifest": str(_brecq_manifest(root, precision)),
        })
    for precision in PRECISION_ORDER:
        for seed in seeds:
            jobs.append({
                "method": "qdrop",
                "precision": precision,
                "seed": int(seed),
                "device": devices[len(jobs) % len(devices)],
                "manifest": str(_qdrop_manifest(root, precision, seed)),
            })
    return tuple(jobs)


def _evaluation_command(row, args, index_csv):
    root = Path(args.out_dir).resolve()
    if row["method"] == "rtn":
        configs = ("FP32", QDROP_EVALUATION_CONFIGS["W4A4"],
                   QDROP_EVALUATION_CONFIGS["W6A6"])
        output = root / "evaluation" / "rtn"
    else:
        configs = (QDROP_EVALUATION_CONFIGS[row["precision"]],)
        output = root / "evaluation" / row["method"] / row["precision"]
        if row["method"] == "qdrop":
            output = output / ("seed_%d" % row["seed"])
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_nyu_edge_quantization.py"),
        "--run-dir", str(Path(args.run_dir).resolve()),
        "--checkpoint", str(Path(args.checkpoint).resolve()),
        "--sample-metrics", str(index_csv),
        "--data-root", str(Path(args.data_root).resolve()),
        "--calibration-indices", str(Path(args.calibration_indices).resolve()),
        "--calibration-samples", "128",
        "--max-eval-samples", "64",
        "--out-dir", str(output),
        "--device", row["device"],
        "--seed", "20260812",
        "--config-names",
    ]
    command.extend(configs)
    command.append("--export-prediction-configs")
    command.extend(configs)
    command.extend(("--quant-backend", QDROP_EVALUATION_BACKEND))
    if row["manifest"]:
        command.extend(("--reconstruction-manifest", row["manifest"]))
    return command


def _prediction_path(root, config, index):
    return Path(root) / "cspn" / "predictions" / config / \
        ("sample_%05d.npz" % int(index))


def _load_prediction(path):
    with np.load(str(path), allow_pickle=False) as payload:
        return dict((name, payload[name].copy()) for name in payload.files)


def _validate_reference_payload(reference, current):
    for field in ("sample_index", "gt", "fp32", "sparse", "rgb"):
        if field not in reference or field not in current:
            raise KeyError("reference payload is missing %s" % field)
    for field in ("sample_index", "gt", "sparse", "rgb"):
        if not np.array_equal(reference[field], current[field]):
            raise ValueError("reference payload %s differs" % field)
    if not np.allclose(
            reference["fp32"], current["fp32"],
            rtol=1.0e-6, atol=1.0e-7):
        raise ValueError("reference payload fp32 differs")


def _depth_metrics(gt, pred):
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    valid = gt > 1.0e-4
    invalid = ~np.isfinite(pred[valid]) | (pred[valid] <= 1.0e-4)
    nonfinite_pixels = int(np.count_nonzero(invalid))
    if nonfinite_pixels:
        return {
            "RMSE": float("inf"), "MAE": float("inf"),
            "ABS_REL": float("inf"), "IRMSE": float("inf"),
            "nonfinite_pixels": nonfinite_pixels,
        }
    error = pred[valid] - gt[valid]
    inverse_error = 1.0 / pred[valid] - 1.0 / gt[valid]
    return {
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "ABS_REL": float(np.mean(np.abs(error) / gt[valid])),
        "IRMSE": float(np.sqrt(np.mean(inverse_error ** 2))),
        "nonfinite_pixels": nonfinite_pixels,
    }


def _prediction_row(model, method, precision, seed, index, path,
                    reference_gt):
    payload = _load_prediction(path)
    gt = payload["gt"]
    if reference_gt is not None and not np.array_equal(gt, reference_gt):
        raise ValueError("prediction GT arrays are not aligned")
    metrics = _depth_metrics(gt, payload["pred"])
    return dict({
        "model": model,
        "method": method,
        "precision": precision,
        "seed": int(seed),
        "sample_index": int(index),
        "prediction": str(Path(path).resolve()),
    }, **metrics), payload


def _select_qdrop_seed(root, precision, seeds):
    rows = []
    for seed in seeds:
        path = root / "reconstruction" / precision / "qdrop" / \
            ("formal_seed_%d" % int(seed)) / "qdrop_validation_metrics.csv"
        values = [float(row["RMSE"]) for row in read_csv(path)]
        if len(values) != 16 or not np.isfinite(values).all():
            raise ValueError("QDrop validation seed metrics are incomplete")
        rows.append((int(seed), float(np.mean(values))))
    ordered = sorted(rows, key=lambda row: (row[1], row[0]))
    return ordered[1][0], rows


def aggregate_outputs(args, seeds):
    root = Path(args.out_dir).resolve()
    evaluation = read_json(args.evaluation_protocol)
    indices = tuple(int(index) for index in evaluation["evaluation_indices"])
    rtn_root = root / "evaluation" / "rtn"
    p3_root = Path(args.p3_t3_root).resolve()
    rows = []
    for index in indices:
        fp_path = _prediction_path(rtn_root, "FP32", index)
        fp_row, fp_payload = _prediction_row(
            "cspn", "fp32", "FP32", 0, index, fp_path, None)
        rows.append(fp_row)
        gt = fp_payload["gt"]
        for precision in PRECISION_ORDER:
            config = QDROP_EVALUATION_CONFIGS[precision]
            rtn_path = _prediction_path(rtn_root, config, index)
            row, _ = _prediction_row(
                "cspn", "rtn", precision, 0, index, rtn_path, gt)
            rows.append(row)
            brecq_root = root / "evaluation" / "brecq" / precision
            brecq_path = _prediction_path(brecq_root, config, index)
            row, _ = _prediction_row(
                "cspn", "brecq", precision, 0, index, brecq_path, gt)
            rows.append(row)
            for seed in seeds:
                qdrop_root = root / "evaluation" / "qdrop" / precision / \
                    ("seed_%d" % int(seed))
                qdrop_path = _prediction_path(qdrop_root, config, index)
                row, _ = _prediction_row(
                    "cspn", "qdrop", precision, seed,
                    index, qdrop_path, gt)
                rows.append(row)
        p3_path = p3_root / "predictions" / P3_T3_CONFIG / \
            ("sample_%05d.npz" % index)
        p3_payload = _load_prediction(p3_path)
        _validate_reference_payload({
            "sample_index": fp_payload["sample_index"],
            "gt": fp_payload["gt"],
            "fp32": fp_payload["pred"],
            "sparse": fp_payload["sparse"],
            "rgb": fp_payload["rgb"],
        }, p3_payload)
        row, _ = _prediction_row(
            "cspn", "p3_t3", "P3T3", 0, index, p3_path, gt)
        rows.append(row)
    validate_aligned_sample_rows(
        rows, "cspn", indices, seeds, PRECISION_ORDER)
    seed_rows = aggregate_seed_rows(rows)
    model_rows, acceptance = aggregate_model_metrics(seed_rows, seeds)
    selections = {}
    selection_rows = []
    for precision in PRECISION_ORDER:
        selected, current = _select_qdrop_seed(root, precision, seeds)
        selections[precision] = selected
        for seed, validation_rmse in current:
            selection_rows.append({
                "precision": precision,
                "seed": seed,
                "validation_rmse": validation_rmse,
                "selected": int(seed == selected),
            })
    write_csv(root / "sample_metrics.csv", rows)
    write_csv(root / "seed_summary.csv", seed_rows)
    write_csv(root / "qdrop_summary.csv", model_rows)
    write_csv(root / "acceptance.csv", acceptance)
    write_csv(root / "qdrop_seed_selection.csv", selection_rows)
    write_json(root / "selected_qdrop_seeds.json", selections)
    return {
        "evaluation_indices": list(indices),
        "sample_rows": len(rows),
        "seed_rows": len(seed_rows),
        "qdrop_rows": len(model_rows),
    }


def artifact_hashes(root):
    root = Path(root)
    rows = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            rows[str(path.relative_to(root))] = file_sha256(path)
    return rows


def _audit(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if artifact_hashes(root) != manifest["artifacts"]:
        raise RuntimeError("unified evaluation artifact hashes changed")
    sample_rows = read_csv(root / "sample_metrics.csv")
    validate_aligned_sample_rows(
        sample_rows, "cspn", manifest["evaluation_indices"],
        manifest["seeds"], PRECISION_ORDER)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--phase", choices=("brecq", "formal", "evaluate", "audit"),
        required=True)
    parser.add_argument("--model", choices=MODEL_ORDER, required=True)
    parser.add_argument("--precisions", nargs="+", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--calibration-indices", required=True)
    parser.add_argument("--calibration-metadata", required=True)
    parser.add_argument("--evaluation-protocol", required=True)
    parser.add_argument("--p3-t3-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--devices", nargs="+", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_qdrop_config(args.config)
    if tuple(args.precisions) != PRECISION_ORDER:
        raise ValueError("precision order must be W4A4 W6A6")
    if args.model != "cspn":
        raise ValueError("unified reconstruction evaluation requires CSPN")
    if file_sha256(args.checkpoint) != read_json(
            args.calibration_metadata)["checkpoint_sha256"]:
        raise ValueError("unified evaluation checkpoint SHA256 differs")
    devices = tuple(str(device) for device in args.devices)
    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.phase == "audit":
        _audit(root)
        print("unified reconstruction audit passed", flush=True)
        return
    if args.phase in ("brecq", "formal"):
        matrix = build_run_matrix(
            args.phase, config.formal.seeds, devices)
        commands = []
        for row in matrix:
            command = _brecq_command(row, args) \
                if args.phase == "brecq" else _qdrop_command(row, args)
            commands.append(dict(
                row, command=command,
                working_directory=str(Path(args.data_root).resolve())))
    else:
        index_csv = _sample_index_csv(args)
        matrix = _evaluation_jobs(args, config.formal.seeds, devices)
        commands = [
            dict(
                row, command=_evaluation_command(row, args, index_csv),
                working_directory=str(Path(args.data_root).resolve()))
            for row in matrix]
    write_json(root / ("%s_commands.json" % args.phase), commands)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT)
    environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    for wave in build_execution_waves(commands):
        run_execution_wave(wave, environment)
    if args.phase == "evaluate":
        result = aggregate_outputs(args, config.formal.seeds)
        manifest = {
            "format_version": 1,
            "model": "cspn",
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "calibration_indices_sha256": file_sha256(
                args.calibration_indices),
            "calibration_metadata_sha256": file_sha256(
                args.calibration_metadata),
            "evaluation_protocol_sha256": file_sha256(
                args.evaluation_protocol),
            "p3_t3_root": str(Path(args.p3_t3_root).resolve()),
            "seeds": list(config.formal.seeds),
            "evaluation_indices": result["evaluation_indices"],
            "sample_rows": result["sample_rows"],
            "seed_rows": result["seed_rows"],
            "qdrop_rows": result["qdrop_rows"],
        }
        manifest["artifacts"] = artifact_hashes(root)
        write_json(root / "manifest.json", manifest)


if __name__ == "__main__":
    main()
