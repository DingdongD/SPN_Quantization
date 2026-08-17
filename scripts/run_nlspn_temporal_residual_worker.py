#!/usr/bin/env python3
"""Run frozen NLSPN stages in the custom deform-convolution environment."""

from __future__ import print_function

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from scripts import nlspn_temporal_residual as residual
from scripts import run_spn_sequence_worker as spn_worker


BASELINE_FIELDS = (
    "pred", "pred_init", "guidance", "confidence", "offset", "aff")


def predict_baseline(model, rgb, sparse, device):
    rgb = np.asarray(rgb, dtype=np.float32)
    sparse = np.asarray(sparse, dtype=np.float32)
    if rgb.ndim != 4 or sparse.shape != (
            rgb.shape[0], rgb.shape[2], rgb.shape[3]):
        raise ValueError("baseline RGB and sparse shapes are incompatible")
    collected = dict((key, []) for key in BASELINE_FIELDS)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for index in range(rgb.shape[0]):
            sample = {
                "rgb": torch.from_numpy(rgb[index:index + 1]).to(device),
                "dep": torch.from_numpy(
                    sparse[index:index + 1, None]).to(device),
            }
            output = model(sample)
            for key in BASELINE_FIELDS:
                value = output[key].detach().cpu().numpy()[0]
                if key in ("pred", "pred_init"):
                    value = value[0]
                collected[key].append(value)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    result = dict(
        (key, np.stack(values).astype(np.float32))
        for key, values in collected.items())
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("baseline output contains non-finite values")
    return result, seconds


def reconstruct_pair(model, previous_depth, current_sparse, backward_flow,
                     guidance, confidence, current_rgb):
    base, _ = residual.backward_warp(previous_depth, backward_flow)
    sparse_mask = current_sparse > 0.0
    seed = torch.zeros_like(base)
    seed[sparse_mask] = current_sparse[sparse_mask] - base[sparse_mask]
    dense_residual = model.prop_layer(
        seed, guidance, confidence, None, current_rgb)[0]
    reconstructed = torch.clamp(
        base + dense_residual, min=0.0, max=residual.MAX_DEPTH)
    return reconstructed, dense_residual


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def predict_propagation(model, payload, baseline, flow, device):
    frame_ids = np.asarray(payload["frame_ids"], dtype=np.int32)
    if not np.array_equal(frame_ids, baseline["frame_ids"]):
        raise ValueError("baseline frame IDs do not match clip")
    if not np.array_equal(frame_ids, flow["frame_ids"]):
        raise ValueError("flow frame IDs do not match clip")
    backward_flow = np.asarray(flow["backward_flow"], dtype=np.float32)
    expected_flow = (
        frame_ids.size - 1, 2, payload["rgb"].shape[2],
        payload["rgb"].shape[3])
    if backward_flow.shape != expected_flow:
        raise ValueError("backward flow shape does not match clip")
    required = ("pred", "guidance", "confidence")
    if not all(key in baseline for key in required):
        raise ValueError("baseline is missing propagation fields")

    collected = dict((key, []) for key in (
        "base", "in_bounds", "oracle_residual", "oracle_prediction",
        "causal_residual", "causal_prediction"))
    oracle_seconds = 0.0
    causal_seconds = 0.0
    with torch.no_grad():
        for index in range(frame_ids.size - 1):
            previous_depth = torch.from_numpy(
                baseline["pred"][index:index + 1, None]).to(device)
            current_sparse = torch.from_numpy(
                payload["sparse"][index + 1:index + 2, None]).to(device)
            pair_flow = torch.from_numpy(
                backward_flow[index:index + 1]).to(device)
            current_rgb = torch.from_numpy(
                payload["rgb"][index + 1:index + 2]).to(device)
            base, in_bounds = residual.backward_warp(
                previous_depth, pair_flow)

            oracle_guidance = torch.from_numpy(
                baseline["guidance"][index + 1:index + 2]).to(device)
            oracle_confidence = torch.from_numpy(
                baseline["confidence"][index + 1:index + 2]).to(device)
            _synchronize(device)
            start = time.perf_counter()
            oracle_prediction, oracle_residual = reconstruct_pair(
                model, previous_depth, current_sparse, pair_flow,
                oracle_guidance, oracle_confidence, current_rgb)
            _synchronize(device)
            oracle_seconds += time.perf_counter() - start

            previous_guidance = torch.from_numpy(
                baseline["guidance"][index:index + 1]).to(device)
            previous_confidence = torch.from_numpy(
                baseline["confidence"][index:index + 1]).to(device)
            causal_guidance = residual.backward_warp(
                previous_guidance, pair_flow)[0]
            causal_confidence = residual.backward_warp(
                previous_confidence, pair_flow)[0]
            _synchronize(device)
            start = time.perf_counter()
            causal_prediction, causal_residual = reconstruct_pair(
                model, previous_depth, current_sparse, pair_flow,
                causal_guidance, causal_confidence, current_rgb)
            _synchronize(device)
            causal_seconds += time.perf_counter() - start

            values = {
                "base": base[0, 0],
                "in_bounds": in_bounds[0, 0],
                "oracle_residual": oracle_residual[0, 0],
                "oracle_prediction": oracle_prediction[0, 0],
                "causal_residual": causal_residual[0, 0],
                "causal_prediction": causal_prediction[0, 0],
            }
            for key, value in values.items():
                collected[key].append(value.detach().cpu().numpy())

    result = {}
    for key, values in collected.items():
        dtype = bool if key == "in_bounds" else np.float32
        result[key] = np.stack(values).astype(dtype)
    if not all(np.isfinite(value).all() for key, value in result.items()
               if key != "in_bounds"):
        raise ValueError("propagation output contains non-finite values")
    return result, oracle_seconds, causal_seconds


def _write_npz_atomic(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".npz", dir=str(path.parent))
    os.close(handle)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_clip(path):
    with np.load(str(path), allow_pickle=False) as data:
        payload = dict((key, data[key]) for key in data.files)
    residual.validate_clip_payload(payload)
    return payload


def _load_npz(path):
    with np.load(str(path), allow_pickle=False) as data:
        return dict((key, data[key]) for key in data.files)


def _read_args(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        args = json.load(stream)
    if not isinstance(args, dict) or args.get("model") != "nlspn":
        raise ValueError("args JSON must describe the nlspn model")
    return args


def run_baseline(cli):
    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for %s" % device)
    payload = _load_clip(cli.input)
    args = _read_args(cli.args_json)
    model, model_metadata = spn_worker.build_model("nlspn", args, device)
    spn_worker.load_checkpoint_strict(model, Path(cli.checkpoint))
    model.eval()
    prediction, seconds = predict_baseline(
        model, payload["rgb"], payload["sparse"], device)
    output = dict(prediction)
    output.update({
        "frame_ids": payload["frame_ids"].astype(np.int32),
        "stage": np.asarray("baseline"),
        "runtime_seconds": np.asarray(seconds, dtype=np.float64),
        "checkpoint_digest": np.asarray(
            residual.file_sha256(cli.checkpoint)),
        "architecture": np.asarray(model_metadata["architecture"]),
        "iteration": np.asarray(model_metadata["iteration"], dtype=np.int32),
    })
    _write_npz_atomic(cli.output, output)
    return {
        "stage": "baseline",
        "output": str(Path(cli.output).resolve()),
        "runtime_seconds": seconds,
        "frame_count": int(payload["frame_ids"].size),
    }


def run_propagate(cli):
    if not cli.baseline or not cli.flow:
        raise ValueError("propagate stage requires --baseline and --flow")
    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for %s" % device)
    payload = _load_clip(cli.input)
    baseline = _load_npz(cli.baseline)
    flow = _load_npz(cli.flow)
    args = _read_args(cli.args_json)
    expected = {
        "iteration": 18,
        "nlspn_network": "resnet34",
    }
    for key, value in expected.items():
        if args.get(key) != value:
            raise ValueError("unexpected NLSPN %s configuration" % key)
    model, model_metadata = spn_worker.build_model("nlspn", args, device)
    spn_worker.load_checkpoint_strict(model, Path(cli.checkpoint))
    model.eval()
    if bool(model.args.preserve_input):
        raise ValueError("NLSPN preserve_input must remain false")
    if model.args.affinity != "TGASS" or model.prop_layer.prop_time != 18:
        raise ValueError("NLSPN propagation configuration changed")
    prediction, oracle_seconds, causal_seconds = predict_propagation(
        model, payload, baseline, flow, device)
    output = dict(prediction)
    output.update({
        "frame_ids": payload["frame_ids"].astype(np.int32),
        "stage": np.asarray("propagate"),
        "oracle_runtime_seconds": np.asarray(
            oracle_seconds, dtype=np.float64),
        "causal_runtime_seconds": np.asarray(
            causal_seconds, dtype=np.float64),
        "checkpoint_digest": np.asarray(
            residual.file_sha256(cli.checkpoint)),
        "architecture": np.asarray(model_metadata["architecture"]),
        "iteration": np.asarray(model_metadata["iteration"], dtype=np.int32),
    })
    _write_npz_atomic(cli.output, output)
    return {
        "stage": "propagate",
        "output": str(Path(cli.output).resolve()),
        "oracle_runtime_seconds": oracle_seconds,
        "causal_runtime_seconds": causal_seconds,
        "pair_count": int(payload["frame_ids"].size - 1),
    }


def make_parser():
    parser = argparse.ArgumentParser(
        description="Run frozen NLSPN temporal-residual stages")
    parser.add_argument(
        "--stage", required=True, choices=("baseline", "propagate"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--flow")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--args-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv=None):
    cli = make_parser().parse_args(argv)
    torch.set_num_threads(1)
    if cli.stage == "baseline":
        result = run_baseline(cli)
    else:
        result = run_propagate(cli)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
