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
        raise RuntimeError("propagate stage is not implemented")
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
