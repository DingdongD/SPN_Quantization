#!/usr/bin/env python3
"""Run one external SPN model on the canonical five-frame payload."""

from __future__ import print_function

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import spn_sequence_io as sequence_io


DYSPN_ROOT = Path("/workspace/external_depth_completion_models/DySPN")
NLSPN_ROOT = Path("/workspace/external_depth_completion_models/NLSPN_ECCV20")
COMPLETIONFORMER_ROOT = Path("/workspace/CompletionFormer")


def nlspn_namespace(args):
    return argparse.Namespace(
        network=args["nlspn_network"],
        from_scratch=args["from_scratch"],
        prop_time=args["iteration"],
        prop_kernel=3,
        conf_prop=True,
        affinity="TGASS",
        affinity_gamma=0.5,
        preserve_input=False,
        legacy=False,
        lr=args["lr"],
    )


def completionformer_namespace(args):
    return argparse.Namespace(
        model=args["completionformer_model"],
        from_scratch=args["from_scratch"],
        prop_time=args["iteration"],
        prop_kernel=3,
        conf_prop=True,
        affinity="TGASS",
        affinity_gamma=0.5,
        preserve_input=False,
        legacy=False,
        max_depth=10.0,
    )


def predict_frames(model_name, model, rgb, sparse, device):
    rgb = np.asarray(rgb, dtype=np.float32)
    sparse = np.asarray(sparse, dtype=np.float32)
    predictions = []
    with torch.no_grad():
        for index in range(rgb.shape[0]):
            rgb_tensor = torch.from_numpy(rgb[index:index + 1]).to(device)
            dep_tensor = torch.from_numpy(
                sparse[index:index + 1, None]).to(device)
            if model_name == "dyspn":
                output = model(rgb_tensor, dep_tensor)
            else:
                output = model({"rgb": rgb_tensor, "dep": dep_tensor})
            if isinstance(output, dict):
                output = output["pred"]
            predictions.append(output.detach().cpu().numpy()[0, 0])
    result = np.stack(predictions).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("model prediction contains non-finite values")
    return result


def _add_path(path):
    path = str(Path(path))
    if path not in sys.path:
        sys.path.insert(0, path)


def _construct_in_directory(root, constructor):
    previous = os.getcwd()
    try:
        os.chdir(str(root))
        return constructor()
    finally:
        os.chdir(previous)


def build_model(model_name, args, device):
    if model_name == "dyspn":
        _add_path(DYSPN_ROOT)
        from DySPN.base import Model

        model = Model(
            iteration=args["iteration"],
            num_neighbor=args["dyspn_neighbors"],
            mode="dyspn",
            res=args["dyspn_resnet"],
            bm=args["dyspn_basemodel"],
            stodepth=not args["from_scratch"],
            norm_depth=[0.0, 10.0],
            norm_layer="bn",
        )
        architecture = "DySPN %s %s neighbor=%d" % (
            args["dyspn_resnet"],
            args["dyspn_basemodel"],
            args["dyspn_neighbors"],
        )
    elif model_name == "nlspn":
        source = NLSPN_ROOT / "src"
        _add_path(source)
        _add_path(source / "model" / "deformconv")
        from model.nlspnmodel import NLSPNModel

        model = _construct_in_directory(
            source, lambda: NLSPNModel(nlspn_namespace(args)))
        architecture = "NLSPN %s" % args["nlspn_network"]
    elif model_name == "completionformer":
        source = COMPLETIONFORMER_ROOT / "src"
        _add_path(source)
        _add_path(source / "model" / "deformconv")
        from model.completionformer import CompletionFormer

        model = _construct_in_directory(
            source,
            lambda: CompletionFormer(completionformer_namespace(args)),
        )
        architecture = args["completionformer_model"]
    else:
        raise ValueError("unsupported model: %s" % model_name)
    return model.to(device), {
        "architecture": architecture,
        "iteration": int(args["iteration"]),
        "from_scratch": bool(args["from_scratch"]),
    }


def load_checkpoint_strict(model, path):
    checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("net"), dict):
        raise RuntimeError("checkpoint must contain a dictionary-valued net")
    model.load_state_dict(checkpoint["net"], strict=True)
    return checkpoint


def make_parser():
    parser = argparse.ArgumentParser(
        description="Run one external SPN model on five canonical frames")
    parser.add_argument(
        "--model", required=True,
        choices=("dyspn", "nlspn", "completionformer"))
    parser.add_argument("--canonical-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--args-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _read_args(path, expected_model):
    with Path(path).open("r", encoding="utf-8") as stream:
        args = json.load(stream)
    if not isinstance(args, dict):
        raise ValueError("args JSON must contain an object")
    if args.get("model") != expected_model:
        raise ValueError(
            "args JSON model %r does not match %r" %
            (args.get("model"), expected_model))
    return args


def main(argv=None):
    cli = make_parser().parse_args(argv)
    checkpoint_path = Path(cli.checkpoint)
    args_path = Path(cli.args_json)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(str(checkpoint_path))
    if not args_path.is_file():
        raise FileNotFoundError(str(args_path))
    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for requested device %s" % device)

    torch.set_num_threads(1)
    canonical = sequence_io.load_canonical_frames(cli.canonical_dir)
    args = _read_args(args_path, cli.model)
    model, model_metadata = build_model(cli.model, args, device)
    load_checkpoint_strict(model, checkpoint_path)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    prediction = predict_frames(
        cli.model, model, canonical["rgb"], canonical["sparse"], device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_seconds = time.perf_counter() - start

    metadata = dict(model_metadata)
    metadata.update({
        "args_json": str(args_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "rgb_shape": list(canonical["rgb"].shape),
        "sparse_shape": list(canonical["sparse"].shape),
    })
    checkpoint_digest = sequence_io.file_sha256(checkpoint_path)
    sequence_io.write_worker_result(
        cli.output,
        cli.model,
        canonical["frame_ids"],
        prediction,
        canonical["input_digest"],
        checkpoint_digest,
        metadata,
        runtime_seconds,
    )
    completion = {
        "model": cli.model,
        "output": str(Path(cli.output).resolve()),
        "runtime_seconds": runtime_seconds,
        "input_digest": canonical["input_digest"],
        "checkpoint_digest": checkpoint_digest,
    }
    print(json.dumps(completion, sort_keys=True))
    return completion


if __name__ == "__main__":
    main()
