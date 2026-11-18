#!/usr/bin/env python3
"""Materialize deterministic NYU full-validation inputs for RKNN."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Optional, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_nyu_four_model_int_mixed_precision as quant  # noqa: E402
from scripts import run_nyu_rtn_quantization as rtn  # noqa: E402
from scripts.nyu_model_runtime import NYUModelRuntime  # noqa: E402


MODELS = ("cspn", "dyspn", "nlspn", "completionformer")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arrays(runtime: NYUModelRuntime, dataset, index: int):
    sample = rtn.seeded_sample(dataset, index, int(runtime.saved_args.seed))
    batch = rtn.batch_from_sample(sample)
    model_args, ground_truth = runtime.model_input(batch, torch.device("cpu"))
    if runtime.model_name == "cspn":
        inputs = (model_args[0],)
    else:
        if runtime.model_name in ("nlspn", "completionformer"):
            payload = model_args[0]
            inputs = (payload["rgb"], payload["dep"])
        else:
            inputs = tuple(model_args)
    arrays = tuple(np.ascontiguousarray(value.numpy(), dtype=np.float32)
                   for value in inputs)
    truth = np.ascontiguousarray(ground_truth.numpy(), dtype=np.float32)
    expected_inputs = ((1, 4, 228, 304),) if runtime.model_name == "cspn" \
        else ((1, 3, 228, 304), (1, 1, 228, 304))
    if tuple(value.shape for value in arrays) != expected_inputs or \
            tuple(truth.shape) != (1, 1, 228, 304):
        raise ValueError("unexpected deployment tensor shape")
    if not all(np.isfinite(value).all() for value in arrays + (truth,)):
        raise ValueError("full-validation tensors must be finite")
    return arrays, truth


def materialize(config_path: Path, model: str, output: Path,
                board_root: str) -> Path:
    if model not in MODELS:
        raise ValueError("unsupported model: %s" % model)
    loaded = json.loads(config_path.read_text(encoding="utf-8"))
    if "models" in loaded:
        source = loaded
    else:
        config = quant._load_run_config(config_path)
        source = quant._model_source(config_path, config)
    payload = source["models"][model]
    runtime = NYUModelRuntime.from_args(
        quant._runtime_args(payload, payload["device"]))
    try:
        dataset = runtime.build_dataset("val")
        if len(dataset) != 654:
            raise ValueError("NYU validation split must contain 654 samples")
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        rows = []
        records = []
        for index in range(len(dataset)):
            inputs, truth = _arrays(runtime, dataset, index)
            stem = "%04d" % index
            paths = []
            for position, value in enumerate(inputs):
                path = output / (stem + ".input%d.f32" % position)
                value.tofile(str(path))
                paths.append(path)
            truth_path = output / (stem + ".gt.f32")
            truth.tofile(str(truth_path))
            board_paths = [
                "%s/%s.input%d.f32" %
                (board_root.rstrip("/"), stem, position)
                for position in range(len(inputs))]
            board_truth = "%s/%s.gt.f32" % (board_root.rstrip("/"), stem)
            rows.append("\t".join(
                [str(index)] + board_paths + [board_truth]))
            records.append({
                "index": index,
                "inputs": [
                    {"path": path.name, "bytes": path.stat().st_size,
                     "sha256": sha256(path)} for path in paths],
                "ground_truth": {
                    "path": truth_path.name,
                    "bytes": truth_path.stat().st_size,
                    "sha256": sha256(truth_path),
                },
            })
        manifest = output / "manifest.tsv"
        manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
        metadata = {
            "schema_version": 1,
            "model": model,
            "dataset": "NYU Depth V2 official validation",
            "sample_count": len(dataset),
            "indices": list(range(len(dataset))),
            "seed": int(runtime.saved_args.seed),
            "board_root": board_root.rstrip("/"),
            "manifest": manifest.name,
            "manifest_sha256": sha256(manifest),
            "records": records,
        }
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        return manifest
    finally:
        runtime.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--board-root", required=True)
    args = parser.parse_args(argv)
    print(materialize(args.config, args.model, args.output,
                      args.board_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
