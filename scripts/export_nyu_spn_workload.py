#!/usr/bin/env python3
"""Export one official NYU SPN inference as an environment-neutral workload."""

from __future__ import annotations

import argparse
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Dict, Iterable, Mapping, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import torch  # noqa: E402

from spn_quant.hardware.collector import WorkloadCollector  # noqa: E402
from spn_quant.hardware.workload import WorkloadProfile  # noqa: E402


MODEL_NAMES = ("cspn", "dyspn", "nlspn", "completionformer")


def runtime_args(spec: Mapping[str, Any], device: str) -> Namespace:
    return Namespace(
        model=str(spec["model"]),
        run_dir=Path(spec["run_dir"]),
        checkpoint=Path(spec["checkpoint"]),
        expected_architecture_class=str(spec["expected_architecture_class"]),
        required_cuda_extension=str(spec["required_cuda_extension"]),
        propagation_iterations=int(spec["propagation_iterations"]),
        data_root=Path(spec["data_root"]), device=str(device),
        checkpoint_architecture=str(spec["checkpoint_architecture"]),
        native_cuda_operator=spec.get("native_cuda_operator"),
    )


def select_sample_index(spec: Mapping[str, Any]) -> int:
    indices = tuple(int(value) for value in spec["evaluation_indices"])
    if not indices:
        raise ValueError("model spec has no fixed evaluation identities")
    return indices[0]


def neighborhood_size(model_name: str, model: Any) -> int:
    if model_name == "cspn":
        return 8
    if model_name == "dyspn":
        base = getattr(model, "base", None)
        value = getattr(base, "num_sample", None)
        if value is not None and int(value) > 0:
            return int(value)
        for module in model.modules():
            if "DySPN" in type(module).__name__ and hasattr(module, "num"):
                value = int(module.num)
                if value > 0:
                    return value
        raise RuntimeError("unable to determine DySPN neighborhood size")
    if model_name in ("nlspn", "completionformer"):
        value = getattr(model, "num_neighbors", None)
        if value is None:
            raise RuntimeError(
                "unable to determine %s neighborhood size" % model_name)
        if int(value) <= 0:
            raise RuntimeError("invalid propagation neighborhood size")
        return int(value)
    raise ValueError("unsupported SPN model: %s" % model_name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)


def _input_shapes(model_args: Any) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
    tensors = tuple(_iter_tensors(model_args))
    if not tensors:
        raise RuntimeError("official runtime produced no tensor model inputs")
    return tuple(
        ("arg%d" % index, tuple(int(value) for value in tensor.shape))
        for index, tensor in enumerate(tensors))


def _load_payload(config_path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if set(MODEL_NAMES) - set(payload.get("models", {})):
        raise ValueError("source config does not define all four SPN models")
    return payload


def export(config_path: Path, model_name: str, device: str,
           output_path: Path) -> Path:
    if model_name not in MODEL_NAMES:
        raise ValueError("unsupported SPN model: %s" % model_name)
    config_path = Path(config_path).resolve()
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("workload output already exists: %s" % output_path)
    payload = _load_payload(config_path)
    spec = payload["models"][model_name]

    # Delayed imports keep configuration/unit tests independent of official
    # CUDA extensions and model-specific Python paths.
    from scripts.nyu_model_runtime import NYUModelRuntime
    from scripts import run_nyu_rtn_quantization as rtn_runner

    runtime = NYUModelRuntime.from_args(runtime_args(spec, device))
    try:
        model = runtime.build_model(runtime.device)
        dataset = runtime.build_dataset("val")
        sample_index = select_sample_index(spec)
        sample = rtn_runner.seeded_sample(
            dataset, sample_index, runtime.saved_args.seed)
        batch = rtn_runner.batch_from_sample(sample)
        model_args, ground_truth = runtime.model_input(batch, runtime.device)
        del ground_truth
        neighbors = neighborhood_size(model_name, model)
        with WorkloadCollector(
                model, model_name=model_name,
                propagation_iterations=int(spec["propagation_iterations"]),
                neighborhood_size=neighbors) as collector:
            with torch.no_grad():
                result = model(*model_args)
        prediction = runtime.prediction(result)
        expected_shape = (1, 1, 228, 304)
        if tuple(prediction.shape) != expected_shape:
            raise RuntimeError(
                "unexpected prediction shape: %s != %s" %
                (tuple(prediction.shape), expected_shape))
        if not bool(torch.isfinite(prediction).all().item()):
            raise RuntimeError("official prediction contains non-finite values")
        layers = collector.layers()
        if not layers:
            raise RuntimeError("workload collector produced no operations")
        checkpoint_path = Path(spec["checkpoint"]).resolve()
        metadata = (
            ("architecture_class", type(model).__name__),
            ("checkpoint_architecture", spec["checkpoint_architecture"]),
            ("checkpoint_sha256", _sha256(checkpoint_path)),
            ("config_sha256", _sha256(config_path)),
            ("cuda_device", str(runtime.device)),
            ("cuda_device_name", torch.cuda.get_device_name(runtime.device)),
            ("neighborhood_size", neighbors),
            ("propagation_iterations", int(spec["propagation_iterations"])),
            ("python_executable", sys.executable),
            ("python_version", platform.python_version()),
            ("torch_version", str(torch.__version__)),
        )
        profile = WorkloadProfile(
            model_name=model_name, source_config=str(config_path),
            checkpoint=str(checkpoint_path), sample_index=sample_index,
            input_shapes=_input_shapes(model_args), layers=layers,
            metadata=metadata)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        profile.write_json(output_path)
        return output_path
    finally:
        runtime.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "configs" /
        "four_model_unified_fp16_task_aware.json")
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Any = None) -> int:
    args = _parser().parse_args(argv)
    path = export(args.config, args.model, args.device, args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
