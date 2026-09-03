#!/usr/bin/env python3
"""Measure task-gradient and propagation sensitivity on official NYU models."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts import run_nyu_rtn_quantization as rtn_runner  # noqa: E402
from scripts import train_nyu_iteration_sweep as sweep  # noqa: E402
from scripts.nyu_model_runtime import NYUModelRuntime  # noqa: E402
from scripts.run_nyu_four_model_propagation_dtype_ablation import (  # noqa: E402
    _build_cspn_uniform_contract,
)
from scripts.run_nyu_model_p3t3_search import (  # noqa: E402
    _calibration_indices as load_calibration_indices,
)
from spn_quant.adapters import install_model_semantic_adapter  # noqa: E402
from spn_quant.experiment_config import MODEL_ORDER  # noqa: E402
from spn_quant.model_contracts import (  # noqa: E402
    build_model_quantization_contract,
)
from scripts.hardware_aligned_quantization import (  # noqa: E402
    SymmetricActivationQuantizer,
    UnsignedActivationQuantizer,
    symmetric_weight_qdq,
)


MODEL_NAMES = ("cspn",) + tuple(MODEL_ORDER)
BIT_OPTIONS = (4, 6, 8)
UNSIGNED_ROLES = frozenset((
    "initial_depth", "confidence", "prediction", "propagation_state",
))
NON_DIFFERENTIABLE_SIGNALS = frozenset(("input::sparse_mask",))
DERIVED_NON_GRAPH_SIGNALS = frozenset(("dyspn::signal::confidence",))


def _tensors(value) -> Iterable[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
        return
    if isinstance(value, Mapping):
        for key in sorted(value):
            yield from _tensors(value[key])
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)
        return
    raise TypeError("captured semantic value is not tensor-like")


def _finite_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise TypeError("%s must be a tensor" % name)
    if tensor.numel() == 0:
        raise ValueError("%s must be nonempty" % name)
    if not bool(torch.isfinite(tensor).all().item()):
        raise FloatingPointError("%s is non-finite" % name)
    return tensor.detach().float()


def _role_quantizer(role: str, bits: int, maximum: float):
    if role in UNSIGNED_ROLES:
        return UnsignedActivationQuantizer(bits, maximum)
    return SymmetricActivationQuantizer(bits, maximum)


def _quant_metrics(reference: torch.Tensor, quantized: torch.Tensor,
                   gradient: torch.Tensor) -> Dict[str, float]:
    reference = _finite_tensor("reference", reference)
    quantized = _finite_tensor("quantized", quantized)
    gradient = _finite_tensor("gradient", gradient)
    if reference.shape != quantized.shape or reference.shape != gradient.shape:
        raise ValueError("quantization metric tensor shapes differ")
    error = quantized - reference
    signal_power = float(reference.square().mean().item())
    noise_power = float(error.square().mean().item())
    if noise_power == 0.0:
        sqnr_db = float("inf")
    elif signal_power == 0.0:
        sqnr_db = float("-inf")
    else:
        sqnr_db = 10.0 * math.log10(signal_power / noise_power)
    weighted = float((gradient * (reference - quantized)).abs().sum().item())
    reference_l1 = float(reference.abs().sum().item())
    normalized = weighted / reference_l1 if reference_l1 > 0.0 else 0.0
    qmin = float(quantized.min().item())
    qmax = float(quantized.max().item())
    return {
        "numel": int(reference.numel()),
        "output_mse": float(error.square().mean().item()),
        "signal_power": signal_power,
        "noise_power": noise_power,
        "sqnr_db": sqnr_db,
        "gradient_weighted_error": weighted,
        "normalized_gradient_weighted_error": normalized,
        "zero_ratio": float((quantized == 0).float().mean().item()),
        "nonpositive_ratio": float((quantized <= 0).float().mean().item()),
        "sign_flip_ratio": float(((reference != 0) & (quantized != 0) &
                                   (reference.sign() != quantized.sign()))
                                  .float().mean().item()),
        "quantized_min": qmin,
        "quantized_max": qmax,
    }


class ModuleOutputCapture(object):
    """Capture differentiable outputs of contract-owned Conv/Linear modules."""

    def __init__(self, model: nn.Module, module_names: Sequence[str]) -> None:
        modules = dict(model.named_modules())
        self.module_names = tuple(module_names)
        if len(set(self.module_names)) != len(self.module_names):
            raise ValueError("module capture names contain duplicates")
        missing = tuple(name for name in self.module_names if name not in modules)
        if missing:
            raise KeyError("module capture names are missing: %s" % (missing,))
        self.values = dict((name, []) for name in self.module_names)
        self.handles = []
        for name in self.module_names:
            self.handles.append(modules[name].register_forward_hook(
                self._hook(name)))

    def _hook(self, name: str):
        def hook(module, inputs, output):
            del module, inputs
            if not torch.is_tensor(output):
                raise TypeError("module output is not a tensor: %s" % name)
            if output.requires_grad:
                output.retain_grad()
            self.values[name].append(output)
        return hook

    def begin(self) -> None:
        self.values = dict((name, []) for name in self.module_names)

    def end(self) -> Dict[str, Tuple[torch.Tensor, ...]]:
        missing = tuple(name for name in self.module_names
                        if not self.values[name])
        if missing:
            raise RuntimeError("module output capture is incomplete: %s" %
                               (missing,))
        return dict((name, tuple(values)) for name, values in self.values.items())

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _update_maximum(maxima: Dict[str, float], key: str,
                    values: Iterable[torch.Tensor]) -> None:
    for tensor in values:
        value = _finite_tensor(key, tensor)
        maximum = float(value.abs().max().item())
        if key in maxima:
            maxima[key] = max(maxima[key], maximum)
        else:
            maxima[key] = maximum


def _semantic_roles(semantic) -> Dict[str, str]:
    return dict((site.name, site.role) for site in semantic.registry)


def _semantic_signal_keys(values: Mapping[str, object]) -> Tuple[Tuple[str, int], ...]:
    keys = []
    for site, value in values.items():
        tensors = tuple(_tensors(value))
        if site == "signal::propagation_state":
            keys.extend((site, index) for index in range(len(tensors)))
        else:
            if len(tensors) != 1:
                raise ValueError("semantic signal must contain one tensor: %s" % site)
            keys.append((site, 0))
    return tuple(keys)


def _semantic_signal_tensors(values: Mapping[str, object]):
    for site in sorted(values):
        tensors = tuple(_tensors(values[site]))
        for index, tensor in enumerate(tensors):
            yield site, index, tensor


def _projection_parts(model_name: str, projection: torch.Tensor):
    if model_name == "cspn":
        return (("guidance", projection),)
    if model_name == "dyspn":
        channels = projection.shape[1] // 3
        offset, affinity = torch.split(projection, [2 * channels, channels], dim=1)
        return (("offset_logits", offset), ("affinity_logits", affinity))
    if model_name in ("nlspn", "completionformer"):
        offset_1, offset_2, affinity = torch.chunk(projection, 3, dim=1)
        return (("offset_logits", torch.cat((offset_1, offset_2), dim=1)),
                ("affinity_logits", affinity))
    raise ValueError("unknown projection model: %s" % model_name)


def _runtime_args(spec: Mapping[str, object], device: str):
    return type("RuntimeArgs", (), {
        "model": spec["model"],
        "run_dir": Path(spec["run_dir"]),
        "checkpoint": Path(spec["checkpoint"]),
        "expected_architecture_class": spec["expected_architecture_class"],
        "required_cuda_extension": spec["required_cuda_extension"],
        "propagation_iterations": int(spec["propagation_iterations"]),
        "data_root": Path(spec["data_root"]),
        "device": device,
        "checkpoint_architecture": spec["checkpoint_architecture"],
        "native_cuda_operator": spec["native_cuda_operator"],
    })()


def _build_contract(model_name: str, model):
    if model_name == "cspn":
        return _build_cspn_uniform_contract(model)
    return build_model_quantization_contract(model_name, model)


def _calibration_indices(spec: Mapping[str, object], trainset) -> Tuple[int, ...]:
    return load_calibration_indices(
        Path(spec["calibration_metadata"]), int(spec["calibration_count"]),
        tuple(int(index) for index in spec["evaluation_indices"]),
        len(trainset))


def _batch(runtime, dataset, index):
    return rtn_runner.batch_from_sample(
        rtn_runner.seeded_sample(dataset, index, runtime.saved_args.seed))


def _calibration_batches(runtime, dataset, indices, batch_size):
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("calibration batch size must be positive")
    batches = []
    for start in range(0, len(indices), batch_size):
        samples = tuple(_batch(runtime, dataset, index)
                        for index in indices[start:start + batch_size])
        keys = tuple(samples[0])
        if any(tuple(sample) != keys for sample in samples):
            raise ValueError("calibration sample fields differ")
        batches.append(dict(
            (key, torch.cat(tuple(sample[key] for sample in samples), dim=0)
             if torch.is_tensor(samples[0][key]) else samples[0][key])
            for key in keys))
    return tuple(batches)


def _forward(runtime, model, semantic, capture, batch, device, need_grad):
    model_args, target = runtime.model_input(batch, device)
    if need_grad:
        for tensor in _tensors(model_args):
            if not tensor.is_floating_point():
                raise TypeError("gradient input must be floating point")
            tensor.requires_grad_(True)
    capture.begin()
    semantic.begin_task_capture()
    if need_grad:
        output = model(*model_args)
    else:
        with torch.no_grad():
            output = model(*model_args)
    semantic_capture = semantic.task_capture()
    signal_values = semantic.task_signal_values()
    module_values = capture.end()
    prediction = runtime.prediction(output)
    if not torch.is_tensor(prediction):
        raise TypeError("runtime prediction is not a tensor")
    return prediction, target, module_values, signal_values, semantic_capture


def _weight_rows(model_name: str, model, module_names, accumulators, bits):
    for name in module_names:
        module = dict(model.named_modules())[name]
        weight = module.weight
        gradient = weight.grad
        if gradient is None:
            raise RuntimeError("weight gradient is missing: %s" % name)
        quantized, scale = symmetric_weight_qdq(weight.detach(), bits)
        metrics = _quant_metrics(weight, quantized, gradient)
        qmax = 2 ** (int(bits) - 1) - 1
        flat_weight = weight.detach().movedim(0, 0).reshape(
            weight.shape[0], -1)
        flat_scale = scale.detach().movedim(0, 0).reshape(
            weight.shape[0], -1)[:, :1]
        weight_codes = torch.round(flat_weight / flat_scale).clamp(-qmax, qmax)
        row = accumulators.setdefault((name, bits), {
            "model": model_name, "module": name, "role": "weight", "bits": bits,
            "calls": 0, "numel": 0, "weight_elements": int(weight.numel()),
            "gradient_weighted_error": 0.0,
            "normalized_gradient_weighted_error": 0.0,
            "output_mse_sum": 0.0, "signal_power_sum": 0.0,
            "noise_power_sum": 0.0,
            "zero_ratio_sum": 0.0, "nonpositive_ratio_sum": 0.0,
            "sign_flip_ratio_sum": 0.0, "quantized_min": float(scale.min().item()),
            "quantized_max": float(scale.max().item()),
            "saturation_ratio_sum": 0.0,
        })
        row["calls"] += 1
        row["numel"] += metrics["numel"]
        row["gradient_weighted_error"] += metrics["gradient_weighted_error"]
        row["normalized_gradient_weighted_error"] += metrics[
            "normalized_gradient_weighted_error"]
        for key in ("output_mse", "zero_ratio", "nonpositive_ratio",
                    "sign_flip_ratio"):
            row[key + "_sum"] += metrics[key]
        row["signal_power_sum"] += metrics["signal_power"] * metrics["numel"]
        row["noise_power_sum"] += metrics["noise_power"] * metrics["numel"]
        row["saturation_ratio_sum"] += float(torch.logical_or(
            weight_codes == -qmax, weight_codes == qmax).float().mean().item())


def _activation_rows(model_name: str, module_values, module_roles, maxima,
                     accumulators, module_names, bits):
    for name in module_names:
        role = module_roles[name]
        maximum = maxima[name]
        quantizer = _role_quantizer(role, bits, maximum)
        for tensor in module_values[name]:
            if tensor.grad is None:
                raise RuntimeError("module activation gradient is missing: %s" % name)
            quantized, codes = quantizer.quantize_with_codes(tensor)
            metrics = _quant_metrics(tensor, quantized, tensor.grad)
            row = accumulators.setdefault((name, bits), {
                "model": model_name, "module": name, "role": role,
                "bits": bits, "calls": 0, "numel": 0,
                "activation_elements": 0, "gradient_weighted_error": 0.0,
                "normalized_gradient_weighted_error": 0.0,
                "output_mse_sum": 0.0, "signal_power_sum": 0.0,
                "noise_power_sum": 0.0,
                "zero_ratio_sum": 0.0, "nonpositive_ratio_sum": 0.0,
                "sign_flip_ratio_sum": 0.0,
                "saturation_ratio_sum": 0.0,
            })
            row["calls"] += 1
            row["numel"] += metrics["numel"]
            row["activation_elements"] += metrics["numel"]
            row["gradient_weighted_error"] += metrics[
                "gradient_weighted_error"]
            row["normalized_gradient_weighted_error"] += metrics[
                "normalized_gradient_weighted_error"]
            for key in ("output_mse", "sqnr_db", "zero_ratio",
                        "nonpositive_ratio", "sign_flip_ratio"):
                if key != "sqnr_db":
                    row[key + "_sum"] += metrics[key]
            row["signal_power_sum"] += metrics["signal_power"] * metrics["numel"]
            row["noise_power_sum"] += metrics["noise_power"] * metrics["numel"]
            row["saturation_ratio_sum"] += float(torch.logical_or(
                codes == quantizer.qmin, codes == quantizer.qmax).float().mean().item())


def _signal_rows(model_name: str, signal_values, roles, maxima, accumulators,
                 bits):
    for site, iteration, tensor in _semantic_signal_tensors(signal_values):
        if not site.startswith("signal::"):
            continue
        if site in ("signal::offset_logits", "signal::affinity_logits"):
            continue
        if site in NON_DIFFERENTIABLE_SIGNALS or site.startswith("input::"):
            continue
        key = "%s[%d]" % (site, iteration)
        role = roles[site]
        maximum = maxima[key]
        quantizer = _role_quantizer(role, bits, maximum)
        if tensor.grad is None:
            if "%s::%s" % (model_name, site) in DERIVED_NON_GRAPH_SIGNALS:
                continue
            raise RuntimeError("semantic signal gradient is missing: %s" % key)
        quantized, codes = quantizer.quantize_with_codes(tensor)
        metrics = _quant_metrics(tensor, quantized, tensor.grad)
        row = accumulators.setdefault((key, bits), {
            "model": model_name, "signal": site, "iteration": iteration,
            "role": role, "bits": bits, "calls": 0, "numel": 0,
            "gradient_weighted_error": 0.0,
            "normalized_gradient_weighted_error": 0.0,
            "output_mse_sum": 0.0, "signal_power_sum": 0.0,
            "noise_power_sum": 0.0,
            "zero_ratio_sum": 0.0, "nonpositive_ratio_sum": 0.0,
            "sign_flip_ratio_sum": 0.0, "saturation_ratio_sum": 0.0,
        })
        row["calls"] += 1
        row["numel"] += metrics["numel"]
        row["gradient_weighted_error"] += metrics[
            "gradient_weighted_error"]
        row["normalized_gradient_weighted_error"] += metrics[
            "normalized_gradient_weighted_error"]
        for metric in ("output_mse", "sqnr_db", "zero_ratio",
                       "nonpositive_ratio", "sign_flip_ratio"):
            if metric != "sqnr_db":
                row[metric + "_sum"] += metrics[metric]
        row["signal_power_sum"] += metrics["signal_power"] * metrics["numel"]
        row["noise_power_sum"] += metrics["noise_power"] * metrics["numel"]
        row["saturation_ratio_sum"] += float(torch.logical_or(
            codes == quantizer.qmin, codes == quantizer.qmax).float().mean().item())


def _finalize_rows(rows, sample_count):
    output = []
    for row in rows:
        row = dict(row)
        row["calibration_samples"] = int(sample_count)
        calls = float(row["calls"])
        if calls <= 0.0:
            raise RuntimeError("sensitivity row has no calls")
        for key in ("output_mse", "zero_ratio",
                    "nonpositive_ratio", "sign_flip_ratio",
                    "saturation_ratio"):
            row[key] = row.pop(key + "_sum") / calls
        signal_power = row.pop("signal_power_sum")
        noise_power = row.pop("noise_power_sum")
        if noise_power == 0.0:
            row["sqnr_db"] = float("inf")
        elif signal_power == 0.0:
            row["sqnr_db"] = float("-inf")
        else:
            row["sqnr_db"] = 10.0 * math.log10(signal_power / noise_power)
        output.append(row)
    return tuple(output)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RuntimeError("cannot write empty sensitivity CSV: %s" % path)
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise RuntimeError("sensitivity CSV row schema differs: %s" % path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(config: Path, model_name: str, device: str, calibration_batch_size: int,
        output: Path) -> Path:
    if model_name not in MODEL_NAMES:
        raise ValueError("unknown model: %s" % model_name)
    if output.exists():
        raise FileExistsError("sensitivity output already exists: %s" % output)
    payload = json.loads(Path(config).read_text(encoding="utf-8"))
    spec = payload["models"][model_name]
    runtime = NYUModelRuntime.from_args(_runtime_args(spec, device))
    model = runtime.build_model(runtime.device)
    contract = _build_contract(model_name, model)
    semantic = install_model_semantic_adapter(
        model, model_name=model_name, strict=True)
    semantic.observe()
    manifest = tuple(semantic.module_manifest(model))
    manifest_by_name = dict((row["name"], row) for row in manifest)
    projection_names = tuple(
        rtn_runner.propagation_projection_outputs(model_name, model))
    module_names = tuple(dict.fromkeys(contract.weight_modules + projection_names))
    module_roles = dict(
        (name, manifest_by_name[name]["role"]) for name in module_names)
    if not set(contract.weight_modules).issubset(set(module_names)):
        raise RuntimeError("semantic module coverage differs from contract")
    capture = ModuleOutputCapture(model, module_names)
    weight_names = tuple(contract.weight_modules + projection_names)
    trainset = runtime.build_dataset("train")
    calibration_indices = _calibration_indices(spec, trainset)
    calibration_batches = _calibration_batches(
        runtime, trainset, calibration_indices, calibration_batch_size)
    maxima = {}
    try:
        for batch in calibration_batches:
            _, _, module_values, signal_values, _ = _forward(
                runtime, model, semantic, capture, batch, runtime.device, False)
            for name, tensors in module_values.items():
                _update_maximum(maxima, name, tensors)
                if name in projection_names and model_name != "cspn":
                    for signal_name, tensor in _projection_parts(
                            model_name, tensors[0]):
                        _update_maximum(
                            maxima, "signal::%s[%d]" % (signal_name, 0),
                            (tensor,))
            for site, iteration, tensor in _semantic_signal_tensors(signal_values):
                if site not in ("signal::offset_logits",
                                "signal::affinity_logits"):
                    _update_maximum(maxima, "%s[%d]" % (site, iteration), (tensor,))
        roles = _semantic_roles(semantic)
        module_maxima = dict((name, maxima[name]) for name in module_names)
        gradient_weights = {}
        gradient_activations = {}
        gradient_signals = {}
        for batch in calibration_batches:
            model.zero_grad(set_to_none=True)
            prediction, target, module_values, signal_values, _ = _forward(
                runtime, model, semantic, capture, batch, runtime.device, True)
            loss = sweep.compute_loss(runtime.saved_args, prediction, target)
            sweep.validate_batch_numerics(prediction, target, loss)
            loss.backward()
            for bits in BIT_OPTIONS:
                _weight_rows(model_name, model, weight_names,
                             gradient_weights, bits)
                _activation_rows(
                    model_name, module_values, module_roles, module_maxima,
                    gradient_activations, module_names, bits)
                _signal_rows(model_name, signal_values, roles,
                             maxima, gradient_signals, bits)
                for projection_name in projection_names:
                    projection = module_values[projection_name][0]
                    if model_name == "cspn":
                        continue
                    for signal_name, tensor in _projection_parts(
                            model_name, projection):
                        key = "signal::%s[0]" % signal_name
                        role = roles[key.split("[")[0]]
                        quantizer = _role_quantizer(
                            role, bits, maxima[key])
                        if projection.grad is None:
                            raise RuntimeError(
                                "propagation projection gradient is missing: %s" %
                                projection_name)
                        offset = projection.shape[1] // 3
                        if model_name == "cspn":
                            gradient = projection.grad
                        elif signal_name == "offset_logits":
                            gradient = projection.grad[:, :2 * offset]
                        else:
                            gradient = projection.grad[:, 2 * offset:3 * offset]
                        quantized, codes = quantizer.quantize_with_codes(tensor)
                        metrics = _quant_metrics(tensor, quantized, gradient)
                        row = gradient_signals.setdefault((key, bits), {
                            "model": model_name, "signal": key.split("[")[0],
                            "iteration": 0, "role": role, "bits": bits,
                            "calls": 0, "numel": 0,
                            "gradient_weighted_error": 0.0,
                            "normalized_gradient_weighted_error": 0.0,
                            "output_mse_sum": 0.0, "signal_power_sum": 0.0,
                            "noise_power_sum": 0.0,
                            "zero_ratio_sum": 0.0, "nonpositive_ratio_sum": 0.0,
                            "sign_flip_ratio_sum": 0.0,
                            "saturation_ratio_sum": 0.0,
                        })
                        row["calls"] += 1
                        row["numel"] += metrics["numel"]
                        row["gradient_weighted_error"] += metrics[
                            "gradient_weighted_error"]
                        row["normalized_gradient_weighted_error"] += metrics[
                            "normalized_gradient_weighted_error"]
                        for metric in ("output_mse", "zero_ratio",
                                       "nonpositive_ratio", "sign_flip_ratio"):
                            row[metric + "_sum"] += metrics[metric]
                        row["signal_power_sum"] += metrics[
                            "signal_power"] * metrics["numel"]
                        row["noise_power_sum"] += metrics[
                            "noise_power"] * metrics["numel"]
                        row["saturation_ratio_sum"] += float(torch.logical_or(
                            codes == quantizer.qmin,
                            codes == quantizer.qmax).float().mean().item())
        weight_rows = _finalize_rows(
            gradient_weights.values(), len(calibration_indices))
        activation_rows = _finalize_rows(
            gradient_activations.values(), len(calibration_indices))
        signal_rows = _finalize_rows(
            gradient_signals.values(), len(calibration_indices))
    finally:
        capture.close()
        semantic.close()
        runtime.close()
    output.mkdir(parents=True)
    _write_csv(output / "weight_sensitivity.csv", weight_rows)
    _write_csv(output / "module_sensitivity.csv", activation_rows)
    _write_csv(output / "propagation_signal_sensitivity.csv", signal_rows)
    ranking = {
        "weight": [
            {"module": row["module"], "bits": row["bits"],
             "score": row["normalized_gradient_weighted_error"]}
            for row in sorted(weight_rows, key=lambda row: (
                -float(row["normalized_gradient_weighted_error"]),
                str(row["module"]), int(row["bits"])))],
        "activation": [
            {"module": row["module"], "bits": row["bits"],
             "score": row["normalized_gradient_weighted_error"]}
            for row in sorted(activation_rows, key=lambda row: (
                -float(row["normalized_gradient_weighted_error"]),
                str(row["module"]), int(row["bits"])))],
        "propagation": [
            {"signal": row["signal"], "iteration": row["iteration"],
             "bits": row["bits"],
             "score": row["normalized_gradient_weighted_error"]}
            for row in sorted(signal_rows, key=lambda row: (
                -float(row["normalized_gradient_weighted_error"]),
                str(row["signal"]), int(row["iteration"]), int(row["bits"])))],
    }
    (output / "gradient_rankings.json").write_text(
        json.dumps(ranking, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "model": model_name,
        "device": device,
        "calibration_count": len(calibration_indices),
        "calibration_batch_size": int(calibration_batch_size),
        "calibration_indices": list(calibration_indices),
        "bits": list(BIT_OPTIONS),
        "weight_quantization": "per_output_channel_symmetric",
        "activation_quantization": "static_per_tensor_minmax",
        "task_loss": str(runtime.saved_args.loss),
        "module_count": len(module_names),
        "weight_rows": len(weight_rows),
        "activation_rows": len(activation_rows),
        "propagation_signal_rows": len(signal_rows),
        "contract_blocks": list(contract.block_names),
    }
    (output / "sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--calibration-batch-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main():
    args = build_parser().parse_args(tuple(sys.argv[1:]))
    print(run(args.config, args.model, args.device,
              args.calibration_batch_size, args.output))


if __name__ == "__main__":
    main()
