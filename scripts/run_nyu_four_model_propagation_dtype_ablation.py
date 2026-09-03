#!/usr/bin/env python3
"""Compare propagation precision across the four official SPN models."""

from __future__ import annotations

import argparse
import csv
from argparse import Namespace
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from scripts.nyu_model_runtime import NYUModelRuntime  # noqa: E402
from scripts.run_nyu_model_p3t3_search import (  # noqa: E402
    HardDeploymentP3T3Evaluator,
    HardDeploymentSettings,
    _calibration_indices,
    _read_activation_cost_rows,
    _read_weight_cost_rows,
)
from scripts import run_nyu_rtn_quantization as rtn_runner  # noqa: E402
from scripts.run_nyu_nlspn_propagation_dtype_ablation import (  # noqa: E402
    FLOAT_STATE_NAMES,
    _aggregate,
    _metrics,
    _state_rows,
)
from spn_quant import mixed_precision  # noqa: E402
from spn_quant.model_contracts import (  # noqa: E402
    QuantizationBlock,
    QuantizationModelContract,
    build_model_quantization_contract,
)
from spn_quant.adapters.cspn import CSPNSemanticAdapter  # noqa: E402
from spn_quant.qdrop_targets import resolve_qdrop_targets  # noqa: E402


MODEL_NAMES = ("cspn", "dyspn", "nlspn", "completionformer")
PROPAGATION_MODES = (
    "PA",
    "FP32_PROP",
    "BF16_STATE",
    "FP16_STATE",
)
EXPECTED_QUANTIZED_FAILURES = (
    "quantized prediction is non-finite",
    "quantized prediction is non-positive",
)


def _mode_label(bits: int, mode: str) -> str:
    if mode == "PA":
        return "PA_W%dA%d" % (bits, bits)
    if mode == "FP32_PROP":
        return "W%dA%d_FP32_PROP" % (bits, bits)
    if mode == "BF16_STATE":
        return "W%dA%d_BF16_STATE" % (bits, bits)
    if mode == "FP16_STATE":
        return "W%dA%d_FP16_STATE" % (bits, bits)
    raise ValueError("unknown propagation mode: %s" % mode)


def _runtime_args(spec, device: str) -> Namespace:
    return Namespace(
        model=spec["model"],
        run_dir=Path(spec["run_dir"]),
        checkpoint=Path(spec["checkpoint"]),
        expected_architecture_class=spec["expected_architecture_class"],
        required_cuda_extension=spec["required_cuda_extension"],
        propagation_iterations=int(spec["propagation_iterations"]),
        data_root=Path(spec["data_root"]),
        device=device,
        checkpoint_architecture=spec["checkpoint_architecture"],
        native_cuda_operator=spec["native_cuda_operator"],
    )


def _settings(spec, hard, device: str, bits: int):
    return HardDeploymentSettings(
        device=device,
        calibration_metadata=Path(spec["calibration_metadata"]),
        calibration_count=int(spec["calibration_count"]),
        evaluation_indices=tuple(int(index) for index in spec[
            "evaluation_indices"]),
        base_weight_bits=bits,
        base_activation_bits=bits,
        promotion_weight_bits=bits,
        promotion_activation_bits=int(hard["promotion_activation_bits"]),
        fold_conv_bn=bool(hard["fold_conv_bn"]),
        fold_max_error=float(hard["fold_max_error"]),
        joint_clip_factors=tuple(float(value) for value in hard[
            "joint_clip_factors"]),
        joint_search_rounds=int(hard["joint_search_rounds"]),
        joint_cache_sample_limit=int(hard["joint_cache_sample_limit"]),
        joint_cache_byte_limit=int(hard["joint_cache_byte_limit"]),
    )


def _build_cspn_uniform_contract(model):
    plan = resolve_qdrop_targets("cspn", model)
    projection_roots = set(
        name.split(".", 1)[0]
        for name in rtn_runner.propagation_projection_outputs("cspn", model))
    manifest = CSPNSemanticAdapter.module_manifest(model)
    modules = dict((row["name"], row["module"]) for row in manifest)
    module_roles = tuple((row["name"], row["role"]) for row in manifest)
    blocks = []
    for block_name in plan.blocks:
        if block_name in projection_roots:
            continue
        weights = tuple(name for name in modules
                        if name == block_name or
                        name.startswith(block_name + "."))
        if not weights:
            raise ValueError("CSPN contract block has no weight modules: %s" %
                             block_name)
        owners = tuple((site.site, site.role)
                       for site in plan.activation_sites
                       if site.owner_name == block_name)
        blocks.append(QuantizationBlock(block_name, weights, owners))
    block_names = tuple(block.name for block in blocks)
    propagation_modules = tuple(
        name for name, module in model.named_modules()
        if name == "post_process_layer" or any(
            name == root or name.startswith(root + ".")
            for root in projection_roots))
    return QuantizationModelContract(
        model_name="cspn",
        blocks=tuple(blocks),
        prefix_groups=(block_names,),
        tail_groups=(block_names,),
        protected_roles=("propagation_state",),
        attention_edges=(),
        concat_edges=(),
        protected_modules=propagation_modules,
        module_roles=module_roles,
    )


def _cspn_activation_cost_rows(plan, model, runtime, calibration_metadata,
                               calibration_count, evaluation_indices):
    projection_roots = set(
        name.split(".", 1)[0]
        for name in rtn_runner.propagation_projection_outputs("cspn", model))
    sites = tuple(site for site in plan.activation_sites
                  if site.owner_name not in projection_roots)
    trainset = runtime.build_dataset("train")
    indices = _calibration_indices(
        calibration_metadata, calibration_count, evaluation_indices,
        len(trainset))
    batch = rtn_runner.batch_from_sample(rtn_runner.seeded_sample(
        trainset, indices[0], runtime.saved_args.seed))
    model_args, ground_truth = runtime.model_input(batch, runtime.device)
    del ground_truth
    modules = dict(model.named_modules())
    observations = {}
    handles = []
    for site in sites:
        if site.owner_kind != "module_input":
            raise ValueError("CSPN uniform contract has unsupported activation owner")
        module = site.site.split("::")[1]
        observations[site.site] = []
        handles.append(modules[module].register_forward_pre_hook(
            lambda current, inputs, site_name=site.site: observations[
                site_name].append(int(inputs[0].numel()))))
    with torch.no_grad():
        model(*model_args)
    for handle in handles:
        handle.remove()
    output = []
    for site in sites:
        values = observations[site.site]
        if len(values) != 1:
            raise RuntimeError("CSPN activation site call count differs from one: %s" %
                               site.site)
        output.append(((site.site, site.role), values[0]))
    return tuple(output)


def _cspn_weight_cost_rows(plan, model, runtime, calibration_metadata,
                           calibration_count, evaluation_indices):
    rows = {}
    modules = dict(model.named_modules())
    expected = tuple(
        name for block in plan.blocks
        for name in modules
        if isinstance(modules[name], (nn.Conv2d, nn.ConvTranspose2d)) and
        name.split(".", 1)[0] not in set(
            projection_name.split(".", 1)[0]
            for projection_name in rtn_runner.propagation_projection_outputs(
                "cspn", model)) and
        (name == block or name.startswith(block + ".")))
    trainset = runtime.build_dataset("train")
    indices = _calibration_indices(
        calibration_metadata, calibration_count, evaluation_indices,
        len(trainset))
    batch = rtn_runner.batch_from_sample(rtn_runner.seeded_sample(
        trainset, indices[0], runtime.saved_args.seed))
    model_args, ground_truth = runtime.model_input(batch, runtime.device)
    del ground_truth
    observations = dict((name, []) for name in expected)
    handles = [modules[name].register_forward_hook(
        lambda current, inputs, output, module_name=name:
            observations[module_name].append(int(output.numel()))
    ) for name in expected]
    with torch.no_grad():
        model(*model_args)
    for handle in handles:
        handle.remove()
    for name in expected:
        values = observations[name]
        if len(values) != 1:
            raise RuntimeError("CSPN weight module call count differs from one: %s" %
                               name)
        module = modules[name]
        kernel = int(module.kernel_size[0]) * int(module.kernel_size[1])
        input_channels = int(module.in_channels) // int(module.groups)
        rows[name] = values[0] * input_channels * kernel
    return tuple((name, rows[name]) for name in expected)


def _evaluate_mode(evaluator, bits: int, mode: str):
    label = _mode_label(bits, mode)
    evaluator.configure_uniform(bits, bits)
    if mode in ("BF16_STATE", "FP16_STATE", "FP32_PROP"):
        evaluator.propagation_adapter.configure_float(FLOAT_STATE_NAMES[
            "W8A8_" + mode])
    with torch.no_grad():
        prediction, ground_truth = evaluator._forward(
            evaluator.evaluation_batch)
    rows = []
    for position, (sample_index, batch) in enumerate(
            evaluator.evaluation_batches):
        del batch
        row = _metrics(prediction[position], ground_truth[position])
        row["mode"] = label
        row["sample_index"] = int(sample_index)
        rows.append(row)
    return rows, tuple(evaluator.propagation_adapter.last_states()), tuple(
        evaluator.propagation_adapter.statistics())


def _evaluate_fp32_baseline(evaluator):
    evaluator.instrumentor.disable()
    if evaluator.propagation_projection_instrumentor is not None:
        evaluator.propagation_projection_instrumentor.disable()
    evaluator.propagation_adapter.disable()
    if evaluator.joint_adapter is not None:
        evaluator.joint_adapter.unbind_qdrop_sites()
    with torch.no_grad():
        prediction, ground_truth = evaluator._forward(
            evaluator.evaluation_batch)
    rows = []
    for position, (sample_index, batch) in enumerate(
            evaluator.evaluation_batches):
        del batch
        row = _metrics(prediction[position], ground_truth[position])
        row["config"] = "FP32_BASELINE"
        row["mode"] = "FP32_BASELINE"
        row["sample_index"] = int(sample_index)
        rows.append(row)
    return rows


def run(config_path: Path, model_name: str, device: str, bits: int,
        output: Path) -> Path:
    if model_name not in MODEL_NAMES:
        raise ValueError("unknown model: %s" % model_name)
    if bits not in (4, 8):
        raise ValueError("ordinary precision must be W4A4 or W8A8")
    if output.exists():
        raise FileExistsError("ablation output already exists: %s" % output)
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    spec = payload["models"][model_name]
    configured_device = spec["device"]
    hard = payload["hard_deployment"]
    runtime = NYUModelRuntime.from_args(_runtime_args(spec, device))
    model = runtime.build_model(runtime.device)
    contract = _build_cspn_uniform_contract(model) if model_name == "cspn" \
        else build_model_quantization_contract(model_name, model)
    plan = resolve_qdrop_targets(model_name, model)
    if model_name == "cspn":
        weight_rows = _cspn_weight_cost_rows(
            plan, model, runtime,
            Path(spec["calibration_metadata"]), int(spec["calibration_count"]),
            tuple(int(index) for index in spec["evaluation_indices"]))
        activation_rows = _cspn_activation_cost_rows(
            plan, model, runtime,
            Path(spec["calibration_metadata"]),
            int(spec["calibration_count"]),
            tuple(int(index) for index in spec["evaluation_indices"]))
    else:
        weight_rows = _read_weight_cost_rows(Path(spec["weight_cost_rows"]))
        activation_rows = _read_activation_cost_rows(
            Path(spec["activation_cost_rows"]))
    registry = mixed_precision.build_registry(contract, mixed_precision.CostBasis(
        weight_macs=weight_rows,
        activation_elements=activation_rows,
    ))
    evaluator = HardDeploymentP3T3Evaluator(
        runtime, model, contract, registry,
        _settings(spec, hard, device, bits),
        propagation_projection_precision=(
            int(hard["propagation_projection_weight_bits"]),
            int(hard["propagation_projection_activation_bits"])),
    )
    output.mkdir(parents=True)
    mode_rows = _evaluate_fp32_baseline(evaluator)
    states_by_mode = {}
    propagation_statistics = {}
    mode_failures = {}
    projection_manifest = tuple(
        {
            "name": name,
            "module_type": type(module).__name__,
            "weight_shape": list(module.weight.shape),
        }
        for name, module in sorted(
            evaluator.propagation_projection_instrumentor.modules.items()))
    try:
        for mode in PROPAGATION_MODES:
            label = _mode_label(bits, mode)
            try:
                rows, states, statistics = _evaluate_mode(evaluator, bits, mode)
            except RuntimeError as error:
                if str(error) not in EXPECTED_QUANTIZED_FAILURES:
                    raise
                mode_failures[label] = str(error)
                continue
            mode_rows.extend(rows)
            states_by_mode[label] = states
            propagation_statistics[label] = statistics
    finally:
        evaluator.close()
        runtime.close()
    (output / "mode_failures.json").write_text(
        json.dumps(mode_failures, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    reference_label = _mode_label(bits, "FP32_PROP")
    if reference_label not in states_by_mode:
        raise RuntimeError("reference propagation mode failed: %s" %
                           json.dumps(mode_failures, sort_keys=True))
    reference_states = states_by_mode[reference_label]
    state_rows = []
    for mode in PROPAGATION_MODES:
        label = _mode_label(bits, mode)
        if label not in states_by_mode:
            continue
        state_rows.extend(_state_rows(
            label, states_by_mode[label], reference_states))
    summary = []
    baseline = _aggregate(tuple(
        row for row in mode_rows if row["mode"] == "FP32_BASELINE"))
    baseline["mode"] = "FP32_BASELINE"
    baseline["delta_vs_fp32_prop"] = None
    baseline["relative_delta_vs_fp32_prop"] = None
    baseline["delta_vs_fp32_baseline"] = 0.0
    baseline["relative_delta_vs_fp32_baseline"] = 0.0
    summary.append(baseline)
    reference = _aggregate(tuple(
        row for row in mode_rows if row["mode"] == reference_label))
    for mode in PROPAGATION_MODES:
        label = _mode_label(bits, mode)
        if label not in states_by_mode:
            continue
        aggregate = _aggregate(tuple(
            row for row in mode_rows if row["mode"] == label))
        aggregate["mode"] = label
        aggregate["delta_vs_fp32_prop"] = \
            aggregate["pooled_rmse"] - reference["pooled_rmse"]
        aggregate["relative_delta_vs_fp32_prop"] = \
            aggregate["pooled_rmse"] / reference["pooled_rmse"] - 1.0
        aggregate["delta_vs_fp32_baseline"] = \
            aggregate["pooled_rmse"] - baseline["pooled_rmse"]
        aggregate["relative_delta_vs_fp32_baseline"] = \
            aggregate["pooled_rmse"] / baseline["pooled_rmse"] - 1.0
        summary.append(aggregate)
    output_metadata = {
        "model": model_name,
        "device": device,
        "configured_device": configured_device,
        "ordinary_precision": "W%dA%d" % (bits, bits),
        "propagation_projection_precision": [
            int(hard["propagation_projection_weight_bits"]),
            int(hard["propagation_projection_activation_bits"])],
        "propagation_projection_modules": list(projection_manifest),
        "propagation_loop_precision": "float_state_mode",
        "calibration_count": int(spec["calibration_count"]),
        "evaluation_indices": list(spec["evaluation_indices"]),
        "reference_mode": reference_label,
        "fp32_baseline_mode": "FP32_BASELINE",
        "propagation_modes": [_mode_label(bits, mode)
                              for mode in PROPAGATION_MODES],
        "successful_modes": sorted(states_by_mode),
        "failed_modes": mode_failures,
    }
    (output / "metadata.json").write_text(
        json.dumps(output_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (output / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(mode_rows[0]))
        writer.writeheader()
        writer.writerows(mode_rows)
    with (output / "propagation_state_metrics.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(state_rows[0]))
        writer.writeheader()
        writer.writerows(state_rows)
    (output / "propagation_statistics.json").write_text(
        json.dumps(propagation_statistics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--ordinary-bits", type=int, choices=(4, 8), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main():
    args = build_parser().parse_args(tuple(sys.argv[1:]))
    print(run(args.config, args.model, args.device, args.ordinary_bits,
              args.output))


if __name__ == "__main__":
    main()
