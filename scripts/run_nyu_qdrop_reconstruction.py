#!/usr/bin/env python3
"""Strict official-aligned QDrop W4A4 reconstruction for NYU models."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train_nyu_iteration_sweep as sweep  # noqa: E402
from scripts import run_nyu_cspn_stem_precision as stem_runner  # noqa: E402
from scripts.export_nyu_predictions import (  # noqa: E402
    build_model,
    load_run_args,
)
from scripts.hardware_aligned_quantization import (  # noqa: E402
    HardwareAlignedInstrumentor,
    prepare_hardware_model,
)
from scripts.nyu_quantization_analysis import classify_module  # noqa: E402
from scripts.run_nyu_rtn_quantization import (  # noqa: E402
    batch_from_sample,
    calibration_dataset,
    seeded_sample,
)
from scripts.run_nyu_strict_reconstruction import (  # noqa: E402
    TargetCapture,
    maximum_primary_fold_error,
    module_at,
)
from spn_quant.adaptive_rounding import AdaptiveRoundingConfig  # noqa: E402
from spn_quant.adapters import CompletionFormerJointAdapter  # noqa: E402
from spn_quant.deployment_contract import (  # noqa: E402
    file_sha256,
    validate_graph_preparation,
)
from spn_quant.propagation import (  # noqa: E402
    PropagationQuantConfig,
    install_propagation_adapter,
    propagation_projection_outputs,
)
from spn_quant.qdrop_config import load_qdrop_config  # noqa: E402
from spn_quant.qdrop_contract import (  # noqa: E402
    build_qdrop_contract,
    save_qdrop_contract,
)
from spn_quant.qdrop_edges import QDropActivationBank  # noqa: E402
from spn_quant.qdrop_reconstruction import (  # noqa: E402
    QDropBlockReconstructor,
    QDropCalibrationRecord,
    QDropOptimizerConfig,
)
from spn_quant.qdrop_targets import (  # noqa: E402
    WEIGHT_TYPES,
    resolve_qdrop_targets,
)
from spn_quant.strict_reconstruction import detach_cpu  # noqa: E402


FOLD_MAX_ABS_ERROR = 0.05


@dataclass(frozen=True)
class CalibrationSplit:
    calibration: tuple[int, ...]
    reconstruction: tuple[int, ...]
    validation: tuple[int, ...]


@dataclass(frozen=True)
class SeededBatch:
    indices: tuple[int, ...]
    sample: dict[str, object]


def write_json(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


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


def build_calibration_split(*, calibration_indices,
                            reconstruction_samples, validation_samples):
    calibration = tuple(int(index) for index in calibration_indices)
    reconstruction_samples = int(reconstruction_samples)
    validation_samples = int(validation_samples)
    if len(calibration) != reconstruction_samples + validation_samples:
        raise ValueError("QDrop calibration split size mismatch")
    if len(set(calibration)) != len(calibration):
        raise ValueError("QDrop calibration indices must be unique")
    if any(index < 0 for index in calibration):
        raise ValueError("QDrop calibration indices must be nonnegative")
    reconstruction = calibration[:reconstruction_samples]
    validation = calibration[reconstruction_samples:]
    if set(reconstruction) & set(validation):
        raise RuntimeError("QDrop calibration split contract failed")
    return CalibrationSplit(
        calibration=calibration,
        reconstruction=reconstruction,
        validation=validation,
    )


def select_probability_candidate(rows, expected_probabilities):
    rows = tuple(rows)
    expected = tuple(float(value) for value in expected_probabilities)
    if len(set(expected)) != len(expected):
        raise ValueError("QDrop probability candidates contain duplicates")
    actual = tuple(float(row["probability"]) for row in rows)
    if len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise ValueError("QDrop probability candidate set is incomplete")
    for row in rows:
        if not math.isfinite(float(row["validation_loss"])):
            raise ValueError("QDrop candidate validation loss is non-finite")
        if int(row["finite"]) != 1:
            raise ValueError("QDrop candidate output is non-finite")
        if int(row["failed_targets"]) != 0:
            raise ValueError("QDrop candidate has failed targets")
    return min(
        rows,
        key=lambda row: (
            float(row["validation_loss"]),
            float(row["probability"])))


def resolve_execution_order(model, targets, model_args):
    modules = dict(model.named_modules())
    targets = tuple(str(name) for name in targets)
    missing = sorted(name for name in targets if name not in modules)
    if missing:
        raise KeyError("unknown QDrop execution targets: %s" % missing)
    calls = []
    handles = []
    for name in targets:
        handles.append(modules[name].register_forward_pre_hook(
            lambda module, inputs, target=name: calls.append(target)))
    with torch.no_grad():
        model(*model_args)
    for handle in handles:
        handle.remove()
    counts = dict((name, calls.count(name)) for name in targets)
    invalid = dict(
        (name, count) for name, count in counts.items() if count != 1)
    if invalid:
        raise RuntimeError(
            "QDrop targets must execute exactly once: %s" % invalid)
    if len(calls) != len(targets):
        raise RuntimeError("QDrop execution order contains duplicate calls")
    return tuple(calls)


def merge_contracts(destination, source, family):
    overlap = set(destination) & set(source)
    if overlap:
        raise RuntimeError(
            "duplicate QDrop %s contracts: %s" %
            (str(family), sorted(overlap)))
    destination.update(source)


def validate_phase_seed(phase, seed, formal_seeds):
    if phase == "formal" and int(seed) not in set(formal_seeds):
        raise ValueError("formal QDrop seed is absent from the configuration")


def algorithm_probability(algorithm):
    if algorithm == "qdrop":
        return 0.5
    if algorithm == "brecq":
        return 1.0
    raise ValueError("unsupported reconstruction algorithm: %s" % algorithm)


def strict_method(algorithm):
    if algorithm == "qdrop":
        return "qdrop_strict"
    if algorithm == "brecq":
        return "brecq_joint_strict"
    raise ValueError("unsupported reconstruction algorithm: %s" % algorithm)


def configure_validation_propagation(adapter):
    adapter.configure(PropagationQuantConfig(
        affinity_bits=8,
        confidence_bits=8,
        offset_bits=8,
        state_bits=8,
        coefficient_fraction_bits=13,
    ))


def build_strict_manifest(method, model, contract, targets, precision,
                          weight_bits, activation_bits, protocol):
    return {
        "format_version": 3,
        "strict": 1,
        "method": str(method),
        "model": str(model),
        "deployment_contract": str(contract),
        "targets": list(targets),
        "weight_bits": int(weight_bits),
        "activation_bits": int(activation_bits),
        "activation_policy": "exact_semantic_edge_contract",
        "precision": str(precision),
        "protocol": dict(protocol),
    }


def _prepare_saved_args(run_dir, data_root, model_name):
    saved_args = load_run_args(run_dir)
    if saved_args.model != model_name:
        raise RuntimeError(
            "QDrop model/run mismatch: %s != %s" %
            (model_name, saved_args.model))
    if not str(saved_args.device).startswith("cuda"):
        raise RuntimeError("QDrop run metadata must select a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for strict QDrop reconstruction")
    saved_args.data_root = str(data_root)
    saved_args.workers = 0
    saved_args.batch_size = 1
    saved_args.val_batch_size = 1
    saved_args.max_train_samples = 0
    saved_args.max_val_samples = 0
    saved_args.cudnn_benchmark = False
    return saved_args


def _resolve_checkpoint(run_dir, checkpoint):
    path = Path(checkpoint)
    if not path.is_absolute():
        path = Path(run_dir) / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def _evaluation_indices(dataset_size, count, seed):
    if int(count) > int(dataset_size):
        raise ValueError("QDrop evaluation dataset is too small")
    return tuple(int(index) for index in np.random.RandomState(
        int(seed)).choice(int(dataset_size), int(count), replace=False))


def _model_input(saved_args, dataset, index, device, seed):
    sample = seeded_sample(dataset, index, seed)
    batch = batch_from_sample(sample)
    return sweep.batch_to_model_input(saved_args.model, batch, device)


def stack_seeded_samples(dataset, indices, seed):
    samples = [
        seeded_sample(dataset, index, seed) for index in indices]
    if not samples:
        raise ValueError("QDrop capture batch cannot be empty")
    keys = tuple(samples[0])
    if any(tuple(sample) != keys for sample in samples):
        raise ValueError("QDrop capture sample keys do not match")
    batch = {}
    for key in keys:
        values = [sample[key] for sample in samples]
        first = values[0]
        if torch.is_tensor(first):
            if any(not torch.is_tensor(value) or
                   value.shape != first.shape or
                   value.dtype != first.dtype for value in values):
                raise ValueError(
                    "QDrop capture tensor structures do not match")
            batch[key] = torch.stack(values, dim=0)
        else:
            if any(type(value) is not type(first) or value != first
                   for value in values):
                raise ValueError(
                    "QDrop capture non-tensor values do not match")
            batch[key] = first
    return batch


def build_seeded_batches(dataset, indices, seed, batch_size):
    indices = tuple(int(index) for index in indices)
    batch_size = int(batch_size)
    if not indices:
        raise ValueError("QDrop calibration indices cannot be empty")
    if batch_size <= 0:
        raise ValueError("QDrop capture batch size must be positive")
    return tuple(
        SeededBatch(
            indices=indices[start:start + batch_size],
            sample=stack_seeded_samples(
                dataset, indices[start:start + batch_size], seed),
        )
        for start in range(0, len(indices), batch_size)
    )


def _model_batch(saved_args, batch, device):
    return sweep.batch_to_model_input(
        saved_args.model, batch.sample, device)


def _prepare_models(saved_args, checkpoint, batch, device):
    student, architecture = build_model(saved_args, checkpoint, device)
    teacher, _ = build_model(saved_args, checkpoint, device)
    student.eval()
    teacher.eval()
    first_sample = {}
    for key in batch.sample:
        value = batch.sample[key]
        first_sample[key] = value[:1] if torch.is_tensor(value) else value
    model_args, _ = sweep.batch_to_model_input(
        saved_args.model, first_sample, device)
    excluded_pairs = (
        (("conv1_1", "bn1"),)
        if saved_args.model == "cspn" else ())
    teacher_preparation = prepare_hardware_model(
        teacher, model_args, excluded_pairs=excluded_pairs, fold=True)
    student_preparation = prepare_hardware_model(
        student, model_args, excluded_pairs=excluded_pairs, fold=True)
    graph_contract = {
        "fold": 1,
        "excluded_pairs": [list(pair) for pair in excluded_pairs],
        "folded_pairs": student_preparation["folded_pairs"],
        "unfolded_fanout_pairs": student_preparation[
            "unfolded_fanout_pairs"],
        "unfolded_conv_bn_pairs": student_preparation[
            "unfolded_conv_bn_pairs"],
    }
    validate_graph_preparation(teacher_preparation, graph_contract)
    if maximum_primary_fold_error(
            teacher_preparation,
            student_preparation) > FOLD_MAX_ABS_ERROR:
        raise RuntimeError("Conv-BN folding exceeded strict QDrop tolerance")
    return student, teacher, architecture, graph_contract, model_args, \
        student_preparation


def _joint_adapter(model_name, model, precision):
    if model_name != "completionformer":
        return None
    return CompletionFormerJointAdapter(
        model=model,
        expected_attention_modules=16,
        expected_concat_modules=16,
        weight_bits=precision.weight_bits,
        qkv_bits=precision.activation_bits,
        probability_bits=8,
        concat_bits=precision.activation_bits,
        output_bits=precision.activation_bits,
        clip_factors=(1.0,),
        search_rounds=1,
        cache_sample_limit=1,
        cache_byte_limit=1,
    )


def _instrumentor(saved_args, model, preparation, joint_adapter):
    propagation_outputs = set(propagation_projection_outputs(
        saved_args.model, model))
    owned_inputs = set()
    owned_outputs = set(propagation_outputs)
    if joint_adapter is not None:
        owned_inputs.update(joint_adapter.externally_owned_inputs())
        owned_outputs.update(joint_adapter.externally_owned_outputs())
    group_fn = lambda name, module: classify_module(
        saved_args.model, name, module)
    return HardwareAlignedInstrumentor(
        model=model,
        group_fn=group_fn,
        fused_relu_producers=preparation["fused_relu_producers"],
        fuse_layernorm=True,
        externally_owned_outputs=owned_outputs,
        externally_owned_inputs=owned_inputs,
    )


def _calibrate(saved_args, model, batches, device, instrumentor,
               propagation_adapter, joint_adapter, precision):
    instrumentor.observe()
    propagation_adapter.observe()
    if joint_adapter is not None:
        joint_adapter.observe_qdrop_ranges()
    total = sum(len(batch.indices) for batch in batches)
    completed = 0
    with torch.no_grad():
        for batch in batches:
            model_args, _ = _model_batch(saved_args, batch, device)
            model(*model_args)
            completed += len(batch.indices)
            if completed % 32 == 0 or completed == total:
                print(
                    "QDrop activation calibration %d/%d" %
                    (completed, total), flush=True)
    if joint_adapter is not None:
        joint_adapter.freeze_qdrop_ranges()
    instrumentor.freeze()
    propagation_adapter.freeze()
    groups = set(instrumentor.groups.values())
    instrumentor.configure(
        w_bits=precision.weight_bits,
        a_bits=precision.activation_bits,
        enabled_groups=groups,
        activation_overrides={},
        weight_bit_overrides={},
        activation_bit_overrides={},
        external_output_ownership=True,
        quantize_bias=False,
    )


class QDropTargetCapture(TargetCapture):
    def __init__(self, module):
        self.inputs = []
        self.outputs = []
        self.handles = [
            module.register_forward_pre_hook(self._pre, prepend=True),
            module.register_forward_hook(self._post),
        ]


def _capture_records(saved_args, teacher, student, teacher_target,
                     student_target, batches, device):
    teacher_capture = QDropTargetCapture(teacher_target)
    student_capture = QDropTargetCapture(student_target)
    records = []
    total = sum(len(batch.indices) for batch in batches)
    completed = 0
    for batch in batches:
        model_args, _ = _model_batch(saved_args, batch, device)
        teacher_capture.reset()
        student_capture.reset()
        with torch.no_grad():
            teacher(*model_args)
            student(*model_args)
        teacher_inputs, teacher_output = teacher_capture.require_one("teacher")
        student_inputs, _ = student_capture.require_one("student")
        records.append(QDropCalibrationRecord(
            quantized_inputs=detach_cpu(student_inputs),
            full_precision_inputs=detach_cpu(teacher_inputs),
            reference=detach_cpu(teacher_output),
        ))
        completed += len(batch.indices)
        if completed % 32 == 0 or completed == total:
            print(
                "QDrop target capture %d/%d" %
                (completed, total), flush=True)
    teacher_capture.close()
    student_capture.close()
    return records


def _optimizer_config(config, phase, probability, seed):
    steps = config.search.steps \
        if phase == "probability-search" else config.reconstruction.steps
    return QDropOptimizerConfig(
        steps=steps,
        batch_size=config.reconstruction.batch_size,
        cache_cuda_byte_limit=
            config.reconstruction.cache_cuda_byte_limit,
        weight_learning_rate=config.reconstruction.weight_learning_rate,
        activation_learning_rate=
            config.reconstruction.activation_learning_rate,
        round_loss_weight=config.reconstruction.round_loss_weight,
        warmup_fraction=config.reconstruction.warmup_fraction,
        beta_start=config.reconstruction.beta_start,
        beta_end=config.reconstruction.beta_end,
        loss_power=config.reconstruction.loss_power,
        quant_probability=float(probability),
        seed=int(seed),
    )


def strict_prediction_metrics(gt, prediction):
    if not torch.is_tensor(gt) or not torch.is_tensor(prediction):
        raise TypeError("strict depth metrics require tensors")
    if gt.shape != prediction.shape:
        raise ValueError("strict depth metric shapes differ")
    valid_gt = torch.isfinite(gt) & (gt > 1.0e-4)
    if not bool(valid_gt.any().item()):
        raise ValueError("strict depth metrics require valid ground truth")
    values = prediction[valid_gt]
    finite = torch.isfinite(values)
    nonfinite_pixels = int((~finite).sum().item())
    nonpositive_pixels = int(
        (finite & (values <= 1.0e-4)).sum().item())
    invalid_pixels = nonfinite_pixels + nonpositive_pixels
    finite_values = values[finite]
    prediction_min = float(finite_values.min().item()) \
        if finite_values.numel() else float("-inf")
    diagnostics = {
        "nonfinite_pixels": nonfinite_pixels,
        "nonpositive_pixels": nonpositive_pixels,
        "invalid_pixels": invalid_pixels,
        "prediction_min": prediction_min,
    }
    if invalid_pixels:
        return {
            "RMSE": float("inf"),
            "MAE": float("inf"),
            "ABS_REL": float("inf"),
            **diagnostics,
        }
    target = gt[valid_gt].double()
    estimate = values.double()
    error = estimate - target
    return {
        "RMSE": float(error.square().mean().sqrt().item()),
        "MAE": float(error.abs().mean().item()),
        "ABS_REL": float((error.abs() / target).mean().item()),
        **diagnostics,
    }


def _evaluate(saved_args, model, dataset, indices, device, seed):
    rows = []
    model.eval()
    with torch.no_grad():
        for index in indices:
            model_args, gt = _model_input(
                saved_args, dataset, index, device, seed)
            prediction = sweep.extract_pred(model(*model_args))
            metric = strict_prediction_metrics(gt, prediction)
            rows.append({
                "sample_index": int(index),
                "RMSE": float(metric["RMSE"]),
                "MAE": float(metric["MAE"]),
                "ABS_REL": float(metric["ABS_REL"]),
                "nonfinite_pixels": int(metric["nonfinite_pixels"]),
                "nonpositive_pixels": int(metric["nonpositive_pixels"]),
                "invalid_pixels": int(metric["invalid_pixels"]),
                "prediction_min": float(metric["prediction_min"]),
            })
    if not rows:
        raise ValueError("strict validation requires evaluation rows")
    return rows


def _target_manifest(plan, model):
    modules = dict(model.named_modules())
    rows = []
    for name in plan.blocks:
        weight_count = sum(
            int(isinstance(module, WEIGHT_TYPES))
            for module in modules[name].modules())
        activation_count = sum(
            int(site.owner_name == name)
            for site in plan.activation_sites)
        rows.append({
            "target": name,
            "module_type": type(modules[name]).__name__,
            "weight_sites": weight_count,
            "activation_sites": activation_count,
        })
    return rows


def run_reconstruction(args, config, probability, split, protocol,
                       phase, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    saved_args = _prepare_saved_args(
        args.run_dir, args.data_root, args.model)
    device = torch.device(saved_args.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = bool(saved_args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(saved_args.allow_tf32)
    checkpoint = _resolve_checkpoint(args.run_dir, args.checkpoint)
    precision = config.precision(args.precision)
    dataset = calibration_dataset(saved_args)
    indices = split.reconstruction \
        if phase == "probability-search" else split.calibration
    calibration_batches = build_seeded_batches(
        dataset,
        split.calibration,
        protocol["evaluation_seed"],
        config.reconstruction.capture_batch_size,
    )
    target_batch_count = math.ceil(
        len(indices) / config.reconstruction.capture_batch_size)
    target_batches = calibration_batches[:target_batch_count]
    target_batch_indices = tuple(
        index for batch in target_batches for index in batch.indices)
    if target_batch_indices != indices:
        raise RuntimeError(
            "QDrop reconstruction split must align with capture batches")
    student, teacher, architecture, graph_contract, model_args, preparation = \
        _prepare_models(
            saved_args, checkpoint, calibration_batches[0], device)
    plan = resolve_qdrop_targets(args.model, student)
    execution_order = resolve_execution_order(
        student, plan.blocks, model_args)
    joint_adapter = _joint_adapter(args.model, student, precision)
    propagation_adapter = install_propagation_adapter(
        args.model, student)
    instrumentor = _instrumentor(
        saved_args, student, preparation, joint_adapter)
    _calibrate(
        saved_args, student, calibration_batches, device,
        instrumentor, propagation_adapter, joint_adapter, precision)
    bank = QDropActivationBank(
        plan=plan,
        instrumentor=instrumentor,
        bits=precision.activation_bits,
        scale_minimum=config.quantization.activation_scale_minimum,
        seed=args.seed,
        joint_adapter=joint_adapter,
    )
    bank.initialize()
    instrumentor._restore_parameters()

    weight_contracts = {}
    activation_contracts = {}
    summaries = []
    histories = {}
    for target_index, target in enumerate(execution_order, 1):
        records = _capture_records(
            saved_args,
            teacher,
            student,
            module_at(teacher, target),
            module_at(student, target),
            target_batches,
            device,
        )
        reconstructor = QDropBlockReconstructor(
            block=module_at(student, target),
            target=target,
            activation_bank=bank,
            weight_config=AdaptiveRoundingConfig(
                bits=precision.weight_bits,
                clip_ratio=config.quantization.weight_clip_ratio),
            optimizer_config=_optimizer_config(
                config, phase, probability,
                args.seed + target_index),
            contract_prefix=target,
        )
        result = reconstructor.fit(records)
        merge_contracts(
            weight_contracts, result.weight_contracts, "weight")
        merge_contracts(
            activation_contracts,
            result.activation_contracts,
            "activation")
        summaries.append({
            "target": target,
            "execution_index": target_index,
            "before_loss": result.before_loss,
            "after_loss": result.after_loss,
            "weight_contracts": len(result.weight_contracts),
            "activation_contracts": len(result.activation_contracts),
            "steps": len(result.history),
        })
        histories[target] = result.history
        print(
            "QDrop reconstructed %d/%d target=%s before=%.8f after=%.8f" %
            (target_index, len(execution_order), target,
             result.before_loss, result.after_loss),
            flush=True)
    bank.disable_randomness()
    if set(activation_contracts) != set(bank.contracts()):
        raise RuntimeError("QDrop activation contract export is incomplete")
    configure_validation_propagation(propagation_adapter)
    validation_rows = _evaluate(
        saved_args, student, dataset, split.validation,
        device, protocol["evaluation_seed"])
    validation_loss = sum(
        row["RMSE"] for row in validation_rows) / len(validation_rows)
    validation_finite = int(all(
        row["invalid_pixels"] == 0 for row in validation_rows))
    contract = build_qdrop_contract(
        method=strict_method(args.algorithm),
        source_checkpoint=checkpoint,
        graph_contract=graph_contract,
        weight_contracts=weight_contracts,
        activation_contracts=activation_contracts,
        targets=plan,
        metadata={
            "model": args.model,
            "algorithm": args.algorithm,
            "architecture": architecture,
            "phase": phase,
            "precision": precision.name,
            "weight_bits": precision.weight_bits,
            "activation_bits": precision.activation_bits,
            "seed": args.seed,
            "data_seed": protocol["evaluation_seed"],
            "quant_probability": float(probability),
            "calibration_indices": list(split.calibration),
            "reconstruction_indices": list(indices),
            "validation_indices": list(split.validation),
            "execution_order": list(execution_order),
            "official_qdrop_commit": config.reference.commit,
            "protocol": dict(protocol),
        },
    )
    contract_path = save_qdrop_contract(
        output / "qdrop_strict_contract.pt", contract)
    write_json(output / "qdrop_configuration.json", {
        "config": asdict(config),
        "phase": phase,
        "model": args.model,
        "algorithm": args.algorithm,
        "precision": precision.name,
        "seed": args.seed,
        "quant_probability": float(probability),
        "execution_order": list(execution_order),
    })
    write_csv(
        output / "qdrop_target_manifest.csv",
        _target_manifest(plan, student))
    write_csv(
        output / "qdrop_activation_manifest.csv",
        bank.manifest())
    write_csv(
        output / "qdrop_reconstruction_summary.csv",
        summaries)
    write_json(
        output / "qdrop_reconstruction_history.json",
        histories)
    write_csv(
        output / "qdrop_validation_metrics.csv",
        validation_rows)
    manifest = build_strict_manifest(
        strict_method(args.algorithm),
        args.model, contract_path.resolve(), plan.blocks,
        precision.name, precision.weight_bits,
        precision.activation_bits, protocol)
    write_json(output / "qdrop_strict_manifest.json", manifest)
    bank.close()
    if joint_adapter is not None:
        joint_adapter.close()
    propagation_adapter.close()
    instrumentor.close()
    return {
        "probability": float(probability),
        "validation_loss": float(validation_loss),
        "finite": validation_finite,
        "failed_targets": 0,
        "contract": str(contract_path.resolve()),
        "output": str(output.resolve()),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--model",
        choices=("cspn", "dyspn", "nlspn", "completionformer"),
        required=True)
    parser.add_argument(
        "--algorithm", choices=("qdrop", "brecq"), required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument(
        "--phase", choices=("probability-search", "formal"),
        required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--calibration-indices", required=True)
    parser.add_argument("--calibration-metadata", required=True)
    parser.add_argument("--evaluation-protocol", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_qdrop_config(args.config)
    config.precision(args.precision)
    if args.algorithm == "qdrop":
        validate_phase_seed(args.phase, args.seed, config.formal.seeds)
    elif args.phase != "formal" or \
            args.seed != config.formal.evaluation_seed:
        raise ValueError(
            "formal BRECQ requires the configured evaluation seed")
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    saved_args = _prepare_saved_args(
        args.run_dir, args.data_root, args.model)
    calibration_indices_path = Path(args.calibration_indices)
    calibration_metadata_path = Path(args.calibration_metadata)
    evaluation_protocol_path = Path(args.evaluation_protocol)
    calibration_payload = json.loads(
        calibration_indices_path.read_text(encoding="utf-8"))
    calibration_metadata = json.loads(
        calibration_metadata_path.read_text(encoding="utf-8"))
    evaluation_metadata = json.loads(
        evaluation_protocol_path.read_text(encoding="utf-8"))
    index_protocol = stem_runner.index_protocol(
        calibration_payload, evaluation_metadata)
    checkpoint = _resolve_checkpoint(args.run_dir, args.checkpoint)
    if Path(calibration_metadata["checkpoint"]).resolve() != checkpoint or \
            Path(evaluation_metadata["checkpoint"]).resolve() != checkpoint:
        raise ValueError("QDrop checkpoint protocol identity differs")
    if Path(calibration_metadata["data_root"]).resolve() != \
            Path(args.data_root).resolve() or \
            Path(evaluation_metadata["data_root"]).resolve() != \
            Path(args.data_root).resolve():
        raise ValueError("QDrop data-root protocol identity differs")
    checkpoint_sha256 = file_sha256(checkpoint)
    if calibration_metadata["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("QDrop checkpoint SHA256 differs")
    if config.formal.evaluation_seed != index_protocol.seed:
        raise ValueError("QDrop evaluation seed differs from protocol")
    trainset = calibration_dataset(saved_args)
    if max(index_protocol.calibration_indices) >= len(trainset):
        raise ValueError("QDrop calibration index exceeds train split")
    split = build_calibration_split(
        calibration_indices=index_protocol.calibration_indices,
        reconstruction_samples=config.search.reconstruction_samples,
        validation_samples=config.search.validation_samples,
    )
    protocol = {
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_indices_sha256": file_sha256(
            calibration_indices_path),
        "calibration_metadata_sha256": file_sha256(
            calibration_metadata_path),
        "evaluation_protocol_sha256": file_sha256(
            evaluation_protocol_path),
        "calibration_indices": list(index_protocol.calibration_indices),
        "reconstruction_indices": list(split.reconstruction),
        "validation_indices": list(split.validation),
        "evaluation_indices": list(index_protocol.evaluation_indices),
        "evaluation_seed": index_protocol.seed,
    }
    if args.phase == "probability-search":
        if args.algorithm != "qdrop":
            raise ValueError("BRECQ does not use probability search")
        rows = []
        for probability in config.search.quant_probabilities:
            label = "candidate_p%03d" % int(round(probability * 100.0))
            rows.append(run_reconstruction(
                args, config, probability, split, protocol,
                args.phase, root / label))
        selected = select_probability_candidate(
            rows, config.search.quant_probabilities)
        write_csv(root / "qdrop_candidate_metrics.csv", rows)
        write_json(root / "selected_probability.json", selected)
        print(
            "QDrop selected probability %.2f validation_RMSE=%.6f" %
            (selected["probability"], selected["validation_loss"]),
            flush=True)
        return
    probability = algorithm_probability(args.algorithm)
    result = run_reconstruction(
        args, config, probability, split, protocol,
        args.phase, root / ("formal_seed_%d" % args.seed))
    write_json(
        root / ("formal_seed_%d.json" % args.seed), result)
    print(
        "%s formal seed=%d validation_RMSE=%.6f" %
        (args.algorithm, args.seed, result["validation_loss"]),
        flush=True)


if __name__ == "__main__":
    main()
