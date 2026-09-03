#!/usr/bin/env python3
"""Train task-aware LSQ+ QAT under the unified FP16 propagation protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.nyu_model_runtime import NYUModelRuntime
from scripts.train_nyu_selected_qat import _sample_batch
from scripts.run_nyu_unified_fp16_task_aware_allocation import (
    _runtime_args,
    _build_contract,
    json_safe,
)
from scripts.train_nyu_selected_qat import (
    ModelActivationRangeCollector,
    _joint_adapter,
    _selected_target_plan,
    install_deterministic_qat_operators,
    set_model_qat_train_mode,
)
from spn_quant import mixed_precision
from spn_quant.model_contracts import (
    build_model_quantization_contract,
    validate_propagation_ownership,
)
from spn_quant.propagation import install_propagation_adapter
from spn_quant.propagation.controller import PropagationQuantConfig
from spn_quant.qat.model_methods import (
    ModelMethodQATConfig,
    ModelMethodQATController,
)
from spn_quant.qat.task_loss import (
    ModelTaskCapture,
    ModelTaskLossWeights,
    model_task_aware_loss,
)
from spn_quant.adapters import install_model_semantic_adapter


MODEL_NAMES = ("cspn", "dyspn", "nlspn", "completionformer")
QAT_CONFIGURATIONS = (
    "TASK_AWARE_W4.00_A4.00",
    "TASK_AWARE_W5.00_A5.00",
    "TASK_AWARE_W6.00_A6.00",
)
BIT_LEVELS = (4, 6, 8)


def validate_qat_payload(payload: Mapping[str, object]) -> None:
    protocol = payload["protocol"]
    if str(protocol["propagation_dtype"]).lower() != "fp16":
        raise ValueError("task-aware QAT propagation must be FP16")
    if int(protocol["calibration_count"]) != 128:
        raise ValueError("task-aware QAT calibration count must equal 128")
    if int(protocol["evaluation_count"]) != 64:
        raise ValueError("task-aware QAT evaluation count must equal 64")
    if tuple(protocol["models"]) != MODEL_NAMES:
        raise ValueError("task-aware QAT protocol model order changed")
    if tuple(protocol["configurations"]) != QAT_CONFIGURATIONS:
        raise ValueError("task-aware QAT configuration order changed")
    training = payload["training"]
    required = {
        "method", "epochs", "batch_size", "workers", "learning_rate",
        "momentum", "weight_decay", "max_gradient_norm", "seed",
        "log_interval", "boundary_threshold_m", "depth_loss_weight",
        "boundary_loss_weight", "teacher_loss_weight",
        "initial_depth_loss_weight", "propagation_loss_weight",
        "hawq_range_momentum", "fold_conv_bn", "fold_max_error",
        "joint_clip_factors", "joint_search_rounds",
        "joint_cache_sample_limit", "joint_cache_byte_limit",
    }
    if set(training) != required:
        raise KeyError("task-aware QAT training fields changed")
    if training["method"] != "task_aware":
        raise ValueError("task-aware QAT requires the task_aware method")
    for name in ("epochs", "batch_size", "workers", "log_interval",
                 "joint_search_rounds", "joint_cache_sample_limit",
                 "joint_cache_byte_limit"):
        if int(training[name]) <= 0:
            raise ValueError("task-aware QAT integer setting is invalid: %s" %
                             name)
    if int(training["workers"]) < 0:
        raise ValueError("task-aware QAT workers must be nonnegative")
    numeric = (
        float(training["learning_rate"]), float(training["momentum"]),
        float(training["weight_decay"]),
        float(training["max_gradient_norm"]),
        float(training["boundary_threshold_m"]),
        float(training["hawq_range_momentum"]),
        float(training["fold_max_error"]),
    ) + tuple(float(value) for value in training["joint_clip_factors"]) + \
        tuple(float(training[name]) for name in (
            "depth_loss_weight", "boundary_loss_weight",
            "teacher_loss_weight", "initial_depth_loss_weight",
            "propagation_loss_weight"))
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("task-aware QAT numeric setting is non-finite")
    if not 0.0 <= numeric[1] < 1.0:
        raise ValueError("task-aware QAT momentum must lie in [0, 1)")
    if numeric[2] < 0.0 or numeric[3] <= 0.0 or numeric[4] <= 0.0 or \
            not 0.0 <= numeric[5] < 1.0 or numeric[6] < 0.0:
        raise ValueError("task-aware QAT numeric setting is invalid")
    if any(value < 0.0 for value in numeric[7:]):
        raise ValueError("task-aware QAT loss weight is negative")


def load_qat_payload(path: Path) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_qat_payload(payload)
    for model_name in MODEL_NAMES:
        payload["models"][model_name]
    return payload


def assignment_counts(assignment: mixed_precision.BitAssignment):
    weight = {}
    activation = {}
    for name, bits in assignment.weight_bits:
        del name
        bits = int(bits)
        if bits not in weight:
            weight[bits] = 0
        weight[bits] += 1
    for owner, bits in assignment.activation_bits:
        del owner
        bits = int(bits)
        if bits not in activation:
            activation[bits] = 0
        activation[bits] += 1
    return {"weight": weight, "activation": activation}


def assignment_budget(manifest, configuration):
    budget = manifest["assignments"][configuration]["budget"]
    return (float(budget["actual_weight_bits"]),
            float(budget["actual_activation_bits"]))


def should_replace_best(best_payload, best_value: float,
                        candidate_value: float) -> bool:
    return best_payload is None or candidate_value < best_value


def assignment_payload(assignment: mixed_precision.BitAssignment):
    return {
        "model_name": assignment.model_name,
        "weight_bits": list(assignment.weight_bits),
        "activation_bits": [
            [list(owner), bits] for owner, bits in assignment.activation_bits],
    }


def validate_training_checkpoint(payload, expected_assignment, max_epochs: int):
    required = {
        "epoch", "model_state", "method_state", "optimizer_state",
        "history", "assignment", "propagation_dtype",
    }
    if set(payload) != required:
        raise KeyError("task-aware QAT checkpoint fields changed")
    epoch = int(payload["epoch"])
    if epoch <= 0 or epoch >= int(max_epochs):
        raise ValueError("task-aware QAT resume epoch is invalid")
    if payload["propagation_dtype"] != "fp16":
        raise ValueError("task-aware QAT resume propagation is not FP16")
    if payload["assignment"] != expected_assignment:
        raise ValueError("task-aware QAT resume assignment changed")
    if not isinstance(payload["model_state"], Mapping) or \
            not isinstance(payload["method_state"], Mapping):
        raise TypeError("task-aware QAT resume model state is invalid")
    if set(payload["optimizer_state"]) != {"state", "param_groups"}:
        raise KeyError("task-aware QAT resume optimizer fields changed")
    history = tuple(payload["history"])
    if len(history) != epoch or tuple(
            int(row["epoch"]) for row in history) != tuple(
                range(1, epoch + 1)):
        raise ValueError("task-aware QAT resume history is invalid")
    return epoch + 1


def _contract_owners(contract):
    return tuple(owner for block in contract.blocks
                 for owner in block.activation_owners)


def _load_assignment(source_root: Path, model_name: str,
                     configuration: str, contract):
    manifest = json.loads(
        (source_root / model_name / "manifest.json").read_text(
            encoding="utf-8"))
    record = manifest["assignments"][configuration]["assignment"]
    assignment = mixed_precision.BitAssignment(
        weight_bits=tuple(
            (str(row[0]), int(row[1])) for row in record["weight_bits"]),
        activation_bits=tuple(
            ((str(row[0][0]), str(row[0][1])), int(row[1]))
            for row in record["activation_bits"]),
        model_name=str(record["model_name"]),
    )
    if assignment.model_name != model_name:
        raise ValueError("QAT assignment model differs from runtime")
    if set(name for name, bits in assignment.weight_bits) != \
            set(contract.weight_modules):
        raise ValueError("QAT assignment weight coverage differs")
    if set(owner for owner, bits in assignment.activation_bits) != \
            set(_contract_owners(contract)):
        raise ValueError("QAT assignment activation coverage differs")
    if any(bits not in BIT_LEVELS for name, bits in assignment.weight_bits) or \
            any(bits not in BIT_LEVELS for owner, bits in assignment.activation_bits):
        raise ValueError("QAT assignment bit level is unsupported")
    mixed_precision.validate_assignment_ownership(contract, assignment)
    return assignment


class PreparedTaskAwareQAT(object):
    def __init__(self, student_runtime, teacher_runtime, model, teacher,
                 contract, assignment, controller, propagation, joint,
                 student_semantic, teacher_semantic, trainset, valset,
                 calibration_indices, evaluation_indices):
        self.student_runtime = student_runtime
        self.teacher_runtime = teacher_runtime
        self.model = model
        self.teacher = teacher
        self.contract = contract
        self.assignment = assignment
        self.controller = controller
        self.propagation = propagation
        self.joint = joint
        self.student_semantic = student_semantic
        self.teacher_semantic = teacher_semantic
        self.trainset = trainset
        self.valset = valset
        self.calibration_indices = tuple(calibration_indices)
        self.evaluation_indices = tuple(evaluation_indices)
        self.closed = False

    def close(self):
        if self.closed:
            raise RuntimeError("task-aware QAT resources were closed twice")
        self.teacher_semantic.close()
        self.student_semantic.close()
        self.controller.remove()
        self.propagation.close()
        if self.joint is not None:
            self.joint.close()
        self.teacher_runtime.close()
        self.student_runtime.close()
        self.closed = True


def _calibration_indices(spec, train_size: int):
    metadata = json.loads(
        Path(spec["calibration_metadata"]).read_text(encoding="utf-8"))
    indices = tuple(int(index) for index in metadata["calibration_indices"])
    if len(indices) != 128 or len(set(indices)) != 128:
        raise ValueError("QAT calibration identities are not 128 unique samples")
    if any(index < 0 or index >= int(train_size) for index in indices):
        raise IndexError("QAT calibration index is outside the train split")
    return indices


def _prepare(payload, model_name: str, configuration: str):
    source = json.loads(
        Path(payload["source_config"]).read_text(encoding="utf-8"))
    spec = source["models"][model_name]
    training = payload["training"]
    student_runtime = NYUModelRuntime.from_args(
        _runtime_args(spec, str(spec["device"])))
    teacher_runtime = NYUModelRuntime.from_args(
        _runtime_args(spec, str(spec["device"])))
    device = student_runtime.device
    model = student_runtime.build_model(device)
    teacher = teacher_runtime.build_model(device)
    deterministic_student = install_deterministic_qat_operators(model_name, model)
    deterministic_teacher = install_deterministic_qat_operators(model_name, teacher)
    del deterministic_student, deterministic_teacher
    contract = _build_contract(model_name, model)
    validate_propagation_ownership(contract, model)
    target_plan = _selected_target_plan(model_name, model, contract)
    assignment = _load_assignment(
        Path(payload["source_output_root"]), model_name, configuration,
        contract)
    trainset = student_runtime.build_dataset("train")
    valset = student_runtime.build_dataset("val")
    calibration_indices = _calibration_indices(spec, len(trainset))
    evaluation_indices = tuple(int(index) for index in spec["evaluation_indices"])
    if len(evaluation_indices) != 64 or len(set(evaluation_indices)) != 64:
        raise ValueError("QAT evaluation identities are not 64 unique samples")
    first = _sample_batch(trainset, calibration_indices[0], int(training["seed"]))
    model_args, target = student_runtime.model_input(first, device)
    del target
    from scripts.hardware_aligned_quantization import prepare_hardware_model
    preparation = prepare_hardware_model(
        model, model_args, fold=bool(training["fold_conv_bn"]))
    if float(preparation["primary_max_abs_error"]) > \
            float(training["fold_max_error"]):
        raise RuntimeError("task-aware QAT Conv-BN preparation exceeds threshold")
    joint = _joint_adapter(model, contract, training)
    propagation = install_propagation_adapter(model_name, model)
    collector = ModelActivationRangeCollector(model, contract, target_plan)
    propagation.observe()
    if joint is not None:
        joint.observe_qdrop_ranges()
    model.eval()
    with torch.no_grad():
        for index in calibration_indices:
            batch = _sample_batch(trainset, index, int(training["seed"]))
            calibration_args, calibration_target = \
                student_runtime.model_input(batch, device)
            del calibration_target
            model(*calibration_args)
    propagation.freeze()
    propagation.configure_float("fp16")
    if joint is not None:
        joint.freeze_qdrop_ranges()
    initialization_rows = collector.initialization_rows(joint)
    collector.close()
    controller = ModelMethodQATController(
        model,
        contract,
        target_plan,
        ModelMethodQATConfig(
            method="task_aware",
            weight_bits=assignment.weight_bits,
            activation_bits=assignment.activation_bits,
            propagation=PropagationQuantConfig(),
            propagation_mode="integer",
            hawq_range_momentum=float(training["hawq_range_momentum"]),
        ),
        joint_adapter=joint,
        propagation_adapter=None,
    )
    controller.initialize_activations(initialization_rows)
    controller.install()
    student_semantic = install_model_semantic_adapter(
        model, model_name, strict=True)
    student_semantic.delegate_quantization()
    teacher.eval()
    teacher_semantic = install_model_semantic_adapter(
        teacher, model_name, strict=True)
    teacher_semantic.delegate_quantization()
    return PreparedTaskAwareQAT(
        student_runtime, teacher_runtime, model, teacher, contract,
        assignment, controller, propagation, joint, student_semantic,
        teacher_semantic, trainset, valset, calibration_indices,
        evaluation_indices)


def _task_forward(prepared, model_input, target, weights, threshold):
    prepared.teacher.eval()
    prepared.teacher_semantic.begin_task_capture()
    with torch.no_grad():
        teacher_output = prepared.teacher(*model_input)
        teacher_capture = prepared.teacher_semantic.task_capture()
    prepared.student_semantic.begin_task_capture()
    student_output = prepared.model(*model_input)
    student_capture = prepared.student_semantic.task_capture()
    prediction = prepared.student_runtime.prediction(student_output)
    if not torch.equal(prediction, student_capture.prediction):
        raise RuntimeError("student semantic prediction capture changed")
    loss = model_task_aware_loss(
        student_capture,
        teacher_capture,
        target,
        target > 0.0,
        weights,
        threshold,
    )
    if any(not bool(torch.isfinite(value).all().item())
           for value in loss.as_dict().values()):
        raise FloatingPointError("task-aware QAT loss is non-finite")
    return prediction, loss


def _metric(prediction, target):
    valid = torch.isfinite(target) & (target > 1e-4)
    values = prediction[valid]
    finite = bool(torch.isfinite(values).all().item())
    positive = finite and bool((values > 1e-4).all().item())
    if not finite:
        return {"squared_error_sum": math.inf, "valid_pixels": int(valid.sum()),
                "rmse": math.inf, "finite": False, "positive": False}
    difference = values.double() - target[valid].double()
    squared = float(difference.square().sum().item())
    return {
        "squared_error_sum": squared,
        "valid_pixels": int(valid.sum()),
        "rmse": math.sqrt(squared / float(max(int(valid.sum()), 1))),
        "finite": finite,
        "positive": positive,
    }


def _evaluate(prepared, indices):
    rows = []
    prepared.model.eval()
    prepared.controller.activation_modules.eval()
    with torch.no_grad():
        for index in indices:
            sample = _sample_batch(
                prepared.valset, index, int(prepared.student_runtime.saved_args.seed))
            model_input, target = prepared.student_runtime.model_input(
                sample, prepared.student_runtime.device)
            prediction = prepared.student_runtime.prediction(
                prepared.model(*model_input))
            row = _metric(prediction[0], target[0])
            row["sample_index"] = int(index)
            rows.append(row)
    pixels = sum(row["valid_pixels"] for row in rows)
    finite = all(row["finite"] for row in rows)
    positive = all(row["positive"] for row in rows)
    squared = sum(row["squared_error_sum"] for row in rows)
    return {
        "pooled_rmse": math.sqrt(squared / float(pixels))
        if finite else math.inf,
        "mean_sample_rmse": sum(row["rmse"] for row in rows) / len(rows)
        if finite else math.inf,
        "valid": finite and positive,
        "finite": finite,
        "positive": positive,
        "sample_count": len(rows),
    }


def load_best_checkpoint_for_evaluation(controller, payload):
    controller.load_canonical_model_state_dict(payload["model_state"])
    controller.load_method_state_dict(payload["method_state"])


def _restore_training_checkpoint(path, prepared, optimizer, max_epochs):
    payload = torch.load(str(path), map_location="cpu")
    expected_assignment = assignment_payload(prepared.assignment)
    start_epoch = validate_training_checkpoint(
        payload, expected_assignment, max_epochs)
    prepared.controller.load_canonical_model_state_dict(
        payload["model_state"])
    prepared.controller.load_method_state_dict(payload["method_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    best_payload = torch.load(
        str(prepared.output / "best.pt"), map_location="cpu")
    validate_training_checkpoint(
        best_payload, expected_assignment, int(max_epochs) + 1)
    history = list(payload["history"])
    best_epoch = min(
        history, key=lambda row: float(row["validation_pooled_rmse"]))["epoch"]
    if int(best_payload["epoch"]) != int(best_epoch):
        raise ValueError("task-aware QAT best checkpoint differs from history")
    best_value = float(best_payload["history"][-1][
        "validation_pooled_rmse"])
    return start_epoch, history, best_value, best_payload


def _train(prepared, payload, resume=None):
    training = payload["training"]
    loader = DataLoader(
        Subset(prepared.trainset, prepared.calibration_indices),
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=int(training["workers"]),
        pin_memory=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(int(training["seed"])),
    )
    weights = ModelTaskLossWeights(
        depth=float(training["depth_loss_weight"]),
        boundary=float(training["boundary_loss_weight"]),
        teacher=float(training["teacher_loss_weight"]),
        initial_depth=float(training["initial_depth_loss_weight"]),
        propagation=float(training["propagation_loss_weight"]),
    )
    optimizer = torch.optim.SGD(
        prepared.controller.parameters(),
        lr=float(training["learning_rate"]),
        momentum=float(training["momentum"]),
        weight_decay=float(training["weight_decay"]),
    )
    history = []
    best = math.inf
    best_payload = None
    start_epoch = 1
    if resume is not None:
        start_epoch, history, best, best_payload = \
            _restore_training_checkpoint(
                resume, prepared, optimizer, int(training["epochs"]))
    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        set_model_qat_train_mode(prepared.student_runtime.model_name,
                                 prepared.model)
        prepared.controller.activation_modules.train()
        loss_sum = 0.0
        steps = 0
        for step, sample in enumerate(loader, 1):
            model_input, target = prepared.student_runtime.model_input(
                sample, prepared.student_runtime.device)
            optimizer.zero_grad(set_to_none=True)
            prediction, loss = _task_forward(
                prepared, model_input, target, weights,
                float(training["boundary_threshold_m"]),
            )
            loss.total.backward()
            gradient_norm = prepared.controller.assert_finite_gradients()
            if gradient_norm > float(training["max_gradient_norm"]):
                torch.nn.utils.clip_grad_norm_(
                    tuple(prepared.controller.parameters()),
                    float(training["max_gradient_norm"]))
            optimizer.step()
            loss_sum += float(loss.total.detach().item())
            steps += 1
            if step % int(training["log_interval"]) == 0:
                print("epoch=%d step=%d/%d loss=%.6f" % (
                    epoch, step, len(loader), loss_sum / float(steps)),
                    flush=True)
        validation = _evaluate(prepared, prepared.evaluation_indices)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / float(steps),
            "validation_pooled_rmse": validation["pooled_rmse"],
            "validation_mean_sample_rmse": validation["mean_sample_rmse"],
            "validation_valid": validation["valid"],
            "validation_finite": validation["finite"],
            "validation_positive": validation["positive"],
            "gradient_norm": gradient_norm,
        }
        history.append(row)
        print("epoch=%d train_loss=%.6f val_rmse=%s valid=%s" % (
            epoch, row["train_loss"], row["validation_pooled_rmse"],
            row["validation_valid"]), flush=True)
        payload_state = {
            "epoch": epoch,
            "model_state": prepared.controller.canonical_model_state_dict(),
            "method_state": prepared.controller.method_state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": list(history),
            "assignment": {
                "model_name": prepared.assignment.model_name,
                "weight_bits": list(prepared.assignment.weight_bits),
                "activation_bits": [
                    [list(owner), bits]
                    for owner, bits in prepared.assignment.activation_bits],
            },
            "propagation_dtype": "fp16",
        }
        torch.save(payload_state, prepared.output / "last.pt")
        if should_replace_best(
                best_payload, best, float(validation["pooled_rmse"])):
            best = float(validation["pooled_rmse"])
            best_payload = payload_state
            torch.save(best_payload, prepared.output / "best.pt")
    if best_payload is None:
        raise RuntimeError("task-aware QAT completed without a best checkpoint")
    torch.save(best_payload, prepared.output / "final.pt")
    return history, best_payload


def _write_summary(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = tuple(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_model(payload, model_name: str, configuration: str, resume=None):
    output = Path(payload["output_root"]) / model_name / configuration.lower()
    if resume is None and output.exists():
        raise FileExistsError("task-aware QAT output already exists: %s" % output)
    if resume is None:
        output.mkdir(parents=True)
    elif Path(resume).resolve() != (output / "last.pt").resolve():
        raise ValueError("QAT resume checkpoint path differs from output")
    prepared = _prepare(payload, model_name, configuration)
    prepared.output = output
    try:
        history, best_payload = _train(prepared, payload, resume)
        load_best_checkpoint_for_evaluation(prepared.controller, best_payload)
        evaluation = _evaluate(prepared, prepared.evaluation_indices)
        source_summary = Path(payload["source_output_root"]) / model_name / \
            "summary.csv"
        source_manifest = json.loads(
            (Path(payload["source_output_root"]) / model_name / "manifest.json")
            .read_text(encoding="utf-8"))
        weighted_weight_bits, weighted_activation_bits = assignment_budget(
            source_manifest, configuration)
        with source_summary.open("r", newline="", encoding="utf-8") as handle:
            source_rows = tuple(csv.DictReader(handle))
        reference = tuple(row for row in source_rows
                          if row["configuration"] == "FP32_REFERENCE")
        if len(reference) != 1:
            raise ValueError("source FP32 reference row is not unique")
        reference_rmse = float(reference[0]["pooled_rmse"])
        result = {
            "model": model_name,
            "configuration": configuration + "_QAT",
            "pooled_rmse": evaluation["pooled_rmse"],
            "mean_sample_rmse": evaluation["mean_sample_rmse"],
            "delta_vs_fp32": evaluation["pooled_rmse"] - reference_rmse,
            "relative_fp_loss": evaluation["pooled_rmse"] / reference_rmse - 1.0,
            "valid": evaluation["valid"],
            "finite": evaluation["finite"],
            "positive": evaluation["positive"],
            "average_weight_bits": weighted_weight_bits,
            "average_activation_bits": weighted_activation_bits,
        }
        _write_summary(output / "summary.csv", (result,))
        manifest = {
            "model": model_name,
            "configuration": configuration,
            "method": "task_aware",
            "propagation_dtype": "fp16",
            "calibration_count": len(prepared.calibration_indices),
            "evaluation_count": len(prepared.evaluation_indices),
            "assignment": best_payload["assignment"],
            "assignment_counts": assignment_counts(prepared.assignment),
            "history": history,
            "result": result,
        }
        (output / "manifest.json").write_text(
            json.dumps(json_safe(manifest), indent=2, sort_keys=True,
                       allow_nan=False) +
            "\n", encoding="utf-8")
    finally:
        prepared.close()
    return output


def run_model(config: Path, model_name: str, configuration: str, resume=None):
    payload = load_qat_payload(config)
    if model_name not in MODEL_NAMES:
        raise ValueError("unknown QAT model: %s" % model_name)
    if configuration not in QAT_CONFIGURATIONS:
        raise ValueError("unknown QAT configuration: %s" % configuration)
    output_root = Path(payload["output_root"])
    if not output_root.is_dir():
        raise FileNotFoundError("QAT output root is missing: %s" % output_root)
    return _run_model(payload, model_name, configuration, resume)


def run(config: Path):
    payload = load_qat_payload(config)
    output_root = Path(payload["output_root"])
    if output_root.exists():
        raise FileExistsError("QAT output root already exists: %s" % output_root)
    output_root.mkdir(parents=True)
    for model_name in MODEL_NAMES:
        spec = payload["models"][model_name]
        environment = dict(os.environ)
        environment["SPN_EXTERNAL_ROOT"] = payload["external_root"]
        environment["COMPLETIONFORMER_ROOT"] = payload["completionformer_root"]
        environment["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + \
            os.environ["PYTHONPATH"]
        for configuration in QAT_CONFIGURATIONS:
            subprocess.run(
                (str(spec["python_executable"]), str(Path(__file__)),
                 "--config", str(Path(config)), "--model", model_name,
                 "--configuration", configuration),
                cwd=str(spec["data_root"]), env=environment, check=True)
    return output_root


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--configuration", choices=QAT_CONFIGURATIONS)
    parser.add_argument("--resume", type=Path)
    return parser


def main():
    args = build_parser().parse_args(tuple(sys.argv[1:]))
    if (args.model is None) != (args.configuration is None):
        raise ValueError("--model and --configuration must be provided together")
    if args.resume is not None and args.model is None:
        raise ValueError("--resume requires a selected model configuration")
    if args.model is None:
        print(run(args.config))
    else:
        print(run_model(
            args.config, args.model, args.configuration, args.resume))


if __name__ == "__main__":
    main()
