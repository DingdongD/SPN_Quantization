#!/usr/bin/env python3
"""Strict folded-graph AdaRound/BRECQ reconstruction for NYU checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train_nyu_iteration_sweep as sweep  # noqa: E402
from scripts.export_nyu_predictions import (  # noqa: E402
    build_model,
    load_run_args,
    prepare_args,
)
from scripts.hardware_aligned_quantization import (  # noqa: E402
    prepare_hardware_model,
)
from scripts.run_nyu_rtn_quantization import (  # noqa: E402
    batch_from_sample,
    calibration_dataset,
    evaluation_dataset,
    seeded_sample,
)
from spn_quant.adaptive_rounding import (  # noqa: E402
    AdaptiveRoundingConfig,
    is_supported_weight_module,
)
from spn_quant.deployment_contract import (  # noqa: E402
    build_deployment_contract,
    save_deployment_contract,
    validate_graph_preparation,
)
from spn_quant.strict_reconstruction import (  # noqa: E402
    StrictBlockReconstructor,
    StrictCalibrationRecord,
    StrictReconstructionConfig,
    detach_cpu,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=True),
        encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def module_at(model: nn.Module, name: str) -> nn.Module:
    try:
        return dict(model.named_modules())[name]
    except KeyError:
        raise KeyError(
            "unknown strict reconstruction target: %s" % name)


def supported_weight_count(module: nn.Module) -> int:
    count = int(is_supported_weight_module(module))
    count += sum(
        1 for name, child in module.named_modules()
        if name and is_supported_weight_module(child))
    return count


def list_targets(model: nn.Module) -> List[Dict[str, Any]]:
    rows = []
    for name, module in model.named_modules():
        if not name:
            continue
        count = supported_weight_count(module)
        if count:
            rows.append({
                "module": name,
                "type": type(module).__name__,
                "supported_weights": count,
                "adaround_strict": int(
                    is_supported_weight_module(module)),
                "brecq_strict": 1,
            })
    return rows


def select_targets(
        model: nn.Module, exact: Sequence[str],
        patterns: Sequence[str], method: str) -> List[str]:
    modules = dict(model.named_modules())
    selected = []
    for name in exact:
        if name not in modules:
            raise KeyError(
                "unknown strict reconstruction target: %s" % name)
        selected.append(name)
    compiled = [re.compile(pattern) for pattern in patterns]
    for name, module in model.named_modules():
        if (not name or
                not any(pattern.search(name) for pattern in compiled)):
            continue
        if (method == "adaround_strict" and
                not is_supported_weight_module(module)):
            continue
        if (method == "brecq_strict" and
                supported_weight_count(module) == 0):
            continue
        selected.append(name)
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError(
            "no strict reconstruction targets selected")
    for name in selected:
        module = modules[name]
        if (method == "adaround_strict" and
                not is_supported_weight_module(module)):
            raise TypeError(
                "strict AdaRound target must be one weight module: %s" %
                name)
        if (method == "brecq_strict" and
                supported_weight_count(module) == 0):
            raise TypeError(
                "strict BRECQ target has no supported weights: %s" %
                name)
    return selected


def validate_non_overlapping_targets(
        targets: Sequence[str]) -> None:
    for index, left in enumerate(targets):
        for right in targets[index + 1:]:
            if (left.startswith(right + ".") or
                    right.startswith(left + ".")):
                raise ValueError(
                    "overlapping strict targets: %s, %s" %
                    (left, right))


def _map_nested(value: Any, function):
    if torch.is_tensor(value):
        return function(value)
    if isinstance(value, Mapping):
        return type(value)((
            key, _map_nested(item, function))
            for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(
            _map_nested(item, function) for item in value)
    if isinstance(value, list):
        return [
            _map_nested(item, function) for item in value]
    return value


def _retain_grad(value: Any) -> None:
    def retain(tensor: torch.Tensor):
        if tensor.requires_grad:
            tensor.retain_grad()
        return tensor

    _map_nested(value, retain)


def _extract_grad(value: Any) -> Any:
    def extract(tensor: torch.Tensor):
        if tensor.grad is None:
            return torch.zeros_like(tensor)
        return tensor.grad.detach().clone()

    return _map_nested(value, extract)


class TargetCapture(object):
    def __init__(self, module: nn.Module) -> None:
        self.inputs = []
        self.outputs = []
        self.handles = [
            module.register_forward_pre_hook(self._pre),
            module.register_forward_hook(self._post),
        ]

    def _pre(
            self, module: nn.Module,
            inputs: Tuple[Any, ...]) -> None:
        del module
        self.inputs.append(inputs)

    def _post(
            self, module: nn.Module,
            inputs: Tuple[Any, ...], output: Any) -> None:
        del module, inputs
        self.outputs.append(output)
        _retain_grad(output)

    def reset(self) -> None:
        self.inputs = []
        self.outputs = []

    def require_one(
            self, label: str) -> Tuple[Tuple[Any, ...], Any]:
        if len(self.inputs) != 1 or len(self.outputs) != 1:
            raise RuntimeError(
                "%s target must execute exactly once; inputs=%d outputs=%d" %
                (label, len(self.inputs), len(self.outputs)))
        return self.inputs[0], self.outputs[0]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def capture_records(
        teacher: nn.Module, student: nn.Module,
        teacher_target: nn.Module, student_target: nn.Module,
        saved_args: Any, dataset: Any,
        indices: Sequence[int], device: torch.device,
        seed: int, loss_mode: str,
        asymmetric: bool) -> List[StrictCalibrationRecord]:
    teacher_capture = TargetCapture(teacher_target)
    student_capture = TargetCapture(student_target)
    records = []
    try:
        for rank, index in enumerate(indices, 1):
            sample = seeded_sample(dataset, index, seed)
            batch = batch_from_sample(sample)
            model_args, gt = sweep.batch_to_model_input(
                saved_args.model, batch, device)

            teacher_capture.reset()
            teacher.zero_grad(set_to_none=True)
            if loss_mode == "mse":
                with torch.no_grad():
                    teacher(*model_args)
            else:
                output = teacher(*model_args)
                prediction = sweep.extract_pred(output)
                loss = sweep.compute_loss(
                    saved_args, prediction, gt)
                loss.backward()
            teacher_inputs, teacher_output = (
                teacher_capture.require_one("teacher"))
            gradients = (
                _extract_grad(teacher_output)
                if loss_mode != "mse" else None)

            if asymmetric:
                student_capture.reset()
                with torch.no_grad():
                    student(*model_args)
                student_inputs, _ = (
                    student_capture.require_one("student"))
                reconstruction_inputs = student_inputs
            else:
                reconstruction_inputs = teacher_inputs

            records.append(StrictCalibrationRecord(
                inputs=detach_cpu(reconstruction_inputs),
                reference=detach_cpu(teacher_output),
                gradients=(
                    detach_cpu(gradients)
                    if gradients is not None else None),
            ))
            teacher.zero_grad(set_to_none=True)
            print(
                "strict capture %d/%d index=%05d" %
                (rank, len(indices), index),
                flush=True)
    finally:
        teacher_capture.close()
        student_capture.close()
    return records


def evaluate_model(
        model: nn.Module, saved_args: Any,
        dataset: Any, indices: Sequence[int],
        device: torch.device, seed: int
        ) -> List[Dict[str, Any]]:
    rows = []
    model.eval()
    with torch.no_grad():
        for index in indices:
            sample = seeded_sample(dataset, index, seed)
            batch = batch_from_sample(sample)
            model_args, gt = sweep.batch_to_model_input(
                saved_args.model, batch, device)
            prediction = sweep.extract_pred(
                model(*model_args))
            metric = sweep.evaluate_error(
                gt_depth=gt, pred_depth=prediction)
            rows.append({
                "sample_index": int(index),
                "RMSE": float(metric["RMSE"]),
                "MAE": float(metric["MAE"]),
                "ABS_REL": float(metric["ABS_REL"]),
                "finite": int(
                    torch.isfinite(prediction).all().item()),
            })
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument(
        "--method",
        choices=("adaround_strict", "brecq_strict"),
        required=True)
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument(
        "--target-regex", action="append", default=[])
    parser.add_argument("--list-targets", action="store_true")
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument(
        "--weight-clip-ratio", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--round-loss-weight", type=float, default=1e-2)
    parser.add_argument(
        "--warmup-fraction", type=float, default=0.2)
    parser.add_argument("--beta-start", type=float, default=20.0)
    parser.add_argument("--beta-end", type=float, default=2.0)
    parser.add_argument(
        "--loss",
        choices=("mse", "fisher_diag", "fisher_full"),
        default="mse")
    parser.add_argument("--asymmetric", action="store_true")
    parser.add_argument(
        "--calibration-samples", type=int, default=1024)
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument(
        "--fold-max-error", type=float, default=0.05)
    parser.add_argument(
        "--skip-conv-bn-fold", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--out-dir",
        default="profile_logs/nyu_strict_reconstruction")
    parser.add_argument(
        "--save-folded-checkpoint",
        default="reconstructed_folded.pt")
    parser.add_argument(
        "--contract-name",
        default="strict_deployment_contract.pt")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = run_dir / checkpoint
    saved_args = prepare_args(
        load_run_args(run_dir), args)
    if (saved_args.device.startswith("cuda") and
            not torch.cuda.is_available()):
        raise RuntimeError(
            "CUDA is required for %s" % saved_args.model)
    device = torch.device(saved_args.device)

    student, meta = build_model(
        saved_args, checkpoint, device)
    teacher, _ = build_model(
        saved_args, checkpoint, device)
    student.eval()
    teacher.eval()
    dataset = calibration_dataset(saved_args)
    calibration_count = min(
        int(args.calibration_samples), len(dataset))
    indices = np.random.RandomState(args.seed).choice(
        len(dataset), calibration_count,
        replace=False).tolist()
    preparation_sample = seeded_sample(
        dataset, indices[0], args.seed)
    preparation_batch = batch_from_sample(
        preparation_sample)
    preparation_args, _ = sweep.batch_to_model_input(
        saved_args.model, preparation_batch, device)
    excluded_pairs = (
        [("conv1_1", "bn1")]
        if saved_args.model == "cspn" else [])
    fold = not args.skip_conv_bn_fold
    teacher_preparation = prepare_hardware_model(
        teacher, preparation_args,
        excluded_pairs=excluded_pairs,
        fold=fold)
    student_preparation = prepare_hardware_model(
        student, preparation_args,
        excluded_pairs=excluded_pairs,
        fold=fold)
    validate_graph_preparation(
        student_preparation, {
            "fold": int(fold),
            "folded_pairs": teacher_preparation[
                "folded_pairs"],
            "unfolded_fanout_pairs": teacher_preparation[
                "unfolded_fanout_pairs"],
            "unfolded_conv_bn_pairs": teacher_preparation[
                "unfolded_conv_bn_pairs"],
        })
    if max(
            teacher_preparation["max_abs_error"],
            student_preparation["max_abs_error"]
            ) > args.fold_max_error:
        raise RuntimeError(
            "Conv-BN folding exceeded strict tolerance")

    if args.list_targets:
        for row in list_targets(student):
            print(
                "{module:70s} {type:24s} "
                "weights={supported_weights} "
                "adaround_strict={adaround_strict} "
                "brecq_strict={brecq_strict}".format(**row))
        return

    targets = select_targets(
        student, args.target,
        args.target_regex, args.method)
    validate_non_overlapping_targets(targets)
    out_dir = (
        Path(args.out_dir) /
        saved_args.model /
        args.method)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    weight_rows = []
    histories = {}
    contracts = {}

    for target_index, target_name in enumerate(targets, 1):
        records = capture_records(
            teacher, student,
            module_at(teacher, target_name),
            module_at(student, target_name),
            saved_args, dataset, indices,
            device, args.seed,
            args.loss, args.asymmetric)
        reconstructor = StrictBlockReconstructor(
            module_at(student, target_name),
            AdaptiveRoundingConfig(
                bits=args.w_bits,
                clip_ratio=args.weight_clip_ratio),
            StrictReconstructionConfig(
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                round_loss_weight=args.round_loss_weight,
                warmup_fraction=args.warmup_fraction,
                beta_start=args.beta_start,
                beta_end=args.beta_end,
                loss=args.loss,
                seed=args.seed + target_index),
            contract_prefix=target_name)
        result = reconstructor.fit(records)
        summary = result.manifest()
        summary.update({
            "target": target_name,
            "method": args.method,
            "model": saved_args.model,
            "asymmetric": int(args.asymmetric),
        })
        summary_rows.append(summary)
        for row in result.weight_manifest:
            row = dict(row)
            local = row.get("module", "")
            row["target"] = target_name
            row["module"] = (
                target_name if not local
                else "%s.%s" % (target_name, local))
            weight_rows.append(row)
        overlap = (
            set(contracts) &
            set(result.weight_contracts))
        if overlap:
            raise RuntimeError(
                "duplicate strict weight contracts: %s" %
                sorted(overlap))
        contracts.update(result.weight_contracts)
        histories[target_name] = result.history
        print(
            "strict reconstructed %d/%d "
            "target=%s before=%.8f after=%.8f" %
            (target_index, len(targets), target_name,
             result.before_loss, result.after_loss),
            flush=True)

    graph_contract = {
        "fold": int(fold),
        "excluded_pairs": [
            list(pair) for pair in excluded_pairs],
        "folded_pairs": student_preparation[
            "folded_pairs"],
        "unfolded_fanout_pairs": student_preparation[
            "unfolded_fanout_pairs"],
        "unfolded_conv_bn_pairs": student_preparation[
            "unfolded_conv_bn_pairs"],
    }
    contract_payload = build_deployment_contract(
        source_checkpoint=checkpoint,
        graph_contract=graph_contract,
        weight_contracts=contracts,
        method=args.method,
        targets=targets,
        metadata={
            "model": saved_args.model,
            "architecture": meta,
            "asymmetric": int(args.asymmetric),
            "loss": args.loss,
            "calibration_indices": indices,
        })
    contract_path = save_deployment_contract(
        out_dir / args.contract_name,
        contract_payload)
    folded_checkpoint = (
        out_dir /
        args.save_folded_checkpoint)
    torch.save({
        "net": student.state_dict(),
        "debug_only_folded_graph": True,
        "source_checkpoint": str(checkpoint),
        "deployment_contract": str(contract_path),
    }, folded_checkpoint)
    write_csv(
        out_dir / "strict_reconstruction_summary.csv",
        summary_rows)
    write_csv(
        out_dir / "strict_weight_rounding_manifest.csv",
        weight_rows)
    write_json(
        out_dir / "strict_reconstruction_history.json",
        histories)
    manifest = {
        "format_version": 1,
        "strict": 1,
        "method": args.method,
        "model": saved_args.model,
        "source_checkpoint": str(checkpoint.resolve()),
        "deployment_checkpoint": str(checkpoint.resolve()),
        "deployment_contract": str(contract_path.resolve()),
        "debug_folded_checkpoint": str(
            folded_checkpoint.resolve()),
        "targets": targets,
        "weight_bits": args.w_bits,
        "activation_bits": 0,
        "activation_manifest": [],
        "graph_contract": graph_contract,
        "calibration_indices": indices,
        "asymmetric": int(args.asymmetric),
        "loss": args.loss,
    }
    write_json(
        out_dir / "strict_reconstruction_manifest.json",
        manifest)

    if int(args.eval_samples) > 0:
        valset = evaluation_dataset(saved_args)
        eval_indices = list(range(min(
            int(args.eval_samples), len(valset))))
        rows = evaluate_model(
            student, saved_args, valset,
            eval_indices, device, args.seed)
        write_csv(
            out_dir / "strict_evaluation_metrics.csv",
            rows)
        print(
            "strict folded evaluation finite=%s "
            "mean_RMSE=%.6f" %
            (all(row["finite"] for row in rows),
             sum(row["RMSE"] for row in rows) /
             max(len(rows), 1)),
            flush=True)
    print(
        "strict deployment contract: %s" % contract_path,
        flush=True)
    print(
        "evaluate with the original checkpoint plus "
        "strict_reconstruction_manifest.json",
        flush=True)


if __name__ == "__main__":
    main()
