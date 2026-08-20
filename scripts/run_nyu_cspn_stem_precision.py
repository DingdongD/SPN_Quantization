#!/usr/bin/env python3
"""Evaluate CSPN stem precision contracts on the fixed NYU protocol."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts import run_nyu_cspn_activation_resolution as base
from scripts.export_nyu_predictions import load_run_args, prepare_args
from scripts.hardware_aligned_quantization import (
    HardwareAlignedInstrumentor,
    prepare_hardware_model,
)
from scripts.run_nyu_rtn_quantization import (
    aggregate_region_rows,
    calibration_dataset,
    evaluation_dataset,
    prediction_payload,
    prepare_prediction_dir,
    seeded_sample,
    write_csv,
    write_json,
    write_prediction_payload,
)
from spn_quant.adapters import install_model_semantic_adapter
from spn_quant.cspn_stem import CSPNStemController
from spn_quant.propagation import install_propagation_adapter
from spn_quant.activation_boundaries import CSPNActivationBoundaryController


CALIBRATION_SAMPLES = 128
EVALUATION_SAMPLES = 64


@dataclass(frozen=True)
class StemConfiguration:
    name: str
    promoted_owners: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class StemOwnership:
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]


@dataclass(frozen=True)
class IndexProtocol:
    calibration_indices: Tuple[int, ...]
    evaluation_indices: Tuple[int, ...]
    calibration_identities: Tuple[Tuple[str, int], ...]
    evaluation_identities: Tuple[Tuple[str, int], ...]
    selection: str
    seed: int


class ConvOperationCounter(object):
    def __init__(self, model: nn.Module,
                 module_names: Sequence[str]) -> None:
        names = tuple(str(name) for name in module_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("operation counter module names must be unique")
        modules = dict(model.named_modules())
        unknown = set(names) - set(modules)
        if unknown:
            raise ValueError("operation counter modules are missing: %s" %
                             sorted(unknown))
        self.module_names = names
        self.records = {}
        self.handles = []
        for name in names:
            module = modules[name]
            if not isinstance(module, nn.Conv2d):
                raise TypeError("operation counter requires Conv2d: %s" % name)
            self.handles.append(module.register_forward_hook(
                self._hook(name)))

    def _hook(self, name: str):
        def hook(module: nn.Conv2d, inputs, output):
            if not inputs or not torch.is_tensor(inputs[0]) or \
                    not torch.is_tensor(output):
                raise TypeError("Conv operation counter requires tensor IO")
            tensor = inputs[0]
            shape = (tuple(tensor.shape), tuple(output.shape))
            if name in self.records:
                if self.records[name]["shape"] != shape:
                    raise ValueError(
                        "Conv operation shape changed: %s" % name)
                return
            reduction = int(module.in_channels // module.groups) * \
                int(module.kernel_size[0]) * int(module.kernel_size[1])
            self.records[name] = {
                "module": name,
                "shape": shape,
                "macs": int(output.numel()) * reduction,
                "weight_elements": int(module.weight.numel()),
                "input_elements": int(tensor.numel()),
            }
        return hook

    def rows(self) -> list[Dict[str, object]]:
        missing = set(self.module_names) - set(self.records)
        if missing:
            raise RuntimeError(
                "operation counter modules were not executed: %s" %
                sorted(missing))
        return [{
            "module": self.records[name]["module"],
            "macs": self.records[name]["macs"],
            "weight_elements": self.records[name]["weight_elements"],
            "input_elements": self.records[name]["input_elements"],
        } for name in self.module_names]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


class TensorCapture(object):
    def __init__(self, module: nn.Module, source: str,
                 argument_index: int = None) -> None:
        if source not in ("output", "argument"):
            raise ValueError("tensor capture source is invalid")
        if source == "argument" and argument_index is None:
            raise ValueError("argument tensor capture requires an index")
        self.source = source
        self.argument_index = argument_index
        self.value = None
        self.handle = module.register_forward_hook(self._hook)

    @classmethod
    def output(cls, module: nn.Module):
        return cls(module, "output")

    @classmethod
    def argument(cls, module: nn.Module, argument_index: int):
        return cls(module, "argument", int(argument_index))

    def _hook(self, module, inputs, output):
        del module
        value = output if self.source == "output" else \
            inputs[self.argument_index]
        if not torch.is_tensor(value):
            raise TypeError("tensor capture source must be a tensor")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("tensor capture source must be finite")
        self.value = value.detach().clone()

    def take(self) -> torch.Tensor:
        if self.value is None:
            raise RuntimeError("tensor capture has no new value")
        value = self.value
        self.value = None
        return value

    def close(self) -> None:
        self.handle.remove()


class PairErrorAccumulator(object):
    def __init__(self, signal: str) -> None:
        self.signal = str(signal)
        self.updates = 0
        self.elements = 0
        self.signal_energy = 0.0
        self.error_energy = 0.0

    def update(self, reference: torch.Tensor,
               quantized: torch.Tensor) -> None:
        if reference.shape != quantized.shape:
            raise ValueError("paired tensors must have equal shapes")
        if not bool(torch.isfinite(reference).all().item()) or \
                not bool(torch.isfinite(quantized).all().item()):
            raise ValueError("paired tensors must be finite")
        reference64 = reference.to(torch.float64)
        error64 = reference64 - quantized.to(torch.float64)
        self.updates += 1
        self.elements += int(reference.numel())
        self.signal_energy += float((reference64 ** 2).sum().item())
        self.error_energy += float((error64 ** 2).sum().item())

    def row(self, config: str) -> Dict[str, object]:
        if self.updates == 0 or self.elements == 0:
            raise RuntimeError("paired error accumulator has no observations")
        if self.error_energy == 0.0:
            sqnr = 300.0
        elif self.signal_energy == 0.0:
            sqnr = -300.0
        else:
            sqnr = 10.0 * np.log10(
                self.signal_energy / self.error_energy)
        return {
            "config": str(config),
            "signal": self.signal,
            "updates": self.updates,
            "elements": self.elements,
            "mse": self.error_energy / float(self.elements),
            "sqnr_db": float(sqnr),
        }


def build_configurations() -> Tuple[StemConfiguration, ...]:
    return (
        StemConfiguration("STRICT_W4A4", ()),
        StemConfiguration("STEM_W8A8", (
            ("relu#0", "relu_output"),
            ("boundary_controller.layer4_signed_skip", "boundary"),
        )),
        StemConfiguration("STEM_FP16", ()),
        StemConfiguration("STEM_BRANCH_A4", ()),
    )


def stem_ownership() -> StemOwnership:
    inputs = set(base.strict_owned_inputs())
    inputs.add("conv1_1")
    outputs = set(base.strict_owned_outputs())
    return StemOwnership(
        inputs=tuple(sorted(inputs)),
        outputs=tuple(sorted(outputs)),
    )


def hardware_configuration(config: StemConfiguration
                           ) -> Dict[str, object]:
    return base._configuration(
        config.name,
        base.ORDINARY_GROUPS,
        base.ORDINARY_GROUPS,
        base.PROPAGATION_A8_Q13,
        granularity="hybrid_group",
        group_size=8,
        promoted_owners=config.promoted_owners,
    )


def _indices(values: Iterable[object], expected: int,
             name: str) -> Tuple[int, ...]:
    output = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("%s indices must be integers" % name)
        if value < 0:
            raise ValueError("%s indices must be nonnegative" % name)
        output.append(int(value))
    if len(output) != expected:
        raise ValueError("%s requires exactly %d indices" % (name, expected))
    if len(set(output)) != len(output):
        raise ValueError("%s indices must be unique" % name)
    return tuple(output)


def index_protocol(calibration: Dict[str, object],
                   evaluation: Dict[str, object]) -> IndexProtocol:
    calibration_indices = _indices(
        calibration["indices"], CALIBRATION_SAMPLES, "calibration")
    evaluation_indices = _indices(
        evaluation["evaluation_indices"],
        EVALUATION_SAMPLES, "evaluation")
    if int(calibration["count"]) != len(calibration_indices):
        raise ValueError("calibration declared count does not match indices")
    if int(evaluation["evaluation_samples"]) != len(evaluation_indices):
        raise ValueError("evaluation declared count does not match indices")
    selection = str(calibration["selection"])
    if not selection:
        raise ValueError("calibration selection must be nonempty")
    seed = int(evaluation["seed"])
    calibration_identities = tuple(
        ("train", index) for index in calibration_indices)
    evaluation_identities = tuple(
        ("validation", index) for index in evaluation_indices)
    if set(calibration_identities) & set(evaluation_identities):
        raise ValueError("calibration and evaluation identities overlap")
    return IndexProtocol(
        calibration_indices=calibration_indices,
        evaluation_indices=evaluation_indices,
        calibration_identities=calibration_identities,
        evaluation_identities=evaluation_identities,
        selection=selection,
        seed=seed,
    )


def load_index_protocol(calibration_path: Path,
                        evaluation_path: Path) -> IndexProtocol:
    calibration = json.loads(Path(calibration_path).read_text())
    evaluation = json.loads(Path(evaluation_path).read_text())
    return index_protocol(calibration, evaluation)


def _sample_rmse(rows: Sequence[Dict[str, object]],
                 name: str) -> Dict[int, float]:
    if len(rows) != EVALUATION_SAMPLES:
        raise ValueError("%s requires exactly 64 sample rows" % name)
    output = {}
    for row in rows:
        index = int(row["sample_index"])
        if index in output:
            raise ValueError("%s sample indices must be unique" % name)
        value = float(row["RMSE"])
        if not np.isfinite(value):
            raise ValueError("%s RMSE must be finite" % name)
        output[index] = value
    return output


def acceptance(baseline_rows: Sequence[Dict[str, object]],
               candidate_rows: Sequence[Dict[str, object]]
               ) -> Dict[str, object]:
    baseline = _sample_rmse(baseline_rows, "baseline")
    candidate = _sample_rmse(candidate_rows, "candidate")
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate sample identities differ")
    baseline_rmse = float(np.mean(np.asarray(
        [baseline[index] for index in sorted(baseline)], dtype=np.float64)))
    candidate_rmse = float(np.mean(np.asarray(
        [candidate[index] for index in sorted(candidate)], dtype=np.float64)))
    wins = sum(candidate[index] < baseline[index] for index in baseline)
    return {
        "baseline_rmse": baseline_rmse,
        "candidate_rmse": candidate_rmse,
        "rmse_delta": candidate_rmse - baseline_rmse,
        "wins": int(wins),
        "accepted": bool(candidate_rmse < baseline_rmse and wins >= 33),
    }


METRIC_FIELDS = (
    "RMSE", "MAE", "ABS_REL", "IRMSE",
    "flat_RMSE", "boundary_RMSE",
)


def aggregate_metrics(rows: Sequence[Dict[str, object]],
                      configurations: Sequence[StemConfiguration]
                      ) -> list[Dict[str, object]]:
    output = []
    for config in configurations:
        selected = [row for row in rows if row["config"] == config.name]
        identities = _sample_rmse(selected, config.name)
        if len(identities) != EVALUATION_SAMPLES:
            raise ValueError("%s requires exactly 64 samples" % config.name)
        aggregate = {
            "config": config.name,
            "samples": len(selected),
        }
        for field in METRIC_FIELDS:
            values = np.asarray(
                [float(row[field]) for row in selected], dtype=np.float64)
            if not bool(np.isfinite(values).all()):
                raise ValueError("%s %s metrics must be finite" %
                                 (config.name, field))
            aggregate[field] = float(values.mean())
        output.append(aggregate)
    return output


def partition_stem_statistics(rows: Sequence[Dict[str, object]]
                              ) -> Tuple[list[Dict[str, object]],
                                         list[Dict[str, object]]]:
    by_signal = {}
    for row in rows:
        signal = str(row["signal"])
        if signal in by_signal:
            raise ValueError("duplicate stem statistic signal: %s" % signal)
        by_signal[signal] = row
    activation_names = ("rgb_input", "depth_input")
    partial_names = ("rgb_partial", "depth_partial", "stem_output")
    expected = set(activation_names + partial_names)
    if set(by_signal) != expected:
        raise ValueError("stem statistic signals do not match the contract")
    return (
        [by_signal[name] for name in activation_names],
        [by_signal[name] for name in partial_names],
    )


def precision_coverage(operation_rows: Sequence[Dict[str, object]],
                       config: StemConfiguration
                       ) -> list[Dict[str, object]]:
    if not operation_rows:
        raise ValueError("precision coverage requires operation rows")
    modules = [str(row["module"]) for row in operation_rows]
    if len(modules) != len(set(modules)):
        raise ValueError("precision coverage modules must be unique")
    if "conv1_1" not in modules:
        raise ValueError("precision coverage requires conv1_1")
    totals = {
        "macs": 0,
        "weight_elements": 0,
        "input_elements": 0,
    }
    grouped = {}
    for source in operation_rows:
        module = str(source["module"])
        values = {
            "macs": int(source["macs"]),
            "weight_elements": int(source["weight_elements"]),
            "input_elements": int(source["input_elements"]),
        }
        if any(value <= 0 for value in values.values()):
            raise ValueError("precision coverage counts must be positive")
        if module == "conv1_1" and config.name == "STEM_W8A8":
            format_name = "W8A8"
        elif module == "conv1_1" and config.name == "STEM_FP16":
            format_name = "FP16"
        else:
            format_name = "W4A4"
        if format_name not in grouped:
            grouped[format_name] = dict(
                (name, 0) for name in totals)
        for name in totals:
            totals[name] += values[name]
            grouped[format_name][name] += values[name]
    order = ("W4A4", "W8A8", "FP16")
    output = []
    for format_name in order:
        if format_name not in grouped:
            continue
        values = grouped[format_name]
        output.append({
            "config": config.name,
            "format": format_name,
            "macs": values["macs"],
            "mac_fraction": values["macs"] / float(totals["macs"]),
            "weight_elements": values["weight_elements"],
            "weight_element_fraction": values["weight_elements"] /
            float(totals["weight_elements"]),
            "input_elements": values["input_elements"],
            "input_element_fraction": values["input_elements"] /
            float(totals["input_elements"]),
        })
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_source_metadata(args, calibration_metadata,
                              evaluation_metadata,
                              checkpoint_sha256: str) -> None:
    checkpoint = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    for name, metadata in (
            ("calibration", calibration_metadata),
            ("evaluation", evaluation_metadata)):
        if Path(metadata["checkpoint"]).resolve() != checkpoint:
            raise ValueError("%s checkpoint identity differs" % name)
        if Path(metadata["data_root"]).resolve() != data_root:
            raise ValueError("%s data root identity differs" % name)
    if calibration_metadata["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("calibration checkpoint SHA256 differs")


def configure_checkpoint_build(saved_args):
    saved_args.from_scratch = True
    return saved_args


def _saved_args(args):
    saved = prepare_args(load_run_args(Path(args.run_dir)), args)
    saved.data_root = args.data_root
    if saved.model != "cspn":
        raise ValueError("stem precision evaluation requires model=cspn")
    return configure_checkpoint_build(saved)


def _prepare_model(saved_args, checkpoint: Path, device: torch.device,
                   preparation_args, fold_max_error: float):
    model, architecture, load_report = base._load_cspn(
        saved_args, checkpoint, device)
    preparation = prepare_hardware_model(
        model, preparation_args,
        excluded_pairs=(("conv1_1", "bn1"),))
    if float(preparation["primary_max_abs_error"]) > fold_max_error:
        raise RuntimeError("Conv-BN fold exceeds declared error threshold")
    return model, architecture, load_report, preparation


def _build_quantization_context(model, preparation, seed: int):
    semantic = install_model_semantic_adapter(model, "cspn", strict=True)
    boundaries = semantic.activation_boundaries()
    semantic.close()
    ownership = stem_ownership()
    instrumentor = HardwareAlignedInstrumentor(
        model, base.cspn_quant_group,
        preparation["fused_relu_producers"],
        externally_owned_outputs=ownership.outputs,
        externally_owned_inputs=ownership.inputs)
    boundary_controller = CSPNActivationBoundaryController(model, boundaries)
    propagation = install_propagation_adapter("cspn", model)
    stem = CSPNStemController(model.conv1_1)
    return instrumentor, boundary_controller, propagation, stem


def _calibrate(model, saved_args, dataset, indices, device, seed,
               instrumentor, boundary_controller, propagation, stem,
               config_name: str) -> None:
    instrumentor.observe()
    boundary_controller.observe()
    propagation.observe()
    stem.observe()
    with torch.no_grad():
        for rank, index in enumerate(indices, 1):
            sample = seeded_sample(dataset, index, seed)
            model(*base._model_args(saved_args, sample, device))
            if rank % 16 == 0 or rank == len(indices):
                print("%s calibration %d/%d" %
                      (config_name, rank, len(indices)), flush=True)
    instrumentor.freeze()
    boundary_controller.freeze()
    propagation.freeze()
    stem.freeze()


def _validate_site_contract(instrumentor, boundary_controller) -> None:
    owners = set(base.activation_owner(key) for key in
                 instrumentor.activation_site_keys(base.ORDINARY_GROUPS))
    expected = set(base.STRICT_ACTIVATION_OWNERS)
    expected.remove(("conv1_1", "input"))
    if owners != expected:
        raise RuntimeError(
            "stem-owned activation contract changed: missing=%s extra=%s" %
            (sorted(expected - owners), sorted(owners - expected)))
    expected_boundary = {
        "decoder_entry": 512,
        "layer4_signed_skip": 64,
    }
    if boundary_controller.channels != expected_boundary:
        raise RuntimeError("CSPN boundary_controller boundary contract changed")


def _configure_context(config: StemConfiguration, instrumentor, boundary_controller,
                       propagation, stem):
    hardware = hardware_configuration(config)
    specs, boundary_specs, active_merge = base._configure_quantized(
        hardware, instrumentor, boundary_controller, propagation, {})
    if active_merge is not None:
        raise RuntimeError("stem precision evaluation forbids merge adapters")
    stem.configure(config.name)
    if "conv1_1" in instrumentor.weight_bits_by_module():
        raise RuntimeError("conv1_1 weight has more than one quantization owner")
    if ("conv1_1", "input") in specs:
        raise RuntimeError("conv1_1 input has more than one quantization owner")
    return specs, boundary_specs


def _model_modules(model: nn.Module) -> Dict[str, nn.Module]:
    modules = dict(model.named_modules())
    for name in ("relu", "gud_up_proj_layer4"):
        if name not in modules:
            raise ValueError("official CSPN module is missing: %s" % name)
    return modules


def executed_operation_modules(instrumentor) -> Tuple[str, ...]:
    return tuple(sorted(
        name for name in instrumentor.modules
        if instrumentor.groups[name] in base.ORDINARY_GROUPS
        and instrumentor.observers[(name, "input")].observed))


def _evaluate_configuration(
        reference_model, quantized_model, saved_args, dataset,
        indices, device, seed, config, instrumentor, boundary_controller,
        propagation, stem, prediction_root):
    reference_capture = base.ModuleOutputCapture(
        reference_model, base.CSPN_BLOCK_SITES)
    quantized_capture = base.ModuleOutputCapture(
        quantized_model, base.CSPN_BLOCK_SITES)
    reference_modules = _model_modules(reference_model)
    quantized_modules = _model_modules(quantized_model)
    reference_relu = TensorCapture.output(reference_modules["relu"])
    quantized_relu = TensorCapture.output(quantized_modules["relu"])
    reference_skip = TensorCapture.argument(
        reference_modules["gud_up_proj_layer4"], 1)
    quantized_skip = TensorCapture.argument(
        quantized_modules["gud_up_proj_layer4"], 1)
    relu_error = PairErrorAccumulator("stem_relu_output")
    skip_error = PairErrorAccumulator("skip4_input")
    operation_modules = executed_operation_modules(instrumentor)
    operation_counter = ConvOperationCounter(
        quantized_model, operation_modules)
    prediction_dir = prepare_prediction_dir(
        prediction_root, config.name)
    block_error = base.BlockErrorAccumulator(base.CSPN_BLOCK_SITES)
    sample_rows = []
    region_rows = []
    propagation_rows = []

    with torch.no_grad():
        for rank, index in enumerate(indices, 1):
            sample = seeded_sample(dataset, index, seed)
            reference_prediction, reference_blocks = base._forward(
                reference_model, saved_args, sample, device,
                reference_capture)
            reference_relu_value = reference_relu.take()
            reference_skip_value = reference_skip.take()
            prediction, blocks = base._forward(
                quantized_model, saved_args, sample, device,
                quantized_capture)
            relu_error.update(
                reference_relu_value, quantized_relu.take())
            skip_error.update(
                reference_skip_value, quantized_skip.take())
            block_error.update(reference_blocks, blocks)
            pred = prediction.numpy()
            if not bool(np.isfinite(pred).all()):
                raise RuntimeError(
                    "non-finite prediction: config=%s sample=%d" %
                    (config.name, index))
            gt = sample["depth"][0].numpy()
            sparse = sample["rgbd"][3].numpy()
            metrics, regions = base.depth_sample_metrics(gt, pred, sparse)
            metrics.update({
                "model": "cspn",
                "config": config.name,
                "sample_index": int(index),
            })
            sample_rows.append(metrics)
            for source in regions:
                row = dict(source)
                row.update({
                    "model": "cspn",
                    "config": config.name,
                    "sample_index": int(index),
                })
                region_rows.append(row)
            payload = prediction_payload(
                gt, reference_prediction.numpy(), pred,
                int(index), "cspn", config.name, sparse=sparse)
            write_prediction_payload(prediction_dir, payload)
            for source in propagation.statistics():
                row = dict(source)
                row.update({
                    "model": "cspn",
                    "config": config.name,
                    "sample_index": int(index),
                })
                propagation_rows.append(row)
            if rank % 16 == 0 or rank == len(indices):
                print("%s evaluation %d/%d" %
                      (config.name, rank, len(indices)), flush=True)

    block_rows = []
    for source in block_error.rows():
        row = dict(source)
        row.update({
            "model": "cspn",
            "config": config.name,
        })
        block_rows.append(row)
    aggregate = block_error.aggregate()
    aggregate.update({
        "model": "cspn",
        "config": config.name,
        "block": "__all__",
    })
    block_rows.append(aggregate)
    stem_rows = stem.statistics()
    stem_rows.extend((
        relu_error.row(config.name),
        skip_error.row(config.name),
    ))
    activation_rows, partial_rows = partition_stem_statistics(
        stem_rows[:5])
    partial_rows.extend(stem_rows[5:])
    operation_rows = operation_counter.rows()
    coverage_rows = precision_coverage(operation_rows, config)
    layer_rows = instrumentor.statistics()
    for row in layer_rows:
        row.update({"model": "cspn", "config": config.name})

    reference_capture.close()
    quantized_capture.close()
    reference_relu.close()
    quantized_relu.close()
    reference_skip.close()
    quantized_skip.close()
    operation_counter.close()
    return {
        "sample_rows": sample_rows,
        "region_rows": region_rows,
        "propagation_rows": propagation_rows,
        "block_rows": block_rows,
        "activation_rows": activation_rows,
        "partial_rows": partial_rows,
        "operation_rows": operation_rows,
        "coverage_rows": coverage_rows,
        "layer_rows": layer_rows,
    }


def _artifact_hashes(root: Path) -> Dict[str, str]:
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json")
    return dict(
        (str(path.relative_to(root)), _sha256(path)) for path in paths)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--calibration-indices", required=True)
    parser.add_argument("--calibration-metadata", required=True)
    parser.add_argument("--evaluation-protocol", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold-max-error", type=float, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.device.startswith("cuda"):
        raise ValueError("CSPN stem evaluation requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.fold_max_error <= 0.0:
        raise ValueError("fold error threshold must be positive")
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False
    checkpoint = Path(args.checkpoint)
    calibration_indices_path = Path(args.calibration_indices)
    calibration_metadata_path = Path(args.calibration_metadata)
    evaluation_protocol_path = Path(args.evaluation_protocol)
    calibration_payload = json.loads(calibration_indices_path.read_text())
    calibration_metadata = json.loads(calibration_metadata_path.read_text())
    evaluation_metadata = json.loads(evaluation_protocol_path.read_text())
    protocol = index_protocol(calibration_payload, evaluation_metadata)
    if int(args.seed) != protocol.seed:
        raise ValueError("runner seed differs from evaluation protocol")
    checkpoint_sha256 = _sha256(checkpoint)
    _validate_source_metadata(
        args, calibration_metadata, evaluation_metadata,
        checkpoint_sha256)
    output = Path(args.out_dir)
    if output.exists():
        raise FileExistsError("output directory already exists: %s" % output)
    output.mkdir(parents=True)
    prediction_root = output
    saved_args = _saved_args(args)
    trainset = calibration_dataset(saved_args)
    valset = evaluation_dataset(saved_args)
    if max(protocol.calibration_indices) >= len(trainset):
        raise ValueError("calibration index exceeds the train split")
    if max(protocol.evaluation_indices) >= len(valset):
        raise ValueError("evaluation index exceeds the validation split")
    preparation_sample = seeded_sample(
        trainset, protocol.calibration_indices[0], args.seed)
    preparation_args = base._model_args(
        saved_args, preparation_sample, device)
    reference_model, architecture, reference_load, reference_preparation = \
        _prepare_model(
            saved_args, checkpoint, device, preparation_args,
            args.fold_max_error)

    configurations = build_configurations()
    sample_rows = []
    region_rows = []
    propagation_rows = []
    block_rows = []
    stem_activation_rows = []
    stem_partial_rows = []
    operation_rows = []
    coverage_rows = []
    layer_rows = []
    contracts = []
    load_reports = []
    site_counts = []

    for config in configurations:
        model, current_architecture, load_report, preparation = \
            _prepare_model(
                saved_args, checkpoint, device, preparation_args,
                args.fold_max_error)
        if current_architecture != architecture:
            raise RuntimeError("fresh CSPN architecture changed")
        if load_report != reference_load:
            raise RuntimeError("fresh CSPN checkpoint load changed")
        if preparation["folded_pairs"] != \
                reference_preparation["folded_pairs"]:
            raise RuntimeError("fresh CSPN fold manifest changed")
        instrumentor, boundary_controller, propagation, stem = \
            _build_quantization_context(model, preparation, args.seed)
        _calibrate(
            model, saved_args, trainset,
            protocol.calibration_indices, device, args.seed,
            instrumentor, boundary_controller, propagation, stem, config.name)
        _validate_site_contract(instrumentor, boundary_controller)
        specs, boundary_specs = _configure_context(
            config, instrumentor, boundary_controller, propagation, stem)
        result = _evaluate_configuration(
            reference_model, model, saved_args, valset,
            protocol.evaluation_indices, device, args.seed,
            config, instrumentor, boundary_controller, propagation,
            stem, prediction_root)
        sample_rows.extend(result["sample_rows"])
        region_rows.extend(result["region_rows"])
        propagation_rows.extend(result["propagation_rows"])
        block_rows.extend(result["block_rows"])
        stem_activation_rows.extend(result["activation_rows"])
        stem_partial_rows.extend(result["partial_rows"])
        for row in result["operation_rows"]:
            current = dict(row)
            current["config"] = config.name
            operation_rows.append(current)
        coverage_rows.extend(result["coverage_rows"])
        layer_rows.extend(result["layer_rows"])
        contract = stem.contract()
        contract["ordinary_activation_sites"] = len(specs)
        contract["boundary_activation_sites"] = len(boundary_specs)
        contracts.append(contract)
        load_reports.append({
            "config": config.name,
            "checkpoint_load": load_report,
        })
        site_counts.append({
            "config": config.name,
            "ordinary": len(specs),
            "boundary_controller": len(boundary_specs),
        })
        stem.close()
        propagation.close()
        boundary_controller.close()
        instrumentor.close()
        del model
        torch.cuda.empty_cache()

    aggregate_rows = aggregate_metrics(sample_rows, configurations)
    baseline_rows = [
        row for row in sample_rows if row["config"] == "STRICT_W4A4"
    ]
    acceptance_rows = []
    for config in configurations[1:]:
        candidate_rows = [
            row for row in sample_rows if row["config"] == config.name
        ]
        row = acceptance(baseline_rows, candidate_rows)
        row["config"] = config.name
        acceptance_rows.append(row)
    regional_rows = []
    for config in configurations:
        selected = [
            row for row in region_rows if row["config"] == config.name
        ]
        for source in aggregate_region_rows(selected):
            row = dict(source)
            row.update({"model": "cspn", "config": config.name})
            regional_rows.append(row)

    write_csv(
        output / "aggregate_metrics.csv", aggregate_rows,
        ("config", "samples") + METRIC_FIELDS)
    write_csv(
        output / "sample_metrics_64.csv", sample_rows,
        ("model", "config", "sample_index") + METRIC_FIELDS)
    write_csv(
        output / "regional_metrics.csv", regional_rows,
        ("model", "config", "region", "RMSE", "MAE", "ABS_REL"))
    write_csv(
        output / "stem_activation_metrics.csv", stem_activation_rows,
        ("config", "signal", "updates", "elements", "mse", "sqnr_db"))
    write_csv(
        output / "stem_partial_conv_metrics.csv", stem_partial_rows,
        ("config", "signal", "updates", "elements", "mse", "sqnr_db"))
    write_csv(
        output / "block_metrics.csv", block_rows,
        ("model", "config", "block", "block_output_mse",
         "block_output_sqnr"))
    write_csv(
        output / "propagation_metrics.csv", propagation_rows,
        ("model", "config", "sample_index", "signal", "iteration"))
    write_csv(
        output / "operation_counts.csv", operation_rows,
        ("config", "module", "macs", "weight_elements", "input_elements"))
    write_csv(
        output / "precision_coverage.csv", coverage_rows,
        ("config", "format", "macs", "mac_fraction",
         "weight_elements", "weight_element_fraction",
         "input_elements", "input_element_fraction"))
    write_csv(
        output / "acceptance.csv", acceptance_rows,
        ("config", "accepted", "wins", "baseline_rmse",
         "candidate_rmse", "rmse_delta"))
    write_csv(
        output / "layer_quantization_metrics.csv", layer_rows,
        ("model", "config", "module", "group", "kind"))
    manifest = {
        "model": "cspn",
        "architecture": architecture,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_load": reference_load,
        "fresh_load_reports": load_reports,
        "data_root": str(Path(args.data_root).resolve()),
        "run_dir": str(Path(args.run_dir).resolve()),
        "device": str(device),
        "seed": int(args.seed),
        "calibration_selection": protocol.selection,
        "calibration_indices": list(protocol.calibration_indices),
        "evaluation_indices": list(protocol.evaluation_indices),
        "calibration_indices_sha256": _sha256(calibration_indices_path),
        "calibration_metadata_sha256": _sha256(calibration_metadata_path),
        "evaluation_protocol_sha256": _sha256(evaluation_protocol_path),
        "configurations": contracts,
        "activation_site_counts": site_counts,
        "guidance": "fp32",
        "bias": "fp32",
        "propagation": dict(base.PROPAGATION_A8_Q13),
        "artifacts": {},
    }
    write_json(output / "manifest.json", manifest)
    manifest["artifacts"] = _artifact_hashes(output)
    write_json(output / "manifest.json", manifest)
    reference_model.cpu()
    del reference_model
    torch.cuda.empty_cache()
    print("CSPN stem precision evaluation complete: %s" % output, flush=True)


if __name__ == "__main__":
    main()
