#!/usr/bin/env python3
"""Generate exact static inputs for one official selected-model lane."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.nyu_model_runtime import NYUModelRuntime  # noqa: E402
from scripts.run_nyu_rtn_quantization import seeded_sample  # noqa: E402
from spn_quant.calibration_selection import (  # noqa: E402
    FeatureSchema,
    deterministic_weighted_kmedoids,
    fit_robust_normalizer,
    grouped_pairwise_distance,
    raw_descriptor,
    select_tail_cover,
)
from spn_quant.experiment_config import (  # noqa: E402
    MODEL_ORDER,
    load_selected_quantization_config,
)
from spn_quant.model_contracts import (  # noqa: E402
    build_model_quantization_contract,
)
from spn_quant.nyu_static_inputs import (  # noqa: E402
    ACTIVATION_DESCRIPTOR_FIELDS,
    CALIBRATION_COUNT,
    DATASET_IDENTITY,
    EVALUATION_COUNT,
    RAW_DESCRIPTOR_FIELDS,
    SELECTION_IDENTITY,
    StaticInputPaths,
    descriptor_schema_sha256,
    file_sha256,
    ordered_split_identity_sha256,
    validate_static_input_bundle,
)


RAW_DESCRIPTOR_GROUPS = (
    "depth", "depth", "depth", "depth", "depth",
    "rgb", "rgb", "rgb", "rgb",
    "diagnostic", "sparse", "sparse", "sparse", "sparse", "sparse",
    "sparse",
)
RAW_DESCRIPTOR_DIAGNOSTIC = tuple(
    name in ("depth_valid_ratio", "sparse_valid_count")
    for name in RAW_DESCRIPTOR_FIELDS)
WEIGHT_TYPES = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)


def _first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for key in value:
            tensor = _first_tensor(value[key])
            if tensor is not None:
                return tensor
        return None
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _tensor_statistics(tensors, channel_last: bool) -> Tuple[float, float, float]:
    values = tuple(tensor for tensor in tensors if torch.is_tensor(tensor))
    if not values:
        raise RuntimeError("activation descriptor capture is empty")
    percentiles = []
    maxima = []
    imbalances = []
    for tensor in values:
        observed = tensor.detach().float().abs()
        if observed.numel() <= 0 or not bool(torch.isfinite(observed).all().item()):
            raise ValueError("activation descriptor tensor must be finite")
        percentiles.append(float(torch.quantile(
            observed.reshape(-1), 0.99).item()))
        maxima.append(float(observed.max().item()))
        if observed.ndim <= 1:
            channel_maximum = observed.reshape(1, -1).max(dim=1).values
        else:
            channel_axis = observed.ndim - 1 if channel_last else 1
            moved = observed.movedim(channel_axis, 0).reshape(
                observed.shape[channel_axis], -1)
            channel_maximum = moved.max(dim=1).values
        denominator = float(channel_maximum.mean().item())
        if denominator <= 0.0:
            raise ValueError("activation descriptor channel scale is zero")
        imbalances.append(float(channel_maximum.max().item()) / denominator)
    return max(percentiles), max(maxima), max(imbalances)


class ActivationDescriptorCapture(object):
    def __init__(self, model_name, model, contract) -> None:
        self.model_name = str(model_name)
        self.model = model
        modules = dict(model.named_modules())
        self.values = {}
        self.handles = []
        stem = contract.prefix_groups[0][0]
        decoder = contract.tail_groups[-1][0]
        if stem not in modules or decoder not in modules:
            raise KeyError("descriptor block is missing from official model")
        self.handles.append(modules[stem].register_forward_hook(
            self._output_hook("encoder_stem")))
        self.handles.append(modules[decoder].register_forward_hook(
            self._output_hook("decoder_fusion")))
        propagation = self._propagation_module(modules)
        self.handles.append(propagation.register_forward_pre_hook(
            self._propagation_hook))
        attention = tuple(
            module for name, module in modules.items()
            if name.startswith("backbone.former") and
            name.endswith(".attn.proj") and isinstance(module, nn.Linear))
        if self.model_name == "completionformer":
            if len(attention) != 16:
                raise RuntimeError(
                    "CompletionFormer descriptor requires 16 projections")
            for module in attention:
                self.handles.append(module.register_forward_hook(
                    self._output_hook("attention_projection", append=True)))
        elif attention:
            raise RuntimeError("non-CompletionFormer attention capture changed")

    def _propagation_module(self, modules):
        if self.model_name in ("nlspn", "completionformer"):
            if "prop_layer" not in modules:
                raise KeyError("official propagation module is missing")
            return modules["prop_layer"]
        names = tuple(
            name for name in modules
            if name.startswith("dyspn_") and "." not in name)
        if len(names) != 1:
            raise RuntimeError("DySPN propagation module is not unique")
        return modules[names[0]]

    def _output_hook(self, group, append=False):
        def hook(module, inputs, output):
            del module, inputs
            tensor = _first_tensor(output)
            if tensor is None:
                raise RuntimeError("descriptor module output has no tensor")
            if append:
                self.values[group].append(tensor)
            else:
                if self.values[group]:
                    raise RuntimeError(
                        "descriptor module executed more than once: %s" % group)
                self.values[group].append(tensor)
        return hook

    def _propagation_hook(self, module, inputs):
        del module
        if len(inputs) < 2 or not torch.is_tensor(inputs[0]) or \
                not torch.is_tensor(inputs[1]):
            raise RuntimeError("propagation descriptor inputs changed")
        if self.values["initial_depth"] or self.values["signed_guidance"]:
            raise RuntimeError("propagation descriptor executed more than once")
        self.values["initial_depth"].append(inputs[0])
        self.values["signed_guidance"].append(inputs[1])

    def reset(self) -> None:
        groups = (
            "encoder_stem", "decoder_fusion", "initial_depth",
            "signed_guidance")
        if self.model_name == "completionformer":
            groups += ("attention_projection",)
        self.values = dict((group, []) for group in groups)

    def descriptor(self) -> dict:
        row = {}
        prefix = self.model_name
        for group in self.values:
            statistics = _tensor_statistics(
                self.values[group], group == "attention_projection")
            for name, value in zip(
                    ("p99", "maximum", "channel_imbalance"), statistics):
                row["%s_%s_%s" % (prefix, group, name)] = value
        if tuple(row) != ACTIVATION_DESCRIPTOR_FIELDS[self.model_name]:
            raise RuntimeError("activation descriptor schema changed")
        return row

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _module_macs(module, output) -> int:
    tensor = _first_tensor(output)
    if tensor is None:
        raise RuntimeError("weight-cost module output has no tensor")
    output_elements = int(tensor.numel())
    if isinstance(module, nn.Linear):
        return output_elements * int(module.in_features)
    kernel = int(module.kernel_size[0]) * int(module.kernel_size[1])
    operations = int(module.in_channels) // int(module.groups) * kernel
    return output_elements * operations


class CostCapture(object):
    def __init__(self, model, contract) -> None:
        self.model = model
        self.contract = contract
        self.modules = dict(model.named_modules())
        self.weight_macs = dict((name, 0) for name in contract.weight_modules)
        self.activation_elements = dict(
            (owner, 0) for block in contract.blocks
            for owner in block.activation_owners)
        self.handles = []
        for name in contract.weight_modules:
            module = self.modules[name]
            if not isinstance(module, WEIGHT_TYPES):
                raise TypeError("weight-cost owner is not Conv/Linear: %s" % name)
            self.handles.append(module.register_forward_hook(
                self._weight_hook(name)))
        for owner in self.activation_elements:
            site, role = owner
            family, module_name, local = site.split("::")
            if family == "activation" and local == "input":
                self.handles.append(
                    self.modules[module_name].register_forward_pre_hook(
                        self._activation_input_hook(owner)))
            elif family == "attention" and local == "q":
                self.handles.append(
                    self.modules[module_name + ".q"].register_forward_hook(
                        self._activation_output_hook(owner, 1)))
            elif family == "attention" and local in ("k", "v"):
                offset = 0 if local == "k" else 1
                self.handles.append(
                    self.modules[module_name + ".kv"].register_forward_hook(
                        self._kv_output_hook(owner, offset)))
            elif family == "concat" and local in (
                    "transformer_input", "cnn_input"):
                offset = 0 if local == "transformer_input" else 1
                self.handles.append(
                    self.modules[module_name].register_forward_pre_hook(
                        self._concat_input_hook(owner, offset)))
            else:
                raise ValueError("unsupported activation-cost site: %s" % site)

    def _weight_hook(self, name):
        def hook(module, inputs, output):
            del inputs
            self.weight_macs[name] += _module_macs(module, output)
        return hook

    def _activation_input_hook(self, owner):
        def hook(module, inputs):
            del module
            tensor = _first_tensor(inputs)
            if tensor is None:
                raise RuntimeError("activation-cost input has no tensor")
            self.activation_elements[owner] += int(tensor.numel())
        return hook

    def _activation_output_hook(self, owner, divisor):
        def hook(module, inputs, output):
            del module, inputs
            tensor = _first_tensor(output)
            if tensor is None or int(tensor.numel()) % int(divisor):
                raise RuntimeError("activation-cost output shape changed")
            self.activation_elements[owner] += \
                int(tensor.numel()) // int(divisor)
        return hook

    def _kv_output_hook(self, owner, offset):
        del offset
        return self._activation_output_hook(owner, 2)

    def _concat_input_hook(self, owner, offset):
        def hook(module, inputs):
            del module
            tensor = _first_tensor(inputs)
            if tensor is None or tensor.ndim != 4 or tensor.shape[1] % 2:
                raise RuntimeError("concat activation-cost shape changed")
            del offset
            self.activation_elements[owner] += int(tensor.numel()) // 2
        return hook

    def rows(self):
        weights = tuple(
            (name, int(self.weight_macs[name]))
            for name in self.contract.weight_modules)
        activations = tuple(
            (owner, int(self.activation_elements[owner]))
            for block in self.contract.blocks
            for owner in block.activation_owners)
        if any(value <= 0 for name, value in weights) or any(
                value <= 0 for owner, value in activations):
            raise RuntimeError("official cost capture has unexecuted owners")
        return weights, activations

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _model_row(configuration, model):
    rows = tuple(row for row in configuration.models if row.model == model)
    if len(rows) != 1:
        raise ValueError("selected model configuration is not unique")
    return rows[0]


def _write_json(path: Path, payload) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")


def _write_cost_rows(path: Path, fields, rows) -> None:
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _descriptor_matrix_sha256(indices, matrix) -> str:
    payload = {
        "indices": [int(index) for index in indices],
        "matrix": np.asarray(matrix, dtype=np.float64).tolist(),
    }
    encoded = json.dumps(
        payload, separators=(",", ":"), allow_nan=False,
        ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate strict official NYU static quantization inputs")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_ORDER, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--calibration-seed", type=int, required=True)
    parser.add_argument("--evaluation-seed", type=int, required=True)
    parser.add_argument("--candidate-samples", type=int, required=True)
    parser.add_argument("--tail-samples", type=int, required=True)
    parser.add_argument("--calibration-metadata", type=Path, required=True)
    parser.add_argument("--calibration-indices", type=Path, required=True)
    parser.add_argument("--evaluation-protocol", type=Path, required=True)
    parser.add_argument("--weight-cost-rows", type=Path, required=True)
    parser.add_argument("--activation-cost-rows", type=Path, required=True)
    return parser


def run(args) -> StaticInputPaths:
    configuration = load_selected_quantization_config(args.config)
    model_config = _model_row(configuration, args.model)
    if str(args.device) != str(model_config.device):
        raise ValueError("static-input device differs from model configuration")
    if int(args.calibration_seed) < 0 or int(args.evaluation_seed) < 0 or \
            int(args.candidate_samples) < CALIBRATION_COUNT or \
            int(args.tail_samples) != 32:
        raise ValueError("static-input selection settings are invalid")
    paths = StaticInputPaths(
        calibration_metadata=args.calibration_metadata.resolve(),
        calibration_indices=args.calibration_indices.resolve(),
        evaluation_protocol=args.evaluation_protocol.resolve(),
        weight_cost_rows=args.weight_cost_rows.resolve(),
        activation_cost_rows=args.activation_cost_rows.resolve(),
    )
    destinations = tuple(vars(paths).values())
    if len(set(destinations)) != len(destinations):
        raise ValueError("static-input output paths must be unique")
    parents = set(path.parent for path in destinations)
    if len(parents) != 1:
        raise ValueError("static-input outputs require one explicit directory")
    output_root = next(iter(parents))
    if output_root.exists() or any(path.exists() for path in destinations):
        raise FileExistsError("static-input output directory already exists: %s" %
                              output_root)
    if paths.calibration_metadata != model_config.calibration_metadata.resolve():
        raise ValueError("calibration metadata output differs from configuration")

    runtime = NYUModelRuntime.from_config(model_config)
    train_list = Path(runtime.saved_args.train_list).resolve()
    evaluation_list = Path(runtime.saved_args.eval_list).resolve()
    train_dataset = runtime.build_dataset("train")
    evaluation_dataset = runtime.build_dataset("val")
    if int(args.candidate_samples) > len(train_dataset):
        raise ValueError("candidate count exceeds the train split")
    evaluation_indices = tuple(model_config.evaluation_indices)
    if len(evaluation_indices) != EVALUATION_COUNT or \
            max(evaluation_indices) >= len(evaluation_dataset):
        raise ValueError("configured evaluation identities exceed val split")
    generator = np.random.default_rng(int(args.calibration_seed))
    candidate_indices = tuple(int(index) for index in generator.choice(
        len(train_dataset), size=int(args.candidate_samples), replace=False))

    device = torch.device(args.device)
    model = runtime.build_model(device)
    contract = build_model_quantization_contract(args.model, model)
    cost_capture = CostCapture(model, contract)
    cost_sample = seeded_sample(
        evaluation_dataset, evaluation_indices[0], int(args.evaluation_seed))
    cost_input, _ = runtime.model_input(
        dict((key, value.unsqueeze(0) if torch.is_tensor(value) else value)
             for key, value in cost_sample.items()), device)
    with torch.no_grad():
        model(*cost_input)
    weight_costs, activation_costs = cost_capture.rows()
    cost_capture.close()

    descriptor_capture = ActivationDescriptorCapture(
        args.model, model, contract)
    descriptor_rows = []
    with torch.no_grad():
        for index in candidate_indices:
            sample = seeded_sample(
                train_dataset, index, int(args.calibration_seed))
            raw = raw_descriptor(
                sample["rgbd"][:3], sample["depth"], sample["rgbd"][3:4])
            batch = dict(
                (key, value.unsqueeze(0) if torch.is_tensor(value) else value)
                for key, value in sample.items())
            model_input, _ = runtime.model_input(batch, device)
            descriptor_capture.reset()
            model(*model_input)
            activation = descriptor_capture.descriptor()
            descriptor_rows.append(tuple(
                float(raw[name]) for name in RAW_DESCRIPTOR_FIELDS) + tuple(
                    float(activation[name])
                    for name in ACTIVATION_DESCRIPTOR_FIELDS[args.model]))
    descriptor_capture.close()
    runtime.close()

    descriptor_matrix = np.asarray(descriptor_rows, dtype=np.float64)
    schema = FeatureSchema(
        names=RAW_DESCRIPTOR_FIELDS + ACTIVATION_DESCRIPTOR_FIELDS[args.model],
        groups=RAW_DESCRIPTOR_GROUPS + tuple(
            "activation" for name in ACTIVATION_DESCRIPTOR_FIELDS[args.model]),
        diagnostic=RAW_DESCRIPTOR_DIAGNOSTIC + tuple(
            False for name in ACTIVATION_DESCRIPTOR_FIELDS[args.model]),
    )
    normalizer = fit_robust_normalizer(descriptor_matrix, schema)
    normalized = normalizer.transform(descriptor_matrix)
    tail = select_tail_cover(
        np.asarray(candidate_indices, dtype=np.int64), normalized,
        normalizer.names, int(args.tail_samples))
    distance = grouped_pairwise_distance(normalized, normalizer.groups)
    medoids = deterministic_weighted_kmedoids(
        np.asarray(candidate_indices, dtype=np.int64), distance,
        np.ones(len(candidate_indices), dtype=np.float64),
        CALIBRATION_COUNT - int(args.tail_samples), tail.selected_indices)
    calibration_indices = tuple(tail.selected_indices) + \
        tuple(medoids.medoid_indices)
    if len(calibration_indices) != CALIBRATION_COUNT or \
            len(set(calibration_indices)) != CALIBRATION_COUNT:
        raise RuntimeError("generated calibration identity count changed")

    checkpoint_sha256 = file_sha256(model_config.checkpoint)
    schema_sha256 = descriptor_schema_sha256(args.model)
    common = {
        "format_version": 1,
        "model": args.model,
        "dataset": DATASET_IDENTITY,
        "checkpoint": str(model_config.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "data_root": str(model_config.data_root.resolve()),
    }
    calibration_identity = ordered_split_identity_sha256(
        "train", calibration_indices)
    evaluation_identity = ordered_split_identity_sha256(
        "validation", evaluation_indices)
    calibration_payload = {
        **common,
        "split": "train",
        "selection": SELECTION_IDENTITY,
        "count": CALIBRATION_COUNT,
        "indices": list(calibration_indices),
        "train_list": str(train_list),
        "train_list_sha256": file_sha256(train_list),
        "descriptor_schema_sha256": schema_sha256,
        "ordered_identity_sha256": calibration_identity,
    }
    evaluation_payload = {
        **common,
        "split": "validation",
        "evaluation_samples": EVALUATION_COUNT,
        "evaluation_indices": list(evaluation_indices),
        "seed": int(args.evaluation_seed),
        "evaluation_list": str(evaluation_list),
        "evaluation_list_sha256": file_sha256(evaluation_list),
        "ordered_identity_sha256": evaluation_identity,
    }
    weight_rows = tuple({"module": name, "macs": value}
                        for name, value in weight_costs)
    activation_rows = tuple({
        "site": owner[0], "role": owner[1], "elements": value,
    } for owner, value in activation_costs)
    if not bool(np.isfinite(descriptor_matrix).all()) or \
            len(tail.selected_indices) != int(args.tail_samples) or \
            len(medoids.medoid_indices) != \
            CALIBRATION_COUNT - int(args.tail_samples):
        raise RuntimeError("static-input selection evidence is incomplete")

    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(paths.calibration_indices, calibration_payload)
    _write_json(paths.evaluation_protocol, evaluation_payload)
    _write_cost_rows(paths.weight_cost_rows, ("module", "macs"), weight_rows)
    _write_cost_rows(
        paths.activation_cost_rows, ("site", "role", "elements"),
        activation_rows)
    metadata = {
        **common,
        "train_list": str(train_list),
        "train_list_sha256": file_sha256(train_list),
        "evaluation_list": str(evaluation_list),
        "evaluation_list_sha256": file_sha256(evaluation_list),
        "calibration_indices": list(calibration_indices),
        "evaluation_indices": list(evaluation_indices),
        "calibration_source": {
            "split": "train",
            "selection": SELECTION_IDENTITY,
            "count": CALIBRATION_COUNT,
            "ordered_identity_sha256": calibration_identity,
        },
        "evaluation_source": {
            "split": "validation",
            "count": EVALUATION_COUNT,
            "seed": int(args.evaluation_seed),
            "ordered_identity_sha256": evaluation_identity,
        },
        "descriptor_schema": {
            "raw": list(RAW_DESCRIPTOR_FIELDS),
            "activation": list(ACTIVATION_DESCRIPTOR_FIELDS[args.model]),
            "sha256": schema_sha256,
        },
        "cost_coverage": {
            "weight_modules": [name for name, value in weight_costs],
            "activation_owners": [list(owner)
                                  for owner, value in activation_costs],
        },
        "artifact_sha256": {
            "calibration_indices": file_sha256(paths.calibration_indices),
            "evaluation_protocol": file_sha256(paths.evaluation_protocol),
            "weight_cost_rows": file_sha256(paths.weight_cost_rows),
            "activation_cost_rows": file_sha256(paths.activation_cost_rows),
        },
    }
    _write_json(paths.calibration_metadata, metadata)
    validate_static_input_bundle(model_config, paths)
    print(
        "%s static inputs candidate_descriptor_sha256=%s" %
        (args.model, _descriptor_matrix_sha256(
            candidate_indices, descriptor_matrix)),
        flush=True)
    return paths


def main(argv=None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
