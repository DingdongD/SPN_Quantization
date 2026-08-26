#!/usr/bin/env python3
"""One-sample hard CUDA smoke matrix for one official selected NYU model."""

from __future__ import annotations

import argparse
from argparse import Namespace
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.nyu_model_runtime import NYUModelRuntime  # noqa: E402
from scripts.run_nyu_model_hawq_trace import (  # noqa: E402
    build_trace_parameter_blocks,
    estimate_model_trace_samples,
    model_hessian_vector_settings,
)
from scripts.run_nyu_model_p3t3_search import (  # noqa: E402
    HardDeploymentP3T3Evaluator,
    HardDeploymentSettings,
    _preserve_input_policy,
    _propagation_valid,
)
from scripts.run_nyu_qdrop_reconstruction import (  # noqa: E402
    CalibrationSplit,
    algorithm_probability,
    ordered_sample_identity_sha256,
    run_contract_reconstruction,
)
from scripts.run_nyu_rtn_quantization import (  # noqa: E402
    batch_from_sample,
    seeded_sample,
)
from scripts.train_nyu_selected_qat import (  # noqa: E402
    ModelActivationRangeCollector,
    _joint_adapter,
    _propagation_config,
    _selected_target_plan,
    set_model_qat_train_mode,
    uniform_qat_assignment,
)
from spn_quant import mixed_precision  # noqa: E402
from spn_quant.adapters import install_model_semantic_adapter  # noqa: E402
from spn_quant.experiment_config import (  # noqa: E402
    MODEL_ORDER,
    load_selected_quantization_config,
)
from spn_quant.hawq_trace import (  # noqa: E402
    HutchinsonTraceConfig,
    masked_curvature_loss,
)
from spn_quant.model_contracts import (  # noqa: E402
    build_model_quantization_contract,
)
from spn_quant.propagation import (  # noqa: E402
    install_propagation_adapter,
)
from spn_quant.qat.model_methods import (  # noqa: E402
    ModelMethodQATConfig,
    ModelMethodQATController,
)
from spn_quant.qat.task_loss import (  # noqa: E402
    ModelTaskLossWeights,
    model_task_aware_loss,
)
from spn_quant.qdrop_config import load_qdrop_config  # noqa: E402


SMOKE_METHODS = (
    "fp32",
    "rtn_w8a8",
    "rtn_w4a4",
    "qdrop_w6a6",
    "brecq_w6a6",
    "lsqplus_w4a4_step",
    "hawq_probe",
    "p3_t3_candidate",
)
EXPECTED_SHAPE = (1, 1, 228, 304)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _revision(path: Path) -> dict:
    source = Path(path).resolve()
    return {
        "path": str(source),
        "size_bytes": int(source.stat().st_size),
        "sha256": _file_sha256(source),
    }


def assert_official_prediction(
        model: str, method: str, prediction: torch.Tensor,
        preserve_input: bool, propagation_rows, native_calls: int,
        official_propagation_calls: int) -> dict:
    if tuple(prediction.shape) != EXPECTED_SHAPE:
        raise RuntimeError(
            "%s %s prediction shape changed: %s" %
            (model, method, tuple(prediction.shape)))
    if not bool(torch.isfinite(prediction).all().item()):
        raise RuntimeError("%s %s prediction is not finite" % (model, method))
    if int(native_calls) <= 0:
        raise RuntimeError(
            "%s %s native extension was not executed" % (model, method))
    if int(official_propagation_calls) <= 0:
        raise RuntimeError(
            "%s %s official propagation operator was not executed" %
            (model, method))
    if not _propagation_valid(
            model, preserve_input, tuple(propagation_rows)):
        raise RuntimeError(
            "%s %s propagation invariants failed" % (model, method))
    return {
        "shape": list(prediction.shape),
        "finite": 1,
        "native_extension_calls": int(native_calls),
        "official_propagation_calls": int(official_propagation_calls),
        "propagation_valid": 1,
    }


def _assert_output_invariants(model_config, output) -> dict:
    if not isinstance(output, dict):
        raise TypeError("official model output must be a dictionary")
    required = {"pred", "pred_init", "offset", "aff"}
    if not required.issubset(output):
        raise KeyError("official model propagation output fields changed")
    state_name = "list_feat" if model_config.model == "dyspn" else \
        "pred_inter"
    states = output[state_name]
    if not isinstance(states, list) or \
            len(states) != int(model_config.propagation_iterations):
        raise RuntimeError("official propagation iteration count changed")
    if any(tuple(state.shape) != EXPECTED_SHAPE for state in states):
        raise RuntimeError("official propagation state shape changed")
    if any(not bool(torch.isfinite(state).all().item()) for state in states):
        raise RuntimeError("official propagation state is not finite")
    if not torch.is_tensor(output["offset"]) and not \
            isinstance(output["offset"], (tuple, list)):
        raise TypeError("official propagation offset representation changed")
    affinity = output["aff"]
    affinity_rows = tuple(affinity) if isinstance(
        affinity, (tuple, list)) else (affinity,)
    if not affinity_rows or any(
            not torch.is_tensor(row) or not bool(torch.isfinite(row).all().item())
            for row in affinity_rows):
        raise RuntimeError("official affinity output is not finite")
    errors = []
    for row in affinity_rows:
        neighbor_axis = 2 if row.ndim == 5 else 1
        errors.append(float((
            row.sum(dim=neighbor_axis) - 1.0).abs().max().item()))
    affinity_error = max(errors)
    if not math.isfinite(affinity_error) or affinity_error > 1.0e-5:
        raise RuntimeError("official affinity normalization invariant failed")
    return {
        "iterations": len(states),
        "state_shape": list(states[0].shape),
        "states_finite": 1,
        "affinity_sum_max_error": affinity_error,
    }


def _assert_hawq_output_invariants(
        model_config, output, preserve_input: bool,
        propagation_rows) -> dict:
    invariants = _assert_output_invariants(model_config, output)
    if not _propagation_valid(
            model_config.model, preserve_input, tuple(propagation_rows)):
        raise RuntimeError(
            "%s HAWQ propagation invariants failed" % model_config.model)
    return invariants


def assert_smoke_matrix_rows(methods) -> None:
    if tuple(methods) != SMOKE_METHODS:
        raise RuntimeError("official smoke method order changed")
    for method in SMOKE_METHODS:
        row = methods[method]
        if "propagation_valid" not in row or \
                int(row["propagation_valid"]) != 1:
            raise RuntimeError(
                "%s propagation validation is missing" % method)


def _collect_hawq_propagation_rows(model_config, model, model_input):
    propagation = install_propagation_adapter(model_config.model, model)
    propagation.observe()
    with torch.no_grad():
        model(*model_input)
    propagation.freeze()
    propagation.configure(_propagation_config())
    with torch.no_grad():
        model(*model_input)
    preserve_input = _preserve_input_policy(
        model_config.model, propagation)
    rows = tuple(propagation.statistics())
    propagation.close()
    return preserve_input, rows


class NativeExtensionTracker(object):
    def __init__(self, model_name: str) -> None:
        self.model_name = str(model_name)
        self.calls = 0
        self.propagation_calls = 0
        self.originals = []
        if self.model_name == "dyspn":
            self._install_dyspn()
        else:
            self._install_dcn()

    def _install_dyspn(self) -> None:
        for name in tuple(sys.modules):
            module = sys.modules[name]
            if module is None or not name.startswith("DySPN") or not \
                    hasattr(module, "deform_conv2d"):
                continue
            original = module.deform_conv2d

            def wrapped(*args, _original=original, **kwargs):
                self.calls += 1
                return _original(*args, **kwargs)

            module.deform_conv2d = wrapped
            self.originals.append((module, "deform_conv2d", original))
        if not self.originals:
            raise RuntimeError("DySPN native deform-conv call site is missing")
        module = importlib.import_module("DySPN.module")
        functional = module.F
        original_grid_sample = functional.grid_sample

        def wrapped_grid_sample(*args, **kwargs):
            self.propagation_calls += 1
            return original_grid_sample(*args, **kwargs)

        functional.grid_sample = wrapped_grid_sample
        self.originals.append(
            (functional, "grid_sample", original_grid_sample))

    def _install_dcn(self) -> None:
        module = importlib.import_module("DCN")
        if not hasattr(module, "modulated_deform_conv_forward"):
            raise RuntimeError("DCN forward entry point is missing")
        original = module.modulated_deform_conv_forward

        def wrapped(*args, **kwargs):
            self.calls += 1
            self.propagation_calls += 1
            return original(*args, **kwargs)

        module.modulated_deform_conv_forward = wrapped
        self.originals.append(
            (module, "modulated_deform_conv_forward", original))

    def since(self, previous: int) -> int:
        return int(self.calls) - int(previous)

    def propagation_since(self, previous: int) -> int:
        return int(self.propagation_calls) - int(previous)

    def exercise_required_extension(self, device) -> None:
        if self.model_name != "dyspn":
            return
        requested = torch.device(device)
        if requested.type != "cuda" or requested.index is None or \
                torch.cuda.current_device() != int(requested.index):
            raise RuntimeError(
                "DySPN extension primitive current CUDA device changed")
        call_sites = tuple(
            module for module, name, original in self.originals
            if name == "deform_conv2d")
        if not call_sites:
            raise RuntimeError("DySPN deform-conv call site is unavailable")
        values = torch.ones(1, 1, 4, 4, device=requested)
        offsets = torch.zeros(1, 18, 4, 4, device=requested)
        mask = torch.ones(1, 9, 4, 4, device=requested)
        weight = torch.ones(1, 1, 3, 3, device=requested)
        bias = torch.zeros(1, device=requested)
        output = call_sites[0].deform_conv2d(
            values, offsets, weight, bias, (1, 1), (1, 1), (1, 1),
            mask=mask)
        if tuple(output.shape) != (1, 1, 4, 4) or not bool(
                torch.isfinite(output).all().item()):
            raise RuntimeError("DySPN deform-conv CUDA primitive failed")

    def close(self) -> None:
        for module, name, original in self.originals:
            setattr(module, name, original)
        self.originals = []


def _model_config(selected, model):
    rows = tuple(row for row in selected.models if row.model == str(model))
    if len(rows) != 1:
        raise ValueError("smoke model configuration is not unique")
    return rows[0]


def _sample(runtime, split: str, index: int, seed: int):
    dataset = runtime.build_dataset(split)
    if int(index) < 0 or int(index) >= len(dataset):
        raise ValueError("smoke sample index is outside the %s split" % split)
    return dataset, batch_from_sample(seeded_sample(dataset, index, seed))


def _fp32_and_rtn_smokes(
        model_config, sample_index: int, seed: int,
        output: Path):
    runtime = NYUModelRuntime.from_config(model_config)
    model = runtime.build_model(runtime.device)
    tracker = NativeExtensionTracker(model_config.model)
    contract = build_model_quantization_contract(model_config.model, model)
    valset, batch = _sample(runtime, "val", sample_index, seed)
    model_args, target = runtime.model_input(batch, runtime.device)
    before = tracker.calls
    propagation_before = tracker.propagation_calls
    tracker.exercise_required_extension(runtime.device)
    with torch.no_grad():
        fp_output = model(*model_args)
    fp_prediction = runtime.prediction(fp_output)
    native_calls = tracker.since(before)
    if tuple(fp_prediction.shape) != EXPECTED_SHAPE or not bool(
            torch.isfinite(fp_prediction).all().item()):
        raise RuntimeError("FP32 prediction shape or finite check failed")
    if native_calls <= 0:
        raise RuntimeError("FP32 native extension was not executed")
    official_calls = tracker.propagation_since(propagation_before)
    if official_calls <= 0:
        raise RuntimeError("FP32 official propagation operator was not executed")
    fp = {
        "shape": list(fp_prediction.shape),
        "finite": 1,
        "native_extension_calls": native_calls,
        "official_propagation_calls": official_calls,
        "propagation_valid": 1,
    }
    fp["model_invariants"] = _assert_output_invariants(
        model_config, fp_output)
    smoke_metadata = output / "one_sample_calibration.json"
    smoke_metadata.write_text(json.dumps({
        "calibration_indices": [int(sample_index)],
        "evaluation_indices": [int(sample_index)],
        "calibration_source": {"selection": "32_tail_96_kmedoids"},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    registry = mixed_precision.AllocationRegistry(
        weights_by_block=dict(
            (block.name, block.weight_modules) for block in contract.blocks),
        activations_by_block=dict(
            (block.name, block.activation_owners) for block in contract.blocks),
        blocks=contract.block_names,
        model_name=contract.model_name,
    )
    settings = HardDeploymentSettings(
        device=model_config.device,
        calibration_metadata=smoke_metadata,
        calibration_count=1,
        evaluation_indices=(int(sample_index),),
        base_weight_bits=4,
        base_activation_bits=4,
        promotion_weight_bits=8,
        promotion_activation_bits=8,
        fold_conv_bn=False,
        fold_max_error=0.05,
        joint_clip_factors=(0.8, 1.0, 1.2),
        joint_search_rounds=1,
        joint_cache_sample_limit=1,
        joint_cache_byte_limit=1073741824,
    )
    evaluator = HardDeploymentP3T3Evaluator(
        runtime, model, contract, registry, settings)
    records = {"fp32": fp}
    assignments = {
        "rtn_w8a8": mixed_precision.uniform_assignment(registry, 8, 8),
        "rtn_w4a4": mixed_precision.uniform_assignment(registry, 4, 4),
        "p3_t3_candidate": mixed_precision.promoted_assignment(
            registry,
            contract.prefix_groups[0],
            4, 4, 8, 8,
        ),
    }
    for method in ("rtn_w8a8", "rtn_w4a4", "p3_t3_candidate"):
        assignment = assignments[method]
        candidate = mixed_precision.P3T3Candidate(
            name=method,
            stage="smoke",
            prefix=contract.prefix_groups[0]
                if method == "p3_t3_candidate" else (),
            tail=(),
            promoted_blocks=contract.prefix_groups[0]
                if method == "p3_t3_candidate" else (),
            assignment=assignment,
        )
        evaluator.configure_hard_candidate(candidate)
        before = tracker.calls
        propagation_before = tracker.propagation_calls
        tracker.exercise_required_extension(runtime.device)
        prediction, ground_truth = evaluator._forward(
            evaluator._sample_batch(valset, sample_index))
        del ground_truth
        row = assert_official_prediction(
            model_config.model, method, prediction,
            _preserve_input_policy(
                model_config.model, evaluator.propagation_adapter),
            evaluator.propagation_adapter.statistics(),
            tracker.since(before),
            tracker.propagation_since(propagation_before))
        row["weight_owner_count"] = len(assignment.weight_bits)
        row["activation_owner_count"] = len(assignment.activation_bits)
        row["protected_module_count"] = len(contract.protected_modules)
        records[method] = row
    evaluator.close()
    tracker.close()
    runtime.close()
    return records, contract


def _hawq_smoke(
        model_config, sample_index: int, seed: int,
        ) -> dict:
    runtime = NYUModelRuntime.from_config(model_config)
    model = runtime.build_model(runtime.device)
    tracker = NativeExtensionTracker(model_config.model)
    contract = build_model_quantization_contract(model_config.model, model)
    dataset, batch = _sample(runtime, "train", sample_index, seed)
    del dataset
    model_input, target = runtime.model_input(batch, runtime.device)
    blocks = build_trace_parameter_blocks(model, contract)
    before = tracker.calls
    propagation_before = tracker.propagation_calls
    tracker.exercise_required_extension(runtime.device)

    def loss_fn():
        output = model(*model_input)
        prediction = runtime.prediction(output)
        return masked_curvature_loss(
            prediction,
            target,
            torch.isfinite(target) & (target > 0.0),
            1.0,
            0.25,
            0.1,
        )

    estimates = estimate_model_trace_samples(
        model_config.model,
        tuple((block.name, block.parameters) for block in blocks),
        loss_fn,
        HutchinsonTraceConfig(1, int(seed)),
    )
    values = tuple(value for name, rows in estimates for value in rows)
    if len(values) != len(blocks) or not all(
            math.isfinite(float(value)) for value in values):
        raise RuntimeError("one-probe HAWQ trace coverage is invalid")
    with torch.no_grad():
        canonical_output = model(*model_input)
    prediction = runtime.prediction(canonical_output)
    if tuple(prediction.shape) != EXPECTED_SHAPE or not bool(
            torch.isfinite(prediction).all().item()):
        raise RuntimeError("HAWQ probe prediction shape or finite check failed")
    preserve_input, propagation_rows = _collect_hawq_propagation_rows(
        model_config, model, model_input)
    model_invariants = _assert_hawq_output_invariants(
        model_config, canonical_output, preserve_input,
        propagation_rows)
    native_calls = tracker.since(before)
    if native_calls <= 0:
        raise RuntimeError("HAWQ probe did not execute the native extension")
    official_calls = tracker.propagation_since(propagation_before)
    if official_calls <= 0:
        raise RuntimeError(
            "HAWQ probe did not execute the official propagation operator")
    tracker.close()
    runtime.close()
    hessian_mode, hessian_epsilon = model_hessian_vector_settings(
        model_config.model)
    return {
        "shape": list(prediction.shape),
        "finite": 1,
        "native_extension_calls": native_calls,
        "official_propagation_calls": official_calls,
        "probe_count": 1,
        "block_count": len(blocks),
        "trace_values_finite": 1,
        "hessian_vector_mode": hessian_mode,
        "hessian_vector_epsilon": hessian_epsilon,
        "propagation_valid": 1,
        "model_invariants": model_invariants,
    }


def _lsqplus_task_forward(
        runtime, student, teacher, student_semantic, teacher_semantic,
        model_input, target, loss_weights, boundary_threshold_m):
    teacher.eval()
    teacher_semantic.begin_task_capture()
    with torch.no_grad():
        teacher_output = teacher(*model_input)
        teacher_capture = teacher_semantic.task_capture()
    if not torch.equal(
            runtime.prediction(teacher_output), teacher_capture.prediction):
        raise RuntimeError("LSQ++ smoke teacher capture changed")
    student_semantic.begin_task_capture()
    student_output = student(*model_input)
    student_capture = student_semantic.task_capture()
    prediction = runtime.prediction(student_output)
    if not torch.equal(prediction, student_capture.prediction):
        raise RuntimeError("LSQ++ smoke student capture changed")
    loss = model_task_aware_loss(
        student_capture,
        teacher_capture,
        target,
        target > 0.0,
        loss_weights,
        boundary_threshold_m,
    )
    if any(not bool(torch.isfinite(value).all().item())
           for value in loss.as_dict().values()):
        raise FloatingPointError("LSQ++ smoke task loss is non-finite")
    return prediction, loss


def _lsqplus_smoke(
        model_config, sample_index: int, seed: int,
        qat_settings, hard_deployment) -> dict:
    from scripts.hardware_aligned_quantization import prepare_hardware_model

    runtime = NYUModelRuntime.from_config(model_config)
    teacher_runtime = NYUModelRuntime.from_config(model_config)
    model = runtime.build_model(runtime.device)
    teacher = teacher_runtime.build_model(runtime.device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    tracker = NativeExtensionTracker(model_config.model)
    contract = build_model_quantization_contract(model_config.model, model)
    target_plan = _selected_target_plan(model_config.model, model, contract)
    dataset, batch = _sample(runtime, "train", sample_index, seed)
    del dataset
    model_input, target = runtime.model_input(batch, runtime.device)
    preparation = prepare_hardware_model(
        model, model_input, fold=bool(hard_deployment["fold_conv_bn"]))
    if float(preparation["primary_max_abs_error"]) > \
            float(hard_deployment["fold_max_error"]):
        raise RuntimeError("LSQ++ smoke graph preparation exceeds threshold")
    training = {
        "joint_clip_factors": tuple(
            float(value) for value in
            hard_deployment["joint_clip_factors"]),
        "joint_search_rounds": 1,
        "joint_cache_sample_limit": 1,
        "joint_cache_byte_limit": int(
            hard_deployment["joint_cache_byte_limit"]),
    }
    joint = _joint_adapter(model, contract, training)
    propagation = install_propagation_adapter(model_config.model, model)
    collector = ModelActivationRangeCollector(model, contract, target_plan)
    propagation.observe()
    if joint is not None:
        joint.observe_qdrop_ranges()
    before = tracker.calls
    propagation_before = tracker.propagation_calls
    tracker.exercise_required_extension(runtime.device)
    with torch.no_grad():
        model(*model_input)
    propagation.freeze()
    propagation.configure(_propagation_config())
    if joint is not None:
        joint.freeze_qdrop_ranges()
    initialization = collector.initialization_rows(joint)
    collector.close()
    assignment = uniform_qat_assignment(contract, 4)
    controller = ModelMethodQATController(
        model,
        contract,
        target_plan,
        ModelMethodQATConfig(
            method="lsqplus",
            weight_bits=assignment.weight_bits,
            activation_bits=assignment.activation_bits,
            propagation=_propagation_config(),
            hawq_range_momentum=float(
                qat_settings["hawq_range_momentum"]),
        ),
        joint_adapter=joint,
        propagation_adapter=propagation,
    )
    controller.initialize_activations(initialization)
    controller.install()
    student_semantic = install_model_semantic_adapter(
        model, model_config.model, strict=True)
    student_semantic.delegate_quantization()
    teacher_semantic = install_model_semantic_adapter(
        teacher, model_config.model, strict=True)
    teacher_semantic.delegate_quantization()
    set_model_qat_train_mode(model_config.model, model)
    optimizer = torch.optim.SGD(
        controller.parameters(),
        lr=float(qat_settings["learning_rate"]),
        momentum=float(qat_settings["momentum"]),
        weight_decay=float(qat_settings["weight_decay"]),
    )
    optimizer.zero_grad(set_to_none=True)
    loss_weights = ModelTaskLossWeights(
        depth=float(qat_settings["depth_loss_weight"]),
        boundary=float(qat_settings["boundary_loss_weight"]),
        teacher=float(qat_settings["teacher_loss_weight"]),
        initial_depth=float(qat_settings["initial_depth_loss_weight"]),
        propagation=float(qat_settings["propagation_loss_weight"]),
    )
    prediction, loss = _lsqplus_task_forward(
        runtime, model, teacher, student_semantic, teacher_semantic,
        model_input, target, loss_weights,
        float(qat_settings["boundary_threshold_m"]))
    loss.total.backward()
    gradient_norm = controller.assert_finite_gradients()
    optimizer.step()
    manifest = controller.hard_deployment_manifest()
    row = assert_official_prediction(
        model_config.model, "lsqplus_w4a4_step", prediction,
        _preserve_input_policy(model_config.model, propagation),
        propagation.statistics(), tracker.since(before),
        tracker.propagation_since(propagation_before))
    row.update({
        "loss": float(loss.total.detach().item()),
        "gradient_norm": gradient_norm,
        "optimizer_steps": 1,
        "materialized_weight_count": manifest["materialized_weight_count"],
        "activation_owner_count": manifest["activation_owner_count"],
        "protected_scale_roles_excluded":
            manifest["protected_scale_roles_excluded"],
    })
    teacher_semantic.close()
    student_semantic.close()
    controller.remove()
    propagation.close()
    if joint is not None:
        joint.close()
    tracker.close()
    teacher_runtime.close()
    runtime.close()
    return row


def _qdrop_smokes(
        model_config, sample_index: int, seed: int,
        qdrop_config_path: Path, output: Path) -> dict:
    runtime = NYUModelRuntime.from_config(model_config)
    model = runtime.build_model(runtime.device)
    contract = build_model_quantization_contract(model_config.model, model)
    runtime.close()
    del model
    tracker = NativeExtensionTracker(model_config.model)
    base = load_qdrop_config(qdrop_config_path)
    smoke_config = replace(
        base,
        reconstruction=replace(
            base.reconstruction,
            batch_size=1,
            capture_batch_size=1,
            steps=1,
        ),
    )
    split = CalibrationSplit(
        calibration=(int(sample_index),),
        reconstruction=(int(sample_index),),
        validation=(int(sample_index),),
    )
    calibration_identity = ordered_sample_identity_sha256(
        "train", split.calibration)
    evaluation_identity = ordered_sample_identity_sha256(
        "train_validation", split.validation)
    protocol = {
        "checkpoint_sha256": _file_sha256(model_config.checkpoint),
        "calibration_indices_sha256": calibration_identity,
        "calibration_metadata_sha256": calibration_identity,
        "evaluation_protocol_sha256": evaluation_identity,
        "calibration_identity_sha256": calibration_identity,
        "reconstruction_identity_sha256": calibration_identity,
        "validation_identity_sha256": calibration_identity,
        "evaluation_identity_sha256": evaluation_identity,
        "calibration_indices": list(split.calibration),
        "reconstruction_indices": list(split.reconstruction),
        "validation_indices": list(split.validation),
        "evaluation_indices": list(split.validation),
        "evaluation_seed": int(seed),
    }
    args = Namespace(
        run_dir=model_config.run_dir,
        checkpoint=model_config.checkpoint,
        data_root=model_config.data_root,
        model=model_config.model,
        device=model_config.device,
        algorithm="qdrop",
        precision="W6A6",
        phase="formal",
        seed=int(seed),
    )
    rows = {}
    for method, algorithm in (
            ("qdrop_w6a6", "qdrop"),
            ("brecq_w6a6", "brecq")):
        args.algorithm = algorithm
        before = tracker.calls
        propagation_before = tracker.propagation_calls
        tracker.exercise_required_extension(model_config.device)
        result = run_contract_reconstruction(
            args,
            smoke_config,
            algorithm_probability(algorithm),
            split,
            protocol,
            "formal",
            output / method,
            contract,
            torch.device(model_config.device),
            False,
            0.05,
        )
        manifest_path = Path(result["hard_deployment_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest["materialized_hard_weights"]) != 1 or \
                tuple(result["prediction_shape"]) != EXPECTED_SHAPE or \
                int(result["finite"]) != 1:
            raise RuntimeError("%s hard smoke validation failed" % method)
        native_calls = tracker.since(before)
        if native_calls <= 0:
            raise RuntimeError("%s did not execute the native extension" % method)
        official_calls = tracker.propagation_since(propagation_before)
        if official_calls <= 0:
            raise RuntimeError(
                "%s did not execute the official propagation operator" %
                method)
        rows[method] = {
            "shape": list(result["prediction_shape"]),
            "finite": 1,
            "native_extension_calls": native_calls,
            "official_propagation_calls": official_calls,
            "propagation_valid": 1,
            "optimizer_steps_per_block": 1,
            "materialized_hard_weights": 1,
            "hard_deployment_manifest": str(manifest_path.resolve()),
            "hard_deployment_manifest_sha256": _file_sha256(manifest_path),
        }
    tracker.close()
    return rows


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run one official-model selected quantization hard smoke")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--launch-spec", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_ORDER, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--qdrop-config", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args) -> Path:
    started = _utc_now()
    selected = load_selected_quantization_config(args.config)
    model_config = _model_config(selected, args.model)
    if str(args.device) != model_config.device:
        raise ValueError("smoke device differs from selected configuration")
    if Path(sys.executable).resolve() != model_config.python_executable.resolve():
        raise RuntimeError("smoke Python differs from selected configuration")
    launch_payload = json.loads(args.launch_spec.read_text(encoding="utf-8"))
    declared_environment = launch_payload["model_environments"][args.model]
    if "CUDA_VISIBLE_DEVICES" in declared_environment or \
            "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("CUDA_VISIBLE_DEVICES remapping is forbidden")
    for name in declared_environment:
        if os.environ[name] != str(declared_environment[name]):
            raise RuntimeError("smoke environment differs: %s" % name)
    if args.output.exists():
        raise FileExistsError("smoke output already exists: %s" % args.output)
    if not args.output.parent.is_dir():
        raise FileNotFoundError(
            "smoke output parent is missing: %s" % args.output.parent)
    if args.sample_index < 0 or args.seed < 0:
        raise ValueError("smoke sample and seed must be nonnegative")
    args.output.mkdir(exist_ok=False)

    methods, contract = _fp32_and_rtn_smokes(
        model_config, args.sample_index, args.seed, args.output)
    del contract
    torch.cuda.empty_cache()
    methods.update(_qdrop_smokes(
        model_config, args.sample_index, args.seed,
        args.qdrop_config, args.output))
    torch.cuda.empty_cache()
    methods["lsqplus_w4a4_step"] = _lsqplus_smoke(
        model_config, args.sample_index, args.seed,
        launch_payload["qat"]["lsqplus_w4a4"],
        launch_payload["hard_deployment"])
    torch.cuda.empty_cache()
    methods["hawq_probe"] = _hawq_smoke(
        model_config, args.sample_index, args.seed)
    if tuple(methods) != SMOKE_METHODS:
        methods = dict((name, methods[name]) for name in SMOKE_METHODS)
    assert_smoke_matrix_rows(methods)

    saved_args = json.loads(
        (model_config.run_dir / "args.json").read_text(encoding="utf-8"))
    input_paths = (
        args.config,
        args.launch_spec,
        args.qdrop_config,
        model_config.checkpoint,
        model_config.run_dir / "args.json",
        model_config.run_dir / "meta.json",
        Path(saved_args["train_list"]),
        Path(saved_args["eval_list"]),
    )
    payload = {
        "format_version": 1,
        "model": model_config.model,
        "architecture": model_config.expected_architecture_class,
        "device": model_config.device,
        "python": str(model_config.python_executable),
        "required_cuda_extension": model_config.required_cuda_extension,
        "native_cuda_operator": model_config.native_cuda_operator,
        "sample_index": int(args.sample_index),
        "seed": int(args.seed),
        "command": [str(Path(sys.executable).resolve())] + list(sys.argv),
        "environment": dict(
            (name, str(declared_environment[name]))
            for name in declared_environment),
        "inputs": [_revision(path) for path in input_paths],
        "methods": methods,
        "start_time_utc": started,
        "end_time_utc": _utc_now(),
        "exit_status": 0,
    }
    destination = args.output / "official_one_sample_smoke.json"
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return destination


def main(argv=None) -> None:
    path = run(build_parser().parse_args(argv))
    print(path, flush=True)


if __name__ == "__main__":
    main()
