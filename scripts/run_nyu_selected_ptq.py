#!/usr/bin/env python3
"""Run the exact selected PTQ matrix for official NYU SPN models."""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
from dataclasses import dataclass
from dataclasses import replace
import math
from pathlib import Path
import sys
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from spn_quant.deployment_contract import file_sha256  # noqa: E402
from spn_quant.mixed_precision import BitAssignment  # noqa: E402
from spn_quant.model_contracts import (  # noqa: E402
    QuantizationModelContract,
    build_model_quantization_contract,
)
from scripts.nyu_model_runtime import NYUModelRuntime  # noqa: E402
from scripts.run_nyu_model_p3t3_search import (  # noqa: E402
    HardDeploymentSettings,
)
from spn_quant.experiment_config import (  # noqa: E402
    MODEL_ORDER,
    load_selected_quantization_config,
)
from spn_quant.qdrop_config import load_qdrop_config  # noqa: E402


SELECTED_PTQ_METHODS = (
    "rtn_w8a8",
    "rtn_w4a4",
    "qdrop_w6a6",
    "brecq_w6a6",
    "p3_t3_mixed_ptq",
)

HARD_DEPLOYMENT_FIELDS = {
    "format_version",
    "strict",
    "method",
    "model",
    "weight_bits",
    "activation_bits",
    "module_names",
    "activation_owners",
    "protected_modules",
    "materialized_hard_weights",
    "hard_weights",
    "hard_weights_sha256",
    "deployment_contract",
    "deployment_contract_sha256",
    "optimization_state",
    "optimization_state_sha256",
    "calibration_identity",
    "evaluation_identity",
}


@dataclass(frozen=True)
class SelectedReconstructionPlan:
    method: str
    model_name: str
    module_names: tuple[str, ...]
    activation_owners: tuple[tuple[str, str], ...]
    attention_edges: tuple[str, ...]
    concat_edges: tuple[str, ...]
    protected_modules: tuple[str, ...]
    weight_bits: int
    activation_bits: int


@dataclass(frozen=True)
class SelectedPTQDependencies:
    runtime_factory: Callable
    contract_builder: Callable
    method_executor: Callable


@dataclass(frozen=True)
class ProductionMethodExecutor:
    qdrop_args: Namespace
    qdrop_config: object
    qdrop_split: object
    qdrop_protocol: object
    rtn_settings: HardDeploymentSettings
    selected_device: str

    def __call__(self, runtime, model, contract, plan, method, output,
                 method_config, calibration_identity, evaluation_identity):
        device = torch.device(self.selected_device)
        if device.type != "cuda" or device.index is None:
            raise ValueError(
                "selected reconstruction device must be explicit CUDA")
        if device != runtime.device:
            raise ValueError(
                "selected reconstruction device differs from runtime")
        if method in ("rtn_w8a8", "rtn_w4a4",
                      "p3_t3_mixed_ptq"):
            from scripts.run_nyu_rtn_quantization import (
                materialize_contract_rtn,
            )
            if method == "p3_t3_mixed_ptq":
                settings = self.rtn_settings
            else:
                settings = replace(
                    self.rtn_settings,
                    base_weight_bits=int(method_config["weight_bits"]),
                    base_activation_bits=int(
                        method_config["activation_bits"]),
                    promotion_weight_bits=int(method_config["weight_bits"]),
                    promotion_activation_bits=int(
                        method_config["activation_bits"]),
                )
            return materialize_contract_rtn(
                runtime=runtime,
                model=model,
                contract=contract,
                plan=plan,
                settings=settings,
                output=output,
                calibration_identity=calibration_identity,
                evaluation_identity=evaluation_identity,
            )
        from scripts.run_nyu_qdrop_reconstruction import (
            algorithm_probability,
            run_contract_reconstruction,
        )
        precision = self.qdrop_config.precision("W6A6")
        if (precision.weight_bits, precision.activation_bits) != (6, 6):
            raise ValueError("QDrop configuration W6A6 precision changed")
        if int(self.qdrop_config.reconstruction.steps) != int(
                method_config["steps"]):
            raise ValueError(
                "selected reconstruction steps differ from QDrop config")
        algorithm = {
            "qdrop_w6a6": "qdrop",
            "brecq_w6a6": "brecq",
        }[method]
        args = Namespace(**vars(self.qdrop_args))
        if torch.device(args.device) != device:
            raise ValueError(
                "reconstruction arguments differ from selected device")
        args.device = str(device)
        args.algorithm = algorithm
        args.precision = "W6A6"
        args.phase = "formal"
        args.seed = int(self.qdrop_config.formal.evaluation_seed)
        result = run_contract_reconstruction(
            args,
            self.qdrop_config,
            algorithm_probability(algorithm),
            self.qdrop_split,
            self.qdrop_protocol,
            "formal",
            output,
            contract,
            device,
            self.rtn_settings.fold_conv_bn,
            self.rtn_settings.fold_max_error,
        )
        return Path(result["hard_deployment_manifest"])


def selected_ptq_methods():
    return SELECTED_PTQ_METHODS


def _assignment_payload(payload):
    if tuple(payload) != (
            "model_name", "weight_bits", "activation_bits"):
        raise ValueError("P3/T3 tuple assignment fields changed")
    return BitAssignment(
        weight_bits=tuple(
            (str(row[0]), int(row[1]))
            for row in payload["weight_bits"]),
        activation_bits=tuple(
            ((str(row[0][0]), str(row[0][1])), int(row[1]))
            for row in payload["activation_bits"]),
        model_name=str(payload["model_name"]),
    )


def load_p3_t3_assignment(path: Path,
                          contract: QuantizationModelContract,
                          precision) -> BitAssignment:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if str(payload["model_name"]) != contract.model_name:
        raise ValueError("P3/T3 assignment model differs from contract")
    expected_precision = {
        "base_weight_bits": int(precision["base_weight_bits"]),
        "base_activation_bits": int(precision["base_activation_bits"]),
        "promotion_weight_bits": int(precision["promotion_weight_bits"]),
        "promotion_activation_bits": int(
            precision["promotion_activation_bits"]),
    }
    actual_precision = dict(
        (name, int(payload["precision"][name]))
        for name in expected_precision)
    if actual_precision != expected_precision or \
            set(payload["precision"]) != set(expected_precision):
        raise ValueError("P3/T3 assignment precision differs from config")
    assignment = _assignment_payload(payload["assignment"])
    if assignment.model_name != contract.model_name:
        raise ValueError("P3/T3 tuple assignment model differs from contract")
    approved_weight_bits = {
        expected_precision["base_weight_bits"],
        expected_precision["promotion_weight_bits"],
    }
    approved_activation_bits = {
        expected_precision["base_activation_bits"],
        expected_precision["promotion_activation_bits"],
    }
    if set(bits for name, bits in assignment.weight_bits) - \
            approved_weight_bits or \
            set(bits for owner, bits in assignment.activation_bits) - \
            approved_activation_bits:
        raise ValueError("P3/T3 assignment uses unapproved precision values")
    from scripts.run_nyu_rtn_quantization import build_contract_rtn_plan
    build_contract_rtn_plan(
        contract,
        "p3_t3_mixed_ptq",
        expected_precision["base_weight_bits"],
        expected_precision["base_activation_bits"],
        assignment,
    )
    return assignment


def _validated_file(path, expected_sha256, family):
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError("%s artifact is missing: %s" %
                                (family, artifact))
    if file_sha256(artifact) != str(expected_sha256):
        raise RuntimeError("%s artifact fingerprint differs" % family)
    return artifact


def validate_hard_deployment_manifest(
        path: Path,
        expected_method: str,
        contract: QuantizationModelContract,
        calibration_identity: str,
        evaluation_identity: str):
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(payload) != HARD_DEPLOYMENT_FIELDS:
        raise KeyError("hard deployment manifest fields mismatch")
    if int(payload["format_version"]) != 1 or int(payload["strict"]) != 1:
        raise ValueError("hard deployment manifest must be strict version 1")
    if str(payload["method"]) != expected_method or \
            expected_method not in SELECTED_PTQ_METHODS:
        raise ValueError("hard deployment method differs from selection")
    if str(payload["model"]) != contract.model_name:
        raise ValueError("hard deployment model differs from contract")
    if expected_method in ("qdrop_w6a6", "brecq_w6a6") and \
            (int(payload["weight_bits"]),
             int(payload["activation_bits"])) != (6, 6):
        raise ValueError("selected reconstruction precision must be W6A6")
    uniform_bits = {
        "rtn_w8a8": 8,
        "rtn_w4a4": 4,
    }
    if expected_method in uniform_bits and \
            (int(payload["weight_bits"]),
             int(payload["activation_bits"])) != (
                uniform_bits[expected_method],
                uniform_bits[expected_method]):
        raise ValueError("selected RTN manifest precision differs")
    if tuple(payload["module_names"]) != contract.weight_modules:
        raise ValueError("hard deployment module ownership differs")
    owners = tuple(
        (str(row[0]), str(row[1]))
        for row in payload["activation_owners"])
    expected_owners = tuple(
        owner for block in contract.blocks for owner in block.activation_owners)
    if owners != expected_owners:
        raise ValueError("hard deployment activation ownership differs")
    if tuple(payload["protected_modules"]) != contract.protected_modules:
        raise ValueError("hard deployment protected module record differs")
    if set(payload["module_names"]).intersection(
            payload["protected_modules"]):
        raise ValueError("hard deployment includes protected contract modules")
    if int(payload["materialized_hard_weights"]) != 1:
        raise ValueError("hard deployment requires materialized hard weights")
    if str(payload["calibration_identity"]) != str(calibration_identity):
        raise ValueError("hard deployment calibration identity differs")
    if str(payload["evaluation_identity"]) != str(evaluation_identity):
        raise ValueError("hard deployment evaluation identity differs")
    _validated_file(
        payload["hard_weights"], payload["hard_weights_sha256"],
        "hard weights")
    _validated_file(
        payload["deployment_contract"],
        payload["deployment_contract_sha256"], "deployment contract")
    _validated_file(
        payload["optimization_state"],
        payload["optimization_state_sha256"], "optimization state")
    return payload


def _validate_manifest_plan(payload, plan):
    if isinstance(plan, SelectedReconstructionPlan):
        if (int(payload["weight_bits"]),
                int(payload["activation_bits"])) != (
                    plan.weight_bits, plan.activation_bits):
            raise ValueError("reconstruction manifest differs from plan")
        return
    expected_weights = tuple(plan.weight_bits)
    expected_activations = tuple(plan.activation_bits)
    if plan.method in ("rtn_w8a8", "rtn_w4a4"):
        weight_bits = int(payload["weight_bits"])
        activation_bits = int(payload["activation_bits"])
        actual_weights = tuple(
            (name, weight_bits) for name in plan.module_names)
        actual_activations = tuple(
            (owner, activation_bits) for owner, bits in plan.activation_bits)
    else:
        actual_weights = tuple(
            (str(row[0]), int(row[1]))
            for row in payload["weight_bits"])
        actual_activations = tuple(
            ((str(row[0][0]), str(row[0][1])), int(row[1]))
            for row in payload["activation_bits"])
    if actual_weights != expected_weights or \
            actual_activations != expected_activations:
        raise ValueError("hard deployment precision differs from method plan")


def validate_selected_reconstruction_pair(
        qdrop_manifest: Path,
        brecq_manifest: Path,
        contract: QuantizationModelContract,
        calibration_identity: str,
        evaluation_identity: str):
    qdrop = validate_hard_deployment_manifest(
        qdrop_manifest, "qdrop_w6a6", contract, calibration_identity,
        evaluation_identity)
    brecq = validate_hard_deployment_manifest(
        brecq_manifest, "brecq_w6a6", contract, calibration_identity,
        evaluation_identity)
    if qdrop["optimization_state"] == brecq["optimization_state"]:
        raise ValueError("QDrop and BRECQ optimization states must differ")
    return qdrop, brecq


def _method_plan(method, method_config, contract, p3_t3_assignment):
    from scripts.run_nyu_rtn_quantization import build_contract_rtn_plan
    expected_fields = {
        "rtn_w8a8": (
            "weight_bits", "activation_bits", "calibration_count"),
        "rtn_w4a4": (
            "weight_bits", "activation_bits", "calibration_count"),
        "qdrop_w6a6": (
            "weight_bits", "activation_bits", "steps",
            "calibration_count"),
        "brecq_w6a6": (
            "weight_bits", "activation_bits", "steps",
            "calibration_count"),
        "p3_t3_mixed_ptq": (
            "base_weight_bits", "base_activation_bits",
            "promotion_weight_bits", "promotion_activation_bits"),
    }
    if method not in expected_fields:
        raise ValueError("unsupported selected PTQ method: %s" % method)
    if tuple(method_config) != expected_fields[method]:
        raise ValueError("selected PTQ method field contract changed: %s" %
                         method)
    if method in ("rtn_w8a8", "rtn_w4a4"):
        if int(method_config["calibration_count"]) != 128:
            raise ValueError("selected RTN calibration count must equal 128")
        return build_contract_rtn_plan(
            contract,
            method,
            int(method_config["weight_bits"]),
            int(method_config["activation_bits"]),
            None,
        )
    if method == "p3_t3_mixed_ptq":
        assignment = load_p3_t3_assignment(
            p3_t3_assignment, contract, method_config)
        return build_contract_rtn_plan(
            contract,
            method,
            int(method_config["base_weight_bits"]),
            int(method_config["base_activation_bits"]),
            assignment,
        )
    bits = (
        int(method_config["weight_bits"]),
        int(method_config["activation_bits"]),
    )
    if bits != (6, 6):
        raise ValueError("selected reconstruction precision must be W6A6")
    if int(method_config["steps"]) <= 0:
        raise ValueError("selected reconstruction steps must be positive")
    if int(method_config["calibration_count"]) != 128:
        raise ValueError(
            "selected reconstruction calibration count must equal 128")
    owners = tuple(
        owner for block in contract.blocks for owner in block.activation_owners)
    return SelectedReconstructionPlan(
        method=method,
        model_name=contract.model_name,
        module_names=contract.weight_modules,
        activation_owners=owners,
        attention_edges=contract.attention_edges,
        concat_edges=contract.concat_edges,
        protected_modules=contract.protected_modules,
        weight_bits=bits[0],
        activation_bits=bits[1],
    )


def run_selected_ptq_matrix(
        *, model_config, method_hyperparameters, p3_t3_assignment,
        output, calibration_identity, evaluation_identity, dependencies):
    """Execute only the selected PTQ methods through fresh official runtimes."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    manifests = {}
    manifest_paths = {}
    contracts = {}
    for method in SELECTED_PTQ_METHODS:
        method_config = method_hyperparameters[method]
        runtime = dependencies.runtime_factory(model_config)
        try:
            model = runtime.build_model(runtime.device)
            contract = dependencies.contract_builder(
                runtime.model_name, model)
            if contract.model_name != runtime.model_name:
                raise ValueError(
                    "selected PTQ contract differs from runtime model")
            plan = _method_plan(
                method, method_config, contract, p3_t3_assignment)
            manifest = dependencies.method_executor(
                runtime,
                model,
                contract,
                plan,
                method,
                root / method,
                method_config,
                calibration_identity,
                evaluation_identity,
            )
            manifest_paths[method] = Path(manifest).resolve()
            manifests[method] = validate_hard_deployment_manifest(
                manifest_paths[method], method, contract,
                calibration_identity, evaluation_identity)
            _validate_manifest_plan(manifests[method], plan)
            contracts[method] = contract
        finally:
            runtime.close()
    qdrop_contract = contracts["qdrop_w6a6"]
    if qdrop_contract != contracts["brecq_w6a6"]:
        raise ValueError("QDrop and BRECQ contracts differ")
    validate_selected_reconstruction_pair(
        manifest_paths["qdrop_w6a6"],
        manifest_paths["brecq_w6a6"],
        qdrop_contract,
        calibration_identity,
        evaluation_identity,
    )
    matrix_path = root / "selected_ptq_matrix.json"
    matrix_path.write_text(json.dumps({
        "format_version": 1,
        "model": model_config.model,
        "methods": list(SELECTED_PTQ_METHODS),
        "calibration_identity": str(calibration_identity),
        "evaluation_identity": str(evaluation_identity),
        "hard_deployment_manifests": dict(
            (method, str(manifest_paths[method]))
            for method in SELECTED_PTQ_METHODS),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifests


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the exact official-model selected PTQ matrix")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_ORDER, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--qdrop-config", type=Path, required=True)
    parser.add_argument("--calibration-indices", type=Path, required=True)
    parser.add_argument("--evaluation-protocol", type=Path, required=True)
    parser.add_argument("--p3-t3-assignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    folding = parser.add_mutually_exclusive_group(required=True)
    folding.add_argument(
        "--fold-conv-bn", dest="fold_conv_bn", action="store_true")
    folding.add_argument(
        "--skip-conv-bn-fold", dest="fold_conv_bn",
        action="store_false")
    parser.add_argument("--fold-max-error", type=float, required=True)
    parser.add_argument(
        "--joint-clip-factors", type=float, nargs="+", required=True)
    parser.add_argument("--joint-search-rounds", type=int, required=True)
    parser.add_argument(
        "--joint-cache-sample-limit", type=int, required=True)
    parser.add_argument("--joint-cache-byte-limit", type=int, required=True)
    return parser


def run_cli(argv):
    args = build_parser().parse_args(tuple(argv))
    selected = load_selected_quantization_config(args.config)
    models = tuple(
        model for model in selected.models if model.model == args.model)
    if len(models) != 1:
        raise ValueError("selected PTQ model entry is not unique")
    model_config = models[0]
    if str(args.device) != model_config.device:
        raise ValueError("explicit device differs from selected model device")
    if args.output.exists():
        raise FileExistsError("selected PTQ output already exists: %s" %
                              args.output)
    if not args.output.parent.is_dir():
        raise FileNotFoundError(
            "selected PTQ output parent is missing: %s" %
            args.output.parent)
    if not args.p3_t3_assignment.is_file():
        raise FileNotFoundError(
            "P3/T3 assignment is missing: %s" % args.p3_t3_assignment)
    clip_factors = tuple(float(value) for value in args.joint_clip_factors)
    if not math.isfinite(float(args.fold_max_error)) or \
            float(args.fold_max_error) < 0.0 or not clip_factors or \
            any(not math.isfinite(value) or value <= 0.0
                for value in clip_factors):
        raise ValueError("selected RTN calibration values are invalid")
    if int(args.joint_search_rounds) <= 0 or \
            int(args.joint_cache_sample_limit) <= 0 or \
            int(args.joint_cache_byte_limit) <= 0:
        raise ValueError("selected RTN calibration limits must be positive")
    p3_method = selected.method_hyperparameters["p3_t3_mixed_ptq"]
    settings = HardDeploymentSettings(
        device=model_config.device,
        calibration_metadata=model_config.calibration_metadata,
        calibration_count=int(model_config.calibration_count),
        evaluation_indices=tuple(model_config.evaluation_indices),
        base_weight_bits=int(p3_method["base_weight_bits"]),
        base_activation_bits=int(p3_method["base_activation_bits"]),
        promotion_weight_bits=int(p3_method["promotion_weight_bits"]),
        promotion_activation_bits=int(
            p3_method["promotion_activation_bits"]),
        fold_conv_bn=bool(args.fold_conv_bn),
        fold_max_error=float(args.fold_max_error),
        joint_clip_factors=clip_factors,
        joint_search_rounds=int(args.joint_search_rounds),
        joint_cache_sample_limit=int(args.joint_cache_sample_limit),
        joint_cache_byte_limit=int(args.joint_cache_byte_limit),
    )
    qdrop_config = load_qdrop_config(args.qdrop_config)
    qdrop_args = Namespace(
        config=args.qdrop_config,
        run_dir=model_config.run_dir,
        checkpoint=model_config.checkpoint,
        data_root=model_config.data_root,
        model=model_config.model,
        device=args.device,
        algorithm="qdrop",
        precision="W6A6",
        phase="formal",
        seed=int(qdrop_config.formal.evaluation_seed),
        calibration_indices=args.calibration_indices,
        calibration_metadata=model_config.calibration_metadata,
        evaluation_protocol=args.evaluation_protocol,
        out_dir=args.output,
    )
    from scripts.run_nyu_qdrop_reconstruction import (
        load_reconstruction_protocol,
    )
    split, protocol = load_reconstruction_protocol(
        qdrop_args, qdrop_config)
    if tuple(protocol["evaluation_indices"]) != tuple(
            model_config.evaluation_indices):
        raise ValueError(
            "reconstruction evaluation identities differ from model config")
    calibration_identity = protocol["calibration_identity_sha256"]
    evaluation_identity = protocol["evaluation_identity_sha256"]
    dependencies = SelectedPTQDependencies(
        runtime_factory=NYUModelRuntime.from_config,
        contract_builder=build_model_quantization_contract,
        method_executor=ProductionMethodExecutor(
            qdrop_args=qdrop_args,
            qdrop_config=qdrop_config,
            qdrop_split=split,
            qdrop_protocol=protocol,
            rtn_settings=settings,
            selected_device=model_config.device,
        ),
    )
    return run_selected_ptq_matrix(
        model_config=model_config,
        method_hyperparameters=selected.method_hyperparameters,
        p3_t3_assignment=args.p3_t3_assignment,
        output=args.output,
        calibration_identity=calibration_identity,
        evaluation_identity=evaluation_identity,
        dependencies=dependencies,
    )


def main():
    manifests = run_cli(tuple(sys.argv[1:]))
    print("completed selected PTQ methods: %s" %
          ",".join(manifests))


if __name__ == "__main__":
    main()
