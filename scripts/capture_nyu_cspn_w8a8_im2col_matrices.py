#!/usr/bin/env python3
"""Capture complete native Conv tensors for CSPN PA-W8A8 Im2Col plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_nyu_cspn_w8a8_im2col as base  # noqa: E402
from spn_quant.im2col_matrix_visualization import (  # noqa: E402
    FullMatrixCaptureRecorder,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--fold-max-error", type=float, required=True)
    return parser.parse_args(argv)


def _read_csv(path: Path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("capture source table is empty: %s" % path)
    return rows


def select_worst_samples(experiment: Path):
    experiment = Path(experiment)
    manifest = json.loads(
        (experiment / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["model"] != "cspn" or \
            manifest["configuration"] != "PA_W8A8":
        raise ValueError("matrix capture requires CSPN PA_W8A8 source")
    modules = tuple(str(name) for name in manifest[
        "selected_plot_modules"])
    if not modules or len(set(modules)) != len(modules):
        raise ValueError("selected module identities changed")
    source = _read_csv(experiment / "top_spatial_tokens.csv")
    selected = {}
    for module in modules:
        candidates = [row for row in source if row["module"] == module]
        if not candidates:
            raise ValueError("selected module token coverage is incomplete")
        ranked = sorted(
            candidates,
            key=lambda row: (
                -float(row["local_output_error"]),
                int(row["sample_index"])))
        selected[module] = int(ranked[0]["sample_index"])
    if set(selected) != set(modules):
        raise ValueError("selected module token coverage changed")
    return selected


def _selection_by_sample(selected):
    by_sample = {}
    for module, sample_index in selected.items():
        sample_index = int(sample_index)
        if sample_index not in by_sample:
            by_sample[sample_index] = set()
        by_sample[sample_index].add(str(module))
    return by_sample


def _module_directory(root: Path, module: str) -> Path:
    return root.joinpath("captures", *str(module).split("."))


def _source_file_digests(experiment: Path):
    return dict(
        (path.name, base.file_sha256(path))
        for path in sorted(Path(experiment).iterdir()) if path.is_file())


def validate_architecture_identity(actual, expected) -> None:
    for field in ("architecture", "iteration", "from_scratch"):
        if actual[field] != expected[field]:
            raise ValueError("source architecture identity changed")
    actual_provenance = actual["model_provenance"]
    expected_provenance = expected["model_provenance"]
    for field in (
            "model_class", "model_module", "source_path", "source_root",
            "source_sha256"):
        if actual_provenance[field] != expected_provenance[field]:
            raise ValueError("model source identity changed")
    if actual_provenance["checkpoint_sha256"] != \
            expected_provenance["checkpoint_sha256"] or \
            actual_provenance["checkpoint_load"] != \
            expected_provenance["checkpoint_load"]:
        raise ValueError("model checkpoint identity changed")


def _validate_source(experiment: Path, manifest, selected) -> None:
    if len(selected) != 6:
        raise ValueError("production matrix capture requires six modules")
    quantization = manifest["quantization"]
    if int(quantization["weight_bits"]) != 8 or \
            int(quantization["activation_bits"]) != 8 or \
            quantization["activation_mode"] != "uniform" or \
            quantization["propagation"] != base.PROPAGATION_W8A8_Q13:
        raise ValueError("source quantization contract changed")
    checkpoint = Path(manifest["checkpoint"])
    if base.file_sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("source checkpoint digest changed")
    protocol = manifest["protocol"]
    metadata = Path(protocol["metadata_path"])
    if base.file_sha256(metadata) != protocol["metadata_sha256"]:
        raise ValueError("source metadata digest changed")
    module_rows = _read_csv(experiment / "module_manifest.csv")
    status = dict((row["module"], row["status"]) for row in module_rows)
    for module in selected:
        if module not in status or status[module] != "collected_conv2d":
            raise ValueError("selected capture module is not active Conv2d")


def main(argv=None):
    args = parse_args(argv)
    if not args.device.startswith("cuda"):
        raise ValueError("CSPN matrix capture requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if float(args.fold_max_error) <= 0.0:
        raise ValueError("fold max error must be positive")
    experiment = Path(args.experiment_dir).resolve()
    output = experiment / "full_matrix_visualization"
    if output.exists():
        raise FileExistsError(str(output))
    manifest = json.loads(
        (experiment / "run_manifest.json").read_text(encoding="utf-8"))
    selected = select_worst_samples(experiment)
    _validate_source(experiment, manifest, selected)
    source_digests = _source_file_digests(experiment)
    checkpoint = Path(manifest["checkpoint"]).resolve()
    data_root = Path(manifest["data_root"]).resolve()
    source_protocol = manifest["protocol"]
    metadata_path = Path(source_protocol["metadata_path"])
    protocol = base.load_protocol(checkpoint, metadata_path)
    if list(protocol["calibration_indices"]) != \
            source_protocol["calibration_indices"] or \
            list(protocol["evaluation_indices"]) != \
            source_protocol["evaluation_indices"] or \
            int(protocol["seed"]) != int(source_protocol["seed"]):
        raise ValueError("source sampling protocol changed")
    output.mkdir()
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False
    saved_args = base._saved_args(checkpoint, data_root, protocol["seed"])
    base.enter_official_cspn_root(data_root)
    model, architecture = base.build_model(saved_args, checkpoint, device)
    validate_architecture_identity(architecture, manifest["architecture"])
    trainset = base.rtn.calibration_dataset(saved_args)
    valset = base.rtn.evaluation_dataset(saved_args)
    calibration_indices = protocol["calibration_indices"]
    evaluation_indices = set(protocol["evaluation_indices"])
    if not set(selected.values()).issubset(evaluation_indices):
        raise ValueError("selected capture sample is outside evaluation set")
    preparation_sample = base.rtn.seeded_sample(
        trainset, calibration_indices[0], saved_args.seed)
    preparation_args = base._model_args(
        saved_args, preparation_sample, device)
    preparation = base.prepare_hardware_model(
        model, preparation_args, excluded_pairs=(("conv1_1", "bn1"),))
    if float(preparation["primary_max_abs_error"]) > \
            float(args.fold_max_error):
        raise RuntimeError("Conv-BN fold exceeds declared error threshold")
    propagation_outputs = set(
        base.propagation_projection_outputs("cspn", model))
    instrumentor = base.HardwareAlignedInstrumentor(
        model, base.cspn.cspn_quant_group,
        preparation["fused_relu_producers"],
        externally_owned_outputs=propagation_outputs)
    propagation = base.install_propagation_adapter("cspn", model)
    merge_adapter = base.CallIndexedConcatAdapter(model)
    groups = sorted(set(instrumentor.module_groups().values()))
    config = base.select_w8a8_configuration(groups)
    started = time.time()
    base._calibrate(
        model, saved_args, trainset, calibration_indices, device,
        instrumentor, propagation, merge_adapter)
    base.rtn.configure_quantized_model(
        config, instrumentor, propagation, None,
        propagation_outputs, "cspn")
    merge_adapter.freeze(8)
    merge_adapter.quantize()
    instrumentor.set_runtime_statistics(False)
    input_modules = set(
        key[0] for key in instrumentor.quantizers
        if isinstance(key, tuple) and key[1] == "input")
    if not set(selected).issubset(input_modules):
        raise ValueError("selected Conv input quantizer coverage changed")
    by_sample = _selection_by_sample(selected)
    recorder = FullMatrixCaptureRecorder(
        instrumentor.modules, instrumentor.original_weights, by_sample)
    rows = []
    with torch.no_grad():
        for rank, sample_index in enumerate(sorted(by_sample), 1):
            sample = base.rtn.seeded_sample(
                valset, sample_index, saved_args.seed)
            model_args = base._model_args(saved_args, sample, device)
            baseline = base.sweep.extract_pred(
                model(*model_args)).detach().cpu()
            recorder.begin_sample(sample_index)
            instrumentor.set_activation_recorder(recorder)
            prediction = base.sweep.extract_pred(
                model(*model_args)).detach().cpu()
            captures = recorder.end_sample()
            recorder.clear_sample()
            instrumentor.clear_activation_recorder()
            if not torch.equal(prediction, baseline):
                raise RuntimeError("matrix recorder changed W8A8 prediction")
            for module in sorted(captures):
                capture = captures[module]
                directory = _module_directory(output, module)
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / (
                    "sample_%05d.npz" % int(sample_index))
                capture.save(path)
                layout = capture.geometry.layout(module)
                height, width = layout.output_shape(
                    capture.reference_input)
                rows.append({
                    "module": module,
                    "sample_index": int(sample_index),
                    "path": str(path.relative_to(output)),
                    "input_shape": "x".join(
                        str(value) for value in capture.reference_input.shape),
                    "weight_shape": "x".join(
                        str(value) for value in capture.original_weight.shape),
                    "k_size": layout.k_size,
                    "token_count": height * width,
                    "activation_bits": capture.activation_bits,
                    "activation_unsigned": int(
                        capture.activation_unsigned),
                    "activation_scale_count": capture.activation_scale.numel(),
                })
            print("matrix capture %d/%d" %
                  (rank, len(by_sample)), flush=True)
    if {row["module"] for row in rows} != set(selected):
        raise RuntimeError("persisted matrix capture coverage changed")
    base.rtn.write_csv(output / "capture_manifest.csv", rows)
    base.rtn.write_json(output / "run_manifest.json", {
        "model": "cspn",
        "configuration": "PA_W8A8",
        "source_experiment": str(experiment),
        "source_file_digests": source_digests,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "metadata_sha256": source_protocol["metadata_sha256"],
        "selected_module_samples": selected,
        "capture_count": len(rows),
        "elapsed_seconds": time.time() - started,
    })
    instrumentor.close()
    propagation.close()
    merge_adapter.close()
    print("CSPN full Im2Col matrix capture complete", flush=True)


if __name__ == "__main__":
    main()
