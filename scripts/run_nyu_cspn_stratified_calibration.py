#!/usr/bin/env python3
"""Select and audit a train-only stratified CSPN calibration set."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_nyu_cspn_activation_resolution as base  # noqa: E402
from scripts import train_nyu_iteration_sweep as sweep  # noqa: E402
from scripts.export_nyu_predictions import (  # noqa: E402
    file_sha256,
    load_run_args,
    prepare_args,
)
from scripts.run_nyu_rtn_quantization import (  # noqa: E402
    calibration_dataset,
    seeded_sample,
    write_csv,
    write_json,
)
from spn_quant.calibration_selection import (  # noqa: E402
    FeatureSchema,
    activation_range_coverage,
    build_disjoint_splits,
    descriptor_coverage,
    deterministic_kmedoids,
    fit_robust_normalizer,
    greedy_kcenter_features,
    grouped_pairwise_distance,
    nearest_distance_summary,
    raw_descriptor,
    select_tail_cover,
)

import data_transform  # noqa: E402


CALIBRATION_SAMPLES = 128
AUDIT_SAMPLES = 512
CANDIDATE_SAMPLES = 1024
CANDIDATE_TAIL_SAMPLES = 256
FINAL_TAIL_SAMPLES = 32
RANDOM_BASELINES = 16
ACTIVATION_EPSILON = 1e-12


@dataclass(frozen=True)
class ActivationOwner:
    identity: str
    module: str
    call_index: int
    kind: str


ACTIVATION_OWNERS = (
    ActivationOwner("encoder_stem_conv", "conv1_1", 0, "output"),
    ActivationOwner("encoder_stem_relu", "relu", 0, "relu_output"),
    ActivationOwner(
        "encoder_layer1_relu", "layer1.0.relu", 1, "relu_output"),
    ActivationOwner(
        "decoder_layer2_fusion", "gud_up_proj_layer2.sc_conv1", 0,
        "output"),
    ActivationOwner(
        "decoder_layer4_fusion", "gud_up_proj_layer4.sc_conv1", 0,
        "output"),
    ActivationOwner(
        "decoder_layer4_relu", "gud_up_proj_layer4.relu", 0,
        "relu_output"),
)

RAW_FEATURE_NAMES = (
    "depth_mean", "depth_p50", "depth_p95", "depth_max",
    "depth_valid_ratio", "rgb_luminance_mean", "rgb_luminance_std",
    "rgb_contrast", "rgb_edge_density", "sparse_valid_count",
    "sparse_quadrant_0", "sparse_quadrant_1", "sparse_quadrant_2",
    "sparse_quadrant_3", "sparse_grid_occupancy",
    "sparse_centroid_spread",
)
RAW_FEATURE_GROUPS = (
    "depth", "depth", "depth", "depth", "depth",
    "rgb", "rgb", "rgb", "rgb",
    "diagnostic", "sparse", "sparse", "sparse", "sparse", "sparse",
    "sparse",
)
RAW_FEATURE_DIAGNOSTIC = tuple(
    name == "sparse_valid_count" for name in RAW_FEATURE_NAMES)
RAW_SCHEMA = FeatureSchema(
    RAW_FEATURE_NAMES, RAW_FEATURE_GROUPS, RAW_FEATURE_DIAGNOSTIC)


def activation_feature_names():
    return tuple(
        "%s_%s" % (owner.identity, statistic)
        for owner in ACTIVATION_OWNERS
        for statistic in ("p99", "max", "p99_over_max",
                          "channel_imbalance"))


def combined_schema():
    activation_names = activation_feature_names()
    return FeatureSchema(
        names=RAW_FEATURE_NAMES + activation_names,
        groups=RAW_FEATURE_GROUPS + tuple(
            "activation" for name in activation_names),
        diagnostic=RAW_FEATURE_DIAGNOSTIC + tuple(
            False for name in activation_names))


def artifact_filenames():
    return (
        "calibration_indices.json", "audit_indices.json",
        "raw_descriptors.csv", "activation_descriptors.csv",
        "candidate_selection.csv", "calibration_selection.csv",
        "descriptor_coverage.csv", "distance_coverage.csv",
        "activation_range_coverage.csv", "metadata.json",
        "coverage_report.md",
    )


class CspnDescriptorDataset(Dataset):
    """Official CSPN train transform with retained pre-normalization RGB."""

    def __init__(self, csv_file, root_dir, n_sample, sample_seed):
        self.root_dir = Path(root_dir)
        self.n_sample = int(n_sample)
        self.sample_seed = int(sample_seed)
        with Path(csv_file).open("r", encoding="utf-8") as stream:
            self.samples = tuple(
                row["Name"] for row in csv.DictReader(stream))
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4)
        self.normalize = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        index = int(index)
        np.random.seed(self.sample_seed + index)
        torch.manual_seed(self.sample_seed + index)
        with h5py.File(str(self.root_dir / self.samples[index]), "r") as source:
            rgb = Image.fromarray(
                source["rgb"][:].transpose(1, 2, 0), mode="RGB")
            depth = Image.fromarray(
                source["depth"][:].astype("float32"), mode="F")
        scale = np.random.uniform(1.0, 1.5)
        size = int(240 * scale)
        degree = np.random.uniform(-5.0, 5.0)
        resize = transforms.Resize(size)
        rotate = data_transform.Rotation(degree)
        crop = transforms.CenterCrop((228, 304))
        rgb = crop(self.color_jitter(rotate(resize(rgb))))
        depth = crop(rotate(resize(depth)))
        if np.random.uniform() < 0.5:
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        raw_rgb = transforms.ToTensor()(rgb)
        depth_tensor = data_transform.ToTensor()(depth).div(float(scale))
        model_rgb = transforms.ToTensor()(
            transforms.ToPILImage()(self.normalize(raw_rgb)))

        pixels = depth_tensor.shape[1] * depth_tensor.shape[2]
        torch.bernoulli(torch.ones_like(depth_tensor) *
                        (self.n_sample / float(pixels)))
        sparse = sweep.create_sparse_depth(depth_tensor, self.n_sample)
        return {
            "sample_index": index,
            "raw_rgb": raw_rgb,
            "depth": depth_tensor,
            "sparse": sparse,
            "rgbd": torch.cat((model_rgb, sparse), dim=0),
            "cspn_preprocessed": True,
        }


class RawDescriptorScanDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        row = raw_descriptor(
            sample["raw_rgb"], sample["depth"], sample["sparse"])
        values = torch.tensor(
            [row[name] for name in RAW_FEATURE_NAMES], dtype=torch.float64)
        return int(sample["sample_index"]), values


def _activation_descriptor(tensor):
    if not torch.is_tensor(tensor) or tensor.ndim != 4 or tensor.numel() == 0:
        raise ValueError("activation must be a nonempty NCHW tensor")
    values = tensor.detach().float()
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("activation must be finite")
    magnitude = values.abs().reshape(-1)
    maximum = float(magnitude.max().item())
    p99 = float(torch.quantile(magnitude, 0.99).item())
    channel_rms = torch.sqrt(values.square().mean(dim=(0, 2, 3)))
    imbalance = float((
        channel_rms.max() /
        (channel_rms.mean() + ACTIVATION_EPSILON)).item())
    return {
        "p99": p99,
        "max": maximum,
        "p99_over_max": p99 / (maximum + ACTIVATION_EPSILON),
        "channel_imbalance": imbalance,
    }


class ActivationDescriptorCollector(object):
    def __init__(self, model, owners):
        self.model = model
        self.owners = tuple(owners)
        if not self.owners or len(set(
                owner.identity for owner in self.owners)) != len(self.owners):
            raise ValueError("activation owners must be unique and nonempty")
        modules = dict(model.named_modules())
        missing = set(owner.module for owner in self.owners) - set(modules)
        if missing:
            raise ValueError("activation owner modules are missing: %s" %
                             sorted(missing))
        self.calls = {}
        self.captured = {}
        self.handles = [
            model.register_forward_pre_hook(self._reset)
        ]
        owners_by_module = {}
        for owner in self.owners:
            if owner.module not in owners_by_module:
                owners_by_module[owner.module] = []
            owners_by_module[owner.module].append(owner)
        for name in sorted(owners_by_module):
            self.handles.append(modules[name].register_forward_hook(
                self._hook(name, tuple(owners_by_module[name]))))

    def _reset(self, module, inputs):
        del module, inputs
        self.calls = {}
        self.captured = {}

    def _hook(self, name, owners):
        def hook(module, inputs, output):
            del module, inputs
            call_index = self.calls[name] if name in self.calls else 0
            self.calls[name] = call_index + 1
            for owner in owners:
                if owner.call_index != call_index:
                    continue
                if owner.identity in self.captured:
                    raise RuntimeError(
                        "activation owner captured more than once: %s" %
                        owner.identity)
                if not torch.is_tensor(output):
                    raise TypeError("activation owner output must be a tensor")
                self.captured[owner.identity] = output
            return None
        return hook

    def capture(self, sample_index, model_args):
        with torch.no_grad():
            self.model(*model_args)
        expected = set(owner.identity for owner in self.owners)
        actual = set(self.captured)
        if actual != expected:
            raise RuntimeError(
                "activation captures are missing: %s" %
                sorted(expected - actual))
        row = {"sample_index": int(sample_index)}
        for owner in self.owners:
            descriptor = _activation_descriptor(
                self.captured[owner.identity])
            for name in ("p99", "max", "p99_over_max",
                         "channel_imbalance"):
                row["%s_%s" % (owner.identity, name)] = descriptor[name]
        return row

    def close(self):
        for handle in self.handles:
            handle.remove()


def _rows_to_matrix(rows, indices, names):
    matrix = np.asarray([
        [rows[int(index)][name] for name in names]
        for index in indices
    ], dtype=np.float64)
    if matrix.shape != (len(indices), len(names)) or \
            not np.isfinite(matrix).all():
        raise ValueError("descriptor matrix coverage is invalid")
    return matrix


def _validate_dataset_parity(
        descriptor_dataset, official_dataset, indices, sample_seed):
    for index in indices:
        descriptor = descriptor_dataset[int(index)]
        official = seeded_sample(
            official_dataset, int(index), int(sample_seed))
        if not torch.equal(descriptor["depth"], official["depth"]):
            raise RuntimeError(
                "official CSPN depth parity failed: %d" % index)
        if not torch.equal(
                descriptor["rgbd"][:3], official["rgbd"][:3]):
            raise RuntimeError(
                "official CSPN RGB parity failed: %d" % index)
        if not torch.equal(
                descriptor["sparse"], official["rgbd"][3:4]):
            raise RuntimeError(
                "official CSPN sparse-depth parity failed: %d" % index)


def _scan_raw_descriptors(dataset, workers):
    loader = DataLoader(
        RawDescriptorScanDataset(dataset), batch_size=1, shuffle=False,
        num_workers=int(workers), persistent_workers=int(workers) > 0)
    rows = {}
    for rank, (index, values) in enumerate(loader, 1):
        sample_index = int(index.item())
        row = {"sample_index": sample_index}
        vector = values[0].numpy()
        for position, name in enumerate(RAW_FEATURE_NAMES):
            row[name] = float(vector[position])
        rows[sample_index] = row
        if rank % 256 == 0 or rank == len(dataset):
            print("CSPN raw descriptor scan %d/%d" %
                  (rank, len(dataset)), flush=True)
    if len(rows) != len(dataset):
        raise RuntimeError("raw descriptor scan is incomplete")
    return rows


def _collect_activation_descriptors(
        collector, model, saved_args, dataset, indices, device, existing):
    rows = dict(existing)
    pending = tuple(
        int(index) for index in indices if int(index) not in rows)
    for rank, index in enumerate(pending, 1):
        sample = dataset[index]
        model_args = base._model_args(saved_args, sample, device)
        rows[index] = collector.capture(index, model_args)
        if rank % 32 == 0 or rank == len(pending):
            print("CSPN activation descriptor scan %d/%d" %
                  (rank, len(pending)), flush=True)
    return rows


def _tail_conditions_for_sample(values, names, low, high):
    conditions = []
    for position, name in enumerate(names):
        if values[position] <= low[position]:
            conditions.append("%s:low" % name)
        if values[position] >= high[position]:
            conditions.append("%s:high" % name)
    return tuple(conditions)


def _selection_rows(
        indices, transformed, names, tail, medoid_clusters, stage):
    tail_position = dict(
        (index, position) for position, index in enumerate(
            tail.selected_indices))
    rows = []
    position_by_index = dict(
        (int(index), position) for position, index in enumerate(indices))
    ordered = list(tail.selected_indices) + [
        index for index in medoid_clusters if index not in tail_position]
    for order, index in enumerate(ordered, 1):
        position = position_by_index[int(index)]
        if index in tail_position:
            tail_order = tail_position[index]
            reason = tail.selection_reasons[tail_order]
            cluster = ""
        else:
            reason = "medoid" if stage == "final" else "kcenter"
            cluster = medoid_clusters[index]
        rows.append({
            "sample_index": int(index),
            "selection_order": order,
            "selection_reason": reason,
            "stage": stage,
            "cluster": cluster,
            "tail_conditions": ";".join(_tail_conditions_for_sample(
                transformed[position], names, tail.low, tail.high)),
        })
    return rows


def _write_report(path, accepted, distance_rows, activation_rows):
    stratified = next(
        row for row in distance_rows
        if row["configuration"] == "stratified_128")
    current = next(
        row for row in distance_rows
        if row["configuration"] == "current_random_128")
    stratified_uncovered = sum(
        row["audit_exceeds_calibration"] for row in activation_rows
        if row["configuration"] == "stratified_128")
    current_uncovered = sum(
        row["audit_exceeds_calibration"] for row in activation_rows
        if row["configuration"] == "current_random_128")
    lines = (
        "# CSPN Stratified Calibration Coverage\n\n"
        "- Accepted: `%s`\n"
        "- Stratified nearest-distance p95: `%.8f`\n"
        "- Current random nearest-distance p95: `%.8f`\n"
        "- Stratified uncovered activation maxima: `%d`\n"
        "- Current random uncovered activation maxima: `%d`\n" % (
            str(bool(accepted)).lower(), stratified["nearest_p95"],
            current["nearest_p95"], stratified_uncovered,
            current_uncovered))
    path.write_text(lines, encoding="utf-8")


def _serialize_rows(rows):
    return [dict(row) for row in rows]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--baseline-seed", type=int, required=True)
    parser.add_argument("--audit-seed", type=int, required=True)
    parser.add_argument(
        "--calibration-samples", type=int,
        choices=(CALIBRATION_SAMPLES,), required=True)
    parser.add_argument(
        "--audit-samples", type=int,
        choices=(AUDIT_SAMPLES,), required=True)
    parser.add_argument(
        "--candidate-samples", type=int,
        choices=(CANDIDATE_SAMPLES,), required=True)
    parser.add_argument(
        "--candidate-tail-samples", type=int,
        choices=(CANDIDATE_TAIL_SAMPLES,), required=True)
    parser.add_argument(
        "--final-tail-samples", type=int,
        choices=(FINAL_TAIL_SAMPLES,), required=True)
    parser.add_argument(
        "--random-baseline-seeds", type=int, nargs=RANDOM_BASELINES,
        required=True)
    parser.add_argument("--parity-samples", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("stratified CSPN selection requires CUDA")
    if args.parity_samples <= 0 or args.workers < 0:
        raise ValueError("parity samples and workers are invalid")
    if len(set(args.random_baseline_seeds)) != RANDOM_BASELINES:
        raise ValueError("random baseline seeds must be unique")
    started = time.time()
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False

    run_dir = Path(args.run_dir)
    checkpoint = Path(args.checkpoint)
    saved_args = prepare_args(load_run_args(run_dir), args)
    saved_args.data_root = args.data_root
    if saved_args.model != "cspn":
        raise ValueError("stratified calibration requires model=cspn")
    descriptor_dataset = CspnDescriptorDataset(
        saved_args.train_list, args.data_root, saved_args.n_sample,
        args.baseline_seed)
    splits = build_disjoint_splits(
        len(descriptor_dataset), args.calibration_samples,
        args.baseline_seed, args.audit_samples, args.audit_seed,
        tuple(args.random_baseline_seeds))
    if len(splits.eligible_indices) != \
            len(descriptor_dataset) - args.audit_samples:
        raise RuntimeError("eligible train split count is invalid")

    official_dataset = calibration_dataset(saved_args)
    parity_indices = tuple(sorted(splits.eligible_indices))[
        :args.parity_samples]
    _validate_dataset_parity(
        descriptor_dataset, official_dataset, parity_indices,
        args.baseline_seed)
    raw_rows = _scan_raw_descriptors(descriptor_dataset, args.workers)

    eligible_indices = np.asarray(splits.eligible_indices, dtype=np.int64)
    eligible_raw = _rows_to_matrix(
        raw_rows, eligible_indices, RAW_FEATURE_NAMES)
    raw_normalizer = fit_robust_normalizer(eligible_raw, RAW_SCHEMA)
    eligible_transformed = raw_normalizer.transform(eligible_raw)
    candidate_tail = select_tail_cover(
        eligible_indices, eligible_transformed, raw_normalizer.names,
        args.candidate_tail_samples)
    candidate_indices = greedy_kcenter_features(
        eligible_indices, eligible_transformed, raw_normalizer.groups,
        args.candidate_samples, candidate_tail.selected_indices)
    candidate_additions = dict(
        (index, "") for index in candidate_indices
        if index not in set(candidate_tail.selected_indices))
    candidate_rows = _selection_rows(
        eligible_indices, eligible_transformed, raw_normalizer.names,
        candidate_tail, candidate_additions, "candidate")

    model, architecture, load_report = base._load_cspn(
        saved_args, checkpoint, device)
    collector = ActivationDescriptorCollector(model, ACTIVATION_OWNERS)
    activation_rows = _collect_activation_descriptors(
        collector, model, saved_args, descriptor_dataset,
        candidate_indices, device, {})

    schema = combined_schema()
    candidate_raw = _rows_to_matrix(
        raw_rows, candidate_indices, RAW_FEATURE_NAMES)
    candidate_activation = _rows_to_matrix(
        activation_rows, candidate_indices, activation_feature_names())
    candidate_combined = np.concatenate(
        (candidate_raw, candidate_activation), axis=1)
    combined_normalizer = fit_robust_normalizer(candidate_combined, schema)
    candidate_transformed = combined_normalizer.transform(candidate_combined)
    final_tail = select_tail_cover(
        np.asarray(candidate_indices), candidate_transformed,
        combined_normalizer.names, args.final_tail_samples)
    tail_set = set(final_tail.selected_indices)
    non_tail_positions = np.asarray([
        position for position, index in enumerate(candidate_indices)
        if index not in tail_set
    ], dtype=np.int64)
    non_tail_indices = np.asarray([
        candidate_indices[position] for position in non_tail_positions
    ], dtype=np.int64)
    non_tail_features = candidate_transformed[non_tail_positions]
    non_tail_distance = grouped_pairwise_distance(
        non_tail_features, combined_normalizer.groups)
    medoids = deterministic_kmedoids(
        non_tail_indices, non_tail_distance,
        args.calibration_samples - args.final_tail_samples)
    final_indices = tuple(final_tail.selected_indices) + \
        tuple(medoids.medoid_indices)
    if len(final_indices) != args.calibration_samples or \
            len(set(final_indices)) != args.calibration_samples:
        raise RuntimeError("final calibration selection count is invalid")
    medoid_clusters = {}
    assignment_by_index = dict(
        (int(index), medoids.assignments[position])
        for position, index in enumerate(non_tail_indices))
    for index in medoids.medoid_indices:
        medoid_clusters[index] = assignment_by_index[index]
    final_rows = _selection_rows(
        np.asarray(candidate_indices), candidate_transformed,
        combined_normalizer.names, final_tail, medoid_clusters, "final")

    profile_indices = set(candidate_indices)
    profile_indices.update(splits.audit_indices)
    profile_indices.update(splits.baseline_indices)
    for baseline in splits.random_baselines:
        profile_indices.update(baseline)
    activation_rows = _collect_activation_descriptors(
        collector, model, saved_args, descriptor_dataset,
        tuple(sorted(profile_indices)), device, activation_rows)
    collector.close()

    configurations = {
        "stratified_128": tuple(final_indices),
        "current_random_128": tuple(splits.baseline_indices),
    }
    for position, baseline in enumerate(splits.random_baselines):
        configurations["random_%02d" % position] = tuple(baseline)
    audit_indices = tuple(splits.audit_indices)
    audit_raw = _rows_to_matrix(raw_rows, audit_indices, RAW_FEATURE_NAMES)
    audit_activation = _rows_to_matrix(
        activation_rows, audit_indices, activation_feature_names())
    audit_combined = np.concatenate((audit_raw, audit_activation), axis=1)
    audit_transformed = combined_normalizer.transform(audit_combined)
    active_positions = combined_normalizer.positions
    audit_actual = audit_combined[:, active_positions]
    activation_max_names = tuple(
        "%s_max" % owner.identity for owner in ACTIVATION_OWNERS)
    activation_max_positions = tuple(
        schema.names.index(name) for name in activation_max_names)

    coverage_rows = []
    distance_rows = []
    activation_coverage_rows = []
    for configuration in sorted(configurations):
        indices = configurations[configuration]
        raw = _rows_to_matrix(raw_rows, indices, RAW_FEATURE_NAMES)
        activation = _rows_to_matrix(
            activation_rows, indices, activation_feature_names())
        combined = np.concatenate((raw, activation), axis=1)
        transformed = combined_normalizer.transform(combined)
        actual = combined[:, active_positions]
        current_coverage = descriptor_coverage(
            configuration, actual, audit_actual,
            combined_normalizer.names, combined_normalizer.groups)
        for column, row in enumerate(current_coverage):
            row["wasserstein"] /= float(combined_normalizer.iqr[column])
        coverage_rows.extend(current_coverage)
        distance_rows.append(nearest_distance_summary(
            configuration, transformed, audit_transformed,
            combined_normalizer.groups))
        activation_coverage_rows.extend(activation_range_coverage(
            configuration, combined[:, activation_max_positions],
            audit_combined[:, activation_max_positions],
            activation_max_names))

    stratified_distance = next(
        row for row in distance_rows
        if row["configuration"] == "stratified_128")
    random_distance = [
        row["nearest_p95"] for row in distance_rows
        if row["configuration"].startswith("random_")]
    current_uncovered = sum(
        row["audit_exceeds_calibration"] for row in activation_coverage_rows
        if row["configuration"] == "current_random_128")
    stratified_uncovered = sum(
        row["audit_exceeds_calibration"] for row in activation_coverage_rows
        if row["configuration"] == "stratified_128")
    accepted = (
        len(final_tail.covered_conditions) ==
        2 * len(combined_normalizer.names) and
        stratified_distance["nearest_p95"] < float(np.mean(random_distance)) and
        stratified_uncovered <= current_uncovered and
        not (set(final_indices) & set(audit_indices)))

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "calibration_indices.json", {
        "indices": list(final_indices),
        "count": len(final_indices),
        "selection": "32_tail_96_kmedoids",
    })
    write_json(output / "audit_indices.json", {
        "indices": list(audit_indices),
        "count": len(audit_indices),
        "seed": args.audit_seed,
    })
    write_csv(
        output / "raw_descriptors.csv",
        [raw_rows[index] for index in sorted(raw_rows)],
        preferred=("sample_index",))
    write_csv(
        output / "activation_descriptors.csv",
        [activation_rows[index] for index in sorted(activation_rows)],
        preferred=("sample_index",))
    write_csv(output / "candidate_selection.csv", candidate_rows,
              preferred=("sample_index", "selection_order"))
    write_csv(output / "calibration_selection.csv", final_rows,
              preferred=("sample_index", "selection_order"))
    write_csv(output / "descriptor_coverage.csv", coverage_rows,
              preferred=("configuration", "group", "feature"))
    write_csv(output / "distance_coverage.csv", distance_rows,
              preferred=("configuration",))
    write_csv(
        output / "activation_range_coverage.csv",
        activation_coverage_rows,
        preferred=("configuration", "feature"))
    metadata = {
        "accepted": bool(accepted),
        "model": "cspn",
        "architecture": architecture,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_load": load_report,
        "train_list": str(Path(saved_args.train_list).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "dataset_samples": len(descriptor_dataset),
        "eligible_samples": len(splits.eligible_indices),
        "candidate_samples": len(candidate_indices),
        "calibration_samples": len(final_indices),
        "audit_samples": len(audit_indices),
        "baseline_seed": args.baseline_seed,
        "audit_seed": args.audit_seed,
        "random_baseline_seeds": list(args.random_baseline_seeds),
        "candidate_tail_samples": args.candidate_tail_samples,
        "final_tail_samples": args.final_tail_samples,
        "activation_owners": [
            {
                "identity": owner.identity,
                "module": owner.module,
                "call_index": owner.call_index,
                "kind": owner.kind,
            } for owner in ACTIVATION_OWNERS
        ],
        "current_uncovered_activation_maxima": current_uncovered,
        "stratified_uncovered_activation_maxima": stratified_uncovered,
        "random_mean_nearest_p95": float(np.mean(random_distance)),
        "stratified_nearest_p95": stratified_distance["nearest_p95"],
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "metadata.json", metadata)
    _write_report(
        output / "coverage_report.md", accepted,
        distance_rows, activation_coverage_rows)
    actual_artifacts = set(path.name for path in output.iterdir())
    if actual_artifacts != set(artifact_filenames()):
        raise RuntimeError("stratified calibration artifact coverage mismatch")
    if not accepted:
        raise RuntimeError("stratified calibration coverage contract failed")


if __name__ == "__main__":
    main()
