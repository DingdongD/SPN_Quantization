"""Train-only descriptor and stratified calibration selection primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


LUMINANCE_WEIGHTS = (0.2126, 0.7152, 0.0722)


@dataclass(frozen=True)
class FeatureSchema:
    names: tuple[str, ...]
    groups: tuple[str, ...]
    diagnostic: tuple[bool, ...]

    def __post_init__(self):
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("feature names must be unique and nonempty")
        if len(self.groups) != len(self.names) or \
                len(self.diagnostic) != len(self.names):
            raise ValueError("feature schema columns must have equal length")


@dataclass(frozen=True)
class RobustNormalizer:
    names: tuple[str, ...]
    groups: tuple[str, ...]
    positions: tuple[int, ...]
    median: np.ndarray
    iqr: np.ndarray

    def transform(self, values):
        matrix = _finite_matrix(values, "normalizer input")
        if matrix.shape[1] <= max(self.positions):
            raise ValueError("normalizer input lacks declared features")
        selected = matrix[:, self.positions]
        transformed = (selected - self.median) / self.iqr
        if not np.isfinite(transformed).all():
            raise ValueError("normalized features must be finite")
        return transformed


@dataclass(frozen=True)
class TailSelection:
    selected_indices: tuple[int, ...]
    covered_conditions: tuple[str, ...]
    selection_reasons: tuple[str, ...]
    low: np.ndarray
    high: np.ndarray


@dataclass(frozen=True)
class KMedoidsResult:
    medoid_indices: tuple[int, ...]
    assignments: tuple[int, ...]


@dataclass(frozen=True)
class DisjointSplits:
    baseline_indices: tuple[int, ...]
    audit_indices: tuple[int, ...]
    eligible_indices: tuple[int, ...]
    random_baselines: tuple[tuple[int, ...], ...]


def _finite_tensor(tensor, name, rank):
    if not torch.is_tensor(tensor) or tensor.ndim != rank:
        raise ValueError("%s must be a rank-%d tensor" % (name, rank))
    if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("%s must be finite and nonempty" % name)
    return tensor.detach().float()


def _finite_matrix(values, name):
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("%s must be a nonempty matrix" % name)
    if not np.isfinite(matrix).all():
        raise ValueError("%s must be finite" % name)
    return matrix


def _luminance(rgb):
    weights = rgb.new_tensor(LUMINANCE_WEIGHTS).reshape(3, 1, 1)
    return (rgb * weights).sum(dim=0)


def _edge_density(luminance):
    kernels = luminance.new_tensor((
        ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0)),
        ((-1.0, -2.0, -1.0), (0.0, 0.0, 0.0), (1.0, 2.0, 1.0)),
    )).unsqueeze(1)
    gradients = F.conv2d(
        luminance.reshape(1, 1, *luminance.shape), kernels, padding=1)
    magnitude = torch.linalg.vector_norm(gradients, dim=1)
    return float((magnitude > 0.1).float().mean().item())


def raw_descriptor(rgb, depth, sparse):
    rgb = _finite_tensor(rgb, "RGB", 3)
    depth = _finite_tensor(depth, "depth", 3)
    sparse = _finite_tensor(sparse, "sparse depth", 3)
    if rgb.shape[0] != 3 or depth.shape[0] != 1 or sparse.shape[0] != 1:
        raise ValueError("RGB/depth channel counts are invalid")
    if tuple(rgb.shape[1:]) != tuple(depth.shape[1:]) or \
            tuple(depth.shape) != tuple(sparse.shape):
        raise ValueError("RGB/depth spatial shapes must match")
    if bool((rgb < 0.0).any().item()) or bool((rgb > 1.0).any().item()):
        raise ValueError("RGB values must be in [0, 1]")
    valid_depth = depth[depth > 0.0001]
    if valid_depth.numel() == 0:
        raise ValueError("depth must contain valid pixels")
    coordinates = torch.nonzero(sparse[0] > 0.0001, as_tuple=False)
    if int(coordinates.shape[0]) != 500:
        raise ValueError("sparse depth must contain exactly 500 valid points")

    height, width = depth.shape[1:]
    luminance = _luminance(rgb)
    quadrants = (
        (coordinates[:, 0] < height / 2.0) &
        (coordinates[:, 1] < width / 2.0),
        (coordinates[:, 0] < height / 2.0) &
        (coordinates[:, 1] >= width / 2.0),
        (coordinates[:, 0] >= height / 2.0) &
        (coordinates[:, 1] < width / 2.0),
        (coordinates[:, 0] >= height / 2.0) &
        (coordinates[:, 1] >= width / 2.0),
    )
    grid_y = torch.clamp(coordinates[:, 0] * 8 // height, max=7)
    grid_x = torch.clamp(coordinates[:, 1] * 8 // width, max=7)
    occupied = torch.unique(grid_y * 8 + grid_x).numel() / 64.0
    normalized_coordinates = torch.stack((
        coordinates[:, 0].float() / float(height),
        coordinates[:, 1].float() / float(width),
    ), dim=1)
    centered = normalized_coordinates - normalized_coordinates.mean(
        dim=0, keepdim=True)
    spread = torch.sqrt(centered.square().sum(dim=1).mean())

    row = {
        "depth_mean": float(valid_depth.mean().item()),
        "depth_p50": float(torch.quantile(valid_depth, 0.5).item()),
        "depth_p95": float(torch.quantile(valid_depth, 0.95).item()),
        "depth_max": float(valid_depth.max().item()),
        "depth_valid_ratio": valid_depth.numel() / float(depth.numel()),
        "rgb_luminance_mean": float(luminance.mean().item()),
        "rgb_luminance_std": float(luminance.std(unbiased=False).item()),
        "rgb_contrast": float(
            rgb.reshape(3, -1).std(dim=1, unbiased=False).mean().item()),
        "rgb_edge_density": _edge_density(luminance),
        "sparse_valid_count": float(coordinates.shape[0]),
        "sparse_grid_occupancy": float(occupied),
        "sparse_centroid_spread": float(spread.item()),
    }
    for index, mask in enumerate(quadrants):
        row["sparse_quadrant_%d" % index] = \
            float(mask.float().mean().item())
    if not all(np.isfinite(value) for value in row.values()):
        raise ValueError("raw descriptors must be finite")
    return row


def fit_robust_normalizer(values, schema):
    if not isinstance(schema, FeatureSchema):
        raise TypeError("schema must be FeatureSchema")
    matrix = _finite_matrix(values, "normalizer fit input")
    if matrix.shape[1] != len(schema.names):
        raise ValueError("normalizer columns do not match schema")
    positions = tuple(
        index for index, diagnostic in enumerate(schema.diagnostic)
        if not diagnostic)
    if not positions:
        raise ValueError("normalizer requires nondiagnostic features")
    selected = matrix[:, positions]
    median = np.quantile(selected, 0.5, axis=0)
    first = np.quantile(selected, 0.25, axis=0)
    third = np.quantile(selected, 0.75, axis=0)
    iqr = third - first
    if not np.isfinite(iqr).all() or np.any(iqr <= 0.0):
        raise ValueError("feature IQR must be finite and positive")
    return RobustNormalizer(
        names=tuple(schema.names[index] for index in positions),
        groups=tuple(schema.groups[index] for index in positions),
        positions=positions, median=median, iqr=iqr)


def grouped_pairwise_distance(values, groups):
    matrix = _finite_matrix(values, "distance input")
    groups = tuple(str(group) for group in groups)
    if len(groups) != matrix.shape[1]:
        raise ValueError("distance groups do not match features")
    unique_groups = tuple(dict.fromkeys(groups))
    distance = np.zeros((matrix.shape[0], matrix.shape[0]), dtype=np.float64)
    for group in unique_groups:
        positions = [index for index, current in enumerate(groups)
                     if current == group]
        difference = matrix[:, None, positions] - matrix[None, :, positions]
        distance += np.mean(difference * difference, axis=2)
    distance /= float(len(unique_groups))
    return distance.astype(np.float32)


def _distance_to_center(matrix, center, groups):
    unique_groups = tuple(dict.fromkeys(groups))
    distance = np.zeros(matrix.shape[0], dtype=np.float64)
    for group in unique_groups:
        positions = [index for index, current in enumerate(groups)
                     if current == group]
        difference = matrix[:, positions] - center[positions]
        distance += np.mean(difference * difference, axis=1)
    return distance / float(len(unique_groups))


def _validated_indices(indices, rows):
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if indices.size != rows or len(set(indices.tolist())) != rows:
        raise ValueError("sample indices must be unique and match rows")
    return indices


def select_tail_cover(indices, values, names, budget):
    matrix = _finite_matrix(values, "tail input")
    indices = _validated_indices(indices, matrix.shape[0])
    names = tuple(str(name) for name in names)
    budget = int(budget)
    if len(names) != matrix.shape[1]:
        raise ValueError("tail names do not match features")
    if budget <= 0 or budget > matrix.shape[0]:
        raise ValueError("tail budget is invalid")
    low = np.quantile(matrix, 0.05, axis=0)
    high = np.quantile(matrix, 0.95, axis=0)
    if np.any(high <= low):
        raise ValueError("tail thresholds must be ordered")
    memberships = np.concatenate(
        (matrix <= low.reshape(1, -1), matrix >= high.reshape(1, -1)),
        axis=1)
    conditions = tuple(
        ["%s:low" % name for name in names] +
        ["%s:high" % name for name in names])
    scale = high - low
    tail_distance = np.maximum(
        (low.reshape(1, -1) - matrix) / scale.reshape(1, -1), 0.0)
    tail_distance += np.maximum(
        (matrix - high.reshape(1, -1)) / scale.reshape(1, -1), 0.0)
    aggregate = tail_distance.sum(axis=1)
    selected = []
    reasons = []
    uncovered = set(range(len(conditions)))
    available = set(range(matrix.shape[0]))
    while uncovered and len(selected) < budget:
        ranked = sorted(
            available,
            key=lambda row: (
                -sum(bool(memberships[row, condition])
                     for condition in uncovered),
                -float(aggregate[row]),
                int(indices[row]),
            ))
        position = ranked[0]
        gain = [condition for condition in uncovered
                if memberships[position, condition]]
        if not gain:
            break
        selected.append(position)
        reasons.append("tail_cover")
        available.remove(position)
        uncovered.difference_update(gain)
    if uncovered:
        raise ValueError("tail conditions exceed selection budget")
    for position in sorted(
            available,
            key=lambda row: (-float(aggregate[row]), int(indices[row]))):
        if len(selected) == budget:
            break
        selected.append(position)
        reasons.append("tail_score")
    covered = tuple(
        condition for column, condition in enumerate(conditions)
        if any(memberships[position, column] for position in selected))
    return TailSelection(
        selected_indices=tuple(int(indices[position]) for position in selected),
        covered_conditions=covered,
        selection_reasons=tuple(reasons), low=low, high=high)


def _validated_distance(indices, distances):
    matrix = _finite_matrix(distances, "distance matrix")
    indices = _validated_indices(indices, matrix.shape[0])
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance matrix must be square")
    if np.any(matrix < 0.0) or not np.allclose(matrix, matrix.T):
        raise ValueError("distance matrix must be symmetric and nonnegative")
    if not np.allclose(np.diag(matrix), 0.0):
        raise ValueError("distance matrix diagonal must be zero")
    return indices, matrix


def greedy_kcenter(indices, distances, count, initial_indices=()):
    indices, matrix = _validated_distance(indices, distances)
    count = int(count)
    if count <= 0 or count > indices.size:
        raise ValueError("k-center count is invalid")
    positions = dict((int(index), position)
                     for position, index in enumerate(indices.tolist()))
    initial = tuple(int(index) for index in initial_indices)
    if len(set(initial)) != len(initial) or len(initial) > count:
        raise ValueError("k-center initial indices are invalid")
    if any(index not in positions for index in initial):
        raise ValueError("k-center initial index is unknown")
    selected = [positions[index] for index in initial]
    if not selected:
        mean_distance = matrix.mean(axis=1)
        selected.append(min(
            range(indices.size),
            key=lambda row: (-float(mean_distance[row]), int(indices[row]))))
    while len(selected) < count:
        selected_set = set(selected)
        minimum = matrix[:, selected].min(axis=1)
        position = min(
            (row for row in range(indices.size) if row not in selected_set),
            key=lambda row: (-float(minimum[row]), int(indices[row])))
        selected.append(position)
    return tuple(int(indices[position]) for position in selected)


def greedy_kcenter_features(
        indices, values, groups, count, initial_indices):
    matrix = _finite_matrix(values, "k-center features")
    indices = _validated_indices(indices, matrix.shape[0])
    groups = tuple(str(group) for group in groups)
    if len(groups) != matrix.shape[1]:
        raise ValueError("k-center groups do not match features")
    count = int(count)
    initial = tuple(int(index) for index in initial_indices)
    if not initial:
        raise ValueError("incremental k-center requires initial indices")
    if count < len(initial) or count > indices.size:
        raise ValueError("k-center count is invalid")
    positions = dict((int(index), position)
                     for position, index in enumerate(indices.tolist()))
    if len(set(initial)) != len(initial) or \
            any(index not in positions for index in initial):
        raise ValueError("k-center initial indices are invalid")
    selected = [positions[index] for index in initial]
    selected_set = set(selected)
    minimum = np.full(matrix.shape[0], np.inf, dtype=np.float64)
    for position in selected:
        minimum = np.minimum(
            minimum,
            _distance_to_center(matrix, matrix[position], groups))
    while len(selected) < count:
        position = min(
            (row for row in range(indices.size) if row not in selected_set),
            key=lambda row: (-float(minimum[row]), int(indices[row])))
        selected.append(position)
        selected_set.add(position)
        minimum = np.minimum(
            minimum,
            _distance_to_center(matrix, matrix[position], groups))
    return tuple(int(indices[position]) for position in selected)


def deterministic_kmedoids(indices, distances, count):
    indices, matrix = _validated_distance(indices, distances)
    count = int(count)
    medoid_indices = greedy_kcenter(indices, matrix, count)
    position_by_index = dict(
        (int(index), position) for position, index in enumerate(indices))
    medoids = [position_by_index[index] for index in medoid_indices]
    while True:
        assignment = np.argmin(matrix[:, medoids], axis=1)
        updated = []
        for cluster in range(count):
            members = np.flatnonzero(assignment == cluster)
            if members.size == 0:
                raise ValueError("k-medoids produced an empty cluster")
            cost = matrix[np.ix_(members, members)].sum(axis=1)
            best = min(
                range(members.size),
                key=lambda offset: (
                    float(cost[offset]), int(indices[members[offset]])))
            updated.append(int(members[best]))
        if updated == medoids:
            break
        medoids = updated
    assignment = np.argmin(matrix[:, medoids], axis=1)
    return KMedoidsResult(
        medoid_indices=tuple(int(indices[position]) for position in medoids),
        assignments=tuple(int(cluster) for cluster in assignment.tolist()))


def build_disjoint_splits(
        length, baseline_count, baseline_seed, audit_count, audit_seed,
        random_seeds):
    length = int(length)
    baseline_count = int(baseline_count)
    audit_count = int(audit_count)
    if length <= 0 or baseline_count <= 0 or audit_count <= 0:
        raise ValueError("split counts must be positive")
    if baseline_count + audit_count > length:
        raise ValueError("split counts exceed dataset length")
    population = np.arange(length, dtype=np.int64)
    baseline = np.random.RandomState(int(baseline_seed)).choice(
        population, baseline_count, replace=False)
    audit_population = np.asarray(
        sorted(set(population.tolist()) - set(baseline.tolist())),
        dtype=np.int64)
    audit = np.random.RandomState(int(audit_seed)).choice(
        audit_population, audit_count, replace=False)
    eligible = np.asarray(
        sorted(set(population.tolist()) - set(audit.tolist())),
        dtype=np.int64)
    random_baselines = tuple(
        tuple(int(index) for index in np.random.RandomState(int(seed)).choice(
            eligible, baseline_count, replace=False).tolist())
        for seed in random_seeds)
    return DisjointSplits(
        baseline_indices=tuple(int(index) for index in baseline.tolist()),
        audit_indices=tuple(int(index) for index in audit.tolist()),
        eligible_indices=tuple(int(index) for index in eligible.tolist()),
        random_baselines=random_baselines)


def _wasserstein_1d(reference, candidate):
    quantiles = np.linspace(0.0, 1.0, 257)
    first = np.quantile(reference, quantiles)
    second = np.quantile(candidate, quantiles)
    return float(np.mean(np.abs(first - second)))


def descriptor_coverage(
        configuration, calibration, audit, names, groups):
    calibration = _finite_matrix(calibration, "calibration descriptors")
    audit = _finite_matrix(audit, "audit descriptors")
    names = tuple(str(name) for name in names)
    groups = tuple(str(group) for group in groups)
    if calibration.shape[1] != audit.shape[1] or \
            len(names) != calibration.shape[1] or len(groups) != len(names):
        raise ValueError("descriptor coverage schema mismatch")
    rows = []
    for column, name in enumerate(names):
        calibration_values = calibration[:, column]
        audit_values = audit[:, column]
        calibration_minimum = float(calibration_values.min())
        calibration_maximum = float(calibration_values.max())
        audit_p01, audit_p50, audit_p99 = np.quantile(
            audit_values, (0.01, 0.5, 0.99))
        audit_maximum = float(audit_values.max())
        inside = (
            (audit_values >= calibration_minimum) &
            (audit_values <= calibration_maximum))
        rows.append({
            "configuration": str(configuration),
            "feature": name,
            "group": groups[column],
            "calibration_min": calibration_minimum,
            "calibration_max": calibration_maximum,
            "audit_p01": float(audit_p01),
            "audit_p50": float(audit_p50),
            "audit_p99": float(audit_p99),
            "audit_max": audit_maximum,
            "range_coverage": float(inside.mean()),
            "p01_ratio": float(np.quantile(calibration_values, 0.01) /
                               audit_p01) if audit_p01 != 0.0 else None,
            "p50_ratio": float(np.quantile(calibration_values, 0.5) /
                               audit_p50) if audit_p50 != 0.0 else None,
            "p99_ratio": float(np.quantile(calibration_values, 0.99) /
                               audit_p99) if audit_p99 != 0.0 else None,
            "maximum_ratio": calibration_maximum / audit_maximum
            if audit_maximum != 0.0 else None,
            "wasserstein": _wasserstein_1d(
                calibration_values, audit_values),
        })
    return rows


def _grouped_cross_distance(first, second, groups):
    first = _finite_matrix(first, "distance reference")
    second = _finite_matrix(second, "distance candidate")
    groups = tuple(str(group) for group in groups)
    if first.shape[1] != second.shape[1] or len(groups) != first.shape[1]:
        raise ValueError("cross-distance schema mismatch")
    unique_groups = tuple(dict.fromkeys(groups))
    distance = np.zeros((first.shape[0], second.shape[0]), dtype=np.float64)
    for group in unique_groups:
        positions = [index for index, current in enumerate(groups)
                     if current == group]
        difference = first[:, None, positions] - second[None, :, positions]
        distance += np.mean(difference * difference, axis=2)
    return distance / float(len(unique_groups))


def nearest_distance_summary(configuration, calibration, audit, groups):
    distance = _grouped_cross_distance(audit, calibration, groups)
    nearest = distance.min(axis=1)
    return {
        "configuration": str(configuration),
        "calibration_samples": int(distance.shape[1]),
        "audit_samples": int(distance.shape[0]),
        "nearest_p50": float(np.quantile(nearest, 0.5)),
        "nearest_p95": float(np.quantile(nearest, 0.95)),
        "nearest_max": float(nearest.max()),
    }


def activation_range_coverage(
        configuration, calibration, audit, names):
    calibration = _finite_matrix(calibration, "calibration activations")
    audit = _finite_matrix(audit, "audit activations")
    names = tuple(str(name) for name in names)
    if calibration.shape[1] != audit.shape[1] or \
            len(names) != calibration.shape[1]:
        raise ValueError("activation range schema mismatch")
    rows = []
    for column, name in enumerate(names):
        calibration_maximum = float(calibration[:, column].max())
        audit_maximum = float(audit[:, column].max())
        rows.append({
            "configuration": str(configuration),
            "feature": name,
            "calibration_max": calibration_maximum,
            "audit_max": audit_maximum,
            "audit_exceeds_calibration": int(
                audit_maximum > calibration_maximum),
            "maximum_ratio": audit_maximum / calibration_maximum
            if calibration_maximum != 0.0 else None,
        })
    return rows
