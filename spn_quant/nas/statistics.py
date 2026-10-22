"""Paired non-inferiority statistics for CSPN encoder candidates."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def paired_hierarchical_bootstrap(
    control_rmse: np.ndarray,
    candidate_rmse: np.ndarray,
    *,
    margin_ratio: float = 0.02,
    confidence: float = 0.95,
    replicates: int = 10_000,
    seed: int = 20260920,
) -> dict[str, Any]:
    """Bootstrap paired seed-by-sample RMSE differences.

    Rows are independent training seeds and columns are matched validation
    samples. Each replicate resamples rows, then columns within each selected
    row, while retaining candidate/control pairing.
    """
    control = np.asarray(control_rmse, dtype=np.float64)
    candidate = np.asarray(candidate_rmse, dtype=np.float64)
    if control.ndim != 2 or candidate.ndim != 2:
        raise ValueError("RMSE arrays must be two-dimensional seed/sample matrices")
    if control.shape != candidate.shape or not all(control.shape):
        raise ValueError("control and candidate RMSE arrays must have equal non-empty shapes")
    if not np.isfinite(control).all() or not np.isfinite(candidate).all():
        raise ValueError("RMSE arrays contain non-finite values")
    if margin_ratio <= 0:
        raise ValueError("margin_ratio must be positive")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")
    if replicates <= 0:
        raise ValueError("replicates must be positive")

    generator = np.random.default_rng(int(seed))
    seed_count, sample_count = control.shape
    deltas = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        sampled_seeds = generator.integers(0, seed_count, size=seed_count)
        total = 0.0
        for source_seed in sampled_seeds:
            sampled_samples = generator.integers(
                0, sample_count, size=sample_count)
            paired_delta = (
                candidate[source_seed, sampled_samples]
                - control[source_seed, sampled_samples])
            total += float(np.mean(paired_delta))
        deltas[replicate] = total / seed_count

    control_mean = float(np.mean(control))
    candidate_mean = float(np.mean(candidate))
    delta = candidate_mean - control_mean
    margin = float(margin_ratio) * control_mean
    upper = float(np.quantile(deltas, confidence))
    return {
        "control_rmse": control_mean,
        "candidate_rmse": candidate_mean,
        "delta_rmse": delta,
        "margin_ratio": float(margin_ratio),
        "margin": margin,
        "confidence": float(confidence),
        "upper_confidence_bound": upper,
        "noninferior": bool(upper <= margin),
        "replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "training_seed_count": int(seed_count),
        "sample_count": int(sample_count),
    }
