#!/usr/bin/env python3
"""Contracts and statistics for strict W4A4 and FP4 evaluation."""

from __future__ import division, print_function

import numpy as np


MODEL_ORDER = ("cspn", "dyspn", "nlspn", "completionformer")
METHOD_ORDER = ("rtn", "adaround", "brecq")
PRIMARY_CONFIGS = (
    "FP32", "FP4V_W4A4", "FP4V_W4E2M1", "FP4V_W4A8")
STRESS_CONFIGS = ("FP32", "HW_W4A4_full")
STRICT_METHODS = {
    "adaround": "adaround_strict",
    "brecq": "brecq_strict",
}
MAX_RELATIVE_RMSE_DEGRADATION = 0.10


def performance_decision(fp32_rmse, quant_rmse, rtn_rmse,
                         nonfinite_samples, nonfinite_pixels):
    values = np.asarray(
        [fp32_rmse, quant_rmse, rtn_rmse], dtype=np.float64)
    if not np.isfinite(values).all() or float(fp32_rmse) <= 0.0:
        raise ValueError("finite positive RMSE values are required")

    relative = (
        (float(quant_rmse) - float(fp32_rmse)) / float(fp32_rmse))
    if int(nonfinite_samples) != 0 or int(nonfinite_pixels) != 0:
        status = "rejected_nonfinite"
    elif relative > MAX_RELATIVE_RMSE_DEGRADATION:
        status = "rejected_fp32_degradation"
    elif float(quant_rmse) > float(rtn_rmse):
        status = "rejected_rtn_regression"
    else:
        status = "preserved"
    return {
        "relative_rmse_degradation": relative,
        "delta_vs_rtn": float(quant_rmse) - float(rtn_rmse),
        "status": status,
    }
