import csv
import json
import math
from pathlib import Path

import pytest
import torch

from spn_quant.gpu_latency import (
    MODEL_ORDER,
    audit_fp32_execution,
    summarize_latencies,
    validate_four_model_results,
    write_latency_reports,
)


def _result(model, uuid="GPU-test", precision="fp32"):
    return {
        "format_version": 1,
        "model": model,
        "architecture_class": {
            "cspn": "ResNet",
            "dyspn": "Model",
            "nlspn": "NLSPNModel",
            "completionformer": "CompletionFormer",
        }[model],
        "precision": precision,
        "batch_size": 1,
        "measurement_scope": "forward_only_gpu_resident_inputs",
        "includes_propagation": True,
        "input_shapes": [[1, 3, 228, 304], [1, 1, 228, 304]],
        "sample_index": 386,
        "propagation_iterations": {
            "cspn": 24, "dyspn": 6, "nlspn": 18,
            "completionformer": 18,
        }[model],
        "parameter_count": 10,
        "parameter_size_mb": 40.0 / (1024.0 * 1024.0),
        "gpu": {"name": "NVIDIA A100-SXM4-40GB", "uuid": uuid},
        "software": {"python": "3.x", "torch": "test", "cuda": "test"},
        "settings": {
            "warmup_iterations": 2,
            "timed_iterations_per_round": 3,
            "rounds": 2,
            "tf32": False,
            "cudnn_benchmark": True,
            "python_gc_enabled_during_timing": False,
        },
        "latency_ms": {
            "count": 6,
            "mean": 2.5,
            "std": 1.0,
            "min": 1.0,
            "p50": 2.5,
            "p90": 3.7,
            "p95": 3.85,
            "p99": 3.97,
            "max": 4.0,
            "fps_from_mean": 400.0,
        },
        "memory_mb": {
            "baseline_allocated": 10.0,
            "peak_allocated": 12.0,
            "peak_reserved": 14.0,
            "inference_workspace_peak": 2.0,
        },
        "rounds": [{"round": 1, "mean_ms": 2.5, "p50_ms": 2.5}],
    }


def test_latency_summary_uses_linear_percentiles_and_population_std():
    summary = summarize_latencies([1.0, 2.0, 3.0, 4.0])

    assert summary["count"] == 4
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["std"] == pytest.approx(math.sqrt(1.25))
    assert summary["p50"] == pytest.approx(2.5)
    assert summary["p90"] == pytest.approx(3.7)
    assert summary["p95"] == pytest.approx(3.85)
    assert summary["p99"] == pytest.approx(3.97)
    assert summary["fps_from_mean"] == pytest.approx(400.0)


@pytest.mark.parametrize("samples", ([], [0.0], [-1.0], [float("nan")]))
def test_latency_summary_rejects_missing_or_nonpositive_samples(samples):
    with pytest.raises(ValueError, match="finite positive"):
        summarize_latencies(samples)


def test_fp32_audit_counts_nested_inputs_and_ignores_integer_buffers():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(4, 2, 1),
        torch.nn.BatchNorm2d(2),
    ).float()
    model_args = (
        {"rgb": torch.ones(1, 3, 2, 3)},
        [torch.ones(1, 1, 2, 3), "metadata"],
    )

    audit = audit_fp32_execution(model, model_args)

    assert audit["precision"] == "fp32"
    assert audit["input_tensor_count"] == 2
    assert audit["input_shapes"] == [[1, 3, 2, 3], [1, 1, 2, 3]]
    assert audit["parameter_count"] == sum(p.numel() for p in model.parameters())
    assert audit["parameter_size_bytes"] == 4 * audit["parameter_count"]


def test_fp32_audit_rejects_bf16_inputs_and_parameters():
    model = torch.nn.Linear(2, 2).float()
    with pytest.raises(RuntimeError, match="input.*torch.bfloat16"):
        audit_fp32_execution(model, (torch.ones(1, 2).bfloat16(),))

    model.bfloat16()
    with pytest.raises(RuntimeError, match="parameter.*torch.bfloat16"):
        audit_fp32_execution(model, (torch.ones(1, 2),))


def test_four_model_validation_orders_results_and_rejects_mixed_gpu():
    unordered = [_result(name) for name in reversed(MODEL_ORDER)]

    ordered = validate_four_model_results(unordered)

    assert tuple(item["model"] for item in ordered) == MODEL_ORDER

    unordered[-1]["gpu"]["uuid"] = "GPU-other"
    with pytest.raises(ValueError, match="same physical GPU"):
        validate_four_model_results(unordered)


def test_four_model_validation_rejects_non_fp32_or_incomplete_results():
    with pytest.raises(ValueError, match="exactly four"):
        validate_four_model_results([_result("cspn")])

    mixed = [_result(name) for name in MODEL_ORDER]
    mixed[2]["precision"] = "bf16"
    with pytest.raises(ValueError, match="FP32"):
        validate_four_model_results(mixed)


def test_report_writer_persists_stable_csv_markdown_and_manifest(tmp_path):
    results = [_result(name) for name in MODEL_ORDER]

    paths = write_latency_reports(tmp_path, results)

    assert set(paths) == {"csv", "markdown", "manifest"}
    with paths["csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["model"] for row in rows] == list(MODEL_ORDER)
    assert rows[0]["p50_ms"] == "2.5"
    report = paths["markdown"].read_text(encoding="utf-8")
    assert "forward-only latency with GPU-resident inputs" in report
    assert "Python cyclic GC disabled during timing" in report
    assert "each request synchronized" in report
    assert "CompletionFormer" in report
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["measurement_scope"] == \
        "forward_only_gpu_resident_inputs"
    assert len(manifest["results"]) == 4


def test_report_writer_labels_native_gc_enabled_runtime(tmp_path):
    results = [_result(name) for name in MODEL_ORDER]
    for result in results:
        result["settings"]["python_gc_enabled_during_timing"] = True

    paths = write_latency_reports(tmp_path, results)

    report = paths["markdown"].read_text(encoding="utf-8")
    assert "Python cyclic GC enabled during timing" in report
