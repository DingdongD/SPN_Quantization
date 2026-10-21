#!/usr/bin/env python3
"""Run and validate a four-model LLMCompass systolic-array study."""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spn_quant.hardware.array_dse import (  # noqa: E402
    ArrayDSEConfig,
    CandidateSummary,
    pareto_frontier,
    rank_shared_candidates,
    simulate_profile,
)
from spn_quant.hardware.llmcompass_adapter import (  # noqa: E402
    LLMCOMPASS_REVISION,
)
from spn_quant.hardware.workload import WorkloadProfile  # noqa: E402


MODEL_NAMES = ("cspn", "dyspn", "nlspn", "completionformer")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ("git",) + tuple(arguments), cwd=str(REPO_ROOT),
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True)
    return completed.stdout.strip()


def _write_csv(path: Path, fieldnames: Sequence[str],
               rows: Iterable[Mapping[str, Any]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _summary_row(summary: CandidateSummary) -> Dict[str, Any]:
    return {
        "model": summary.model_name,
        "array_height": summary.array.height,
        "array_width": summary.array.width,
        "array_label": summary.array.label,
        "pe_count": summary.pe_count,
        "dense_macs": summary.dense_macs,
        "irregular_macs": summary.irregular_macs,
        "cycles": summary.cycles,
        "ideal_cycles": summary.ideal_cycles,
        "effective_utilization": summary.effective_utilization,
        "ideal_cycle_overhead": summary.ideal_cycle_overhead,
        "cycles_per_mac": summary.cycles_per_mac,
        "low_utilization_mac_fraction":
            summary.low_utilization_mac_fraction,
        "low_utilization_layer_fraction":
            summary.low_utilization_layer_fraction,
        "array_coverage_fraction": summary.array_coverage_fraction,
        "analytical_gemm_fraction": summary.analytical_gemm_fraction,
        "gemm_count": summary.gemm_count,
    }


def _plot(path: Path, summaries: Sequence[CandidateSummary],
          config: ArrayDSEConfig) -> None:
    by_model = {
        model: {row.array: row for row in summaries if row.model_name == model}
        for model in MODEL_NAMES
    }
    x = list(range(len(config.array_shapes)))
    labels = [shape.label for shape in config.array_shapes]
    figure, axis = plt.subplots(figsize=(12, 5.5))
    for model in MODEL_NAMES:
        axis.plot(
            x, [by_model[model][shape].effective_utilization
                for shape in config.array_shapes], marker="o", linewidth=1.5,
            label=model)
    shared = {
        row.array: row for row in rank_shared_candidates(summaries)}
    axis.plot(
        x, [shared[shape].geometric_mean_utilization
            for shape in config.array_shapes], marker="s", linewidth=2.5,
        linestyle="--", color="black", label="four-model geometric mean")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_xlabel("Array shape (height x width)")
    axis.set_ylabel("Effective PE utilization")
    axis.set_ylim(0.0, 1.02)
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(ncol=3, fontsize=9)
    figure.tight_layout()
    figure.savefig(str(path), dpi=180)
    plt.close(figure)


def run(config_path: Path, workload_paths: Mapping[str, Path],
        output: Path) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError("DSE output already exists: %s" % output)
    if set(workload_paths) != set(MODEL_NAMES):
        raise ValueError("workload paths must cover exactly four SPN models")
    config_path = Path(config_path).resolve()
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = ArrayDSEConfig.from_dict(config_payload)
    resolved_paths = {
        model: Path(workload_paths[model]).resolve() for model in MODEL_NAMES}
    profiles = {
        model: WorkloadProfile.read_json(resolved_paths[model])
        for model in MODEL_NAMES}
    for model, profile in profiles.items():
        if profile.model_name != model:
            raise ValueError("workload model identity mismatch: %s" % model)

    revision = _git("rev-parse", "HEAD")
    dirty_before_output = bool(_git("status", "--porcelain"))
    simulations = tuple(
        simulation for model in MODEL_NAMES
        for simulation in simulate_profile(profiles[model], config))
    summaries = tuple(simulation.summary for simulation in simulations)
    shared_ranking = rank_shared_candidates(summaries)

    output.mkdir(parents=True)
    layer_rows = []
    for simulation in sorted(
            simulations,
            key=lambda row: (MODEL_NAMES.index(row.model_name),
                             row.array.pe_count, row.array.height,
                             row.array.width)):
        for row in sorted(simulation.rows, key=lambda item: item.execution_index):
            layer_rows.append({
                "model": simulation.model_name,
                "array_height": row.array.height,
                "array_width": row.array.width,
                "array_label": row.array.label,
                "pe_count": row.array_pe_count,
                "execution_index": row.execution_index,
                "layer": row.name,
                "kind": row.kind,
                "m": row.m,
                "n": row.n,
                "k": row.k,
                "repetitions": row.repetitions,
                "macs": row.macs,
                "cycles": row.cycles,
                "ideal_cycles": row.ideal_cycles,
                "mac_per_cycle": row.mac_per_cycle,
                "dataflow": row.dataflow,
                "effective_utilization": row.effective_utilization,
                "ideal_cycle_overhead": row.ideal_cycle_overhead,
                "backend": row.backend,
                "simulation_source": row.simulation_source,
            })
    layer_fields = (
        "model", "array_height", "array_width", "array_label", "pe_count",
        "execution_index", "layer", "kind", "m", "n", "k",
        "repetitions", "macs", "cycles", "ideal_cycles", "mac_per_cycle",
        "dataflow", "effective_utilization", "ideal_cycle_overhead",
        "backend", "simulation_source",
    )
    _write_csv(output / "layer_results.csv", layer_fields, layer_rows)

    summary_rows = tuple(_summary_row(row) for row in sorted(
        summaries,
        key=lambda row: (MODEL_NAMES.index(row.model_name), row.pe_count,
                         row.array.height, row.array.width)))
    summary_fields = tuple(summary_rows[0])
    _write_csv(output / "model_summary.csv", summary_fields, summary_rows)

    iso_best = {}
    for pe_count, shapes in config.iso_pe_groups:
        shape_set = set(shapes)
        candidates = tuple(
            row for row in shared_ranking if row.array in shape_set)
        best = sorted(
            candidates,
            key=lambda row: (-row.geometric_mean_utilization,
                             row.total_cycles, row.array.height,
                             row.array.width))[0]
        iso_best[int(pe_count)] = best.array
    ranking_rows = []
    for rank, row in enumerate(shared_ranking, 1):
        ranking_rows.append({
            "rank": rank,
            "array_height": row.array.height,
            "array_width": row.array.width,
            "array_label": row.array.label,
            "pe_count": row.array.pe_count,
            "geometric_mean_utilization": row.geometric_mean_utilization,
            "model_count": row.model_count,
            "total_cycles": row.total_cycles,
            "mean_array_coverage_fraction":
                row.mean_array_coverage_fraction,
            "best_within_iso_pe_group": int(
                iso_best.get(row.array.pe_count) == row.array),
        })
    _write_csv(
        output / "candidate_ranking.csv", tuple(ranking_rows[0]),
        ranking_rows)

    frontier_rows = []
    for model in MODEL_NAMES:
        model_frontier = pareto_frontier(tuple(
            row for row in summaries if row.model_name == model))
        for row in model_frontier:
            frontier_rows.append({
                "model": model,
                "array_height": row.array.height,
                "array_width": row.array.width,
                "array_label": row.array.label,
                "pe_count": row.pe_count,
                "cycles": row.cycles,
                "effective_utilization": row.effective_utilization,
            })
    _write_csv(
        output / "pareto_frontier.csv", tuple(frontier_rows[0]),
        frontier_rows)

    coverage_rows = []
    for model in MODEL_NAMES:
        profile = profiles[model]
        estimated = profile.mapped_macs + profile.irregular_macs
        kinds = sorted(set(layer.kind for layer in profile.layers
                           if not layer.array_mappable))
        coverage_rows.append({
            "model": model,
            "mapped_macs": profile.mapped_macs,
            "irregular_macs": profile.irregular_macs,
            "estimated_total_macs": estimated,
            "array_coverage_fraction": profile.mapped_macs / float(estimated),
            "mappable_gemm_count": profile.mappable_gemm_count,
            "irregular_layer_count": sum(
                not layer.array_mappable for layer in profile.layers),
            "irregular_kinds": ";".join(kinds),
        })
    _write_csv(
        output / "workload_coverage.csv", tuple(coverage_rows[0]),
        coverage_rows)

    _plot(output / "shape_utilization.png", summaries, config)
    metadata = {
        "format_version": 1,
        "llmcompass_revision": LLMCOMPASS_REVISION,
        "scalesim_version": importlib_metadata.version("scalesim"),
        "repository_revision": revision,
        "repository_dirty_before_output": dirty_before_output,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "config": config_payload,
        "input_paths": {
            model: str(resolved_paths[model]) for model in MODEL_NAMES},
        "input_sha256": {
            model: _sha256(resolved_paths[model]) for model in MODEL_NAMES},
        "formulas": {
            "effective_utilization":
                "sum(macs)/sum(cycles*height*width*mac_per_cycle)",
            "array_coverage_fraction":
                "dense_macs/(dense_macs+irregular_macs)",
            "shared_score":
                "geometric_mean(per_model_effective_utilization)",
        },
        "limitations": [
            "Dense-array cycles exclude propagation gather/scatter and vector ops.",
            "Deformable sampling and memory-system latency are not calibrated.",
            "PE count is a resource proxy, not an area or power estimate.",
        ],
        "open_hardware_questions": [
            "process, voltage, frequency",
            "array count and per-core SRAM",
            "global-buffer and DRAM bandwidth",
            "latency, throughput, area, and power budgets",
            "separate sparse/vector engine contract",
        ],
        "confidence": {
            "dense_gemm_shape_comparison": "MEDIUM",
            "whole_model_latency": "LOW",
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return output


def _read_csv(path: Path) -> Tuple[Dict[str, str], ...]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def validate_artifacts(output: Path) -> str:
    output = Path(output)
    layers = _read_csv(output / "layer_results.csv")
    summaries = _read_csv(output / "model_summary.csv")
    ranking = _read_csv(output / "candidate_ranking.csv")
    frontier = _read_csv(output / "pareto_frontier.csv")
    coverage = _read_csv(output / "workload_coverage.csv")
    if not layers or not summaries or not ranking or not frontier:
        raise ValueError("DSE artifacts contain an empty required table")
    by_candidate: Dict[Tuple[str, int, int], list] = {}
    for row in layers:
        key = (row["model"], int(row["array_height"]),
               int(row["array_width"]))
        by_candidate.setdefault(key, []).append(row)
        capacity = (int(row["cycles"]) * int(row["pe_count"]) *
                    float(row["mac_per_cycle"]))
        expected = int(row["macs"]) / capacity
        if not math.isclose(
                expected, float(row["effective_utilization"]),
                rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("layer utilization arithmetic mismatch")
    summary_map = {}
    for row in summaries:
        key = (row["model"], int(row["array_height"]),
               int(row["array_width"]))
        detail = by_candidate.get(key)
        if not detail:
            raise ValueError("summary has no detailed rows")
        dense_macs = sum(int(item["macs"]) for item in detail)
        cycles = sum(int(item["cycles"]) for item in detail)
        capacity = sum(
            int(item["cycles"]) * int(item["pe_count"]) *
            float(item["mac_per_cycle"]) for item in detail)
        if dense_macs != int(row["dense_macs"]) or cycles != int(row["cycles"]):
            raise ValueError("summary totals do not match layer rows")
        if not math.isclose(
                dense_macs / capacity, float(row["effective_utilization"]),
                rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("summary utilization arithmetic mismatch")
        summary_map[key] = row
    if set(row["model"] for row in coverage) != set(MODEL_NAMES):
        raise ValueError("coverage table does not contain all four models")
    for row in ranking:
        height, width = int(row["array_height"]), int(row["array_width"])
        values = [float(summary_map[(model, height, width)][
            "effective_utilization"]) for model in MODEL_NAMES]
        expected = math.exp(sum(math.log(value) for value in values) /
                            len(values))
        if not math.isclose(
                expected, float(row["geometric_mean_utilization"]),
                rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("shared ranking geometric mean mismatch")
    for model in MODEL_NAMES:
        candidates = tuple(
            (int(row["pe_count"]), int(row["cycles"]), row["array_label"])
            for row in summaries if row["model"] == model)
        expected_labels = set()
        for pe_count, cycles, label in candidates:
            if not any(
                    other_pe <= pe_count and other_cycles <= cycles and
                    (other_pe < pe_count or other_cycles < cycles)
                    for other_pe, other_cycles, other_label in candidates
                    if other_label != label):
                expected_labels.add(label)
        actual_labels = set(
            row["array_label"] for row in frontier if row["model"] == model)
        if actual_labels != expected_labels:
            raise ValueError("Pareto frontier mismatch for %s" % model)
    return "SPN_ARRAY_DSE_VALIDATION_OK"


def _parse_workloads(values: Sequence[str]) -> Dict[str, Path]:
    result = {}
    for value in values:
        model, separator, path = value.partition("=")
        if not separator or model not in MODEL_NAMES or not path:
            raise ValueError("workload must use model=/path/to/profile.json")
        if model in result:
            raise ValueError("duplicate workload model: %s" % model)
        result[model] = Path(path)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", type=Path)
    return parser


def main(argv: Any = None) -> int:
    args = _parser().parse_args(argv)
    if args.validate_only is not None:
        if args.config is not None or args.workload or args.output is not None:
            raise ValueError("--validate-only cannot be combined with run inputs")
        print(validate_artifacts(args.validate_only))
        return 0
    if args.config is None or args.output is None:
        raise ValueError("--config and --output are required for a DSE run")
    output = run(args.config, _parse_workloads(args.workload), args.output)
    print(validate_artifacts(output))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
