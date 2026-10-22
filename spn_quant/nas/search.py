"""Deterministic data splitting and multi-objective NAS selection."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence


OBJECTIVES = ("rmse", "parameters", "latency_ms")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_search_split(
    total_count: int,
    dev_count: int,
    *,
    seed: int,
    source_list: Path,
) -> dict[str, Any]:
    total_count = int(total_count)
    dev_count = int(dev_count)
    if total_count <= 1:
        raise ValueError("total_count must be greater than one")
    if dev_count <= 0 or dev_count >= total_count:
        raise ValueError("dev_count must be between zero and total_count")
    generator = random.Random(int(seed))
    dev_indices = sorted(generator.sample(range(total_count), dev_count))
    dev_set = set(dev_indices)
    train_indices = [index for index in range(total_count) if index not in dev_set]
    return {
        "format_version": 1,
        "seed": int(seed),
        "total_count": total_count,
        "source_list": str(Path(source_list).resolve()),
        "source_list_sha256": _file_sha256(Path(source_list)),
        "train_indices": train_indices,
        "dev_indices": dev_indices,
    }


def validate_search_split(
    manifest: Mapping[str, Any],
    *,
    total_count: int,
    source_list: Path,
) -> None:
    train = [int(value) for value in manifest["train_indices"]]
    dev = [int(value) for value in manifest["dev_indices"]]
    if len(train) != len(set(train)) or len(dev) != len(set(dev)):
        raise ValueError("split contains duplicate indices")
    if set(train).intersection(dev):
        raise ValueError("train and dev indices overlap")
    expected = set(range(int(total_count)))
    if set(train).union(dev) != expected:
        raise ValueError("split does not cover the source dataset exactly")
    if int(manifest.get("total_count", -1)) != int(total_count):
        raise ValueError("split total_count mismatch")
    if manifest.get("source_list_sha256") != _file_sha256(Path(source_list)):
        raise ValueError("split source-list hash mismatch")


def _objective_values(row: Mapping[str, Any]) -> tuple[float, ...]:
    values = tuple(float(row[key]) for key in OBJECTIVES)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("candidate contains a non-finite objective")
    return values


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a <= b for a, b in zip(left, right)) and any(
        a < b for a, b in zip(left, right))


def pareto_rank(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranked = [dict(row) for row in rows]
    if not ranked:
        return []
    identifiers = [str(row["candidate"]) for row in ranked]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate identifiers must be unique")
    values = [_objective_values(row) for row in ranked]
    dominated = [[] for _ in ranked]
    domination_count = [0 for _ in ranked]
    for left in range(len(ranked)):
        for right in range(left + 1, len(ranked)):
            if _dominates(values[left], values[right]):
                dominated[left].append(right)
                domination_count[right] += 1
            elif _dominates(values[right], values[left]):
                dominated[right].append(left)
                domination_count[left] += 1

    fronts = [[index for index, count in enumerate(domination_count) if count == 0]]
    rank = 0
    while fronts[rank]:
        next_front = []
        for index in fronts[rank]:
            ranked[index]["pareto_rank"] = rank
            for other in dominated[index]:
                domination_count[other] -= 1
                if domination_count[other] == 0:
                    next_front.append(other)
        rank += 1
        fronts.append(sorted(set(next_front)))

    for front in fronts[:-1]:
        _assign_crowding_distance(ranked, front)
    return sorted(ranked, key=lambda row: str(row["candidate"]))


def _assign_crowding_distance(rows: list[dict[str, Any]], front: list[int]) -> None:
    for index in front:
        rows[index]["crowding_distance"] = 0.0
    if len(front) <= 2:
        for index in front:
            rows[index]["crowding_distance"] = float("inf")
        return
    for objective in OBJECTIVES:
        ordered = sorted(
            front,
            key=lambda index: (float(rows[index][objective]),
                               str(rows[index]["candidate"])),
        )
        rows[ordered[0]]["crowding_distance"] = float("inf")
        rows[ordered[-1]]["crowding_distance"] = float("inf")
        low = float(rows[ordered[0]][objective])
        high = float(rows[ordered[-1]][objective])
        if high == low:
            continue
        for position in range(1, len(ordered) - 1):
            index = ordered[position]
            if math.isinf(rows[index]["crowding_distance"]):
                continue
            before = float(rows[ordered[position - 1]][objective])
            after = float(rows[ordered[position + 1]][objective])
            rows[index]["crowding_distance"] += (after - before) / (high - low)


def select_successive_halving(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ranked = pareto_rank(rows)
    keep = min(len(ranked), max(8, int(math.ceil(len(ranked) / 4.0))))
    ordered = sorted(
        ranked,
        key=lambda row: (
            int(row["pareto_rank"]),
            -float(row["crowding_distance"]),
            str(row["candidate"]),
        ),
    )
    return ordered[:keep]
