"""Shared experiment paths, benchmark metadata, and CSV writers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
INSTANCE_NAMES = ("att48", "ch130", "a280")
BKS = {"att48": 10628.0, "ch130": 6110.0, "a280": 2579.0}
TUNED_CONFIGS = {
    "att48": {
        "target_acceptance": 0.7,
        "temperature_scale": 0.75,
        "final_temperature": 0.1,
        "iterations_per_temperature": 200,
    },
    "ch130": {
        "target_acceptance": 0.3,
        "temperature_scale": 0.25,
        "final_temperature": 0.5,
        "iterations_per_temperature": 50,
    },
    "a280": {
        "target_acceptance": 0.1,
        "temperature_scale": 0.25,
        "final_temperature": 0.5,
        "iterations_per_temperature": 200,
    },
}
TRACE_FIELDS = (
    "iteration",
    "evaluations_used",
    "best_cost",
    "elapsed_seconds",
    "temperature",
)


def write_trace(path: Path, history: Iterable[Mapping[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        for record in history:
            row = {}
            for field in TRACE_FIELDS:
                row[field] = record.get(field, "")
            writer.writerow(row)


def write_summary(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
