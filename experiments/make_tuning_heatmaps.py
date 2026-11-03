"""Create parameter-sensitivity heat maps from the tuning grid."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from experiments.common import INSTANCE_NAMES, ROOT


def mean_by(rows: list[dict[str, str]], row_key: str, column_key: str) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row[row_key], row[column_key])].append(float(row["mean_cost"]))
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def plot_heatmap(
    values: dict[tuple[str, str], float],
    row_labels: list[str],
    column_labels: list[str],
    row_label: str,
    column_label: str,
    title: str,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    matrix = [
        [values.get((row, column), float("nan")) for column in column_labels]
        for row in row_labels
    ]
    display_columns = [
        f"{float(label):.3g}" if column_label == "Initial temperature" else label
        for label in column_labels
    ]
    figure, axis = plt.subplots(figsize=(10, 5))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(display_columns)), display_columns, rotation=45, ha="right")
    axis.set_yticks(range(len(row_labels)), row_labels)
    axis.set_xlabel(column_label)
    axis.set_ylabel(row_label)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="Mean final cost")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "tuning" / "grid.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "plots" / "tuning")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before making plots") from exc

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for instance in INSTANCE_NAMES:
        instance_rows = [row for row in rows if row["instance"] == instance]
        if not instance_rows:
            continue

        target_initial = mean_by(instance_rows, "target_acceptance", "initial_temperature")
        targets = sorted({row["target_acceptance"] for row in instance_rows}, key=float)
        initials = sorted(
            {row["initial_temperature"] for row in instance_rows},
            key=float,
        )
        plot_heatmap(
            target_initial,
            targets,
            initials,
            "Target acceptance",
            "Initial temperature",
            f"{instance}: acceptance and initial-temperature sensitivity",
            args.output_dir / f"{instance}_acceptance_temperature.png",
        )

        final_block = mean_by(instance_rows, "final_temperature", "iterations_per_temperature")
        finals = sorted({row["final_temperature"] for row in instance_rows}, key=float)
        blocks = sorted(
            {row["iterations_per_temperature"] for row in instance_rows},
            key=float,
        )
        plot_heatmap(
            final_block,
            finals,
            blocks,
            "Final temperature",
            "Moves per temperature block",
            f"{instance}: cooling-schedule sensitivity",
            args.output_dir / f"{instance}_cooling_schedule.png",
        )

    print(args.output_dir)


if __name__ == "__main__":
    main()
