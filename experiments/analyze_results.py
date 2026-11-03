"""Summarize per-run tables and compare SA with the baseline."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

from experiments.common import BKS, ROOT, read_summary, write_summary


def summarize(input_dir: Path, output: Path) -> None:
    rows = []
    for path in sorted(input_dir.glob("*_summary.csv")):
        records = read_summary(path)
        if not records:
            continue
        instance = records[0]["instance"]
        costs = [float(record["final_cost"]) for record in records]
        bks = BKS.get(instance, math.nan)
        mean = statistics.mean(costs)
        rows.append({
            "algorithm": records[0]["algorithm"],
            "instance": instance,
            "runs": len(costs),
            "best": min(costs),
            "mean": mean,
            "median": statistics.median(costs),
            "std_dev": statistics.stdev(costs) if len(costs) > 1 else 0.0,
            "bks": bks,
            "best_gap_percent": ((min(costs) - bks) / bks * 100) if not math.isnan(bks) else "",
            "mean_gap_percent": ((mean - bks) / bks * 100) if not math.isnan(bks) else "",
        })
    write_summary(output, rows)


def compare_algorithms(results_dir: Path, output: Path) -> None:
    """Run two-sided Mann–Whitney tests and report both medians."""

    from scipy.stats import mannwhitneyu

    rows = []
    for sa_path in sorted((results_dir / "sa").glob("*_summary.csv")):
        baseline_path = results_dir / "baseline" / sa_path.name
        if not baseline_path.exists():
            continue
        sa = [float(row["final_cost"]) for row in read_summary(sa_path)]
        baseline = [float(row["final_cost"]) for row in read_summary(baseline_path)]
        statistic, p_value = mannwhitneyu(sa, baseline, alternative="two-sided")
        rows.append({
            "instance": sa_path.stem.removesuffix("_summary"),
            "sa_median": statistics.median(sa),
            "baseline_median": statistics.median(baseline),
            "u_statistic": statistic,
            "p_value": p_value,
            "alpha": 0.05,
            "significant": p_value < 0.05,
        })
    write_summary(output, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    summary_dir = args.results_dir / "summary"
    for algorithm in ("sa", "baseline"):
        input_dir = args.results_dir / algorithm
        if input_dir.exists():
            output = summary_dir / f"{algorithm}.csv"
            summarize(input_dir, output)
            print(output)
    comparison = summary_dir / "mann_whitney.csv"
    compare_algorithms(args.results_dir, comparison)
    print(comparison)


if __name__ == "__main__":
    main()
