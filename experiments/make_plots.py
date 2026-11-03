"""Plot convergence spreads, final-cost boxes, and optimality gaps."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from experiments.common import BKS, ROOT, read_summary


def read_trace(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before making plots") from exc

    plot_dir = args.results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for name in ("att48", "ch130", "a280"):
        summaries = {}
        traces = {}
        for algorithm in ("sa", "baseline"):
            summary_path = args.results_dir / algorithm / f"{name}_summary.csv"
            if not summary_path.exists():
                continue
            summaries[algorithm] = read_summary(summary_path)
            traces[algorithm] = [
                read_trace(args.results_dir / algorithm / row["trace_file"])
                for row in summaries[algorithm]
            ]
        if not traces:
            continue

        plt.figure(figsize=(9, 5))
        for algorithm, run_traces in traces.items():
            by_evaluation = {}
            for trace in run_traces:
                for row in trace:
                    by_evaluation.setdefault(int(row["evaluations_used"]), []).append(
                        float(row["best_cost"])
                    )
            x = sorted(by_evaluation)
            mean = [statistics.mean(by_evaluation[value]) for value in x]
            std = [statistics.stdev(by_evaluation[value]) if len(by_evaluation[value]) > 1 else 0 for value in x]
            plt.plot(x, mean, label=algorithm)
            plt.fill_between(x, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], alpha=0.15)
        plt.axhline(BKS[name], color="black", linestyle="--", label="BKS")
        plt.xlabel("Evaluations used")
        plt.ylabel("Best-so-far tour distance")
        plt.title(f"Convergence: {name}")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{name}_convergence.png", dpi=160)
        plt.close()

        labels, values = [], []
        for algorithm, records in summaries.items():
            labels.append(algorithm)
            values.append([float(row["final_cost"]) for row in records])
        plt.figure(figsize=(7, 5))
        plt.boxplot(values, labels=labels)
        plt.axhline(BKS[name], color="black", linestyle="--", label="BKS")
        plt.ylabel("Final tour distance")
        plt.title(f"Final costs and gap: {name}")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{name}_final_costs.png", dpi=160)
        plt.close()
    print(plot_dir)


if __name__ == "__main__":
    main()
