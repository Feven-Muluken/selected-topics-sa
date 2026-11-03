"""Run the equal-budget random-restart 2-opt baseline with one trace per run."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.baseline import solve_random_restart_2opt
from src.tsp_problem import TSPProblem

from experiments.common import INSTANCE_NAMES, ROOT, write_summary, write_trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "baseline")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--max-evaluations", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-interval", type=int, default=100)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")

    for name in INSTANCE_NAMES:
        path = args.data_dir / f"{name}.tsp"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run data/download_data.py first")
        problem = TSPProblem.from_tsplib(path)
        summary_rows = []
        for run in range(args.runs):
            run_seed = args.seed + run
            result = solve_random_restart_2opt(
                problem, args.max_evaluations, run_seed, args.log_interval
            )
            trace_path = args.output_dir / f"{name}_seed{run_seed}.csv"
            write_trace(trace_path, result.history)
            summary_rows.append({
                "algorithm": "random_restart_2opt",
                "instance": name,
                "run": run + 1,
                "seed": run_seed,
                "final_cost": result.best_cost,
                "evaluations_used": result.evaluations_used,
                "tour": " ".join(map(str, result.best_solution)),
                "trace_file": trace_path.name,
            })
        summary_path = args.output_dir / f"{name}_summary.csv"
        write_summary(summary_path, summary_rows)
        print(summary_path)


if __name__ == "__main__":
    main()
