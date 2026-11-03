"""Grid-search SA parameters using the same 50,000-evaluation budget."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.simulated_annealing import SimulatedAnnealingConfig
from src.tsp_problem import TSPProblem

from experiments.common import INSTANCE_NAMES, ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "tuning" / "grid.csv")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in INSTANCE_NAMES:
        problem = TSPProblem.from_tsplib(args.data_dir / f"{name}.tsp")
        target_values = (0.1, 0.3, 0.5) if name in {"a280", "ch130"} else (0.7, 0.8)
        scale_values = (0.25, 0.5, 0.75) if name in {"a280", "ch130"} else (0.75, 1.0, 1.25)
        for target_acceptance in target_values:
            estimated = problem.estimate_initial_temperature(
                seed=2026, target_acceptance=target_acceptance
            )
            for temperature_scale in scale_values:
                final_values = (0.1, 0.5) if name in {"a280", "ch130"} else (0.1, 0.25, 0.5)
                block_values = (50, 200) if name in {"a280", "ch130"} else (50, 100, 200)
                for final_temperature in final_values:
                    for iterations_per_temperature in block_values:
                        initial_temperature = estimated * temperature_scale
                        config = SimulatedAnnealingConfig(
                            initial_temperature=initial_temperature,
                            final_temperature=final_temperature,
                            iterations_per_temperature=iterations_per_temperature,
                            max_evaluations=50_000,
                            log_interval=100,
                        )
                        costs = [
                            problem.solve_with_sa(config=config, seed=9000 + run).best_cost
                            for run in range(args.runs)
                        ]
                        rows.append({
                            "instance": name,
                            "target_acceptance": target_acceptance,
                            "initial_temperature": initial_temperature,
                            "final_temperature": final_temperature,
                            "iterations_per_temperature": iterations_per_temperature,
                            "cooling_rate": config.cooling_rate,
                            "runs": args.runs,
                            "mean_cost": sum(costs) / len(costs),
                            "best_cost": min(costs),
                        })
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
