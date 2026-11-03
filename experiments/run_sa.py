"""Run reproducible simulated-annealing trials with one trace per run."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.simulated_annealing import SimulatedAnnealingConfig
from src.tsp_problem import TSPProblem

from experiments.common import INSTANCE_NAMES, ROOT, TUNED_CONFIGS, write_summary, write_trace


def run_instance(
    problem: TSPProblem,
    instance_name: str,
    output_dir: Path,
    runs: int,
    config: SimulatedAnnealingConfig,
    seed: int,
) -> Path:
    summary_rows = []
    for run in range(runs):
        run_seed = seed + run
        result = problem.solve_with_sa(config=config, seed=run_seed)
        trace_path = output_dir / f"{instance_name}_seed{run_seed}.csv"
        write_trace(trace_path, result.history)
        summary_rows.append({
            "algorithm": "simulated_annealing",
            "instance": instance_name,
            "run": run + 1,
            "seed": run_seed,
            "final_cost": result.best_cost,
            "evaluations_used": result.evaluations_used,
            "tour": " ".join(map(str, result.best_solution)),
            "trace_file": trace_path.name,
        })
    summary_path = output_dir / f"{instance_name}_summary.csv"
    write_summary(summary_path, summary_rows)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "sa")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--initial-temperature", type=float, default=None)
    parser.add_argument("--final-temperature", type=float, default=None)
    parser.add_argument("--iterations-per-temperature", type=int, default=None)
    parser.add_argument("--max-evaluations", type=int, default=50_000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--target-acceptance", type=float, default=None)
    parser.add_argument("--use-tuned", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")

    for name in INSTANCE_NAMES:
        path = args.data_dir / f"{name}.tsp"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run data/download_data.py first")
        problem = TSPProblem.from_tsplib(path)
        tuned = TUNED_CONFIGS[name] if args.use_tuned else {}
        target_acceptance = (
            args.target_acceptance
            if args.target_acceptance is not None
            else tuned.get("target_acceptance", 0.8)
        )
        initial_temperature = args.initial_temperature
        if initial_temperature is None:
            initial_temperature = problem.estimate_initial_temperature(
                args.seed, target_acceptance=target_acceptance
            ) * tuned.get("temperature_scale", 1.0)
        final_temperature = (
            args.final_temperature
            if args.final_temperature is not None
            else tuned.get("final_temperature", 0.25)
        )
        iterations_per_temperature = (
            args.iterations_per_temperature
            if args.iterations_per_temperature is not None
            else tuned.get("iterations_per_temperature", 100)
        )
        print(
            f"{name}: T0={initial_temperature:.6g}, target={target_acceptance}, "
            f"Tf={final_temperature}, block={iterations_per_temperature}"
        )
        config = SimulatedAnnealingConfig(
            initial_temperature=initial_temperature,
            final_temperature=final_temperature,
            iterations_per_temperature=iterations_per_temperature,
            max_evaluations=args.max_evaluations,
            log_interval=args.log_interval,
        )
        print(run_instance(problem, name, args.output_dir, args.runs, config, args.seed))


if __name__ == "__main__":
    main()
