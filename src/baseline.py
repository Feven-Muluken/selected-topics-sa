"""Nearest-neighbour and equal-budget random-restart 2-opt baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from time import perf_counter
from src.tsp_problem import TSPProblem, Tour


@dataclass
class BaselineResult:
    best_solution: Tour
    best_cost: float
    evaluations_used: int
    history: list[dict[str, float | int]]


def solve_random_restart_2opt(
    problem: TSPProblem,
    max_evaluations: int = 50_000,
    seed: int | None = None,
    log_interval: int = 100,
    restart_after: int | None = None,
) -> BaselineResult:
    """Run improving 2-opt with random restarts under an evaluation budget."""

    if max_evaluations < 2:
        raise ValueError("max_evaluations must be at least 2")
    if log_interval < 1:
        raise ValueError("log_interval must be positive")
    rng = Random(seed)
    restart_limit = restart_after or 2 * len(problem.cities)
    start_time = perf_counter()
    current = problem.random_tour(rng)
    current_cost = problem.cost(current, validate=False)
    best_tour = list(current)
    best_cost = current_cost
    evaluations_used = 1
    non_improving = 0
    history: list[dict[str, float | int]] = [{
        "iteration": 0,
        "evaluations_used": 1,
        "best_cost": best_cost,
        "elapsed_seconds": perf_counter() - start_time,
    }]

    while evaluations_used < max_evaluations:
        candidate, candidate_cost = problem.neighbour_with_cost(
            current, current_cost, rng
        )
        evaluations_used += 1
        if candidate_cost < current_cost:
            current, current_cost = candidate, candidate_cost
            non_improving = 0
            if current_cost < best_cost:
                best_tour, best_cost = list(current), current_cost
        else:
            non_improving += 1
            if non_improving >= restart_limit and evaluations_used < max_evaluations:
                current = problem.random_tour(rng)
                current_cost = problem.cost(current, validate=False)
                evaluations_used += 1
                non_improving = 0
                if current_cost < best_cost:
                    best_tour, best_cost = list(current), current_cost

        if (
            evaluations_used == max_evaluations
            or (evaluations_used - 1) % log_interval == 0
        ):
            history.append({
                "iteration": evaluations_used - 1,
                "evaluations_used": evaluations_used,
                "best_cost": best_cost,
                "elapsed_seconds": perf_counter() - start_time,
            })

    problem.validate_tour(best_tour)
    return BaselineResult(best_tour, best_cost, evaluations_used, history)


def solve_nearest_neighbour(instance_path: str | Path, start: int = 0) -> tuple[Tour, float]:
    problem = TSPProblem.from_tsplib(instance_path)
    tour = problem.nearest_neighbour(start=start)
    return tour, problem.cost(tour)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    tour, cost = solve_nearest_neighbour(args.instance, args.start)
    print(f"cost={cost:.3f}")
    print("tour=" + ",".join(str(city + 1) for city in tour))


if __name__ == "__main__":
    main()
