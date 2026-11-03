"""Generic from-scratch simulated annealing minimizer."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from random import Random
from time import perf_counter
from typing import Callable, Generic, List, Optional, TypeVar


Solution = TypeVar("Solution") # solution is a flexablo type


@dataclass(frozen=True)
class SimulatedAnnealingConfig:
    """Controls SA; the initial cost counts as one evaluation."""

    initial_temperature: float = 1_000.0
    final_temperature: float = 0.25
    max_evaluations: int = 50_000
    iterations_per_temperature: int = 100
    log_interval: int = 100

    def validate(self) -> None:
        if self.initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        if self.final_temperature <= 0:
            raise ValueError("final_temperature must be positive")
        if self.final_temperature >= self.initial_temperature:
            raise ValueError("final_temperature must be below initial_temperature")
        if self.max_evaluations < 2:
            raise ValueError("max_evaluations must be at least 2")
        if self.iterations_per_temperature < 1:
            raise ValueError("iterations_per_temperature must be positive")
        if self.log_interval < 1:
            raise ValueError("log_interval must be positive")

    @property
    def temperature_steps(self) -> int:
        return (self.max_evaluations - 1 + self.iterations_per_temperature - 1) // self.iterations_per_temperature

    @property
    def cooling_rate(self) -> float:
        """Rate that reaches ``final_temperature`` on the last block."""

        return (self.final_temperature / self.initial_temperature) ** (1 / self.temperature_steps)


@dataclass
class SimulatedAnnealingResult(Generic[Solution]):
    best_solution: Solution
    best_cost: float
    iterations: int
    evaluations_used: int
    accepted_moves: int
    history: List[dict[str, float | int]]


def simulated_annealing(
    initial_solution: Solution,
    cost_function: Callable[[Solution], float],
    neighbour_function: Callable[[Solution, Random], Solution],
    config: Optional[SimulatedAnnealingConfig] = None,
    seed: Optional[int] = None,
    candidate_cost_function: Optional[
        Callable[[Solution, float, Random], tuple[Solution, float]]
    ] = None,
) -> SimulatedAnnealingResult[Solution]:
    """Minimize ``cost_function`` using the Metropolis acceptance rule.

    A worse candidate is accepted with probability
    ``exp(-(candidate_cost - current_cost) / temperature)``.  The caller
    owns the solution representation and supplies a problem-specific move.
    """

    settings = config or SimulatedAnnealingConfig()
    settings.validate()
    rng = Random(seed)

    start_time = perf_counter()
    current_solution = initial_solution
    current_cost = float(cost_function(current_solution))
    if not current_cost == current_cost:
        raise ValueError("cost_function returned NaN")

    best_solution = current_solution
    best_cost = current_cost
    temperature = settings.initial_temperature
    iterations = 0
    accepted_moves = 0
    history: List[dict[str, float | int]] = [{
        "iteration": 0,
        "evaluations_used": 1,
        "best_cost": best_cost,
        "elapsed_seconds": perf_counter() - start_time,
        "temperature": settings.initial_temperature,
    }]

    while iterations < settings.max_evaluations - 1:
        for _ in range(settings.iterations_per_temperature):
            if iterations >= settings.max_evaluations - 1:
                break

            if candidate_cost_function is None:
                candidate = neighbour_function(current_solution, rng)
                candidate_cost = float(cost_function(candidate))
            else:
                candidate, candidate_cost = candidate_cost_function(
                    current_solution, current_cost, rng
                )
                candidate_cost = float(candidate_cost)
            if not candidate_cost == candidate_cost:
                raise ValueError("cost_function returned NaN")

            delta = candidate_cost - current_cost
            accepted = delta <= 0 or rng.random() < exp(-delta / temperature)
            if accepted:
                current_solution = candidate
                current_cost = candidate_cost
                accepted_moves += 1

                if current_cost < best_cost:
                    best_solution = current_solution
                    best_cost = current_cost

            iterations += 1
            evaluations_used = iterations + 1
            if (
                iterations == settings.max_evaluations - 1
                or iterations % settings.log_interval == 0
            ):
                history.append({
                "iteration": iterations,
                "evaluations_used": evaluations_used,
                "best_cost": best_cost,
                "elapsed_seconds": perf_counter() - start_time,
                "temperature": temperature,
                })

        temperature *= settings.cooling_rate

    return SimulatedAnnealingResult(
        best_solution=best_solution,
        best_cost=best_cost,
        iterations=iterations,
        evaluations_used=iterations + 1,
        accepted_moves=accepted_moves,
        history=history,
    )
