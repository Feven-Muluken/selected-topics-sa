"""TSPLIB loading, distance calculations, and TSP operators."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, log, sqrt
from pathlib import Path
from random import Random
from typing import List, Optional, Sequence, Tuple

from src.simulated_annealing import (
    SimulatedAnnealingConfig,
    SimulatedAnnealingResult,
    simulated_annealing,
)


City = Tuple[float, float]
Tour = List[int]


@dataclass(frozen=True)
class TSPProblem:
    """A symmetric Euclidean TSP instance."""

    cities: Tuple[City, ...]
    edge_weight_type: str = "EUC_2D"
    distances: Tuple[Tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.distances:
            return
        matrix = tuple(
            tuple(self._calculate_distance(first, second) for second in range(len(self.cities)))
            for first in range(len(self.cities))
        )
        object.__setattr__(self, "distances", matrix)

    @classmethod
    def from_tsplib(cls, path: str | Path) -> "TSPProblem":
        """Load a TSPLIB file containing NODE_COORD_SECTION."""

        coordinates: List[City] = []
        edge_weight_type = "EUC_2D"
        in_coordinates = False
        path = Path(path)
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.upper() == "NODE_COORD_SECTION":
                in_coordinates = True
                continue
            if not in_coordinates and ":" in line:
                key, value = (part.strip() for part in line.split(":", 1))
                if key.upper() == "EDGE_WEIGHT_TYPE":
                    edge_weight_type = value.upper()
                continue
            if line.upper() in {"EOF", "DISPLAY_DATA_SECTION"}:
                break
            if in_coordinates:
                fields = line.split()
                if len(fields) < 3:
                    raise ValueError(f"Invalid coordinate row: {raw_line!r}")
                coordinates.append((float(fields[1]), float(fields[2])))

        if len(coordinates) < 2:
            raise ValueError(f"No usable coordinates found in {path}")
        if edge_weight_type not in {"EUC_2D", "ATT"}:
            raise ValueError(f"Unsupported TSPLIB EDGE_WEIGHT_TYPE: {edge_weight_type}")
        return cls(tuple(coordinates), edge_weight_type)

    def distance(self, first: int, second: int) -> float:
        return self.distances[first][second]

    def _calculate_distance(self, first: int, second: int) -> float:
        dx = self.cities[first][0] - self.cities[second][0]
        dy = self.cities[first][1] - self.cities[second][1]
        squared_distance = dx * dx + dy * dy
        if self.edge_weight_type == "ATT":
            root = sqrt(squared_distance / 10.0)
            truncated = int(root)
            return float(truncated + 1 if truncated < root else truncated)
        return float(int(hypot(dx, dy) + 0.5))

    def cost(self, tour: Sequence[int], validate: bool = True) -> float:
        if validate:
            self.validate_tour(tour)
        return sum(
            self.distance(tour[index], tour[(index + 1) % len(tour)])
            for index in range(len(tour))
        )

    def validate_tour(self, tour: Sequence[int]) -> None:
        if len(tour) != len(self.cities):
            raise ValueError("tour must contain every city exactly once")
        if set(tour) != set(range(len(self.cities))):
            raise ValueError("tour must be a permutation of city indices")

    def random_tour(self, rng: Random) -> Tour:
        tour = list(range(len(self.cities)))
        rng.shuffle(tour)
        return tour

    def estimate_initial_temperature(
        self, seed: int = 0, samples: int = 128, target_acceptance: float = 0.8
    ) -> float:
        """Choose T0 for the requested acceptance of a typical worsening move."""

        if not 0 < target_acceptance < 1:
            raise ValueError("target_acceptance must be between 0 and 1")
        rng = Random(seed)
        tour = self.random_tour(rng)
        current_cost = self.cost(tour, validate=False)
        worsening = []
        for _ in range(samples):
            candidate, candidate_cost = self.neighbour_with_cost(tour, current_cost, rng)
            if candidate_cost > current_cost:
                worsening.append(candidate_cost - current_cost)
        if not worsening:
            return 1.0
        worsening.sort()
        typical_delta = worsening[len(worsening) // 2]
        return max(1.0, -typical_delta / log(target_acceptance))

    def neighbour(self, tour: Sequence[int], rng: Random) -> Tour:
        """Create a feasible candidate by reversing one tour segment (2-opt)."""

        candidate = list(tour)
        first, second = sorted(rng.sample(range(len(candidate)), 2))
        candidate[first : second + 1] = reversed(candidate[first : second + 1])
        return candidate

    def neighbour_with_cost(
        self, tour: Sequence[int], current_cost: float, rng: Random
    ) -> tuple[Tour, float]:
        """Return a 2-opt candidate and its cost using four matrix lookups."""

        first, second = sorted(rng.sample(range(1, len(tour)), 2))
        candidate = list(tour)
        previous_first = candidate[first - 1]
        first_city = candidate[first]
        second_city = candidate[second]
        next_second = candidate[(second + 1) % len(candidate)]
        candidate[first : second + 1] = reversed(candidate[first : second + 1])
        removed = self.distance(previous_first, first_city) + self.distance(second_city, next_second)
        added = self.distance(previous_first, second_city) + self.distance(first_city, next_second)
        return candidate, current_cost - removed + added

    def nearest_neighbour(self, start: int = 0) -> Tour:
        """Construct the required deterministic baseline tour."""

        if not 0 <= start < len(self.cities):
            raise ValueError("start city is outside the problem")
        unvisited = set(range(len(self.cities)))
        tour = [start]
        unvisited.remove(start)
        while unvisited:
            current = tour[-1]
            next_city = min(unvisited, key=lambda city: self.distance(current, city))
            tour.append(next_city)
            unvisited.remove(next_city)
        return tour

    def solve_with_sa(
        self,
        config: Optional[SimulatedAnnealingConfig] = None,
        seed: Optional[int] = None,
        initial_tour: Optional[Sequence[int]] = None,
    ) -> SimulatedAnnealingResult[Tour]:
        rng = Random(seed)
        starting_tour = list(initial_tour) if initial_tour is not None else self.random_tour(rng)
        self.validate_tour(starting_tour)
        settings = config or SimulatedAnnealingConfig(
            initial_temperature=self.estimate_initial_temperature(seed or 0)
        )
        result = simulated_annealing(
            initial_solution=starting_tour,
            cost_function=lambda tour: self.cost(tour, validate=False),
            neighbour_function=self.neighbour,
            config=settings,
            seed=rng.randrange(2**63),
            candidate_cost_function=self.neighbour_with_cost,
        )
        self.validate_tour(result.best_solution)
        exact_cost = self.cost(result.best_solution, validate=False)
        if exact_cost != result.best_cost:
            raise RuntimeError("2-opt incremental cost disagrees with full tour cost")
        return result


if __name__ == "__main__":
    example = TSPProblem(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    result = example.solve_with_sa(seed=7)
    baseline = example.nearest_neighbour()
    print(f"SA best cost: {result.best_cost:.3f}")
    print(f"Nearest-neighbour cost: {example.cost(baseline):.3f}")
