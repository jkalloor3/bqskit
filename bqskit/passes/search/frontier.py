"""This module implements the Frontier class."""
from __future__ import annotations

import heapq
import itertools
from typing import Any
from typing import NamedTuple

from bqskit.ir.circuit import Circuit
from bqskit.passes.search.heuristic import HeuristicFunction
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class FrontierElement(NamedTuple):
    """The Frontier contains FrontierElements."""
    cost: float
    element_id: int
    circuit: Circuit
    extra_data: Any


class Frontier:
    """The Frontier class."""

    def __init__(
        self,
        target: UnitaryMatrix | StateVector | StateSystem,
        heuristic_function: HeuristicFunction,
        max_solutions: int = 1,
        success_threshold: float = 1e-8
    ) -> None:
        """
        Construct an empty frontier.

        Args:
            target (UnitaryMatrix | StateVector | StateSystem): The target to
                pass to the heuristic_function.

            heuristic_function (HeuristicFunction): The heuristic used
                to sort the Frontier.

            max_solutions (int): The maximum number of solutions to store.

            success_threshold (float): The threshold for considering a
            solution successful.

        """

        if not isinstance(target, (UnitaryMatrix, StateVector, StateSystem)):
            raise TypeError(
                'Expected unitary or state, got %s.' % type(target),
            )

        if not isinstance(heuristic_function, HeuristicFunction):
            raise TypeError(
                'Expected HeursiticFunction, got %s.'
                % type(heuristic_function),
            )

        self.target = target
        self.heuristic_function = heuristic_function
        self._frontier: list[FrontierElement] = []
        self._counter = itertools.count()
        self.solutions: list[Circuit] = []
        self.max_solutions = max_solutions
        self.psols: dict[int, list[tuple[Circuit, float]]] = {}
        self.success_threshold = success_threshold

    def add(self, circuit: Circuit, extra_data: Any = None) -> None:
        """Add `circuit` into the frontier."""
        heuristic_value = self.heuristic_function(circuit, self.target)
        count = next(self._counter)
        elem = FrontierElement(heuristic_value, count, circuit, extra_data)
        heapq.heappush(self._frontier, elem)

    def pop(self) -> tuple[Circuit, Any]:
        """Pop the top circuit."""
        elem = heapq.heappop(self._frontier)
        return elem.circuit, elem.extra_data

    def empty(self) -> bool:
        """Return true if the frontier is empty."""
        return len(self._frontier) == 0
    
    def should_continue(self) -> bool:
        """We should continue if the number of solutions < max solutions
        and we are not empty"""
        return len(self.solutions) < self.max_solutions and not self.empty()

    def add_solution(self, circuit: Circuit, dist: float) -> None:
        """Add a solution to the frontier."""
        if dist < self.success_threshold:
            self.solutions.append(circuit)

    def add_partial_solution(self, circuit: Circuit, layer: int, dist: float) -> None:
        """Add a partial solution to the frontier."""
        if layer not in self.psols:
            self.psols[layer] = []

        self.psols[layer].append((circuit.copy(), dist))

        if len(self.psols[layer]) > self.max_solutions:
            self.psols[layer].sort(key=lambda x: x[1])
            del self.psols[layer][-1]

    def final_solution(self, default: Circuit) -> Circuit | list[Circuit]:
        """Return the best solution found or default."""
        if len(self.solutions) == 0:
            self.solutions.append(default.copy())
        if self.max_solutions == 1:
            return self.solutions[0]
        return self.solutions

    def clear(self) -> None:
        """Remove all elements from the frontier."""
        self._frontier.clear()
        self.solutions.clear()
        self.psols.clear()
