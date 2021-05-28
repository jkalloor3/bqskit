"""This module defines the Logical Topology pass."""
from __future__ import annotations

from typing import Any
from typing import Iterable

from itertools import permutations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.machine import MachineModel
from bqskit.ir.circuit import Circuit

class LogicalTopology(BasePass):
    def __init__(self):
        self.logical_connectivity_graph = set()

    def run(self, circuit: Circuit, data: dict[str, Any]) -> None:
        """
        Create a logical coupling map for a given circuit.

        Args:
            circuit (Circuit): The circuit to be analyzed.

            data (dict): Extra information about the run.
        """
        coup_map = set()
        for operation in circuit:
            edges = list(permutations([q for q in operation.location], 2))
            for edge in edges:
                coup_map.add(edge)
        self.logical_connectivity_graph = coup_map
    
    def get_logical_machine(self, circuit : Circuit) -> MachineModel:
        """
        Return a MachineModel with the logical topology.

        Args:
            circuit (Circuit): The circuit whose logical topology is returned.
        """
        data = {}
        self.run(circuit, data)
        return MachineModel(circuit.get_size(), self.logical_connectivity_graph)

