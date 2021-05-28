"""This module defines the SimplePartitioner pass."""
from __future__ import annotations

from typing import Any
from typing import Iterable

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit

# TODO:
#   Layout should be a separate pass from partitioning. The partitioner may
#   need to be changed so that it can accept a layout assignment, but by
#   default assumes the numberings in the algorithm and topology are equal.


class SimpleLayout(BasePass):
    def __init__(
        self,
        layout : dict[int,int] | None
    ) -> None:
        self.layout = layout if layout is not None else {}

    def _generate_layout(self, circuit : Circuit) -> dict[int,int]:
        pass

    def _apply_layout(self, circuit : Circuit) -> None:
        # For each cycle
        for point, op in circuit.operations_on_qudit_with_points:
            # Pop each operation from the cycle into a list
            
            # Change the locations of each operation
            # Pop operations from the list and into their new circuit locations
        pass

    def run(self, circuit: Circuit, data: dict[str, Any]) -> None:
        self.layout = data['layout'] if 'layout' in data else \
            self.layout if self.layout is not None else \
            self._generate_layout

        self._apply_layout(circuit)