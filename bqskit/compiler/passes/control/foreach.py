# type: ignore
# TODO: Remove type: ignore, when new mypy comes out with TypeGuards
"""This module implements the ForEachBlockPass class."""
from __future__ import annotations

import logging
from typing import Any
from typing import Callable
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
from bqskit.utils.typing import is_sequence

_logger = logging.getLogger(__name__)


class ForEachBlockPass(BasePass):
    """
    The ForEachBlockPass class.

    This is a control pass that executes another pass or passes on every block
    in the circuit.
    """

    def __init__(
        self,
        loop_body: BasePass | Sequence[BasePass],
        replace_filter: Callable[[Circuit, Operation], bool] | None = None,
    ) -> None:
        """
        Construct a ForEachBlockPass.

        Args:
            loop_body (BasePass | Sequence[BasePass]): The pass or passes
                to execute on every block.

            replace_filter (Callable[[Circuit, Operation], bool] | None):
                A predicate that determines if the resulting circuit, after
                calling `loop_body` on a block, should replace the original
                operation. Called with the circuit output from `loop_body`
                and the original operation. If this returns true, the
                operation will be replaced with the new circuit.
                Defaults to always replace.

        Raises:
            ValueError: If a Sequence[BasePass] is given, but it is empty.
        """

        if not is_sequence(loop_body) and not isinstance(loop_body, BasePass):
            raise TypeError(
                'Expected Pass or sequence of Passes, got %s.'
                % type(loop_body),
            )

        if is_sequence(loop_body):
            truth_list = [isinstance(elem, BasePass) for elem in loop_body]
            if not all(truth_list):
                raise TypeError(
                    'Expected Pass or sequence of Passes, got %s.'
                    % type(loop_body[truth_list.index(False)]),
                )
            if len(loop_body) == 0:
                raise ValueError('Expected at least one pass.')

        self.loop_body = loop_body if is_sequence(loop_body) else [loop_body]
        self.replace_filter = replace_filter or default_replace_filter

        if not callable(self.replace_filter):
            raise TypeError(
                'Expected callable method that maps Circuit and Operations to'
                ' booleans for replace_filter'
                ', got %s.' % type(self.replace_filter),
            )

    def run(self, circuit: Circuit, data: dict[str, Any]) -> None:
        """Perform the pass's operation, see BasePass for more info."""

        # Collect CircuitGate blocks
        blocks: list[tuple[CircuitPoint, Operation]] = []
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, CircuitGate):
                blocks.append((cycle, op))

        # Perform work
        points: list[CircuitPoint] = []
        ops: list[Operation] = []
        for cycle, op in blocks:
            gate: CircuitGate = op.gate
            subcircuit = gate._circuit.copy()
            subcircuit.set_params(op.params)

            for loop_pass in self.loop_body:
                # TODO: Pass only subtopology when topology avail
                loop_pass.run(subcircuit, data)

            if self.replace_filter(subcircuit, op):
                points.append(CircuitPoint(cycle, op.location[0]))
                ops.append(
                    Operation(
                        CircuitGate(subcircuit, True),
                        op.location,
                        subcircuit.get_params(),
                    ),
                )

        circuit.batch_replace(points, ops)


def default_replace_filter(circuit: Circuit, op: Operation) -> bool:
    return True
