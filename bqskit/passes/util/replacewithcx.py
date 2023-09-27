"""This module implements the ToU3Pass."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant import CNOTGate
from bqskit.ir.point import CircuitPoint

_logger = logging.getLogger(__name__)


class ReplaceWithCXPass(BasePass):
    """Converts all two qubit gates with CX gates. """

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug('Converting single-qubit general gates to U3Gates.')
        for cycle, op in circuit.operations_with_cycles():
            if (op.num_qudits == 2):
                point = CircuitPoint(cycle, op.location[0])
                circuit.replace_gate(point, CNOTGate(), op.location)
