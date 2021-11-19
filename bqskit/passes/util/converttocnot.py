"""This module implements the ToCNOTPass."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.swap import SwapGate
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint

_logger = logging.getLogger(__name__)


class ToCNOTPass(BasePass):
    """Converts SWAP gates to 3 CNOT Gates."""

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug('Converting SWAPs to CNOTGates.')
        # Find the maximum number of cycles the circuit can expand by
        swapcount = 0
        for op in circuit:
            if isinstance(op.gate, SwapGate):
                swapcount += 1
        _logger.debug(f'Found a total of {swapcount} SwapGates.')
        # 
        cycle = 0
        while swapcount > 0:
            for qudit in range(circuit.num_qudits):
                if not circuit.is_point_idle((cycle, qudit)):
                    op = circuit.get_operation((cycle, qudit))
                    if isinstance(op.gate, SwapGate):
                        point = CircuitPoint(cycle, op.location[0])
                        new_op_1 = Operation(CNOTGate(), [op.location[1], op.location[0]])
                        new_op_2 = Operation(CNOTGate(), [op.location[0], op.location[1]])
                        circuit.replace_gate(point, CNOTGate(), op.location)
                        circuit.insert(cycle, new_op_1)
                        circuit.insert(cycle, new_op_2)
                        swapcount -= 1
            cycle += 1
