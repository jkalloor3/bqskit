"""This module implements the CZToCNOTPass."""
from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.ir.gates.parameterized.u1 import U1Gate


class U1ToRZPass(BasePass):
    """
    The U1ToRZPass class.

    This uses a rule to convert U1 gates to RZ gates.
    """
    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        # Find all u1s
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, U1Gate):
                circuit.replace_gate((cycle, op.location[0]), 
                                     RZGate(), 
                                     op.location, 
                                     op.params)