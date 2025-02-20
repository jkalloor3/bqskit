"""This module implements the ToU3Pass."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.generalgate import GeneralGate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.point import CircuitPoint
from bqskit.passes.partitioning import GroupSingleQuditGatePass

_logger = logging.getLogger(__name__)


class ToU3Pass(BasePass):
    """Converts single-qubit general unitary gates to U3 Gates."""

    def __init__(self, 
                 convert_all_single_qubit_gates: bool = False, 
                 ensemble: bool = False,
                 group: bool = False) -> None:
        """
        Construct a ToU3Pass.

        Args:
            convert_all_single_qubit_gates (bool): Indicates wheter to convert
            only the general gates, or every single qubit gate.
        """

        self.convert_all_single_qubit_gates = convert_all_single_qubit_gates
        self.ensemble = ensemble
        self.group = group

    async def run_circ(self, circuit: Circuit) -> None:
        """Perform the pass's operation"""
        _logger.debug('Converting single-qubit general gates to U3Gates.')
        for cycle, op in circuit.operations_with_cycles():
            if (
                op.radixes == (2,) and (
                    isinstance(op.gate, GeneralGate)
                    or self.convert_all_single_qubit_gates
                )
            ):
                params = U3Gate().calc_params(op.get_unitary())
                point = CircuitPoint(cycle, op.location[0])
                circuit.replace_gate(point, U3Gate(), op.location, params)
   
    @staticmethod
    def run_group_circ(circuit: Circuit) -> None:
        """Perform the pass's operation"""
        GroupSingleQuditGatePass.group(circuit)
        for cycle, op in circuit.operations_with_cycles():
            if (op.num_params >= 3 and op.radixes == (2,)):
                params = U3Gate().calc_params(op.get_unitary())
                # print(params, flush=True)
                point = CircuitPoint(cycle, op.location[0])
                circuit.replace_gate(point, U3Gate(), op.location, params)
        
        circuit.unfold_all()


    async def run(self, circuit: Circuit, data: PassData) -> None:
        if self.ensemble:
            for circ, dist in data['scan_sols']:
                if self.group:
                    ToU3Pass.run_group_circ(circ)
                else:
                    await self.run_circ(circ)
        else:
            if self.group:
                    ToU3Pass.run_group_circ(circ)
            await self.run_circ(circuit)
