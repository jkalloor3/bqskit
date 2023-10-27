"""This module implements the Rebase2QuditGatePass."""
from __future__ import annotations

import logging
from typing import Any
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit, CircuitGate
from bqskit.ir.gate import Gate
from bqskit.ir.point import CircuitPoint as Point
from bqskit.qis import UnitaryMatrix
from bqskit.utils.typing import is_sequence


_logger = logging.getLogger(__name__)


class GetUnitariesPass(BasePass):
    """
    The Rebase2QuditGatePass class.

    Will use instantiation to change the a 2-qudit gate to a different one.
    """

    def __init__(
        self,
        gate_in_circuit: Gate | Sequence[Gate],
    ) -> None:
        """
        Construct a Rebase2QuditGatePass.

        Args:
            gate_in_circuit (Gate | Sequence[Gate]): The two-qudit gate
                or gates in the circuit that you want to replace.

            new_gate (Gate | Sequence[Gate]): The two-qudit new gate or
                gates you want to put in the circuit.

            max_depth (int): The maximum number of new gates to replace
                an old gate with. (Default: 3)

            max_retries (int): The number of retries for the same gate
                before we increase the maximum depth. If left as -1,
                then never increase max depth. (Default: -1)

            success_threshold (float): The distance threshold that
                determines successful termintation. Measured in cost
                described by the hilbert schmidt cost function.
                (Default: 1e-8)

            cost (CostFunction | None): The cost function that determines
                successful removal of a gate.
                (Default: HilbertSchmidtResidualsGenerator())

            instantiate_options (dict[str: Any]): Options passed directly
                to circuit.instantiate when instantiating circuit
                templates. (Default: {})

            single_qudit_gate (Gate): A single-qudit gate to fill
                in between two-qudit gates.

        Raises:
            ValueError: If `gate_in_circuit` or `new_gate` is not a 2-qudit
                gate.

            ValueError: if `max_depth` is nonnegative.
        """

        if is_sequence(gate_in_circuit):
            if any(not isinstance(g, Gate) for g in gate_in_circuit):
                raise TypeError('Expected Gate or Gate list.')

        elif not isinstance(gate_in_circuit, Gate):
            raise TypeError(f'Expected Gate, got {type(gate_in_circuit)}.')

        else:
            gate_in_circuit = [gate_in_circuit]

        if any(g.num_qudits != 2 for g in gate_in_circuit):
            raise ValueError('Expected 2-qudit gate.')


        self.gates: list[Gate] = list(gate_in_circuit)

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        circuit_copy = circuit.copy()
        unitaries = []

        for g in self.gates:

            while circuit_copy.count(g) > 0:
                # Group together a 2-qubit block composed of gates from old set
                point, utry = self.group_near_gates(
                    circuit_copy,
                    circuit_copy.point(g),
                    self.gates,
                )

                unitaries.append(utry)
                circuit_copy.remove(circuit_copy[point])
                circuit_copy.unfold_all()

        
        data["block_unitaries"] = unitaries

                
    def group_near_gates(
        self,
        circuit: Circuit,
        center: Point,
        gates: list[Gate],
    ) -> tuple[Point, UnitaryMatrix]:
        """Group gates similar to the gate at center on the same qubits."""
        op = circuit[center]
        qubits = op.location
        counts = {g: 0.0 for g in gates}
        counts[op.gate] += 1.0

        # Go to the left until cant
        i = 0
        moving_left = True
        while moving_left:
            i += 1
            if center.cycle - i < 0:
                i = center.cycle
                break
            for q in qubits:
                point = (center.cycle - i, q)
                if not circuit.is_point_idle(point):
                    lop = circuit[point]
                    if any(p not in qubits for p in lop.location):
                        i -= 1
                        moving_left = False
                        break
                    if lop.num_qudits != 1 and lop.gate not in gates:
                        i -= 1
                        moving_left = False
                        break
                    if lop.num_qudits == 2:
                        counts[lop.gate] += 0.5

        j = 0
        moving_right = True
        while moving_right:
            j += 1
            if center.cycle + j >= circuit.num_cycles:
                j = circuit.num_cycles - center.cycle - 1
                break
            for q in qubits:
                point = (center.cycle + j, q)
                if not circuit.is_point_idle(point):
                    rop = circuit[point]
                    if any(p not in qubits for p in rop.location):
                        j -= 1
                        moving_right = False
                        break
                    if rop.num_qudits != 1 and rop.gate not in gates:
                        j -= 1
                        moving_right = False
                        break
                    if rop.num_qudits == 2:
                        counts[rop.gate] += 0.5

        region = {q: (center.cycle - i, center.cycle + j) for q in qubits}
        grouped_gate_str = ', '.join([
            f'{int(c)} {g}' + ('s' if c > 1 else '')
            for g, c in counts.items()
        ])
        _logger.debug(f'Grouped together {grouped_gate_str}.')
        pt = circuit.fold(region)

        c_gate: CircuitGate = circuit[pt].gate
        utry = c_gate._circuit.get_unitary()

        return pt, utry

    