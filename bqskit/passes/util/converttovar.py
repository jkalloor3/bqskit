"""This module implements the ToVariablePass."""
from __future__ import annotations

import logging
import pickle

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.generalgate import GeneralGate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.point import CircuitPoint

_logger = logging.getLogger(__name__)


class ToVariablePass(BasePass):
    """Converts single-qudit general unitary gates to Variable Unitary Gates."""

    def __init__(self, 
                 convert_all_single_qudit_gates: bool = False,
                 all_ensembles: bool = False) -> None:
        """
        Construct a ToVariablePass.

        Args:
            convert_all_single_qudit_gates (bool): Indicates wheter to convert
            only the general gates, or every single qudit gate.
        """
        self.convert_all_single_qudit_gates = convert_all_single_qudit_gates
        self.all_ensembles = all_ensembles

    async def run(self, circuit: Circuit, data: PassData) -> None:
        if self.all_ensembles:
            if "converted_to_var" in data:
                return
            
            for ensemble in data["ensemble"]:
                for circ,_ in ensemble:
                    await self.run_circ(circ, data)

            if "checkpoint_dir" in data:
                checkpoint_data_file = data["checkpoint_data_file"]
                data["converted_to_var"] = True
                pickle.dump(data, open(checkpoint_data_file, "wb"))
        else:
            await self.run_circ(circuit, data)

    async def run_circ(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug(
            'Converting single-qudit general gates to VariableUnitaryGate.',
        )
        for cycle, op in circuit.operations_with_cycles():
            if (
                op.gate.num_qudits == 1 and (
                    isinstance(op.gate, GeneralGate)
                    or self.convert_all_single_qudit_gates
                )
            ):
                params = VariableUnitaryGate.get_params(op.get_unitary())
                point = CircuitPoint(cycle, op.location[0])
                vgate = VariableUnitaryGate(op.num_qudits, op.radixes)
                circuit.replace_gate(
                    point, vgate, op.location, params,
                )
