"""This module implements the ApplyPlacement class."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
import random

_logger = logging.getLogger(__name__)


class ApplyPlacement(BasePass):
    """Place the circuit on the machine model."""

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        print("HITTING APPLY PLACEMENT!!")
        model = data.model
        placement = data.placement
        physical_circuit = Circuit(model.num_qudits, model.radixes)
        physical_circuit.append_circuit(circuit, placement)
        circuit.become(physical_circuit)
        basic_placement = list(range(circuit.num_qudits))
        random.shuffle(basic_placement)
        data.placement = basic_placement
        if 'final_mapping' in data:
            data['final_mapping'] = [placement[p] for p in range(circuit.num_qudits)]
        if 'initial_mapping' in data:
            data['initial_mapping'] = [placement[p] for p in range(circuit.num_qudits)]
