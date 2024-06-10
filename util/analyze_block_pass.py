"""This module implements the InstantiateCount pass"""
from __future__ import annotations

import logging
from typing import Any

from bqskit.ir import Gate, Circuit
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ext.supermarq import supermarq_parallelism
from bqskit.ir.gates import U3Gate, CNOTGate
from bqskit.runtime import get_runtime
from itertools import product
import numpy as np



class AnalyzeBlockPass(BasePass):
    def __init__(
        self,
        gate_to_count: Gate,
    ) -> None:
        """
        Construct a Instantiate Count pass and then 

        """
        self.gate_to_count = gate_to_count
        return
     
    async def run(
            self, 
            circuit : Circuit, 
            data: PassData
    ) -> None:
            
        unitary = circuit.get_unitary()

        # Get Unitary structure quantities
        # coeffs = unitary.schmidt_coefficients
        # data["schmidt_number"] = len(coeffs[np.nonzero(coeffs)])
        # _logger.debug(unitary)
        # _logger.debug(coeffs)
        # data["entanglement_entropy"] = unitary.entanglement_entropy() 

        # Get circuit structure quantities
        data["block_depth"] = circuit.depth
        data["block_twoq_count"] = circuit.count(self.gate_to_count)
        data["num_gates"] = circuit.num_operations
        data["num_qubits"] = circuit.num_qudits
        data["parallelism"] = supermarq_parallelism(circuit=circuit)