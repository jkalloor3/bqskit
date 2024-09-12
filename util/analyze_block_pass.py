"""This module implements the InstantiateCount pass"""
from __future__ import annotations

import logging
from typing import Any

from bqskit.ir import Gate, Circuit
from bqskit.ir.gates import RZGate, U3Gate, FixedRZGate
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



class TCountPass(BasePass):
    def __init__(
        self,
        t_gates_per_rz: int,
        count_ensemble: bool = False
    ) -> None:
        """
        Construct a Instantiate Count pass and then 

        """
        self.t_gates_per_rz = t_gates_per_rz
        self.count_ensemble = count_ensemble
        return

    async def run(
            self, 
            circuit : Circuit, 
            data: PassData
    ) -> None:
        
        print(circuit.gate_counts)
        

        # TODO: Call Gridsynth here
        if not self.count_ensemble:
            data["circuit_t_count"] = circuit.num_params * self.t_gates_per_rz
            final_t_count = data["circuit_t_count"]

        else:
            # Get all the circuits in the ensemble
            ensemble = data["final_ensemble"]
            t_counts = []
            for circ in ensemble:
                t_counts.append(circ.num_params * self.t_gates_per_rz)
            data["ensemble_t_counts"] = t_counts
            final_t_count = np.mean(t_counts)

        print("T Count", final_t_count)