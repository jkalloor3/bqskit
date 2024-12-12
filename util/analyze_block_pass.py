"""This module implements the InstantiateCount pass"""
from __future__ import annotations

import logging
from typing import Any

from bqskit.passes import ForEachBlockPass
from bqskit.ir import Gate, Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
import numpy as np
import matplotlib.pyplot as plt
import pickle


class AnalyzeBlockPass(BasePass):
    def __init__(
        self,
        gate_to_count: Gate = CNOTGate(),
    ) -> None:
        """
        Construct a Instantiate Count pass and then 

        """
        self.gate_to_count = gate_to_count
        return
     
    async def run(
            self, 
            circ : Circuit, 
            data: PassData
    ) -> None:
            
        # unitary = circuit.get_unitary()

        # Get Unitary structure quantities
        # coeffs = unitary.schmidt_coefficients
        # data["schmidt_number"] = len(coeffs[np.nonzero(coeffs)])
        # _logger.debug(unitary)
        # _logger.debug(coeffs)
        # data["entanglement_entropy"] = unitary.entanglement_entropy() 

        # Get circuit structure quantitieså
        circuit = circ.copy()
        circuit.unfold_all()
        data["depth"] = circuit.depth
        data["2q_count"] = circuit.count(self.gate_to_count)
        data["num_gates"] = circuit.num_operations
        data["free_params"] = circuit.num_params
        data["num_qubits"] = circuit.num_qudits

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


class MakeHistogramPass(BasePass):

    def create_histogram(counts: list[list], labels: list[str], filename: str):
        fig, axes = plt.subplots(1, len(counts), figsize=(5 * len(counts), 5))
        axes: list[plt.Axes] = list(axes)
        for i, count in enumerate(counts):
            axes[i].hist(count)
            axes[i].set_ylabel(labels[i])
        
        fig.savefig(filename)


    async def run(self, circuit: Circuit, data: PassData) -> None:
        block_data = data[ForEachBlockPass.key][0]
        counts = []
        depths = []
        params = []
        widths = []

        for i, block in enumerate(block_data):
            counts.append(block["2q_count"])
            depths.append(block["depth"])
            params.append(block["free_params"])
            widths.append(block["num_qubits"])

        save_data_dir = data["checkpoint_dir"]
        filename = f"{save_data_dir}/block_data.png"

        data["2q_counts"] = counts
        data["depths"] = depths
        data["params"] = params
        data["widths"] = widths

        save_data_file = data["checkpoint_data_file"]
        if save_data_file:
            pickle.dump(data, open(save_data_file, "wb"))

        
        MakeHistogramPass.create_histogram([counts, depths, params, widths], 
                                           ["Two Qubit Count", "Depth", 
                                            "Free Params", "Width"], 
                                            filename)