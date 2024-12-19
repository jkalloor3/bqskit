"""This module implements the InstantiateCount pass"""
from __future__ import annotations

import logging
from typing import Any

from bqskit.passes import ForEachBlockPass
from bqskit.ir import Gate, Circuit
from bqskit.ir.gates import CNOTGate, CircuitGate
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



def rem_outliers(points: np.ndarray, thresh=3.5) -> np.ndarray[bool]:
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score <= thresh

from collections import Counter

class MakeHistogramPass(BasePass):

    def create_histogram(counts: dict[str, list], filename: str):
        fig, axes = plt.subplots(1, len(counts.keys()), figsize=(5 * len(counts.keys()), 5))
        axes: list[plt.Axes] = list(axes)
        i = 0
        for label, count in counts.items():
            filtered = np.array(count)
            if not (label == "Widths"):
                filtered = filtered[rem_outliers(filtered)]
            axes[i].hist(filtered)
            axes[i].set_ylabel(label)
            i += 1
        
        fig.savefig(filename)

    def get_block_data(self, block: CircuitGate) -> dict[str, Any]:
        data = {}
        circ = block._circuit.copy()
        circ.unfold_all()
        data["2q_count"] = circ.count(CNOTGate())
        data["depth"] = circ.depth
        data["free_params"] = circ.num_params
        data["num_qubits"] = circ.num_qudits
        return data


    async def run(self, circuit: Circuit, data: PassData) -> None:
        all_data = {}
        all_data["2Q Count"] = []
        all_data["Depth"] = []
        all_data["Free Params"] = []
        all_data["Widths"] = []
        for op in circuit:
            if isinstance(op.gate, CircuitGate):
                block_data = self.get_block_data(op.gate)
                all_data["2Q Count"].append(block_data["2q_count"])
                all_data["Depth"].append(block_data["depth"])
                all_data["Free Params"].append(block_data["free_params"])
                all_data["Widths"].append(block_data["num_qubits"])

        save_data_file: str = data["checkpoint_data_file"]
        filename = save_data_file.replace(".data", ".png")
        if save_data_file:
            data.update(all_data)
            pickle.dump(data, open(save_data_file, "wb"))

        
        MakeHistogramPass.create_histogram(all_data,
                                            filename)