"""This module implements the InstantiateCount pass"""
from __future__ import annotations

import logging
from typing import Any

from bqskit.ir import Gate, Circuit
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
import numpy as np
import matplotlib.pyplot as plt
from util import get_upperbound_error_mean_vec, load_circuit, load_compiled_circuits, get_unitary_vec, get_average_distance_vec


class AnalyzeDistributionPass(BasePass):
    def __init__(
        self,
        circ_name: str,
        tol: float
    ) -> None:
        """
        Construct a Instantiate Count pass and then 

        """
        self.circ_name = circ_name
        self.tol = tol
        return
     
    async def run(
            self, 
            circuit : Circuit, 
            data: PassData
    ) -> None:
            
        target = data.target.get_flat_vector()

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        orig_circs: list[Circuit] = data["original_ensemble"]
        proc_circs: list[Circuit] = data["ensemble"]
        print("Got Circuits")

        all_utries_vec = np.array([circ.get_unitar().get_flat_vector() for circ in orig_circs])
        
        print("------------------")
        print(len(all_utries_vec))
        full_e1, full_e2, true_mean = get_upperbound_error_mean_vec(all_utries_vec, target)
        avg_distance_all = get_average_distance_vec(all_utries_vec)

        min_val = min(full_e1, full_e2)
        max_val = max(full_e1, full_e2)


        ax.scatter(full_e1, avg_distance_all, c="black", marker="*", label="Full Ensemble")

        all_utries_vec_new = np.array([circ.get_unitar().get_flat_vector() for circ in orig_circs])
        
        print("------------------")
        print(len(all_utries_vec_new))
        full_e1, full_e2, true_mean = get_upperbound_error_mean_vec(all_utries_vec_new, target)
        avg_distance_all = get_average_distance_vec(all_utries_vec_new)

        min_val = min(min_val, full_e1, full_e2)
        max_val = max(max_val, full_e1, full_e2)


        ax.scatter(full_e1, avg_distance_all, c="red", marker="*", label="Full Ensemble")

        ax.set_xlabel("E1")
        ax.set_ylabel("Average Distance")
        full_min = min_val * 0.9
        full_max = max_val * 1.1

        ax.set_xlim(full_min, full_max)
        ax.set_ylim(full_min, full_max)

        ax.plot([full_min, full_max], [full_min, full_max], c="black", linestyle="--")

        block_id = data.get("block_num", "")

        ax.legend()
        fig.savefig(f"{self.circ_name}_{self.tol}_{block_id}_errors_comp.png")