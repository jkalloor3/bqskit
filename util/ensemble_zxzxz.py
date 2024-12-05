"""This module implements the LEAPSynthesisPass."""
from __future__ import annotations

import logging
from typing import Any
import pickle

import numpy as np
import os
import csv
from pathlib import Path

from bqskit.runtime import get_runtime
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.qis import UnitaryMatrix
from bqskit.ir.gates import CNOTGate, VariableUnitaryGate
from bqskit.passes import FullBlockZXZPass, TreeScanningGateRemovalPass, ToVariablePass, ToU3Pass

from util import normalized_frob_cost, normalized_gp_frob_cost

_logger = logging.getLogger(__name__)


class EnsembleZXZXZ(BasePass):
    """
    A pass implementing the LEAP search synthesis algorithm, but for
    an ensemble of targets.

    References:
        Ethan Smith, Marc G. Davis, Jeffrey M. Larson, Ed Younis,
        Lindsay Bassman, Wim Lavrijsen, and Costin Iancu. 2022. LEAP:
        Scaling Numerical Optimization Based Synthesis Using an
        Incremental Approach. ACM Transactions on Quantum Computing
        (June 2022). https://doi.org/10.1145/3548693
    """

    def __init__(
        self,
        extract_diagonal: bool = False,
        synthesis_epsilon: float = 1e-5,
        tree_depth: int = 2,
    ) -> None:
        self.extract_diagonal = extract_diagonal
        self.zxzxz = FullBlockZXZPass(min_qudit_size=2, 
                                      perform_extract=extract_diagonal,
                                      extract_epsilon=synthesis_epsilon * 1e-1)
        self.zxzxz_2 = FullBlockZXZPass(min_qudit_size=1, perform_extract=False,
                                    extract_epsilon=synthesis_epsilon * 1e-1)
        self.variable = ToVariablePass(convert_all_single_qudit_gates=True)
        self.u3 = ToU3Pass(convert_all_single_qubit_gates=True)
        self.scan = TreeScanningGateRemovalPass(success_threshold=synthesis_epsilon, tree_depth=tree_depth)
        self.synthesis_epsilon = synthesis_epsilon

    async def run_target(self, i: int, data: PassData) -> tuple[Circuit, float]:
        """Run the LEAP synthesis pass on a single target."""
        target = data["ensemble_targets"][i]
        checkpoint_data_file = data["checkpoint_data_file"].replace(".data", f"_{i}.data")
        # checkpoint_circ_file = data["checkpoint_circuit_file"].replace(".pickle", f"_{i}.pickle")
        if os.path.exists(checkpoint_data_file):
            new_data = pickle.load(open(checkpoint_data_file, "rb"))
            # un_circ = pickle.load(open(checkpoint_circ_file, "rb"))
        else:
            new_data = data.copy()
            new_data["checkpoint_data_file"] = checkpoint_data_file
            # new_data["checkpoint_circuit_file"] = checkpoint_circ_file
            new_data.target = target
            pickle.dump(new_data, open(checkpoint_data_file, "wb"))
            un_circ = Circuit(target.num_qudits)
            utry_params = np.concatenate((
                np.real(target.numpy).flatten(),
                np.imag(target.numpy).flatten(),
            ))
            un_circ.append_gate(
                VariableUnitaryGate(target.num_qudits),
                list(range(target.num_qudits)),
                utry_params,
            )

            await self.zxzxz.run(un_circ, new_data)
            # pickle.dump(un_circ, open(checkpoint_circ_file, "wb"))
            pickle.dump(new_data, open(checkpoint_data_file, "wb"))
        
        print("UN CIRC Gates: ", un_circ.gate_counts, flush=True)
        # if un_circ.num_qudits < 6:
            # Only run the scan pass if the circuit is small enough
            # await self.variable.run(un_circ, new_data)
            # await self.scan.run(un_circ, new_data)

        await self.zxzxz_2.run(un_circ, new_data)
        await self.u3.run(un_circ, new_data)

        print("Final Gate Counts: ", un_circ.gate_counts, flush=True)
        
        final_dist = normalized_frob_cost(un_circ.get_unitary(), data.target)
        return un_circ, final_dist


    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        circ_dists: list[tuple[Circuit, float]] = await get_runtime().map(
            self.run_target, 
            list(range(len(data["ensemble_targets"]))), 
            data=data
        )
        data["scan_sols"] = circ_dists
        data["ensemble"] = [circ for circ, _ in circ_dists]

        # Calculate the bias of the ensemble
        bias = np.mean([sol[0].get_unitary() for sol in circ_dists], axis=0)
        target_dists = [normalized_frob_cost(t, data.target) for t in data["ensemble_targets"]]
        avg_target_dist = np.mean(target_dists)
        actual_dists = [normalized_frob_cost(c.get_unitary(), data.target) for c, _ in circ_dists]
        avg_actual_dist = np.mean(actual_dists)
        bias_dist = normalized_frob_cost(bias, data.target)
        ratio = bias_dist / (avg_actual_dist ** 2)
        actual_counts = [c.count(CNOTGate()) for c, _ in circ_dists]
        save_data_file = data["checkpoint_data_file"]
        print("Save Data File: ", save_data_file, flush=True)
        if save_data_file:
            Path(save_data_file).parent.mkdir(parents=True, exist_ok=True)
            save_csv_file = save_data_file.replace(".data", ".csv")
            with open(save_csv_file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["Ensemble Size", "Hamiltonian Noise Epsilon", 
                                 "Synthesis Epsilon","Avg. Actual Distance", 
                                 "CNOT Count", "Bias", "Ratio"])
                writer.writerow([len(circ_dists), avg_target_dist, 
                                 self.synthesis_epsilon, avg_actual_dist, 
                                 np.mean(actual_counts), bias_dist, ratio])
                pickle.dump(data, open(save_data_file, "wb"))