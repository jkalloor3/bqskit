"""This module implements the ToU3Pass."""
from __future__ import annotations

import logging
import csv

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate, TGate, TdgGate
from bqskit.runtime import get_runtime
from typing import Any
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.gates import CNOTGate
from bqskit.qis import UnitaryMatrix
import numpy as np
from math import ceil
from itertools import chain

from .distance import normalized_frob_cost

import pickle

class SubselectEnsemblePass(BasePass):
    """ Subselects a subset of the ensemble to use for further analysis. Uses K-means with PCA and TSNE.
        By default selects does Frobenius distance of circuit matrices as the distance metric."""

    def __init__(self, success_threshold = 1e-4, 
                 num_circs = 1000,
                 count_t: bool = False
                 ) -> None:
        """

        Args:
        """

        self.success_threshold = success_threshold
        self.num_circs = num_circs
        self.pca_components = 24
        self.tsne_components = 20
        self.get_all_data = True
        self.count_t = count_t

    def get_inds(cluster_ids, k: int):
        inds = []
        all_inds = np.arange(len(cluster_ids))
        for id in range(k):
            id_inds = all_inds[cluster_ids == id]
            if len(id_inds) == 0:
                continue
            rand_ind = np.random.choice(id_inds)
            inds.append(rand_ind)

        return np.array(inds)

    def subselect_circs_by_bias(self, circuits: list[tuple[Circuit, float]], target: UnitaryMatrix) -> list[Circuit]:
        '''
        Given a list of circuits and distances, subselect the circuits
        that minimize the bias between the target and the mean unitary
        of the ensemble.


        The circuits are sorted by the gate reduction, so try to use
        the circuits that have the fewest gates to minimize the bias.
        '''

        unitaries: list[UnitaryMatrix] = [x[0].get_unitary() for x in circuits]

        # Initially randomly sort circs
        order = np.random.permutation(len(unitaries))

        # Grab first 25 as initial set
        inds = list(order[:25])
        avg_un = np.mean([unitaries[i] for i in inds], axis=0)
        bias = normalized_frob_cost(avg_un, target)
        
        cur_ind = 25
        nn = 1
        while cur_ind < len(unitaries) and len(inds) < 1000:
            num_to_check = min(nn, len(unitaries) - cur_ind)
            # Get next unitaries to try and add
            next_inds = [order[cur_ind + i] for i in range(num_to_check)]
            next_uns = [unitaries[i] for i in next_inds]

            # Try to add all of them
            num_inds = len(inds)
            new_avg_un = (num_inds / (num_inds + num_to_check)) * avg_un + (num_to_check / (num_inds + num_to_check)) * np.mean(next_uns, axis=0)
            new_bias = normalized_frob_cost(new_avg_un, target)

            # If the bias is less, add the unitaries
            if new_bias < bias:
                bias = new_bias
                avg_un = new_avg_un
                inds.extend(next_inds)

            cur_ind = cur_ind + num_to_check

        # Return the ensemble with the best bias
        new_circs = [circuits[i][0] for i in inds]
        return new_circs


    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        if "finished_subselect" in data:
            print("Already Subselected", flush=True)
            return

        data["finished_subselect"] = False

        
        all_ensembles: list[list[Circuit]] = await get_runtime().map(self.subselect_circs_by_bias, data["ensemble"], target=data.target)


        print("FINISHED SUBSELECTING", flush=True)
        # Sort all ensembles by average CNOT count
        all_ensembles = sorted(all_ensembles, key=lambda x: np.mean([circ.count(CNOTGate()) for circ in x]))
        avg_cnot_counts = [np.mean([circ.count(CNOTGate()) for circ in ens]) for ens in all_ensembles]
        print("Average CNOT Counts of Subselected Ensembles", avg_cnot_counts)

        data["sub_select_ensemble"] = all_ensembles

        if "checkpoint_dir" in data:
            data["finished_subselect"] = True
            data.pop("ensemble")
            checkpoint_data_file = data["checkpoint_data_file"]
            pickle.dump(data, open(checkpoint_data_file, "wb"))

        return

        
