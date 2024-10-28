"""This module implements the ToU3Pass."""
from __future__ import annotations

import logging
import csv

from sklearn.cluster import AgglomerativeClustering

from bqskit.compiler.basepass import BasePass
from bqskit.passes import ForEachBlockPass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit, CircuitPoint, Operation
from bqskit.runtime import get_runtime
from typing import Any
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator, HilbertSchmidtCostGenerator
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates import CNOTGate
from bqskit.qis import UnitaryMatrix
import numpy as np
from math import ceil
from itertools import chain

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle


_logger = logging.getLogger(__name__)

def bias_cost(utry: UnitaryMatrix, target: UnitaryMatrix):
    '''
    Calculates the normalized Frobenius distance between two unitaries
    '''
    diff = utry- target
    # This is Frob(u - v)
    cost = np.sqrt(np.real(np.trace(diff @ diff.conj().T)))

    N = utry.shape[0]
    cost = cost / np.sqrt(2 * N)

    # This quantity should be less than HS distance as defined by 
    # Quest Paper 
    return cost

def frobenius_cost(utry: UnitaryMatrix, target: UnitaryMatrix):
    '''
    Calculates the Frobenius distance between two unitaries
    '''
    diff = utry- target
    # This is Frob(u - v)
    cost = np.sqrt(np.real(np.trace(diff @ diff.conj().T)))

    return cost


class SubselectEnsemblePass(BasePass):
    """ Subselects a subset of the ensemble to use for further analysis. Uses K-means with PCA and TSNE.
        By default selects does Frobenius distance of circuit matrices as the distance metric."""

    def __init__(self, success_threshold = 1e-4, 
                 num_circs = 1000,
                 cost: CostFunctionGenerator = HilbertSchmidtCostGenerator()) -> None:
        """

        Args:
        """

        self.success_threshold = success_threshold
        self.num_circs = num_circs
        self.pca_components = 24
        self.tsne_components = 20
        self.get_all_data = True

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


    async def get_new_ensemble(self, ensemble: list[Circuit]) -> list[Circuit]:
        ensemble_vec = np.array([c.get_unitary().get_flat_vector() for c, _ in ensemble])
        print("Running Subselect Ensemble Pass!", flush=True)
        print(ensemble_vec.shape, flush=True)
        # print(ensemble_vec[0], flush=True)
        # print(ensemble_vec[1], flush=True)
        # pca = PCA(n_components=self.pca_components).fit_transform(ensemble_vec)
        # tsne = TSNE(n_components=self.tsne_components, method='exact').fit_transform(pca)
        k_means = KMeans(n_clusters=self.num_circs, random_state=0, n_init="auto").fit(ensemble_vec)
        
        new_ensemble_inds = SubselectEnsemblePass.get_inds(k_means.labels_, self.num_circs)
        new_ensemble = [ensemble[i][0] for i in new_ensemble_inds]

        print("Subselected Ensemble Size: ", len(new_ensemble_inds), flush=True)

        return new_ensemble


    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        if "finished_subselect" in data:
            print("Already Subselected", flush=True)
            return

        data["finished_subselect"] = False

        all_ensembles: list[list[Circuit]] = await get_runtime().map(self.get_new_ensemble, data["ensemble"])

        csv_dict = []
        ensemble_names = ["Random Sub-Sample", "Least CNOTs", "Medium CNOTs", "Valid CNOTs"]
        target = data.target
        for i,ens in enumerate(all_ensembles):
            ensemble_data = {}
            unitaries: list[UnitaryMatrix] = [x.get_unitary() for x in ens]
            e1s = [bias_cost(un, target) for un in unitaries]
            e1s_actual = [frobenius_cost(un, target) for un in unitaries]
            e1 = np.mean(e1s)
            e1_actual = np.mean(e1s_actual)
            mean_un = np.mean(unitaries, axis=0)
            orig_bias = bias_cost(mean_un, target)
            orig_bias_actual = frobenius_cost(mean_un, target)
            
            final_counts = [circ.count(CNOTGate()) for circ in ens]
            ensemble_data["Ensemble Generation Method"] = ensemble_names[i]
            ensemble_data["Epsilon"] = e1
            ensemble_data["Epsilon Actual"] = e1_actual
            ensemble_data["Avg CNOT Count after K Means"] = np.mean(final_counts)
            ensemble_data["Bias after K Means"] = orig_bias
            ensemble_data["Bias Actual after K Means"] = orig_bias_actual
            ensemble_data["Num Circs"] = len(ens)

            csv_dict.append(ensemble_data)

        data["sub_select_ensemble"] = all_ensembles

        if "checkpoint_dir" in data:
            # data["finished_subselect"] = True
            # data.pop("ensemble")
            checkpoint_data_file = data["checkpoint_data_file"]
            csv_file = checkpoint_data_file.replace(".data", ".csv_subselect3")
            writer = csv.DictWriter(open(csv_file, "w", newline=""), fieldnames=csv_dict[0].keys())
            writer.writeheader()
            for row in csv_dict:
                writer.writerow(row)
            # pickle.dump(data, open(checkpoint_data_file, "wb"))

        return

        
