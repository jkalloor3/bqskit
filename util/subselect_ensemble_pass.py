"""This module implements the ToU3Pass."""
from __future__ import annotations

import logging

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


    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        all_ensembles = []
        original_ensembles = data["ensemble"]

        if "finished_subselect" in data:
            print("Already Subselected", flush=True)
            return

        for ensemble in data["ensemble"]:

            ensemble_vec = np.array([c.get_unitary().get_flat_vector() for c in ensemble])
            # print("Running Subselect Ensemble Pass!", flush=True)
            # print(ensemble_vec.shape, flush=True)
            # print(ensemble_vec[0], flush=True)
            # print(ensemble_vec[1], flush=True)
            # pca = PCA(n_components=self.pca_components).fit_transform(ensemble)
            # tsne = TSNE(n_components=self.tsne_components, method='exact').fit_transform(pca)
            k_means = KMeans(n_clusters=self.num_circs, random_state=0, n_init="auto").fit(ensemble_vec)
            
            new_ensemble_inds = SubselectEnsemblePass.get_inds(k_means.labels_, self.num_circs)
            new_ensemble = [ensemble[i] for i in new_ensemble_inds]

            print("Post sub-select Ensemble Size", len(new_ensemble), flush=True)

            all_ensembles.append(new_ensemble)

        data["original_ensemble"] = original_ensembles
        data["ensemble"] = all_ensembles

        if "checkpoint_dir" in data:
            data["finished_subselect"] = True
            checkpoint_data_file = data["checkpoint_data_file"]
            pickle.dump(data, open(checkpoint_data_file, "wb"))

        return

        
