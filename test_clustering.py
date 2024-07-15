from bqskit.ir.circuit import Circuit
from sys import argv
import numpy as np
# Generate a super ensemble for some error bounds
from bqskit.passes import *
from bqskit.qis import UnitaryMatrix

from util import get_upperbound_error_mean_vec, load_circuit, load_compiled_circuits, get_unitary_vec, get_average_distance_vec, get_upperbound_error_mean, get_average_distance

import matplotlib.pyplot as plt

import multiprocessing as mp

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from qpsolvers import solve_qp


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


def get_quad_subensemble(ensemble: list[np.ndarray], V: np.ndarray, ensemble_size = 1000):
        M = len(ensemble)
        print(ensemble[0].shape)
        print(V.shape)
        tr_V_Us = np.zeros(M)
        tr_Us = np.zeros((M, M))

        for jj in range(M):
            tr_V_Us[jj] = np.trace(V.conj().T @ ensemble[jj])
            for kk in range(M):
                tr_Us[jj, kk] = np.trace((ensemble[jj].conj().T * ensemble[kk]))

        f = -2 * np.real(tr_V_Us)
        H = 2 * np.real(tr_Us)

        # ev = np.linalg.eigvals(H)
        # isposdef = np.all(ev > 0)
        # while not isposdef:
        #     if np.abs(np.min(ev)) < 1e-10:
        #         print('H not positive definite. Perturbing...')
        #         H += 1e-10 * np.eye(M)
        #     else:
        #         print('H not positive definite by a lot!')
        #         break
        #     ev = np.linalg.eigvals(H)
        #     isposdef = np.all(ev > 0)

        Aeq = np.ones((1, M))
        beq = np.array([1])
        lbound = np.zeros(M)
        ubound = np.ones(M)

        x = solve_qp(H, f, None, None, Aeq, beq, lbound, ubound, solver='clarabel')

        # Sample according to x
        sample = np.random.choice(M, ensemble_size, p=x, replace=False)
        return sample

def get_all_subensembles(ensemble: np.ndarray[np.float128], ensemble_size=1000, pca_components: int = 90, tsne_components: int = 32):
    rand_inds = np.random.choice(range(len(ensemble)), ensemble_size, replace=False)
    all_ens = []

    rand_ensemble = ensemble[rand_inds]
    all_ens.append(rand_ensemble)
    labels = ["Random"]

    # Do K-means
    k_means = KMeans(n_clusters=ensemble_size, random_state=0, n_init="auto").fit(ensemble)
    k_ens = ensemble[get_inds(k_means.labels_, ensemble_size)]
    all_ens.append(k_ens)
    labels.append("Full K-Means")
    
    # Do Agg
    agg = AgglomerativeClustering(n_clusters=ensemble_size).fit(ensemble)
    agg_ens = ensemble[get_inds(agg.labels_, ensemble_size)]
    all_ens.append(agg_ens)
    labels.append("Full Agglomerative")

    # Do PCA and K-Means
    pca = PCA(n_components=pca_components).fit_transform(ensemble)
    # Do PCA and Agg

    k_means = KMeans(n_clusters=ensemble_size, random_state=0, n_init="auto").fit(pca)
    k_ens = ensemble[get_inds(k_means.labels_, ensemble_size)]
    all_ens.append(k_ens)
    labels.append("PCA K-Means")
    
    # Do Agg
    agg = AgglomerativeClustering(n_clusters=ensemble_size).fit(pca)
    agg_ens = ensemble[get_inds(agg.labels_, ensemble_size)]
    all_ens.append(agg_ens)
    labels.append("PCA Agglomerative")


    tsne = TSNE(n_components=tsne_components, method='exact').fit_transform(pca)
    # Do TSNE and K-Means
    # Do TSNE and Agg
    k_means = KMeans(n_clusters=ensemble_size, random_state=0, n_init="auto").fit(tsne)
    k_ens = ensemble[get_inds(k_means.labels_, ensemble_size)]
    all_ens.append(k_ens)
    labels.append("TSNE K-Means")
    
    # Do Agg
    agg = AgglomerativeClustering(n_clusters=ensemble_size).fit(tsne)
    agg_ens = ensemble[get_inds(agg.labels_, ensemble_size)]
    all_ens.append(agg_ens)
    labels.append("TSNE Agglomerative")

    return all_ens, labels


# Circ 
if __name__ == '__main__':
    circ_name = argv[1]
    timestep = int(argv[2])
    tol = int(argv[3])
    timestep = 0

    # tols  = [3,5,7]

    ensemble_sizes = [100, 500]

    colors = ["red", "blue", "green", "purple", "orange", "pink", "cyan", "magenta", "yellow"]
    markers = ["o", "s", "v", "D"]
    circ = load_circuit(circ_name)
    target_un = circ.get_unitary().numpy
    target = circ.get_unitary().get_flat_vector()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    min_val = 1e10
    max_val = -1e10

    circs = load_compiled_circuits(circ_name, tol, timestep, extra_str="_bounded_2")[:1000]
    print("Got Circuits")
    # with mp.Pool() as pool:
        # all_utries_vec = np.array(pool.map(get_unitary_vec, circs))

    all_utries: list[UnitaryMatrix] = [circ.get_unitary() for circ in circs]
    all_utries_vec = np.array([u.get_flat_vector() for u in all_utries])
    all_utries: list[np.ndarray] = [u.numpy for u in all_utries]
        
    print("------------------")
    print(len(all_utries_vec))
    print(all_utries[0].shape)
    full_e1, full_e2, true_mean = get_upperbound_error_mean_vec(all_utries_vec, target)
    avg_distance_all = get_average_distance_vec(all_utries_vec)


    # ax.scatter(full_e1, avg_distance_all, c="black", marker="*", label="Full Ensemble")
    for ind, ens_size in enumerate(ensemble_sizes):
        print("Calculating All Ensembles")
        quad_ens_inds = get_quad_subensemble(all_utries, target_un, ensemble_size=ens_size)
        quad_ens = all_utries_vec[quad_ens_inds]

        e1, _, _ = get_upperbound_error_mean_vec(quad_ens, target)
        avg_dist = get_average_distance_vec(quad_ens)
        all_xs = [e1]
        all_ys = [avg_dist]

        all_ens, labels = get_all_subensembles(all_utries_vec, ensemble_size=ens_size)

        labels.insert(0, "Quadratic Ensemble")

        for ens in all_ens:
            e1, _, _ = get_upperbound_error_mean_vec(ens, target)
            avg_dist = get_average_distance_vec(ens)
            all_xs.append(e1)
            all_ys.append(avg_dist)
        for i, x in enumerate(all_xs):
            ax.scatter(x, all_ys[i], c=colors[i], marker=markers[ind], label=f"Ens Size: {ens_size} {labels[i]}")

    ax.set_xlabel("E1")
    ax.set_ylabel("Average Distance")

    min_val = min(np.min(all_xs), np.min(all_ys))
    max_val = max(np.max(all_xs), np.max(all_ys))   
    full_min = min_val * 0.9
    full_max = max_val * 1.1

    ax.set_xlim(full_min, full_max)
    ax.set_ylim(full_min, full_max)

    ax.plot([full_min, full_max], [full_min, full_max], c="black", linestyle="--")

    ax.legend()
    fig.savefig(f"{circ_name}_{tol}_errors_comp_with_quad_swap.png")