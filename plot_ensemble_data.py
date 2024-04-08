from bqskit.ir.circuit import Circuit
from sys import argv
from bqskit import compile
import numpy as np
# Generate a super ensemble for some error bounds
from bqskit.passes import *
import pickle

from bqskit.qis.unitary import UnitaryMatrix

from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator, HilbertSchmidtCostGenerator

from pathlib import Path

import glob

import matplotlib.pyplot as plt
import random

import multiprocessing as mp

from sklearn.decomposition import PCA


def transform_mat_to_vec(unitary: UnitaryMatrix):
    vec = unitary.flatten()
    re_vec = np.abs(vec)
    im_vec = np.angle(vec)

    return np.hstack([re_vec, im_vec])


def get_circ_unitary_diff_jiggle(circ_file):
    global basic_circ
    global target
    params: Circuit = pickle.load(open(circ_file, "rb"))
    return target - basic_circ.get_unitary(params)

def get_circ_unitary_diff(circ_file):
    global target
    circ: Circuit = pickle.load(open(circ_file, "rb"))
    return target - circ.get_unitary()

def get_unitary_diff(circ_file):
    global target
    utry: UnitaryMatrix = pickle.load(open(circ_file, "rb"))
    # print(utry)
    return target - utry

def get_covar_elem(matrices):
    A, B = matrices
    elem =  2*np.real(np.trace(A.conj().T @ B))
    return elem

# Circ 
if __name__ == '__main__':
    global basic_circ
    global target

    circ_type = argv[1]
    timestep = int(argv[2])
    method = argv[3]
    tol = int(argv[4])
    block_size = int(argv[5])
    if len(argv) == 8:
        prev_tol = argv[6]
        prev_block_size = argv[7]
    else:
        prev_tol = None
        prev_block_size = None

    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)

    actual_target = None

    if circ_type == "TFIM":
        initial_circ = Circuit.from_file("ensemble_benchmarks/tfim_3.qasm")
        # target = np.loadtxt("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/tfim_4-1.unitary", dtype=np.complex128)
    elif circ_type == "Heisenberg":
        initial_circ = Circuit.from_file("ensemble_benchmarks/heisenberg_3.qasm")
        # target = np.loadtxt("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/tfim_4-1.unitary", dtype=np.complex128)
    elif circ_type == "Heisenberg_7" or circ_type == "TFXY_8":
        initial_circ = Circuit.from_file(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/{circ_type}/{circ_type}_{timestep}.qasm")
        initial_circ.remove_all_measurements()
        if prev_tol:
            actual_initial_circ: Circuit = pickle.load(open(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_approx_circuits_qfactor/gpu_real/{circ_type}/jiggled_circs/{prev_tol}/{prev_block_size}/{timestep}/jiggled_circ.pickle", "rb"))
            target = initial_circ.get_unitary()
            print("ORIG GPU DIST", actual_initial_circ.get_unitary().get_frobenius_distance(target))
            print(f"Orig Depth: {initial_circ.depth}, New Depth: {actual_initial_circ.depth}")
            print(f"Orig Count: {initial_circ.num_operations}, New Count: {actual_initial_circ.num_operations}")
            initial_circ = actual_initial_circ
    elif circ_type == "Hubbard":
        initial_circ = Circuit.from_file("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/hubbard_4.qasm")
    elif circ_type == "TFXY":
        initial_circ = Circuit.from_file("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/tfxy_6.qasm")
    else:
        target = np.loadtxt("ensemble_benchmarks/qite_3.unitary", dtype=np.complex128)
        initial_circ = Circuit.from_unitary(target)

    if actual_target:
        target = actual_target
    else:
        target = initial_circ.get_unitary()

    method = argv[3]
    max_tol = float(argv[4])
    # Store approximate solutions
    all_utries = []
    basic_circs = []
    circ_files = []

    min_tol = 1

    if method.startswith("jiggle"):
        # for tol in range(min_tol, max_tol):
        # tol = min_tol
        if method.startswith("jiggle"):
            basic_circ: Circuit = pickle.load(open(f"ensemble_approx_circuits_qfactor/gpu_real/{circ_type}/jiggled_circs/7/6/{timestep}/jiggled_circ.pickle", "rb"))
            print(dir)
            print("Got Circ")
            print(basic_circ.gate_counts)
        
        dir = f"ensemble_approx_circuits_qfactor/{method}/{circ_type}/{tol}/{timestep}/params_*.pickle"
        circ_files = glob.glob(dir)[:500]

        with mp.Pool() as pool:
            if method.startswith("jiggle"):
                utries = pool.map(get_circ_unitary_diff_jiggle, circ_files)
            else:
                utries = pool.map(get_circ_unitary_diff, circ_files)
            # circ_files.extend(glob.glob(dir)[:1000])
                
        all_utries.extend(utries)
    elif method == "noise":
        dir = f"ensemble_approx_circuits_qfactor/{method}/{circ_type}/{int(tol)}/{block_size}/utry_*.pickle"
        circ_files = glob.glob(dir)[:2000]
        with mp.Pool() as pool:
            all_utries = pool.map(get_unitary_diff, circ_files)
        
    print("------------------")
    print(len(all_utries))

    # Now, run PCA on all of the differences to get 2-vectors
    with mp.Pool() as pool:
        X = pool.map(transform_mat_to_vec, all_utries)

    # Perform PCA with two components
    pca = PCA(n_components=2)
    X = np.array(X)
    print(X.shape)
    X_pca = pca.fit_transform(X)

    x_mean = np.mean(X_pca, axis=0)

    print(x_mean)

    # Visualize the results
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c="r", cmap='viridis', edgecolor='k', s=60)
    plt.scatter(x_mean[0], x_mean[1], c= "b")
    plt.title('PCA of Unitary Diffs')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    # plt.show()
    plt.savefig(f"pca_{circ_type}_{timestep}_{method}_combo_{min_tol}_to_{max_tol}.png")


    # ys = xs ** (2)

    # ax.plot(xs, ys)

    # ax.set_xlabel("E2s")
    # ax.set_ylabel("E1s")
    # ax.set_yscale("log")


    # fig.savefig(f"{circ_type}_{method}_errors_comp_{min_tol}_to_{max_tol}_rev.png")

