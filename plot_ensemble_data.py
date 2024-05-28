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

from util import load_circuit, load_compiled_circuits


def transform_mat_to_vec(unitary: UnitaryMatrix):
    vec = unitary.flatten()
    re_vec = np.abs(vec)
    im_vec = np.angle(vec)

    return np.hstack([re_vec, im_vec])


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
    global target

    circ_name = argv[1]
    timestep = int(argv[2])
    tol = int(argv[3])
    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)

    circ = load_circuit(circ_name)
    target = circ.get_unitary()

    # Store approximate solutions
    all_utries = []
    basic_circs = []
    circ_files = []

    min_tol = 1

    circuits = load_compiled_circuits(circ_name, tol, timestep)[:1000]
    # Get differences
    all_utries = [c.get_unitary() - target for c in circuits]

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
    plt.savefig(f"pca_{circ_name}_{timestep}_{tol}.png")



