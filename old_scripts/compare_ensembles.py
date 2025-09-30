from bqskit.ir.circuit import Circuit
from sys import argv
from bqskit import compile
import numpy as np
# Generate a super ensemble for some error bounds
from bqskit.passes import *
import pickle

from bqskit.qis.unitary import UnitaryMatrix

from pathlib import Path

import glob

import matplotlib.pyplot as plt
import random

import pandas as pd

import multiprocessing as mp


from bqskit.utils.math import canonical_unitary, global_phase

import itertools


def get_upperbound_error(unitaries: list[UnitaryMatrix], target):
    np_uns = [u.numpy for u in unitaries]
    mean = np.mean(np_uns, axis=0)

    target_unitary = UnitaryMatrix(target)
    mean_unitary = UnitaryMatrix(mean, check_arguments=False)

    print("Starting MP")

    with mp.Pool() as pool:
        errors = pool.map(target_unitary.get_frobenius_distance, unitaries)
        vars = pool.map(mean_unitary.get_frobenius_distance, unitaries)

    print("Finishing")

    return np.sqrt(np.mean(errors)), np.sqrt(np.mean(vars))

def get_covar_elem(matrices):
    A, B = matrices
    return 2*np.real(np.sum(np.multiply(A.conj(), B)))

def get_bias_var_covar(ensemble: list[UnitaryMatrix], target):
    # ensemble is of size M
    M = len(ensemble)
    ensemble_mean = np.mean(ensemble, axis=0)
    bias = UnitaryMatrix(target).get_frobenius_distance(ensemble_mean)

    mean_unitary = UnitaryMatrix(ensemble_mean, check_arguments=False)

    print("Calculating Vars")

    with mp.Pool() as pool:
        vars = pool.map(mean_unitary.get_frobenius_distance, ensemble)
        var = np.mean(vars)

    print("Finished Calculating Vars")

    covar = 0

    # Get all subsets
    subsets = []

    ensemble_diffs = [A - ensemble_mean for A in ensemble]

    for i in range(1, M):
        for j in range(i):
            subsets.append((ensemble_diffs[i], ensemble_diffs[j]))

    print("Created Subsets")

    with mp.Pool() as pool:
        covars = pool.map(get_covar_elem, subsets)

    print("Calculated")

    covar = np.sum(covars) / M / (M - 1)

    return bias, var, covar


def get_circ_unitary(circ_file):
    circ: Circuit = pickle.load(open(circ_file, "rb"))
    return circ.get_unitary()

# Circ 
if __name__ == '__main__':

    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)
    all_data = []

    for circ_type in ["Heisenberg", "TFIM"]:
        if circ_type == "TFIM":
            initial_circ = Circuit.from_file("ensemble_benchmarks/tfim_3.qasm")
            # target = np.loadtxt("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/tfim_4-1.unitary", dtype=np.complex128)
        elif circ_type == "Heisenberg":
            initial_circ = Circuit.from_file("ensemble_benchmarks/heisenberg_3.qasm")
            # target = np.loadtxt("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/tfim_4-1.unitary", dtype=np.complex128)
        else:
            target = np.loadtxt("ensemble_benchmarks/qite_3.unitary", dtype=np.complex128)
            initial_circ = Circuit.from_unitary(target)

        for method in ["leap", "treescan"]:
            target = initial_circ.get_unitary()
            for tol in range(1, 7):
                dir = f"ensemble_approx_circuits_frobenius/{method}/{circ_type}/{tol}/*.pickle"
                circ_files = glob.glob(dir)

                with mp.Pool() as pool:
                    all_utries = pool.map(get_circ_unitary, circ_files)
                
                print("------------------")
                full_e1, full_e2 = get_upperbound_error(all_utries, target)

                bias, var, covar = get_bias_var_covar(all_utries, target)

                actual_tol = 10 ** (tol / -2)

                all_data.append([circ_type, method, actual_tol, full_e1, full_e2, bias, var, covar])


    pickle.dump(all_data, open("all_data.pickle", "wb"))

    df = pd.DataFrame(all_data, columns=["Model", "Method", "Targeted Distance", "e1", "e2", "bias", "var", "covar"])

    df.to_csv("all_ensemble_data.csv")

    # Now try to create a smart ensemble!

    # ensemble = []
    # remaining_utries = all_utries.copy()
    # np.random.shuffle(remaining_utries)

    # while len(ensemble) < M:
    #     next_utry = remaining_utries.pop(0)

    #     ensemble.append(next_utry)

    #     _, _ , cov = get_bias_var_covar(ensemble, all_utries, target)

    #     if cov > 0:
    #         # Do not add this value
    #         ensemble.pop(-1)

    
    # Final Ensemble!!