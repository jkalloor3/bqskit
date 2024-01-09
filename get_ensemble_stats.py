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

import multiprocessing as mp

def get_upperbound_error_mean(unitaries: list[UnitaryMatrix], target):
    np_uns = [u.numpy for u in unitaries]
    mean = np.mean(np_uns, axis=0)
    errors = [u.get_frobenius_distance(target) for u in unitaries]

    max_error = np.mean(errors)
    max_variance = np.mean([u.get_frobenius_distance(mean) for u in unitaries])
    return np.sqrt(max_error), np.sqrt(max_variance), mean

def get_tvd_magnetization(ensemble: list[UnitaryMatrix], target):
    ensemble_mean = np.mean(ensemble, axis=0)
    final_output = ensemble_mean[:, 0]
    target_output = target[:, 0]
    diff = np.sum(np.abs(final_output - target_output))
    return diff / 2

def get_bias_var_covar(ensemble: list[UnitaryMatrix], target):
    # ensemble is of size M
    M = len(ensemble)
    ensemble_mean = np.mean(ensemble, axis=0)
    bias = UnitaryMatrix(target).get_frobenius_distance(ensemble_mean)

    var = np.mean([u.get_frobenius_distance(ensemble_mean) for u in ensemble])
    covar = 0
    for i in range(M):
        A = UnitaryMatrix(ensemble[i] - ensemble_mean, check_arguments=False)
        for j in range(M):
            if j < i:
                B = ensemble[j] - ensemble_mean
                covar += 2*np.real(np.trace(A.conj().T @ B))

    covar *= (1 /M)*(1/(M-1))

    return bias, var, covar

def get_circ_unitary(circ_file):
    circ: Circuit = pickle.load(open(circ_file, "rb"))
    return circ.get_unitary()

def get_covar_elem(matrices):
    A, B = matrices
    elem =  2*np.real(np.trace(A.conj().T @ B))
    return elem

def get_bias_var_covar_fast(ensemble: list[UnitaryMatrix], target, true_mean):
    # ensemble is of size M
    M = len(ensemble)
    ensemble_mean = np.mean(ensemble, axis=0)
    # bias = UnitaryMatrix(target).get_frobenius_distance(ensemble_mean)

    bias = (ensemble_mean - target)

    mean_unitary = UnitaryMatrix(true_mean, check_arguments=False)

    print("Calculating Vars")

    with mp.Pool() as pool:
        vars = pool.map(mean_unitary.get_frobenius_distance, ensemble)
        var = np.mean(vars)

    print("Finished Calculating Vars")

    covar = 0

    # Get all subsets
    subsets = []
    for i in range(1, M):
        for j in range(i):
            subsets.append((ensemble[i] - true_mean, ensemble[j] - true_mean))

    print("Created Subsets")

    with mp.Pool() as pool:
        covars = pool.map(get_covar_elem, subsets)

    print("Calculated")

    covar = np.sum(covars) / M / (M - 1)

    return bias, var, covar


def get_covar_diff(ensemble: list[UnitaryMatrix], next_unitary: UnitaryMatrix):
    ensemble_mean = np.mean(ensemble + [next_unitary], axis=0)

    # Get all new terms in CoVariance
    subsets = []
    for i in ensemble:
        subsets.append((i - ensemble_mean, next_unitary - ensemble_mean))

    with mp.Pool() as pool:
        covars = pool.map(get_covar_elem, subsets)

    return np.mean(covars)


# Circ 
if __name__ == '__main__':
    circ_type = argv[1]

    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)

    if circ_type == "TFIM":
        initial_circ = Circuit.from_file("ensemble_benchmarks/tfim_3.qasm")
        # target = np.loadtxt("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/tfim_4-1.unitary", dtype=np.complex128)
    elif circ_type == "Heisenberg":
        initial_circ = Circuit.from_file("ensemble_benchmarks/heisenberg_3.qasm")
        # target = np.loadtxt("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/tfim_4-1.unitary", dtype=np.complex128)
    elif circ_type == "Heisenberg_7":
        initial_circ = Circuit.from_file("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/heisenberg7.qasm")
    elif circ_type == "Hubbard":
        initial_circ = Circuit.from_file("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/hubbard_4.qasm")
    else:
        target = np.loadtxt("ensemble_benchmarks/qite_3.unitary", dtype=np.complex128)
        initial_circ = Circuit.from_unitary(target)

    target = initial_circ.get_unitary()

    method = argv[2]
    tol = int(argv[3])
    # Store approximate solutions
    dir = f"ensemble_approx_circuits_frobenius/{method}/{circ_type}/{tol}/*.pickle"

    print(dir)

    circ_files = glob.glob(dir)

    print(len(circ_files))

    with mp.Pool() as pool:
        all_utries = pool.map(get_circ_unitary, circ_files)
    
    print("------------------")
    full_e1, full_e2, true_mean = get_upperbound_error_mean(all_utries, target)



    # # Get random sub samples - Vary this in first plot
    M = int(argv[4])

    # Vary this as well - K
    num_ensembles = 150

    errors = []
    biases = []
    vars = []
    covars = []
    tvds = []
    magnetizations = []

    # utry_inds = np.arange(len(all_utries))

    for num in range(num_ensembles):
        print(num)
        ensemble = random.sample(all_utries, M)

        # bias, var, cov = get_bias_var_covar(ensemble, target)
        bias, var, cov = get_bias_var_covar_fast(ensemble, target, true_mean)

        e1, e2, sample_mean = get_upperbound_error_mean(ensemble, target)

        tvd = get_tvd_magnetization(ensemble, target)

        errors.append(e1)
        biases.append(np.abs(bias))
        vars.append(var)
        covars.append(cov)
        tvds.append(tvd)

    print(biases)

    final_bias_dist = UnitaryMatrix(np.mean(biases, axis=0), check_arguments=False).get_frobenius_norm()
    final_var = np.mean(vars)
    final_covar = np.mean(covars)
    final_tvd = np.mean(tvds)

    final_sample_err = np.mean(errors)

    print("Sample Stats:", final_sample_err, final_bias_dist, final_var, final_covar)

    # fig = plt.figure(figsize=(26, 20))
    # axes: list[list[plt.Axes]] = fig.subplots(2, 2, sharex=True)

    # axes[0][0].scatter(errors, biases)
    # axes[0][0].set_xlabel("e1 Error", fontdict={"size": 20})
    # axes[0][0].set_ylabel("Bias^2", fontdict={"size": 20})
    # axes[0][0].axhline(full_e1 ** 2 / M, c="red")
    # axes[0][0].axhline(final_bias, c="blue")
    # # axes[0][0].axhline(full_e1 ** 4, c="red")

    # axes[0][1].scatter(errors, tvds)
    # axes[0][1].set_xlabel("e1 Error", fontdict={"size": 20})
    # axes[0][1].set_ylabel("TVD", fontdict={"size": 20})

    # axes[1][0].scatter(errors, vars)
    # axes[1][0].set_xlabel("e1 Error", fontdict={"size": 20})
    # axes[1][0].set_ylabel("Variance", fontdict={"size": 20})
    # axes[1][0].axhline(final_var, c="blue")
    # axes[1][0].axhline(full_e2 ** 2, c="red")

    # axes[1][1].scatter(errors, covars)
    # axes[1][1].set_xlabel("e1 Error", fontdict={"size": 20})
    # axes[1][1].set_ylabel("Covariance", fontdict={"size": 20})
    # axes[1][1].axhline(final_covar, c="blue")
    # axes[1][1].axhline(-1 * (full_e1 ** 2), c="red")

    ensemble_dict = {}

    ensemble_dict["errrors"] = errors
    ensemble_dict["biases"] = biases
    ensemble_dict["variances"] = vars
    ensemble_dict["covariances"] = covars
    ensemble_dict["final_bias_dist"] = final_bias_dist
    ensemble_dict["final_var"] = final_var
    ensemble_dict["final_covar"] = final_covar

    pickle.dump(ensemble_dict, open(f"{circ_type}_{method}_{tol}_{M}.data", "wb"))

    # Now try to create a smart ensemble!

    # ensemble = []
    # remaining_utries = all_utries.copy()
    # np.random.shuffle(remaining_utries)
    # ensemble.append(remaining_utries.pop(0))

    # failed = 0
    # while len(ensemble) < M:
    #     next_utry = remaining_utries.pop(0)

    #     cov = get_covar_diff(ensemble, next_utry)
    #     if cov > 0 and failed < 6:
    #         print("Positive Covariance!!!")
    #         failed += 1
    #         # Do not add this value
    #     else:
    #         ensemble.append(next_utry)
    #         failed = 0
    #         print(len(ensemble))


    # bias, var, cov = get_bias_var_covar_fast(ensemble, target)

    # e1, e2 = get_upperbound_error(ensemble, target)


    # axes[0][0].scatter([e1], [bias], c="red")
    # axes[1][0].scatter([e1], [var], c="red")
    # axes[1][1].scatter([e1], [cov], c="red")

    # for axs in axes:
    #     for ax in axs:
    #         ax.yaxis.get_offset_text().set_fontsize(20)
    #         for item in (ax.get_yticklabels()):
    #             item.set_fontsize(20)

    #         for item in (ax.get_xticklabels()):
    #             item.set_fontsize(20)

    # print("Sample Stats:", final_sample_err, final_bias, final_var, final_covar)
    
    # # Final Ensemble!!


    # fig.suptitle(f"{circ_type} {method} M: {M}, e1: {full_e1}, Ensemble avg e1: {np.mean(errors)}", fontsize=20)

    # # axes[0][1].plot(errors, np.imag(biases))
    # # axes[0][1].set_xlabel("e1 Error")
    # # axes[0][1].set_ylabel("Bias-Imaginary")

    # fig.savefig(f"{circ_type}_{method}_{tol}_{M}.png")