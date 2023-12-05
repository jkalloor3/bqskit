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


from bqskit.utils.math import canonical_unitary, global_phase

def get_upperbound_error(unitaries: list[UnitaryMatrix], target):
    np_uns = [u.numpy for u in unitaries]
    mean = np.mean(np_uns, axis=0)
    # print(UnitaryMatrix(target).get_distance_from(mean))
    mean_test = UnitaryMatrix(mean)
    print(UnitaryMatrix(mean_test).get_distance_from(target))
    errors = [u.get_distance_from(target) for u in unitaries]

    max_error = np.max(errors)
    max_variance = np.max([u.get_distance_from(mean) for u in unitaries])
    return np.sqrt(max_error), np.sqrt(max_variance)

def get_bias_var_covar(ensemble: list[UnitaryMatrix], all_unitaries: list[UnitaryMatrix], target):
    # ensemble is of size M
    M = len(ensemble)
    ensemble_mean = np.mean(ensemble, axis=0)
    bias = ensemble_mean - target

    var = np.mean([u.get_distance_from(ensemble_mean) for u in ensemble])
    covar = 0
    for i in range(M):
        A = UnitaryMatrix(ensemble[i] - ensemble_mean, check_arguments=False)
        for j in range(M):
            if j < i:
                B = ensemble[j] - ensemble_mean
                covar += 2*np.real(np.trace(A.conj().T @ B))

    covar *= ((1 /M), (1/(M-1)))

    return bias, var, covar


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
    else:
        target = np.loadtxt("ensemble_benchmarks/qite_3.unitary", dtype=np.complex128)
        initial_circ = Circuit.from_unitary(target)

    target = initial_circ.get_unitary()

    method = argv[2]
    tol = int(argv[3])
    # Store approximate solutions
    dir = f"ensemble_approx_circuits_correct/{method}/{circ_type}/{tol}/*.pickle"

    print(dir)

    circ_files = glob.glob(dir)

    print(len(circ_files))

    all_circs = []
    all_utries = []

    phase_diff = global_phase(target.numpy)

    for circ_file in circ_files:
        circ: Circuit = pickle.load(open(circ_file, "rb"))
        all_circs.append(circ)
        utry = circ.get_unitary()
        can_utry = UnitaryMatrix(canonical_unitary(utry.numpy))
        fixed_utry = UnitaryMatrix(can_utry.numpy * phase_diff)

        print(utry.get_distance_from(target, 2))
        # print(can_utry.get_distance_from(target))
        print(can_utry.get_frobenius_distance(canonical_unitary(target.numpy)))
        print(fixed_utry.get_frobenius_distance(target))
        print("--------------------")

    exit(0)
    print(get_upperbound_error(all_utries, target))


    # Get random sub samples

    M = 200
    num_ensembles = 150

    errors = []
    biases = []
    vars = []
    covars = []

    for _ in range(num_ensembles):
        ensemble = np.random.choice(all_utries, M, replace=False)

        bias, var, cov = get_bias_var_covar(ensemble, all_utries, target)

        e1, e2 = get_upperbound_error(ensemble, target)

        errors.append(e1)
        biases.append(bias)
        vars.append(var)
        covars.append(cov)

    final_bias = np.mean(biases)
    final_var = np.mean(vars)
    final_covar = np.mean(covars)

    final_sample_err = np.mean(errors)

    print("Sample Stats:", final_sample_err, final_bias, final_var, final_covar)

    # fig = plt.figure(20, 20)
    # axes: list[list[plt.Axes]] = fig.subplots(2, 2, sharex=True)

    # axes[0][0].plot(errors, biases)
    # axes[0][0].set_xlabel("e1 Error")
    # axes[0][0].set_ylabel("Bias")

    # axes[0][1].plot(errors, vars)
    # axes[0][1].set_xlabel("e1 Error")
    # axes[0][1].set_ylabel("Variance")

    # axes[1][0].plot(errors, covars)
    # axes[1][0].set_xlabel("e1 Error")
    # axes[1][0].set_ylabel("Covariance")

    # axes[1][1].plot(errors, biases)
    # axes[1][1].set_xlabel("e1 Error")
    # axes[1][1].set_ylabel("Bias")

    # # Now try to create a smart ensemble!

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