from bqskit.ir.circuit import Circuit
from sys import argv
import numpy as np
# Generate a super ensemble for some error bounds
from bqskit.passes import *
import pickle

from bqskit.qis.unitary import UnitaryMatrix

from util import load_circuit

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

def get_circ_unitary_jiggle(circ_file):
    global basic_circ
    params: Circuit = pickle.load(open(circ_file, "rb"))
    return basic_circ.get_unitary(params)

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

    # bias = (ensemble_mean - target)

    # mean_unitary = UnitaryMatrix(true_mean, check_arguments=False)

    # print("Calculating Vars")

    # with mp.Pool() as pool:
    #     vars = pool.map(mean_unitary.get_frobenius_distance, ensemble)
    #     var = np.mean(vars)

    # print("Finished Calculating Vars")

    # covar = 0

    # # Get all subsets
    # subsets = []
    # for i in range(1, M):
    #     for j in range(i):
    #         subsets.append((ensemble[i] - true_mean, ensemble[j] - true_mean))

    # print("Created Subsets")

    # with mp.Pool() as pool:
    #     covars = pool.map(get_covar_elem, subsets)

    # print("Calculated")

    # covar = np.sum(covars) / M / (M - 1)

    return ensemble_mean #,  var, covar


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
    global basic_circ

    circ_name = argv[1]
    timestep = int(argv[2])

    circ = load_circuit(circ_name)
    # Store approximate solutions
    all_utries = []

    all_tols = [1,1.3,2,3,4,5,6]

    for tol in all_tols:
        # for i in range(tol + 1, tol + 5):
        # dir = f"ensemble_approx_circuits_qfactor/{method}/{circ_type}/{tol}/{timestep}/params_*_{tol}.pickle"
        if method.startswith("jiggle"):
            basic_circ_2 = pickle.load(open(f"ensemble_approx_circuits_qfactor/{method}/{circ_type}/jiggled_circs/2/6/{timestep}/jiggled_circ.pickle", "rb"))
            basic_circ_3 = pickle.load(open(f"ensemble_approx_circuits_qfactor/{method}/{circ_type}/jiggled_circs/3/6/{timestep}/jiggled_circ.pickle", "rb"))
        # print(dir)
        print("Got Circ")

        dir2 = f"ensemble_approx_circuits_qfactor/{method}/{circ_type}/{tol}/{timestep}/params_*2*.pickle"
        dir3 = f"ensemble_approx_circuits_qfactor/{method}/{circ_type}/{tol}/{timestep}/params_*3*.pickle"
        print(dir2)
        circ_files_2 = glob.glob(dir2)[:1000]
        circ_files_3 = glob.glob(dir3)


        # print(len(circ_files))
        basic_circ = basic_circ_2

        with mp.Pool() as pool:
            if method.startswith("jiggle"):
                utries = pool.map(get_circ_unitary_jiggle, circ_files_2)
                # all_utries.extend(utries)
                # basic_circ = basic_circ_3
                # utries = pool.map(get_circ_unitary_jiggle, circ_files_3)
                # utries = pool.map(get_circ_unitary_jiggle, circ_files)
            else:
                utries = pool.map(get_circ_unitary, circ_files)
        all_utries.extend(utries)
        
    print("------------------")
    print(len(all_utries))
    full_e1, full_e2, true_mean = get_upperbound_error_mean(all_utries, target)

    print(full_e1, full_e2)

    # # Get random sub samples - Vary this in first plot
    M = int(argv[5])

    num_points = 100

    # Vary this as well - K
    num_ensembles = 400

    # vars = []
    # covars = []
    # tvds = []
    # magnetizations = []

    tot_e1_uppers = []
    tot_e1_actuals = []
    tot_e2_uppers = []

    Ms = []

    # utry_inds = np.arange(len(all_utries))
    for pt in range(num_points):
        errors = []
        errors_2 = []
        biases = []
        for num in range(num_ensembles):
            # print(num)
            ensemble = random.sample(all_utries, M)

            # bias, var, cov = get_bias_var_covar(ensemble, target)
            # bias, var, cov = get_bias_var_covar_fast(ensemble, target, true_mean)
            bias = get_bias_var_covar_fast(ensemble, target, true_mean)

            e1, e2, sample_mean = get_upperbound_error_mean(ensemble, target)

            tvd = get_tvd_magnetization(ensemble, target)

            errors.append(e1)
            errors_2.append(e2)
            biases.append(bias)

            print(e1)
            print(e2)

            # vars.append(var)
            # covars.append(cov)
            # tvds.append(tvd)
        tot_e1_uppers.append(np.mean(errors))
        actual_mean = np.mean(biases, axis = 0)
        actual_e1 = target.get_frobenius_distance(actual_mean)
        tot_e1_actuals.append(np.sqrt(actual_e1))
        tot_e2_uppers.append(np.mean(errors_2))
        Ms.append(M)

        print("Got pt", np.mean(errors), np.sqrt(actual_e1))

    # print(biases)

    # final_bias_dist = UnitaryMatrix(np.mean(biases, axis=0), check_arguments=False).get_frobenius_norm()
    # final_var = np.mean(vars)
    # final_covar = np.mean(covars)
    # final_tvd = np.mean(tvds)

    # final_sample_err = np.mean(errors)

    # print("Sample Stats:", final_sample_err, final_bias_dist, final_var, final_covar)

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

    # ensemble_dict = {}

    # ensemble_dict["errrors"] = errors
    # ensemble_dict["biases"] = biases
    # ensemble_dict["variances"] = vars
    # ensemble_dict["covariances"] = covars
    # ensemble_dict["final_bias_dist"] = final_bias_dist
    # ensemble_dict["final_var"] = final_var
    # ensemble_dict["final_covar"] = final_covar

    # pickle.dump(ensemble_dict, open(f"{circ_type}_{method}_{tol}_{M}.data", "wb"))

    fig, ax = plt.subplots(1, 1)

    ax.scatter(tot_e2_uppers, tot_e1_uppers, label="E1")
    ax.scatter(tot_e2_uppers, tot_e1_actuals, c="red", label = "Full Distance")

    e2_min = min(tot_e2_uppers)
    e2_max = max(tot_e2_uppers)

    xs = np.linspace(e2_min, e2_max, 100)
    ys = xs ** (2)

    ax.plot(xs, ys)

    ax.set_xlabel("E2s")
    ax.set_ylabel("E1s")
    ax.set_yscale("log")
    ax.legend()


    fig.savefig(f"{circ_type}_{method}_errors_comp_rev.png")



    # fig, ax = plt.subplots(1, 1)

    # ax.scatter(Ms, tot_e2_uppers)
    # # ax.scatter(tot_e2_uppers, tot_e1_actuals, c="red")
    # ax.set_xlabel("M")
    # ax.set_ylabel("E2s")
    # ax.set_yscale("log")


    # fig.savefig(f"{circ_type}_E2vM.png")

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