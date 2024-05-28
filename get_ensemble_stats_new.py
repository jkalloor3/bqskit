from bqskit.ir.circuit import Circuit
from sys import argv
import numpy as np
# Generate a super ensemble for some error bounds
from bqskit.passes import *
import pickle

from bqskit.qis.unitary import UnitaryMatrix

from util import get_upperbound_error_mean, get_tvd_magnetization, load_circuit, get_ensemble_mean, load_compiled_circuits, get_unitary

import glob

import matplotlib.pyplot as plt
import random

import multiprocessing as mp

# Circ 
if __name__ == '__main__':
    global basic_circ

    circ_name = argv[1]
    timestep = int(argv[2])
    tol = int(argv[3])
    # Get random sub samples - Vary this in first plot
    M = int(argv[4])

    circ = load_circuit(circ_name)
    target = circ.get_unitary()
    circs = load_compiled_circuits(circ_name, tol, timestep)

    print("Got Circuits")
    with mp.Pool() as pool:
        all_utries = pool.map(get_unitary, circs)
        
    print("------------------")
    print(len(all_utries))
    full_e1, full_e2, true_mean = get_upperbound_error_mean(all_utries, target)

    print(full_e1, full_e2)

    num_points = 100

    # Vary this as well - K
    num_ensembles = 400

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

            bias = get_ensemble_mean(ensemble)

            e1, e2, sample_mean = get_upperbound_error_mean(ensemble, target)

            tvd = get_tvd_magnetization(ensemble, target)

            errors.append(e1)
            errors_2.append(e2)
            biases.append(bias)

            print(e1)
            print(e2)

        tot_e1_uppers.append(np.mean(errors))
        actual_mean = np.mean(biases, axis = 0)
        actual_e1 = target.get_frobenius_distance(actual_mean)
        tot_e1_actuals.append(np.sqrt(actual_e1))
        tot_e2_uppers.append(np.mean(errors_2))
        Ms.append(M)

        print("Got pt", np.mean(errors), np.sqrt(actual_e1))

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


    fig.savefig(f"{circ_name}_{timestep}_{tol}_errors_comp_rev.png")