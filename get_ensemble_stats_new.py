from bqskit.ir.circuit import Circuit
from sys import argv
import numpy as np
# Generate a super ensemble for some error bounds
from bqskit.passes import *
import pickle

from bqskit.qis.unitary import UnitaryMatrix

from util import get_upperbound_error_mean, get_tvd_magnetization, load_circuit, get_ensemble_mean, load_compiled_circuits, get_unitary, get_average_distance

from bqskit.ir.opt.cost.functions import  HilbertSchmidtCostGenerator
import glob

import matplotlib.pyplot as plt
import random

import multiprocessing as mp

# Circ 
if __name__ == '__main__':
    circ_name = argv[1]
    timestep = int(argv[2])
    tol = int(argv[3])
    timestep = 0

    # tols  = [3,5,7]
    tols = [3]

    # unique_circs = [1, 5, 20, 100, 1000, 10000]
    unique_circs = [100]

    colors = ["red", "blue", "green", "purple", "orange", "black"]
    markers = ["o", "s", "v", "D"]
    circ = load_circuit(circ_name)
    target = circ.get_unitary()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    min_val = 1e10
    max_val = -1e10

    for i, unique_circ in enumerate(unique_circs):
        # for j, tol in enumerate(tols):
        try:
            circs = load_compiled_circuits(circ_name, tol, timestep, extra_str=f"_{unique_circ}_circ_final", ignore_timestep=True)
            print("Got Circuits")
            # with mp.Pool() as pool:
            #     all_utries = pool.map(get_unitary, circs)

            all_utries = [c.get_unitary() for c in circs]
                
            print("------------------")
            print(len(all_utries))
            full_e1, full_e2, true_mean = get_upperbound_error_mean(all_utries, target)
            # avg_distance = get_average_distance(all_utries)
            # ensemble = random.sample(all_utries, M)
            avg_distance = get_average_distance(all_utries)

            print(full_e1, avg_distance)

            min_val = min(min_val, min(full_e1, avg_distance))
            max_val = max(max_val, max(full_e1, avg_distance))

            ax.scatter(full_e1, avg_distance, c=colors[i], marker=markers[0], label=f"Unique Circuits: {unique_circ}, Tol: {tol}")
        except Exception as e:
            print(e)
            continue

    ax.set_xlabel("E1")
    ax.set_ylabel("Average Distance")

    # ax.set_yscale("log")
    # ax.set_xscale("log")

    # x_min, x_max = ax.get_xlim()
    # y_min, y_max = ax.get_ylim()

    full_min = min_val * 0.9
    full_max = max_val * 1.1

    ax.set_xlim(full_min, full_max)
    ax.set_ylim(full_min, full_max)

    ax.plot([full_min, full_max], [full_min, full_max], c="black", linestyle="--")

    ax.legend()
    fig.savefig(f"{circ_name}_{tol}_unique_circuits_errors_comp.png")

    # print(full_e1, full_e2, avg_distance)

    # num_points = 100

    # # Vary this as well - K
    # num_ensembles = 400

    # tot_e1_uppers = []
    # tot_e1_actuals = []
    # tot_e2_uppers = []

    # Ms = []

    # # utry_inds = np.arange(len(all_utries))
    # for pt in range(num_points):
    #     errors = []
    #     errors_2 = []
    #     biases = []
    #     for num in range(num_ensembles):
    #         # print(num)
    #         ensemble = random.sample(all_utries, M)

    #         bias = get_ensemble_mean(ensemble)

    #         e1, e2, sample_mean = get_upperbound_error_mean(ensemble, target)

    #         tvd = get_tvd_magnetization(ensemble, target)

    #         errors.append(e1)
    #         errors_2.append(e2)
    #         biases.append(bias)

    #         print(e1)
    #         print(e2)

    #     tot_e1_uppers.append(np.mean(errors))
    #     actual_mean = np.mean(biases, axis = 0)
    #     actual_e1 = target.get_frobenius_distance(actual_mean)
    #     tot_e1_actuals.append(np.sqrt(actual_e1))
    #     tot_e2_uppers.append(np.mean(errors_2))
    #     Ms.append(M)

    #     print("Got pt", np.mean(errors), np.sqrt(actual_e1))

    # fig, ax = plt.subplots(1, 1)

    # ax.scatter(tot_e2_uppers, tot_e1_uppers, label="E1")
    # ax.scatter(tot_e2_uppers, tot_e1_actuals, c="red", label = "Full Distance")

    # e2_min = min(tot_e2_uppers)
    # e2_max = max(tot_e2_uppers)

    # xs = np.linspace(e2_min, e2_max, 100)
    # ys = xs ** (2)

    # ax.plot(xs, ys)

    # ax.set_xlabel("E2s")
    # ax.set_ylabel("E1s")
    # ax.set_yscale("log")
    # ax.legend()


    # fig.savefig(f"{circ_name}_{timestep}_{tol}_errors_comp_rev.png")