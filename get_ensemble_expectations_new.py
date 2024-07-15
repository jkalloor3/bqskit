from bqskit.ir.circuit import Circuit
from sys import argv
import numpy as np
# Generate a super ensemble for some error bounds
import pickle

from bqskit.qis.unitary import UnitaryMatrix

import glob

import matplotlib.pyplot as plt
import random

import multiprocessing as mp
import time

from util import get_upperbound_error_mean, get_tvd_magnetization, load_circuit, get_ensemble_mean, load_compiled_circuits, get_unitary


def local_magnetization(u: UnitaryMatrix, N,  qub: int):
    """Compute average magnetization from results of qk.execution.
    Args:
    - N: number of spins
    - result (dict): a dictionary with the counts for each qubit, see qk.result.result module
    - shots (int): number of trials
    Return:
    - average_mag (float)
    """
    # Look at first column, assuming initial state
    values = (u.numpy[:, 0]).flatten()
    values = np.abs(values) ** 2
    # print(values)
    mag = 0
    for i, val in enumerate(values):
        q_val = float(np.binary_repr(i, N)[qub])
        spin_int = [1 - 2 * q_val]
        mag += (sum(spin_int) / len(spin_int)) * val
    return mag


# TODO: Fix these for unitary as well
def staggered_magnetization(result: dict, shots: int):
    sm_val = 0
    for spin_str, count in result.items():
        spin_int = [1 - 2 * float(s) for s in spin_str]
        for i in range(len(spin_int)):
            spin_int[i] = spin_int[i]*(-1)**i
        sm_val += (sum(spin_int) / len(spin_int)) * count
    average_sm = sm_val/shots
    return average_sm

def system_magnetization(result: dict, shots: int):
    mag_val = 0
    for spin_str, count in result.items():
        spin_int = [1 - 2 * float(s) for s in spin_str]
        mag_val += (sum(spin_int) / len(spin_int)) * count
    average_mag = mag_val/shots
    return average_mag

def excitation_displacement(u: UnitaryMatrix):
    dis = 0
    N = u.num_qudits
    for qub in range(1, N):
        z = local_magnetization(u, N, qub)
        dis += qub*((1.0 - z)/2.0)
    return dis

def get_ensemble_mags(i):
    global all_utries
    global target
    ens_size = 1
    ensemble = random.sample(all_utries[i], ens_size)
    ensemble_mean = np.mean(ensemble, axis=0)
    dist = target.get_frobenius_distance(ensemble_mean)
    print(dist)
    # print(dist)
    # if dist > max_dist:
    #     max_dist = dist
    return excitation_displacement(UnitaryMatrix(ensemble_mean, check_arguments=False))


def transform_mat_to_vec(unitary: UnitaryMatrix):
    vec = unitary.flatten()
    re_vec = np.abs(vec)
    im_vec = np.angle(vec)

    return np.hstack([re_vec, im_vec])


def get_circ_unitary_diff_jiggle(circ_args):
    start = time.time()
    basic_circ_file, circ_file = circ_args
    basic_circ: Circuit = pickle.load(open(basic_circ_file, "rb"))
    params: Circuit = pickle.load(open(circ_file, "rb"))
    final_utry =  basic_circ.get_unitary(params)
    print("Took", time.time() - start)
    return final_utry

def get_circ_unitary_diff(circ_file):
    circ: Circuit = pickle.load(open(circ_file, "rb"))
    return circ.get_unitary()

def get_covar_elem(matrices):
    A, B = matrices
    elem =  2*np.real(np.trace(A.conj().T @ B))
    return elem

from qiskit import Aer

# Circ 
if __name__ == '__main__':
    global basic_circ
    global target
    global all_utries

    circ_name = argv[1]
    timestep = int(argv[2])
    tol = int(argv[3])

    circ = load_circuit(circ_name)
    target = circ.get_unitary()
    circs = load_compiled_circuits(circ_name, tol, timestep)

    dists = [target.get_frobenius_distance(c.get_unitary()) for c in circs[:20]]

    print(np.mean(dists))

    base_excitations = []
    base_excitations.append(excitation_displacement(target))

    timesteps = [0]

    ensemble_sizes = [1, 10, 100, 500, 1000, 1500]

    ensemble_mags = [[] for _ in ensemble_sizes]

    all_utries = [[c.get_unitary() for c in circs]]

    max_dists = {}

    for j, ens_size in enumerate(ensemble_sizes):
        max_dist = 0
        avg_dist = 0
        # print(ens_size)
        with mp.Pool() as pool:
            ensemble_mags[j] = pool.map(get_ensemble_mags, range(len(timesteps)))
        # max_dists[ens_size] = [max_dist, avg_dist]


    base_excitations = np.array(base_excitations)
    ensemble_mags = np.array(ensemble_mags)

    print("Base Excitation", base_excitations)
    for i, ens_size in enumerate(ensemble_sizes):
        print(f"Ensemble Excitations {ens_size}, Ensemble Magnitude: {ensemble_mags[i]}, Diff: {ensemble_mags[i] - base_excitations[0]}")


