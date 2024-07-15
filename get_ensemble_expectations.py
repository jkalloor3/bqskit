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

# def local_magnetization_orig(N, result: dict, shots: int, qub: int):
#     """Compute average magnetization from results of qk.execution.
#     Args:
#     - N: number of spins
#     - result (dict): a dictionary with the counts for each qubit, see qk.result.result module
#     - shots (int): number of trials
#     Return:
#     - average_mag (float)
#     """
#     mag = 0
#     q_idx = N - qub -1
#     pers = [0 for _ in range(2 ** N)]
#     for spin_str, count in result.items():
#         # print(count / shots, end=",")
#         ind = int(spin_str, base=2)
#         pers[ind] = count / shots
#         spin_int = [1 - 2 * float(spin_str[q_idx])]
#         mag += (sum(spin_int) / len(spin_int)) * count
#     print(pers)
#     average_mag = mag / shots
#     return average_mag

# def excitation_displacement_orig(N, result: dict, shots: int):
#     dis = 0
#     for qub in range(1, N):
#         z = local_magnetization_orig(N,result, shots, qub)
#         dis += qub*((1.0 - z)/2.0)
#         print(dis)
#         exit(0)
#     return dis

def get_ensemble_mags(i):
    global all_utries
    global targets
    ens_size = 1
    ensemble = random.sample(all_utries[i], ens_size)
    ensemble_mean = np.mean(ensemble, axis=0)
    dist = targets[i].get_frobenius_distance(ensemble_mean)
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
    global targets

    circ_type = argv[1]

    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)

    method = argv[2]
    max_tol = int(argv[3])
    min_tol = max_tol
    # Store approximate solutions
    all_utries = []
    basic_circs = []
    circ_files = []

    min_tol = max_tol

    timesteps = list(range(10,20))
    base_excitations = []

    num_spins = 5

    targets = []

    for timestep in timesteps:

        # Get Original Circuit
        if circ_type == "TFIM":
                initial_circ = Circuit.from_file(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/TFIM_3_timesteps/TFIM_3_{timestep}.qasm")
                initial_circ.remove_all_measurements()
                # target = np.loadtxt("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/tfim_4-1.unitary", dtype=np.complex128)
        elif circ_type == "Heisenberg":
            initial_circ = Circuit.from_file("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/heisenberg_3.qasm")
        elif circ_type == "Heisenberg_7" or circ_type == "TFXY_8":
            initial_circ = Circuit.from_file(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/{circ_type}/{circ_type}_{timestep}.qasm")
            initial_circ.remove_all_measurements()
        elif circ_type == "Hubbard":
            initial_circ = Circuit.from_file("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/hubbard_4.qasm")
            # target = np.loadtxt("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/tfim_4-1.unitary", dtype=np.complex128)
        elif circ_type == "TFXY":
            initial_circ = Circuit.from_file("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/tfxy_6.qasm")
        elif circ_type == "TFXY_t":
            initial_circ: Circuit =  Circuit.from_file(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/TFXY_5_timesteps/TFXY_5_{timestep}.qasm")
            # qiskit_circ = QuantumCircuit.from_qasm_file(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/TFXY_5_timesteps/TFXY_5_{timestep}.qasm")
            initial_circ.remove_all_measurements()
        else:
            target = np.loadtxt("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/qite_3.unitary", dtype=np.complex128)
            initial_circ = Circuit.from_unitary(target)

        print(initial_circ.num_cycles, end=",")
        target = initial_circ.get_unitary()
        targets.append(target)
        # result = simulator.run(qiskit_circ, shots=num_shots).result()
        # result_dict = result.get_counts(qiskit_circ)
        # base_excitations.append(excitation_displacement_orig(num_spins, result_dict, num_shots))
        base_excitations.append(excitation_displacement(target))
        # print("Calculated excitation")


    # Visualize the results
    fig, axs = plt.subplots(1,1)

    # axs.plot(timesteps, base_excitations)
    axs.set_title('Base Behavior')
    axs.set_xlabel('Timestep')
    axs.set_ylabel('Excitation Displacement')
    print("")

    # Now get ensemble results

    # ensemble_sizes = [3, 10, 100, 500, 1000]
    ensemble_sizes = [1, 10, 100]

    ensemble_mags = [[] for _ in ensemble_sizes]

    all_utries = [[] for _ in timesteps]

    max_dists = {}

    if method.startswith("jiggle"):
        for i, timestep in enumerate(timesteps):
            for j in range(max_tol, max_tol + 1):
                # dir = f"ensemble_approx_circuits_qfactor/{method}/{circ_type}/{j}/{timestep}/params_*_{max_tol}.pickle"
                dir = f"ensemble_approx_circuits_qfactor/{method}/{circ_type}/{j}/{timestep}/params_*.pickle"
                if method.startswith("jiggle"):
                    # basic_circ = pickle.load(open(f"ensemble_approx_circuits_qfactor/{method}/{circ_type}/jiggled_circs/{max_tol}/{timestep}/jiggled_circ.pickle", "rb"))
                    basic_circ: Circuit = pickle.load(open(f"ensemble_approx_circuits_qfactor/gpu_real/{circ_type}/jiggled_circs/7/6/{timestep}/jiggled_circ.pickle", "rb"))
                    print(dir)
                    print("Got Circ")
                
                circ_files = glob.glob(dir)[:200]
                circ_args = [(f"ensemble_approx_circuits_qfactor/gpu_real/{circ_type}/jiggled_circs/7/6/{timestep}/jiggled_circ.pickle", cf) for cf in circ_files]

                with mp.Pool() as pool:
                    if method.startswith("jiggle"):
                        utries = pool.map(get_circ_unitary_diff_jiggle, circ_args)
                    else:
                        utries = pool.map(get_circ_unitary_diff, circ_files)

                # avg_utry_dist = np.mean([targets[i].get_frobenius_distance(utry) for utry in utries])  
                # print("Avg Utry Dist", avg_utry_dist, end=",")   
                # exit(0)
                    # circ_files.extend(glob.glob(dir)[:1000])
                        
                all_utries[i].extend(utries)
    print("")
    print([len(x) for x in all_utries])

    for j, ens_size in enumerate(ensemble_sizes):
        max_dist = 0
        avg_dist = 0
        # print(ens_size)
        with mp.Pool() as pool:
            ensemble_mags[j] = pool.map(get_ensemble_mags, range(len(timesteps)))
        # max_dists[ens_size] = [max_dist, avg_dist]

    colors = ["c", "g", "y", "b", "r"]

    # print(base_excitations)
    # print(ensemble_mags)

    base_excitations = np.array(base_excitations)
    ensemble_mags = np.array(ensemble_mags)

    for i, ens_size in enumerate(ensemble_sizes):
        print(f"Avg Exp. Dist for Ens of size: {ens_size}", np.mean(np.abs(ensemble_mags[i] - base_excitations)))
        axs.plot(timesteps, ensemble_mags[i] - base_excitations, c=colors[i], label=f"Ensemble Size: {ens_size}")

    # plt.show()
    axs.legend() 
    fig.savefig(f"excitation_{circ_type}_{method}_{max_tol}_1_diff.png")

    # print(max_dists)

        
    # print("------------------")
    # print(len(all_utries))

