import cirq.circuits
import cirq.circuits.circuit
import cirq.sim
from bqskit.ir.circuit import Circuit
from sys import argv
import numpy as np
# Generate a super ensemble for some error bounds
import pickle

from bqskit.qis.unitary import UnitaryMatrix

import glob

from os.path import exists, join

import matplotlib.pyplot as plt
import random

import multiprocessing as mp
import time

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from concurrent.futures import ThreadPoolExecutor
from qiskit_experiments.framework import BatchExperiment
from qiskit.providers.fake_provider import FakeCasablancaV2

from bqskit.ext import bqskit_to_qiskit, bqskit_to_cirq

from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)
from qiskit_experiments.library import StandardRB

import cirq

class MyNoiseModel(cirq.NoiseModel):

    def __init__(self, one_qubit_error_rate, two_qubit_error_rate):
        self._one_qubit_error_rate = one_qubit_error_rate
        self._two_qubit_error_rate = two_qubit_error_rate

    def noisy_operation(self, op):        
        n_qubits = len(op.qubits)
        if n_qubits > 2:
            return op
        error_rate = self._one_qubit_error_rate if n_qubits == 1 else self._two_qubit_error_rate
        depolarize_channel = cirq.depolarize(error_rate, n_qubits=n_qubits)
        return [op, depolarize_channel.on(*op.qubits)]



def create_pauli_noise_model(rb_fid_1q, rb_fid_2q):
    rb_err_1q = 1 - rb_fid_1q
    rb_err_2q = 1 - rb_fid_2q

    # Create an empty noise model
    noise_model = NoiseModel()

    # Add depolarizing error to all single qubit u1, u2, u3 gates
    one_q_error = pauli_error([('X', rb_err_1q /3), ('Y', rb_err_1q /3), ('Z', rb_err_1q / 3), ('I', rb_fid_1q)])
    two_q_error = pauli_error([('XX', rb_err_2q / 15), ('YX', rb_err_2q / 15), ('ZX', rb_err_2q / 15),
                            ('XY', rb_err_2q / 15), ('YY', rb_err_2q / 15), ('ZY', rb_err_2q / 15),
                                ('XZ', rb_err_2q / 15), ('YZ', rb_err_2q / 15), ('ZZ', rb_err_2q / 15),
                                ('XI', rb_err_2q / 15), ('YI', rb_err_2q / 15), ('ZI', rb_err_2q / 15),
                                ('IX', rb_err_2q / 15), ('IY', rb_err_2q / 15), ('IZ', rb_err_2q / 15), 
                                ('II', rb_fid_2q)])
    noise_model.add_all_qubit_quantum_error(one_q_error, ['u3'])
    noise_model.add_all_qubit_quantum_error(two_q_error, ['cx'])

    return noise_model

def create_pauli_noise_model_backend(rb_fid_1q, rb_fid_2q):
    rb_err_1q = 1 - rb_fid_1q
    rb_err_2q = 1 - rb_fid_2q

    # Create an empty noise model
    noise_model = NoiseModel()

    # Add depolarizing error to all single qubit u1, u2, u3 gates
    one_q_error = pauli_error([('X', rb_err_1q /3), ('Y', rb_err_1q /3), ('Z', rb_err_1q / 3), ('I', rb_fid_1q)])
    two_q_error = pauli_error([('XX', rb_err_2q / 15), ('YX', rb_err_2q / 15), ('ZX', rb_err_2q / 15),
                            ('XY', rb_err_2q / 15), ('YY', rb_err_2q / 15), ('ZY', rb_err_2q / 15),
                                ('XZ', rb_err_2q / 15), ('YZ', rb_err_2q / 15), ('ZZ', rb_err_2q / 15),
                                ('XI', rb_err_2q / 15), ('YI', rb_err_2q / 15), ('ZI', rb_err_2q / 15),
                                ('IX', rb_err_2q / 15), ('IY', rb_err_2q / 15), ('IZ', rb_err_2q / 15), 
                                ('II', rb_fid_2q)])
    noise_model.add_all_qubit_quantum_error(one_q_error, ['u3'])
    noise_model.add_all_qubit_quantum_error(two_q_error, ['cx'])


# sim = AerSimulator()
device_backend = FakeCasablancaV2()
one_q_fid = 0.9999
two_q_fid = 0.995
noisy_backend = create_pauli_noise_model(one_q_fid, two_q_fid)
noise_model_backend = create_pauli_noise_model(one_q_fid, two_q_fid)
sim = AerSimulator(noise_model=noisy_backend)
# sim = AerSimulator()
sim_perf = AerSimulator()
shots = 1024

def get_oneq_rb(noisy_backend):
    lengths = np.arange(1, 800, 200)
    num_samples = 20
    seed = 1010

    num_qubits = 7

    qubits = range(num_qubits)

    errors = {}

    for qubit in qubits:
        # Run an RB experiment on all_qubits
        exp1 = StandardRB([qubit], lengths, num_samples=num_samples, seed=seed)
        expdata1 = exp1.run(noisy_backend).block_for_results()
        results1 = expdata1.analysis_results()

        one_q_error = results1[3].value.nominal_value
        errors[(qubit,)] = (one_q_error)

    print(errors)
    return errors


# Get 2-qubit noise also
def get_twoq_rb(noisy_backend):
    lengths_2_qubit = np.arange(1, 200, 30)
    lengths_1_qubit = np.arange(1, 800, 200)
    num_samples = 10
    seed = 1010

    edges = noisy_backend.coupling_map

    errors = {}

    for qubits in edges:
        # Run a 1-qubit RB experiment on qubits 1, 2 to determine the error-per-gate of 1-qubit gates
        single_exps = BatchExperiment(
            [
                StandardRB([qubit], lengths_1_qubit, num_samples=num_samples, seed=seed)
                for qubit in qubits
            ],
            flatten_results=True,
        )
        expdata_1q = single_exps.run(noisy_backend).block_for_results()

        # Run an RB experiment on qubits 1, 2
        exp_2q = StandardRB(qubits, lengths_2_qubit, num_samples=num_samples, seed=seed)

        # Use the EPG data of the 1-qubit runs to ensure correct 2-qubit EPG computation
        exp_2q.analysis.set_options(epg_1_qubit=expdata_1q.analysis_results())

        # Run the 2-qubit experiment
        expdata_2q = exp_2q.run(noisy_backend).block_for_results()


        results2 = expdata_2q.analysis_results()
        two_q_error = results2[4].value.nominal_value

        errors[qubits] = two_q_error

    return errors



def staggered_magnetization(N, result: dict, shots: int):
    sm_val = 0
    for spin_str, count in result.items():
        spin_int = [1 - 2 * float(s) for s in spin_str]
        for i in range(len(spin_int)):
            spin_int[i] = spin_int[i]*(-1)**i
        sm_val += (sum(spin_int) / len(spin_int)) * count
    average_sm = sm_val/shots
    return average_sm

def local_magnetization(N, result: dict, shots: int, qub: int):
    """Compute average magnetization from results of qk.execution.
    Args:
    - N: number of spins
    - result (dict): a dictionary with the counts for each qubit, see qk.result.result module
    - shots (int): number of trials
    Return:
    - average_mag (float)
    """
    mag = 0
    q_idx = N - qub -1
    pers = [0 for _ in range(2 ** N)]
    for spin_str, count in result.items():
        # print(count / shots, end=",")
        ind = int(spin_str, base=2)
        pers[ind] = count / shots
        spin_int = [1 - 2 * float(spin_str[q_idx])]
        mag += (sum(spin_int) / len(spin_int)) * count
    average_mag = mag / shots
    return average_mag

def excitation_displacement(N, result: dict, shots: int):
    dis = 0
    for qub in range(1, N):
        z = local_magnetization(N,result, shots, qub)
        dis += qub*((1.0 - z)/2.0)
    return dis

import itertools
def get_ensemble_mags(data):
    global all_qcircs
    global calc_func
    i, ens_size = data


    ensemble: list[QuantumCircuit] = random.sample(all_qcircs[i], ens_size)
    results = sim.run(ensemble, num_shots=shots).result()
    # Sum over all c
    total_dict = {}
    for c in ensemble:
        results_dict = results.get_counts(c)
        x = total_dict
        y = results_dict
        total_dict = {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}
        
    return calc_func(c.num_qubits, total_dict, shots=shots * len(ensemble))


def execute_circuit(circuit: QuantumCircuit):
    global calc_func
    results = sim.run(circuit, num_shots=shots).result()
    noisy_counts = results.get_counts(circuit)
    return calc_func(circuit.num_qubits, noisy_counts, shots=shots)

cirq_noise = MyNoiseModel(1 - one_q_fid, 1-two_q_fid)
cirq_sim = cirq.sim.Simulator()
def execute_circuit_cirq(circuit: cirq.circuits.circuit):
    global calc_func
    samples = cirq_sim.sample(circuit, repetitions=shots)
    total_dict = {}
    for sample in samples[:, 1]:
        total_dict[sample] = total_dict.get(sample, 0) + 1

    return calc_func(7, total_dict, shots=shots)

from mitiq.zne import execute_with_zne
from mitiq.interface import convert_to_mitiq
def get_ensemble_mags_zne(data):
    global all_qcircs
    global calc_func
    i, ens_size = data


    ensemble: list[QuantumCircuit] = random.sample(all_qcircs[i], ens_size)
    # Sum over all c
    exp_vals = []
    for c in ensemble:
        new_c = transpile(c, basis_gates=["unitary", "cx"])
        # print(c.)
        exp_val = execute_with_zne(new_c, execute_circuit)
        exp_vals.append(exp_val)

    return np.mean(exp_vals)


def get_qcirc(params):
    global basic_circ
    # start = time.time()
    basic_circ.set_params(params)
    q_circ = bqskit_to_qiskit(basic_circ)
    q_circ.measure_all()
    # (time.time() - start)
    return q_circ

def get_covar_elem(matrices):
    A, B = matrices
    elem =  2*np.real(np.trace(A.conj().T @ B))
    return elem

# Circ 
if __name__ == '__main__':

    # print(get_oneq_rb(device_backend))
    # print(get_twoq_rb(device_backend))
    global basic_circ
    global all_qcircs
    global calc_func

    circ_type = argv[1]

    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)

    method = argv[2]
    tols = [1, 3, 4]
    # Store approximate solutions
    all_utries = []
    basic_circs = []
    circ_files = []

    timesteps = [2,4,5,7,8]
    base_excitations = []
    noisy_excitations = []

    if circ_type.startswith("TF"):
        calc_func = excitation_displacement
    else:
        calc_func = staggered_magnetization


    for timestep in timesteps:

        # Get Original Circuit
        if circ_type.startswith("vqe"):
            initial_circ = Circuit.from_file(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/{circ_type}.qasm")
        else:
            initial_circ = Circuit.from_file(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/{circ_type}/{circ_type}_{timestep}.qasm")
            initial_circ.remove_all_measurements()
        if initial_circ.num_qudits < 8:
            target = initial_circ.get_unitary()

        qiskit_circ = bqskit_to_qiskit(initial_circ)
        qiskit_circ.measure_all()
        result = sim_perf.run(qiskit_circ, shots=shots*100).result()
        result_dict = result.get_counts(qiskit_circ)
        base_excitations.append(calc_func(initial_circ.num_qudits, result_dict, shots*100))
        noisy_result = sim.run(qiskit_circ, shots=shots*100).result()
        result_dict = noisy_result.get_counts(qiskit_circ)
        noisy_excitations.append(calc_func(initial_circ.num_qudits, result_dict, shots*100))
        # print("Calculated excitation")


    # Now get ensemble results
    # Visualize the results
    fig = plt.figure(figsize=(25, 5))
    axes = fig.subplots(1,6)

    for axes_ind in range(3):
        print("Getting Circuits")

        # ensemble_sizes = [1, 10, 100, 500, 1000]
        ensemble_sizes = [1, 10, 100]

        ensemble_mags = [[] for _ in ensemble_sizes]

        all_qcircs: list[QuantumCircuit] = [[] for _ in timesteps]

        max_dists = {}
        max_tol = tols[axes_ind]
        if method.startswith("jiggle"):
            for i, timestep in enumerate(timesteps):
                for j in range(max_tol, max_tol + 1):
                    q_circs_file = f"ensemble_approx_circuits_qfactor/{method}/{circ_type}/{j}/{timestep}/all_qcircs.pickle"
                    try:
                        qcircs = pickle.load(open(q_circs_file, "rb"))
                        print("Loaded Circuits")
                    except:
                        print("Getting all params")
                        param_file = f"ensemble_approx_circuits_qfactor/{method}/{circ_type}/{j}/{timestep}/all_params.pickle"
                        if method.startswith("jiggle"):
                            # basic_circ = pickle.load(open(f"ensemble_approx_circuits_qfactor/{method}/{circ_type}/jiggled_circs/{max_tol}/{timestep}/jiggled_circ.pickle", "rb"))
                            basic_circ: Circuit = pickle.load(open(f"ensemble_approx_circuits_qfactor/gpu_post_pam/{circ_type}/jiggled_circs/8/6/{timestep}/jiggled_circ.pickle", "rb"))
                            q_circ = bqskit_to_qiskit(basic_circ)
                            # q_circ = bqskit_to_cirq(basic_circ)
                            print("Got Circ")
                        
                        if exists(param_file):
                            params = pickle.load(open(param_file, "rb"))
                        else:
                            param_files = glob.glob(f"ensemble_approx_circuits_qfactor/{method}/{circ_type}/{j}/{timestep}/params_*.pickle")
                            params = [pickle.load(open(pf, "rb")) for pf in param_files]
                            pickle.dump(params, open(param_file, "wb"))
                        
                        start = time.time()
                        with mp.Pool() as pool:
                            qcircs = pool.map(get_qcirc, params)

                        # times = [get_qcirc(p) for p in params]
                        full_time = time.time() - start
                        pickle.dump(qcircs, open(q_circs_file, "wb"))

                        print(full_time)

                    all_qcircs[i].extend(qcircs)


        # print([len(x) for x in all_qcircs])
        # run_params = list(itertools.product(range(len(timesteps)), ensemble_sizes))
        # print(run_params)

        # pickle.dump(all_qcircs, open(f"ensemble_approx_circuits_qfactor/{method}/{circ_type}/{j}/{timestep}/all_qcircs.pickle", "wb"))

        start = time.time()
        # with mp.Pool(processes=min(len(run_params), 256)) as pool:
        #     all_results = pool.map(get_ensemble_mags, run_params)
        # n = len(timesteps)
        # ensemble_mags = [all_results[i: i + n] for i in range(0, len(all_results), n)]

        print("Runing QCIRCS")
        for j, ens_size in enumerate(ensemble_sizes):
            for i in range(len(timesteps)):
                ensemble_mags[j].append(get_ensemble_mags_zne((i, ens_size)))
        
        # print(ensemble_mags)
        
        print(f"RUNNING AND CALCULATING TIME for all circuits: {time.time() - start}")

        colors = ["c", "g", "y", "r", "m"]

        # print(base_excitations)
        # print(ensemble_mags)

        base_excitations = np.array(base_excitations)
        ensemble_mags = np.array(ensemble_mags)


        axs = axes[axes_ind*2]
        axs_diff = axes[axes_ind*2 + 1]

        # axs.plot(timesteps, base_excitations)
        axs.set_title('Base Behavior')
        axs.set_xlabel('Timestep')
        axs.set_ylabel('Excitation Displacement')


        for i, ens_size in enumerate(ensemble_sizes):
            # print(f"Avg Exp. Dist for Ens of size: {ens_size}", np.mean(np.abs(ensemble_mags[i] - base_excitations)))
            axs.plot(timesteps, ensemble_mags[i], c=colors[i], label=f"Ensemble Size: {ens_size}")

        axs.plot(timesteps, noisy_excitations, c="blue", label=f"Noisy Base Sim")
        axs.plot(timesteps, base_excitations, c="black", label=f"Base Sim")



        for i, ens_size in enumerate(ensemble_sizes):
            # print(f"Avg Exp. Dist for Ens of size: {ens_size}", np.mean(np.abs(ensemble_mags[i] - base_excitations)))
            axs_diff.plot(timesteps, ensemble_mags[i] - base_excitations, c=colors[i], label=f"Ensemble Size: {ens_size}")

        axs_diff.plot(timesteps, noisy_excitations - base_excitations, c="blue", label=f"Noisy Base Sim")


        # plt.show()
        axs.legend() 
    fig.savefig(f"pam_staggered_{circ_type}_{method}_1_3_4_2_6_noisy_sim_{one_q_fid}_{two_q_fid}_3.png")
    exit(0)

        


