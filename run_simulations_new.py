import cirq.circuits
import cirq.circuits.circuit
import cirq.sim
from bqskit.ir.circuit import Circuit
from sys import argv
import numpy as np

import matplotlib.pyplot as plt
import random

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import FakeCasablancaV2

from util import load_circuit, load_compiled_circuits

from bqskit.ext import bqskit_to_qiskit

from qiskit_aer.noise import NoiseModel, pauli_error

import multiprocessing as mp

import cirq



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
def get_ensemble_mags(ens_size):
    global all_qcircs
    global calc_func

    ensemble: list[QuantumCircuit] = random.sample(all_qcircs, ens_size)
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


def get_qcirc(circ: Circuit):
    q_circ = bqskit_to_qiskit(circ)
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


    circ_name = argv[1]
    timestep = int(argv[2])
    tol = int(argv[3])

    initial_circ = load_circuit(circ_name)
    target = initial_circ.get_unitary()
    circs = load_compiled_circuits(circ_name, tol, timestep)

    # Store approximate solutions
    all_utries = []
    basic_circs = []
    circ_files = []
    base_excitations = []
    noisy_excitations = []

    calc_func = excitation_displacement


    qiskit_circ = bqskit_to_qiskit(initial_circ)
    qiskit_circ.measure_all()
    result = sim_perf.run(qiskit_circ, shots=shots*100).result()
    result_dict = result.get_counts(qiskit_circ)
    noisy_result = sim.run(qiskit_circ, shots=shots*100).result()
    noisy_result_dict = noisy_result.get_counts(qiskit_circ)
    base_excitations.append(calc_func(initial_circ.num_qudits, result_dict, shots*100))
    noisy_excitations.append(calc_func(initial_circ.num_qudits, noisy_result_dict, shots*100))

    with mp.Pool() as pool:
       all_qcircs = pool.map(get_qcirc, circs)

    ensemble_mags = []
    ensemble_sizes = [1, 10, 100, 1000, 2000]


    print("Runing QCIRCS")
    print(f"Base Excitation: {base_excitations[0]}, Noisy Excitation {noisy_excitations[0]}")
    for j, ens_size in enumerate(ensemble_sizes):
        ensemble_mags[j] = get_ensemble_mags(ens_size)
        print(f"Ensemble Size: {ens_size},  Ensemble Excitation {ensemble_mags[j]}")


