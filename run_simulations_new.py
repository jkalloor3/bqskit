from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from sys import argv
from scipy.stats import entropy
import numpy as np

import matplotlib.pyplot as plt
import random
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.point import CircuitPoint

from itertools import chain
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, DensityMatrix
# from qiskit_aer.primitives import PrimitiveResult
from qiskit_aer.noise import NoiseModel, pauli_error
# from qiskit.providers.fake_provider import FakeCasablancaV2

from util import load_circuit, load_compiled_circuits

from bqskit.ext import bqskit_to_qiskit
# from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# from qiskit_aer.noise import NoiseModel, pauli_error

import multiprocessing as mp

import cirq

def run_circuit(circuit: QuantumCircuit, shots: int = 1024):
    global calc_func
    results = sim.run(circuit, num_shots=shots).result()
    noisy_counts = results.get_counts(circuit)
    return noisy_counts


def prob_squared(N, result: dict, shots: int):
    # Calculate sum(p(x)^2)
    probs = [v / shots for v in result.values()]
    return sum(p*p for p in probs)


shots = 100

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

def tvd(p, q, shots):
    p = {k: v/shots for k, v in p.items()}
    q = {k: v/shots for k, v in q.items()}
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))

def get_ensemble_mags(ens_size, random_states: list[np.ndarray] = None):
    global all_qcircs
    global calc_func
    global circs

    ensemble_inds: list[int] = np.random.choice(len(all_qcircs), ens_size)
    ensemble: list[QuantumCircuit] = [all_qcircs[i] for i in ensemble_inds]
    
    if ensemble[0].num_qubits <= 10:
        if ens_size < 10:
            print([normalized_frob_dist(circs[i].get_unitary()) for i in ensemble_inds])
        mean_un = np.mean(np.array([circs[i].get_unitary().numpy for i in ensemble_inds]), axis=0)
    else:
        mean_un = None

    print("Avg CNOT count: ", np.mean([c.count_ops()['cx'] for c in ensemble]))
    all_circs = ensemble
    noisy_svs = np.array([Statevector.from_instruction(circ).data for circ in all_circs])
    final_sv = np.mean(noisy_svs, axis=0)
    return final_sv, mean_un

def execute_circuit(circuit: QuantumCircuit):
    global calc_func
    results = sim.run(circuit, shots=shots).result()
    noisy_counts = results.get_counts(circuit)
    return calc_func(circuit.num_qubits, noisy_counts, shots=shots)


def get_qcirc(circ: Circuit):
    for cycle, op in circ.operations_with_cycles():
        if op.num_qudits == 1 and not isinstance(op.gate, U3Gate):
            params = U3Gate().calc_params(op.get_unitary())
            point = CircuitPoint(cycle, op.location[0])
            circ.replace_gate(point, U3Gate(), op.location, params)

    q_circ = bqskit_to_qiskit(circ)
    return q_circ

def get_covar_elem(matrices):
    A, B = matrices
    elem =  2*np.real(np.trace(A.conj().T @ B))
    return elem

def normalized_frob_dist(mat: np.ndarray):
    global target
    frob_dist = target.get_frobenius_distance(mat) / np.sqrt(mat.shape[0] * 2)
    return frob_dist

def get_random_states(num_qubits: int, num_random_states: int = 4):
    states = []
    for i in range(4):
        state = np.random.randint(0, 2, num_qubits)
        states.append(state)
    return states

def get_random_init_state_circuits(qcirc: QuantumCircuit, random_states: list[np.ndarray]):
    circs = []
    for state in random_states:
        init_circ = qcirc.copy()
        for i in range(init_circ.num_qubits):
            if state[i] == 1:
                init_circ.h(i)
        init_circ.compose(qcirc, inplace=True)
        circs.append(transpile(init_circ, backend=sim, optimization_level=0))
    return circs

def aggregate_results(results: list):
    total_dict = {}
    for r in results:
        # print("Result: ", r)
        x = total_dict
        y = r.data.meas.get_counts()
        total_dict = {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}
    return total_dict

def trace_distance(sv1: np.ndarray, sv2: np.ndarray):
    sv1 = sv1 / np.linalg.norm(sv1)
    sv2 = sv2 / np.linalg.norm(sv2)
    # diff_2 = np.outer(sv1, sv1.conj()) - np.outer(sv2, sv2.conj())
    # eigvals = np.linalg.eigh(diff_2)[0]
    # trace = np.sum(np.abs(eigvals)) / 2
    trace_3 = 1 - np.abs(np.inner(sv1, sv2.conj())) ** 2
    trace_3 = np.sqrt(trace_3)
    return trace_3

# Circ 
if __name__ == '__main__':

    # print(get_oneq_rb(device_backend))
    # print(get_twoq_rb(device_backend))
    global basic_circ
    global all_qcircs
    global calc_func
    global circs
    global target

    circ_type = argv[1]

    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)


    circ_name = argv[1]
    timestep = int(argv[2])
    tol = int(argv[3])
    cliff = bool(int(argv[4])) if len(argv) > 4 else False

    initial_circ = load_circuit(circ_name, opt=False)
    print("Original CX Count: ", initial_circ.count(CNOTGate()))
    if initial_circ.num_qudits <= 10:
        target = initial_circ.get_unitary()
    num_unique_circs = int(argv[4])

    if cliff:
        circs = load_compiled_circuits(circ_name, tol, timestep, ignore_timestep=True, extra_str=f"_{num_unique_circs}_circ_cliff_t_final_noqp")
    else:
        circs = load_compiled_circuits(circ_name, tol, timestep, ignore_timestep=True, extra_str=f"_{num_unique_circs}_circ_final_min_post_noqp")

    print("LOADED CIRCUITS", flush=True)

    # Store approximate solutions
    all_utries = []
    basic_circs = []
    circ_files = []
    base_excitations = []
    noisy_excitations = []

    calc_func = prob_squared

    ensemble_sizes = [1, 10, 100, 1000] #, 2000, 4000]
    shot_ratio = max(ensemble_sizes)

    # sampler = Sampler(mode=sim)

    random_states = get_random_states(initial_circ.num_qudits, num_random_states=5)
    qiskit_circ = bqskit_to_qiskit(initial_circ)
    qiskit_circs = [qiskit_circ]
    print("Got all circuits", flush=True)
    svs = [Statevector.from_instruction(circ) for circ in qiskit_circs]
    noisy_svs = [Statevector.from_instruction(circ) for circ in qiskit_circs]
    noisy_dists = [trace_distance(svs[i].data, sv.data) for i,sv in enumerate(noisy_svs)]
    print("Noisy Distances: ", noisy_dists)
    
    
    # print("Noisy Counts: ", noisy_result_dict)
    print("Finished Running", flush=True)
    # noisy_result_dict = noisy_result.get_counts(qiskit_circ)
    base_excitations.append(0)
    noisy_excitations.append(np.mean(noisy_dists))

    print("Base TVD: ", base_excitations[0], "Noisy TVD: ", noisy_excitations[0])

    print("Finished Base Circuits", flush=True)


    all_qcircs = [get_qcirc(c) for c in circs]
    print("Got MAP", flush=True)

    ensemble_mags = [0,0,0,0,0,0]


    print("Runing PERFECT ENSEMBLES: ")
    print(f"Base TVD: {base_excitations[0]}, Noisy TVD {noisy_excitations[0]}")
    for j, ens_size in enumerate(ensemble_sizes):
        final_sv, mean_un = get_ensemble_mags(ens_size)
        ensemble_mags[j] = trace_distance(final_sv, svs[0].data)
        if mean_un is not None:
            frob_dist = normalized_frob_dist(mean_un)
            print(f"Ensemble Size: {ens_size},  Trace Distance: {ensemble_mags[j]}, Frobenius Distance Normalized: {frob_dist} Full Frobenius Distance: {target.get_frobenius_distance(mean_un)}")
        else:
            print(f"Ensemble Size: {ens_size},  Trace Distance: {ensemble_mags[j]}")


