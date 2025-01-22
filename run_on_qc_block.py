from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from sys import argv
from scipy.stats import entropy
import numpy as np
from pathlib import Path
import json

import matplotlib.pyplot as plt
import random
from bqskit.ir.gates.parameterized import U3Gate, VariableUnitaryGate
from bqskit.ir.point import CircuitPoint
from bqskit.compiler.passdata import PassData

from itertools import chain
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, DensityMatrix, hellinger_fidelity
from qiskit_aer.noise import NoiseModel, pauli_error, depolarizing_error
import multiprocessing as mp

from bqskit.ext import bqskit_to_qiskit, qiskit_to_bqskit

from util import load_block, load_compiled_block_circuits, frobenius_cost, tvd_dict, cross_entropy_fidelity_dict

shots = 1000
one_q_err = 1e-4
two_q_err = 4e-3

def run_circuits(all_circs: list[list[QuantumCircuit]], shots: int, backend: AerSimulator = None) -> list[dict[str, int]]:
    '''
    Run a set of noisy circuits on a given backend
    '''
    if backend is None:
        sampler = AerSimulator()
        print("Running on Aer Simulator: ", sampler.description, flush=True)
    else:
        # sampler = BackendSampler(backend=backend)
        print("Running on Noisy Sim with Noise Model: ", backend.description, flush=True)
        sampler = backend

    # Return aggregated results for each random circuit
    all_results = []
    for circs in all_circs:
        result = sampler.run(circs, shots=shots).result()
        # print(f">>> Job ID: {job.job_id()}")
        # print(f">>> Job Status: {job.status()}")
        if len(all_results) == 0:
            all_results = [result.get_counts(j)for j in range(len(circs))]
        else:
            results = [result.get_counts(j) for j in range(len(circs))]
            all_results = [aggregate_results([all_results[i], results[i]]) for i in range(len(circs))]
    return all_results

def run_noisy_ensemble(ens_size, shots: int = 1, random_states: list[np.ndarray] = None, backend: AerSimulator = None) -> tuple[list[dict[str, int]], np.ndarray]:
    '''
    Run an ensemble of size `ens_size` on a noisy backend.

    Returns a list of list of TVDs with shape (ens_size, num_random_states)

    Also, if the number of qubits is less than 10, returns the mean unitary of the ensemble.
    '''
    global all_qcircs
    # global uns

    ensemble_inds: list[int] = np.random.choice(len(all_qcircs), ens_size)
    ensemble: list[QuantumCircuit] = [all_qcircs[i] for i in ensemble_inds]
    # ensemble_uns = [uns[i] for i in ensemble_inds]
    # mean_un = np.mean(np.array(ensemble_uns), axis=0)
    print("Avg CNOT count: ", np.mean([c.count_ops()['cx'] for c in ensemble]))
    all_circs = [[c] for c in ensemble]
    # all_circs = [get_random_init_state_circuits(c, random_states, backend=backend) for c in ensemble]
    final_svs = run_circuits(all_circs, shots=shots, backend=backend)
    return final_svs #, mean_un

def get_qcirc(circ: Circuit):
    for cycle, op in circ.operations_with_cycles():
        if op.num_qudits == 1 and not isinstance(op.gate, U3Gate):
            params = U3Gate().calc_params(op.get_unitary())
            point = CircuitPoint(cycle, op.location[0])
            circ.replace_gate(point, U3Gate(), op.location, params)
        if op.num_qudits == 2 and isinstance(op.gate, VariableUnitaryGate):
            # get decomp 
            mini_qcirc = QuantumCircuit(2)
            mini_qcirc.append(UnitaryGate(op.get_unitary()), [0, 1])
            trans_qcirc = transpile(mini_qcirc, basis_gates=['cx', 'u3'])
            # print(trans_qcirc.count_ops())
            bqskit_circ = qiskit_to_bqskit(trans_qcirc)
            circ.replace_with_circuit((cycle, op.location[0]), bqskit_circ, as_circuit_gate=True)
    circ.unfold_all()
    # print(circ.gate_counts)
    q_circ = bqskit_to_qiskit(circ)
    q_circ.measure_all()
    # (time.time() - start)
    return q_circ

def get_random_states(num_qubits: int, num_random_states: int = 4) -> list[np.ndarray]:
    states = []
    for i in range(num_random_states):
        state = np.random.randint(0, 5, num_qubits)
        states.append(state)
    return states

def get_random_init_state_circuits(qcirc: QuantumCircuit, random_states: list[np.ndarray], backend: AerSimulator) -> list[QuantumCircuit]:
    circs = []
    for state in random_states:
        init_circ: QuantumCircuit = QuantumCircuit(qcirc.num_qubits)
        for i in range(init_circ.num_qubits):
            if state[i] == 1:
                init_circ.h(i)
            elif state[i] == 2:
                init_circ.x(i)
            elif state[i] == 3:
                init_circ.z(i)
            elif state[i] == 4:
                angles = np.random.uniform(0, 2 * np.pi, 3)
                init_circ.u(*angles, i)
        init_circ.compose(qcirc, inplace=True)
        circs.append(init_circ)
    return circs

def aggregate_results(results: list[dict[str, int]]) -> dict[str, int]:
    total_dict = {}
    for y in results:
        x = total_dict
        total_dict = {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}
    return total_dict

def create_pauli_noise_model(rb_fid_1q, rb_fid_2q):
    rb_err_1q = 1 - rb_fid_1q
    rb_err_2q = 1 - rb_fid_2q

    print("1Q Error: ", rb_err_1q, "2Q Error: ", rb_err_2q)

    # Create an empty noise model
    noise_model = NoiseModel()

    # Add depolarizing error to all single qubit u1, u2, u3 gates
    # one_q_error = pauli_error([('X', rb_err_1q /3), ('Y', rb_err_1q /3), ('Z', rb_err_1q / 3), ('I', rb_fid_1q)])
    one_q_error = depolarizing_error(rb_err_1q, 1)
    two_q_error = depolarizing_error(rb_err_2q, 2)
    # two_q_error = one_q_error.tensor(one_q_error)
    noise_model.add_all_qubit_quantum_error(one_q_error, ['u', 'u3'])
    noise_model.add_all_qubit_quantum_error(two_q_error, ['cx'])

    return noise_model

def get_sim_backend(num_1q_gates: int, num_2q_gates: int):
    x_fid = 1 - one_q_err
    y_fid = 1 - two_q_err
    pred_fid = x_fid ** num_1q_gates * y_fid ** num_2q_gates
    print("Final pred fid: ", pred_fid, "X: ", x_fid, "Y: ", y_fid, flush=True)

    return AerSimulator(noise_model=create_pauli_noise_model(x_fid, y_fid)), pred_fid

if __name__ == '__main__':
    global all_qcircs
    # global uns
    global target

    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)

    circ_name = argv[1]
    block_num = argv[2]
    tol = float(argv[3])
    num_unique_circs = int(argv[4])
    # target_err = int(argv[5])
    # target_fid = 1 - (10 ** (-1 * target_err))
    cliff = False

    circ_path = load_block(circ_name, block_num, good=True)
    initial_circ = Circuit.from_file(circ_path)
    target = initial_circ.get_unitary()

    # _ , backend = setup_ibm(initial_circ.num_qudits)
    num_1q_gates = len([op for op in initial_circ.operations() if op.num_qudits == 1])
    num_2q_gates = len([op for op in initial_circ.operations() if op.num_qudits == 2])
    noisy_backend, pred_fid = get_sim_backend(num_1q_gates, num_2q_gates)

    # exit(0)

    circs = load_compiled_block_circuits(circ_name, block_num, tol, num_unique_circs)
    print("Num Circs: ", len(circs), flush=True)

    # dists = [target.get_frobenius_distance(c.get_unitary()) for c in circs[:20]]
    bqskit_circs = circs
    # uns = [c.get_unitary() for c in bqskit_circs]
    # frob_dists = [frobenius_cost(target, un) for un in uns]
    # print("Avg Norm. Dist: ", np.mean(dists))
    # print("Avg Dist: ", np.mean(frob_dists))
    print("Original CX Count: ", initial_circ.count(CNOTGate()))

    print("LOADED CIRCUITS", flush=True)

    all_qcircs = [get_qcirc(c) for c in bqskit_circs]
    print("Got MAP", flush=True)

    print("LOADED CIRCUITS", flush=True)
    # print("NUM Circs: ", len(circs), flush=True)

    # Store approximate solutions
    all_utries = []
    basic_circs = []
    circ_files = []
    base_excitations = []
    noisy_excitations = []

    ensemble_sizes = [1, 10, 50, 500] #, 2000, 4000]
    shot_ratio = max(ensemble_sizes)

    num_random_states = 1
    random_states = get_random_states(initial_circ.num_qudits, num_random_states=num_random_states)
    qiskit_circ = bqskit_to_qiskit(initial_circ)
    qiskit_circ.measure_all()
    qiskit_circs = [qiskit_circ]
    # qiskit_circs = get_random_init_state_circuits(qiskit_circ, random_states, backend=noisy_backend)
    print("Num Random Circuits: ", len(qiskit_circs))

    # Get the base results
    base_svs = run_circuits([qiskit_circs], shots=shots*shot_ratio)
    noisy_svs = run_circuits([qiskit_circs], shots=shots*shot_ratio, backend=noisy_backend)
    
    # print(base_svs)
    # print(noisy_svs)

    orig_noisy_tvds = [tvd_dict(base_svs[i], noisy_svs[i]) for i in range(num_random_states)]
    orig_hellinger_fids = [hellinger_fidelity(base_svs[i], noisy_svs[i]) for i in range(num_random_states)]

    print("Noisy TVDs: ", orig_noisy_tvds)
    # print("Noisy TVD: ", np.mean(orig_noisy_tvds))
    # print("Noisy Hellinger Fids ", np.mean(orig_hellinger_fids))

    '''
    Calculate values for ensemble sizes
    '''
    # ensemble_mags = [0,0,0,0,0,0]

    print("Runing Noisy ENSEMBLES: ")
    # print(f"Base TVD: {base_excitations[0]}, Noisy TVD {noisy_excitations[0]}")
    final_tvds = []
    final_frobs = []
    final_fids = []

    # Get outputs using mp Pool
    # with mp.Pool(mp.cpu_count()) as pool:
    results = []
    for ens_size in ensemble_sizes:
        ens_result = run_noisy_ensemble(ens_size, shots * (shot_ratio // ens_size), random_states, noisy_backend)
        print("Ensemble Size: ", ens_size)
        results.append(ens_result)


    for j, data in enumerate(results):
        ens_size = ensemble_sizes[j]
        shots_per_circuit = shots * (shot_ratio // ens_size)
        final_svs = data
        avg_noisy_tvds = np.array([tvd_dict(base_svs[i], final_svs[i]) 
                                    for i in range(num_random_states)])
        avg_hell_fids = np.array([hellinger_fidelity(base_svs[i], final_svs[i]) 
                                    for i in range(num_random_states)])
        # frob_dist = frobenius_cost(mean_un, target)
        frob_dist = 0
        print(f"Ensemble Size: {ens_size}, Frobenius Distance: {frob_dist}")

        print("Avg Noisy TVD: ", avg_noisy_tvds)
        final_tvds.append(np.mean(avg_noisy_tvds))
        # final_frobs.append(np.mean(frob_dist))
        final_fids.append(np.mean(avg_hell_fids))

    
    headers = ["Ensemble Size", "TVD", "Frobenius Distance"]
    out_data = {}
    out_data["Ensemble Size"] = ensemble_sizes
    out_data["Original Circuit TVD"] = np.mean(orig_noisy_tvds)
    out_data["Original Circuit Hellinger Fidelity"] = np.mean(orig_hellinger_fids)
    out_data["TVD"] = final_tvds
    out_data["Hellinger Fidelity"] = final_fids
    out_data["Frobenius Distance"] = final_frobs
    file_name = f"{circ_name}_conv_data_noisy_{two_q_err:.1e}__{one_q_err:.1e}/{circ_name}_{block_num}_{tol}_{num_unique_circs}.json"
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_data, open(file_name, "w"))


