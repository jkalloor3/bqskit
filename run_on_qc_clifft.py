from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from sys import argv
from scipy.stats import entropy
import numpy as np
from pathlib import Path
import json

from qiskit_aer import AerSimulator

import matplotlib.pyplot as plt
from bqskit.ir.gates.parameterized import U3Gate, VariableUnitaryGate
from bqskit.ir.point import CircuitPoint
from bqskit.compiler.passdata import PassData

from itertools import chain
from qiskit import QuantumCircuit, transpile
from qiskit_aer.noise import NoiseModel, depolarizing_error
import multiprocessing as mp

from bqskit.ext import bqskit_to_qiskit, qiskit_to_bqskit

from util import load_cliff_circ, frobenius_cost, load_circuit, gp_frob_cost, tvd_dict


num_random_states = 16

# def run_circuits(all_circs: list[list[QuantumCircuit]], shots: int, backend: IBMBackend = None) -> list[dict[str, int]]:
#     '''
#     Run a set of noisy circuits on a given backend
#     '''
#     if backend is None:
#         sampler = Sampler()
#     else:
#         sampler = BackendSampler(backend=backend)
#     # Return aggregated results for each random circuit
#     all_results = []
#     for circs in all_circs:
#         job = sampler.run(circs, shots=shots)
#         # print(f">>> Job ID: {job.job_id()}")
#         # print(f">>> Job Status: {job.status()}")
#         result = job.result()
#         if len(all_results) == 0:
#             all_results = [result[j].data.meas.get_counts() for j in range(len(circs))]
#         else:
#             results = [result[j].data.meas.get_counts() for j in range(len(circs))]
#             all_results = [aggregate_results([all_results[i], results[i]]) for i in range(len(circs))]
#     return all_results

# def run_noisy_ensemble(ens_size, shots: int = 1, random_states: list[np.ndarray] = None, backend: IBMBackend = None) -> tuple[list[dict[str, int]], np.ndarray]:
#     '''
#     Run an ensemble of size `ens_size` on a noisy backend.

#     Returns a list of list of TVDs with shape (ens_size, num_random_states)

#     Also, if the number of qubits is less than 10, returns the mean unitary of the ensemble.
#     '''
#     global all_qcircs
#     global uns

#     ensemble_inds: list[int] = np.random.choice(len(all_qcircs), ens_size)
#     ensemble: list[QuantumCircuit] = [all_qcircs[i] for i in ensemble_inds]
#     ensemble_uns = [uns[i] for i in ensemble_inds]
#     mean_un = np.mean(np.array(ensemble_uns), axis=0)
#     print("Avg CNOT count: ", np.mean([c.count_ops()['cx'] for c in ensemble]))
#     all_circs = [get_random_init_state_circuits(c, random_states, backend=backend) for c in ensemble]
#     final_svs = run_circuits(all_circs, shots=shots, backend=backend)
#     return final_svs, mean_un

def get_qcirc(circ: Circuit):
    q_circ = bqskit_to_qiskit(circ)
    q_circ.measure_all()
    return q_circ

def get_random_states(num_qubits: int, num_random_states: int = 4) -> list[np.ndarray]:
    states = []
    for i in range(num_random_states):
        state = np.random.randint(0, 2, num_qubits)
        states.append(state)
    return states

def get_random_init_state_circuits(qcirc: QuantumCircuit, random_states: list[np.ndarray]) -> list[QuantumCircuit]:
    circs = []
    for state in random_states:
        init_circ = QuantumCircuit(qcirc.num_qubits)
        for i in range(init_circ.num_qubits):
            if state[i] == 1:
                init_circ.h(i)
        init_circ.compose(qcirc, inplace=True)
        circs.append(init_circ)
    return circs

def aggregate_results(results: list[dict[str, int]]) -> dict[str, int]:
    total_dict = {}
    for y in results:
        x = total_dict
        total_dict = {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}
    return total_dict

def create_noise_model(t_gate_err: float, logical_gate_err: float, one_q_gates: list[str]):

    # Create an empty noise model
    noise_model = NoiseModel()

    # Add depolarizing error to all single qubit u1, u2, u3 gates
    one_q_error = depolarizing_error(logical_gate_err, 1)
    two_q_error = one_q_error.tensor(one_q_error)
    t_error = depolarizing_error(t_gate_err, 1)
    noise_model.add_all_qubit_quantum_error(t_error, ['t', 'tdg'])
    # noise_model.add_all_qubit_quantum_error(one_q_error, one_q_gates)
    # noise_model.add_all_qubit_quantum_error(two_q_error, ['cx'])

    return noise_model

def get_sim_backend(circ: QuantumCircuit, t_err: float, logical_err: float):
    one_q_gates = list(circ.count_ops().keys())
    gates_to_remove = ['cx', 'measure', 'barrier']
    for gate in gates_to_remove:
        if gate in one_q_gates:
            one_q_gates.remove(gate)
    print("One Q Gates: ", one_q_gates)
    return AerSimulator(noise_model=create_noise_model(t_err, logical_err, one_q_gates))

def get_all_circuits(circ_name: str, precision: int, rand_states: list[np.ndarray]) -> list[Circuit]:
    circ_path = load_cliff_circ(circ_name, precision)
    circ = Circuit.from_file(circ_path)
    qcirc = QuantumCircuit.from_qasm_file(circ_path)
    qcirc.measure_all()
    return circ, circ.get_unitary(), get_random_init_state_circuits(qcirc, rand_states)


def run_circuits(shots: int, backend: AerSimulator) -> list[dict[str, int]]:
    global qcircs
    all_counts = []
    circs_to_run = list(chain.from_iterable(qcircs))
    print(f"Running {len(circs_to_run)} Circuits", flush=True)
    results = backend.run(circs_to_run, shots=shots).result()
    all_counts = [[] for _ in range(len(qcircs))]
    for i in range(len(circs_to_run)):
        ind = i // num_random_states
        all_counts[ind].append(results.get_counts(i))
    return all_counts

if __name__ == '__main__':
    global qcircs
    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)

    circ_name = argv[1]
    # tol = int(argv[3])
    # num_unique_circs = int(argv[4])
    precisions = range(1, 10)
    t_errs = [10 ** (-i) for i in range(1, 6)]
    logical_err = 10 ** (-8)
    cliff = False

    shots = 1024

    orig_circ = load_circuit(circ_name)
    rand_states = get_random_states(orig_circ.num_qudits, num_random_states)
    
    backend = AerSimulator()
    target = orig_circ.get_unitary()
    orig_qcirc = bqskit_to_qiskit(orig_circ)
    orig_qcirc.measure_all()

    orig_qcircs = get_random_init_state_circuits(orig_qcirc, rand_states)

    print("Got Random States", flush=True)

    qcircs = [orig_qcircs]

    orig_statevectors = run_circuits(shots, backend)[0]
    # with mp.Pool(num_random_states) as pool:
    #     orig_statevectors = pool.starmap(backend.run, [(qc, shots) for qc in orig_qcircs])
    #     orig_statevectors = [result.result().get_counts() for result in orig_statevectors]

    print("Got Original Statevectors", flush=True)

    qcircs = []
    bqskit_circs = []
    uns = []

    with mp.Pool(num_random_states) as pool:
        circ_data =  pool.starmap(get_all_circuits, [(circ_name, prec, rand_states) for prec in precisions])
        bqskit_circs = [circ_data[0] for circ_data in circ_data]
        uns = [circ_data[1] for circ_data in circ_data]
        qcircs = [circ_data[2] for circ_data in circ_data]

    print("Loaded All Circuits", flush=True)

    # frob_dists = [gp_frob_cost(target, un) for un in uns]

    # with mp.Pool(len(qcircs)) as pool:
    #     perfect_statevectors = pool.starmap(run_circuits, [(qc, shots, backend) for qc in qcircs])

    perfect_statevectors = run_circuits(shots, backend)

    print("Ran Perfect Circuits", flush=True)

    tvds = []
    for statevector in perfect_statevectors:
        tvds.append([tvd_dict(orig_statevectors[i], statevector, shots=shots) for i, statevector in enumerate(statevector)])

    print("TVDs: ", tvds)

    all_t_data = []
    for t_err in t_errs:
        with mp.Pool(len(qcircs)) as pool:
            # statevectors = pool.starmap(run_circuits, [(qc, shots, get_sim_backend(qc[0], t_err, logical_err)) for qc in qcircs])
            statevectors = run_circuits(shots, get_sim_backend(qcircs[0][0], t_err, logical_err))
        t_data = []
        for statevector in statevectors:
            t_data.append([tvd_dict(orig_statevectors[i], sv, shots=shots) for i, sv in enumerate(statevector)])
        
        all_t_data.append(t_data)

    final_data = {}
    final_data["No Error"] = tvds
    for t_err, t_data in zip(t_errs, all_t_data):
        final_data[f"T Err: {t_err}"] = t_data

    # Plot all data onto one graph
    fig, ax = plt.subplots()

    for key, data in final_data.items():
        plot_data = [np.mean(d) for d in data]
        ax.plot(precisions, plot_data, label=key)

    ax.set_xlabel("Precision")
    ax.set_ylabel("TVD")

    ax.legend()
    fig.savefig(f"tvd_vs_t_error_{num_random_states}.png")

    json.dump(final_data, open(f"tvd_vs_t_error_{num_random_states}.json", "w"))

