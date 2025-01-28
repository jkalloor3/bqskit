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
from qiskit_ibm_runtime import SamplerV2 as Sampler, IBMBackend
from qiskit.primitives import BackendSamplerV2 as BackendSampler
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit_aer.noise import NoiseModel, depolarizing_error
import multiprocessing as mp

from bqskit.ext import bqskit_to_qiskit, qiskit_to_bqskit

from util import load_block, load_compiled_block_circuits, frobenius_cost, tvd_dict

shots = 100

def run_circuits(all_circs: list[list[QuantumCircuit]], shots: int, backend: IBMBackend = None) -> list[dict[str, int]]:
    '''
    Run a set of noisy circuits on a given backend
    '''
    if backend is None:
        sampler = Sampler()
    else:
        sampler = BackendSampler(backend=backend)
    # Return aggregated results for each random circuit
    all_results = []
    for circs in all_circs:
        job = sampler.run(circs, shots=shots)
        # print(f">>> Job ID: {job.job_id()}")
        # print(f">>> Job Status: {job.status()}")
        result = job.result()
        if len(all_results) == 0:
            all_results = [result[j].data.meas.get_counts() for j in range(len(circs))]
        else:
            results = [result[j].data.meas.get_counts() for j in range(len(circs))]
            all_results = [aggregate_results([all_results[i], results[i]]) for i in range(len(circs))]
    return all_results

def run_noisy_ensemble(ens_size, shots: int = 1, random_states: list[np.ndarray] = None, backend: IBMBackend = None) -> tuple[list[dict[str, int]], np.ndarray]:
    '''
    Run an ensemble of size `ens_size` on a noisy backend.

    Returns a list of list of TVDs with shape (ens_size, num_random_states)

    Also, if the number of qubits is less than 10, returns the mean unitary of the ensemble.
    '''
    global all_qcircs
    global uns

    ensemble_inds: list[int] = np.random.choice(len(all_qcircs), ens_size)
    ensemble: list[QuantumCircuit] = [all_qcircs[i] for i in ensemble_inds]
    ensemble_uns = [uns[i] for i in ensemble_inds]
    mean_un = np.mean(np.array(ensemble_uns), axis=0)
    print("Avg CNOT count: ", np.mean([c.count_ops()['cx'] for c in ensemble]))
    all_circs = [get_random_init_state_circuits(c, random_states, backend=backend) for c in ensemble]
    final_svs = run_circuits(all_circs, shots=shots, backend=backend)
    return final_svs, mean_un

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
        state = np.random.randint(0, 2, num_qubits)
        states.append(state)
    return states

def get_random_init_state_circuits(qcirc: QuantumCircuit, random_states: list[np.ndarray], backend: IBMBackend) -> list[QuantumCircuit]:
    circs = []
    for state in random_states:
        init_circ = qcirc.copy()
        for i in range(init_circ.num_qubits):
            if state[i] == 1:
                init_circ.h(i)
        init_circ.compose(qcirc, inplace=True)
        circs.append(transpile(init_circ, backend=backend, optimization_level=0))
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
    noise_model.add_all_qubit_quantum_error(one_q_error, one_q_gates)
    noise_model.add_all_qubit_quantum_error(two_q_error, ['cx'])

    return noise_model

def get_sim_backend(circ: QuantumCircuit, t_err: float, logical_err: float):
    one_q_gates = list(circ.count_ops().keys())
    one_q_gates.pop('cx')
    print("One Q Gates: ", one_q_gates)
    return AerSimulator(noise_model=create_noise_model(t_err, logical_err, one_q_gates))

if __name__ == '__main__':
    global all_qcircs
    global uns
    global target

    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)

    circ_name = argv[1]
    block_num = argv[2]
    tol = int(argv[3])
    num_unique_circs = int(argv[4])
    target_err = int(argv[5])
    target_fid = 1 - (10 ** (-1 * target_err))
    cliff = False

    circ_path = load_block(circ_name, block_num, good=True)
    initial_circ = Circuit.from_file(circ_path)
    target = initial_circ.get_unitary()

    # _ , backend = setup_ibm(initial_circ.num_qudits)
    num_1q_gates = len([op for op in initial_circ.operations() if op.num_qudits == 1])
    num_2q_gates = len([op for op in initial_circ.operations() if op.num_qudits == 2])
    backend = get_sim_backend(num_1q_gates, num_2q_gates, target_fid=target_fid)

    # exit(0)

    circs = load_compiled_block_circuits(circ_name, block_num, tol, num_unique_circs)
    print("Num Circs: ", len(circs), flush=True)

    # dists = [target.get_frobenius_distance(c.get_unitary()) for c in circs[:20]]
    bqskit_circs = circs
    uns = [c.get_unitary() for c in bqskit_circs]
    frob_dists = [frobenius_cost(target, un) for un in uns]
    # print("Avg Norm. Dist: ", np.mean(dists))
    print("Avg Dist: ", np.mean(frob_dists))
    print("Original CX Count: ", initial_circ.count(CNOTGate()))

    print("LOADED CIRCUITS", flush=True)

    all_qcircs = [get_qcirc(c) for c in bqskit_circs]
    print("Got MAP", flush=True)

    print("LOADED CIRCUITS", flush=True)
    print("NUM Circs: ", len(circs), flush=True)

    # Store approximate solutions
    all_utries = []
    basic_circs = []
    circ_files = []
    base_excitations = []
    noisy_excitations = []

    ensemble_sizes = [1, 10, 32, 500, 1000] #, 2000, 4000]
    shot_ratio = max(ensemble_sizes)

    num_random_states = 16
    random_states = get_random_states(initial_circ.num_qudits, num_random_states=num_random_states)
    qiskit_circ = bqskit_to_qiskit(initial_circ)
    qiskit_circ.measure_all()
    qiskit_circs = get_random_init_state_circuits(qiskit_circ, random_states, backend=backend)
    print("Num Random Circuits: ", len(qiskit_circs))

    # Get the base results
    base_svs = run_circuits([qiskit_circs], shots=shots*shot_ratio)
    noisy_svs = run_circuits([qiskit_circs], shots=shots*shot_ratio, 
                            backend=backend)
    
    # print(base_svs)
    # print(noisy_svs)

    noisy_tvds = [tvd_dict(base_svs[i], noisy_svs[i], shots=shots*shot_ratio) for i in range(num_random_states)]

    print("Noisy TVD: ", noisy_tvds)

    '''
    Calculate values for ensemble sizes
    '''
    # ensemble_mags = [0,0,0,0,0,0]

    print("Runing Noisy ENSEMBLES: ")
    # print(f"Base TVD: {base_excitations[0]}, Noisy TVD {noisy_excitations[0]}")
    final_tvds = []
    final_frobs = []

    # Get outputs using mp Pool
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(run_noisy_ensemble, [(ens_size, shots * (shot_ratio // ens_size), random_states, backend) for ens_size in ensemble_sizes])


    for j, data in enumerate(results):
        ens_size = ensemble_sizes[j]
        shots_per_circuit = shots * (shot_ratio // ens_size)
        final_svs, mean_un = data
        avg_noisy_tvds = np.array([tvd_dict(base_svs[i], final_svs[i],
                                    shots=shots_per_circuit*ens_size) 
                                    for i in range(num_random_states)])
        frob_dist = frobenius_cost(mean_un, target)
        print(f"Ensemble Size: {ens_size}, Frobenius Distance: {frob_dist}")

        print("Avg Noisy TVD: ", avg_noisy_tvds)
        final_tvds.append(np.mean(avg_noisy_tvds))
        final_frobs.append(np.mean(frob_dist))

    
    headers = ["Ensemble Size", "TVD", "Frobenius Distance"]
    out_data = {}
    out_data["Ensemble Size"] = ensemble_sizes
    out_data["TVD"] = final_tvds
    out_data["Frobenius Distance"] = final_frobs
    file_name = f"no_qp_conv_data_noisy_{target_fid}/{circ_name}_{block_num}_{tol}_{num_unique_circs}.json"
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_data, open(file_name, "w"))


