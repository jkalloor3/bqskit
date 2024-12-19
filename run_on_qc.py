from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from sys import argv
from scipy.stats import entropy
import numpy as np
import pickle

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
from qiskit_aer.noise import NoiseModel, pauli_error

from bqskit.ext import bqskit_to_qiskit, qiskit_to_bqskit

from util import load_circuit, load_compiled_circuits

import multiprocessing as mp

shots = 100

def tvd(p: dict[str, int], q: dict[str, int], shots: int):
    p = {k: v/shots for k, v in p.items()}
    q = {k: v/shots for k, v in q.items()}
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))

def run_circuits(all_circs: list[list[QuantumCircuit]], shots: int, backend: IBMBackend = None) -> list[list[dict[str, int]]]:
    '''
    Run a set of noisy circuits on a given backend
    '''
    if backend is None:
        sampler = Sampler()
    else:
        sampler = BackendSampler(backend=backend)
    all_results = []
    for circs in all_circs:
        job = sampler.run(circs, shots=shots)
        print(f">>> Job ID: {job.job_id()}")
        print(f">>> Job Status: {job.status()}")
        result = job.result()
        results = [result[j].data.meas.get_counts() for j in range(len(circs))]
        all_results.append(results)
    # final_sv = aggregate_results(results)
    return all_results

def run_noisy_ensemble(ens_size, shots: int = 1, random_states: list[np.ndarray] = None, backend: IBMBackend = None) -> tuple[list[list[dict[str, int]]], np.ndarray]:
    '''
    Run an ensemble of size `ens_size` on a noisy backend.

    Returns a list of list of TVDs with shape (ens_size, num_random_states)

    Also, if the number of qubits is less than 10, returns the mean unitary of the ensemble.
    '''
    global all_qcircs
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

def normalized_frob_dist(mat: np.ndarray):
    global target
    frob_dist = target.get_frobenius_distance(mat) / np.sqrt(mat.shape[0] * 2)
    return frob_dist

def get_random_states(num_qubits: int, num_random_states: int = 4) -> list[np.ndarray]:
    states = []
    for i in range(4):
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

def setup_ibm(num_qubits: int) -> tuple[QiskitRuntimeService, IBMBackend]:
    '''
    Set up the IBM Quantum account and get the least busy backend with the required number of qubits.
    '''
    # service = QiskitRuntimeService.save_account(
    #                                 channel="ibm_quantum", 
    #                                 token="c04b6dfb98ed86857ab1b56cc7aeffeab68467dc0d61f60ad5b47a7797f30a55af880991a02859f88165e1cdf7d3c23161f5384ecbb54cc6e68e4588a695e36b",
    #                                 set_as_default=True,
    #                                 overwrite=True
    #                                )
    
    service = QiskitRuntimeService()
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=num_qubits)
    print(backend.name, backend.status())
    return service, backend


# Circ 
if __name__ == '__main__':

    # print(get_oneq_rb(device_backend))
    # print(get_twoq_rb(device_backend))
    global basic_circ
    global all_qcircs
    global calc_func
    global circs
    global target

    # circ_type = argv[1]

    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)


    # circ_name = argv[1]
    # timestep = int(argv[2])
    # tol = int(argv[3])

    # initial_circ = load_circuit(circ_name, opt=True)
    initial_circ = Circuit.from_file(
        '/pscratch/sd/j/jkalloor/bqskit/fixed_block_checkpoints_min' + 
        '/adder9_0_2_8_3/block_2.qasm')
    service, backend = setup_ibm(initial_circ.num_qudits)

    print("Original CX Count: ", initial_circ.count(CNOTGate()))
    if initial_circ.num_qudits <= 10:
        target = initial_circ.get_unitary()

    # num_unique_circs = int(argv[4])
    # opt_str = "_post_opt" if int(argv[5]) == 1 else ""
    # circs = load_compiled_circuits(circ_name, tol, timestep, 
    #                                ignore_timestep=True, 
    #                                extra_str=f"_{num_unique_circs}_circ_final_min{opt_str}")

    data: PassData = pickle.load(open("/pscratch/sd/j/jkalloor/bqskit/hamiltonian_perturbation_checkpoints_zxzxz/adder9_hard_0.0001_1e-06_32/" + 
                                      "data.data", 
                                      "rb"))
    circs = data.get("ensemble")

    all_qcircs = [get_qcirc(c) for c in circs]
    print("Got MAP", flush=True)

    print("LOADED CIRCUITS", flush=True)
    print("NUM Circs: ", len(circs), flush=True)

    # Store approximate solutions
    all_utries = []
    basic_circs = []
    circ_files = []
    base_excitations = []
    noisy_excitations = []

    ensemble_sizes = [1, 10, 32] #, 1000] #, 2000, 4000]
    shot_ratio = max(ensemble_sizes)

    num_random_states = 2
    random_states = get_random_states(initial_circ.num_qudits, num_random_states=num_random_states)
    qiskit_circ = bqskit_to_qiskit(initial_circ)
    qiskit_circ.measure_all()
    qiskit_circs = get_random_init_state_circuits(qiskit_circ, random_states, backend=backend)

    # Get the base results
    base_svs = run_circuits([qiskit_circs], shots=shots*shot_ratio)[0]
    noisy_svs = run_circuits([qiskit_circs], shots=shots*shot_ratio, 
                            backend=backend)[0]

    noisy_tvds = [tvd(base_svs[i], noisy_svs[i], shots=shots*shot_ratio) for i in range(num_random_states)]

    print("Noisy TVD: ", noisy_tvds)

    '''
    Calculate values for ensemble sizes
    '''
    # ensemble_mags = [0,0,0,0,0,0]

    print("Runing Noisy ENSEMBLES: ")
    # print(f"Base TVD: {base_excitations[0]}, Noisy TVD {noisy_excitations[0]}")
    for j, ens_size in enumerate(ensemble_sizes):
        shots_per_circuit = shots * (shot_ratio // ens_size)
        final_svs, mean_un = run_noisy_ensemble(ens_size, shots=shots_per_circuit, random_states=random_states, backend=backend)
        # ensemble_mags[j] = tvd(final_sv, base_sv, shots=shots*ens_size*num_random_states)
        # noisy_tvds = [tvd(base_svs[i], final_svs[i], shots=shots*ens_size) for i in range(num_random_states)]
        # noisy_tvds = np.array([[tvd(base_svs[i], final_svs[j][i],
        #                             shots=shots_per_circuit) 
        #                             for i in range(num_random_states)] 
        #                             for j in range(ens_size)])
        # noisy_tvds is a float array of shape (ens_size, num_random_states)
        # avg_noisy_tvds = np.mean(noisy_tvds, axis=0)
        print("Final SVS Shape: ", len(final_svs), len(final_svs[0]))
        aggregated_svs = [aggregate_results([final_svs[j][i] for j in range(ens_size)]) for i in range(num_random_states)]
        avg_noisy_tvds = np.array([tvd(base_svs[i], aggregated_svs[j][i],
                                    shots=shots_per_circuit*ens_size) 
                                    for i in range(num_random_states)])
        if mean_un is not None:
            frob_dist = normalized_frob_dist(mean_un)
            print(f"Ensemble Size: {ens_size}, Frobenius Distance Normalized: {frob_dist}")
        else:
            print(f"Ensemble Size: {ens_size}")

        print("Avg Noisy TVD: ", avg_noisy_tvds)


