from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from sys import argv
from scipy.stats import entropy
import numpy as np

from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.point import CircuitPoint

from itertools import chain
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit.quantum_info import Statevector
from qiskit_aer.noise import NoiseModel, pauli_error

from bqskit.ext import bqskit_to_qiskit

from util import load_circuit, load_compiled_circuits
from util.distance import normalized_frob_dist, tvd, trace_distance, get_density_matrix, get_average_density_matrix

import multiprocessing as mp

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

one_q_fid = 1
two_q_fid = 0.999
noisy_backend = create_pauli_noise_model(one_q_fid, two_q_fid)
sim = AerSimulator(noise_model=noisy_backend)
sim_perf = AerSimulator(method='statevector')

sv_sim = StatevectorSimulator()

shots = 100

def run_sv_circuits(circs: list[QuantumCircuit], noise_model: NoiseModel = None) -> list[np.ndarray]:
    '''
    Run a set of circuits with a given Noise Model. Calculate the full StateVector.
    '''
    svs: list[np.ndarray] = []

    for circ in circs:
        sv: Statevector = sv_sim.run(circ, noise_model=noise_model).result().get_statevector()
        svs.append(sv.data)
    
    return svs

def get_ensemble_mags(ens_size, random_states: list[np.ndarray] = None) -> tuple[list[np.ndarray[np.float64]], 
                                                                                 list[np.ndarray[np.float64]], 
                                                                                 np.ndarray[np.complex128] | None]:
    global all_qcircs
    global bqskit_circs
    global target
    ensemble_inds: list[int] = np.random.choice(len(bqskit_circs), ens_size)
    ensemble: list[QuantumCircuit] = [all_qcircs[i] for i in ensemble_inds]
    
    if ensemble[0].num_qubits <= 10:
        mean_un = np.mean(np.array([bqskit_circs[i].get_unitary().numpy for i in ensemble_inds]), axis=0)
    else:
        mean_un = None

    print("Avg CNOT count: ", np.mean([c.count_ops()['cx'] for c in ensemble]))
    random_qcircs = get_random_init_state_circuits(ensemble, random_states)
    noisy_rhos = []
    noisy_probs = []
    for circs in random_qcircs:
        noisy_svs = run_sv_circuits(circs, noise_model=noisy_backend)
        probs = np.array([np.abs(sv)**2 for sv in noisy_svs], dtype=np.float64)
        avg_probs = np.mean(probs, axis=0)
        noisy_rho = get_average_density_matrix(noisy_svs)
        noisy_rhos.append(noisy_rho)
        noisy_probs.append(avg_probs)
    return noisy_rhos, noisy_probs, mean_un

def get_qcirc(circ: Circuit):
    for cycle, op in circ.operations_with_cycles():
        if op.num_qudits == 1 and not isinstance(op.gate, U3Gate):
            params = U3Gate().calc_params(op.get_unitary())
            point = CircuitPoint(cycle, op.location[0])
            circ.replace_gate(point, U3Gate(), op.location, params)

    q_circ = bqskit_to_qiskit(circ)
    return q_circ


def get_random_states(num_qubits: int, num_random_states: int = 4):
    states = []
    for i in range(4):
        state = np.random.randint(0, 2, num_qubits)
        states.append(state)
    return states

def get_random_init_state_circuits(qcircs: list[QuantumCircuit], random_states: list[np.ndarray]) -> list[list[QuantumCircuit]]:
    all_qcircs = []
    for state in random_states:
        circs = []
        for qcirc in qcircs:
            init_circ: QuantumCircuit = qcirc.copy()
            for i in range(init_circ.num_qubits):
                if state[i] == 1:
                    init_circ.h(i)
            init_circ.compose(qcirc, inplace=True)
            circs.append(transpile(init_circ, optimization_level=0))
        all_qcircs.append(circs)
    return all_qcircs
# Circ 
if __name__ == '__main__':
    global all_qcircs
    global bqskit_circs
    global target

    circ_type = argv[1]

    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)


    circ_name = argv[1]
    timestep = int(argv[2])
    tol = int(argv[3])
    num_unique_circs = int(argv[4])
    cliff = bool(int(argv[5])) if len(argv) > 5 else False

    initial_circ = load_circuit(circ_name, opt=False)
    print("Original CX Count: ", initial_circ.count(CNOTGate()))
    if initial_circ.num_qudits <= 10:
        target = initial_circ.get_unitary()

    if cliff:
        bqskit_circs = load_compiled_circuits(circ_name, tol, timestep, ignore_timestep=True, extra_str=f"_{num_unique_circs}_circ_cliff_t_final")
    else:
        bqskit_circs = load_compiled_circuits(circ_name, tol, timestep, ignore_timestep=True, extra_str=f"_{num_unique_circs}_circ_final_min")

    print("LOADED CIRCUITS", flush=True)

    # Store approximate solutions
    all_utries = []
    basic_circs = []
    circ_files = []
    base_excitations = []
    noisy_excitations = []

    ensemble_sizes = [1, 10, 100, 1000] #, 2000, 4000]
    shot_ratio = max(ensemble_sizes)

    # sampler = Sampler(mode=sim)

    random_states = get_random_states(initial_circ.num_qudits, num_random_states=5)
    qiskit_circ = bqskit_to_qiskit(initial_circ)
    qiskit_circs = get_random_init_state_circuits([qiskit_circ], random_states)
    qiskit_circs = list(chain(*qiskit_circs))
    print("Got all circuits", flush=True)
    svs = [Statevector.from_instruction(circ).data for circ in qiskit_circs]
    rhos = [get_density_matrix(sv) for sv in svs]
    print("Len of rhos: ", len(rhos))
    true_probs = [np.abs(sv)**2 for sv in svs]
    noisy_svs = run_sv_circuits(qiskit_circs, noise_model=noisy_backend)
    noisy_rhos = [get_density_matrix(sv) for sv in noisy_svs]
    noisy_dists = [trace_distance(rhos[i], noisy_rho) for i,noisy_rho in enumerate(noisy_rhos)]
    noisy_probs = [np.abs(sv)**2 for sv in noisy_svs]
    noisy_tvds = [tvd(true_probs[i], noisy_probs[i]) for i in range(len(noisy_probs))]
    print("Noisy Trace Distances: ", noisy_dists)
    print("Noisy TVDS: ", noisy_tvds)
    
    # print("Noisy Counts: ", noisy_result_dict)
    print("Finished Running", flush=True)
    # noisy_result_dict = noisy_result.get_counts(qiskit_circ)

    all_qcircs = [get_qcirc(c) for c in bqskit_circs]
    print("Got MAP", flush=True)

    print("Runing PERFECT ENSEMBLES: ")
    for j, ens_size in enumerate(ensemble_sizes):
        final_rhos, final_probs, mean_un = get_ensemble_mags(ens_size, random_states)

        print("Len of final rhos: ", len(final_rhos))

        tds = [trace_distance(final_rho, rhos[i]) for i,final_rho in enumerate(final_rhos)]
        tvds = [tvd(prob, true_probs[i]) for i,prob in enumerate(final_probs)]
        if mean_un is not None:
            frob_dist = normalized_frob_dist(target, mean_un)
            print(f"Ensemble Size: {ens_size},  Trace Distance: {tds}, TVDS: {tvds}, Frobenius Distance Normalized: {frob_dist}")
        else:
            print(f"Ensemble Size: {ens_size},  Trace Distance: {tds}, TVDS: {tvds}")




